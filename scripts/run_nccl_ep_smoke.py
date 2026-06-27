"""General N-rank NCCL/Gloo smoke for the full EP data path.

Validates, on real ranks (NCCL when CUDA is available, else a Gloo CPU
fallback), that every distributed path matches the single-process reference:

1. top-1 distributed EP (``run_distributed_ep_moe``) vs the logical reference;
2. top-k EP, no capacity (``run_distributed_topk_ep_moe``) vs the top-k reference;
3. top-k EP with a capacity factor (token dropping) vs the top-k reference, also
   checking that distributed and reference drop counts agree.

Cases 2 and 3 use a skewed router and a load-aware ``balanced_placement`` so the
non-contiguous placement path is exercised too.

Launch (single node, N GPUs):

    torchrun --standalone --nproc_per_node=8 scripts/run_nccl_ep_smoke.py

Set ``NANO_MOE_EP_BACKEND=gloo`` to force the CPU fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_moe_ep.distributed_ep import (  # noqa: E402
    DistributedEPConfig,
    run_distributed_ep_moe,
    run_distributed_topk_ep_moe,
)
from nano_moe_ep.dispatch_combine import run_logical_ep_moe  # noqa: E402
from nano_moe_ep.placement import balanced_placement, contiguous_placement, max_rank_load  # noqa: E402
from nano_moe_ep.reference import ReferenceMoEFFN, TopKReferenceMoEFFN  # noqa: E402
from nano_moe_ep.routing import expert_load, route_round_robin  # noqa: E402
from nano_moe_ep.types import TopKRouterOutput  # noqa: E402

RTOL = 1e-4
ATOL = 1e-5
SEED = 2029
HIDDEN_DIM = 16
FFN_DIM = 32


def _choose_backend(world_size: int) -> str:
    requested = os.environ.get("NANO_MOE_EP_BACKEND")
    if requested:
        return requested.lower()
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def _build_skewed_topk_router(num_tokens: int, num_experts: int, k: int) -> TopKRouterOutput:
    """Deterministic skewed top-k router (expert 0 is hot), identical on every rank."""

    generator = torch.Generator().manual_seed(SEED)
    probs = torch.full((num_experts,), 0.5 / (num_experts - 1))
    probs[0] = 0.5
    slot0 = torch.multinomial(probs, num_tokens, replacement=True, generator=generator)
    rows = []
    for token in range(num_tokens):
        chosen = [int(slot0[token].item())]
        remaining = [e for e in range(num_experts) if e not in chosen]
        perm = torch.randperm(len(remaining), generator=generator)
        for j in range(k - 1):
            chosen.append(remaining[int(perm[j].item())])
        rows.append(chosen)
    expert_indices = torch.tensor(rows, dtype=torch.long)
    weights = torch.full((num_tokens, k), 1.0 / k, dtype=torch.float32)
    return TopKRouterOutput(expert_indices=expert_indices, weights=weights)


def _max_abs_error(distributed: torch.Tensor, reference: torch.Tensor) -> float:
    return float((distributed - reference).abs().max().item()) if distributed.numel() else 0.0


def main() -> int:
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    backend = _choose_backend(world_size_env)
    if backend == "nccl" and (not dist.is_nccl_available() or torch.cuda.device_count() < world_size_env):
        print("NCCL requested but unavailable or too few CUDA devices.", flush=True)
        return 2

    dist.init_process_group(backend=backend)
    exit_code = 0
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        device = torch.device(f"cuda:{local_rank}") if backend == "nccl" else torch.device("cpu")
        if backend == "nccl":
            torch.cuda.set_device(device)

        num_experts = 2 * world_size
        num_tokens = 64 * world_size
        config = DistributedEPConfig(backend=backend, world_size=world_size, rank=rank, device=device)

        # The reference oracles are single-process CPU paths; the distributed paths
        # run on `device`. Both expert sets are built from the same seed, so their
        # weights are identical, and every rank sees identical inputs/routers.
        torch.manual_seed(SEED)
        ref_model_top1 = ReferenceMoEFFN(HIDDEN_DIM, FFN_DIM, num_experts)
        torch.manual_seed(SEED)
        ref_model_topk = TopKReferenceMoEFFN(HIDDEN_DIM, FFN_DIM, num_experts)
        torch.manual_seed(SEED)
        dist_model_top1 = ReferenceMoEFFN(HIDDEN_DIM, FFN_DIM, num_experts).to(device)
        torch.manual_seed(SEED)
        dist_model_topk = TopKReferenceMoEFFN(HIDDEN_DIM, FFN_DIM, num_experts).to(device)
        torch.manual_seed(SEED)
        inputs_cpu = torch.randn(num_tokens, HIDDEN_DIM)
        inputs = inputs_cpu.to(device)

        contiguous = contiguous_placement(num_experts, world_size)
        topk_router = _build_skewed_topk_router(num_tokens, num_experts, k=2)
        balanced = balanced_placement(expert_load(topk_router, num_experts), world_size)
        top1_router = route_round_robin(num_tokens, num_experts)

        results = []
        with torch.no_grad():
            # 1. top-1 distributed (on device) vs logical reference (on CPU).
            dist_top1, _ = run_distributed_ep_moe(
                inputs, top1_router, dist_model_top1.experts, contiguous, config=config, replicate_output=True
            )
            ref_top1, _ = run_logical_ep_moe(inputs_cpu, top1_router, ref_model_top1.experts, contiguous)
            dist_top1 = dist_top1.cpu()
            results.append(("top-1", _max_abs_error(dist_top1, ref_top1),
                            torch.allclose(dist_top1, ref_top1, rtol=RTOL, atol=ATOL)))

            # 2. top-k, no capacity, balanced placement.
            dist_topk, _ = run_distributed_topk_ep_moe(
                inputs, topk_router, dist_model_topk.experts, balanced, config=config, replicate_output=True
            )
            ref_topk, _ = ref_model_topk(inputs_cpu, topk_router)
            dist_topk = dist_topk.cpu()
            results.append(("top-k", _max_abs_error(dist_topk, ref_topk),
                            torch.allclose(dist_topk, ref_topk, rtol=RTOL, atol=ATOL)))

            # 3. top-k with capacity dropping, balanced placement.
            dist_cap, trace_cap = run_distributed_topk_ep_moe(
                inputs, topk_router, dist_model_topk.experts, balanced, config=config,
                capacity_factor=1.25, replicate_output=True,
            )
            ref_cap, ref_trace = ref_model_topk(inputs_cpu, topk_router, capacity_factor=1.25)
            dropped = torch.tensor(trace_cap.num_local_dropped, device=device)
            dist.all_reduce(dropped, op=dist.ReduceOp.SUM)
            drops_match = int(dropped.item()) == ref_trace.num_dropped
            dist_cap = dist_cap.cpu()
            results.append(("top-k+cap", _max_abs_error(dist_cap, ref_cap),
                            torch.allclose(dist_cap, ref_cap, rtol=RTOL, atol=ATOL) and drops_match))

        all_pass = all(passed for _name, _err, passed in results)
        passed_flag = torch.tensor(1 if all_pass else 0, dtype=torch.int, device=device)
        dist.all_reduce(passed_flag, op=dist.ReduceOp.MIN)

        summary = "; ".join(f"{name} maxerr={err:.2e} {'OK' if ok else 'FAIL'}" for name, err, ok in results)
        print(
            f"rank {rank}/{world_size} [{backend}] device={device} "
            f"contig_max_load={max_rank_load(contiguous, expert_load(topk_router, num_experts))} "
            f"balanced_max_load={max_rank_load(balanced, expert_load(topk_router, num_experts))} | {summary}",
            flush=True,
        )
        if rank == 0:
            print(f"OVERALL: {'PASS' if int(passed_flag.item()) == 1 else 'FAIL'}", flush=True)
        if int(passed_flag.item()) != 1:
            exit_code = 1
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
