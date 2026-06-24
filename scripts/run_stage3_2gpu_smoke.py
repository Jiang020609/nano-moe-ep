from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_moe_ep.distributed_ep import DistributedEPConfig, run_distributed_ep_moe  # noqa: E402
from nano_moe_ep.dispatch_combine import run_logical_ep_moe  # noqa: E402
from nano_moe_ep.reference import ReferenceMoEFFN  # noqa: E402
from nano_moe_ep.routing import route_explicit  # noqa: E402
from nano_moe_ep.types import ExpertPlacement  # noqa: E402


RTOL = 1e-5
ATOL = 1e-6


def _choose_backend(world_size: int) -> str:
    requested = os.environ.get("NANO_MOE_EP_BACKEND")
    if requested:
        return requested.lower()
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def _device_for_backend(backend: str, local_rank: int) -> torch.device:
    if backend == "nccl":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def main() -> int:
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    backend = _choose_backend(world_size_env)
    if backend == "nccl" and not dist.is_nccl_available():
        print("NCCL requested but torch.distributed reports NCCL unavailable.", flush=True)
        return 2
    if backend == "nccl" and torch.cuda.device_count() < world_size_env:
        print("NCCL requested but fewer CUDA devices than WORLD_SIZE are available.", flush=True)
        return 2

    dist.init_process_group(backend=backend)
    exit_code = 0
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        if world_size != 2:
            if rank == 0:
                print(f"Stage 3 smoke expects world_size=2, got {world_size}.", flush=True)
            return 2

        device = _device_for_backend(backend, local_rank)
        config = DistributedEPConfig(
            backend=backend,
            world_size=world_size,
            rank=rank,
            device=device,
            deterministic_seed=2029,
        )
        torch.manual_seed(config.deterministic_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.deterministic_seed)

        placement = ExpertPlacement.from_rank_experts(
            {0: [0, 1], 1: [2, 3]},
            num_experts=4,
            num_ep_ranks=2,
        )
        model = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4).to(device)
        inputs = torch.randn(11, 8, device=device)
        router_output = route_explicit(
            [0, 2, 2, 1, 3, 0, 2, 3, 1, 2, 2],
            num_experts=model.num_experts,
            device=device,
        )

        with torch.no_grad():
            distributed_output, trace = run_distributed_ep_moe(
                inputs,
                router_output,
                model.experts,
                placement,
                config=config,
            )
            reference_output, _ = run_logical_ep_moe(inputs, router_output, model.experts, placement)

        max_abs_error = (distributed_output - reference_output).abs().max()
        max_abs_error_all = max_abs_error.detach().clone()
        dist.all_reduce(max_abs_error_all, op=dist.ReduceOp.MAX)
        passed = torch.tensor(
            1 if torch.allclose(distributed_output, reference_output, rtol=RTOL, atol=ATOL) else 0,
            dtype=torch.int,
            device=device,
        )
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)

        print(
            "rank id: "
            f"{rank}; owned experts: {trace.owned_experts}; "
            f"send counts: {[int(v) for v in trace.send_plan.send_counts_by_rank.cpu()]}; "
            f"receive counts: {[int(v) for v in trace.dispatch_counts.recv_counts_by_rank.cpu()]}; "
            f"max absolute error vs reference: {float(max_abs_error_all.item()):.8g}; "
            f"{'PASS' if int(passed.item()) == 1 else 'FAIL'}",
            flush=True,
        )
        if int(passed.item()) != 1:
            exit_code = 1
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
