"""Real-rank timing benchmark for the distributed EP forward paths.

Launch with ``torchrun`` on one node:

    torchrun --standalone --nproc_per_node=8 scripts/bench_nccl_ep.py

By default this uses NCCL when enough CUDA devices are visible and falls back to
CPU/Gloo otherwise. Rank 0 prints a Markdown table and can also write Markdown
and CSV files via ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_moe_ep.distributed_ep import (  # noqa: E402
    DistributedEPConfig,
    run_distributed_ep_moe,
    run_distributed_topk_ep_moe,
)
from nano_moe_ep.placement import balanced_placement, contiguous_placement  # noqa: E402
from nano_moe_ep.reference import ReferenceMoEFFN, TopKReferenceMoEFFN  # noqa: E402
from nano_moe_ep.routing import expert_load, route_round_robin  # noqa: E402
from nano_moe_ep.types import TopKRouterOutput  # noqa: E402


@dataclass(frozen=True)
class BenchRow:
    case: str
    combine: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _choose_backend(world_size: int) -> str:
    requested = os.environ.get("NANO_MOE_EP_BACKEND")
    if requested:
        return requested.lower()
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unknown dtype {name!r}")


def _build_skewed_topk_router(
    num_tokens: int,
    num_experts: int,
    k: int,
    hot_frac: float,
    seed: int,
) -> TopKRouterOutput:
    """Deterministic skewed top-k router with expert 0 as the hot expert."""

    if num_experts < 2:
        raise ValueError("num_experts must be at least 2 for the skewed router")
    if k < 1 or k > num_experts:
        raise ValueError("k must be in [1, num_experts]")
    if not 0.0 <= hot_frac <= 1.0:
        raise ValueError("hot_frac must be in [0, 1]")

    generator = torch.Generator().manual_seed(seed)
    rest = (1.0 - hot_frac) / (num_experts - 1)
    probs = torch.full((num_experts,), rest)
    probs[0] = hot_frac
    slot0 = torch.multinomial(probs, num_tokens, replacement=True, generator=generator)

    rows: list[list[int]] = []
    for token in range(num_tokens):
        chosen = [int(slot0[token].item())]
        remaining = [expert for expert in range(num_experts) if expert not in chosen]
        perm = torch.randperm(len(remaining), generator=generator)
        for j in range(k - 1):
            chosen.append(remaining[int(perm[j].item())])
        rows.append(chosen)
    expert_indices = torch.tensor(rows, dtype=torch.long)
    weights = torch.full((num_tokens, k), 1.0 / k, dtype=torch.float32)
    return TopKRouterOutput(expert_indices=expert_indices, weights=weights)


def _barrier(config: DistributedEPConfig) -> None:
    if config.backend == "nccl":
        dist.barrier(device_ids=[config.device.index or 0])
    else:
        dist.barrier()


def _bottleneck_elapsed_ms(fn: Callable[[], object], config: DistributedEPConfig) -> float:
    """Return max elapsed time across ranks for one synchronized iteration."""

    _barrier(config)
    if config.device.type == "cuda":
        torch.cuda.synchronize(config.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device=config.device)
    else:
        start_time = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        elapsed = torch.tensor(elapsed_ms, dtype=torch.float64)

    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _run_case(
    name: str,
    combine: str,
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    config: DistributedEPConfig,
) -> BenchRow:
    for _ in range(warmup):
        _bottleneck_elapsed_ms(fn, config)

    timings = [_bottleneck_elapsed_ms(fn, config) for _ in range(iters)]
    return BenchRow(
        case=name,
        combine=combine,
        mean_ms=statistics.fmean(timings),
        p50_ms=_percentile(timings, 0.50),
        p95_ms=_percentile(timings, 0.95),
        min_ms=min(timings),
        max_ms=max(timings),
    )


def _markdown_report(args: argparse.Namespace, rows: list[BenchRow], *, backend: str, world_size: int, device_label: str) -> str:
    lines = [
        "# NCCL/Gloo EP Timing Benchmark",
        "",
        f"- backend: `{backend}`",
        f"- world size: `{world_size}`",
        f"- device: `{device_label}`",
        f"- num tokens: `{args.num_tokens}`",
        f"- hidden dim: `{args.hidden_dim}`",
        f"- ffn dim: `{args.ffn_dim}`",
        f"- num experts: `{2 * world_size}`",
        f"- top-k: `{args.k}`",
        f"- dtype: `{args.dtype}`",
        f"- warmup / iters: `{args.warmup}` / `{args.iters}`",
        f"- capacity factor: `{args.capacity_factor}`",
        "",
        "| case | combine | mean ms | p50 ms | p95 ms | min ms | max ms |",
        "|------|---------|---------|--------|--------|--------|--------|",
    ]
    for row in rows:
        lines.append(
            f"| {row.case} | {row.combine} | {row.mean_ms:.3f} | {row.p50_ms:.3f} | "
            f"{row.p95_ms:.3f} | {row.min_ms:.3f} | {row.max_ms:.3f} |"
        )
    lines.extend(
        [
            "",
            "Times are max latency across ranks for each synchronized iteration.",
            "Run the correctness smoke before treating these as benchmark numbers.",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    args: argparse.Namespace,
    rows: list[BenchRow],
    *,
    backend: str,
    world_size: int,
    device_label: str,
) -> None:
    if not args.output_dir:
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"bench_ep_{backend}_{world_size}ranks_{timestamp}"

    md_path = output_dir / f"{stem}.md"
    md_path.write_text(
        _markdown_report(args, rows, backend=backend, world_size=world_size, device_label=device_label) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "combine",
                "mean_ms",
                "p50_ms",
                "p95_ms",
                "min_ms",
                "max_ms",
                "backend",
                "world_size",
                "device",
                "num_tokens",
                "hidden_dim",
                "ffn_dim",
                "num_experts",
                "k",
                "dtype",
                "warmup",
                "iters",
                "capacity_factor",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.case,
                    row.combine,
                    f"{row.mean_ms:.6f}",
                    f"{row.p50_ms:.6f}",
                    f"{row.p95_ms:.6f}",
                    f"{row.min_ms:.6f}",
                    f"{row.max_ms:.6f}",
                    backend,
                    world_size,
                    device_label,
                    args.num_tokens,
                    args.hidden_dim,
                    args.ffn_dim,
                    2 * world_size,
                    args.k,
                    args.dtype,
                    args.warmup,
                    args.iters,
                    args.capacity_factor,
                ]
            )
    print(f"wrote {md_path} and {csv_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--ffn-dim", type=int, default=4096)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--hot-frac", type=float, default=0.5)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--output-dir", default=None, help="Optional directory for rank-0 Markdown and CSV output.")
    parser.add_argument(
        "--combine-modes",
        nargs="+",
        choices=["sharded", "replicated"],
        default=["sharded", "replicated"],
    )
    args = parser.parse_args()

    if args.num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if args.hidden_dim < 1 or args.ffn_dim < 1:
        raise ValueError("hidden_dim and ffn_dim must be positive")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive")

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    backend = _choose_backend(world_size_env)
    if backend == "nccl" and (not dist.is_nccl_available() or torch.cuda.device_count() < world_size_env):
        print("NCCL requested but unavailable or too few CUDA devices.", flush=True)
        return 2

    dist.init_process_group(backend=backend)
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        device = torch.device(f"cuda:{local_rank}") if backend == "nccl" else torch.device("cpu")
        if backend == "nccl":
            torch.cuda.set_device(device)
        dtype = _dtype_from_name(args.dtype)
        if device.type != "cuda" and dtype is not torch.float32:
            raise ValueError("non-fp32 dtypes are only supported for CUDA benchmarks")

        config = DistributedEPConfig(backend=backend, world_size=world_size, rank=rank, device=device)
        num_experts = 2 * world_size
        device_label = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

        torch.manual_seed(args.seed)
        model_top1 = ReferenceMoEFFN(args.hidden_dim, args.ffn_dim, num_experts).to(device=device, dtype=dtype)
        torch.manual_seed(args.seed)
        model_topk = TopKReferenceMoEFFN(args.hidden_dim, args.ffn_dim, num_experts).to(device=device, dtype=dtype)
        torch.manual_seed(args.seed)
        inputs = torch.randn(args.num_tokens, args.hidden_dim, device=device, dtype=dtype)

        contiguous = contiguous_placement(num_experts, world_size)
        top1_router = route_round_robin(args.num_tokens, num_experts)
        topk_router = _build_skewed_topk_router(
            args.num_tokens,
            num_experts,
            args.k,
            args.hot_frac,
            args.seed,
        )
        balanced = balanced_placement(expert_load(topk_router, num_experts), world_size)

        rows: list[BenchRow] = []
        with torch.no_grad():
            for combine in args.combine_modes:
                replicate_output = combine == "replicated"
                rows.append(
                    _run_case(
                        "top-1",
                        combine,
                        lambda replicate_output=replicate_output: run_distributed_ep_moe(
                            inputs,
                            top1_router,
                            model_top1.experts,
                            contiguous,
                            config=config,
                            replicate_output=replicate_output,
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        config=config,
                    )
                )
                rows.append(
                    _run_case(
                        "top-k",
                        combine,
                        lambda replicate_output=replicate_output: run_distributed_topk_ep_moe(
                            inputs,
                            topk_router,
                            model_topk.experts,
                            balanced,
                            config=config,
                            replicate_output=replicate_output,
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        config=config,
                    )
                )
                rows.append(
                    _run_case(
                        "top-k+cap",
                        combine,
                        lambda replicate_output=replicate_output: run_distributed_topk_ep_moe(
                            inputs,
                            topk_router,
                            model_topk.experts,
                            balanced,
                            config=config,
                            capacity_factor=args.capacity_factor,
                            replicate_output=replicate_output,
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        config=config,
                    )
                )

        if rank == 0:
            report = _markdown_report(args, rows, backend=backend, world_size=world_size, device_label=device_label)
            print(report, flush=True)
            _write_outputs(args, rows, backend=backend, world_size=world_size, device_label=device_label)
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
