"""Communication cost model for the EP combine: all_reduce vs all-to-all.

This quantifies the win from the Stage 3 combine refactor. Both strategies are
implemented in ``run_distributed_ep_moe``:

* ``replicate_output=True`` (legacy): reverse all-to-all of expert outputs,
  then an ``all_reduce`` of the full ``[num_tokens, hidden]`` output so every
  rank holds the complete result.
* ``replicate_output=False`` (default, *sharded*): reverse all-to-all only;
  each rank keeps its own source-token rows.

The dispatch count matrix is derived from the *actual* planning code
(``build_distributed_payload_plan``), so the per-rank token movement is real.
The collective byte counts use standard hardware-independent cost models:

* ring all-reduce moves ``2 * (P - 1) / P * N * H * dtype`` bytes per rank;
* all-to-all moves, per rank, the bytes that rank sends.

End-to-end numerical correctness of both paths is covered by
``tests/test_distributed_ep_e2e.py``. This script computes volume only; wall
clock on real interconnects is deferred to the GPU stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_moe_ep.distributed_ep import build_distributed_payload_plan  # noqa: E402
from nano_moe_ep.routing import route_explicit, route_round_robin  # noqa: E402
from nano_moe_ep.types import ExpertPlacement  # noqa: E402

MIB = 1024 * 1024


def dispatch_count_matrix(router_output, placement, world_size: int) -> torch.Tensor:
    """Build the [world_size, world_size] send-count matrix from the real planner."""

    rows = []
    for source_rank in range(world_size):
        plan = build_distributed_payload_plan(
            router_output, placement, source_rank=source_rank, world_size=world_size
        )
        rows.append(plan.send_counts_by_rank)
    return torch.stack(rows, dim=0)


def _one_expert_per_rank(world_size: int) -> ExpertPlacement:
    return ExpertPlacement.from_rank_experts(
        {rank: [rank] for rank in range(world_size)},
        num_experts=world_size,
        num_ep_ranks=world_size,
    )


def scenario_counts(kind: str, num_tokens: int, world_size: int) -> torch.Tensor:
    """Return per-rank received-token counts for a named routing scenario."""

    placement = _one_expert_per_rank(world_size)
    if kind == "balanced":
        router = route_round_robin(num_tokens, world_size)
    elif kind == "all-to-one":
        router = route_explicit([0] * num_tokens, num_experts=world_size)
    else:
        raise ValueError(f"unknown scenario {kind!r}")
    matrix = dispatch_count_matrix(router, placement, world_size)
    return matrix.sum(dim=0)  # tokens received (and later returned) per rank


def combine_volume(recv_per_rank: torch.Tensor, *, num_tokens: int, hidden: int, dtype_bytes: int):
    """Per-rank combine bytes for the sharded vs replicated strategies."""

    world_size = recv_per_rank.numel()
    reverse_a2a = recv_per_rank.to(torch.float64) * hidden * dtype_bytes
    allreduce = 2.0 * (world_size - 1) / world_size * num_tokens * hidden * dtype_bytes
    sharded = reverse_a2a
    replicated = reverse_a2a + allreduce
    return sharded, replicated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--dtype-bytes", type=int, default=2, help="2=bf16/fp16, 4=fp32")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--scenarios", nargs="+", default=["balanced", "all-to-one"])
    args = parser.parse_args()

    print(
        f"combine communication per rank  (N={args.num_tokens}, H={args.hidden}, "
        f"dtype={args.dtype_bytes}B)\n"
    )
    header = f"{'scenario':<12}{'P':>3}{'sharded(MiB)':>16}{'replicated(MiB)':>18}{'reduction':>12}"
    print(header)
    print("-" * len(header))
    for scenario in args.scenarios:
        for world_size in args.world_sizes:
            if args.num_tokens % world_size != 0 and scenario == "balanced":
                continue
            recv = scenario_counts(scenario, args.num_tokens, world_size)
            sharded, replicated = combine_volume(
                recv, num_tokens=args.num_tokens, hidden=args.hidden, dtype_bytes=args.dtype_bytes
            )
            sharded_max = float(sharded.max().item()) / MIB
            replicated_max = float(replicated.max().item()) / MIB
            ratio = replicated_max / sharded_max if sharded_max > 0 else float("inf")
            print(
                f"{scenario:<12}{world_size:>3}{sharded_max:>16.1f}{replicated_max:>18.1f}{ratio:>11.1f}x"
            )
    print(
        "\nPer-rank max shown (the bottleneck rank). 'sharded' is the default path; "
        "'replicated' adds the all_reduce.\nThe all_reduce term grows with P and is "
        "independent of routing skew, so it dominates as the cluster scales."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
