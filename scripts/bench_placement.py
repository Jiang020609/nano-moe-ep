"""Expert placement experiment: contiguous vs. load-aware balanced placement.

Builds a Zipf-skewed per-expert load (expert i gets load proportional to
1/(i+1), the kind of imbalance real routers produce) and compares the naive
contiguous placement against the capacitated-LPT `balanced_placement`. Both keep
the same per-rank expert count, so the comparison is apples-to-apples; the metric
that matters is the max-rank (bottleneck) load, which gates the EP step.

Reports per-rank load, max-rank load, and load imbalance for each placement,
plus the bottleneck reduction from load-aware placement. Counts only; wall clock
on real devices is deferred to the GPU stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_moe_ep.placement import (  # noqa: E402
    balanced_placement,
    contiguous_placement,
    load_imbalance,
    max_rank_load,
    rank_load,
)


def zipf_load(num_experts: int, total: int, exponent: float) -> torch.Tensor:
    weights = 1.0 / (torch.arange(1, num_experts + 1, dtype=torch.float64) ** exponent)
    weights = weights / weights.sum()
    return (weights * total).round().to(dtype=torch.long)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--total", type=int, default=8192, help="total assignments across experts")
    parser.add_argument("--zipf", type=float, default=1.0, help="Zipf skew exponent")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()

    load = zipf_load(args.num_experts, args.total, args.zipf)
    print(
        f"expert placement  (E={args.num_experts}, total={int(load.sum())}, zipf={args.zipf})\n"
    )
    print(f"per-expert load: {[int(v) for v in load]}\n")

    header = (
        f"{'P':>3}{'contig max':>12}{'balanced max':>14}"
        f"{'contig imbal':>14}{'balanced imbal':>16}{'reduction':>11}"
    )
    print(header)
    print("-" * len(header))
    for world_size in args.world_sizes:
        if world_size > args.num_experts:
            continue
        contig = contiguous_placement(args.num_experts, world_size)
        balanced = balanced_placement(load, world_size)
        contig_max = max_rank_load(contig, load)
        balanced_max = max_rank_load(balanced, load)
        ratio = contig_max / balanced_max if balanced_max > 0 else float("inf")
        print(
            f"{world_size:>3}{contig_max:>12}{balanced_max:>14}"
            f"{load_imbalance(contig, load):>13.2f}x{load_imbalance(balanced, load):>15.2f}x"
            f"{ratio:>10.2f}x"
        )
    print(
        "\nBoth placements use the same per-rank expert count; balanced placement only "
        "reassigns which experts share a rank.\nLower max-rank load means a faster "
        "bottleneck rank, so the EP step is gated less by routing skew."
    )

    # Show the per-rank breakdown for the largest world size as a concrete example.
    example = min(args.world_sizes[-1], args.num_experts)
    contig = contiguous_placement(args.num_experts, example)
    balanced = balanced_placement(load, example)
    print(f"\nper-rank load at P={example}:")
    print(f"  contiguous: {[int(v) for v in rank_load(contig, load)]}")
    print(f"  balanced:   {[int(v) for v in rank_load(balanced, load)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
