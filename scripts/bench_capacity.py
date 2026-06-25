"""Routing-skew vs. capacity experiment for top-k MoE.

Generates a skewed top-k assignment (one hot expert holds ``--hot-frac`` of the
slot-0 mass; the remaining experts share the rest) and reports, across a sweep
of capacity factors, the fundamental MoE trade-off:

* load imbalance   - max / mean assignments per expert (how skewed routing is);
* drop rate        - fraction of all assignments dropped at that capacity;
* capacity util    - kept assignments / (num_experts * capacity), i.e. how full
                     the capacity buffers are. Low utilization means padded,
                     wasted expert compute; high utilization with a high drop
                     rate means the capacity is too small for the skew.

All counts come from the real drop policy in ``routing.capacity``. This reports
load/drop statistics only; expert-compute wall clock is deferred to the GPU
stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_moe_ep.routing import (  # noqa: E402
    build_capacity_mask,
    compute_expert_capacity,
    expert_load,
    route_topk_explicit,
)


def build_skewed_router(num_tokens, num_experts, k, hot_frac, generator):
    """Build a skewed top-k assignment with distinct experts per token."""

    rest = (1.0 - hot_frac) / (num_experts - 1)
    slot0_prob = torch.full((num_experts,), rest)
    slot0_prob[0] = hot_frac
    slot0 = torch.multinomial(slot0_prob, num_tokens, replacement=True, generator=generator)

    rows = []
    for token in range(num_tokens):
        chosen = [int(slot0[token].item())]
        remaining = [e for e in range(num_experts) if e not in chosen]
        perm = torch.randperm(len(remaining), generator=generator)
        for j in range(k - 1):
            chosen.append(remaining[int(perm[j].item())])
        rows.append(chosen)
    return route_topk_explicit(rows, num_experts=num_experts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--hot-frac", type=float, default=0.5, help="slot-0 mass on the hot expert")
    parser.add_argument("--capacity-factors", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--seed", type=int, default=2029)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    router = build_skewed_router(args.num_tokens, args.num_experts, args.k, args.hot_frac, generator)
    load = expert_load(router, args.num_experts)
    total = int(load.sum().item())
    mean = total / args.num_experts
    imbalance = float(load.max().item()) / mean if mean > 0 else float("inf")

    print(
        f"top-k skew/capacity  (N={args.num_tokens}, E={args.num_experts}, k={args.k}, "
        f"hot_frac={args.hot_frac})\n"
    )
    print(f"per-expert load: {[int(v) for v in load]}")
    print(f"load imbalance (max/mean): {imbalance:.2f}x\n")

    header = f"{'cap_factor':>11}{'capacity':>10}{'dropped':>10}{'drop_rate':>11}{'cap_util':>10}"
    print(header)
    print("-" * len(header))
    for capacity_factor in args.capacity_factors:
        capacity = compute_expert_capacity(args.num_tokens, args.num_experts, args.k, capacity_factor)
        keep = build_capacity_mask(router, args.num_experts, capacity)
        kept = int(keep.sum().item())
        dropped = total - kept
        drop_rate = dropped / total if total > 0 else 0.0
        cap_util = kept / (args.num_experts * capacity) if capacity > 0 else 0.0
        print(
            f"{capacity_factor:>11.2f}{capacity:>10}{dropped:>10}{drop_rate:>10.1%}{cap_util:>10.1%}"
        )
    print(
        "\nHigher capacity factor lowers drops but raises padding (lower utilization); "
        "the right point depends on the routing skew above."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
