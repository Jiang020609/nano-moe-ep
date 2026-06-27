"""Load-aware expert placement and a simple cost model.

Expert parallelism pins each expert to one rank. The default contiguous/round-
robin placement is balanced in *expert count* but oblivious to *load*: under
skewed routing a few hot experts can land on the same rank, and the EP step is
gated by the slowest (max-load) rank. This module adds:

* a cost model (`rank_load`, `max_rank_load`, `load_imbalance`) that scores a
  placement against a per-expert load vector; and
* `balanced_placement`, a capacitated longest-processing-time (LPT) heuristic
  that keeps the per-rank expert *count* as even as the contiguous baseline
  while minimizing the max-rank *load*.

Both return / consume the existing immutable `ExpertPlacement`, so a load-aware
placement plugs straight into the dispatch/combine and distributed paths.
"""

from __future__ import annotations

import torch

from nano_moe_ep.types import ExpertPlacement


def _validate_num_ep_ranks(num_ep_ranks: int) -> None:
    if not isinstance(num_ep_ranks, int) or isinstance(num_ep_ranks, bool):
        raise ValueError("num_ep_ranks must be an integer")
    if num_ep_ranks <= 0:
        raise ValueError("num_ep_ranks must be positive")


def _as_load_vector(expert_load: torch.Tensor) -> torch.Tensor:
    if not isinstance(expert_load, torch.Tensor):
        raise ValueError("expert_load must be a torch.Tensor")
    if expert_load.ndim != 1:
        raise ValueError("expert_load must be a 1D tensor")
    if expert_load.numel() == 0:
        raise ValueError("expert_load must contain at least one expert")
    load = expert_load.to(dtype=torch.long)
    if (load < 0).any().item():
        raise ValueError("expert_load must be non-negative")
    return load


def _rank_capacities(num_experts: int, num_ep_ranks: int) -> list[int]:
    """Per-rank expert-count capacity, as even as possible (ceil for the first ranks)."""

    base = num_experts // num_ep_ranks
    remainder = num_experts % num_ep_ranks
    return [base + (1 if rank < remainder else 0) for rank in range(num_ep_ranks)]


def rank_load(placement: ExpertPlacement, expert_load: torch.Tensor) -> torch.Tensor:
    """Return the total load assigned to each rank under ``placement``."""

    if not isinstance(placement, ExpertPlacement):
        raise ValueError("placement must be an ExpertPlacement")
    load = _as_load_vector(expert_load)
    if load.numel() != placement.num_experts:
        raise ValueError("expert_load must have one entry per expert")
    owners = torch.tensor(placement.owner_rank_by_expert, dtype=torch.long, device=load.device)
    totals = torch.zeros(placement.num_ep_ranks, dtype=torch.long, device=load.device)
    totals.scatter_add_(0, owners, load)
    return totals


def max_rank_load(placement: ExpertPlacement, expert_load: torch.Tensor) -> int:
    """Return the bottleneck (maximum) per-rank load under ``placement``."""

    return int(rank_load(placement, expert_load).max().item())


def load_imbalance(placement: ExpertPlacement, expert_load: torch.Tensor) -> float:
    """Return max-rank load divided by mean-rank load (1.0 is perfectly balanced)."""

    totals = rank_load(placement, expert_load).to(dtype=torch.float64)
    mean = totals.mean().item()
    if mean == 0.0:
        return 1.0
    return float(totals.max().item() / mean)


def contiguous_placement(num_experts: int, num_ep_ranks: int) -> ExpertPlacement:
    """Block placement: consecutive experts fill each rank to its count capacity."""

    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    _validate_num_ep_ranks(num_ep_ranks)
    if num_ep_ranks > num_experts:
        raise ValueError("num_ep_ranks must not exceed num_experts")
    capacities = _rank_capacities(num_experts, num_ep_ranks)
    owner_by_expert: list[int] = []
    for rank, capacity in enumerate(capacities):
        owner_by_expert.extend([rank] * capacity)
    return ExpertPlacement(owner_rank_by_expert=tuple(owner_by_expert), num_ep_ranks=num_ep_ranks)


def balanced_placement(expert_load: torch.Tensor, num_ep_ranks: int) -> ExpertPlacement:
    """Capacitated LPT placement: even expert counts, minimized max-rank load.

    Experts are assigned in descending load order to the least-loaded rank that
    still has expert-count capacity. Ties break toward the lowest rank index, so
    the result is deterministic. The per-rank expert count matches the
    contiguous baseline, so this trades no extra memory for better balance.
    """

    load = _as_load_vector(expert_load)
    num_experts = load.numel()
    _validate_num_ep_ranks(num_ep_ranks)
    if num_ep_ranks > num_experts:
        raise ValueError("num_ep_ranks must not exceed num_experts")

    capacities = _rank_capacities(num_experts, num_ep_ranks)
    remaining = capacities[:]
    rank_totals = [0] * num_ep_ranks
    owner_by_expert = [-1] * num_experts

    # Descending load; ties broken by ascending expert id (stable sort on -load).
    order = torch.argsort(-load, stable=True)
    for expert_id in order.tolist():
        best_rank = -1
        best_total = None
        for rank in range(num_ep_ranks):
            if remaining[rank] == 0:
                continue
            if best_total is None or rank_totals[rank] < best_total:
                best_total = rank_totals[rank]
                best_rank = rank
        owner_by_expert[expert_id] = best_rank
        rank_totals[best_rank] += int(load[expert_id].item())
        remaining[best_rank] -= 1

    return ExpertPlacement(owner_rank_by_expert=tuple(owner_by_expert), num_ep_ranks=num_ep_ranks)
