"""Expert capacity and token-drop policy for top-k MoE routing.

Real MoE layers cap how many tokens each expert processes. The capacity is

    capacity = ceil(capacity_factor * num_tokens * k / num_experts)

(the per-expert share of all ``num_tokens * k`` assignments, scaled by the
slack ``capacity_factor``). Assignments beyond an expert's capacity are dropped.

Drop policy: priority is row-major over the ``[num_tokens, k]`` assignment grid,
i.e. by token index, then by gate slot. Each expert keeps the first ``capacity``
assignments it sees in that order; the rest are dropped. This is deterministic
and is the single source of truth shared by the grouped reference and its
token-by-token oracle, so both drop exactly the same assignments.
"""

from __future__ import annotations

import math

import torch

from nano_moe_ep.types import TopKRouterOutput


def _validate_num_experts(num_experts: int) -> None:
    if not isinstance(num_experts, int) or isinstance(num_experts, bool):
        raise ValueError("num_experts must be an integer")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")


def compute_expert_capacity(num_tokens: int, num_experts: int, k: int, capacity_factor: float) -> int:
    """Return the per-expert token capacity for a top-k layer."""

    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or num_tokens < 0:
        raise ValueError("num_tokens must be a non-negative integer")
    _validate_num_experts(num_experts)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(capacity_factor, (int, float)) or isinstance(capacity_factor, bool):
        raise ValueError("capacity_factor must be a number")
    if capacity_factor < 0:
        raise ValueError("capacity_factor must be non-negative")
    return math.ceil(capacity_factor * num_tokens * k / num_experts)


def expert_load(router_output: TopKRouterOutput, num_experts: int) -> torch.Tensor:
    """Return the number of assignments routed to each expert (before dropping)."""

    _validate_num_experts(num_experts)
    indices = router_output.expert_indices
    if indices.numel() > 0 and (indices >= num_experts).any().item():
        raise ValueError("expert indices must be in [0, num_experts)")
    flat = indices.reshape(-1)
    load = torch.zeros(num_experts, dtype=torch.long, device=indices.device)
    if flat.numel() > 0:
        load.scatter_add_(0, flat.to(torch.long), torch.ones_like(flat, dtype=torch.long))
    return load


def build_capacity_mask(router_output: TopKRouterOutput, num_experts: int, capacity: int) -> torch.Tensor:
    """Return a ``[num_tokens, k]`` bool mask of assignments kept under ``capacity``."""

    _validate_num_experts(num_experts)
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")
    indices = router_output.expert_indices
    if indices.numel() > 0 and (indices >= num_experts).any().item():
        raise ValueError("expert indices must be in [0, num_experts)")
    if indices.numel() == 0:
        return torch.zeros_like(indices, dtype=torch.bool)

    flat = indices.reshape(-1).to(torch.long)
    # rank_within_expert[p] = how many earlier flat positions share this expert.
    onehot = flat.unsqueeze(1) == torch.arange(num_experts, device=flat.device).unsqueeze(0)
    cumulative = onehot.to(torch.long).cumsum(dim=0)
    rank_within_expert = cumulative.gather(1, flat.unsqueeze(1)).squeeze(1) - 1
    keep_flat = rank_within_expert < capacity
    return keep_flat.reshape(indices.shape)
