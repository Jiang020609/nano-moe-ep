from __future__ import annotations

from collections.abc import Sequence

import torch

from nano_moe_ep.types import RouterOutput, _is_integer_tensor


def _validate_num_experts(num_experts: int) -> None:
    if not isinstance(num_experts, int) or isinstance(num_experts, bool):
        raise ValueError("num_experts must be an integer")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")


def _validate_num_tokens(num_tokens: int) -> None:
    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
        raise ValueError("num_tokens must be an integer")
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")


def _contains_bool(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_bool(item) for item in value)
    return False


def _assignment_tensor(assignments: Sequence[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(assignments, torch.Tensor):
        indices = assignments.detach().clone()
    else:
        if not isinstance(assignments, Sequence):
            raise ValueError("assignments must be a tensor or sequence of integers")
        if _contains_bool(assignments):
            raise ValueError("assignments must contain integer-like expert indices")
        try:
            indices = torch.as_tensor(assignments)
        except (TypeError, ValueError) as exc:
            raise ValueError("assignments must be a tensor or sequence of integers") from exc

    if indices.ndim == 2 and indices.shape[1] == 1:
        indices = indices.reshape(-1)
    elif indices.ndim != 1:
        raise ValueError("assignments must have shape [num_tokens] or [num_tokens, 1]")

    if not _is_integer_tensor(indices):
        raise ValueError("assignments must contain integer-like expert indices")
    return indices.to(dtype=torch.long)


def _validate_expert_range(indices: torch.Tensor, num_experts: int) -> None:
    if indices.numel() == 0:
        return
    if (indices < 0).any().item() or (indices >= num_experts).any().item():
        raise ValueError("expert indices must be in [0, num_experts)")


def route_round_robin(
    num_tokens: int,
    num_experts: int,
    *,
    device: torch.device | str | None = None,
) -> RouterOutput:
    """Assign token i to expert i % num_experts with unit top-1 weights."""

    _validate_num_tokens(num_tokens)
    _validate_num_experts(num_experts)
    expert_indices = torch.arange(num_tokens, device=device, dtype=torch.long).remainder(num_experts)
    expert_indices = expert_indices.reshape(num_tokens, 1)
    weights = torch.ones((num_tokens, 1), device=device, dtype=torch.float32)
    return RouterOutput(expert_indices=expert_indices, weights=weights)


def route_explicit(
    assignments: Sequence[int] | torch.Tensor,
    num_experts: int,
    *,
    num_tokens: int | None = None,
    device: torch.device | str | None = None,
) -> RouterOutput:
    """Use a caller-supplied top-1 expert assignment with unit weights."""

    _validate_num_experts(num_experts)
    indices = _assignment_tensor(assignments)
    if num_tokens is not None:
        _validate_num_tokens(num_tokens)
        if indices.numel() != num_tokens:
            raise ValueError("assignment length must match num_tokens")
    _validate_expert_range(indices, num_experts)
    if device is not None:
        indices = indices.to(device=device)
    weights = torch.ones((indices.numel(), 1), device=indices.device, dtype=torch.float32)
    return RouterOutput(expert_indices=indices.reshape(-1, 1), weights=weights)
