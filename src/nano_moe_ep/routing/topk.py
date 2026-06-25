"""Deterministic synthetic top-k routers.

These mirror the Stage 1 top-1 routers but select ``k`` distinct experts per
token. Weights are uniform (``1/k`` when normalized, else ``1``); a real learned
softmax gate is out of scope for the synthetic harness.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from nano_moe_ep.types import TopKRouterOutput, _is_integer_tensor


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _uniform_weights(num_tokens: int, k: int, *, normalize: bool, device) -> torch.Tensor:
    value = 1.0 / k if normalize else 1.0
    return torch.full((num_tokens, k), value, dtype=torch.float32, device=device)


def route_topk_round_robin(
    num_tokens: int,
    num_experts: int,
    k: int,
    *,
    normalize: bool = True,
    device: torch.device | str | None = None,
) -> TopKRouterOutput:
    """Assign token ``i`` to experts ``i, i+1, ..., i+k-1`` (mod num_experts)."""

    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or num_tokens < 0:
        raise ValueError("num_tokens must be a non-negative integer")
    _validate_positive_int("num_experts", num_experts)
    _validate_positive_int("k", k)
    if k > num_experts:
        raise ValueError("k must not exceed num_experts (experts must be distinct)")

    base = torch.arange(num_tokens, device=device, dtype=torch.long).reshape(num_tokens, 1)
    offsets = torch.arange(k, device=device, dtype=torch.long).reshape(1, k)
    expert_indices = (base + offsets).remainder(num_experts)
    weights = _uniform_weights(num_tokens, k, normalize=normalize, device=device)
    return TopKRouterOutput(expert_indices=expert_indices, weights=weights)


def route_topk_explicit(
    assignments: Sequence[Sequence[int]] | torch.Tensor,
    num_experts: int,
    *,
    weights: Sequence[Sequence[float]] | torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> TopKRouterOutput:
    """Use a caller-supplied ``[num_tokens, k]`` top-k assignment."""

    _validate_positive_int("num_experts", num_experts)
    if isinstance(assignments, torch.Tensor):
        indices = assignments.detach().clone()
    else:
        indices = torch.as_tensor(assignments)
    if indices.ndim != 2:
        raise ValueError("assignments must have shape [num_tokens, k]")
    if not _is_integer_tensor(indices):
        raise ValueError("assignments must contain integer expert indices")
    indices = indices.to(dtype=torch.long)
    if indices.numel() > 0 and (
        (indices < 0).any().item() or (indices >= num_experts).any().item()
    ):
        raise ValueError("expert indices must be in [0, num_experts)")
    if device is not None:
        indices = indices.to(device=device)

    if weights is None:
        weight_tensor = _uniform_weights(indices.shape[0], indices.shape[1], normalize=True, device=indices.device)
    else:
        weight_tensor = (
            weights.detach().clone() if isinstance(weights, torch.Tensor) else torch.as_tensor(weights)
        )
        weight_tensor = weight_tensor.to(device=indices.device, dtype=torch.float32)
    return TopKRouterOutput(expert_indices=indices, weights=weight_tensor)
