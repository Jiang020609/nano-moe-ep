from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    if tensor.dtype is torch.bool:
        return False
    try:
        torch.iinfo(tensor.dtype)
    except TypeError:
        return False
    return True


@dataclass(frozen=True)
class RouterOutput:
    """Top-1 router output for Stage 1."""

    expert_indices: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.expert_indices, torch.Tensor):
            raise ValueError("expert_indices must be a torch.Tensor")
        if not isinstance(self.weights, torch.Tensor):
            raise ValueError("weights must be a torch.Tensor")
        if self.expert_indices.ndim != 2 or self.expert_indices.shape[1] != 1:
            raise ValueError("expert_indices must have shape [num_tokens, 1]")
        if self.weights.ndim != 2 or self.weights.shape[1] != 1:
            raise ValueError("weights must have shape [num_tokens, 1]")
        if self.expert_indices.shape[0] != self.weights.shape[0]:
            raise ValueError("expert_indices and weights must have the same num_tokens")
        if not _is_integer_tensor(self.expert_indices):
            raise ValueError("expert_indices must use an integer dtype")
        if not self.weights.dtype.is_floating_point:
            raise ValueError("weights must use a floating dtype")
        if not torch.isfinite(self.weights).all().item():
            raise ValueError("weights must be finite")


@dataclass(frozen=True)
class TokenLayout:
    """Permutation metadata for grouping tokens by assigned expert."""

    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor

    def __post_init__(self) -> None:
        for name, tensor in (
            ("permutation", self.permutation),
            ("inverse_permutation", self.inverse_permutation),
            ("expert_offsets", self.expert_offsets),
            ("expert_counts", self.expert_counts),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be a 1D tensor")
            if not _is_integer_tensor(tensor):
                raise ValueError(f"{name} must use an integer dtype")

        num_tokens = self.permutation.numel()
        if self.inverse_permutation.numel() != num_tokens:
            raise ValueError("inverse_permutation must match permutation length")
        if self.expert_offsets.numel() != self.expert_counts.numel() + 1:
            raise ValueError("expert_offsets must have shape [num_experts + 1]")
        if self.expert_offsets.numel() == 0:
            raise ValueError("expert_offsets must include a starting zero")
        if self.expert_offsets[0].item() != 0:
            raise ValueError("expert_offsets must start at 0")
        if self.expert_offsets[-1].item() != num_tokens:
            raise ValueError("expert_offsets must end at num_tokens")
        if (self.expert_counts < 0).any().item():
            raise ValueError("expert_counts must be non-negative")
        if not torch.equal(self.expert_offsets[1:] - self.expert_offsets[:-1], self.expert_counts):
            raise ValueError("expert_offsets differences must match expert_counts")

        expected = torch.arange(num_tokens, dtype=self.permutation.dtype, device=self.permutation.device)
        if not torch.equal(torch.sort(self.permutation).values, expected):
            raise ValueError("permutation must contain every token index exactly once")
        if not torch.equal(torch.sort(self.inverse_permutation).values, expected):
            raise ValueError("inverse_permutation must contain every token index exactly once")
        if num_tokens > 0:
            restored = self.permutation[self.inverse_permutation]
            if not torch.equal(restored, expected):
                raise ValueError("inverse_permutation must restore original token order")


@dataclass(frozen=True)
class ExpertPlacement:
    """Immutable Stage 1 expert ownership metadata."""

    owner_rank_by_expert: Sequence[int]

    def __post_init__(self) -> None:
        owners = tuple(self.owner_rank_by_expert)
        if len(owners) == 0:
            raise ValueError("owner_rank_by_expert must contain at least one expert")
        for rank in owners:
            if not isinstance(rank, int) or isinstance(rank, bool):
                raise ValueError("owner_rank_by_expert must contain integer ranks")
            if rank < 0:
                raise ValueError("owner_rank_by_expert ranks must be non-negative")
        object.__setattr__(self, "owner_rank_by_expert", owners)

    @classmethod
    def single_rank(cls, num_experts: int) -> "ExpertPlacement":
        if not isinstance(num_experts, int) or isinstance(num_experts, bool):
            raise ValueError("num_experts must be an integer")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        return cls(owner_rank_by_expert=(0,) * num_experts)


@dataclass(frozen=True)
class ReferenceTrace:
    """Trace emitted by the grouped Stage 1 reference path."""

    router_output: RouterOutput
    token_layout: TokenLayout
    expert_placement: ExpertPlacement
