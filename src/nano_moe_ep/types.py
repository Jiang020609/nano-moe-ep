from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import torch


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    if tensor.dtype is torch.bool:
        return False
    try:
        torch.iinfo(tensor.dtype)
    except TypeError:
        return False
    return True


class ExecutionMode(str, Enum):
    """Execution mode metadata for one EP forward path."""

    REFERENCE = "reference"
    LOGICAL_SINGLE_PROCESS = "logical_single_process"
    FUTURE_DISTRIBUTED = "future_distributed"


@dataclass(frozen=True)
class EPContext:
    """Execution metadata for one logical Expert Parallel run."""

    num_ep_ranks: int
    local_rank: int | None = None
    execution_mode: ExecutionMode = ExecutionMode.LOGICAL_SINGLE_PROCESS
    device: str | None = "cpu"
    deterministic: bool = True
    phase: str = "forward"

    def __post_init__(self) -> None:
        if not isinstance(self.num_ep_ranks, int) or isinstance(self.num_ep_ranks, bool):
            raise ValueError("num_ep_ranks must be an integer")
        if self.num_ep_ranks <= 0:
            raise ValueError("num_ep_ranks must be positive")
        if self.local_rank is not None:
            if not isinstance(self.local_rank, int) or isinstance(self.local_rank, bool):
                raise ValueError("local_rank must be an integer or None")
            if self.local_rank < 0 or self.local_rank >= self.num_ep_ranks:
                raise ValueError("local_rank must be in [0, num_ep_ranks)")
        if not isinstance(self.execution_mode, ExecutionMode):
            try:
                execution_mode = ExecutionMode(self.execution_mode)
            except ValueError as exc:
                raise ValueError("execution_mode must be a valid ExecutionMode") from exc
            object.__setattr__(self, "execution_mode", execution_mode)
        if self.device is not None and self.device != "cpu":
            raise ValueError("device must be None or 'cpu' for Stage 2 logical execution")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a bool")
        if not isinstance(self.phase, str) or self.phase == "":
            raise ValueError("phase must be a non-empty string")

    @classmethod
    def single_process(cls, *, num_ep_ranks: int, phase: str = "forward") -> "EPContext":
        """Create the default Stage 2 logical single-process context."""

        return cls(
            num_ep_ranks=num_ep_ranks,
            local_rank=None,
            execution_mode=ExecutionMode.LOGICAL_SINGLE_PROCESS,
            device="cpu",
            deterministic=True,
            phase=phase,
        )

    def require_compatible_placement(self, expert_placement: "ExpertPlacement") -> None:
        """Check that execution metadata agrees with expert ownership metadata."""

        if not isinstance(expert_placement, ExpertPlacement):
            raise ValueError("expert_placement must be an ExpertPlacement")
        if self.num_ep_ranks != expert_placement.num_ep_ranks:
            raise ValueError("EPContext num_ep_ranks must match ExpertPlacement num_ep_ranks")


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
    rank_offsets: torch.Tensor | None = None
    rank_counts: torch.Tensor | None = None
    rank_expert_offsets: torch.Tensor | None = None
    rank_expert_counts: torch.Tensor | None = None

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

        if any(
            tensor is not None
            for tensor in (
                self.rank_offsets,
                self.rank_counts,
                self.rank_expert_offsets,
                self.rank_expert_counts,
            )
        ):
            if (
                self.rank_offsets is None
                or self.rank_counts is None
                or self.rank_expert_offsets is None
                or self.rank_expert_counts is None
            ):
                raise ValueError("rank layout metadata must be provided as a complete set")
            self._validate_rank_layout(num_tokens)

    def _validate_rank_layout(self, num_tokens: int) -> None:
        assert self.rank_offsets is not None
        assert self.rank_counts is not None
        assert self.rank_expert_offsets is not None
        assert self.rank_expert_counts is not None

        for name, tensor in (
            ("rank_offsets", self.rank_offsets),
            ("rank_counts", self.rank_counts),
            ("rank_expert_offsets", self.rank_expert_offsets),
            ("rank_expert_counts", self.rank_expert_counts),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            if not _is_integer_tensor(tensor):
                raise ValueError(f"{name} must use an integer dtype")

        if self.rank_offsets.ndim != 1:
            raise ValueError("rank_offsets must be a 1D tensor")
        if self.rank_counts.ndim != 1:
            raise ValueError("rank_counts must be a 1D tensor")
        if self.rank_expert_offsets.ndim != 2:
            raise ValueError("rank_expert_offsets must be a 2D tensor")
        if self.rank_expert_counts.ndim != 2:
            raise ValueError("rank_expert_counts must be a 2D tensor")
        if self.rank_offsets.numel() != self.rank_counts.numel() + 1:
            raise ValueError("rank_offsets must have shape [num_ep_ranks + 1]")
        if self.rank_offsets[0].item() != 0:
            raise ValueError("rank_offsets must start at 0")
        if self.rank_offsets[-1].item() != num_tokens:
            raise ValueError("rank_offsets must end at num_tokens")
        if (self.rank_counts < 0).any().item():
            raise ValueError("rank_counts must be non-negative")
        if not torch.equal(self.rank_offsets[1:] - self.rank_offsets[:-1], self.rank_counts):
            raise ValueError("rank_offsets differences must match rank_counts")

        num_ranks = self.rank_counts.numel()
        num_experts = self.expert_counts.numel()
        if self.rank_expert_counts.shape != (num_ranks, num_experts):
            raise ValueError("rank_expert_counts must have shape [num_ep_ranks, num_experts]")
        if self.rank_expert_offsets.shape != (num_ranks, num_experts + 1):
            raise ValueError("rank_expert_offsets must have shape [num_ep_ranks, num_experts + 1]")
        if (self.rank_expert_counts < 0).any().item():
            raise ValueError("rank_expert_counts must be non-negative")
        if not torch.equal(
            self.rank_expert_offsets[:, 0],
            torch.zeros(num_ranks, dtype=self.rank_expert_offsets.dtype, device=self.rank_expert_offsets.device),
        ):
            raise ValueError("rank_expert_offsets must start each rank at 0")
        if not torch.equal(self.rank_expert_offsets[:, -1], self.rank_counts):
            raise ValueError("rank_expert_offsets must end each rank at its rank_count")
        if not torch.equal(
            self.rank_expert_offsets[:, 1:] - self.rank_expert_offsets[:, :-1],
            self.rank_expert_counts,
        ):
            raise ValueError("rank_expert_offsets differences must match rank_expert_counts")
        if not torch.equal(self.rank_expert_counts.sum(dim=1), self.rank_counts):
            raise ValueError("rank_expert_counts row sums must match rank_counts")
        if not torch.equal(self.rank_expert_counts.sum(dim=0), self.expert_counts):
            raise ValueError("rank_expert_counts column sums must match expert_counts")


@dataclass(frozen=True)
class ExpertPlacement:
    """Immutable Stage 1 expert ownership metadata."""

    owner_rank_by_expert: Sequence[int]
    num_ep_ranks: int | None = None

    def __post_init__(self) -> None:
        owners = tuple(self.owner_rank_by_expert)
        if len(owners) == 0:
            raise ValueError("owner_rank_by_expert must contain at least one expert")
        for rank in owners:
            if not isinstance(rank, int) or isinstance(rank, bool):
                raise ValueError("owner_rank_by_expert must contain integer ranks")
            if rank < 0:
                raise ValueError("owner_rank_by_expert ranks must be non-negative")
        num_ep_ranks = self.num_ep_ranks
        if num_ep_ranks is None:
            num_ep_ranks = max(owners) + 1
        if not isinstance(num_ep_ranks, int) or isinstance(num_ep_ranks, bool):
            raise ValueError("num_ep_ranks must be an integer")
        if num_ep_ranks <= 0:
            raise ValueError("num_ep_ranks must be positive")
        if any(rank >= num_ep_ranks for rank in owners):
            raise ValueError("owner_rank_by_expert ranks must be in [0, num_ep_ranks)")
        object.__setattr__(self, "owner_rank_by_expert", owners)
        object.__setattr__(self, "num_ep_ranks", num_ep_ranks)

    @classmethod
    def single_rank(cls, num_experts: int) -> "ExpertPlacement":
        if not isinstance(num_experts, int) or isinstance(num_experts, bool):
            raise ValueError("num_experts must be an integer")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        return cls(owner_rank_by_expert=(0,) * num_experts, num_ep_ranks=1)

    @classmethod
    def from_rank_experts(
        cls,
        rank_to_experts: Mapping[int, Sequence[int]],
        *,
        num_experts: int,
        num_ep_ranks: int,
    ) -> "ExpertPlacement":
        if not isinstance(num_experts, int) or isinstance(num_experts, bool):
            raise ValueError("num_experts must be an integer")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not isinstance(num_ep_ranks, int) or isinstance(num_ep_ranks, bool):
            raise ValueError("num_ep_ranks must be an integer")
        if num_ep_ranks <= 0:
            raise ValueError("num_ep_ranks must be positive")

        owner_by_expert = [-1] * num_experts
        for rank, expert_ids in rank_to_experts.items():
            if not isinstance(rank, int) or isinstance(rank, bool):
                raise ValueError("rank ids must be integers")
            if rank < 0:
                raise ValueError("rank ids must be non-negative")
            if rank >= num_ep_ranks:
                raise ValueError("rank ids must be in [0, num_ep_ranks)")
            for expert_id in expert_ids:
                if not isinstance(expert_id, int) or isinstance(expert_id, bool):
                    raise ValueError("expert ids must be integers")
                if expert_id < 0 or expert_id >= num_experts:
                    raise ValueError("expert ids must be in [0, num_experts)")
                if owner_by_expert[expert_id] != -1:
                    raise ValueError("each expert must appear exactly once")
                owner_by_expert[expert_id] = rank

        if any(rank == -1 for rank in owner_by_expert):
            raise ValueError("each expert must appear exactly once")
        return cls(owner_rank_by_expert=tuple(owner_by_expert), num_ep_ranks=num_ep_ranks)

    @property
    def num_experts(self) -> int:
        return len(self.owner_rank_by_expert)

    def owner_rank(self, expert_id: int) -> int:
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError("expert_id must be in [0, num_experts)")
        return self.owner_rank_by_expert[expert_id]

    def experts_for_rank(self, ep_rank: int) -> tuple[int, ...]:
        if ep_rank < 0 or ep_rank >= self.num_ep_ranks:
            raise ValueError("ep_rank must be in [0, num_ep_ranks)")
        return tuple(
            expert_id
            for expert_id, owner_rank in enumerate(self.owner_rank_by_expert)
            if owner_rank == ep_rank
        )


@dataclass(frozen=True)
class TokenAssignment:
    """One top-1 routed token visit for the logical EP simulation."""

    token_index: int
    expert_id: int
    ep_rank: int
    routing_weight: float
    original_position: int

    def __post_init__(self) -> None:
        for name, value in (
            ("token_index", self.token_index),
            ("expert_id", self.expert_id),
            ("ep_rank", self.ep_rank),
            ("original_position", self.original_position),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.routing_weight, float):
            raise ValueError("routing_weight must be a float")


@dataclass(frozen=True)
class DispatchPlan:
    """Variable-size logical-rank dispatch metadata for single-process simulation."""

    layout: TokenLayout
    assignments: tuple[TokenAssignment, ...]
    payload_token_indices_by_rank: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.layout.rank_counts is None:
            raise ValueError("DispatchPlan requires rank-aware TokenLayout")
        if len(self.payload_token_indices_by_rank) != self.layout.rank_counts.numel():
            raise ValueError("payload_token_indices_by_rank must have one payload per rank")
        payload_count = sum(len(payload) for payload in self.payload_token_indices_by_rank)
        if payload_count != self.layout.permutation.numel():
            raise ValueError("dispatch payload token count must match layout token count")
        if len(self.assignments) != self.layout.permutation.numel():
            raise ValueError("assignments must match layout token count")
        payload_token_indices = tuple(
            token_index
            for payload in self.payload_token_indices_by_rank
            for token_index in payload
        )
        layout_token_indices = tuple(int(token_index.item()) for token_index in self.layout.permutation)
        if payload_token_indices != layout_token_indices:
            raise ValueError("dispatch payload token order must match layout permutation")
        assignment_token_indices = tuple(assignment.token_index for assignment in self.assignments)
        if assignment_token_indices != layout_token_indices:
            raise ValueError("dispatch assignments must match layout permutation")


@dataclass(frozen=True)
class CombinePlan:
    """Plan for restoring logical EP expert outputs to original token order."""

    token_indices: torch.Tensor
    routing_weights: torch.Tensor
    num_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.token_indices, torch.Tensor):
            raise ValueError("token_indices must be a torch.Tensor")
        if self.token_indices.ndim != 1:
            raise ValueError("token_indices must be a 1D tensor")
        if not _is_integer_tensor(self.token_indices):
            raise ValueError("token_indices must use an integer dtype")
        if not isinstance(self.routing_weights, torch.Tensor):
            raise ValueError("routing_weights must be a torch.Tensor")
        if self.routing_weights.shape != (self.token_indices.numel(), 1):
            raise ValueError("routing_weights must have shape [num_assignments, 1]")
        if not self.routing_weights.dtype.is_floating_point:
            raise ValueError("routing_weights must use a floating dtype")
        if not torch.isfinite(self.routing_weights).all().item():
            raise ValueError("routing_weights must be finite")
        if not isinstance(self.num_tokens, int) or isinstance(self.num_tokens, bool):
            raise ValueError("num_tokens must be an integer")
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if self.token_indices.numel() != self.num_tokens:
            raise ValueError("token_indices must contain one entry per token")
        expected = torch.arange(self.num_tokens, dtype=self.token_indices.dtype, device=self.token_indices.device)
        if not torch.equal(torch.sort(self.token_indices).values, expected):
            raise ValueError("token_indices must contain every token index exactly once")


@dataclass(frozen=True)
class ReferenceTrace:
    """Trace emitted by the grouped Stage 1 reference path."""

    router_output: RouterOutput
    token_layout: TokenLayout
    expert_placement: ExpertPlacement
