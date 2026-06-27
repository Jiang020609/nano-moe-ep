from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn

from nano_moe_ep.types import (
    CombinePlan,
    DispatchPlan,
    EPContext,
    ExpertPlacement,
    RouterOutput,
    TokenAssignment,
    TokenLayout,
)


@dataclass(frozen=True)
class LogicalEPTrace:
    """Trace for the Stage 2 single-process logical EP simulation."""

    assignments: tuple[TokenAssignment, ...]
    token_layout: TokenLayout
    dispatch_plan: DispatchPlan
    combine_plan: CombinePlan
    expert_placement: ExpertPlacement
    ep_context: EPContext


def _validate_router(router_output: RouterOutput, *, num_tokens: int, num_experts: int) -> None:
    if router_output.expert_indices.shape != (num_tokens, 1):
        raise ValueError("router expert_indices must have shape [num_tokens, 1]")
    if router_output.weights.shape != (num_tokens, 1):
        raise ValueError("router weights must have shape [num_tokens, 1]")
    expert_indices = router_output.expert_indices
    if expert_indices.numel() > 0 and (
        (expert_indices < 0).any().item() or (expert_indices >= num_experts).any().item()
    ):
        raise ValueError("router expert indices must be in [0, num_experts)")


def _validate_inputs(
    inputs: torch.Tensor,
    router_output: RouterOutput,
    experts: Sequence[nn.Module],
    expert_placement: ExpertPlacement,
) -> None:
    if not isinstance(inputs, torch.Tensor):
        raise ValueError("inputs must be a torch.Tensor")
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [num_tokens, hidden_dim]")
    if not isinstance(router_output, RouterOutput):
        raise ValueError("router_output must be a RouterOutput")
    if not isinstance(expert_placement, ExpertPlacement):
        raise ValueError("expert_placement must be an ExpertPlacement")
    if len(experts) != expert_placement.num_experts:
        raise ValueError("experts length must match expert placement num_experts")
    _validate_router(router_output, num_tokens=inputs.shape[0], num_experts=expert_placement.num_experts)


def build_token_assignments(
    router_output: RouterOutput,
    expert_placement: ExpertPlacement,
) -> tuple[TokenAssignment, ...]:
    """Build one top-1 assignment per token."""

    _validate_router(
        router_output,
        num_tokens=router_output.expert_indices.shape[0],
        num_experts=expert_placement.num_experts,
    )
    assignments: list[TokenAssignment] = []
    for token_index in range(router_output.expert_indices.shape[0]):
        expert_id = int(router_output.expert_indices[token_index, 0].item())
        assignments.append(
            TokenAssignment(
                token_index=token_index,
                expert_id=expert_id,
                ep_rank=expert_placement.owner_rank(expert_id),
                routing_weight=float(router_output.weights[token_index, 0].item()),
                original_position=token_index,
            )
        )
    return tuple(assignments)


def build_logical_ep_layout(
    assignments: Sequence[TokenAssignment],
    *,
    num_tokens: int,
    num_experts: int,
    num_ep_ranks: int,
) -> TokenLayout:
    """Group tokens by destination rank, then expert id, then original token order."""

    sorted_assignments = sorted(
        assignments,
        key=lambda assignment: (assignment.ep_rank, assignment.expert_id, assignment.original_position),
    )
    permutation = torch.tensor([assignment.token_index for assignment in sorted_assignments], dtype=torch.long)
    inverse_permutation = torch.empty_like(permutation)
    if permutation.numel() > 0:
        inverse_permutation[permutation] = torch.arange(permutation.numel(), dtype=torch.long)

    expert_counts = torch.zeros(num_experts, dtype=torch.long)
    rank_counts = torch.zeros(num_ep_ranks, dtype=torch.long)
    rank_expert_counts = torch.zeros((num_ep_ranks, num_experts), dtype=torch.long)
    for assignment in sorted_assignments:
        expert_counts[assignment.expert_id] += 1
        rank_counts[assignment.ep_rank] += 1
        rank_expert_counts[assignment.ep_rank, assignment.expert_id] += 1

    expert_offsets = torch.empty(num_experts + 1, dtype=torch.long)
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    rank_offsets = torch.empty(num_ep_ranks + 1, dtype=torch.long)
    rank_offsets[0] = 0
    rank_offsets[1:] = torch.cumsum(rank_counts, dim=0)

    rank_expert_offsets = torch.empty((num_ep_ranks, num_experts + 1), dtype=torch.long)
    rank_expert_offsets[:, 0] = 0
    rank_expert_offsets[:, 1:] = torch.cumsum(rank_expert_counts, dim=1)

    return TokenLayout(
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        expert_offsets=expert_offsets,
        expert_counts=expert_counts,
        rank_offsets=rank_offsets,
        rank_counts=rank_counts,
        rank_expert_offsets=rank_expert_offsets,
        rank_expert_counts=rank_expert_counts,
    )


def build_dispatch_plan(
    assignments: Sequence[TokenAssignment],
    layout: TokenLayout,
) -> DispatchPlan:
    """Create one logical payload per destination rank."""

    if layout.rank_offsets is None or layout.rank_counts is None:
        raise ValueError("rank-aware TokenLayout is required")
    sorted_assignments = sorted(
        assignments,
        key=lambda assignment: (assignment.ep_rank, assignment.expert_id, assignment.original_position),
    )
    payloads: list[tuple[int, ...]] = []
    for ep_rank in range(layout.rank_counts.numel()):
        start = int(layout.rank_offsets[ep_rank].item())
        end = int(layout.rank_offsets[ep_rank + 1].item())
        payloads.append(tuple(assignment.token_index for assignment in sorted_assignments[start:end]))
    return DispatchPlan(
        layout=layout,
        assignments=tuple(sorted_assignments),
        payload_token_indices_by_rank=tuple(payloads),
    )


def simulate_dispatch(inputs: torch.Tensor, dispatch_plan: DispatchPlan) -> tuple[torch.Tensor, ...]:
    """Create one local tensor payload per logical destination rank."""

    payloads: list[torch.Tensor] = []
    for token_indices in dispatch_plan.payload_token_indices_by_rank:
        index = torch.tensor(token_indices, dtype=torch.long, device=inputs.device)
        payloads.append(inputs.index_select(0, index))
    return tuple(payloads)


def execute_local_experts(
    payloads: Sequence[torch.Tensor],
    dispatch_plan: DispatchPlan,
    experts: Sequence[nn.Module],
) -> torch.Tensor:
    """Run each logical rank's owned expert slices and return outputs in dispatch order."""

    layout = dispatch_plan.layout
    if layout.rank_expert_offsets is None or layout.rank_offsets is None:
        raise ValueError("rank-aware TokenLayout is required")

    rank_outputs: list[torch.Tensor] = []
    for ep_rank, payload in enumerate(payloads):
        rank_output = torch.empty_like(payload)
        rank_start = int(layout.rank_offsets[ep_rank].item())
        for expert_id, expert in enumerate(experts):
            local_start = int(layout.rank_expert_offsets[ep_rank, expert_id].item())
            local_end = int(layout.rank_expert_offsets[ep_rank, expert_id + 1].item())
            if local_start == local_end:
                continue
            rank_output[local_start:local_end] = expert(payload[local_start:local_end])
        if int(layout.rank_offsets[ep_rank + 1].item()) - rank_start != payload.shape[0]:
            raise ValueError("rank payload size must match rank layout count")
        rank_outputs.append(rank_output)

    if not rank_outputs:
        raise ValueError("at least one logical EP rank is required")
    return torch.cat(rank_outputs, dim=0)


def build_combine_plan(dispatch_plan: DispatchPlan, router_output: RouterOutput) -> CombinePlan:
    """Build the plan that restores expert outputs to original token order."""

    token_indices = dispatch_plan.layout.permutation
    weight_indices = token_indices.to(device=router_output.weights.device)
    weights = router_output.weights.index_select(0, weight_indices)
    return CombinePlan(
        token_indices=token_indices,
        routing_weights=weights,
        num_tokens=router_output.expert_indices.shape[0],
    )


def apply_combine_plan(expert_outputs: torch.Tensor, combine_plan: CombinePlan) -> torch.Tensor:
    """Apply routing weights once and restore original token order."""

    if expert_outputs.ndim != 2:
        raise ValueError("expert_outputs must have shape [num_assignments, hidden_dim]")
    if expert_outputs.shape[0] != combine_plan.token_indices.numel():
        raise ValueError("expert_outputs must contain one row per assignment")
    token_indices = combine_plan.token_indices.to(device=expert_outputs.device)
    weights = combine_plan.routing_weights.to(device=expert_outputs.device, dtype=expert_outputs.dtype)
    weighted_outputs = expert_outputs * weights
    output = torch.empty(
        (combine_plan.num_tokens, expert_outputs.shape[1]),
        dtype=expert_outputs.dtype,
        device=expert_outputs.device,
    )
    output.index_copy_(0, token_indices, weighted_outputs)
    return output


def run_logical_ep_moe(
    inputs: torch.Tensor,
    router_output: RouterOutput,
    experts: Sequence[nn.Module],
    expert_placement: ExpertPlacement,
    ep_context: EPContext | None = None,
) -> tuple[torch.Tensor, LogicalEPTrace]:
    """Run the Stage 2 single-process logical EP dispatch/combine simulation."""

    _validate_inputs(inputs, router_output, experts, expert_placement)
    if ep_context is None:
        ep_context = EPContext.single_process(num_ep_ranks=expert_placement.num_ep_ranks)
    elif not isinstance(ep_context, EPContext):
        raise ValueError("ep_context must be an EPContext")
    ep_context.require_compatible_placement(expert_placement)

    assignments = build_token_assignments(router_output, expert_placement)
    layout = build_logical_ep_layout(
        assignments,
        num_tokens=inputs.shape[0],
        num_experts=expert_placement.num_experts,
        num_ep_ranks=expert_placement.num_ep_ranks,
    )
    dispatch_plan = build_dispatch_plan(assignments, layout)
    payloads = simulate_dispatch(inputs, dispatch_plan)
    expert_outputs = execute_local_experts(payloads, dispatch_plan, experts)
    combine_plan = build_combine_plan(dispatch_plan, router_output)
    output = apply_combine_plan(expert_outputs, combine_plan)
    return output, LogicalEPTrace(
        assignments=assignments,
        token_layout=layout,
        dispatch_plan=dispatch_plan,
        combine_plan=combine_plan,
        expert_placement=expert_placement,
        ep_context=ep_context,
    )
