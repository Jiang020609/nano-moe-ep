"""Forward-only top-k reference MoE FFN with capacity and token dropping.

This extends the Stage 1 top-1 reference to ``k`` experts per token and adds an
optional capacity factor. With ``capacity_factor=None`` no tokens are dropped
and the result is the plain weighted sum of every selected expert. With a
capacity factor, assignments beyond an expert's capacity are dropped per the
policy in ``routing.capacity``; a dropped assignment simply contributes nothing
to its token's output (the token keeps the contributions of its kept experts).
"""

from __future__ import annotations

import torch
from torch import nn

from nano_moe_ep.routing.capacity import build_capacity_mask, compute_expert_capacity, expert_load
from nano_moe_ep.types import TopKReferenceTrace, TopKRouterOutput


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


class TopKReferenceMoEFFN(nn.Module):
    """Forward-only, single-process CPU top-k reference MoE FFN."""

    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int) -> None:
        super().__init__()
        _validate_positive_int("hidden_dim", hidden_dim)
        _validate_positive_int("ffn_dim", ffn_dim)
        _validate_positive_int("num_experts", num_experts)
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def _validate_inputs(self, inputs: torch.Tensor, router_output: TopKRouterOutput) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise ValueError("inputs must be a torch.Tensor")
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape [num_tokens, hidden_dim]")
        if inputs.shape[1] != self.hidden_dim:
            raise ValueError("inputs hidden_dim must match model hidden_dim")
        if not isinstance(router_output, TopKRouterOutput):
            raise ValueError("router_output must be a TopKRouterOutput")
        if router_output.num_tokens != inputs.shape[0]:
            raise ValueError("router_output num_tokens must match inputs")
        if router_output.expert_indices.numel() > 0 and (
            router_output.expert_indices >= self.num_experts
        ).any().item():
            raise ValueError("router expert indices must be in [0, num_experts)")

    def _keep_mask_and_capacity(
        self, router_output: TopKRouterOutput, capacity_factor: float | None
    ) -> tuple[torch.Tensor, int | None]:
        if capacity_factor is None:
            return torch.ones_like(router_output.expert_indices, dtype=torch.bool), None
        capacity = compute_expert_capacity(
            router_output.num_tokens, self.num_experts, router_output.k, capacity_factor
        )
        keep = build_capacity_mask(router_output, self.num_experts, capacity)
        return keep, capacity

    def forward(
        self,
        inputs: torch.Tensor,
        router_output: TopKRouterOutput,
        *,
        capacity_factor: float | None = None,
    ) -> tuple[torch.Tensor, TopKReferenceTrace]:
        self._validate_inputs(inputs, router_output)
        keep, capacity = self._keep_mask_and_capacity(router_output, capacity_factor)

        output = torch.zeros_like(inputs)
        for slot in range(router_output.k):
            slot_experts = router_output.expert_indices[:, slot]
            slot_weights = router_output.weights[:, slot].to(dtype=inputs.dtype)
            slot_keep = keep[:, slot]
            for expert_id, expert in enumerate(self.experts):
                selected = (slot_experts == expert_id) & slot_keep
                positions = torch.nonzero(selected, as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                expert_out = expert(inputs.index_select(0, positions))
                weighted = expert_out * slot_weights.index_select(0, positions).unsqueeze(1)
                output.index_add_(0, positions, weighted)

        trace = TopKReferenceTrace(
            router_output=router_output,
            expert_load=expert_load(router_output, self.num_experts),
            capacity=capacity,
            num_dropped=int((~keep).sum().item()),
        )
        return output, trace

    def token_by_token_oracle(
        self,
        inputs: torch.Tensor,
        router_output: TopKRouterOutput,
        *,
        capacity_factor: float | None = None,
    ) -> torch.Tensor:
        """Intentionally slow oracle using the same capacity mask as the grouped path."""

        self._validate_inputs(inputs, router_output)
        keep, _ = self._keep_mask_and_capacity(router_output, capacity_factor)

        output = torch.zeros_like(inputs)
        for token_index in range(inputs.shape[0]):
            for slot in range(router_output.k):
                if not bool(keep[token_index, slot].item()):
                    continue
                expert_id = int(router_output.expert_indices[token_index, slot].item())
                weight = router_output.weights[token_index, slot].to(dtype=inputs.dtype)
                token_out = self.experts[expert_id](inputs[token_index : token_index + 1])
                output[token_index] += token_out.squeeze(0) * weight
        return output
