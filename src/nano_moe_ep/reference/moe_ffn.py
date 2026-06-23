from __future__ import annotations

import torch
from torch import nn

from nano_moe_ep.types import ExpertPlacement, ReferenceTrace, RouterOutput, TokenLayout


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_router_for_experts(router_output: RouterOutput, num_tokens: int, num_experts: int) -> None:
    if router_output.expert_indices.shape != (num_tokens, 1):
        raise ValueError("router expert_indices must have shape [num_tokens, 1]")
    if router_output.weights.shape != (num_tokens, 1):
        raise ValueError("router weights must have shape [num_tokens, 1]")
    indices = router_output.expert_indices
    if indices.numel() > 0 and ((indices < 0).any().item() or (indices >= num_experts).any().item()):
        raise ValueError("router expert indices must be in [0, num_experts)")


def build_token_layout(router_output: RouterOutput, num_experts: int) -> TokenLayout:
    """Group top-1 token assignments by expert and record the inverse order."""

    _validate_positive_int("num_experts", num_experts)
    num_tokens = router_output.expert_indices.shape[0]
    _validate_router_for_experts(router_output, num_tokens, num_experts)

    assignments = router_output.expert_indices[:, 0]
    grouped_indices: list[torch.Tensor] = []
    counts: list[int] = []
    for expert_id in range(num_experts):
        token_indices = torch.nonzero(assignments == expert_id, as_tuple=False).flatten()
        grouped_indices.append(token_indices)
        counts.append(int(token_indices.numel()))

    if grouped_indices:
        permutation = torch.cat(grouped_indices).to(dtype=torch.long)
    else:
        permutation = torch.empty((0,), dtype=torch.long, device=assignments.device)

    expert_counts = torch.tensor(counts, dtype=torch.long, device=assignments.device)
    expert_offsets = torch.empty((num_experts + 1,), dtype=torch.long, device=assignments.device)
    expert_offsets[0] = 0
    if num_experts > 0:
        expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    inverse_permutation = torch.empty_like(permutation)
    if permutation.numel() > 0:
        inverse_permutation[permutation] = torch.arange(
            permutation.numel(),
            dtype=torch.long,
            device=permutation.device,
        )

    return TokenLayout(
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        expert_offsets=expert_offsets,
        expert_counts=expert_counts,
    )


class ReferenceMoEFFN(nn.Module):
    """Forward-only, single-process CPU reference MoE FFN for Stage 1."""

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

    def _validate_inputs(self, inputs: torch.Tensor, router_output: RouterOutput) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise ValueError("inputs must be a torch.Tensor")
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape [num_tokens, hidden_dim]")
        if inputs.shape[1] != self.hidden_dim:
            raise ValueError("inputs hidden_dim must match model hidden_dim")
        if not isinstance(router_output, RouterOutput):
            raise ValueError("router_output must be a RouterOutput")
        _validate_router_for_experts(router_output, inputs.shape[0], self.num_experts)

    def forward(self, inputs: torch.Tensor, router_output: RouterOutput) -> tuple[torch.Tensor, ReferenceTrace]:
        self._validate_inputs(inputs, router_output)
        layout = build_token_layout(router_output, self.num_experts)
        placement = ExpertPlacement.single_rank(self.num_experts)

        permutation = layout.permutation.to(device=inputs.device)
        inverse_permutation = layout.inverse_permutation.to(device=inputs.device)
        packed_inputs = inputs.index_select(0, permutation)
        packed_outputs = torch.empty_like(packed_inputs)

        for expert_id, expert in enumerate(self.experts):
            start = int(layout.expert_offsets[expert_id].item())
            end = int(layout.expert_offsets[expert_id + 1].item())
            if start == end:
                continue
            packed_outputs[start:end] = expert(packed_inputs[start:end])

        restored = packed_outputs.index_select(0, inverse_permutation)
        weights = router_output.weights.to(device=inputs.device, dtype=inputs.dtype)
        output = restored * weights
        trace = ReferenceTrace(
            router_output=router_output,
            token_layout=layout,
            expert_placement=placement,
        )
        return output, trace

    def token_by_token_oracle(self, inputs: torch.Tensor, router_output: RouterOutput) -> torch.Tensor:
        """Intentionally slow oracle that does not use grouped layout metadata."""

        self._validate_inputs(inputs, router_output)
        output = torch.empty_like(inputs)
        for token_index in range(inputs.shape[0]):
            expert_id = int(router_output.expert_indices[token_index, 0].item())
            weight = router_output.weights[token_index, 0].to(device=inputs.device, dtype=inputs.dtype)
            token_output = self.experts[expert_id](inputs[token_index : token_index + 1])
            output[token_index] = token_output.squeeze(0) * weight
        return output
