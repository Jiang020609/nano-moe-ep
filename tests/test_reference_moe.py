import pytest
import torch

from nano_moe_ep.reference import ReferenceMoEFFN, build_token_layout
from nano_moe_ep.routing import route_explicit, route_round_robin
from nano_moe_ep.types import RouterOutput


RTOL = 1e-5
ATOL = 1e-6


def _model_and_inputs(seed: int = 1234):
    torch.manual_seed(seed)
    model = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)
    inputs = torch.randn(13, 8)
    return model, inputs


def test_output_shape_is_num_tokens_by_hidden_dim():
    model, inputs = _model_and_inputs()
    router_output = route_round_robin(num_tokens=inputs.shape[0], num_experts=model.num_experts)

    output, trace = model(inputs, router_output)

    assert output.shape == (13, 8)
    assert trace.router_output is router_output
    assert tuple(trace.expert_placement.owner_rank_by_expert) == (0, 0, 0, 0)


def test_grouped_output_matches_token_by_token_oracle():
    model, inputs = _model_and_inputs()
    router_output = route_round_robin(num_tokens=inputs.shape[0], num_experts=model.num_experts)

    grouped_output, _ = model(inputs, router_output)
    oracle_output = model.token_by_token_oracle(inputs, router_output)

    torch.testing.assert_close(grouped_output, oracle_output, rtol=RTOL, atol=ATOL)


def test_one_token_path_matches_oracle_and_records_empty_experts():
    torch.manual_seed(11)
    model = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)
    inputs = torch.randn(1, 8)
    router_output = route_explicit([1], num_experts=model.num_experts, num_tokens=1)

    grouped_output, trace = model(inputs, router_output)
    oracle_output = model.token_by_token_oracle(inputs, router_output)

    torch.testing.assert_close(grouped_output, oracle_output, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(trace.token_layout.expert_counts, torch.tensor([0, 1, 0, 0]))
    torch.testing.assert_close(trace.token_layout.expert_offsets, torch.tensor([0, 0, 1, 1, 1]))


def test_all_tokens_to_one_expert_matches_oracle_and_preserves_tokens():
    torch.manual_seed(22)
    model = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)
    inputs = torch.randn(9, 8)
    router_output = route_explicit([2] * inputs.shape[0], num_experts=model.num_experts, num_tokens=9)

    grouped_output, trace = model(inputs, router_output)
    oracle_output = model.token_by_token_oracle(inputs, router_output)

    torch.testing.assert_close(grouped_output, oracle_output, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(trace.token_layout.expert_counts, torch.tensor([0, 0, 9, 0]))
    torch.testing.assert_close(torch.sort(trace.token_layout.permutation).values, torch.arange(9))


def test_non_unit_weights_are_applied_once_and_match_oracle():
    torch.manual_seed(33)
    model = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)
    inputs = torch.randn(4, 8)
    expert_indices = torch.tensor([[0], [1], [2], [3]])
    weights = torch.tensor([[0.25], [0.5], [1.0], [1.75]])
    router_output = RouterOutput(expert_indices=expert_indices, weights=weights)

    grouped_output, _ = model(inputs, router_output)
    oracle_output = model.token_by_token_oracle(inputs, router_output)

    torch.testing.assert_close(grouped_output, oracle_output, rtol=RTOL, atol=ATOL)

    unweighted_router_output = RouterOutput(
        expert_indices=expert_indices,
        weights=torch.ones_like(weights),
    )
    unweighted_output = model.token_by_token_oracle(inputs, unweighted_router_output)
    twice_weighted_output = unweighted_output * weights * weights

    assert not torch.allclose(grouped_output, unweighted_output, rtol=RTOL, atol=ATOL)
    assert not torch.allclose(grouped_output, twice_weighted_output, rtol=RTOL, atol=ATOL)


def test_non_contiguous_input_matches_oracle():
    torch.manual_seed(44)
    model = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)
    base = torch.randn(8, 6)
    inputs = base.t()
    assert inputs.shape == (6, 8)
    assert not inputs.is_contiguous()
    router_output = route_round_robin(num_tokens=inputs.shape[0], num_experts=model.num_experts)

    grouped_output, _ = model(inputs, router_output)
    oracle_output = model.token_by_token_oracle(inputs, router_output)

    torch.testing.assert_close(grouped_output, oracle_output, rtol=RTOL, atol=ATOL)


def test_skewed_assignment_still_matches_oracle():
    model, _ = _model_and_inputs()
    torch.manual_seed(2026)
    inputs = torch.randn(17, 8)
    assignments = [2] * 15 + [0, 3]
    router_output = route_explicit(assignments, num_experts=model.num_experts, num_tokens=17)

    grouped_output, trace = model(inputs, router_output)
    oracle_output = model.token_by_token_oracle(inputs, router_output)

    torch.testing.assert_close(grouped_output, oracle_output, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(trace.token_layout.expert_counts, torch.tensor([1, 0, 15, 1]))


def test_every_token_appears_once_and_inverse_restores_order():
    assignments = [0, 2, 2, 1, 2, 0, 3]
    router_output = route_explicit(assignments, num_experts=4)
    layout = build_token_layout(router_output, num_experts=4)

    torch.testing.assert_close(layout.permutation, torch.tensor([0, 5, 3, 1, 2, 4, 6]))
    torch.testing.assert_close(torch.sort(layout.permutation).values, torch.arange(7))

    marker = torch.arange(7)
    packed = marker.index_select(0, layout.permutation)
    restored = packed.index_select(0, layout.inverse_permutation)
    torch.testing.assert_close(restored, marker)


def test_expert_counts_and_offsets_match_assignments():
    assignments = [0, 2, 2, 1, 2, 0, 3]
    router_output = route_explicit(assignments, num_experts=4)
    layout = build_token_layout(router_output, num_experts=4)

    torch.testing.assert_close(layout.expert_counts, torch.tensor([2, 1, 3, 1]))
    torch.testing.assert_close(layout.expert_offsets, torch.tensor([0, 2, 3, 6, 7]))


def test_no_token_is_dropped_or_silently_duplicated():
    router_output = route_explicit([3, 3, 0, 1, 3, 2, 0, 1, 3, 2, 2], num_experts=4)
    layout = build_token_layout(router_output, num_experts=4)

    assert layout.permutation.numel() == router_output.expert_indices.shape[0]
    assert layout.expert_counts.sum().item() == router_output.expert_indices.shape[0]
    assert torch.unique(layout.permutation).numel() == router_output.expert_indices.shape[0]


def test_repeated_fixed_seed_runs_are_deterministic():
    def run_once():
        model, inputs = _model_and_inputs(seed=777)
        router_output = route_round_robin(num_tokens=inputs.shape[0], num_experts=model.num_experts)
        output, trace = model(inputs, router_output)
        return output, trace.token_layout.permutation

    output_a, permutation_a = run_once()
    output_b, permutation_b = run_once()

    torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)
    torch.testing.assert_close(permutation_a, permutation_b)


def test_invalid_expert_ids_fail_with_clear_value_error():
    model, inputs = _model_and_inputs()
    router_output = RouterOutput(
        expert_indices=torch.tensor([[0], [1], [4], [2], [0], [1], [2], [3], [0], [1], [2], [3], [0]]),
        weights=torch.ones(13, 1),
    )

    with pytest.raises(ValueError, match="router expert indices must be in \\[0, num_experts\\)"):
        model(inputs, router_output)


def test_invalid_input_shape_fails_with_clear_value_error():
    model, _ = _model_and_inputs()
    router_output = route_round_robin(num_tokens=13, num_experts=model.num_experts)

    with pytest.raises(ValueError, match="inputs must have shape \\[num_tokens, hidden_dim\\]"):
        model(torch.randn(13, 8, 1), router_output)


def test_invalid_router_shapes_and_weight_shapes_fail_with_clear_value_error():
    with pytest.raises(ValueError, match="expert_indices must have shape \\[num_tokens, 1\\]"):
        RouterOutput(expert_indices=torch.tensor([0, 1, 2]), weights=torch.ones(3, 1))

    with pytest.raises(ValueError, match="weights must have shape \\[num_tokens, 1\\]"):
        RouterOutput(expert_indices=torch.tensor([[0], [1], [2]]), weights=torch.ones(3))


@pytest.mark.parametrize("weights", [torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]])])
def test_router_output_rejects_non_finite_weights(weights):
    with pytest.raises(ValueError, match="finite"):
        RouterOutput(expert_indices=torch.tensor([[0]]), weights=weights)


def test_router_output_rejects_non_floating_weight_dtype():
    with pytest.raises(ValueError, match="floating dtype"):
        RouterOutput(expert_indices=torch.tensor([[0]]), weights=torch.tensor([[1]]))
