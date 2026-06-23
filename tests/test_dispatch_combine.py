import pytest
import torch

from nano_moe_ep.dispatch_combine import build_token_assignments, run_logical_ep_moe
from nano_moe_ep.reference import ReferenceMoEFFN
from nano_moe_ep.routing import route_explicit, route_round_robin
from nano_moe_ep.types import CombinePlan, ExpertPlacement, RouterOutput


RTOL = 1e-5
ATOL = 1e-6


def _model_and_inputs(seed: int = 2027, num_tokens: int = 11):
    torch.manual_seed(seed)
    model = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)
    inputs = torch.randn(num_tokens, 8)
    return model, inputs


def _assert_matches_stage1(model, inputs, router_output, placement):
    stage1_output, _ = model(inputs, router_output)
    stage2_output, trace = run_logical_ep_moe(inputs, router_output, model.experts, placement)
    torch.testing.assert_close(stage2_output, stage1_output, rtol=RTOL, atol=ATOL)
    return stage2_output, trace


def test_basic_local_dispatch_combine_output_matches_stage1_reference():
    model, inputs = _model_and_inputs()
    router_output = route_round_robin(num_tokens=inputs.shape[0], num_experts=model.num_experts)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0, 2], 1: [1, 3]},
        num_experts=model.num_experts,
        num_ep_ranks=2,
    )

    _, trace = _assert_matches_stage1(model, inputs, router_output, placement)

    torch.testing.assert_close(trace.token_layout.rank_counts, torch.tensor([6, 5]))
    assert len(trace.dispatch_plan.payload_token_indices_by_rank) == 2


def test_one_token_case_represents_empty_experts_and_ranks():
    model, inputs = _model_and_inputs(seed=1, num_tokens=1)
    router_output = route_explicit([1], num_experts=model.num_experts, num_tokens=1)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0], 1: [1], 2: [2, 3]},
        num_experts=model.num_experts,
        num_ep_ranks=4,
    )

    _, trace = _assert_matches_stage1(model, inputs, router_output, placement)

    torch.testing.assert_close(trace.token_layout.expert_counts, torch.tensor([0, 1, 0, 0]))
    torch.testing.assert_close(trace.token_layout.rank_counts, torch.tensor([0, 1, 0, 0]))
    assert trace.dispatch_plan.payload_token_indices_by_rank == ((), (0,), (), ())


def test_all_tokens_routed_to_one_expert_matches_reference_and_preserves_tokens():
    model, inputs = _model_and_inputs(seed=2, num_tokens=9)
    router_output = route_explicit([2] * inputs.shape[0], num_experts=model.num_experts, num_tokens=9)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0], 1: [1], 2: [2], 3: [3]},
        num_experts=model.num_experts,
        num_ep_ranks=4,
    )

    _, trace = _assert_matches_stage1(model, inputs, router_output, placement)

    torch.testing.assert_close(trace.token_layout.expert_counts, torch.tensor([0, 0, 9, 0]))
    torch.testing.assert_close(trace.token_layout.rank_counts, torch.tensor([0, 0, 9, 0]))
    torch.testing.assert_close(torch.sort(trace.token_layout.permutation).values, torch.arange(9))


def test_empty_experts_are_represented_in_layout_metadata():
    model, inputs = _model_and_inputs(seed=3, num_tokens=7)
    router_output = route_explicit([0, 0, 3, 0, 3, 0, 3], num_experts=model.num_experts, num_tokens=7)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0, 1], 1: [2, 3]},
        num_experts=model.num_experts,
        num_ep_ranks=2,
    )

    _, trace = _assert_matches_stage1(model, inputs, router_output, placement)

    torch.testing.assert_close(trace.token_layout.expert_counts, torch.tensor([4, 0, 0, 3]))
    torch.testing.assert_close(
        trace.token_layout.rank_expert_counts,
        torch.tensor([[4, 0, 0, 0], [0, 0, 0, 3]]),
    )


def test_empty_logical_ranks_are_represented_in_dispatch_metadata():
    model, inputs = _model_and_inputs(seed=4, num_tokens=8)
    router_output = route_explicit([0, 1, 0, 1, 0, 1, 0, 1], num_experts=model.num_experts, num_tokens=8)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0, 1], 1: [], 2: [2, 3]},
        num_experts=model.num_experts,
        num_ep_ranks=3,
    )

    _, trace = _assert_matches_stage1(model, inputs, router_output, placement)

    torch.testing.assert_close(trace.token_layout.rank_counts, torch.tensor([8, 0, 0]))
    assert trace.dispatch_plan.payload_token_indices_by_rank[1] == ()
    assert trace.dispatch_plan.payload_token_indices_by_rank[2] == ()


def test_uneven_expert_placement_uses_rank_then_expert_then_token_ordering():
    model, inputs = _model_and_inputs(seed=5, num_tokens=10)
    router_output = route_explicit([3, 1, 0, 2, 3, 1, 0, 2, 1, 3], num_experts=model.num_experts)
    placement = ExpertPlacement.from_rank_experts(
        {0: [1, 3], 1: [0], 2: [2]},
        num_experts=model.num_experts,
        num_ep_ranks=3,
    )

    _, trace = _assert_matches_stage1(model, inputs, router_output, placement)

    torch.testing.assert_close(trace.token_layout.permutation, torch.tensor([1, 5, 8, 0, 4, 9, 2, 6, 3, 7]))
    torch.testing.assert_close(trace.token_layout.rank_counts, torch.tensor([6, 2, 2]))


def test_non_unit_routing_weights_are_applied_exactly_once():
    model, inputs = _model_and_inputs(seed=6, num_tokens=4)
    expert_indices = torch.tensor([[0], [1], [2], [3]])
    weights = torch.tensor([[0.25], [0.5], [1.0], [1.75]])
    router_output = RouterOutput(expert_indices=expert_indices, weights=weights)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0, 2], 1: [1, 3]},
        num_experts=model.num_experts,
        num_ep_ranks=2,
    )

    stage2_output, trace = _assert_matches_stage1(model, inputs, router_output, placement)
    unweighted_output, _ = run_logical_ep_moe(
        inputs,
        RouterOutput(expert_indices=expert_indices, weights=torch.ones_like(weights)),
        model.experts,
        placement,
    )
    twice_weighted = unweighted_output * weights * weights

    assert not torch.allclose(stage2_output, unweighted_output, rtol=RTOL, atol=ATOL)
    assert not torch.allclose(stage2_output, twice_weighted, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(trace.combine_plan.routing_weights, weights.index_select(0, trace.token_layout.permutation))


def test_token_order_is_restored_exactly_after_combine():
    model, inputs = _model_and_inputs(seed=7, num_tokens=7)
    router_output = route_explicit([2, 0, 3, 1, 2, 0, 3], num_experts=model.num_experts)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0, 1], 1: [2, 3]},
        num_experts=model.num_experts,
        num_ep_ranks=2,
    )

    _, trace = _assert_matches_stage1(model, inputs, router_output, placement)

    marker = torch.arange(inputs.shape[0])
    packed = marker.index_select(0, trace.token_layout.permutation)
    restored = packed.index_select(0, trace.token_layout.inverse_permutation)
    torch.testing.assert_close(restored, marker)


def test_no_token_is_silently_lost_or_duplicated():
    model, inputs = _model_and_inputs(seed=8, num_tokens=13)
    router_output = route_explicit([0, 3, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2], num_experts=4)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0], 1: [1, 2], 2: [3]},
        num_experts=model.num_experts,
        num_ep_ranks=3,
    )

    _, trace = _assert_matches_stage1(model, inputs, router_output, placement)

    assert len(trace.assignments) == inputs.shape[0]
    assert trace.token_layout.permutation.numel() == inputs.shape[0]
    assert torch.unique(trace.token_layout.permutation).numel() == inputs.shape[0]
    assert trace.combine_plan.token_indices.numel() == inputs.shape[0]


def test_combine_plan_rejects_missing_or_duplicated_token_indices():
    with pytest.raises(ValueError, match="every token index exactly once"):
        CombinePlan(
            token_indices=torch.tensor([0, 0, 1]),
            routing_weights=torch.ones(3, 1),
            num_tokens=3,
        )


def test_non_contiguous_input_matches_stage1_reference():
    torch.manual_seed(9)
    model = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)
    base = torch.randn(8, 6)
    inputs = base.t()
    assert inputs.shape == (6, 8)
    assert not inputs.is_contiguous()
    router_output = route_round_robin(num_tokens=inputs.shape[0], num_experts=model.num_experts)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0], 1: [1], 2: [2, 3]},
        num_experts=model.num_experts,
        num_ep_ranks=3,
    )

    _assert_matches_stage1(model, inputs, router_output, placement)


def test_invalid_expert_placement_is_rejected():
    with pytest.raises(ValueError, match="exactly once"):
        ExpertPlacement.from_rank_experts({0: [0], 1: [1]}, num_experts=3, num_ep_ranks=2)
    with pytest.raises(ValueError, match="exactly once"):
        ExpertPlacement.from_rank_experts({0: [0, 1], 1: [1, 2]}, num_experts=3, num_ep_ranks=2)
    with pytest.raises(ValueError, match="rank ids must be in"):
        ExpertPlacement.from_rank_experts({0: [0], 2: [1]}, num_experts=2, num_ep_ranks=2)
    with pytest.raises(ValueError, match="rank ids must be non-negative"):
        ExpertPlacement.from_rank_experts({-1: [0], 0: [1]}, num_experts=2, num_ep_ranks=2)


def test_deterministic_layout_ordering_is_stable_across_repeated_runs():
    router_output = route_explicit([2, 0, 3, 1, 2, 0, 3, 1], num_experts=4)
    placement = ExpertPlacement.from_rank_experts(
        {0: [1, 3], 1: [0], 2: [2]},
        num_experts=4,
        num_ep_ranks=3,
    )

    assignments_a = build_token_assignments(router_output, placement)
    _, trace_a = run_logical_ep_moe(torch.randn(8, 8), router_output, ReferenceMoEFFN(8, 16, 4).experts, placement)
    assignments_b = build_token_assignments(router_output, placement)
    _, trace_b = run_logical_ep_moe(torch.randn(8, 8), router_output, ReferenceMoEFFN(8, 16, 4).experts, placement)

    assert assignments_a == assignments_b
    torch.testing.assert_close(trace_a.token_layout.permutation, trace_b.token_layout.permutation)
    torch.testing.assert_close(trace_a.token_layout.rank_counts, trace_b.token_layout.rank_counts)
    torch.testing.assert_close(trace_a.token_layout.rank_offsets, trace_b.token_layout.rank_offsets)
    torch.testing.assert_close(trace_a.token_layout.rank_expert_counts, trace_b.token_layout.rank_expert_counts)
