import pytest
import torch

from nano_moe_ep.reference import ReferenceMoEFFN, TopKReferenceMoEFFN
from nano_moe_ep.routing import (
    build_capacity_mask,
    compute_expert_capacity,
    expert_load,
    route_explicit,
    route_topk_explicit,
    route_topk_round_robin,
)
from nano_moe_ep.types import TopKRouterOutput

RTOL = 1e-5
ATOL = 1e-6


# --- routers and router type ------------------------------------------------


def test_topk_round_robin_selects_distinct_experts_with_normalized_weights():
    router = route_topk_round_robin(num_tokens=3, num_experts=4, k=2)

    torch.testing.assert_close(router.expert_indices, torch.tensor([[0, 1], [1, 2], [2, 3]]))
    torch.testing.assert_close(router.weights, torch.full((3, 2), 0.5))
    assert router.k == 2 and router.num_tokens == 3


def test_topk_round_robin_rejects_k_larger_than_num_experts():
    with pytest.raises(ValueError, match="k must not exceed num_experts"):
        route_topk_round_robin(num_tokens=2, num_experts=2, k=3)


def test_topk_router_output_rejects_duplicate_experts_per_token():
    with pytest.raises(ValueError, match="distinct experts"):
        TopKRouterOutput(expert_indices=torch.tensor([[1, 1]]), weights=torch.tensor([[0.5, 0.5]]))


# --- capacity and drop policy ----------------------------------------------


def test_compute_expert_capacity_uses_ceil_of_share():
    # ceil(1.25 * 8 tokens * 2 slots / 4 experts) = ceil(5.0) = 5
    assert compute_expert_capacity(num_tokens=8, num_experts=4, k=2, capacity_factor=1.25) == 5
    # ceil(1.0 * 5 * 2 / 4) = ceil(2.5) = 3
    assert compute_expert_capacity(num_tokens=5, num_experts=4, k=2, capacity_factor=1.0) == 3


def test_expert_load_counts_all_assignments():
    router = route_topk_explicit([[0, 1], [0, 2], [0, 1]], num_experts=4)
    torch.testing.assert_close(expert_load(router, 4), torch.tensor([3, 2, 1, 0]))


def test_build_capacity_mask_keeps_first_assignments_in_row_major_order():
    router = route_topk_explicit([[0, 1], [0, 1], [0, 1]], num_experts=2)

    mask = build_capacity_mask(router, num_experts=2, capacity=2)

    # Each expert keeps its first two occurrences (tokens 0 and 1); token 2 drops.
    torch.testing.assert_close(
        mask, torch.tensor([[True, True], [True, True], [False, False]])
    )


def test_build_capacity_mask_zero_capacity_drops_everything():
    router = route_topk_explicit([[0, 1], [2, 3]], num_experts=4)
    mask = build_capacity_mask(router, num_experts=4, capacity=0)
    assert not mask.any().item()


# --- reference forward vs oracle -------------------------------------------


def _topk_model(seed=7, num_experts=4):
    torch.manual_seed(seed)
    return TopKReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=num_experts)


def test_topk_grouped_matches_oracle_without_capacity():
    model = _topk_model()
    inputs = torch.randn(13, 8)
    router = route_topk_round_robin(num_tokens=13, num_experts=4, k=2)

    grouped, trace = model(inputs, router)
    oracle = model.token_by_token_oracle(inputs, router)

    torch.testing.assert_close(grouped, oracle, rtol=RTOL, atol=ATOL)
    assert trace.capacity is None
    assert trace.num_dropped == 0


def test_topk_grouped_matches_oracle_with_capacity_dropping():
    model = _topk_model()
    inputs = torch.randn(20, 8)
    # Skew toward expert 0 so capacity actually bites.
    assignments = [[0, 1]] * 12 + [[2, 3]] * 8
    router = route_topk_explicit(assignments, num_experts=4)

    grouped, trace = model(inputs, router, capacity_factor=1.0)
    oracle = model.token_by_token_oracle(inputs, router, capacity_factor=1.0)

    torch.testing.assert_close(grouped, oracle, rtol=RTOL, atol=ATOL)
    # capacity = ceil(1.0 * 20 * 2 / 4) = 10; experts 0 and 1 each have 12 -> 2 dropped each.
    assert trace.capacity == 10
    assert trace.num_dropped == 4


def test_dropped_assignment_contributes_nothing():
    model = _topk_model()
    inputs = torch.randn(3, 8)
    # All three tokens pick experts {0,1}; capacity 2 drops token 2's two slots.
    router = route_topk_explicit([[0, 1], [0, 1], [0, 1]], num_experts=4)

    grouped, trace = model(inputs, router, capacity_factor=1.0)  # capacity = ceil(1.0*3*2/4)=2

    assert trace.capacity == 2
    assert trace.num_dropped == 2
    torch.testing.assert_close(grouped[2], torch.zeros(8), rtol=RTOL, atol=ATOL)
    assert not torch.allclose(grouped[0], torch.zeros(8))


def test_topk_k1_matches_top1_reference():
    # A k=1 top-k layer with unit weights must equal the Stage 1 top-1 reference.
    torch.manual_seed(99)
    top1 = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)
    torch.manual_seed(99)
    topk = TopKReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)

    inputs = torch.randn(10, 8)
    assignment = [0, 3, 1, 2, 0, 3, 1, 2, 0, 1]
    top1_out, _ = top1(inputs, route_explicit(assignment, num_experts=4))
    topk_out, _ = topk(
        inputs,
        route_topk_explicit([[e] for e in assignment], num_experts=4),
    )

    torch.testing.assert_close(top1_out, topk_out, rtol=RTOL, atol=ATOL)


def test_topk_handles_empty_input():
    model = _topk_model()
    inputs = torch.empty(0, 8)
    router = route_topk_round_robin(num_tokens=0, num_experts=4, k=2)

    grouped, trace = model(inputs, router, capacity_factor=1.0)

    assert grouped.shape == (0, 8)
    assert trace.num_dropped == 0
