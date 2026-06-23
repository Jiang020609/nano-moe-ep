import pytest
import torch

from nano_moe_ep.routing import route_explicit, route_round_robin


def test_round_robin_routing_creates_expected_assignment_and_unit_weights():
    router_output = route_round_robin(num_tokens=10, num_experts=3)

    assert router_output.expert_indices.shape == (10, 1)
    assert router_output.weights.shape == (10, 1)
    torch.testing.assert_close(
        router_output.expert_indices,
        torch.tensor([[0], [1], [2], [0], [1], [2], [0], [1], [2], [0]]),
    )
    torch.testing.assert_close(router_output.weights, torch.ones(10, 1))


def test_explicit_routing_preserves_supplied_assignment():
    router_output = route_explicit([2, 0, 1, 2, 2, 0, 1], num_experts=3, num_tokens=7)

    assert router_output.expert_indices.shape == (7, 1)
    assert router_output.weights.shape == (7, 1)
    torch.testing.assert_close(router_output.expert_indices[:, 0], torch.tensor([2, 0, 1, 2, 2, 0, 1]))
    torch.testing.assert_close(router_output.weights, torch.ones(7, 1))


def test_explicit_routing_accepts_column_tensor_assignment():
    assignments = torch.tensor([[0], [3], [1], [2], [3]])
    router_output = route_explicit(assignments, num_experts=4)

    torch.testing.assert_close(router_output.expert_indices, assignments)
    torch.testing.assert_close(router_output.weights, torch.ones(5, 1))


@pytest.mark.parametrize(
    "assignments, message",
    [
        ([0, 1.5, 2], "integer-like"),
        ([0, True, 2], "integer-like"),
        (torch.tensor([0.0, 1.0]), "integer-like"),
        ([[0, 1], [2, 0]], "shape"),
    ],
)
def test_explicit_routing_rejects_non_integer_or_invalid_assignment_shapes(assignments, message):
    with pytest.raises(ValueError, match=message):
        route_explicit(assignments, num_experts=3)


def test_explicit_routing_rejects_length_mismatch():
    with pytest.raises(ValueError, match="assignment length must match num_tokens"):
        route_explicit([0, 1, 2], num_experts=3, num_tokens=4)


@pytest.mark.parametrize("assignments", [[0, 3], [-1, 1]])
def test_explicit_routing_rejects_invalid_expert_ids(assignments):
    with pytest.raises(ValueError, match="expert indices must be in \\[0, num_experts\\)"):
        route_explicit(assignments, num_experts=3)


def test_round_robin_rejects_invalid_sizes():
    with pytest.raises(ValueError, match="num_tokens must be non-negative"):
        route_round_robin(num_tokens=-1, num_experts=3)
    with pytest.raises(ValueError, match="num_experts must be positive"):
        route_round_robin(num_tokens=3, num_experts=0)
