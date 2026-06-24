import pytest
import torch

from nano_moe_ep.distributed_ep import (
    CountExchange,
    DistributedEPConfig,
    apply_partial_combine,
    build_distributed_payload_plan,
    reverse_count_exchange,
    source_token_indices,
)
from nano_moe_ep.routing import route_explicit
from nano_moe_ep.types import ExecutionMode, ExpertPlacement, RouterOutput


def test_source_token_indices_partition_tokens_without_overlap():
    rank0 = source_token_indices(7, rank=0, world_size=2)
    rank1 = source_token_indices(7, rank=1, world_size=2)

    torch.testing.assert_close(rank0, torch.tensor([0, 2, 4, 6]))
    torch.testing.assert_close(rank1, torch.tensor([1, 3, 5]))
    torch.testing.assert_close(torch.sort(torch.cat([rank0, rank1])).values, torch.arange(7))


def test_distributed_payload_plan_uses_destination_rank_then_expert_order():
    router_output = route_explicit([0, 2, 3, 1, 2], num_experts=4)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0, 1], 1: [2, 3]},
        num_experts=4,
        num_ep_ranks=2,
    )

    rank0_plan = build_distributed_payload_plan(
        router_output,
        placement,
        source_rank=0,
        world_size=2,
    )
    rank1_plan = build_distributed_payload_plan(
        router_output,
        placement,
        source_rank=1,
        world_size=2,
    )

    torch.testing.assert_close(rank0_plan.send_counts_by_rank, torch.tensor([1, 2]))
    torch.testing.assert_close(rank0_plan.send_offsets, torch.tensor([0, 1, 3]))
    torch.testing.assert_close(rank0_plan.token_indices, torch.tensor([0, 4, 2]))
    torch.testing.assert_close(rank0_plan.expert_ids, torch.tensor([0, 2, 3]))
    torch.testing.assert_close(rank1_plan.send_counts_by_rank, torch.tensor([1, 1]))
    torch.testing.assert_close(rank1_plan.token_indices, torch.tensor([3, 1]))
    torch.testing.assert_close(rank0_plan.token_layout.rank_counts, torch.tensor([2, 3]))


def test_reverse_count_exchange_swaps_dispatch_and_return_directions():
    dispatch_counts = CountExchange(
        send_counts_by_rank=torch.tensor([1, 2]),
        recv_counts_by_rank=torch.tensor([1, 1]),
        count_matrix=torch.tensor([[1, 2], [1, 1]]),
    )

    reversed_counts = reverse_count_exchange(dispatch_counts, rank=0)

    torch.testing.assert_close(reversed_counts.count_matrix, torch.tensor([[1, 1], [2, 1]]))
    torch.testing.assert_close(reversed_counts.send_counts_by_rank, torch.tensor([1, 1]))
    torch.testing.assert_close(reversed_counts.recv_counts_by_rank, torch.tensor([1, 2]))


def test_partial_combine_restores_source_tokens_and_applies_weights_once():
    expert_outputs = torch.tensor([[2.0, 4.0], [10.0, 20.0]])
    token_indices = torch.tensor([3, 1])
    weights = torch.tensor([[0.5], [2.0]])

    output = apply_partial_combine(expert_outputs, token_indices, weights, num_tokens=5)

    expected = torch.tensor(
        [
            [0.0, 0.0],
            [20.0, 40.0],
            [0.0, 0.0],
            [1.0, 2.0],
            [0.0, 0.0],
        ]
    )
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output[1], expert_outputs[1])
    assert not torch.allclose(output[3], expert_outputs[0] * weights[0] * weights[0])


def test_partial_combine_rejects_duplicate_token_indices():
    with pytest.raises(ValueError, match="duplicates"):
        apply_partial_combine(
            torch.ones(2, 3),
            torch.tensor([1, 1]),
            torch.ones(2, 1),
            num_tokens=4,
        )


def test_distributed_config_exports_distributed_ep_context():
    config = DistributedEPConfig(backend="gloo", world_size=2, rank=1, device="cpu")

    context = config.to_ep_context()

    assert context.num_ep_ranks == 2
    assert context.local_rank == 1
    assert context.execution_mode is ExecutionMode.DISTRIBUTED
    assert context.device == "cpu"


def test_distributed_config_rejects_invalid_rank_and_backend_device():
    with pytest.raises(ValueError, match="rank"):
        DistributedEPConfig(backend="gloo", world_size=2, rank=2, device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        DistributedEPConfig(backend="nccl", world_size=2, rank=0, device="cpu")


def test_distributed_payload_plan_preserves_non_unit_weights():
    router_output = RouterOutput(
        expert_indices=torch.tensor([[0], [2], [3], [1]]),
        weights=torch.tensor([[0.25], [0.5], [1.5], [2.0]]),
    )
    placement = ExpertPlacement.from_rank_experts(
        {0: [0, 1], 1: [2, 3]},
        num_experts=4,
        num_ep_ranks=2,
    )

    plan = build_distributed_payload_plan(router_output, placement, source_rank=0, world_size=2)

    torch.testing.assert_close(plan.token_indices, torch.tensor([0, 2]))
    torch.testing.assert_close(plan.routing_weights, torch.tensor([[0.25], [1.5]]))
