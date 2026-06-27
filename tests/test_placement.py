import pytest
import torch

from nano_moe_ep.placement import (
    balanced_placement,
    contiguous_placement,
    load_imbalance,
    max_rank_load,
    rank_load,
)
from nano_moe_ep.types import ExpertPlacement


def test_contiguous_placement_blocks_experts_evenly():
    placement = contiguous_placement(num_experts=4, num_ep_ranks=2)
    assert placement.owner_rank_by_expert == (0, 0, 1, 1)


def test_contiguous_placement_uses_ceil_for_leading_ranks():
    placement = contiguous_placement(num_experts=5, num_ep_ranks=2)
    # rank 0 gets ceil (3 experts), rank 1 gets 2.
    assert placement.owner_rank_by_expert == (0, 0, 0, 1, 1)


def test_rank_load_and_max_and_imbalance():
    placement = ExpertPlacement(owner_rank_by_expert=(0, 0, 1, 1), num_ep_ranks=2)
    load = torch.tensor([5, 4, 1, 1])

    torch.testing.assert_close(rank_load(placement, load), torch.tensor([9, 2]))
    assert max_rank_load(placement, load) == 9
    assert load_imbalance(placement, load) == pytest.approx(9 / 5.5)


def test_balanced_placement_lowers_max_rank_load_under_skew():
    load = torch.tensor([5, 4, 1, 1])
    naive = contiguous_placement(4, 2)
    balanced = balanced_placement(load, 2)

    # Contiguous puts experts 0 and 1 together (load 9); balanced splits them.
    assert max_rank_load(naive, load) == 9
    assert max_rank_load(balanced, load) == 6
    assert max_rank_load(balanced, load) < max_rank_load(naive, load)


def test_balanced_placement_keeps_even_expert_counts():
    load = torch.tensor([10, 9, 8, 1, 1, 1])
    balanced = balanced_placement(load, 3)

    counts = torch.bincount(
        torch.tensor(balanced.owner_rank_by_expert), minlength=3
    )
    # Six experts over three ranks: exactly two each, like the contiguous baseline.
    torch.testing.assert_close(counts, torch.tensor([2, 2, 2]))


def test_balanced_placement_assigns_each_expert_once_and_is_deterministic():
    load = torch.tensor([3, 3, 3, 3])
    first = balanced_placement(load, 2)
    second = balanced_placement(load, 2)

    assert first.owner_rank_by_expert == second.owner_rank_by_expert
    assert sorted(set(first.owner_rank_by_expert)) == [0, 1]
    assert len(first.owner_rank_by_expert) == 4


def test_balanced_placement_single_rank_owns_all_experts():
    placement = balanced_placement(torch.tensor([1, 2, 3]), 1)
    assert placement.owner_rank_by_expert == (0, 0, 0)


def test_balanced_placement_rejects_more_ranks_than_experts():
    with pytest.raises(ValueError, match="must not exceed num_experts"):
        balanced_placement(torch.tensor([1, 2]), 3)
