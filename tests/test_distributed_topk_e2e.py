"""End-to-end multi-process tests for the distributed top-k EP path.

Spawns real Gloo processes on CPU and asserts that the distributed top-k
dispatch/combine output (with and without capacity dropping) matches the
single-process ``TopKReferenceMoEFFN`` element for element, on every rank.
"""

from __future__ import annotations

import pytest
import torch

from _dist_harness import gloo_available, run_in_processes, topk_ep_worker

pytestmark = pytest.mark.skipif(not gloo_available(), reason="torch.distributed Gloo backend is unavailable")

HIDDEN_DIM = 8
FFN_DIM = 16
SEED = 2029
RTOL = 1e-5
ATOL = 1e-6


def _run(world_size, assignments, owner_by_expert, num_experts, *, weights=None, capacity_factor=None):
    num_tokens = len(assignments)
    if weights is None:
        weights = [[1.0 / len(row)] * len(row) for row in assignments]
    generator = torch.Generator().manual_seed(4242)
    inputs = torch.randn(num_tokens, HIDDEN_DIM, generator=generator)
    return run_in_processes(
        world_size,
        topk_ep_worker,
        args=(
            inputs,
            assignments,
            weights,
            owner_by_expert,
            num_experts,
            HIDDEN_DIM,
            FFN_DIM,
            capacity_factor,
            SEED,
        ),
    )


def _assert_matches_reference(results):
    reference = results[0]["reference"]
    num_tokens = reference.shape[0]

    for result in results:
        torch.testing.assert_close(result["replicated"], result["reference"], rtol=RTOL, atol=ATOL)
        idx = result["sharded_token_indices"]
        torch.testing.assert_close(
            result["sharded"], reference.index_select(0, idx), rtol=RTOL, atol=ATOL
        )
        assert torch.equal(idx, torch.sort(idx).values)

    for result in results[1:]:
        torch.testing.assert_close(result["replicated"], results[0]["replicated"], rtol=RTOL, atol=ATOL)

    # Shards tile every token exactly once and reconstruct the reference.
    all_idx = torch.cat([result["sharded_token_indices"] for result in results])
    assert torch.equal(torch.sort(all_idx).values, torch.arange(num_tokens))
    reconstructed = torch.zeros_like(reference)
    for result in results:
        reconstructed.index_copy_(0, result["sharded_token_indices"], result["sharded"])
    torch.testing.assert_close(reconstructed, reference, rtol=RTOL, atol=ATOL)


def test_two_rank_topk_no_capacity():
    assignments = [[0, 2], [1, 3], [0, 3], [2, 1], [0, 2]]
    results = _run(2, assignments, (0, 0, 1, 1), 4)
    _assert_matches_reference(results)
    for result in results:
        assert result["capacity"] is None
        assert result["num_local_dropped"] == 0


def test_two_rank_topk_non_unit_weights():
    assignments = [[0, 2], [1, 3], [3, 0]]
    weights = [[0.25, 0.75], [1.5, 0.5], [2.0, 1.0]]
    results = _run(2, assignments, (0, 0, 1, 1), 4, weights=weights)
    _assert_matches_reference(results)


def test_two_rank_topk_with_capacity_dropping():
    # Skew toward experts 0 and 1 (rank 0) so capacity bites.
    assignments = [[0, 1]] * 6 + [[2, 3]] * 2
    results = _run(2, assignments, (0, 0, 1, 1), 4, capacity_factor=1.0)
    _assert_matches_reference(results)
    # capacity = ceil(1.0 * 8 * 2 / 4) = 4; experts 0 and 1 each see 6 -> drops happen.
    total_dropped = sum(result["num_local_dropped"] for result in results)
    assert results[0]["capacity"] == 4
    assert total_dropped == results[0]["ref_num_dropped"]
    assert total_dropped > 0


def test_four_rank_topk_balanced():
    assignments = [[i % 4, (i + 1) % 4] for i in range(12)]
    results = _run(4, assignments, (0, 1, 2, 3), 4)
    _assert_matches_reference(results)


def test_four_rank_topk_with_capacity_dropping():
    assignments = [[0, 1]] * 8 + [[2, 3], [1, 2], [0, 3], [3, 0]]
    results = _run(4, assignments, (0, 1, 2, 3), 4, capacity_factor=1.0)
    _assert_matches_reference(results)
    total_dropped = sum(result["num_local_dropped"] for result in results)
    assert total_dropped == results[0]["ref_num_dropped"]
