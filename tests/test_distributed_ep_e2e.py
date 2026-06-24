"""End-to-end multi-process correctness tests for the Stage 3 distributed EP path.

These spawn real Gloo processes on CPU and assert that the distributed
dispatch/combine output matches both the Stage 2 logical simulation and the
Stage 1 reference, element for element, on every rank.

Gloo exercises the ``_all_gather_variable_tensors`` fallback; the NCCL
``all_to_all_single`` branch remains covered only by the manual 2-GPU smoke.
"""

from __future__ import annotations

import pytest
import torch

from _dist_harness import ep_worker, gloo_available, run_in_processes

pytestmark = pytest.mark.skipif(not gloo_available(), reason="torch.distributed Gloo backend is unavailable")

HIDDEN_DIM = 8
FFN_DIM = 16
SEED = 2029
RTOL = 1e-5
ATOL = 1e-6


def _run(world_size, assignments, owner_by_expert, num_experts, *, weights=None, num_tokens=None):
    if num_tokens is None:
        num_tokens = len(assignments)
    generator = torch.Generator().manual_seed(1234)
    inputs = torch.randn(num_tokens, HIDDEN_DIM, generator=generator)
    return run_in_processes(
        world_size,
        ep_worker,
        args=(inputs, assignments, owner_by_expert, num_experts, HIDDEN_DIM, FFN_DIM, weights, SEED),
    )


def _assert_matches_references(results):
    for result in results:
        torch.testing.assert_close(result["dist"], result["reference"], rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(result["dist"], result["logical"], rtol=RTOL, atol=ATOL)
    # The all-reduce combine replicates the full output, so every rank agrees.
    for result in results[1:]:
        torch.testing.assert_close(result["dist"], results[0]["dist"], rtol=RTOL, atol=ATOL)


def test_two_rank_balanced_round_robin():
    results = _run(2, [0, 1, 2, 3, 0, 1, 2, 3], (0, 0, 1, 1), 4)
    _assert_matches_references(results)
    assert results[0]["owned_experts"] == (0, 1)
    assert results[1]["owned_experts"] == (2, 3)


def test_two_rank_all_to_one_expert_leaves_a_rank_empty():
    # Every token routes to expert 0 (owned by rank 0); rank 1 receives nothing.
    results = _run(2, [0] * 9, (0, 0, 1, 1), 4)
    _assert_matches_references(results)
    # Rank 1 owns experts 2/3 which receive no tokens: zero recv on every peer.
    assert int(results[1]["recv_counts"].sum().item()) == 0


def test_two_rank_empty_expert_within_active_rank():
    # Experts 0 and 2 are hit; expert 1 (rank 0) and expert 3 (rank 1) stay empty.
    results = _run(2, [0, 2, 0, 2, 0], (0, 0, 1, 1), 4)
    _assert_matches_references(results)


def test_two_rank_non_unit_routing_weights_applied_once():
    results = _run(
        2,
        [0, 2, 3, 1],
        (0, 0, 1, 1),
        4,
        weights=[0.25, 0.5, 1.5, 2.0],
    )
    _assert_matches_references(results)


def test_two_rank_skew_with_single_token():
    results = _run(2, [3], (0, 0, 1, 1), 4)
    _assert_matches_references(results)


def test_four_rank_balanced():
    results = _run(4, [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], (0, 1, 2, 3), 4)
    _assert_matches_references(results)
    for rank, result in enumerate(results):
        assert result["owned_experts"] == (rank,)


def test_four_rank_skew_hot_expert():
    # Heavy skew toward expert 1 (rank 1); ranks 2 and 3 stay cold.
    results = _run(4, [1, 1, 1, 1, 0, 1, 1], (0, 1, 2, 3), 4)
    _assert_matches_references(results)
    assert int(results[2]["recv_counts"].sum().item()) == 0
    assert int(results[3]["recv_counts"].sum().item()) == 0
