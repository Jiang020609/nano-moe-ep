# Test Strategy

The project tests correctness before distributed scale or optimization. Tests should prove token identity, layout reversibility, count reconciliation, weight application, and reference equivalence before any performance claim is accepted.

## Existing Verified Tests

The current verified baseline is `python -m pytest -q` with `74 passed`.

Existing tests cover:

- Round-robin and explicit synthetic top-1 routing.
- Invalid routing assignments, invalid expert ids, invalid assignment length, and invalid router/weight shapes.
- Stage 1 grouped/permuted output compared to an independent token-by-token oracle.
- One-token input, all tokens routed to one expert, skewed routing, empty experts, and non-contiguous inputs.
- Non-unit finite weights, including checks that weights are neither ignored nor applied twice.
- Stage 1 permutation/inverse permutation and expert counts/offsets.
- Fixed-seed determinism.
- Stage 2 table-driven expert placement, including empty ranks and uneven placement.
- Stage 2 deterministic ordering: destination rank, then expert id, then original token order.
- Stage 2 `DispatchPlan` payload order matching layout permutation.
- Stage 2 `CombinePlan` rejecting missing or duplicated token indices.
- Stage 2 logical EP output matching Stage 1 reference.
- Stage 2.75 `EPContext` validation, placement world-size agreement, and unchanged logical EP output when context is provided.
- Stage 3 source-token partitioning, distributed payload count/offset metadata, return-count reversal, and partial combine validation.

## Stage 2 Layout And Plan Tests

Stage 2 is implemented as a single-process logical EP simulation. The tests intentionally avoid real distributed execution and cover:

- `TokenAssignment` construction from `RouterOutput` plus `ExpertPlacement`.
- Rank-aware `TokenLayout` with `rank_counts`, `rank_offsets`, `rank_expert_counts`, and `rank_expert_offsets`.
- Empty expert and empty logical-rank metadata.
- `DispatchPlan` payload construction without expert computation.
- `execute_local_experts` running expert slices according to rank/expert offsets.
- `CombinePlan` alignment and exact once-only weight application.
- `EPContext` validation before dispatch/combine ownership begins.
- Final output equivalence to the Stage 1 reference.

## Randomized Tests

Future randomized tests should use fixed seeds and small shapes:

- Random token counts, including one-token and zero-assignment-to-expert cases.
- Random expert counts with deterministic placement.
- Random top-1 assignments with valid ids.
- Random skew patterns that create empty experts and hot experts.
- Random finite non-unit weights.
- Random non-contiguous input views where shape remains `[num_tokens, hidden_dim]`.
- Random plan counts whose sums must reconcile with assignment counts.

## Reference-vs-Distributed Equivalence Tests

Stage 3 distributed tests must compare against the Stage 1/2 local references:

- Same input activations, expert weights, routing metadata, and expert placement.
- Same dtype policy and tolerance rule.
- Same assignment ids before and after dispatch/combine.
- Per-rank outputs gathered or compared so the final token-major output matches reference.
- Failure reports include rank id, peer id, phase id, expected counts, observed counts, max absolute error, and max relative error.

These equivalence tests now run automatically as multi-process Gloo end-to-end
tests in `tests/test_distributed_ep_e2e.py` (driven by the `tests/_dist_harness.py`
`mp.spawn` harness). They spawn 2- and 4-rank processes on CPU and assert that
`run_distributed_ep_moe` matches both the Stage 2 logical simulation and the
Stage 1 reference, on every rank, across balanced, all-to-one-skew, empty-expert,
empty-rank, single-token, and non-unit-weight fixtures. Because Gloo lacks
`all_to_all_single`, these cover the `_all_gather_variable_tensors` fallback and
all surrounding orchestration; the NCCL `all_to_all_single` collective itself is
covered only by the manual smoke below.

Both combine strategies are checked in the same fixtures: the default sharded
combine (each rank's rows must equal the reference rows for its tokens, and the
shards must tile every token exactly once and reconstruct the reference) and the
legacy `replicate_output=True` combine (the full output must match the reference
on every rank). `apply_sharded_combine` also has direct CPU unit tests for sort
order, weight-once application, the empty shard, and duplicate-index rejection.

The manual Stage 3 smoke is launched with:

```bash
torchrun --standalone --nproc_per_node=2 scripts/run_stage3_2gpu_smoke.py
```

It uses NCCL only when two CUDA devices and NCCL are available; otherwise it falls back to CPU/Gloo. The smoke is correctness-only and must not be interpreted as a benchmark.

## Collective Ordering Tests

Distributed stages must prove ranks agree before entering collectives:

- All ranks use the same peer order.
- All ranks use the same collective phase order.
- Send/receive counts reconcile pairwise.
- Empty sends and receives are represented explicitly.
- Rank-local `ExpertPlacement` views match a shared placement definition.
- A failure in ordering checks should happen before a hang-prone collective call.

## Numerical Tolerance Rules

- Metadata equality is exact: token ids, assignment ids, expert ids, rank ids, offsets, counts, and top-k slots.
- Stage 1 and Stage 2 fp32 output comparisons use `rtol=1e-5` and `atol=1e-6`.
- Future GPU comparisons must state dtype, device, backend, max absolute error, and max relative error.
- Missing assignments, duplicated assignments, invalid ids, and count mismatches are correctness failures, not tolerance issues.
- Token dropping is not part of the first project version.

## Tests Before Optimization

Before any packing/permutation or communication optimization:

- Stage 1 reference tests pass.
- Stage 2 deterministic dispatch/combine tests pass.
- Stage 3 2-GPU reference-equivalence tests pass if the optimization touches distributed movement.
- Empty, balanced, skewed, one-token, all-to-one, uneven-placement, and non-contiguous input cases pass.
- The targeted phase has a baseline measurement and named fixture.
- The fallback non-optimized path remains available and tested.

## Tests Before 4-GPU Work

Before moving from 2 GPUs to 4 GPUs:

- 2-GPU balanced, empty-peer, empty-expert, and hot-expert fixtures pass.
- Per-rank count reconciliation passes for dispatch and combine.
- Collective ordering checks are logged and auditable.
- Expert placement is immutable within a forward pass.
- Distributed output matches the Stage 1 reference within tolerance.
- Profile output includes per-rank token counts, bytes moved, and phase timings.
