# Test Strategy

The project tests correctness before distributed scale or optimization. Tests should prove token identity, layout reversibility, count reconciliation, weight application, and reference equivalence before any performance claim is accepted.

## Existing Verified Tests

The current verified baseline is `python -m pytest -q` with `28 passed in 3.96s`.

Existing tests cover:

- Round-robin synthetic top-1 routing assignments and unit weights.
- Explicit top-1 routing, including column tensor assignments.
- Invalid explicit assignments: non-integer values, bools, invalid shape, invalid expert ids, invalid length, invalid token/expert counts.
- Output shape `[num_tokens, hidden_dim]`.
- Grouped/permuted output compared to an independent token-by-token oracle.
- One-token input with empty experts.
- All tokens routed to one expert.
- Skewed routing with empty experts.
- Non-unit finite weights, including assertions that weights are neither ignored nor applied twice.
- Non-contiguous input tensor views.
- Permutation contains every token exactly once.
- Inverse permutation restores original token order.
- Expert counts and offsets match assignments.
- No silent token drop or duplication.
- Fixed-seed determinism.
- Invalid expert ids, invalid input shape, invalid router shape, invalid weight shape, non-finite weights, and non-floating weight dtype.

## Stage 2 Deterministic Layout Tests

Stage 2 should add local-only dispatch/combine tests before any distributed process exists:

- Build `TokenAssignment` records from `RouterOutput` plus `ExpertPlacement`.
- Convert assignments into a `DispatchPlan` with deterministic peer order and expert segments.
- Pack tokens according to the plan and verify each assignment id appears once.
- Simulate variable-size sends locally, including empty peers and empty experts.
- Build a `CombinePlan` and verify every dispatched assignment returns once.
- Unpack, unpermute, and reduce outputs, then compare to the Stage 1 reference.
- Test balanced routing, all-to-one skew, hot expert skew, and empty expert/rank segments.

## Randomized Tests

Randomized tests should use fixed seeds and small shapes:

- Random token counts, including one-token and zero-assignment-to-expert cases.
- Random expert counts where placement is deterministic.
- Random top-1 assignments with valid ids.
- Random skew patterns that create empty experts and hot experts.
- Random finite non-unit weights.
- Random non-contiguous input views when shapes remain `[num_tokens, hidden_dim]`.
- Random plan counts whose sums must reconcile with assignment counts.

## Reference-vs-Distributed Equivalence Tests

Stage 3 distributed tests must compare against the Stage 1 reference:

- Same input activations, expert weights, routing metadata, and expert placement.
- Same dtype policy and tolerance rule.
- Same assignment ids before and after dispatch/combine.
- Per-rank outputs gathered or compared so the final token-major output matches reference.
- Failure reports include rank id, peer id, phase id, expected counts, observed counts, max absolute error, and max relative error.

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
- Stage 1 fp32 output comparisons use `rtol=1e-5` and `atol=1e-6`.
- Future GPU comparisons must state dtype, device, backend, max absolute error, and max relative error.
- Missing assignments, duplicated assignments, invalid ids, and count mismatches are correctness failures, not tolerance issues.
- Token dropping is not part of the first project version.

## Tests Before Optimization

Before any packing/permutation or communication optimization:

- Stage 1 reference tests pass.
- Stage 2 deterministic dispatch/combine tests pass.
- Stage 3 2-GPU reference-equivalence tests pass if the optimization touches distributed movement.
- Empty, balanced, skewed, one-token, all-to-one, and non-contiguous input cases pass.
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
