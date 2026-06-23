# Test Strategy

The test strategy is correctness-first. Optimizations and distributed scale-ups should wait until tests can prove that routed assignments, token order, weights, and rank agreement are intact.

## Unit Tests

- Router tests: expert id range, top-k slot count, finite weights, deterministic output under fixed seeds, and explicit weight normalization rule.
- Placement tests: every expert has exactly one owner, all ranks derive the same placement, local expert slots are valid, and placement is immutable during a forward pass.
- Assignment tests: each token produces the expected number of assignments, top-k slot ids are stable, and assignment ids survive dispatch and combine.
- Dispatch plan tests: send counts sum to assignment count, peer order is stable, offsets are monotonic, empty peers are represented, and expert segment boundaries match assignments.
- Layout tests: group, permute, pack, unpack, and unpermute round-trip without changing token identity.
- Expert tests: local expert input and output shapes match; empty expert batches are valid; experts receive only assigned tokens.
- Combine and reduction tests: every dispatched assignment has one returned result, original token order is restored, and routing weights are applied exactly once.
- Profiling tests: profile events use explicit units, phase names, rank ids, and backend labels without changing execution behavior.

## Property-Based Or Randomized Tests

These tests can start with a deterministic random generator and do not require a new dependency at first.

- Random token counts, including zero-token edge cases where supported by the implementation contract.
- Random expert counts and world sizes that divide or do not evenly divide expert ownership, depending on the placement policy being tested.
- Random top-k assignments with valid weights, including repeated hot experts and empty experts.
- Skewed routing distributions where most assignments target one expert or one rank.
- Random peer send counts with empty sends and uneven receive counts.
- Round-trip checks that pack/unpack and dispatch/combine preserve assignment identity for every generated case.
- Reference comparison checks across multiple seeds, dtypes, and shape combinations once dtype support exists.

## Distributed Correctness Tests

- 2-rank tests must run before any 4-rank test is considered meaningful.
- Each rank must verify the same world size, rank order, expert placement, peer order, and collective phase sequence.
- Per-rank planned send counts and observed receive counts must reconcile.
- Empty send, empty receive, empty expert, and hot expert cases must be included.
- Distributed output must be compared against the single-process reference output for the same input, routing decisions, expert weights, and placement.
- Hangs or mismatched counts should produce enough logged metadata to identify the rank, phase id, peer, and expected count.

## Determinism Checks

- Fixed seeds must reproduce router decisions, expert initialization, input fixtures, and reference outputs.
- The same metadata plan should produce the same packed order across repeated runs.
- Distributed tests should record collective phase ids so rank ordering can be audited.
- Determinism tests should separate expected floating-point tolerance from metadata determinism. Metadata identity should match exactly.

## Failure Injection Ideas

- Duplicate an assignment id and verify the checker rejects it.
- Drop an assignment from the combine path and verify the missing token visit is reported.
- Send a payload to the wrong expert id and verify expert ownership checks fail.
- Corrupt a routing weight and verify weighted reduction comparison fails.
- Change peer order on one simulated rank and verify collective ordering checks fail before distributed execution.
- Truncate a receive buffer and verify count reconciliation fails.
- Mutate expert placement during a forward pass and verify immutability checks fail.

## Numerical Tolerance Rules

- Metadata equality is exact: token ids, assignment ids, expert ids, rank ids, offsets, and top-k slot ids should not use numerical tolerance.
- CPU fp32 reference comparisons should start with strict tolerances suitable for deterministic fixtures, then document any relaxation with a specific reason.
- GPU comparisons should use explicit absolute and relative tolerances per dtype and operation path.
- Tolerance checks must report maximum absolute error, maximum relative error, shape, dtype, seed, and backend.
- Token dropping is not part of the first version, so missing routed assignments are correctness failures, not tolerance issues.

## Required Before Any Optimization

- Reference output is deterministic under fixed seeds.
- Assignment, dispatch, combine, unpermute, and reduction invariants pass on balanced, empty, and skewed fixtures.
- Distributed output matches reference output within tolerance for the 2-GPU baseline.
- Profile events exist for the phase targeted by the optimization.
- The baseline path remains available so before/after measurements use the same fixture.

## Required Before Moving From 2 GPUs To 4 GPUs

- 2-GPU distributed correctness passes for balanced routing, empty peer sends, empty expert batches, and hot expert skew.
- Per-rank collective phase ordering is audited and logged.
- Expert placement is identical across ranks and immutable during the forward pass.
- Dispatch and combine count reconciliation passes on every rank.
- Reference comparison passes for the same fixtures used in distributed runs.
- Benchmark output includes per-rank token counts, bytes moved, and phase timings so 4-GPU skew can be interpreted rather than guessed.
