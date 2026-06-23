# Implementation Roadmap

This roadmap is strict: each stage depends on the previous stage, and later stages should not begin until the acceptance tests and required benchmark artifact for the current stage exist.

## Stage 0 - Architecture and invariants

- Goal: document the architecture, boundaries, invariants, staged plan, and test strategy before source code exists.
- Deliverables: `README.md`, `docs/architecture.md`, `docs/design-decisions.md`, `docs/implementation-roadmap.md`, and `docs/test-strategy.md`.
- Dependency on previous stages: none.
- Acceptance tests: documentation states current repository state accurately; all proposed modules are labeled as future proposals; no source code, package files, CUDA code, dependencies, CI, or benchmark scripts are created.
- Benchmark required before proceeding: a benchmark plan naming future metrics, with no measured performance claims.
- Non-goals: implementing runtime code, installing packages, running GPU commands, or designing a production framework.
- Kill criteria that prevent moving forward: unclear ownership of communication, missing token-loss invariants, or a roadmap that requires multi-GPU work before reference correctness.

## Stage 1 - Single-process reference MoE FFN

- Goal: implement the smallest forward-only CPU/reference MoE FFN that exercises routing, local expert execution, and weighted output assembly in one process.
- Deliverables: a standalone MoE FFN reference path, deterministic fixture inputs, explicit metadata objects, and unit tests for top-1 routing.
- Dependency on previous stages: Stage 0 must define invariants and metadata responsibilities.
- Acceptance tests: fixed-seed reference output is reproducible; every token has exactly one assignment for top-1; output shape and token order match input; no token is dropped.
- Benchmark required before proceeding: record CPU wall-clock time for the reference forward on fixed small fixtures, labeled as a reference baseline only.
- Non-goals: distributed execution, backward propagation, custom CUDA, capacity factor, token dropping, full Transformer integration, or top-2 execution.
- Kill criteria that prevent moving forward: nondeterministic reference output under a fixed seed, unclear weight application semantics, or inability to inspect assignments per token.

## Stage 2 - Deterministic dispatch/combine harness

- Goal: validate grouping, permutation, packing, combine, unpermute, and weighted reduction without real cross-rank communication.
- Deliverables: deterministic local harness that simulates EP ranks, dispatch plans, combine plans, empty expert cases, empty peer cases, and skewed routing cases.
- Dependency on previous stages: Stage 1 reference output must exist and be reproducible.
- Acceptance tests: packed order is reversible; simulated variable-size dispatch preserves every assignment; combine restores original token order; top-k metadata supports at least top-1 and is shaped for later top-2.
- Benchmark required before proceeding: record counts and CPU time for pack, simulated dispatch, simulated combine, unpack, and reduction on fixed skew fixtures.
- Non-goals: launching distributed processes, using GPUs, adding CUDA kernels, or claiming communication performance.
- Kill criteria that prevent moving forward: any assignment is lost, duplicated outside top-k semantics, returned to the wrong token, or reduced with a weight more than once.

## Stage 3 - 2-GPU EP baseline

- Goal: run the forward EP path on 2 GPUs using PyTorch distributed collectives while matching the single-process reference output.
- Deliverables: 2-rank execution path, distributed transport backend, collective phase ordering checks, per-rank profile events, and reference comparison tests.
- Dependency on previous stages: Stage 2 must prove the same plans work in the no-op/simulated transport path.
- Acceptance tests: both ranks agree on placement, peer order, counts, and collective sequence; distributed output matches reference within tolerance; empty sends and skewed sends are tested.
- Benchmark required before proceeding: record per-phase timings and byte counts for routing, packing, dispatch, expert compute, combine, unpack, and reduction on fixed 2-GPU fixtures.
- Non-goals: custom CUDA packing, communication-computation overlap, 4-GPU scaling, backward propagation, or production deployment.
- Kill criteria that prevent moving forward: collective ordering can diverge, rank counts do not reconcile, output differs from reference beyond tolerance, or a distributed hang cannot be diagnosed from profile events.

## Stage 4 - 4-GPU scaling and skew experiments

- Goal: validate that the 2-GPU design generalizes to 4 GPUs and measure routing skew effects.
- Deliverables: 4-rank test configuration, expert placement variants, skewed routing fixtures, and comparison against 2-GPU baseline metrics.
- Dependency on previous stages: Stage 3 must be correct and measured on 2 GPUs.
- Acceptance tests: all ranks agree on placement and collective ordering; distributed output matches reference within tolerance; empty-rank, empty-expert, and hot-expert cases are covered.
- Benchmark required before proceeding: record bytes moved, per-rank token counts, collective timings, and expert compute timings for balanced and skewed 4-GPU fixtures.
- Non-goals: multi-node execution, RDMA, InfiniBand, dynamic expert migration, or custom kernels.
- Kill criteria that prevent moving forward: 4-GPU correctness failures that do not reproduce in the simulated harness, unbounded rank disagreement in counts, or profiling gaps that hide skew behavior.

## Stage 5 - GPU-side packing/permutation optimization

- Goal: reduce a named, measured packing or permutation cost while preserving the same metadata contracts and reference-equivalent output.
- Deliverables: GPU-side packing/permutation prototype, fallback baseline path, correctness tests shared with earlier stages, and before/after phase measurements.
- Dependency on previous stages: Stage 4 must identify packing or permutation as a measured bottleneck worth addressing.
- Acceptance tests: output matches reference within tolerance; dispatch and combine metadata remain compatible; empty and skewed cases still pass; fallback path remains available.
- Benchmark required before proceeding: compare CPU/framework-side packing against GPU-side packing for the same fixtures, reporting phase time and bytes copied.
- Non-goals: changing routing semantics, changing placement semantics, adding overlap, or replacing the whole transport stack.
- Kill criteria that prevent moving forward: metadata contracts must be weakened, correctness depends on fixture-specific assumptions, or measured improvement does not appear in the targeted phase.

## Stage 6 - Communication-computation overlap

- Goal: overlap communication and local expert computation only after both are separately correct and measured.
- Deliverables: overlap schedule, event timeline, correctness tests under deterministic scheduling, and comparison with non-overlapped baseline.
- Dependency on previous stages: Stage 5 or the measured baseline must identify enough separate communication and compute time to justify overlap work.
- Acceptance tests: output matches reference within tolerance; collective ordering remains provable; profile events show phase boundaries and overlap windows; fallback non-overlap path remains correct.
- Benchmark required before proceeding: record end-to-end time and phase timeline for overlapped and non-overlapped runs on the same fixtures.
- Non-goals: speculative scheduling, dynamic expert migration, production serving, or multi-node communication.
- Kill criteria that prevent moving forward: overlap makes correctness nondeterministic, hides collective ordering, or cannot be measured separately from unrelated changes.

## Stage 7 - Final benchmark report

- Goal: produce a final report explaining correctness coverage, bottlenecks, and measured behavior across the implemented stages.
- Deliverables: benchmark report, reproducibility notes, hardware and software labels, fixture descriptions, correctness summary, and known limitations.
- Dependency on previous stages: prior stages must have produced accepted correctness results and benchmark artifacts.
- Acceptance tests: every reported number has a fixture, backend, hardware label, and measurement method; every optimization claim is paired with a baseline; limitations are explicit.
- Benchmark required before proceeding: this is the final benchmark artifact; it must include reference, 2-GPU, and any completed later-stage results without extrapolating unmeasured performance.
- Non-goals: claiming production competitiveness, adding new runtime features, or benchmarking systems outside the project scope.
- Kill criteria that prevent completion: missing correctness evidence, unlabeled hardware, unrepeatable fixture setup, or performance claims not tied to measured project artifacts.
