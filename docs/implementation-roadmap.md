# Implementation Roadmap

This roadmap is strict. Each stage must pass its validation gate before later-stage work begins.

## Stage 0 - Architecture and invariants

- Goal: document project scope, invariants, metadata, staged execution, and non-goals.
- Deliverables: README plus architecture, design decision, roadmap, and test strategy documents.
- Acceptance tests: docs distinguish implemented baseline from proposed future work.
- Non-goals: runtime implementation, GPU work, package expansion, benchmark scripts, or CI.
- Validation gate: reviewers can identify baseline, future boundaries, and correctness invariants from docs alone.
- Status: complete.
- Kill criteria: docs imply distributed/CUDA functionality exists before it is implemented, or metadata ownership is unclear.

## Stage 1 - Single-process reference MoE FFN

- Goal: implement and test a forward-only CPU reference MoE FFN with deterministic synthetic top-1 routing.
- Deliverables: `RouterOutput`, Stage 1 `TokenLayout`, `ExpertPlacement.single_rank`, `ReferenceTrace`, synthetic routers, grouped/permuted reference FFN, token-by-token oracle, and CPU tests.
- Acceptance tests: grouped output matches oracle; token order and counts are validated; non-unit weights are applied once; non-contiguous input works.
- Non-goals: distributed EP, CUDA/NCCL, top-2, capacity factor, token dropping, backward, serving, or benchmarking.
- Validation gate: Stage 1 tests pass on CPU with fixed seeds and `rtol=1e-5`, `atol=1e-6`.
- Status: complete and still covered by the current `64 passed` suite.
- Kill criteria: nondeterministic reference output, token loss/duplication, ambiguous inverse permutation semantics, or hidden weight application.

## Stage 2 - Deterministic dispatch/combine harness

- Goal: simulate dispatch/combine locally without distributed processes so logical EP metadata can be tested deterministically.
- Deliverables: `TokenAssignment`, rank-aware `TokenLayout`, table-driven `ExpertPlacement`, `DispatchPlan`, `CombinePlan`, `LogicalEPTrace`, `EPContext`, local payload simulation, per-rank expert execution, and combine restore.
- Acceptance tests: every assignment is packed once, every output combines once, empty experts/ranks are represented, payload order matches layout permutation, invalid placement is rejected, and logical EP output matches Stage 1 reference.
- Non-goals: real `torch.distributed`, GPUs, CUDA kernels, NCCL, benchmarks, overlap, top-2, or backward.
- Validation gate: deterministic fixtures cover balanced routing, all-to-one skew, empty experts, empty ranks, uneven placement, non-unit weights, non-contiguous input, and failure injection for duplicate/missing tokens.
- Status: implemented; Stage 2.75 execution-context metadata is in place; current verified suite is `python -m pytest -q` with `64 passed`.
- Kill criteria: assignment identity is lost, counts do not reconcile, combine order is ambiguous, payload order can diverge from layout, or output cannot be compared to Stage 1 reference.

## Stage 3 - 2-GPU EP baseline

- Goal: validate the EP forward path on 2 GPUs with a PyTorch distributed baseline.
- Deliverables: 2-rank launch path, transport backend, collective phase ordering checks, per-rank count reconciliation, reference-equivalence smoke script, and CPU unit tests for distributed planning metadata.
- Acceptance tests: both ranks agree on placement, peer order, counts, and phase ids; distributed output matches Stage 1 reference within tolerance.
- Non-goals: 4-GPU scaling, custom CUDA packing, overlap, backward, serving, or production deployment.
- Validation gate: 2-GPU balanced, empty-peer, and skewed fixtures pass repeatedly without hangs.
- Status: baseline implemented with `torch.distributed` and validated end-to-end by 2- and 4-process Gloo tests in CI (bit-for-bit vs. the single-process reference); the NCCL `all_to_all_single` path still requires a CUDA/NCCL environment via the manual smoke. The combine is now sharded (reverse all-to-all only) by default, removing the full-output `all_reduce`; `scripts/bench_combine.py` reports the per-rank communication reduction (about 3x at P=2 up to 15x at P=8 for balanced routing) and the legacy `all_reduce` combine is retained as `replicate_output=True` for comparison.
- Kill criteria: collectives can be entered in different order, counts are rank-inconsistent, or output differs from reference beyond tolerance.

## Stage 3.5 - Top-k routing, capacity, and token dropping (reference)

- Goal: add real MoE routing semantics (top-k, capacity factor, token dropping) in the single-process reference before pushing them into the distributed path.
- Deliverables: `TopKRouterOutput`, top-k synthetic routers, a shared capacity/drop policy (`compute_expert_capacity`, `build_capacity_mask`, `expert_load`), `TopKReferenceMoEFFN` with a token-by-token oracle, and a routing-skew/capacity benchmark (`scripts/bench_capacity.py`).
- Acceptance tests: grouped output matches the oracle with and without dropping; dropped assignments contribute nothing; `k=1` matches the top-1 reference; capacity formula, drop ordering, and expert-load counts are exact; empty input works.
- Non-goals: distributed top-k dispatch/combine, a learned softmax gate, and backward.
- Validation gate: the drop policy is the single source of truth for both the grouped path and the oracle, and the skew benchmark reports load imbalance, drop rate, and capacity utilization from the real policy.
- Status: implemented in the single-process reference; the distributed path is still top-1.
- Kill criteria: the grouped path and oracle can drop different assignments, or dropping changes a kept token's output.

## Stage 4 - 4-GPU scaling and skew experiments

- Goal: check that the 2-GPU design generalizes to 4 GPUs and exposes routing skew behavior.
- Deliverables: 4-rank fixtures, placement variants, skew summaries, and comparison to the 2-GPU baseline.
- Acceptance tests: 4-rank output matches reference; empty-rank and hot-expert cases pass; per-rank token counts and bytes are recorded.
- Non-goals: multi-node, RDMA, InfiniBand, dynamic expert migration, or custom kernels.
- Validation gate: every 4-GPU correctness failure is reproducible in a deterministic local or 2-GPU fixture.
- Status: not started.
- Kill criteria: profiling cannot explain skew, rank count disagreement persists, or correctness depends on fixture-specific assumptions.

## Stage 5 - GPU-side packing/permutation optimization

- Goal: reduce a measured packing/permutation bottleneck while preserving metadata contracts.
- Deliverables: GPU-side packing/permutation prototype, CPU/framework fallback, shared correctness tests, and before/after phase measurements.
- Acceptance tests: output matches reference; empty and skewed cases pass; fallback remains correct; measured phase is named.
- Non-goals: changing routing semantics, changing placement semantics, replacing all transport, or adding overlap.
- Validation gate: the targeted phase shows a measured bottleneck before optimization work starts.
- Status: not started.
- Kill criteria: correctness weakens, metadata contracts change incompatibly, or the targeted phase does not improve.

## Stage 6 - Communication-computation overlap

- Goal: overlap communication and expert compute only after both are separately correct and measured.
- Deliverables: overlap schedule, event timeline, fallback non-overlap path, and correctness tests under deterministic scheduling.
- Acceptance tests: output matches reference; phase order is auditable; profile events show overlap windows.
- Non-goals: speculative scheduling, production serving, multi-node communication, or dynamic expert migration.
- Validation gate: separate communication and compute measurements justify overlap work.
- Status: not started.
- Kill criteria: overlap makes correctness nondeterministic, hides collective ordering, or cannot be measured independently.

## Stage 7 - Final benchmark report

- Goal: produce a reproducible report explaining correctness coverage, bottlenecks, and measured behavior.
- Deliverables: benchmark report, hardware/software labels, fixture descriptions, correctness matrix, limitations, and future work.
- Acceptance tests: every reported number has a fixture, backend, hardware label, and measurement method; every improvement is compared to a baseline.
- Non-goals: claiming production competitiveness, adding features during report writing, or benchmarking unrelated systems.
- Validation gate: all included measurements are reproducible from documented commands and fixtures.
- Status: not started.
- Kill criteria: missing correctness evidence, unlabeled hardware, unreproducible fixtures, or unmeasured performance claims.
