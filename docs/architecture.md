# nano-moe-ep Architecture

## Current Repository State

This repository was inspected on 2026-06-22. Before this documentation pass, the root contained `LICENSE` only. There was no `README.md`, `docs/` directory, source package, test suite, CI configuration, benchmark script, CUDA code, or Python package metadata. Everything below is a design proposal for future work, not a description of existing implementation.

## Project Thesis

`nano-moe-ep` is a correctness-first, from-scratch multi-GPU Expert Parallel runtime learning project for Mixture-of-Experts models. The project should make one standalone MoE FFN layer explicit, testable, and measurable: top-k routing, token grouping and permutation, variable-size cross-rank dispatch, local expert computation, cross-rank combine, and unpermute plus weighted reduction. It is not intended to beat production frameworks; its purpose is to implement the essential Expert Parallel data path narrowly enough that correctness, communication cost, routing skew, expert compute, and later overlap can be explained and measured.

## Assumptions

- Development may begin on Windows and CPU.
- The first complete correctness path is single-process and communication-free.
- The first distributed implementation targets one Linux host with 2 NVIDIA GPUs.
- A 4-GPU run is a later validation target, not the first milestone.
- Multi-node execution, RDMA, InfiniBand, FP8, production deployment, and serving are outside the first project version.
- The first model target is a standalone MoE FFN block, not a full Transformer.

## In Scope

- One MoE FFN forward path with explicit routing, token layout, dispatch, expert compute, combine, unpermute, and weighted reduction.
- A CPU/reference execution mode that is numerically testable in one process.
- Metadata structures that make token movement and ownership auditable.
- Variable token counts per expert and per EP rank.
- A distributed transport abstraction that can start with PyTorch distributed collectives.
- Profiling events for routing skew, bytes moved, collective timing, local expert work, and packing/unpacking cost.
- A strict path from single-process correctness to 2-GPU correctness before any CUDA-side specialization.

## Out of Scope

- Full LLM or full Transformer architecture.
- Tensor parallelism, pipeline parallelism, data parallelism, speculative decoding, MoE serving, dynamic expert migration, autoscaling, or production scheduling.
- Backward propagation in the first end-to-end milestone.
- Custom CUDA kernels before the reference and 2-GPU PyTorch distributed baselines are correct.
- Multi-node networking, RDMA, InfiniBand, FP8, or production inference deployment.
- Copying implementation details from DeepEP, Megatron-Core, vLLM, SGLang, or similar systems.

## Layered Architecture

```text
+--------------------------------------------------------------+
| Benchmark and profile harness                                |
+--------------------------------------------------------------+
| Public MoE FFN layer contract                                |
+--------------------------------------------------------------+
| Router: logits, top-k expert ids, top-k weights              |
+--------------------------------------------------------------+
| Planning: assignment, placement, dispatch plan, combine plan |
+--------------------------------------------------------------+
| Layout: group, permute, pack, unpack, unpermute              |
+--------------------------------------------------------------+
| Transport: no-op reference, PyTorch distributed, future CUDA |
+--------------------------------------------------------------+
| Local expert MLP execution                                   |
+--------------------------------------------------------------+
| Correctness oracle and invariant checks                      |
+--------------------------------------------------------------+
```

The router, planner, layout, local experts, and reducer must be usable without distributed communication. The transport layer owns cross-rank movement. The public MoE FFN layer composes these pieces but should not hide metadata needed for tests and profiling.

## End-to-End Data Flow For One MoE Layer

1. Input activations enter the MoE FFN layer as a token-major activation matrix with a stable original token index for every row.
2. The router reads activations and produces router logits over the configured expert set.
3. Top-k selection converts logits into expert assignments and routing weights for each token. The design must represent `k` slots per token even if the first implementation uses `k = 1`.
4. Expert placement maps each expert id to exactly one owning EP rank for the duration of the forward pass.
5. Token assignments are grouped by destination EP rank and then by destination expert. Empty groups are valid and must be represented.
6. The layout layer builds a permutation from original token order to packed dispatch order. It records enough metadata to reverse the permutation after combine.
7. Dispatch sends variable-size packed token payloads to destination ranks. In reference mode this is a local reorder; in distributed mode this is the only cross-rank communication step for expert inputs.
8. Each rank runs the local expert MLP for the token slices owned by its local experts. Expert compute receives only local expert ids and packed local token activations.
9. Combine returns each expert result to the rank that owns the original token position. In reference mode this is a local reorder; in distributed mode this is the return communication step.
10. The combine planner restores results to original token order and aligns every result with its original token id and top-k slot.
11. Weighted reduction applies routing weights exactly once per routed result and sums across top-k slots.
12. Output activations leave the layer in the same token order and hidden size as the input activations.

## Proposed Future Package Boundaries

These package boundaries are proposals only. They should not be created until the corresponding implementation stage needs them.

### `nano_moe_ep.moe_ffn`

- Responsibility: expose the standalone MoE FFN block contract and compose router, planner, layout, transport, expert execution, combine, and reducer.
- Inputs: input activations, router configuration, expert modules, EP context, execution mode.
- Outputs: output activations, optional debug metadata, optional profile events.
- Invariants: output token order matches input token order; top-k weights are applied once; the layer can run with reference transport.
- Must never own: collective communication details, expert placement mutation policy, benchmark reporting, or CUDA-specialized packing logic.

### `nano_moe_ep.router`

- Responsibility: produce router logits, top-k expert ids, and top-k weights.
- Inputs: token activations, router parameters, top-k configuration.
- Outputs: `RouterOutput` and token-level assignment candidates.
- Invariants: every token has exactly `k` routing slots unless the run is explicitly configured to test failure handling; weights are finite; expert ids are in range.
- Must never own: token packing order, rank communication, expert placement, or expert MLP execution.

### `nano_moe_ep.metadata`

- Responsibility: define conceptual records used across routing, layout, dispatch, combine, context, and profiling.
- Inputs: values emitted by router, planner, layout, transport, and instrumentation.
- Outputs: validated metadata objects passed across modules.
- Invariants: metadata is explicit enough to audit every token movement and reduction contribution.
- Must never own: tensor computation, distributed collectives, benchmark execution, or policy decisions.

### `nano_moe_ep.placement`

- Responsibility: map global expert ids to EP ranks and local expert slots.
- Inputs: number of experts, EP world size, rank id, placement policy.
- Outputs: `ExpertPlacement` and local expert ownership views.
- Invariants: every expert has exactly one owner during a forward pass; placement is immutable while that pass executes; all ranks agree on placement.
- Must never own: routing logits, token permutation, transport calls, or expert math.

### `nano_moe_ep.planning`

- Responsibility: convert router assignments and expert placement into dispatch and combine plans.
- Inputs: `RouterOutput`, `ExpertPlacement`, original token indices, EP context.
- Outputs: `TokenAssignment`, `DispatchPlan`, and `CombinePlan`.
- Invariants: no token assignment is dropped; duplicated assignments occur only because top-k requires multiple expert visits; source and destination counts reconcile.
- Must never own: tensor communication, expert MLP weights, profiler storage, or routing policy.

### `nano_moe_ep.layout`

- Responsibility: build and apply token grouping, permutation, packing, unpacking, and unpermutation.
- Inputs: input activations, `DispatchPlan`, returned expert activations, `CombinePlan`.
- Outputs: packed send buffers, unpacked receive buffers, restored token-major outputs before or after weighted reduction depending on the call boundary.
- Invariants: layout transformations are reversible using recorded metadata; empty expert and rank segments are valid; segment boundaries match the plan.
- Must never own: route selection, expert placement, collective scheduling, or numerical tolerance policy.

### `nano_moe_ep.transport`

- Responsibility: move packed token payloads and returned expert outputs according to plans.
- Inputs: packed payloads, variable counts, peer ordering, `EPContext`, transport backend selection.
- Outputs: received payloads and transport profile events.
- Invariants: each rank executes collectives in the same order; received counts match planned counts; no payload is interpreted by transport.
- Must never own: routing, token grouping policy, expert compute, weight application, or correctness oracle decisions.

### `nano_moe_ep.experts`

- Responsibility: run local expert MLP computation on packed token slices.
- Inputs: local expert activations grouped by expert, local expert parameters, local expert ids.
- Outputs: local expert outputs in the same packed local order.
- Invariants: each local expert receives only tokens assigned to that expert; empty expert batches are valid; output shape matches input token count and hidden size.
- Must never own: dispatch communication, original token ordering, global expert placement, or top-k reduction.

### `nano_moe_ep.reference`

- Responsibility: provide the single-process correctness path and comparison oracle.
- Inputs: deterministic inputs, router outputs or seeded router configuration, expert parameters, proposed plans.
- Outputs: reference outputs, metadata validation results, numerical comparison reports.
- Invariants: no distributed communication is required; execution is deterministic under a fixed seed; every distributed result can be compared to this path within tolerance.
- Must never own: production transport, CUDA-specific kernels, benchmark interpretation, or placement mutation.

### `nano_moe_ep.profiling`

- Responsibility: collect structured events for timing, counts, bytes, skew, and phase boundaries.
- Inputs: phase names, ranks, sizes, timestamps, counters, backend labels.
- Outputs: `ProfileEvent` records and later benchmark summaries.
- Invariants: profile events are descriptive and do not change execution behavior; units are explicit; measurements are labeled with hardware and backend assumptions.
- Must never own: routing choices, communication implementation, expert math, or correctness pass/fail decisions.

## Metadata Design

The following structures are conceptual contracts, not implementation code.

### RouterOutput

- Purpose: represent router decisions before any token movement.
- Fields: token count, expert count, top-k value, logits summary or optional logits tensor reference, selected expert ids per token and slot, routing weights per token and slot, optional router diagnostics.
- Invariants: selected expert ids are valid; weights are finite; shape records match token count and top-k; weight normalization rule is explicit.
- Never includes: dispatch rank ordering, packed buffer offsets, transport handles, or expert placement mutation.

### TokenAssignment

- Purpose: represent one routed visit from one token to one expert.
- Fields: original token index, top-k slot index, expert id, routing weight, destination EP rank, optional local expert index.
- Invariants: one token with `k` routing slots produces exactly `k` assignments; assignment identity is stable through dispatch and combine.
- Never includes: raw communication buffers, expert parameter tensors, or benchmark summaries.

### DispatchPlan

- Purpose: describe how assignments become packed sends.
- Fields: source rank, destination rank order, send counts per destination rank, expert segment offsets, packed row order, original token indices, top-k slot indices, assignment ids.
- Invariants: sum of send counts equals assignment count on the source rank; segment offsets are monotonic; empty destinations and experts are represented explicitly.
- Never includes: router logits, expert weights, or transport backend implementation details.

### ExpertPlacement

- Purpose: define immutable expert ownership for one forward pass.
- Fields: world size, rank id, global expert count, mapping from expert id to owner rank, mapping from expert id to local expert slot on the owning rank, placement policy label.
- Invariants: every global expert has one owner; all ranks construct the same mapping; placement does not change inside one forward pass.
- Never includes: token-level routing decisions, packed buffers, performance results, or load-balancing side effects.

### CombinePlan

- Purpose: describe how expert outputs return to original token ownership and reduction order.
- Fields: return source and destination rank order, receive counts, assignment ids, original token indices, top-k slot indices, routing weights or stable references to them, output restore order.
- Invariants: every dispatched assignment has exactly one returned expert result; reduction can group by original token index; routing weights are applied exactly once.
- Never includes: router training losses, local expert parameters, or transport-specific handles.

### EPContext

- Purpose: hold execution context shared by planning and transport.
- Fields: execution mode, rank id, world size, device label, dtype policy label, backend label, collective sequence id or phase id, deterministic seed reference.
- Invariants: all ranks agree on world size, rank mapping, backend, and collective phase ordering for a distributed pass.
- Never includes: model parameters, token activations, routing logits, or benchmark conclusions.

### ProfileEvent

- Purpose: record a measured or counted execution phase.
- Fields: event name, rank id, phase id, start and end timestamps when measured, token count, byte count, expert id or rank peer when applicable, backend label, units.
- Invariants: event units are explicit; event names are stable enough to compare across runs; profiling does not affect correctness.
- Never includes: raw token payloads, secrets, mutable execution policy, or pass/fail correctness status.

## Transport Abstraction

The transport boundary accepts already-packed payloads and a plan. It returns received payloads and count metadata. It must not choose routes, reorder assignments outside the plan, apply weights, run experts, or decide correctness.

### Reference No-Op Transport

- Runs inside one process.
- Treats dispatch and combine as local data movement using the same plans as distributed mode.
- Exists to make dispatch and combine metadata testable before GPUs are involved.
- Must expose the same logical send and receive count checks as distributed transport.

### PyTorch Distributed Baseline Transport

- Runs after the reference path is correct.
- Uses PyTorch distributed collectives as the first 2-GPU baseline.
- Must support variable-size payloads per peer. The planning layer owns counts and offsets; the transport layer executes movement according to those counts.
- Must record collective phase ids so all ranks can prove they entered collectives in the same order.

### Future CUDA/NCCL-Oriented Path

- May replace CPU-side or framework-side packing and transport only after the 2-GPU PyTorch distributed path is correct.
- "Optimization" in this project means a measured reduction in a named phase, such as packing time, unpacking time, dispatch time, combine time, or bytes copied, while preserving reference-equivalent output.
- Must preserve the same metadata contracts so correctness tests can compare it against the reference and PyTorch distributed paths.

Communication is owned only by `nano_moe_ep.transport`. Router, placement, planning, layout metadata construction, expert computation, and weighted reduction must remain communication-agnostic.

## Execution Modes

### Correctness-First Reference Mode

- Single process, CPU-compatible, no distributed initialization required.
- Executes the full logical path: route, assign, group, pack, no-op dispatch, local expert compute, no-op combine, unpermute, weighted reduction.
- Prioritizes deterministic outputs, metadata inspection, invariant checks, and direct comparison against simple fixtures.

### Distributed Execution Mode

- Introduced only after the single-process path and deterministic dispatch/combine harness pass.
- Starts with 2 GPUs and PyTorch distributed collectives.
- Uses the same metadata concepts as reference mode, with transport replacing local movement.
- Must compare distributed output to reference output within a documented tolerance before any performance conclusion is accepted.

## Core Invariants

- Every routed token assignment arrives at the expert identified by its expert id and placement.
- No token assignment is silently duplicated or lost; the only intentional duplication is the `k` expert visits produced by top-k routing.
- Top-k routing weights are applied exactly once during reduction.
- Original token order is restored correctly before output activations leave the MoE FFN layer.
- Distributed output matches reference output within the configured numerical tolerance.
- Each rank agrees on collective ordering, peer order, counts, and phase ids.
- Expert ownership remains immutable within one forward pass.
- Empty expert segments, empty peer sends, and skewed routing distributions are valid states.
- Dispatch and combine plans reconcile: every dispatched assignment has exactly one returned result.
- Profiling cannot change routing, layout, transport, expert compute, or reduction behavior.
