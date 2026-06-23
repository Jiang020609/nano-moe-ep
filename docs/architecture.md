# nano-moe-ep Architecture

## Current Verified Baseline

This repository currently contains a Stage 1 single-process CPU reference implementation and tests. The verified baseline for this documentation pass is:

- Command: `python -m pytest -q`
- Result: `28 passed in 3.96s`
- Implemented runtime scope: deterministic synthetic top-1 routing, explicit `RouterOutput`, `TokenLayout`, `ExpertPlacement`, `ReferenceTrace`, a grouped/permuted reference MoE FFN, and an independent token-by-token oracle.
- Implemented tests cover round-robin routing, explicit routing, one-token input, all tokens routed to one expert, skewed expert counts, non-contiguous input tensors, invalid routing metadata, finite floating weights, inverse permutation, no silent token loss or duplication, determinism, and grouped output compared to the oracle.
- Not implemented: distributed EP, `torch.distributed`, CUDA/NCCL kernels, transport, `DispatchPlan`, `CombinePlan`, `EPContext`, profiling events, top-2 routing, capacity factor, token dropping, backward-specific logic, or benchmarks.

Everything beyond Stage 1 in this document is a design proposal. It must not be treated as existing implementation.

## Project Thesis

`nano-moe-ep` is a correctness-first, from-scratch educational Expert Parallel runtime for Mixture-of-Experts models. Its purpose is to expose the essential MoE EP data path in a small, testable system: top-k routing, token grouping and permutation, variable-size cross-rank dispatch, local expert computation, cross-rank combine, unpermute, and weighted reduction. It is not a DeepEP, Megatron-Core, vLLM, or SGLang clone; it is a narrow systems-learning project that makes correctness invariants and bottlenecks explicit.

## In Scope

- A standalone MoE FFN layer before any full Transformer integration.
- Single-process CPU/reference execution with deterministic tests.
- Explicit routing, layout, placement, dispatch, combine, context, and profiling metadata.
- Variable token counts per expert and, later, per EP rank.
- A future 2-GPU PyTorch distributed baseline after the reference path and local dispatch/combine harness are correct.
- Later measurement of routing skew, packing/permutation cost, communication cost, expert compute, and communication-computation overlap.

## Out of Scope

- Full LLM training or inference serving.
- Tensor parallelism, pipeline parallelism, data parallelism, speculative decoding, autoscaling, schedulers, HTTP services, or production control planes.
- CUDA/NCCL kernels before the 2-GPU PyTorch distributed baseline is correct.
- Top-2 routing, capacity factor, token dropping, and backward propagation in Stage 1.
- Multi-node, RDMA, InfiniBand, FP8, or production deployment.
- Copying implementation code from DeepEP, Megatron-Core, vLLM, SGLang, or similar systems.

## Layered Architecture

```text
+--------------------------------------------------------------+
| Tests and correctness oracle                                 |
+--------------------------------------------------------------+
| Public MoE FFN reference layer                               |
+--------------------------------------------------------------+
| Router: top-k expert ids and routing weights                 |
+--------------------------------------------------------------+
| Metadata: assignment, layout, placement, dispatch, combine   |
+--------------------------------------------------------------+
| Layout: group, permute, pack, unpack, unpermute              |
+--------------------------------------------------------------+
| Transport: reference no-op, future torch.distributed, CUDA   |
+--------------------------------------------------------------+
| Local expert MLP computation                                 |
+--------------------------------------------------------------+
| Profiling and benchmark event records                        |
+--------------------------------------------------------------+
```

Stage 1 implements only the single-process router, metadata, layout, local expert compute, and oracle parts. The transport and profiling layers are future boundaries.

## One-Layer MoE Data Flow

1. Input activations enter as `[num_tokens, hidden_dim]`.
2. The router emits expert ids and routing weights. Stage 1 uses deterministic synthetic top-1 routing only.
3. `RouterOutput` validates shape `[num_tokens, 1]`, integer expert ids, floating finite weights, and matching token count.
4. Token assignments are grouped by expert id.
5. `TokenLayout` records a permutation from original token order to expert-grouped order.
6. `TokenLayout.inverse_permutation` restores original token order from grouped order.
7. `expert_counts` and `expert_offsets` describe per-expert token segments, including empty experts.
8. Stage 1 runs each local expert MLP only on its assigned token slice.
9. Expert outputs are unpermuted back to original token order.
10. Routing weights are applied exactly once.
11. Output activations leave with the same token order and shape as input activations.
12. The grouped implementation is checked against a token-by-token oracle that does not use layout metadata.

Future distributed stages insert variable-size dispatch after packing and cross-rank combine before unpermute/reduction.

## Proposed Module Boundaries

### `nano_moe_ep.routing`

- Current responsibility: deterministic synthetic top-1 routing.
- Future responsibility: top-k router output formatting without owning layout or transport.
- Inputs: token count or explicit assignment, number of experts.
- Outputs: `RouterOutput`.
- Invariants: expert ids are in range; weights are finite; Stage 1 synthetic weights are all `1.0`.
- Must never own: token permutation, expert execution, dispatch, combine, or distributed communication.

### `nano_moe_ep.types`

- Current responsibility: explicit Stage 1 metadata dataclasses.
- Future responsibility: shared metadata contracts for assignment, layout, dispatch, combine, context, and profiling.
- Inputs: tensors and immutable placement metadata produced by router/layout/planner layers.
- Outputs: validated metadata objects.
- Invariants: metadata must be auditable and must not hide token movement.
- Must never own: tensor compute kernels, communication, training losses, or benchmark interpretation.

### `nano_moe_ep.reference`

- Current responsibility: Stage 1 grouped/permuted reference MoE FFN and independent token-by-token oracle.
- Future responsibility: correctness oracle for distributed outputs.
- Inputs: input activations, `RouterOutput`, local expert MLPs.
- Outputs: output activations and `ReferenceTrace`.
- Invariants: no distributed initialization is required; output matches the oracle within tolerance.
- Must never own: production transport, distributed scheduling, CUDA kernels, or serving logic.

### Future `planning`

- Responsibility: convert routing plus placement into `TokenAssignment`, `DispatchPlan`, and `CombinePlan`.
- Inputs: `RouterOutput`, `TokenLayout`, `ExpertPlacement`, rank/world metadata.
- Outputs: auditable send/receive and return plans.
- Invariants: all assignments are represented exactly once for top-1; later top-k duplication is explicit.
- Must never own: collectives, expert MLP parameters, or router policy.

### Future `layout`

- Responsibility: pack and unpack tensors according to plans.
- Inputs: token activations and planning metadata.
- Outputs: packed send buffers and restored output buffers.
- Invariants: pack/unpack round trips preserve token identity and segment boundaries.
- Must never own: routing decisions, communication backend, or expert ownership policy.

### Future `transport`

- Responsibility: move already-packed buffers according to dispatch/combine plans.
- Inputs: packed tensors, counts, peer order, context.
- Outputs: received tensors and transport profile events.
- Invariants: all ranks agree on collective order and counts.
- Must never own: route selection, weight application, expert compute, or correctness comparison.

### Future `profiling`

- Responsibility: record phase timings, token counts, bytes, skew, and backend labels.
- Inputs: phase boundaries and counters.
- Outputs: `ProfileEvent` records.
- Invariants: profiling must not alter correctness behavior.
- Must never own: execution policy or pass/fail correctness decisions.

## Metadata Concepts

### RouterOutput

- Current status: implemented.
- Concept: selected expert ids and routing weights.
- Current fields: `expert_indices` with shape `[num_tokens, 1]`; `weights` with shape `[num_tokens, 1]`.
- Invariants: integer expert ids; floating finite weights; matching first dimension; Stage 1 router-generated weights are unit weights.

### TokenAssignment

- Current status: proposed for Stage 2+.
- Concept: one routed token visit to one expert.
- Proposed fields: original token index, top-k slot, expert id, routing weight, owner rank, assignment id.
- Invariants: assignment identity survives packing, dispatch, combine, and reduction.

### TokenLayout

- Current status: implemented.
- Concept: local token grouping metadata.
- Current fields: `permutation`, `inverse_permutation`, `expert_offsets`, `expert_counts`.
- Invariants: every token appears once; inverse restores original order; offsets match counts; empty experts are valid.

### DispatchPlan

- Current status: proposed for Stage 2+.
- Concept: variable-size send plan from source rank to destination rank/expert segments.
- Proposed fields: source rank, destination rank order, send counts, expert segment offsets, packed row order, assignment ids.
- Invariants: send counts reconcile with assignment count; peer order is deterministic.

### ExpertPlacement

- Current status: implemented in minimal Stage 1 form.
- Concept: immutable expert ownership metadata.
- Current fields: `owner_rank_by_expert`; Stage 1 defaults every expert to rank 0.
- Invariants: each expert has exactly one owner during a forward pass.

### CombinePlan

- Current status: proposed for Stage 2+.
- Concept: return path from expert outputs back to original token ownership.
- Proposed fields: return peer order, receive counts, assignment ids, original token indices, top-k slots, weight references.
- Invariants: every dispatched assignment returns exactly one result; weights are applied once.

### EPContext

- Current status: proposed for Stage 3+.
- Concept: execution context shared across ranks.
- Proposed fields: rank id, world size, backend, device label, dtype policy, deterministic seed, collective phase id.
- Invariants: all ranks agree on world size, peer order, backend, and phase ordering.

### ProfileEvent

- Current status: proposed for Stage 3+.
- Concept: structured measurement record.
- Proposed fields: phase name, rank id, timestamps, token count, byte count, peer/expert id, backend, units.
- Invariants: measurements are labeled and never change execution behavior.

## Transport Abstraction

### Single-Process Reference

- Current status: Stage 1 uses local grouping and unpermutation only.
- Purpose: validate correctness before communication exists.
- Contract: no distributed initialization, no collectives, no CUDA/NCCL dependency.

### Future PyTorch Distributed Baseline

- Target stage: Stage 3.
- Purpose: validate 2-GPU variable-size dispatch/combine using a standard collective backend.
- Contract: planning owns counts and peer order; transport executes movement without interpreting payloads.

### Future CUDA/NCCL Path

- Target stage: Stage 5 or later.
- Purpose: replace measured bottlenecks in packing/permutation or communication after correctness gates pass.
- Contract: must preserve the same metadata and reference-equivalence tests.

## Correctness Invariants

- Every top-1 token assignment is represented exactly once.
- No token is silently dropped or duplicated.
- Expert ids are valid before expert execution.
- Empty expert segments are valid.
- `expert_counts` and `expert_offsets` reconcile with the number of tokens.
- `permutation` contains every token index exactly once.
- `inverse_permutation` restores original token ordering.
- Routing weights are finite, floating point, shaped correctly, and applied exactly once.
- Grouped/permuted execution matches the independent token-by-token oracle within tolerance.
- Future distributed output must match the single-process reference within tolerance.
- Future ranks must agree on peer order, counts, placement, and collective phase order.
- Expert placement is immutable within one forward pass.
