# nano-moe-ep Architecture

## Current Verified Baseline

The repository currently contains Stage 1 and Stage 2 CPU-compatible implementations.

- Command: `python -m pytest -q`
- Result: `43 passed in 2.65s`
- Stage 1 implemented: deterministic synthetic top-1 routing, `RouterOutput`, `TokenLayout`, `ExpertPlacement`, `ReferenceTrace`, grouped/permuted reference MoE FFN, and an independent token-by-token oracle.
- Stage 2 implemented: `TokenAssignment`, rank-aware `TokenLayout`, table-driven `ExpertPlacement`, `DispatchPlan`, `CombinePlan`, `LogicalEPTrace`, and a single-process logical EP dispatch/combine simulation.
- Not implemented: real distributed EP, `torch.distributed`, CUDA, NCCL, multiprocessing, custom kernels, top-2 routing, capacity factor, token dropping, backward-specific logic, benchmarks, `EPContext`, or `ProfileEvent`.

Stage 2 simulates logical EP ranks inside one process. It is not a real multi-GPU implementation.

## Project Thesis

`nano-moe-ep` is a correctness-first, from-scratch educational runtime for the Expert Parallel data path in a standalone Mixture-of-Experts FFN block. Its purpose is to expose and test the core path: top-k routing, token grouping and permutation, variable-size cross-rank dispatch, local expert computation, cross-rank combine, unpermute, and weighted reduction. It is not a DeepEP, Megatron-Core, vLLM, or SGLang clone.

## In Scope

- Standalone MoE FFN before full Transformer integration.
- CPU reference correctness and independent oracle comparison.
- Deterministic synthetic top-1 routing.
- Explicit metadata for routing, placement, token layout, assignment, dispatch, and combine.
- Single-process logical-rank dispatch/combine simulation.
- Future 2-GPU PyTorch distributed baseline after Stage 2 metadata remains stable.
- Later measurement of routing skew, packing/permutation cost, communication cost, expert compute, and overlap.

## Out of Scope

- Full LLM training or inference serving.
- Tensor parallelism, pipeline parallelism, data parallelism, speculative decoding, schedulers, autoscaling, HTTP services, or production control planes.
- Real distributed communication before Stage 3.
- CUDA/NCCL kernels before correctness and baseline measurements justify them.
- Top-2 routing, capacity factor, token dropping, and backward propagation in the current implementation.
- Multi-node, RDMA, InfiniBand, FP8, or production deployment.

## Layered Architecture

```text
+--------------------------------------------------------------+
| Tests and correctness oracle                                 |
+--------------------------------------------------------------+
| Public MoE FFN reference and logical EP entry points         |
+--------------------------------------------------------------+
| Router: top-1 expert ids and routing weights                 |
+--------------------------------------------------------------+
| Metadata: placement, assignment, layout, dispatch, combine   |
+--------------------------------------------------------------+
| Layout: rank -> expert -> original-token order               |
+--------------------------------------------------------------+
| Single-process logical dispatch/combine simulation           |
+--------------------------------------------------------------+
| Local expert MLP computation                                 |
+--------------------------------------------------------------+
| Future transport/profiling boundaries                        |
+--------------------------------------------------------------+
```

The current transport is only a local simulation. Future Stage 3 code should be able to replace the simulated movement boundary without changing router semantics or expert execution.

## Implemented One-Layer Data Flow

1. Input activations enter as `[num_tokens, hidden_dim]`.
2. Synthetic top-1 routing emits `RouterOutput.expert_indices` and `RouterOutput.weights`, both shaped `[num_tokens, 1]`.
3. `ExpertPlacement` maps every expert to exactly one logical EP rank.
4. `build_token_assignments` creates one `TokenAssignment` per token with token id, expert id, rank id, routing weight, and original position.
5. `build_logical_ep_layout` sorts assignments by destination rank, then expert id, then original token order.
6. Rank-aware `TokenLayout` records token permutation, inverse permutation, expert counts/offsets, rank counts/offsets, and rank/expert counts/offsets.
7. `build_dispatch_plan` creates one logical payload per destination rank and validates payload order against the layout permutation.
8. `simulate_dispatch` gathers local payload tensors without communication.
9. `execute_local_experts` runs each expert only on the slices assigned to that rank/expert segment.
10. `build_combine_plan` records token indices and routing weights in dispatch order.
11. `apply_combine_plan` applies routing weights exactly once and restores original token order.
12. `run_logical_ep_moe` returns output activations plus `LogicalEPTrace`.

## Module Boundaries

### `nano_moe_ep.routing`

- Responsibility: deterministic synthetic top-1 routing.
- Inputs: token count or explicit assignment, number of experts.
- Outputs: `RouterOutput`.
- Must never own: token layout, expert placement, dispatch, combine, or expert execution.

### `nano_moe_ep.reference`

- Responsibility: Stage 1 reference MoE FFN and independent oracle.
- Inputs: input activations, `RouterOutput`, expert modules.
- Outputs: output activations and `ReferenceTrace`.
- Must never own: logical-rank simulation, distributed communication, or profiling.

### `nano_moe_ep.dispatch_combine`

- Responsibility: Stage 2 logical-rank assignment, layout, dispatch simulation, local expert execution orchestration, combine planning, and final restore.
- Inputs: input activations, `RouterOutput`, expert modules, `ExpertPlacement`.
- Outputs: output activations and `LogicalEPTrace`.
- Must never own: route selection, distributed collectives, CUDA kernels, top-2 reduction policy, or benchmarks.

### `nano_moe_ep.types`

- Responsibility: shared metadata validation.
- Current note: `types.py` is larger after Stage 2, but still acceptable because all records are small metadata contracts. A split into `metadata.py`, `placement.py`, or `plans.py` should wait until Stage 3 pressure justifies API movement.
- Must never own: tensor computation, communication execution, or benchmark interpretation.

## Metadata Concepts

### RouterOutput

- Status: implemented.
- Fields: `expert_indices`, `weights`.
- Invariants: shape `[num_tokens, 1]`; integer expert ids; floating finite weights; matching token count.

### ExpertPlacement

- Status: implemented.
- Fields: `owner_rank_by_expert`, `num_ep_ranks`.
- Invariants: every expert appears exactly once; empty ranks are allowed; rank ids are in `[0, num_ep_ranks)`.

### TokenAssignment

- Status: implemented.
- Fields: `token_index`, `expert_id`, `ep_rank`, `routing_weight`, `original_position`.
- Invariants: one top-1 assignment per token; non-negative ids; stable original order key.

### TokenLayout

- Status: implemented.
- Fields: `permutation`, `inverse_permutation`, `expert_offsets`, `expert_counts`, `rank_offsets`, `rank_counts`, `rank_expert_offsets`, `rank_expert_counts`.
- Invariants: every token appears exactly once; inverse restores original order; count and offset metadata reconcile; empty experts and empty ranks are valid.
- Stage 2 semantic note: `rank_expert_offsets` describe contiguous expert slices inside each rank-major payload. `expert_offsets` remain global count-prefix metadata and should not be used to slice a rank-major payload directly.

### DispatchPlan

- Status: implemented.
- Fields: rank-aware layout, sorted assignments, payload token indices by rank.
- Invariants: one payload per logical rank; payload count equals assignment count; payload token order matches layout permutation.

### CombinePlan

- Status: implemented.
- Fields: token indices, routing weights, token count.
- Invariants: every token index appears exactly once; weights are finite and aligned to dispatch order.

### LogicalEPTrace

- Status: implemented.
- Fields: assignments, token layout, dispatch plan, combine plan, expert placement.
- Purpose: inspect Stage 2 behavior without hiding metadata.

### EPContext

- Status: proposed for Stage 3.
- Purpose: rank/world/backend/device/dtype/phase metadata for real distributed execution.

### ProfileEvent

- Status: proposed for Stage 3+.
- Purpose: named phase timing/count/byte records that do not alter execution.

## Transport Boundary

### Current Single-Process Simulation

- Uses local `index_select`, Python tuples, and tensors.
- Simulates one payload per logical destination rank.
- Has no `torch.distributed`, CUDA, NCCL, or multiprocessing.

### Future PyTorch Distributed Baseline

- Target: Stage 3.
- Should reuse `ExpertPlacement`, `TokenAssignment`, `TokenLayout`, `DispatchPlan`, and `CombinePlan` concepts.
- Transport should execute movement according to counts and peer order, not choose routes or apply weights.

### Future CUDA/NCCL Path

- Target: Stage 5 or later.
- May replace measured packing/permutation or communication bottlenecks only after reference and 2-GPU correctness gates pass.

## Correctness Invariants

- Every top-1 token assignment is represented exactly once.
- No token is silently dropped or duplicated.
- Expert ids and rank ids are valid before local expert execution.
- Empty expert segments and empty logical ranks are valid.
- Rank/expert counts and offsets reconcile with token counts.
- Dispatch payload order matches layout permutation.
- Combine token indices contain every token exactly once.
- Routing weights are applied exactly once in `apply_combine_plan`.
- Logical EP output matches the Stage 1 reference within tolerance.
- Future distributed output must match the Stage 1 reference within tolerance.
