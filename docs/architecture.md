# nano-moe-ep Architecture

## Current Verified Baseline

The repository currently contains Stage 1 and Stage 2 CPU-compatible implementations, Stage 2.75 execution-context metadata, a Stage 3 PyTorch distributed EP baseline, and Stage 4 multi-rank NCCL/Gloo correctness validation.

- Command: `python -m pytest -q`
- Result: `104 passed, 1 skipped`
- Stage 1 implemented: deterministic synthetic top-1 routing, `RouterOutput`, `TokenLayout`, `ExpertPlacement`, `ReferenceTrace`, grouped/permuted reference MoE FFN, and an independent token-by-token oracle.
- Top-k implemented: `TopKRouterOutput`, top-k synthetic routers, `TopKReferenceMoEFFN` with an expert capacity factor and deterministic token dropping, a shared capacity-mask drop policy, an independent token-by-token oracle, and communication / capacity cost-model benchmarks. The same routing, capacity, and drop policy run in the distributed path (`run_distributed_topk_ep_moe`), validated bit-for-bit against the reference in multi-process Gloo tests.
- Stage 2 implemented: `TokenAssignment`, rank-aware `TokenLayout`, table-driven `ExpertPlacement`, `DispatchPlan`, `CombinePlan`, `LogicalEPTrace`, and a single-process logical EP dispatch/combine simulation.
- Stage 2.75 implemented: `ExecutionMode` and `EPContext` metadata for one logical EP execution, with validation against `ExpertPlacement`.
- Stage 3 implemented: minimal `torch.distributed` forward path with explicit count exchange, variable-size dispatch/combine helpers, local expert execution, sharded combine, top-k capacity dropping, and a manual `torchrun` smoke script.
- Stage 4 validated: the general N-rank smoke has passed on 2, 4, and 8 CUDA/NCCL ranks for top-1, top-k, and top-k with capacity dropping; CI covers 2- and 4-rank Gloo E2E correctness.
- Not implemented: custom CUDA kernels, raw NCCL calls, Triton, a learned softmax gate, backward-specific logic, wall-clock benchmarks on real interconnects, or `ProfileEvent`. (Top-k routing, capacity factor, and token dropping now exist in both the single-process reference and the distributed forward path.)

Stage 2 simulates logical EP ranks inside one process. Stage 3 adds a real distributed transport boundary, but it is a correctness baseline rather than a performance implementation.

## Project Thesis

`nano-moe-ep` is a correctness-first, from-scratch educational runtime for the Expert Parallel data path in a standalone Mixture-of-Experts FFN block. Its purpose is to expose and test the core path: top-k routing, token grouping and permutation, variable-size cross-rank dispatch, local expert computation, cross-rank combine, unpermute, and weighted reduction. It is not a DeepEP, Megatron-Core, vLLM, or SGLang clone.

## In Scope

- Standalone MoE FFN before full Transformer integration.
- CPU reference correctness and independent oracle comparison.
- Deterministic synthetic top-1 and top-k routing.
- Single-process top-k reference with capacity factor and token dropping.
- Load-aware expert placement cost model and a balanced placement heuristic.
- Explicit metadata for routing, execution context, placement, token layout, assignment, dispatch, and combine.
- Single-process logical-rank dispatch/combine simulation.
- Minimal 2-rank PyTorch distributed baseline after Stage 2 metadata remains stable.
- Later measurement of routing skew, packing/permutation cost, communication cost, expert compute, and overlap.

## Out of Scope

- Full LLM training or inference serving.
- Tensor parallelism, pipeline parallelism, data parallelism, speculative decoding, schedulers, autoscaling, HTTP services, or production control planes.
- Real distributed communication before Stage 3.
- CUDA/NCCL kernels before correctness and baseline measurements justify them.
- A learned softmax gate, and backward propagation anywhere.
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
| Single-process logical and Stage 3 distributed transport     |
+--------------------------------------------------------------+
| Local expert MLP computation                                 |
+--------------------------------------------------------------+
| Future transport/profiling boundaries                        |
+--------------------------------------------------------------+
```

The Stage 2 transport is only a local simulation. Stage 3 replaces that movement boundary with PyTorch distributed collectives without changing router semantics or expert ownership.

## Implemented One-Layer Data Flow

1. Input activations enter as `[num_tokens, hidden_dim]`.
2. Synthetic top-1 routing emits `RouterOutput.expert_indices` and `RouterOutput.weights`, both shaped `[num_tokens, 1]`.
3. `ExpertPlacement` maps every expert to exactly one logical EP rank.
4. Optional `EPContext` metadata records execution mode, logical world size, optional local rank, CPU device marker, deterministic flag, and phase name.
5. `run_logical_ep_moe` validates `EPContext.num_ep_ranks` against `ExpertPlacement.num_ep_ranks`.
6. `build_token_assignments` creates one `TokenAssignment` per token with token id, expert id, rank id, routing weight, and original position.
7. `build_logical_ep_layout` sorts assignments by destination rank, then expert id, then original token order.
8. Rank-aware `TokenLayout` records token permutation, inverse permutation, expert counts/offsets, rank counts/offsets, and rank/expert counts/offsets.
9. `build_dispatch_plan` creates one logical payload per destination rank and validates payload order against the layout permutation.
10. `simulate_dispatch` gathers local payload tensors without communication.
11. `execute_local_experts` runs each expert only on the slices assigned to that rank/expert segment.
12. `build_combine_plan` records token indices and routing weights in dispatch order.
13. `apply_combine_plan` applies routing weights exactly once and restores original token order.
14. `run_logical_ep_moe` returns output activations plus `LogicalEPTrace`.

## Stage 3 Distributed Data Flow

1. Every rank constructs the same deterministic input, router output, expert weights, and `ExpertPlacement`.
2. Each rank owns a deterministic source-token shard by `token_index % world_size`.
3. Each rank builds the same global assignment/layout metadata, then derives its local variable-size send payloads by destination expert rank.
4. `exchange_counts` gathers pairwise send counts before any payload exchange.
5. Tokens, token ids, expert ids, and routing weights are exchanged to destination expert ranks.
6. Each rank validates that received expert ids belong to experts it owns, then executes only those local experts.
7. Expert outputs, token ids, and routing weights are exchanged back to source-token ranks.
8. Each source rank applies routing weights exactly once into a zero-padded full output tensor.
9. An `all_reduce` sums the disjoint source-token contributions so every rank can compare the full output to the local reference.

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
- Inputs: input activations, `RouterOutput`, expert modules, `ExpertPlacement`, and optional `EPContext`.
- Outputs: output activations and `LogicalEPTrace`.
- Must never own: route selection, distributed collectives, CUDA kernels, top-2 reduction policy, or benchmarks.

### `nano_moe_ep.distributed_ep`

- Responsibility: Stage 3 distributed count exchange, variable-size tensor exchange, source-token reconstruction, and distributed forward orchestration.
- Inputs: input activations, `RouterOutput`, expert modules, `ExpertPlacement`, and initialized `torch.distributed` rank metadata.
- Outputs: output activations and `DistributedEPTrace`.
- Must never own: route selection, expert placement semantics, custom kernels, top-2 reduction policy, backward, or benchmarks.

### `nano_moe_ep.types`

- Responsibility: shared metadata validation.
- Current note: `types.py` is larger after Stage 2.75, but still acceptable because all records are small metadata contracts. A split into `metadata.py`, `placement.py`, or `plans.py` should wait until Stage 3 pressure justifies API movement.
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
- Fields: assignments, token layout, dispatch plan, combine plan, expert placement, EP context.
- Purpose: inspect Stage 2 behavior without hiding metadata.

### EPContext

- Status: implemented as Stage 2.75 preparation metadata.
- Fields: `num_ep_ranks`, optional `local_rank`, `execution_mode`, optional CPU-only `device`, `deterministic`, and `phase`.
- Purpose: describe one logical EP execution and validate that context world size agrees with expert placement before future distributed transport is added.
- Non-purpose: route selection, expert ownership, token layout, communication, or device execution.

### ProfileEvent

- Status: proposed for Stage 3+.
- Purpose: named phase timing/count/byte records that do not alter execution.

### DistributedEPConfig / DistributedPayloadPlan / DistributedEPTrace

- Status: implemented for Stage 3.
- Purpose: keep distributed runtime metadata, local source payload metadata, count exchange metadata, and trace fields explicit.
- Invariants: config matches the initialized process group; send and receive counts are non-negative and pairwise consistent; returned token ids match the rank's deterministic source-token shard.

## Transport Boundary

### Current Single-Process Simulation

- Uses local `index_select`, Python tuples, and tensors.
- Simulates one payload per logical destination rank.
- Has no `torch.distributed`, CUDA, NCCL, or multiprocessing.

### PyTorch Distributed Baseline

- Target: Stage 3.
- Reuses `ExpertPlacement`, `TokenAssignment`, `TokenLayout`, and once-only weight application concepts.
- Transport should execute movement according to counts and peer order, not choose routes or apply weights.
- CPU/Gloo smoke is supported for correctness-only environments; NCCL is used only when CUDA/NCCL are available.

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
