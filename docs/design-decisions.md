# Design Decisions

These ADRs describe the current project direction after Stage 2. They are allowed to change when tests or measurements show that a decision is blocking correctness or learning value.

## ADR-0001: Start With A Standalone MoE FFN

- Decision: implement a standalone MoE FFN before any full Transformer or LLM.
- Rationale: the project is about the EP data path: routing, layout, dispatch, local expert compute, combine, unpermute, and weighted reduction. A full Transformer would add unrelated attention, normalization, embeddings, and training concerns.
- Tradeoff: early results do not prove full-model integration.
- Revisit condition: the standalone FFN path is correct in reference and 2-GPU modes, and integration becomes the next smallest useful task.

## ADR-0002: Correctness Before Multi-GPU

- Decision: single-process correctness must precede distributed execution.
- Rationale: token identity, permutation, expert counts, offsets, and weight application are easier to debug without rank state, collectives, or asynchronous device behavior.
- Tradeoff: Stage 1 does not measure communication or GPU behavior.
- Revisit condition: a future invariant cannot be simulated locally and must be validated with a distributed-only test.

## ADR-0003: Top-1 First, Top-k Later

- Decision: Stage 1 to Stage 3 implement top-1 only; top-k is then added in the single-process reference (`TopKReferenceMoEFFN`) before being pushed into the distributed path.
- Rationale: top-1 proves the essential layout, dispatch/combine, and oracle path without multi-assignment reduction complexity; top-k is then validated against an oracle in isolation before distributed dispatch/combine inherits its complexity.
- Status: top-k routing with multi-slot weighted reduction is implemented and oracle-tested in the single-process reference, and the distributed path (`run_distributed_topk_ep_moe`) now runs the same top-k dispatch/combine with capacity dropping, validated bit-for-bit against the reference in multi-process tests.
- Revisit condition: a learned softmax gate or a top-k-aware expert placement cost model becomes the next correctness or efficiency gap.

## ADR-0004: Variable-Size Dispatch Is Core

- Decision: variable token counts per expert and per rank are core to the design.
- Rationale: routing skew is the interesting EP systems problem; fixed-size padding would hide count reconciliation, empty experts, hot experts, and load imbalance.
- Tradeoff: planning and tests are more complex than a dense or padded toy implementation.
- Revisit condition: a temporary fixed-size fixture is needed for debugging, and it remains isolated from the architecture.

## ADR-0005: Routing Weights Are Validated And Applied Once

- Decision: routing weights must be finite floating tensors with shape `[num_tokens, 1]` in Stage 1, and the reference path applies them exactly once.
- Rationale: incorrect weight application can silently make a grouped implementation appear structurally correct while producing wrong activations.
- Tradeoff: Stage 1 tests include manually constructed non-unit weights even though synthetic routers emit unit weights.
- Revisit condition: top-2 routing requires a generalized reduction contract across multiple slots per token.

## ADR-0006: PyTorch Distributed Before Custom CUDA

- Decision: the first real distributed implementation should use PyTorch distributed before custom CUDA/NCCL-oriented kernels.
- Rationale: correctness of counts, peer order, dispatch/combine, and reference equivalence should be proven before maintaining custom kernels.
- Tradeoff: PyTorch distributed baseline timings may include overhead not present in later specialized paths.
- Revisit condition: 2-GPU correctness passes and profiling identifies a named phase worth replacing.

## ADR-0007: 2 GPUs Before 4 GPUs

- Decision: the first distributed target is 2 GPUs; 4 GPUs are a later scaling and skew validation target.
- Rationale: 2 ranks are enough to exercise cross-rank dispatch/combine and collective ordering with the smallest debugging surface.
- Tradeoff: early distributed tests do not show 4-rank skew or placement behavior.
- Revisit condition: 2-GPU reference equivalence, empty-peer cases, and skewed-count cases are stable.

## ADR-0008: Backward Is Deferred

- Decision: Stage 1, Stage 2, and the first distributed milestones are forward-only.
- Rationale: backward adds gradient routing, parameter gradients, optimizer state, and more collectives before forward EP correctness is established.
- Tradeoff: the project cannot train end to end in early stages.
- Revisit condition: forward reference, Stage 2 harness, and Stage 3 2-GPU baseline all pass correctness gates.

## ADR-0009: Capacity Factor And Token Dropping (Reference Implemented)

- Decision: all-token delivery was proven first; capacity factor and token dropping are now implemented as a separately tested mode in the single-process top-k reference (`capacity_factor=None` keeps every token; a finite factor drops per a deterministic policy).
- Rationale: dropping tokens intentionally changes outputs, so it is validated in isolation against a token-by-token oracle that shares the exact same drop mask before the distributed path inherits it.
- Drop policy: per-expert capacity is `ceil(capacity_factor * num_tokens * k / num_experts)`; assignments beyond capacity are dropped in row-major (token, then gate slot) order. A dropped assignment contributes nothing; the token keeps its kept experts' contributions.
- Status: the distributed dispatch/combine path adopts the exact same capacity and drop policy (the mask is recomputed identically on every rank from the replicated router output), using the reference as its oracle in multi-process tests.

## ADR-0010: Avoid A Vague Mini-DeepEP Clone

- Decision: the project must stay narrow: explicit metadata, reference correctness, staged distributed validation, and measured bottlenecks.
- Rationale: copying broad production framework shapes would obscure the learning objective and create a worse version of existing systems.
- Tradeoff: many production features are intentionally absent.
- Revisit condition: the final benchmark report is complete and a new project objective is chosen deliberately.
