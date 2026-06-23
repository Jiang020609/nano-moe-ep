# Design Decisions

These records describe current design proposals. They are not final architecture law; they should be revisited when implementation evidence contradicts an assumption.

## ADR-0001: Start With A Standalone MoE FFN

- Decision: the first implementation target is a standalone MoE FFN block, not a full Transformer or LLM.
- Reasoning: Expert Parallel correctness depends on routing, permutation, variable-size dispatch, local expert execution, combine, and unpermute. A full Transformer would add attention, residual paths, layer normalization, embeddings, and training concerns before the EP data path is proven.
- Tradeoff: this postpones realistic model integration and hides interactions with full-model memory layout.
- Revisit when: the standalone FFN forward path is correct in reference mode and 2-GPU mode, and integration issues become the next blocker.

## ADR-0002: Single-Process Correctness Comes First

- Decision: the reference path must work in one process before any multi-GPU execution is attempted.
- Reasoning: token routing and reduction bugs are easier to isolate without distributed process state, collective ordering, device placement, or asynchronous execution.
- Tradeoff: early work will not demonstrate GPU communication behavior.
- Revisit when: the reference path cannot represent a distributed-only invariant; that invariant should then be added as a simulated check before GPU work starts.

## ADR-0003: Use PyTorch Distributed Before Custom CUDA

- Decision: the first distributed baseline should use PyTorch distributed collectives.
- Reasoning: PyTorch distributed provides a known communication substrate, allowing the project to validate counts, peer ordering, payload shape, and correctness before maintaining custom kernels.
- Tradeoff: the baseline may include overhead that a later CUDA/NCCL-oriented path can reduce, so its timings should be interpreted as baseline measurements rather than final performance goals.
- Revisit when: the 2-GPU PyTorch distributed path passes correctness tests and profiling identifies a named phase whose measured cost justifies custom work.

## ADR-0004: Variable-Size Dispatch Is Core

- Decision: variable token counts per expert and per rank are a first-class requirement.
- Reasoning: real top-k routing can be skewed. Padding every expert or rank to a fixed capacity would hide the core EP problem this project exists to study.
- Tradeoff: plans, offsets, counts, tests, and collectives become more complex from the beginning.
- Revisit when: an implementation stage needs a temporary fixed-size fixture for debugging; that fixture must remain a test helper and not become the architecture.

## ADR-0005: Expert Placement Is Explicit Metadata

- Decision: expert placement should be represented as immutable metadata mapping global expert ids to owner ranks and local expert slots.
- Reasoning: routing produces expert ids, while dispatch needs rank destinations. Keeping placement explicit prevents router, planner, and transport responsibilities from blending together.
- Tradeoff: every mode must carry placement metadata, even in single-process reference runs where all experts are local.
- Revisit when: dynamic expert migration becomes an intentional post-MVP feature. It is not part of the first project version.

## ADR-0006: Design For Top-k, Implement Top-1 First

- Decision: metadata and tests should support general top-k routing, while the smallest Stage 1 implementation should use top-1.
- Reasoning: top-2 and higher make weighted reduction and duplicate token visits unavoidable, so the design must not hard-code top-1 assumptions. Starting with top-1 keeps the first executable target small enough to validate thoroughly.
- Tradeoff: top-2 behavior will be designed before it is fully exercised by the first implementation.
- Revisit when: top-1 passes reference and dispatch/combine tests; then top-2 should be enabled before judging the architecture complete.

## ADR-0007: No Backward Propagation In The First Milestone

- Decision: Stage 1 should be forward-only.
- Reasoning: the project thesis is about the EP forward data path. Backward propagation introduces gradient routing, parameter gradients, optimizer state, and additional collectives before forward correctness is established.
- Tradeoff: early implementation cannot train a model end to end.
- Revisit when: forward reference mode, deterministic dispatch/combine, and the 2-GPU forward baseline are correct and measured.

## ADR-0008: No Capacity Factor Or Token Dropping Initially

- Decision: the first version should not drop tokens and should not enforce a capacity factor.
- Reasoning: token dropping intentionally changes outputs and adds policy choices. The first correctness target should prove that all routed assignments are delivered and reduced.
- Tradeoff: pathological skew may produce large variable-size batches and is not bounded by capacity.
- Revisit when: all-token delivery is correct and the project needs to study capacity-limited routing as a separate, explicitly tested mode.

## ADR-0009: Avoid Becoming A Mini Production Framework

- Decision: the project should stay scoped to a standalone MoE FFN EP data path and its measurements.
- Reasoning: adding a service, scheduler, full Megatron-style training stack, serving path, or broad parallelism framework would dilute the learning objective and make correctness harder to audit.
- Tradeoff: the project will not look like a complete MoE training or inference system.
- Revisit when: the final benchmark report is complete and a new project goal is explicitly chosen.

## ADR-0010: Postpone Non-Essential Functionality Until After 2-GPU Correctness

- Decision: backward propagation, custom CUDA packing, communication-computation overlap, 4-GPU scaling, capacity-limited routing, full-Transformer integration, advanced load balancing, and production-serving concerns are postponed until after a correct 2-GPU forward path.
- Reasoning: the first meaningful distributed milestone is small: prove that variable-size EP dispatch and combine match the single-process reference output.
- Tradeoff: several interesting systems topics wait behind correctness gates.
- Revisit when: Stage 3 acceptance tests pass and profiling data identifies the next bottleneck to study.
