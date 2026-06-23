# nano-moe-ep

`nano-moe-ep` is a correctness-first, from-scratch educational runtime for the Expert Parallel data path in a standalone Mixture-of-Experts FFN block.

## Status

Current status: Stage 1 single-process CPU reference implementation. The verified baseline is `python -m pytest -q` with `28 passed`.

The repository contains deterministic synthetic top-1 routing, explicit metadata, a grouped/permuted reference MoE FFN, an independent token-by-token oracle, and CPU tests.

It does not yet contain distributed Expert Parallel execution, CUDA, NCCL, multiprocessing, custom kernels, top-2 routing, capacity factor, token dropping, backward logic, or benchmarking.

## Documentation

- [Architecture](docs/architecture.md)
- [Design decisions](docs/design-decisions.md)
- [Implementation roadmap](docs/implementation-roadmap.md)
- [Test strategy](docs/test-strategy.md)

## Milestones

- Stage 0 — Architecture and invariants
- Stage 1 — Single-process reference MoE FFN
- Stage 2 — Deterministic dispatch/combine harness
- Stage 3 — 2-GPU EP baseline
- Stage 4 — 4-GPU scaling and skew experiments
- Stage 5 — GPU-side packing/permutation optimization
- Stage 6 — Communication-computation overlap
- Stage 7 — Final benchmark report

## Quick Start

```bash
python -m pytest -q
```

## Non-Goals

- No full LLM, full Transformer, inference server, scheduler, autoscaler, or training platform.
- No tensor parallelism, pipeline parallelism, data parallelism, speculative decoding, serving stack, or dynamic expert migration in the first project version.
- No custom CUDA kernels until the CPU/reference path and a 2-GPU PyTorch distributed baseline are correct and measurable.
- No copied implementation from DeepEP, Megatron-Core, vLLM, SGLang, or other production MoE frameworks.
