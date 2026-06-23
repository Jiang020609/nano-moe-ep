# nano-moe-ep

`nano-moe-ep` is a correctness-first, from-scratch learning project for the essential Expert Parallel data path in a standalone Mixture-of-Experts FFN block.

## Status

Current status: Stage 1 single-process CPU reference implementation. The repository contains deterministic synthetic top-1 routing, a forward-only reference MoE FFN, explicit trace metadata, and CPU tests.

It does not yet contain distributed Expert Parallel execution, CUDA, NCCL, multiprocessing, custom kernels, or benchmarking.

## Quick Start

```bash
python -m pytest -q
```

## Milestones

- [Architecture and invariants](docs/architecture.md)
- [Design decisions](docs/design-decisions.md)
- [Implementation roadmap](docs/implementation-roadmap.md)
- [Test strategy](docs/test-strategy.md)

## Non-Goals

- No full LLM, full Transformer, inference server, scheduler, autoscaler, or training platform.
- No tensor parallelism, pipeline parallelism, data parallelism, speculative decoding, serving stack, or dynamic expert migration in the first project version.
- No custom CUDA kernels until the CPU/reference path and a 2-GPU PyTorch distributed baseline are correct and measurable.
- No copied implementation from DeepEP, Megatron-Core, vLLM, SGLang, or other production MoE frameworks.
