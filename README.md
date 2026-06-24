# nano-moe-ep

[![CI](https://github.com/Jiang020609/nano-moe-ep/actions/workflows/ci.yml/badge.svg)](https://github.com/Jiang020609/nano-moe-ep/actions/workflows/ci.yml)

`nano-moe-ep` is a correctness-first, from-scratch educational runtime for the Expert Parallel data path in a standalone Mixture-of-Experts FFN block.

## Status

Current status: Stage 3 minimal PyTorch distributed EP baseline on top of the Stage 2 logical dispatch/combine harness. The verified pytest baseline is `python -m pytest -q` with `71 passed`. The distributed dispatch/combine path is validated end-to-end on 2- and 4-process Gloo in CI (asserting bit-for-bit equality with the single-process reference); the NCCL `all_to_all_single` branch is covered by the manual 2-GPU smoke script.

The repository contains deterministic synthetic top-1 routing, explicit metadata including `EPContext`, a grouped/permuted reference MoE FFN, an independent token-by-token oracle, a single-process logical-rank dispatch/combine simulation, and a minimal `torch.distributed` Stage 3 forward path.

It does not contain custom CUDA kernels, raw NCCL calls, Triton, top-2 routing, capacity factor, token dropping, backward logic, or benchmarking.

## Documentation

- [Architecture](docs/architecture.md)
- [Design decisions](docs/design-decisions.md)
- [Implementation roadmap](docs/implementation-roadmap.md)
- [Test strategy](docs/test-strategy.md)

## Milestones

- Stage 0 - Architecture and invariants
- Stage 1 - Single-process reference MoE FFN
- Stage 2 - Deterministic dispatch/combine harness
- Stage 3 - 2-GPU EP baseline
- Stage 4 - 4-GPU scaling and skew experiments
- Stage 5 - GPU-side packing/permutation optimization
- Stage 6 - Communication-computation overlap
- Stage 7 - Final benchmark report

## Quick Start

```bash
python -m pytest -q
```

Manual Stage 3 smoke:

```bash
torchrun --standalone --nproc_per_node=2 scripts/run_stage3_2gpu_smoke.py
```

The smoke uses NCCL only when CUDA/NCCL and two devices are available; otherwise it falls back to a CPU/Gloo correctness smoke. It prints per-rank owned experts, send/receive counts, maximum absolute error versus the local reference, and PASS/FAIL.

## Non-Goals

- No full LLM, full Transformer, inference server, scheduler, autoscaler, or training platform.
- No tensor parallelism, pipeline parallelism, data parallelism, speculative decoding, serving stack, or dynamic expert migration in the first project version.
- No custom CUDA kernels until the CPU/reference path, logical dispatch/combine harness, and 2-GPU PyTorch distributed baseline are correct and measurable.
- No copied implementation from DeepEP, Megatron-Core, vLLM, SGLang, or other production MoE frameworks.
