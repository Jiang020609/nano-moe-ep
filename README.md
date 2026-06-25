# nano-moe-ep

[![CI](https://github.com/Jiang020609/nano-moe-ep/actions/workflows/ci.yml/badge.svg)](https://github.com/Jiang020609/nano-moe-ep/actions/workflows/ci.yml)

`nano-moe-ep` is a correctness-first, from-scratch educational runtime for the Expert Parallel data path in a standalone Mixture-of-Experts FFN block.

## Status

Current status: Stage 3 minimal PyTorch distributed EP baseline on top of the Stage 2 logical dispatch/combine harness. The verified pytest baseline is `python -m pytest -q` with `74 passed`. The distributed dispatch/combine path is validated end-to-end on 2- and 4-process Gloo in CI (asserting bit-for-bit equality with the single-process reference); the NCCL `all_to_all_single` branch is covered by the manual 2-GPU smoke script.

The combine is *sharded* by default: each rank returns only its own source-token rows via the reverse all-to-all, with no extra collective. A legacy `replicate_output=True` mode reproduces the full output with a final `all_reduce` and is kept only for comparison; see [Combine communication](#combine-communication) for the cost difference.

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

## Combine communication

The original combine assembled the full output with an `all_reduce`, which moves
`2 * (P - 1) / P * num_tokens * hidden` elements per rank regardless of routing
skew. The default sharded combine drops that collective entirely and moves only
each rank's own returned outputs. The dispatch counts below come from the real
planner (`build_distributed_payload_plan`); the collective byte counts use the
standard ring-all-reduce / all-to-all cost models.

Per-rank combine traffic at the bottleneck rank (`N=4096`, `H=4096`, bf16):

| scenario   | P | sharded (MiB) | replicated (MiB) | reduction |
|------------|---|---------------|------------------|-----------|
| balanced   | 2 | 16.0          | 48.0             | 3.0x      |
| balanced   | 4 | 8.0           | 56.0             | 7.0x      |
| balanced   | 8 | 4.0           | 60.0             | 15.0x     |
| all-to-one | 2 | 32.0          | 64.0             | 2.0x      |
| all-to-one | 4 | 32.0          | 80.0             | 2.5x      |
| all-to-one | 8 | 32.0          | 88.0             | 2.8x      |

The `all_reduce` term grows with `P` and is independent of skew, so it dominates
as the cluster scales: at `P=8` balanced routing the sharded combine moves ~15x
less. Reproduce with:

```bash
python scripts/bench_combine.py
```

This reports communication volume only; wall-clock on real interconnects is
deferred to the GPU stages. Numerical equivalence of both combine paths is
covered by `tests/test_distributed_ep_e2e.py`.

## Non-Goals

- No full LLM, full Transformer, inference server, scheduler, autoscaler, or training platform.
- No tensor parallelism, pipeline parallelism, data parallelism, speculative decoding, serving stack, or dynamic expert migration in the first project version.
- No custom CUDA kernels until the CPU/reference path, logical dispatch/combine harness, and 2-GPU PyTorch distributed baseline are correct and measurable.
- No copied implementation from DeepEP, Megatron-Core, vLLM, SGLang, or other production MoE frameworks.
