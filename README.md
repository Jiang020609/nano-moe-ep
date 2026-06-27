# nano-moe-ep

[![CI](https://github.com/Jiang020609/nano-moe-ep/actions/workflows/ci.yml/badge.svg)](https://github.com/Jiang020609/nano-moe-ep/actions/workflows/ci.yml)

`nano-moe-ep` is a correctness-first, from-scratch educational runtime for the Expert Parallel data path in a standalone Mixture-of-Experts FFN block.

## Status

Current status: Stage 3 minimal PyTorch distributed EP baseline on top of the Stage 2 logical dispatch/combine harness. The verified pytest baseline is `python -m pytest -q` with `104 passed`. The distributed dispatch/combine path is validated end-to-end on 2- and 4-process Gloo in CI (asserting bit-for-bit equality with the single-process reference); the NCCL `all_to_all_single` branch is covered by the manual 2-GPU smoke script.

The combine is *sharded* by default: each rank returns only its own source-token rows via the reverse all-to-all, with no extra collective. A legacy `replicate_output=True` mode reproduces the full output with a final `all_reduce` and is kept only for comparison; see [Combine communication](#combine-communication) for the cost difference.

Top-k routing with an expert capacity factor and token dropping is implemented both in the single-process reference (`TopKReferenceMoEFFN`) and in the distributed path (`run_distributed_topk_ep_moe`), the latter validated bit-for-bit against the reference in multi-process Gloo tests; see [Top-k routing and capacity](#top-k-routing-and-capacity).

The repository contains deterministic synthetic top-1 and top-k routing, a top-k reference MoE FFN with capacity and token dropping, explicit metadata including `EPContext`, a grouped/permuted reference MoE FFN, independent token-by-token oracles, a single-process logical-rank dispatch/combine simulation, minimal `torch.distributed` top-1 and top-k forward paths (the latter with capacity dropping), a load-aware expert placement cost model, and communication / capacity / placement cost-model benchmarks.

It does not contain custom CUDA kernels, raw NCCL calls, Triton, backward logic, a learned softmax gate, or wall-clock benchmarks on real interconnects.

## Documentation

- [Architecture](docs/architecture.md)
- [Design decisions](docs/design-decisions.md)
- [Implementation roadmap](docs/implementation-roadmap.md)
- [Test strategy](docs/test-strategy.md)
- [Benchmark report](docs/benchmark-report.md)
- [Comparison to production MoE/EP systems](docs/comparison.md)

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

## Top-k routing and capacity

`TopKReferenceMoEFFN` selects `k` distinct experts per token and supports an
expert capacity factor with token dropping. The per-expert capacity is
`ceil(capacity_factor * num_tokens * k / num_experts)`; assignments beyond an
expert's capacity are dropped in a deterministic row-major (token, then gate
slot) order. A dropped assignment contributes nothing to its token's output; the
token keeps the contributions of its kept experts. The grouped forward and an
independent token-by-token oracle share the exact same drop mask and are
asserted equal.

The skew/capacity trade-off, with a hot expert holding 50% of the slot-0 mass
(`N=4096`, `E=8`, `k=2`), measured by the real drop policy:

| capacity factor | capacity | drop rate | capacity utilization |
|-----------------|----------|-----------|----------------------|
| 1.00            | 1024     | 16.2%     | 83.8%                |
| 1.25            | 1280     | 13.1%     | 69.6%                |
| 1.50            | 1536     | 9.9%      | 60.0%                |
| 2.00            | 2048     | 3.7%      | 48.2%                |

A higher capacity factor lowers drops but raises padding (lower utilization).
The hot expert here carries ~2.3x the mean load, so small capacity factors drop
heavily. Reproduce with:

```bash
python scripts/bench_capacity.py
```

The same top-k routing, capacity, and drop policy run in the distributed path via
`run_distributed_topk_ep_moe`: the capacity mask is computed identically on every
rank from the replicated router output, so distributed dropping matches the
reference exactly. Each token's kept slots originate on, and return to, its owner
rank, where they are summed; multi-process Gloo tests assert bit-for-bit equality
with `TopKReferenceMoEFFN`, including capacity-dropping fixtures.

## Expert placement

The default contiguous placement pins consecutive experts to each rank: it is
balanced in expert *count* but oblivious to *load*, so under skew the hot experts
cluster on one rank and gate the EP step. `balanced_placement` is a capacitated
longest-processing-time heuristic that keeps the same per-rank expert count while
minimizing the max-rank (bottleneck) load, and `rank_load` / `max_rank_load` /
`load_imbalance` score any placement against a per-expert load vector. A
load-aware `ExpertPlacement` plugs straight into the distributed top-k path.

Bottleneck per-rank load on a Zipf-skewed load (`E=16`, total `8192`, exponent 1):

| P | contiguous max | balanced max | contiguous imbalance | balanced imbalance | reduction |
|---|----------------|--------------|----------------------|--------------------|-----------|
| 2 | 6587           | 4242         | 1.61x                | 1.04x              | 1.55x     |
| 4 | 5049           | 2909         | 2.47x                | 1.42x              | 1.74x     |
| 8 | 3635           | 2574         | 3.55x                | 2.51x              | 1.41x     |

Same per-rank expert count, only the assignment changes, so this trades no extra
memory for a 1.4-1.7x lower bottleneck. (At `P=8` the single hottest expert,
load 2423, is a hard floor under equal cardinality, which motivates future expert
splitting/replication.) Reproduce with:

```bash
python scripts/bench_placement.py
```

## Non-Goals

- No full LLM, full Transformer, inference server, scheduler, autoscaler, or training platform.
- No tensor parallelism, pipeline parallelism, data parallelism, speculative decoding, serving stack, or dynamic expert migration in the first project version.
- No custom CUDA kernels until the CPU/reference path, logical dispatch/combine harness, and 2-GPU PyTorch distributed baseline are correct and measurable.
- No copied implementation from DeepEP, Megatron-Core, vLLM, SGLang, or other production MoE frameworks.
