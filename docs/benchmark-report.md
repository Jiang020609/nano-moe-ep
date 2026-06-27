# nano-moe-ep Benchmark Report

This report aggregates the three reproducible experiments in `scripts/`. All
numbers are **deterministic** and **hardware-independent**: they come from the
real planning/routing/placement code combined with standard cost models
(ring-all-reduce and all-to-all communication volume, the GShard-style capacity
formula, and per-rank load). They measure *communication volume* and *load*, not
wall-clock time; wall-clock on real interconnects is deferred to the GPU stages.

Numerical correctness behind every number is covered by the test suite
(`python -m pytest -q`, currently `104 passed`), including multi-process Gloo
end-to-end tests that match the distributed paths to the single-process
reference bit-for-bit.

## 1. Combine communication: sharded vs. replicated

`scripts/bench_combine.py` — the combine refactor replaced a full-output
`all_reduce` with a sharded reverse-all-to-all. Per-rank bottleneck traffic at
`N=4096`, `H=4096`, bf16:

| scenario   | P | sharded (MiB) | replicated (MiB) | reduction |
|------------|---|---------------|------------------|-----------|
| balanced   | 2 | 16.0          | 48.0             | 3.0x      |
| balanced   | 4 | 8.0           | 56.0             | 7.0x      |
| balanced   | 8 | 4.0           | 60.0             | 15.0x     |
| all-to-one | 2 | 32.0          | 64.0             | 2.0x      |
| all-to-one | 4 | 32.0          | 80.0             | 2.5x      |
| all-to-one | 8 | 32.0          | 88.0             | 2.8x      |

The dropped `all_reduce` term is `2 * (P - 1) / P * N * H * dtype` per rank: it
grows with `P` and is independent of routing skew, so it dominates as the
cluster scales (15x less combine traffic at `P=8`, balanced).

## 2. Routing skew vs. capacity

`scripts/bench_capacity.py` — the top-k capacity/drop trade-off with a hot
expert holding 50% of the slot-0 mass (`N=4096`, `E=8`, `k=2`). The hot expert
carries 2.29x the mean load:

| capacity factor | capacity | drop rate | capacity utilization |
|-----------------|----------|-----------|----------------------|
| 1.00            | 1024     | 16.2%     | 83.8%                |
| 1.25            | 1280     | 13.1%     | 69.6%                |
| 1.50            | 1536     | 9.9%      | 60.0%                |
| 2.00            | 2048     | 3.7%      | 48.2%                |

Raising the capacity factor lowers drops but raises padding (lower utilization).
The right operating point depends on the routing skew — there is no free lunch.

## 3. Load-aware expert placement

`scripts/bench_placement.py` — contiguous vs. capacitated-LPT `balanced_placement`
on a Zipf-skewed load (`E=16`, total `8192`, exponent 1). Both keep the same
per-rank expert count, so this trades no memory for a lower bottleneck:

| P | contiguous max | balanced max | contiguous imbalance | balanced imbalance | reduction |
|---|----------------|--------------|----------------------|--------------------|-----------|
| 2 | 6587           | 4242         | 1.61x                | 1.04x              | 1.55x     |
| 4 | 5049           | 2909         | 2.47x                | 1.42x              | 1.74x     |
| 8 | 3635           | 2574         | 3.55x                | 2.51x              | 1.41x     |

Per-rank load at `P=8`:

- contiguous: `[3635, 1414, 889, 649, 511, 422, 359, 313]`
- balanced:   `[2574, 1374, 981, 792, 687, 624, 588, 572]`

The single hottest expert (load 2423) is a hard floor under equal cardinality —
placement alone cannot split it, which motivates expert splitting/replication as
future work.

## Reproduce

```bash
python scripts/bench_combine.py
python scripts/bench_capacity.py
python scripts/bench_placement.py
```

## What these do and do not show

- **Do**: exact communication volume, drop rates, and per-rank load from the real
  code paths, under controlled skew, with closed-form collective cost models.
- **Do not**: wall-clock latency/throughput on real GPUs or interconnects, kernel
  efficiency, or overlap. Those require the GPU stages and are explicitly out of
  scope for the current CPU-validated baseline.
