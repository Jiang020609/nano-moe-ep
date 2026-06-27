# GPU Benchmarking

This project separates correctness smokes from timing benchmarks. Run
`scripts/run_nccl_ep_smoke.py` first; it checks top-1, top-k, and top-k with
capacity dropping against the single-process references. Then use
`scripts/bench_nccl_ep.py` to collect real-rank latency numbers.

## Timing Command

Example 8-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/bench_nccl_ep.py \
  --num-tokens 4096 \
  --hidden-dim 1024 \
  --ffn-dim 4096 \
  --warmup 5 \
  --iters 20 \
  --output-dir docs/benchmarks
```

The script writes Markdown and CSV files from rank 0 when `--output-dir` is
provided. Each reported latency is the maximum elapsed time across ranks for a
synchronized iteration, which approximates the bottleneck rank for the EP step.

For 2- and 4-GPU runs, change only `CUDA_VISIBLE_DEVICES` and
`--nproc_per_node`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/bench_nccl_ep.py --output-dir docs/benchmarks
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/bench_nccl_ep.py --output-dir docs/benchmarks
```

## Reported Cases

The benchmark reports:

- `top-1`: distributed top-1 EP with contiguous placement.
- `top-k`: distributed top-k EP with load-aware balanced placement.
- `top-k+cap`: distributed top-k EP with balanced placement and capacity dropping.
- `sharded`: default reverse all-to-all combine.
- `replicated`: legacy full-output `all_reduce` combine for comparison.

## Current Timing Results

Timing results are not checked in yet. The correctness baseline has passed the
manual 2-, 4-, and 8-GPU NCCL smoke tests; the next step is to run this timing
benchmark on the same GPU host and commit the generated Markdown/CSV outputs.
