# How nano-moe-ep Relates to Production MoE/EP Systems

`nano-moe-ep` is a correctness-first reference, not a competitor to production
frameworks. The goal of this page is to place its design next to DeepEP,
Megatron-Core MoE, and Tutel so the shared **semantics** and the deliberately
omitted **performance engineering** are explicit.

> Caveat: production frameworks evolve quickly. The statements below describe
> stable, architecture-level behavior to the best of current knowledge; verify
> specifics against each project's official documentation before quoting them.

## Side-by-side

| Dimension | nano-moe-ep | DeepEP (DeepSeek) | Megatron-Core MoE | Tutel |
|-----------|-------------|-------------------|-------------------|-------|
| Primary purpose | Educational, correctness-first EP data path | Production EP communication library | Production training framework MoE layer | Optimized MoE layer/library |
| Dispatch/combine | Variable-size token all-to-all; reverse all-to-all combine; sharded output | GPU all-to-all dispatch/combine kernels | Token permutation + all-to-all (or all-gather) dispatcher | Optimized all-to-all (incl. hierarchical/2D) |
| Comm primitive | `all_to_all_single` (NCCL) / all-gather (Gloo); counts via all-gather | NVLink intranode + RDMA internode; normal + low-latency kernels | NCCL collectives across EP/TP/DP/PP groups | Tuned all-to-all kernels |
| Top-k routing | Yes (distinct experts, weighted sum) | Yes (built around group-limited gating) | Yes (configurable k) | Yes |
| Capacity / token drop | Yes; GShard-style capacity, deterministic drop; also dropless (`capacity_factor=None`) | Dropless-oriented (model uses aux-loss-free balancing) | Both token-drop and dropless modes | Adaptive/dynamic capacity |
| Expert placement | Static table + load-aware capacitated-LPT balanced placement | EP across GPUs; relies on model load balancing | EP groups; balancing via aux loss | Flexible parallelism, runtime expert switching |
| Expert compute | Per-expert `index_select` + `nn.Linear` loop | External (grouped GEMM) | GroupedGEMM / TransformerEngine | Batched expert GEMM |
| Compute–comm overlap | None (correctness baseline) | Yes (overlap hooks, background RDMA) | Yes (comm overlap features) | Yes (all-to-all overlapped with compute) |
| Precision | fp32 reference (bf16 only in cost models) | FP8 dispatch supported | bf16/fp16, FP8 via TransformerEngine | fp16/bf16 |
| Kernels | Pure PyTorch | Custom CUDA (PTX-tuned) | TransformerEngine + GroupedGEMM | Custom CUDA |
| Backward / training | Forward only | Transport supports training | Full training | Full training |

## What nano-moe-ep reproduces faithfully

- The **token-level all-to-all dispatch/combine** structure that defines expert
  parallelism, including variable-size per-rank payloads and a count-exchange
  protocol — the same shape these frameworks use.
- **Top-k routing** with multi-slot weighted reduction.
- **Capacity factor and token dropping** with a deterministic, documented policy,
  matching the GShard/Switch capacity formula, plus a dropless mode.
- **Load-aware expert placement** scored by a max-rank-load cost model — the same
  bottleneck these systems care about under routing skew.
- A **sharded combine** that avoids a redundant full-output collective, which is
  how real EP keeps the output partitioned across ranks.

## What it deliberately omits (and where each framework provides it)

- **Custom GPU kernels / fused all-to-all** (DeepEP, Tutel) — this project uses
  stock `torch.distributed` collectives for clarity.
- **GroupedGEMM for expert compute** (Megatron-Core) — this project runs a simple
  per-expert loop; it measures communication and load, not GEMM efficiency.
- **Compute–communication overlap** (all three) — the baseline is sequential.
- **FP8 / low-latency decode kernels** (DeepEP) and **multi-axis parallelism**
  (Megatron-Core: EP×TP×DP×PP) — out of scope for a single-layer reference.
- **Backward / end-to-end training** — this project is forward-only.

## Takeaway

The value of this project is that the EP **semantics** — dispatch/combine,
top-k, capacity/drop, placement, sharded combine — are implemented from scratch,
validated against independent oracles, and measured with reproducible cost-model
benchmarks. The production frameworks exist to make those same semantics fast on
real hardware; this project exists to make them *legible and correct* first.
