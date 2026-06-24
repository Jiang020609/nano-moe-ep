"""Reusable multi-process harness for end-to-end distributed EP tests.

This module spawns real ``torch.distributed`` (Gloo) processes on CPU so the
Stage 3 distributed dispatch/combine path can be validated end-to-end without a
GPU. Both the harness and the generic worker live here (rather than in a test
module) so that ``mp.spawn`` can re-import them by a stable top-level module
name on every platform, including Windows ``spawn`` start semantics.

Coverage note: the Gloo backend exercises the ``_all_gather_variable_tensors``
fallback, not the NCCL ``all_to_all_single`` branch. The orchestration, count
exchange, reverse exchange, expert execution, and combine are all covered here;
the NCCL collective itself is only covered by the manual 2-GPU smoke script.
"""

from __future__ import annotations

import os
import socket
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent / "src"

# Make both this harness module and the `nano_moe_ep` package importable inside
# freshly spawned child interpreters (which do not inherit pytest's sys.path).
import sys

for _path in (str(_THIS_DIR), str(_SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nano_moe_ep.distributed_ep import DistributedEPConfig, run_distributed_ep_moe
from nano_moe_ep.dispatch_combine import run_logical_ep_moe
from nano_moe_ep.reference import ReferenceMoEFFN
from nano_moe_ep.types import ExpertPlacement, RouterOutput


def gloo_available() -> bool:
    return dist.is_available() and dist.is_gloo_available()


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _worker_entry(
    rank: int,
    world_size: int,
    port: int,
    out_dir: str,
    worker_fn: Callable[..., object],
    args: tuple,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        result = worker_fn(rank, world_size, *args)
        torch.save(result, str(Path(out_dir) / f"rank_{rank}.pt"))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_in_processes(
    world_size: int,
    worker_fn: Callable[..., object],
    args: tuple = (),
) -> list[object]:
    """Spawn ``world_size`` Gloo processes, run ``worker_fn`` on each, collect results.

    ``worker_fn`` must be a top-level (picklable) function with signature
    ``worker_fn(rank, world_size, *args)`` and return a picklable result.
    """

    if world_size < 1:
        raise ValueError("world_size must be positive")

    # Propagate import paths to spawned children via the environment so that the
    # pickled `worker_fn` reference resolves before our own code runs.
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(_THIS_DIR), str(_SRC_DIR)]
    if existing:
        parts.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)

    port = _free_port()
    out_dir = tempfile.mkdtemp(prefix="nano_moe_ep_dist_")
    try:
        mp.spawn(
            _worker_entry,
            args=(world_size, port, out_dir, worker_fn, args),
            nprocs=world_size,
            join=True,
        )
        return [
            torch.load(str(Path(out_dir) / f"rank_{rank}.pt"), weights_only=False)
            for rank in range(world_size)
        ]
    finally:
        for child in Path(out_dir).glob("rank_*.pt"):
            child.unlink(missing_ok=True)
        try:
            Path(out_dir).rmdir()
        except OSError:
            pass


def ep_worker(
    rank: int,
    world_size: int,
    inputs: torch.Tensor,
    assignments: Sequence[int],
    owner_by_expert: Sequence[int],
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    weights: Sequence[float] | None,
    seed: int,
) -> dict:
    """Run the distributed EP path and both references inside one rank.

    Expert weights are rebuilt from a shared ``seed`` so every rank holds
    identical expert parameters; this is required because the distributed
    all-reduce combine sums each rank's locally computed expert outputs, which
    must agree with the single-process reference that runs every expert.
    """

    torch.manual_seed(seed)
    model = ReferenceMoEFFN(hidden_dim=hidden_dim, ffn_dim=ffn_dim, num_experts=num_experts)
    placement = ExpertPlacement(owner_rank_by_expert=tuple(owner_by_expert), num_ep_ranks=world_size)

    expert_indices = torch.tensor(list(assignments), dtype=torch.long).reshape(-1, 1)
    if weights is None:
        weight_tensor = torch.ones((expert_indices.shape[0], 1), dtype=torch.float32)
    else:
        weight_tensor = torch.tensor(list(weights), dtype=torch.float32).reshape(-1, 1)
    router_output = RouterOutput(expert_indices=expert_indices, weights=weight_tensor)

    config = DistributedEPConfig(backend="gloo", world_size=world_size, rank=rank, device="cpu")

    with torch.no_grad():
        dist_output, trace = run_distributed_ep_moe(
            inputs, router_output, model.experts, placement, config=config
        )
        logical_output, _ = run_logical_ep_moe(inputs, router_output, model.experts, placement)
        reference_output, _ = model(inputs, router_output)

    return {
        "rank": rank,
        "dist": dist_output,
        "logical": logical_output,
        "reference": reference_output,
        "owned_experts": trace.owned_experts,
        "send_counts": trace.send_plan.send_counts_by_rank.cpu(),
        "recv_counts": trace.dispatch_counts.recv_counts_by_rank.cpu(),
    }
