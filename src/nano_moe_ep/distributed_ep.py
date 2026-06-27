from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from nano_moe_ep.dispatch_combine import build_logical_ep_layout, build_token_assignments
from nano_moe_ep.routing.capacity import build_capacity_mask, compute_expert_capacity
from nano_moe_ep.types import (
    EPContext,
    ExecutionMode,
    ExpertPlacement,
    RouterOutput,
    TokenLayout,
    TopKRouterOutput,
)


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    if tensor.dtype is torch.bool:
        return False
    try:
        torch.iinfo(tensor.dtype)
    except TypeError:
        return False
    return True


def _validate_rank_world(rank: int, world_size: int) -> None:
    if not isinstance(world_size, int) or isinstance(world_size, bool):
        raise ValueError("world_size must be an integer")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise ValueError("rank must be an integer")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")


def _prefix_offsets(counts: torch.Tensor) -> torch.Tensor:
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D tensor")
    if not _is_integer_tensor(counts):
        raise ValueError("counts must use an integer dtype")
    if (counts < 0).any().item():
        raise ValueError("counts must be non-negative")
    offsets = torch.empty(counts.numel() + 1, dtype=torch.long, device=counts.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts.to(dtype=torch.long), dim=0)
    return offsets


def source_token_indices(num_tokens: int, *, rank: int, world_size: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Return the deterministic source-token shard for one EP rank."""

    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
        raise ValueError("num_tokens must be an integer")
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    _validate_rank_world(rank, world_size)
    token_indices = torch.arange(num_tokens, dtype=torch.long, device=device)
    return token_indices[token_indices.remainder(world_size) == rank]


@dataclass(frozen=True)
class DistributedEPConfig:
    """Runtime metadata for a launched Stage 3 distributed EP process."""

    backend: str
    world_size: int
    rank: int
    device: torch.device | str = "cpu"
    deterministic_seed: int = 2029

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend == "":
            raise ValueError("backend must be a non-empty string")
        backend = self.backend.lower()
        _validate_rank_world(self.rank, self.world_size)
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("device must be a valid torch device") from exc
        if backend == "nccl" and device.type != "cuda":
            raise ValueError("backend='nccl' requires a CUDA device")
        if backend == "gloo" and device.type != "cpu":
            raise ValueError("backend='gloo' is CPU-only in Stage 3")
        if not isinstance(self.deterministic_seed, int) or isinstance(self.deterministic_seed, bool):
            raise ValueError("deterministic_seed must be an integer")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "device", device)

    @classmethod
    def from_process_group(
        cls,
        *,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
        deterministic_seed: int = 2029,
    ) -> "DistributedEPConfig":
        if not dist.is_available() or not dist.is_initialized():
            raise ValueError("torch.distributed must be initialized")
        backend = str(dist.get_backend(group=group)).lower()
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        if device is None:
            device = torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu")
        return cls(
            backend=backend,
            world_size=world_size,
            rank=rank,
            device=device,
            deterministic_seed=deterministic_seed,
        )

    def to_ep_context(self) -> EPContext:
        return EPContext(
            num_ep_ranks=self.world_size,
            local_rank=self.rank,
            execution_mode=ExecutionMode.DISTRIBUTED,
            device=str(self.device),
            deterministic=True,
            phase="distributed_forward",
        )

    @property
    def uses_all_to_all_single(self) -> bool:
        return self.backend != "gloo"


@dataclass(frozen=True)
class DistributedPayloadPlan:
    """Local source-rank payload metadata before distributed dispatch."""

    send_counts_by_rank: torch.Tensor
    send_offsets: torch.Tensor
    token_indices: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    token_layout: TokenLayout

    def __post_init__(self) -> None:
        if self.send_counts_by_rank.ndim != 1:
            raise ValueError("send_counts_by_rank must be a 1D tensor")
        if not _is_integer_tensor(self.send_counts_by_rank):
            raise ValueError("send_counts_by_rank must use an integer dtype")
        if (self.send_counts_by_rank < 0).any().item():
            raise ValueError("send_counts_by_rank must be non-negative")
        if self.send_offsets.shape != (self.send_counts_by_rank.numel() + 1,):
            raise ValueError("send_offsets must have shape [world_size + 1]")
        if self.send_offsets[0].item() != 0:
            raise ValueError("send_offsets must start at 0")
        if not torch.equal(self.send_offsets[1:] - self.send_offsets[:-1], self.send_counts_by_rank):
            raise ValueError("send_offsets differences must match send_counts_by_rank")
        if self.token_indices.ndim != 1 or self.expert_ids.ndim != 1:
            raise ValueError("token_indices and expert_ids must be 1D tensors")
        if not _is_integer_tensor(self.token_indices) or not _is_integer_tensor(self.expert_ids):
            raise ValueError("token_indices and expert_ids must use integer dtypes")
        if self.routing_weights.shape != (self.token_indices.numel(), 1):
            raise ValueError("routing_weights must have shape [num_local_tokens, 1]")
        if not self.routing_weights.dtype.is_floating_point:
            raise ValueError("routing_weights must use a floating dtype")
        if not torch.isfinite(self.routing_weights).all().item():
            raise ValueError("routing_weights must be finite")
        if self.expert_ids.numel() != self.token_indices.numel():
            raise ValueError("expert_ids must match token_indices length")
        if self.send_offsets[-1].item() != self.token_indices.numel():
            raise ValueError("send_offsets must end at local token count")


@dataclass(frozen=True)
class CountExchange:
    """Pairwise variable-size count metadata after a distributed count exchange."""

    send_counts_by_rank: torch.Tensor
    recv_counts_by_rank: torch.Tensor
    count_matrix: torch.Tensor

    def __post_init__(self) -> None:
        if self.send_counts_by_rank.ndim != 1 or self.recv_counts_by_rank.ndim != 1:
            raise ValueError("send and recv counts must be 1D tensors")
        if self.count_matrix.ndim != 2:
            raise ValueError("count_matrix must be a 2D tensor")
        world_size = self.send_counts_by_rank.numel()
        if self.recv_counts_by_rank.numel() != world_size:
            raise ValueError("recv_counts_by_rank must match world_size")
        if self.count_matrix.shape != (world_size, world_size):
            raise ValueError("count_matrix must have shape [world_size, world_size]")
        for name, tensor in (
            ("send_counts_by_rank", self.send_counts_by_rank),
            ("recv_counts_by_rank", self.recv_counts_by_rank),
            ("count_matrix", self.count_matrix),
        ):
            if not _is_integer_tensor(tensor):
                raise ValueError(f"{name} must use an integer dtype")
            if (tensor < 0).any().item():
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class DistributedEPTrace:
    """Rank-local trace for the Stage 3 distributed EP forward path."""

    config: DistributedEPConfig
    ep_context: EPContext
    token_layout: TokenLayout
    send_plan: DistributedPayloadPlan
    dispatch_counts: CountExchange
    return_counts: CountExchange
    owned_experts: tuple[int, ...]
    returned_token_indices: torch.Tensor
    output_token_indices: torch.Tensor
    replicate_output: bool


def build_distributed_payload_plan(
    router_output: RouterOutput,
    expert_placement: ExpertPlacement,
    *,
    source_rank: int,
    world_size: int,
    device: torch.device | str | None = None,
) -> DistributedPayloadPlan:
    """Build the local source-rank payload for the Stage 3 distributed path."""

    _validate_rank_world(source_rank, world_size)
    if expert_placement.num_ep_ranks != world_size:
        raise ValueError("world_size must match ExpertPlacement num_ep_ranks")
    target_device = torch.device(device) if device is not None else router_output.expert_indices.device
    assignments = build_token_assignments(router_output, expert_placement)
    token_layout = build_logical_ep_layout(
        assignments,
        num_tokens=router_output.expert_indices.shape[0],
        num_experts=expert_placement.num_experts,
        num_ep_ranks=world_size,
    )
    local_assignments = [
        assignment
        for assignment in assignments
        if assignment.token_index % world_size == source_rank
    ]
    sorted_assignments = sorted(
        local_assignments,
        key=lambda assignment: (assignment.ep_rank, assignment.expert_id, assignment.original_position),
    )
    send_counts = torch.zeros(world_size, dtype=torch.long, device=target_device)
    for assignment in sorted_assignments:
        send_counts[assignment.ep_rank] += 1
    send_offsets = _prefix_offsets(send_counts)

    token_indices = torch.tensor(
        [assignment.token_index for assignment in sorted_assignments],
        dtype=torch.long,
        device=target_device,
    )
    expert_ids = torch.tensor(
        [assignment.expert_id for assignment in sorted_assignments],
        dtype=torch.long,
        device=target_device,
    )
    router_weights = router_output.weights.to(device=target_device)
    if token_indices.numel() == 0:
        routing_weights = router_weights.new_empty((0, 1))
    else:
        routing_weights = router_weights.index_select(0, token_indices)
    return DistributedPayloadPlan(
        send_counts_by_rank=send_counts,
        send_offsets=send_offsets,
        token_indices=token_indices,
        expert_ids=expert_ids,
        routing_weights=routing_weights,
        token_layout=token_layout,
    )


def exchange_counts(send_counts_by_rank: torch.Tensor, *, group: dist.ProcessGroup | None = None) -> CountExchange:
    """Exchange one send-count vector per rank using a real distributed collective."""

    if not dist.is_available() or not dist.is_initialized():
        raise ValueError("torch.distributed must be initialized")
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if send_counts_by_rank.shape != (world_size,):
        raise ValueError("send_counts_by_rank must have shape [world_size]")
    if not _is_integer_tensor(send_counts_by_rank):
        raise ValueError("send_counts_by_rank must use an integer dtype")
    if (send_counts_by_rank < 0).any().item():
        raise ValueError("send_counts_by_rank must be non-negative")

    send_counts = send_counts_by_rank.to(dtype=torch.long).contiguous()
    gathered = [torch.empty_like(send_counts) for _ in range(world_size)]
    dist.all_gather(gathered, send_counts, group=group)
    count_matrix = torch.stack(gathered, dim=0)
    return CountExchange(
        send_counts_by_rank=send_counts,
        recv_counts_by_rank=count_matrix[:, rank].contiguous(),
        count_matrix=count_matrix,
    )


def reverse_count_exchange(count_exchange: CountExchange, *, rank: int) -> CountExchange:
    """Invert source/destination roles for the return combine exchange."""

    world_size = count_exchange.send_counts_by_rank.numel()
    _validate_rank_world(rank, world_size)
    reversed_matrix = count_exchange.count_matrix.transpose(0, 1).contiguous()
    return CountExchange(
        send_counts_by_rank=reversed_matrix[rank].contiguous(),
        recv_counts_by_rank=reversed_matrix[:, rank].contiguous(),
        count_matrix=reversed_matrix,
    )


def all_to_all_variable_tensors(
    send_buffer: torch.Tensor,
    count_exchange: CountExchange,
    *,
    config: DistributedEPConfig,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Exchange a first-dimension variable-size tensor according to count metadata."""

    if send_buffer.ndim == 0:
        raise ValueError("send_buffer must have at least one dimension")
    if send_buffer.shape[0] != int(count_exchange.send_counts_by_rank.sum().item()):
        raise ValueError("send_buffer first dimension must equal total send count")
    if config.uses_all_to_all_single:
        output = send_buffer.new_empty(
            (int(count_exchange.recv_counts_by_rank.sum().item()), *send_buffer.shape[1:])
        )
        dist.all_to_all_single(
            output,
            send_buffer.contiguous(),
            output_split_sizes=[int(count.item()) for count in count_exchange.recv_counts_by_rank],
            input_split_sizes=[int(count.item()) for count in count_exchange.send_counts_by_rank],
            group=group,
        )
        return output
    return _all_gather_variable_tensors(send_buffer, count_exchange, rank=config.rank, group=group)


def _all_gather_variable_tensors(
    send_buffer: torch.Tensor,
    count_exchange: CountExchange,
    *,
    rank: int,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """CPU/Gloo fallback for variable all-to-all using one padded all-gather."""

    world_size = count_exchange.send_counts_by_rank.numel()
    _validate_rank_world(rank, world_size)
    total_send_by_rank = count_exchange.count_matrix.sum(dim=1)
    max_total_send = int(total_send_by_rank.max().item()) if total_send_by_rank.numel() > 0 else 0
    output_shape = (int(count_exchange.recv_counts_by_rank.sum().item()), *send_buffer.shape[1:])
    if max_total_send == 0:
        return send_buffer.new_empty(output_shape)

    padded = send_buffer.new_zeros((max_total_send, *send_buffer.shape[1:]))
    if send_buffer.shape[0] > 0:
        padded[: send_buffer.shape[0]] = send_buffer
    gathered = [send_buffer.new_empty(padded.shape) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)

    received_segments: list[torch.Tensor] = []
    for source_rank in range(world_size):
        source_counts = count_exchange.count_matrix[source_rank]
        start = int(source_counts[:rank].sum().item())
        count = int(source_counts[rank].item())
        if count > 0:
            received_segments.append(gathered[source_rank][start : start + count])
    if not received_segments:
        return send_buffer.new_empty(output_shape)
    return torch.cat(received_segments, dim=0)


def execute_received_experts(
    received_tokens: torch.Tensor,
    received_expert_ids: torch.Tensor,
    experts: Sequence[nn.Module],
    expert_placement: ExpertPlacement,
    *,
    rank: int,
) -> torch.Tensor:
    """Run only the experts owned by this rank on received token rows."""

    if received_tokens.ndim != 2:
        raise ValueError("received_tokens must have shape [num_tokens, hidden_dim]")
    if received_expert_ids.shape != (received_tokens.shape[0],):
        raise ValueError("received_expert_ids must have one entry per received token")
    if len(experts) != expert_placement.num_experts:
        raise ValueError("experts length must match expert placement num_experts")
    owned_experts = expert_placement.experts_for_rank(rank)
    owned_set = set(owned_experts)
    unique_experts = [int(expert_id.item()) for expert_id in torch.unique(received_expert_ids.cpu())]
    invalid_experts = [expert_id for expert_id in unique_experts if expert_id not in owned_set]
    if invalid_experts:
        raise ValueError("received expert ids must belong to the local rank")

    local_outputs = torch.empty_like(received_tokens)
    for expert_id in owned_experts:
        positions = torch.nonzero(received_expert_ids == expert_id, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        expert_inputs = received_tokens.index_select(0, positions.to(device=received_tokens.device))
        expert_outputs = experts[expert_id](expert_inputs)
        local_outputs.index_copy_(0, positions.to(device=received_tokens.device), expert_outputs)
    return local_outputs


def apply_partial_combine(
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    num_tokens: int,
) -> torch.Tensor:
    """Restore this rank's source-token outputs into a full-size zero-padded tensor."""

    if expert_outputs.ndim != 2:
        raise ValueError("expert_outputs must have shape [num_tokens, hidden_dim]")
    if token_indices.shape != (expert_outputs.shape[0],):
        raise ValueError("token_indices must have one entry per expert output")
    if not _is_integer_tensor(token_indices):
        raise ValueError("token_indices must use an integer dtype")
    if routing_weights.shape != (expert_outputs.shape[0], 1):
        raise ValueError("routing_weights must have shape [num_outputs, 1]")
    if not routing_weights.dtype.is_floating_point:
        raise ValueError("routing_weights must use a floating dtype")
    if not torch.isfinite(routing_weights).all().item():
        raise ValueError("routing_weights must be finite")
    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
        raise ValueError("num_tokens must be an integer")
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if token_indices.numel() > 0:
        if (token_indices < 0).any().item() or (token_indices >= num_tokens).any().item():
            raise ValueError("token_indices must be in [0, num_tokens)")
        if torch.unique(token_indices).numel() != token_indices.numel():
            raise ValueError("token_indices must not contain duplicates")

    output = torch.zeros(
        (num_tokens, expert_outputs.shape[1]),
        dtype=expert_outputs.dtype,
        device=expert_outputs.device,
    )
    weighted_outputs = expert_outputs * routing_weights.to(device=expert_outputs.device, dtype=expert_outputs.dtype)
    output.index_copy_(0, token_indices.to(device=expert_outputs.device), weighted_outputs)
    return output


def apply_sharded_combine(
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply routing weights and return only this rank's source-token rows.

    Unlike :func:`apply_partial_combine`, this performs no zero-padding to the
    global token count and no cross-rank collective: the reverse all-to-all has
    already delivered every source token's expert output back to its owning
    rank. Rows are returned in ascending source-token-index order so the shard
    has a deterministic, reconstructable layout.
    """

    if expert_outputs.ndim != 2:
        raise ValueError("expert_outputs must have shape [num_tokens, hidden_dim]")
    if token_indices.shape != (expert_outputs.shape[0],):
        raise ValueError("token_indices must have one entry per expert output")
    if not _is_integer_tensor(token_indices):
        raise ValueError("token_indices must use an integer dtype")
    if routing_weights.shape != (expert_outputs.shape[0], 1):
        raise ValueError("routing_weights must have shape [num_outputs, 1]")
    if not routing_weights.dtype.is_floating_point:
        raise ValueError("routing_weights must use a floating dtype")
    if not torch.isfinite(routing_weights).all().item():
        raise ValueError("routing_weights must be finite")
    if token_indices.numel() > 0:
        if (token_indices < 0).any().item():
            raise ValueError("token_indices must be non-negative")
        if torch.unique(token_indices).numel() != token_indices.numel():
            raise ValueError("token_indices must not contain duplicates")

    order = torch.argsort(token_indices)
    sorted_indices = token_indices.index_select(0, order)
    weighted_outputs = expert_outputs * routing_weights.to(
        device=expert_outputs.device, dtype=expert_outputs.dtype
    )
    sorted_outputs = weighted_outputs.index_select(0, order.to(device=expert_outputs.device))
    return sorted_outputs, sorted_indices


def _validate_distributed_inputs(
    inputs: torch.Tensor,
    router_output: RouterOutput,
    experts: Sequence[nn.Module],
    expert_placement: ExpertPlacement,
    config: DistributedEPConfig,
) -> None:
    if not isinstance(inputs, torch.Tensor):
        raise ValueError("inputs must be a torch.Tensor")
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [num_tokens, hidden_dim]")
    if not isinstance(router_output, RouterOutput):
        raise ValueError("router_output must be a RouterOutput")
    if not isinstance(expert_placement, ExpertPlacement):
        raise ValueError("expert_placement must be an ExpertPlacement")
    if len(experts) != expert_placement.num_experts:
        raise ValueError("experts length must match expert placement num_experts")
    if expert_placement.num_ep_ranks != config.world_size:
        raise ValueError("ExpertPlacement num_ep_ranks must match distributed world_size")
    if router_output.expert_indices.shape != (inputs.shape[0], 1):
        raise ValueError("router expert_indices must have shape [num_tokens, 1]")
    if router_output.weights.shape != (inputs.shape[0], 1):
        raise ValueError("router weights must have shape [num_tokens, 1]")
    if inputs.device != config.device:
        raise ValueError("inputs device must match DistributedEPConfig device")


def _validate_returned_source_tokens(token_indices: torch.Tensor, *, num_tokens: int, config: DistributedEPConfig) -> None:
    expected = source_token_indices(
        num_tokens,
        rank=config.rank,
        world_size=config.world_size,
        device=token_indices.device,
    )
    if not torch.equal(torch.sort(token_indices).values, expected):
        raise ValueError("returned token indices must match this rank's source-token shard")


def run_distributed_ep_moe(
    inputs: torch.Tensor,
    router_output: RouterOutput,
    experts: Sequence[nn.Module],
    expert_placement: ExpertPlacement,
    *,
    config: DistributedEPConfig | None = None,
    group: dist.ProcessGroup | None = None,
    replicate_output: bool = False,
) -> tuple[torch.Tensor, DistributedEPTrace]:
    """Run the minimal Stage 3 distributed EP forward path.

    By default the combine is *sharded*: each rank returns only its own
    source-token rows (shape ``[num_local_tokens, hidden_dim]``) in ascending
    token-index order, with the matching token indices available on the trace.
    This is the realistic EP combine and avoids any extra collective.

    With ``replicate_output=True`` the legacy behaviour is used: every rank
    returns the full ``[num_tokens, hidden_dim]`` output, assembled with a
    final ``all_reduce``. This is kept for comparison and convenience; it moves
    roughly ``2 * (world_size - 1) / world_size * num_tokens * hidden_dim``
    extra elements per rank versus the sharded path.
    """

    if config is None:
        config = DistributedEPConfig.from_process_group(group=group, device=inputs.device)
    if not dist.is_available() or not dist.is_initialized():
        raise ValueError("torch.distributed must be initialized")
    if config.rank != dist.get_rank(group=group) or config.world_size != dist.get_world_size(group=group):
        raise ValueError("DistributedEPConfig must match the initialized process group")
    _validate_distributed_inputs(inputs, router_output, experts, expert_placement, config)

    ep_context = config.to_ep_context()
    ep_context.require_compatible_placement(expert_placement)
    send_plan = build_distributed_payload_plan(
        router_output,
        expert_placement,
        source_rank=config.rank,
        world_size=config.world_size,
        device=config.device,
    )
    dispatch_counts = exchange_counts(send_plan.send_counts_by_rank, group=group)

    send_token_payload = inputs.index_select(0, send_plan.token_indices.to(device=inputs.device))
    received_tokens = all_to_all_variable_tensors(send_token_payload, dispatch_counts, config=config, group=group)
    received_token_indices = all_to_all_variable_tensors(
        send_plan.token_indices,
        dispatch_counts,
        config=config,
        group=group,
    )
    received_expert_ids = all_to_all_variable_tensors(
        send_plan.expert_ids,
        dispatch_counts,
        config=config,
        group=group,
    )
    received_routing_weights = all_to_all_variable_tensors(
        send_plan.routing_weights,
        dispatch_counts,
        config=config,
        group=group,
    )

    local_expert_outputs = execute_received_experts(
        received_tokens,
        received_expert_ids,
        experts,
        expert_placement,
        rank=config.rank,
    )

    return_counts = reverse_count_exchange(dispatch_counts, rank=config.rank)
    returned_outputs = all_to_all_variable_tensors(local_expert_outputs, return_counts, config=config, group=group)
    returned_token_indices = all_to_all_variable_tensors(
        received_token_indices,
        return_counts,
        config=config,
        group=group,
    )
    returned_routing_weights = all_to_all_variable_tensors(
        received_routing_weights,
        return_counts,
        config=config,
        group=group,
    )
    _validate_returned_source_tokens(returned_token_indices, num_tokens=inputs.shape[0], config=config)

    if replicate_output:
        output = apply_partial_combine(
            returned_outputs,
            returned_token_indices,
            returned_routing_weights,
            num_tokens=inputs.shape[0],
        )
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        output_token_indices = source_token_indices(
            inputs.shape[0], rank=config.rank, world_size=config.world_size, device=output.device
        )
    else:
        output, output_token_indices = apply_sharded_combine(
            returned_outputs,
            returned_token_indices,
            returned_routing_weights,
        )

    return output, DistributedEPTrace(
        config=config,
        ep_context=ep_context,
        token_layout=send_plan.token_layout,
        send_plan=send_plan,
        dispatch_counts=dispatch_counts,
        return_counts=return_counts,
        owned_experts=expert_placement.experts_for_rank(config.rank),
        returned_token_indices=returned_token_indices,
        output_token_indices=output_token_indices,
        replicate_output=replicate_output,
    )


@dataclass(frozen=True)
class DistributedTopKPayloadPlan:
    """Local source-rank payload for the distributed top-k path (one row per kept assignment)."""

    send_counts_by_rank: torch.Tensor
    send_offsets: torch.Tensor
    token_indices: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.send_counts_by_rank.ndim != 1 or not _is_integer_tensor(self.send_counts_by_rank):
            raise ValueError("send_counts_by_rank must be a 1D integer tensor")
        if (self.send_counts_by_rank < 0).any().item():
            raise ValueError("send_counts_by_rank must be non-negative")
        if self.send_offsets.shape != (self.send_counts_by_rank.numel() + 1,):
            raise ValueError("send_offsets must have shape [world_size + 1]")
        if self.token_indices.ndim != 1 or self.expert_ids.ndim != 1:
            raise ValueError("token_indices and expert_ids must be 1D tensors")
        if self.token_indices.numel() != self.expert_ids.numel():
            raise ValueError("token_indices and expert_ids must have equal length")
        if not _is_integer_tensor(self.token_indices) or not _is_integer_tensor(self.expert_ids):
            raise ValueError("token_indices and expert_ids must use integer dtypes")
        if self.routing_weights.shape != (self.token_indices.numel(), 1):
            raise ValueError("routing_weights must have shape [num_assignments, 1]")
        if not self.routing_weights.dtype.is_floating_point:
            raise ValueError("routing_weights must use a floating dtype")
        if self.send_offsets[-1].item() != self.token_indices.numel():
            raise ValueError("send_offsets must end at the assignment count")


@dataclass(frozen=True)
class DistributedTopKTrace:
    """Rank-local trace for the distributed top-k EP forward path."""

    config: DistributedEPConfig
    ep_context: EPContext
    send_plan: DistributedTopKPayloadPlan
    dispatch_counts: CountExchange
    return_counts: CountExchange
    owned_experts: tuple[int, ...]
    capacity: int | None
    num_local_dropped: int
    output_token_indices: torch.Tensor
    replicate_output: bool


def build_distributed_topk_payload_plan(
    router_output: TopKRouterOutput,
    expert_placement: ExpertPlacement,
    keep_mask: torch.Tensor,
    *,
    source_rank: int,
    world_size: int,
    device: torch.device | str | None = None,
) -> DistributedTopKPayloadPlan:
    """Build this source rank's kept (token, slot) assignments for top-k dispatch."""

    _validate_rank_world(source_rank, world_size)
    if expert_placement.num_ep_ranks != world_size:
        raise ValueError("world_size must match ExpertPlacement num_ep_ranks")
    if keep_mask.shape != router_output.expert_indices.shape:
        raise ValueError("keep_mask must match router expert_indices shape")
    target_device = torch.device(device) if device is not None else router_output.expert_indices.device

    num_tokens, k = router_output.expert_indices.shape
    records: list[tuple[int, int, int, float]] = []
    for token_index in range(num_tokens):
        if token_index % world_size != source_rank:
            continue
        for slot in range(k):
            if not bool(keep_mask[token_index, slot].item()):
                continue
            expert_id = int(router_output.expert_indices[token_index, slot].item())
            dest_rank = expert_placement.owner_rank(expert_id)
            weight = float(router_output.weights[token_index, slot].item())
            records.append((dest_rank, expert_id, token_index, weight))

    records.sort(key=lambda record: (record[0], record[1], record[2]))
    send_counts = torch.zeros(world_size, dtype=torch.long, device=target_device)
    for dest_rank, _expert_id, _token_index, _weight in records:
        send_counts[dest_rank] += 1
    send_offsets = _prefix_offsets(send_counts)

    token_indices = torch.tensor([r[2] for r in records], dtype=torch.long, device=target_device)
    expert_ids = torch.tensor([r[1] for r in records], dtype=torch.long, device=target_device)
    if records:
        routing_weights = torch.tensor(
            [[r[3]] for r in records], dtype=router_output.weights.dtype, device=target_device
        )
    else:
        routing_weights = router_output.weights.new_empty((0, 1)).to(device=target_device)
    return DistributedTopKPayloadPlan(
        send_counts_by_rank=send_counts,
        send_offsets=send_offsets,
        token_indices=token_indices,
        expert_ids=expert_ids,
        routing_weights=routing_weights,
    )


def apply_sharded_topk_combine(
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    owned_token_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum each owned source token's kept-slot contributions into a dense shard.

    ``owned_token_indices`` is this rank's full ascending set of source tokens.
    A token with every slot dropped keeps a zero row. Unlike the top-1 sharded
    combine, duplicate token indices are expected (one per kept slot) and summed.
    """

    if expert_outputs.ndim != 2:
        raise ValueError("expert_outputs must have shape [num_assignments, hidden_dim]")
    if token_indices.shape != (expert_outputs.shape[0],):
        raise ValueError("token_indices must have one entry per expert output")
    if routing_weights.shape != (expert_outputs.shape[0], 1):
        raise ValueError("routing_weights must have shape [num_assignments, 1]")
    if not routing_weights.dtype.is_floating_point:
        raise ValueError("routing_weights must use a floating dtype")
    if not torch.isfinite(routing_weights).all().item():
        raise ValueError("routing_weights must be finite")
    if owned_token_indices.ndim != 1 or not _is_integer_tensor(owned_token_indices):
        raise ValueError("owned_token_indices must be a 1D integer tensor")

    output = torch.zeros(
        (owned_token_indices.numel(), expert_outputs.shape[1]),
        dtype=expert_outputs.dtype,
        device=expert_outputs.device,
    )
    if expert_outputs.shape[0] > 0:
        owned = owned_token_indices.to(device=expert_outputs.device)
        positions = torch.searchsorted(owned, token_indices.to(device=expert_outputs.device))
        if (positions >= owned.numel()).any().item() or not torch.equal(
            owned.index_select(0, positions), token_indices.to(device=expert_outputs.device)
        ):
            raise ValueError("token_indices must be a subset of owned_token_indices")
        weighted = expert_outputs * routing_weights.to(device=expert_outputs.device, dtype=expert_outputs.dtype)
        output.index_add_(0, positions, weighted)
    return output, owned_token_indices


def _validate_returned_topk_tokens(token_indices: torch.Tensor, *, num_tokens: int, config: DistributedEPConfig) -> None:
    owned = source_token_indices(
        num_tokens, rank=config.rank, world_size=config.world_size, device=token_indices.device
    )
    if token_indices.numel() > 0:
        if not torch.isin(token_indices, owned).all().item():
            raise ValueError("returned token indices must belong to this rank's source-token shard")


def run_distributed_topk_ep_moe(
    inputs: torch.Tensor,
    router_output: TopKRouterOutput,
    experts: Sequence[nn.Module],
    expert_placement: ExpertPlacement,
    *,
    config: DistributedEPConfig | None = None,
    group: dist.ProcessGroup | None = None,
    capacity_factor: float | None = None,
    replicate_output: bool = False,
) -> tuple[torch.Tensor, DistributedTopKTrace]:
    """Run the distributed top-k EP forward path with optional capacity dropping.

    The capacity mask is computed identically on every rank from the replicated
    router output, so dropping matches the single-process reference exactly. Each
    token's kept slots all originate on its owner rank and return there, where
    they are summed. Output is sharded by default; ``replicate_output=True``
    assembles the full output with a final ``all_reduce`` (for comparison).
    """

    if config is None:
        config = DistributedEPConfig.from_process_group(group=group, device=inputs.device)
    if not dist.is_available() or not dist.is_initialized():
        raise ValueError("torch.distributed must be initialized")
    if config.rank != dist.get_rank(group=group) or config.world_size != dist.get_world_size(group=group):
        raise ValueError("DistributedEPConfig must match the initialized process group")
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
        raise ValueError("inputs must have shape [num_tokens, hidden_dim]")
    if not isinstance(router_output, TopKRouterOutput):
        raise ValueError("router_output must be a TopKRouterOutput")
    if not isinstance(expert_placement, ExpertPlacement):
        raise ValueError("expert_placement must be an ExpertPlacement")
    if len(experts) != expert_placement.num_experts:
        raise ValueError("experts length must match expert placement num_experts")
    if expert_placement.num_ep_ranks != config.world_size:
        raise ValueError("ExpertPlacement num_ep_ranks must match distributed world_size")
    if router_output.num_tokens != inputs.shape[0]:
        raise ValueError("router_output num_tokens must match inputs")
    if inputs.device != config.device:
        raise ValueError("inputs device must match DistributedEPConfig device")

    num_tokens = inputs.shape[0]
    num_experts = expert_placement.num_experts
    if capacity_factor is None:
        keep_mask = torch.ones_like(router_output.expert_indices, dtype=torch.bool)
        capacity = None
    else:
        capacity = compute_expert_capacity(num_tokens, num_experts, router_output.k, capacity_factor)
        keep_mask = build_capacity_mask(router_output, num_experts, capacity)

    ep_context = config.to_ep_context()
    ep_context.require_compatible_placement(expert_placement)
    send_plan = build_distributed_topk_payload_plan(
        router_output,
        expert_placement,
        keep_mask,
        source_rank=config.rank,
        world_size=config.world_size,
        device=config.device,
    )
    dispatch_counts = exchange_counts(send_plan.send_counts_by_rank, group=group)

    send_token_payload = inputs.index_select(0, send_plan.token_indices.to(device=inputs.device))
    received_tokens = all_to_all_variable_tensors(send_token_payload, dispatch_counts, config=config, group=group)
    received_token_indices = all_to_all_variable_tensors(send_plan.token_indices, dispatch_counts, config=config, group=group)
    received_expert_ids = all_to_all_variable_tensors(send_plan.expert_ids, dispatch_counts, config=config, group=group)
    received_routing_weights = all_to_all_variable_tensors(send_plan.routing_weights, dispatch_counts, config=config, group=group)

    local_expert_outputs = execute_received_experts(
        received_tokens, received_expert_ids, experts, expert_placement, rank=config.rank
    )

    return_counts = reverse_count_exchange(dispatch_counts, rank=config.rank)
    returned_outputs = all_to_all_variable_tensors(local_expert_outputs, return_counts, config=config, group=group)
    returned_token_indices = all_to_all_variable_tensors(received_token_indices, return_counts, config=config, group=group)
    returned_routing_weights = all_to_all_variable_tensors(received_routing_weights, return_counts, config=config, group=group)
    _validate_returned_topk_tokens(returned_token_indices, num_tokens=num_tokens, config=config)

    owned = source_token_indices(num_tokens, rank=config.rank, world_size=config.world_size, device=inputs.device)
    if replicate_output:
        output = torch.zeros((num_tokens, inputs.shape[1]), dtype=returned_outputs.dtype, device=inputs.device)
        if returned_outputs.shape[0] > 0:
            weighted = returned_outputs * returned_routing_weights.to(device=output.device, dtype=output.dtype)
            output.index_add_(0, returned_token_indices.to(device=output.device), weighted)
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        output_token_indices = torch.arange(num_tokens, dtype=torch.long, device=output.device)
    else:
        output, output_token_indices = apply_sharded_topk_combine(
            returned_outputs, returned_token_indices, returned_routing_weights, owned
        )

    if owned.numel() > 0:
        local_keep = keep_mask[owned.to(device=keep_mask.device)]
    else:
        local_keep = keep_mask.new_zeros((0, router_output.k))
    num_local_dropped = int((~local_keep).sum().item())
    return output, DistributedTopKTrace(
        config=config,
        ep_context=ep_context,
        send_plan=send_plan,
        dispatch_counts=dispatch_counts,
        return_counts=return_counts,
        owned_experts=expert_placement.experts_for_rank(config.rank),
        capacity=capacity,
        num_local_dropped=num_local_dropped,
        output_token_indices=output_token_indices,
        replicate_output=replicate_output,
    )
