import pytest
import torch

from nano_moe_ep.dispatch_combine import run_logical_ep_moe
from nano_moe_ep.reference import ReferenceMoEFFN
from nano_moe_ep.routing import route_round_robin
from nano_moe_ep.types import EPContext, ExecutionMode, ExpertPlacement


RTOL = 1e-5
ATOL = 1e-6


def test_valid_single_process_ep_context_metadata():
    context = EPContext.single_process(num_ep_ranks=2, phase="dispatch")

    assert context.num_ep_ranks == 2
    assert context.local_rank is None
    assert context.execution_mode is ExecutionMode.LOGICAL_SINGLE_PROCESS
    assert context.device == "cpu"
    assert context.deterministic is True
    assert context.phase == "dispatch"


def test_ep_context_accepts_distributed_device_metadata():
    context = EPContext(
        num_ep_ranks=2,
        local_rank=1,
        execution_mode=ExecutionMode.DISTRIBUTED,
        device="cuda:1",
        phase="distributed_forward",
    )

    assert context.execution_mode is ExecutionMode.DISTRIBUTED
    assert context.device == "cuda:1"


@pytest.mark.parametrize("num_ep_ranks", [0, -1, True, "2"])
def test_ep_context_rejects_invalid_num_ep_ranks(num_ep_ranks):
    with pytest.raises(ValueError, match="num_ep_ranks"):
        EPContext(num_ep_ranks=num_ep_ranks)


@pytest.mark.parametrize("local_rank", [-1, 2, True, "0"])
def test_ep_context_rejects_invalid_local_rank(local_rank):
    with pytest.raises(ValueError, match="local_rank"):
        EPContext(num_ep_ranks=2, local_rank=local_rank)


def test_ep_context_num_ep_ranks_must_match_expert_placement():
    placement = ExpertPlacement.from_rank_experts(
        {0: [0], 1: [1]},
        num_experts=2,
        num_ep_ranks=2,
    )

    EPContext.single_process(num_ep_ranks=2).require_compatible_placement(placement)
    with pytest.raises(ValueError, match="num_ep_ranks"):
        EPContext.single_process(num_ep_ranks=3).require_compatible_placement(placement)


def test_logical_ep_with_context_still_matches_stage1_reference():
    torch.manual_seed(2030)
    model = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)
    inputs = torch.randn(7, 8)
    router_output = route_round_robin(num_tokens=inputs.shape[0], num_experts=model.num_experts)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0, 2], 1: [1, 3]},
        num_experts=model.num_experts,
        num_ep_ranks=2,
    )
    context = EPContext.single_process(num_ep_ranks=2)

    stage1_output, _ = model(inputs, router_output)
    stage2_output, trace = run_logical_ep_moe(
        inputs,
        router_output,
        model.experts,
        placement,
        ep_context=context,
    )

    torch.testing.assert_close(stage2_output, stage1_output, rtol=RTOL, atol=ATOL)
    assert trace.ep_context == context
    assert trace.ep_context.num_ep_ranks == trace.expert_placement.num_ep_ranks


def test_logical_ep_rejects_context_with_mismatched_num_ep_ranks():
    torch.manual_seed(2031)
    model = ReferenceMoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4)
    inputs = torch.randn(5, 8)
    router_output = route_round_robin(num_tokens=inputs.shape[0], num_experts=model.num_experts)
    placement = ExpertPlacement.from_rank_experts(
        {0: [0, 2], 1: [1, 3]},
        num_experts=model.num_experts,
        num_ep_ranks=2,
    )

    with pytest.raises(ValueError, match="num_ep_ranks"):
        run_logical_ep_moe(
            inputs,
            router_output,
            model.experts,
            placement,
            ep_context=EPContext.single_process(num_ep_ranks=3),
        )
