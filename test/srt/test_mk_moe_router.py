from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.mk_moe_router import (
    MkMoeRouterMode,
    WelmV45_80A3MkMoeRouterAdapter,
)
from sglang.srt.layers.welmv4_op import (
    mmq_style_expert_bias_topk,
    mmq_style_router_linear,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _is_h20() -> bool:
    if not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return "H20" in properties.name and properties.multi_processor_count == 78


pytestmark = pytest.mark.skipif(not _is_h20(), reason="requires NVIDIA H20")


def _mk_api():
    pytest.importorskip("mk")
    from mk.kernels.welm_v45_80a3_moe_router import (
        welm_v45_80a3_moe_router_init_workspace,
        welm_v45_80a3_moe_router_plan,
        welm_v45_80a3_moe_router_run,
    )

    return (
        welm_v45_80a3_moe_router_plan,
        welm_v45_80a3_moe_router_init_workspace,
        welm_v45_80a3_moe_router_run,
    )


def _inputs(
    m: int,
    seed: int,
    gate_weight_dtype: torch.dtype = torch.float32,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        (m, 2048), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    gate_weight = (
        (torch.randn((512, 2048), device="cuda", generator=generator) * 0.02)
        .to(torch.bfloat16)
        .to(gate_weight_dtype)
    )
    expert_bias = (
        torch.randn((512,), device="cuda", dtype=torch.float32, generator=generator)
        * 0.01
    )
    return hidden_states, gate_weight, expert_bias


def _reference(hidden_states, gate_weight, expert_bias):
    router_logits = mmq_style_router_linear(hidden_states, gate_weight)
    return mmq_style_expert_bias_topk(
        torch.sigmoid(router_logits), expert_bias, topk=10
    )


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64, 128])
def test_native_kernel_matches_sglang(m: int):
    plan_fn, init_fn, run_fn = _mk_api()
    hidden_states, gate_weight, expert_bias = _inputs(m, seed=1700 + m)
    expected_weights, expected_ids = _reference(hidden_states, gate_weight, expert_bias)
    actual_weights = torch.empty((m, 10), device="cuda", dtype=torch.float32)
    actual_ids = torch.empty((m, 10), device="cuda", dtype=torch.int32)
    plan = plan_fn(m=m, slot_count=1, device="cuda")
    workspace = init_fn(plan)

    run_fn(
        workspace,
        slot_id=0,
        hidden_states=hidden_states,
        gate_weight=gate_weight,
        expert_bias=expert_bias,
        topk_weights=actual_weights,
        topk_ids=actual_ids,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(actual_weights, expected_weights, rtol=3e-6, atol=3e-6)


def test_native_kernel_cuda_graph_replay_matches_sglang():
    plan_fn, init_fn, run_fn = _mk_api()
    m = 8
    hidden_states, gate_weight, expert_bias = _inputs(m, seed=1808)
    expected_weights, expected_ids = _reference(hidden_states, gate_weight, expert_bias)
    actual_weights = torch.empty((m, 10), device="cuda", dtype=torch.float32)
    actual_ids = torch.empty((m, 10), device="cuda", dtype=torch.int32)
    plan = plan_fn(m=m, slot_count=1, device="cuda")
    workspace = init_fn(plan)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_fn(
            workspace,
            slot_id=0,
            hidden_states=hidden_states,
            gate_weight=gate_weight,
            expert_bias=expert_bias,
            topk_weights=actual_weights,
            topk_ids=actual_ids,
        )

    for _ in range(3):
        init_fn(plan, workspace=workspace)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(actual_ids, expected_ids.to(torch.int32))
        torch.testing.assert_close(
            actual_weights, expected_weights, rtol=3e-6, atol=3e-6
        )


@pytest.mark.parametrize(
    "mode",
    [
        MkMoeRouterMode.TF32,
        MkMoeRouterMode.BF16,
        MkMoeRouterMode.FP32_EXACT,
    ],
)
def test_adapter_dispatches_selected_kernel(mode: MkMoeRouterMode):
    config = SimpleNamespace(
        hidden_size=2048,
        num_experts=512,
        num_experts_per_tok=10,
        router_score_func="sigmoid",
        moe_routing_type="expert_bias",
        norm_topk_prob=False,
        scale_seq_times=0,
    )
    server_args = SimpleNamespace(
        ep_size=1,
        moe_a2a_backend="none",
        enable_eplb=False,
        expert_distribution_recorder_mode=None,
        enable_expert_distribution_metrics=False,
        enable_moe_router_replay=False,
        enable_return_routed_experts=False,
        enable_dp_attention=False,
        enable_pdmux=False,
        enable_torch_compile=False,
        enable_lora=False,
        moe_runner_backend="auto",
    )
    adapter = WelmV45_80A3MkMoeRouterAdapter(
        mode=mode,
        config=config,
        server_args=server_args,
        slot_count=1,
        use_previous_precision=False,
        use_mxfp8=False,
    )
    m = 8
    hidden_states, gate_weight, expert_bias = _inputs(
        m, seed=1908, gate_weight_dtype=mode.gate_weight_dtype
    )
    expected_weights, expected_ids = _reference(hidden_states, gate_weight, expert_bias)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        can_run_tbo=False,
    )

    adapter.begin_forward(
        forward_batch=forward_batch,
        num_tokens=m,
        allow_fused_router=True,
    )
    if mode in (MkMoeRouterMode.TF32, MkMoeRouterMode.FP32_EXACT):
        assert adapter._active_state is not None
        assert adapter._active_state.plan.bitwise_sglang is (
            mode is MkMoeRouterMode.FP32_EXACT
        )
    output = adapter.route(
        slot_id=0,
        hidden_states=hidden_states,
        gate_weight=gate_weight,
        expert_bias=expert_bias,
    )
    torch.cuda.synchronize()

    assert output is not None
    assert torch.equal(output.topk_ids, expected_ids.to(torch.int32))
    if mode is MkMoeRouterMode.FP32_EXACT:
        assert torch.equal(output.topk_weights, expected_weights)
    else:
        torch.testing.assert_close(
            output.topk_weights, expected_weights, rtol=3e-6, atol=3e-6
        )


def test_adapter_validation_allows_independent_parallel_features():
    config = SimpleNamespace(
        hidden_size=2048,
        num_experts=512,
        num_experts_per_tok=10,
        router_score_func="sigmoid",
        moe_routing_type="expert_bias",
        norm_topk_prob=False,
        scale_seq_times=0,
    )
    server_args = SimpleNamespace(
        pp_size=2,
        dp_size=2,
        attn_cp_size=2,
        ep_size=1,
        moe_a2a_backend="none",
        enable_eplb=False,
        expert_distribution_recorder_mode=None,
        enable_expert_distribution_metrics=False,
        enable_moe_router_replay=False,
        enable_return_routed_experts=False,
        enable_dp_attention=False,
        enable_two_batch_overlap=True,
        enable_pdmux=False,
        enable_torch_compile=False,
        speculative_algorithm="EAGLE",
        enable_lora=False,
        moe_runner_backend="auto",
    )

    WelmV45_80A3MkMoeRouterAdapter._validate_config(
        mode=MkMoeRouterMode.TF32,
        config=config,
        server_args=server_args,
        slot_count=24,
        use_previous_precision=False,
        use_mxfp8=False,
    )


@pytest.mark.parametrize(
    "feature",
    ["enable_return_routed_experts", "enable_dp_attention", "enable_pdmux"],
)
def test_adapter_validation_rejects_incompatible_server_features(feature: str):
    config = SimpleNamespace(
        hidden_size=2048,
        num_experts=512,
        num_experts_per_tok=10,
        router_score_func="sigmoid",
        moe_routing_type="expert_bias",
        norm_topk_prob=False,
        scale_seq_times=0,
    )
    server_args = SimpleNamespace(
        ep_size=1,
        moe_a2a_backend="none",
        enable_eplb=False,
        expert_distribution_recorder_mode=None,
        enable_expert_distribution_metrics=False,
        enable_moe_router_replay=False,
        enable_return_routed_experts=False,
        enable_dp_attention=False,
        enable_pdmux=False,
        enable_torch_compile=False,
        enable_lora=False,
        moe_runner_backend="auto",
    )
    setattr(server_args, feature, True)

    with pytest.raises(RuntimeError, match=feature):
        WelmV45_80A3MkMoeRouterAdapter._validate_config(
            mode=MkMoeRouterMode.TF32,
            config=config,
            server_args=server_args,
            slot_count=48,
            use_previous_precision=False,
            use_mxfp8=False,
        )


def _stub_adapter():
    adapter = object.__new__(WelmV45_80A3MkMoeRouterAdapter)
    adapter._active_state = None
    adapter._active_m = 0
    adapter._active_forward_mode = ""
    adapter._states = {}
    adapter._logged_fallbacks = set()
    state = object()
    adapter._create_state = lambda num_tokens: state
    adapter._is_capturing = lambda: False
    adapter._init_workspace_fn = lambda *args, **kwargs: None
    return adapter, state


@pytest.mark.parametrize("m", [16, 32, 128, 512])
def test_adapter_does_not_fallback_by_decode_batch_size(m: int):
    adapter, state = _stub_adapter()
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        can_run_tbo=False,
    )

    adapter.begin_forward(
        forward_batch=forward_batch,
        num_tokens=m,
        allow_fused_router=True,
    )

    assert adapter._active_state is state
    assert adapter._active_m == m


@pytest.mark.parametrize(
    "forward_mode",
    [
        ForwardMode.DECODE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.DRAFT_EXTEND,
        ForwardMode.DRAFT_EXTEND_V2,
    ],
)
def test_adapter_supports_decode_and_mtp_forward_modes(forward_mode: ForwardMode):
    adapter, state = _stub_adapter()
    forward_batch = SimpleNamespace(
        forward_mode=forward_mode,
        can_run_tbo=False,
    )

    adapter.begin_forward(
        forward_batch=forward_batch,
        num_tokens=16,
        allow_fused_router=True,
    )

    assert adapter._active_state is state
    assert adapter._active_m == 16
    assert adapter._active_forward_mode == forward_mode.name


def test_adapter_falls_back_above_kernel_limit():
    adapter, _ = _stub_adapter()
    adapter._create_state = lambda num_tokens: pytest.fail("must not create a plan")
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        can_run_tbo=False,
    )

    adapter.begin_forward(
        forward_batch=forward_batch,
        num_tokens=513,
        allow_fused_router=True,
    )

    assert adapter._active_state is None
    assert adapter._active_m == 0


def test_adapter_logs_actual_dispatch_once(caplog):
    m = 1
    adapter = object.__new__(WelmV45_80A3MkMoeRouterAdapter)
    adapter._active_m = m
    adapter._active_state = SimpleNamespace(
        workspace=object(),
        topk_weights=(torch.empty((m, 10), device="cuda", dtype=torch.float32),),
        topk_ids=(torch.empty((m, 10), device="cuda", dtype=torch.int32),),
        empty_router_logits=torch.empty((m, 0), device="cuda", dtype=torch.float32),
    )
    adapter._run_fn = lambda *args, **kwargs: None
    adapter.mode = MkMoeRouterMode.TF32
    adapter._active_forward_mode = ForwardMode.TARGET_VERIFY.name
    adapter._logged_dispatches = set()
    adapter._logged_fallbacks = set()
    hidden_states = torch.empty((m, 2048), device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.empty((512, 2048), device="cuda", dtype=torch.float32)
    expert_bias = torch.empty((512,), device="cuda", dtype=torch.float32)

    with caplog.at_level(logging.INFO):
        for _ in range(2):
            adapter.route(
                slot_id=0,
                hidden_states=hidden_states,
                gate_weight=gate_weight,
                expert_bias=expert_bias,
            )

    active_logs = [
        record.message
        for record in caplog.records
        if "MK WeLM MoE router ACTIVE" in record.message
    ]
    assert active_logs == [
        "MK WeLM MoE router ACTIVE: mode=TARGET_VERIFY "
        "fused kernel dispatched for M=1; "
        "native Router bypassed"
    ]


def test_adapter_logs_native_fallback_once(caplog):
    adapter = object.__new__(WelmV45_80A3MkMoeRouterAdapter)
    adapter._active_state = None
    adapter._active_m = 0
    adapter._logged_fallbacks = set()
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        can_run_tbo=False,
    )

    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            adapter.begin_forward(
                forward_batch=forward_batch,
                num_tokens=8,
                allow_fused_router=True,
            )

    fallback_logs = [
        record.message
        for record in caplog.records
        if "MK WeLM MoE router FALLBACK" in record.message
    ]
    assert len(fallback_logs) == 1
    assert "native SGLang Router" in fallback_logs[0]
    assert "forward_mode=EXTEND is not a supported decode/MTP mode" in fallback_logs[0]
