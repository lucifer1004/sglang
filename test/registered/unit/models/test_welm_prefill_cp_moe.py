"""CPU contracts for WeLM persistent-token prefill CP MoE layouts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.models import welmv4 as welmv4_model
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _Backend:
    def __init__(self, is_none: bool):
        self._is_none = is_none

    def is_none(self):
        return self._is_none


def test_shared_expert_stays_tp_sharded_without_ep_dispatch(monkeypatch):
    monkeypatch.setattr(
        welmv4_model, "get_moe_a2a_backend", lambda: _Backend(is_none=True)
    )

    assert welmv4_model._welm_shared_expert_parallel_kwargs() == {}


def test_shared_expert_is_replicated_for_ep_dispatch(monkeypatch):
    monkeypatch.setattr(
        welmv4_model, "get_moe_a2a_backend", lambda: _Backend(is_none=False)
    )

    assert welmv4_model._welm_shared_expert_parallel_kwargs() == {
        "tp_rank": 0,
        "tp_size": 1,
    }


@pytest.mark.parametrize("ep_dispatch", [False, True])
def test_prefill_cp_ppln_keeps_communicator_fp32_residual(
    monkeypatch, ep_dispatch
):
    captured = {}
    fp32_residual = torch.tensor([[1.0039061]], dtype=torch.float32)

    class PrefillCommunicator:
        use_ep_dispatch = ep_dispatch

        def validate_mlp(self, _mlp):
            captured["validated"] = True

        def prepare_attn(
            self, hidden_states, residual, forward_batch, **kwargs
        ):
            captured["prepare_attn_kwargs"] = kwargs
            return hidden_states, fp32_residual

        def prepare_mlp(self, hidden_states, residual, forward_batch):
            captured["prepare_mlp_residual"] = residual
            return hidden_states, residual

        def has_active_mlp_tokens(self, forward_batch):
            return True

        def build_router_context(self, forward_batch):
            assert not ep_dispatch
            captured["router_context_batch"] = forward_batch
            return "router-context"

        def postprocess_layer(self, hidden_states, residual, forward_batch):
            return hidden_states, residual

    class FakeAttention:
        use_o_norm = False
        o_norm_needs_attn_tp_reduce = False
        kv_mirror_layer_idx = -1

        def __call__(self, **kwargs):
            return kwargs["hidden_states"]

    class FakeMLP:
        tp_size = 2

        def __call__(
            self,
            hidden_states,
            hidden_states_fp32,
            forward_batch,
            use_reduce_scatter,
            **kwargs,
        ):
            captured["mlp_hidden_states_fp32"] = hidden_states_fp32
            captured["mlp_kwargs"] = kwargs
            return hidden_states

    layer = SimpleNamespace(
        layer_communicator=object(),
        prefill_cp_communicator=PrefillCommunicator(),
        _prefill_cp_mlp_validated=False,
        ppln=True,
        config_layer_id=1,
        prenorm_layer_idx=[],
        self_attn=FakeAttention(),
        kv_mirror_layers=[],
        is_nextn=False,
        is_final_layer=False,
        layer_id=0,
        mlp=FakeMLP(),
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=object(),
        dp_padding_mode=None,
    )

    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_suffix_parallel_enabled", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "_welm_needs_empty_dp_collectives",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_dispatch_attention", lambda *_args: True
    )
    monkeypatch.setattr(
        welmv4_model,
        "_welm_should_contract_idle_extend_dp_metadata",
        lambda *_: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda *_: False
    )

    welmv4_model.Qwen2MoeDecoderLayer.forward(
        layer,
        positions=torch.arange(1),
        hidden_states=torch.ones((1, 1), dtype=torch.bfloat16),
        forward_batch=forward_batch,
        residual=None,
        kv_mirror_states={},
    )

    assert captured["validated"] is True
    assert captured["prepare_attn_kwargs"] == {
        "residual_after_layernorm": True
    }
    assert captured["prepare_mlp_residual"] is fp32_residual
    assert captured["mlp_hidden_states_fp32"] is None
    if ep_dispatch:
        assert "router_context_batch" not in captured
        assert "prefill_cp_router_context" not in captured["mlp_kwargs"]
    else:
        assert captured["router_context_batch"] is forward_batch
        assert (
            captured["mlp_kwargs"]["prefill_cp_router_context"]
            == "router-context"
        )


def test_prefill_cp_moe_routes_only_owner_rows_but_keeps_full_expert_input(
    monkeypatch,
):
    hidden = torch.arange(24, dtype=torch.bfloat16).view(6, 4)
    hidden_fp32 = hidden.float()
    local_hidden = hidden[2:4]
    full_weights = torch.arange(12, dtype=torch.float32).view(6, 2)
    full_ids = torch.arange(12, dtype=torch.int64).view(6, 2)
    captured = {}

    class RouterContext:
        def local_rows(self, tensor):
            captured.setdefault("local_rows", []).append(tensor)
            return tensor[2:4]

        def gather_routing_metadata(self, weights, ids):
            captured["local_weights"] = weights
            captured["local_ids"] = ids
            return full_weights, full_ids

    class SharedExpert:
        def __call__(self, tensor):
            captured["shared_hidden"] = tensor
            return torch.zeros_like(tensor)

    class TopK:
        def __call__(self, routing_hidden, router_logits):
            captured["topk_hidden"] = routing_hidden
            captured["router_logits"] = router_logits
            return StandardTopKOutput(
                topk_weights=torch.ones((2, 2), dtype=torch.float32),
                topk_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
                router_logits=router_logits,
            )

    class Experts:
        def __call__(self, tensor, topk_output):
            captured["expert_hidden"] = tensor
            captured["expert_topk"] = topk_output
            return tensor.clone()

    def router_linear(tensor, weight):
        captured["router_hidden"] = tensor
        return torch.arange(8, dtype=torch.float32).view(2, 4)

    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4_model, "mmq_style_router_linear", router_linear)
    monkeypatch.setattr(
        welmv4_model,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(recording=False),
    )
    monkeypatch.setattr(
        welmv4_model, "get_global_experts_capturer", lambda: None
    )
    block = SimpleNamespace(
        layer_id=3,
        shared_expert=SharedExpert(),
        shared_expert_gate=None,
        gate=SimpleNamespace(weight=torch.empty((4, 4))),
        topk=TopK(),
        experts=Experts(),
        tp_size=1,
        router_score_func="sigmoid",
    )

    output = welmv4_model.Qwen2MoeSparseMoeBlock.forward(
        block,
        hidden,
        hidden_fp32,
        forward_batch=SimpleNamespace(),
        prefill_cp_router_context=RouterContext(),
    )

    assert torch.equal(output, hidden)
    assert torch.equal(captured["shared_hidden"], hidden)
    assert torch.equal(captured["router_hidden"], local_hidden)
    assert torch.equal(captured["topk_hidden"], local_hidden)
    assert torch.equal(captured["expert_hidden"], hidden)
    assert captured["expert_topk"].topk_weights is full_weights
    assert captured["expert_topk"].topk_ids is full_ids
    assert captured["expert_topk"].router_logits.shape == (6, 0)


def test_prefill_cp_default_router_does_not_require_fp32_hidden(monkeypatch):
    hidden = torch.arange(24, dtype=torch.bfloat16).view(6, 4)
    full_weights = torch.ones((6, 2), dtype=torch.float32)
    full_ids = torch.zeros((6, 2), dtype=torch.int64)
    captured = {}

    class RouterContext:
        def local_rows(self, tensor):
            assert torch.equal(tensor, hidden)
            captured["local_rows_calls"] = captured.get("local_rows_calls", 0) + 1
            return tensor[2:4]

        def gather_routing_metadata(self, weights, ids):
            return full_weights, full_ids

    class TopK:
        def __call__(self, routing_hidden, router_logits):
            return StandardTopKOutput(
                topk_weights=torch.ones((2, 2), dtype=torch.float32),
                topk_ids=torch.zeros((2, 2), dtype=torch.int64),
                router_logits=router_logits,
            )

    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "mmq_style_router_linear",
        lambda tensor, _weight: torch.ones((tensor.shape[0], 4)),
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(recording=False),
    )
    monkeypatch.setattr(welmv4_model, "get_global_experts_capturer", lambda: None)
    block = SimpleNamespace(
        layer_id=3,
        shared_expert=None,
        shared_expert_gate=None,
        gate=SimpleNamespace(weight=torch.empty((4, 4))),
        topk=TopK(),
        experts=lambda tensor, _topk: tensor,
        tp_size=1,
        router_score_func="sigmoid",
    )

    output = welmv4_model.Qwen2MoeSparseMoeBlock.forward(
        block,
        hidden,
        None,
        forward_batch=SimpleNamespace(),
        prefill_cp_router_context=RouterContext(),
    )

    assert torch.equal(output, hidden)
    assert captured["local_rows_calls"] == 1


def test_prefill_cp_moe_rejects_nonstandard_local_topk(monkeypatch):
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    context = SimpleNamespace(
        local_rows=lambda tensor: tensor[:1],
    )
    block = SimpleNamespace(
        layer_id=0,
        shared_expert=None,
        shared_expert_gate=None,
        gate=SimpleNamespace(weight=torch.empty((4, 4))),
        topk=lambda *_args: object(),
        experts=MagicMock(),
        tp_size=1,
        router_score_func="sigmoid",
    )
    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "mmq_style_router_linear",
        lambda tensor, _weight: torch.ones((tensor.shape[0], 4)),
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(recording=False),
    )
    monkeypatch.setattr(
        welmv4_model, "get_global_experts_capturer", lambda: None
    )

    with pytest.raises(RuntimeError, match="Standard TopK"):
        welmv4_model.Qwen2MoeSparseMoeBlock.forward(
            block,
            hidden,
            hidden.float(),
            forward_batch=SimpleNamespace(),
            prefill_cp_router_context=context,
        )


@pytest.mark.parametrize(
    "backend",
    [
        welmv4_model.MoeRunnerBackend.TRITON,
        welmv4_model.MoeRunnerBackend.DEEP_GEMM,
    ],
)
def test_prefill_cp_local_router_accepts_metadata_only_moe_runners(backend):
    block = SimpleNamespace(
        experts=SimpleNamespace(
            quant_method=SimpleNamespace(
                runner=SimpleNamespace(runner_backend=backend)
            )
        )
    )

    welmv4_model.Qwen2MoeSparseMoeBlock.validate_prefill_cp_local_router(block)


def test_prefill_cp_local_router_rejects_runner_that_consumes_logits():
    block = SimpleNamespace(
        experts=SimpleNamespace(
            quant_method=SimpleNamespace(
                runner=SimpleNamespace(
                    runner_backend=welmv4_model.MoeRunnerBackend.MARLIN
                )
            )
        )
    )

    with pytest.raises(NotImplementedError, match="got marlin"):
        welmv4_model.Qwen2MoeSparseMoeBlock.validate_prefill_cp_local_router(
            block
        )


def test_non_dp_decode_does_not_select_layer_communicator_reduce_scatter(monkeypatch):
    communicator = SimpleNamespace(
        should_use_reduce_scatter=MagicMock(return_value=True)
    )
    captured = {}

    class FakeNorm:
        weight = torch.ones(1)

        def __call__(self, hidden_states, residual, **kwargs):
            if kwargs.get("clone_fp32_out"):
                return hidden_states, residual, hidden_states.float()
            return hidden_states, residual

    class FakeAttention:
        use_o_norm = False
        o_norm_needs_attn_tp_reduce = False
        kv_mirror_layer_idx = -1

        def __call__(self, **kwargs):
            return kwargs["hidden_states"]

    class FakeMLP:
        tp_size = 2

        def __call__(
            self,
            hidden_states,
            hidden_states_fp32,
            forward_batch,
            use_reduce_scatter,
            **kwargs,
        ):
            captured["use_reduce_scatter"] = use_reduce_scatter
            return hidden_states

    layer = SimpleNamespace(
        layer_communicator=communicator,
        prefill_cp_communicator=object(),
        _prefill_cp_mlp_validated=False,
        ppln=False,
        config_layer_id=0,
        prenorm_layer_idx=[],
        input_layernorm=FakeNorm(),
        post_attention_layernorm=FakeNorm(),
        self_attn=FakeAttention(),
        kv_mirror_layers=[],
        is_nextn=False,
        is_final_layer=False,
        layer_id=0,
        mlp=FakeMLP(),
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=None,
        dp_padding_mode=object(),
    )

    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_suffix_parallel_enabled", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "_welm_needs_empty_dp_collectives",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_dispatch_attention", lambda *_args: True
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_idle_extend_dp_metadata", lambda *_: False
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda *_: False
    )

    welmv4_model.Qwen2MoeDecoderLayer.forward(
        layer,
        positions=torch.arange(1),
        hidden_states=torch.ones((1, 1)),
        forward_batch=forward_batch,
        residual=None,
        kv_mirror_states={},
    )

    assert captured["use_reduce_scatter"] is False
    communicator.should_use_reduce_scatter.assert_not_called()
