import pytest
import torch

from sglang.srt.layers.moe.mk_moe_router import (
    MkMoeRouterMode,
    get_mk_moe_router_mode,
)
from sglang.srt.server_args import prepare_server_args

_ENV_NAME = "SGLANG_WELM_V45_80A3_MK_MOE_ROUTER_MODE"


def test_mk_moe_router_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(_ENV_NAME, raising=False)

    assert get_mk_moe_router_mode() is MkMoeRouterMode.OFF


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("tf32", MkMoeRouterMode.TF32),
        ("BF16", MkMoeRouterMode.BF16),
        ("fp32_exact", MkMoeRouterMode.FP32_EXACT),
    ],
)
def test_mk_moe_router_mode_is_selected_by_environment(monkeypatch, value, expected):
    monkeypatch.setenv(_ENV_NAME, value)

    assert get_mk_moe_router_mode() is expected


def test_mk_moe_router_mode_selects_gate_weight_dtype():
    assert MkMoeRouterMode.OFF.gate_weight_dtype is torch.float32
    assert MkMoeRouterMode.TF32.gate_weight_dtype is torch.float32
    assert MkMoeRouterMode.BF16.gate_weight_dtype is torch.bfloat16
    assert MkMoeRouterMode.FP32_EXACT.gate_weight_dtype is torch.float32


def test_mk_moe_router_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv(_ENV_NAME, "fastest")

    with pytest.raises(RuntimeError, match="must be one of"):
        get_mk_moe_router_mode()


def test_mk_moe_router_server_argument_is_removed():
    with pytest.raises(SystemExit):
        prepare_server_args(
            ["--model-path", "dummy", "--enable-welm-v45-80a3-mk-moe-router"]
        )
