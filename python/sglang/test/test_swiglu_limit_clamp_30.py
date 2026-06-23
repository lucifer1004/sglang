import unittest

import torch
import torch.nn.functional as F


def _reference_silu_and_mul(gateup: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = torch.chunk(gateup.to(torch.float32), chunks=2, dim=-1)
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    # Match act_and_mul_kernel: activation is computed in fp32, cast back to the
    # input dtype, then multiplied and cast to the output dtype.
    return (F.silu(gate).to(gateup.dtype) * up.to(gateup.dtype)).to(gateup.dtype)


def _reference_silu_and_mul_fused(gateup: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = torch.chunk(gateup, chunks=2, dim=-1)
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    return (F.silu(gate) * up).to(gateup.dtype)


def _require_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available")


def test_act_and_mul_triton_uses_runtime_swiglu_limit_30():
    _require_cuda()
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        act_and_mul_triton,
    )

    gate = torch.tensor(
        [
            [25.0, 40.0, -35.0, 8.0],
            [9.0, 31.0, -5.0, 0.5],
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    up = torch.tensor(
        [
            [20.0, -45.0, 35.0, -12.0],
            [11.0, 29.0, -31.0, 0.25],
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    gateup = torch.cat([gate, up], dim=-1).contiguous()
    actual = torch.empty_like(gate)
    topk_ids = torch.zeros((gate.shape[0], 1), device="cuda", dtype=torch.int64)

    act_and_mul_triton(
        gateup,
        actual,
        config={"BLOCK_SIZE_M": 1},
        topk_ids=topk_ids,
        down_moe_use_tma=False,
        activation="silu",
        swiglu_limit=30.0,
    )

    expected_30 = _reference_silu_and_mul(gateup, 30.0)
    expected_10 = _reference_silu_and_mul(gateup, 10.0)

    torch.testing.assert_close(actual, expected_30, rtol=0, atol=0)
    assert not torch.equal(actual, expected_10)


def test_deep_gemm_apply_swiglu_limit_uses_runtime_limit_30():
    _require_cuda()
    from sglang.srt.layers.moe.moe_runner.deep_gemm import _apply_swiglu_limit

    gate = torch.tensor(
        [[25.0, 40.0, -35.0, 8.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    up = torch.tensor(
        [[20.0, -45.0, 35.0, -12.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    gateup = torch.cat([gate, up], dim=-1).contiguous()

    actual = _apply_swiglu_limit(gateup, 30.0)
    expected_gate = torch.clamp(gate, max=30.0)
    expected_up = torch.clamp(up, min=-30.0, max=30.0)
    expected_30 = torch.cat([expected_gate, expected_up], dim=-1)
    expected_10 = torch.cat(
        [torch.clamp(gate, max=10.0), torch.clamp(up, min=-10.0, max=10.0)],
        dim=-1,
    )

    torch.testing.assert_close(actual, expected_30, rtol=0, atol=0)
    assert not torch.equal(actual, expected_10)


def test_jit_silu_and_mul_clamp_uses_runtime_limit_30():
    _require_cuda()
    from sglang.jit_kernel.deepseek_v4 import silu_and_mul_clamp

    base_gate = torch.tensor(
        [
            [25.0, 40.0, -35.0, 8.0],
            [9.0, 31.0, -5.0, 0.5],
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    base_up = torch.tensor(
        [
            [20.0, -45.0, 35.0, -12.0],
            [11.0, 29.0, -31.0, 0.25],
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = base_gate.repeat_interleave(16, dim=-1).contiguous()
    up = base_up.repeat_interleave(16, dim=-1).contiguous()
    gateup = torch.cat([gate, up], dim=-1).contiguous()
    actual = torch.empty_like(gate)

    silu_and_mul_clamp(gateup, actual, 30.0)

    expected_30 = _reference_silu_and_mul_fused(gateup, 30.0)
    expected_10 = _reference_silu_and_mul_fused(gateup, 10.0)

    torch.testing.assert_close(actual, expected_30, rtol=0, atol=0.01)
    assert not torch.equal(actual, expected_10)
