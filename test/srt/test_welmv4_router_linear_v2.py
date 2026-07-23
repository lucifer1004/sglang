import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.welmv4_op import (
    _router_matmul_v2_m_bucket,
    mmq_style_router_linear,
    mmq_style_router_linear_v2,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.welmv4 import (
    Qwen2MoeSparseMoeBlock,
    _select_welm_router_linear_mode,
)


class TestWelmRouterLinearV2Mode(unittest.TestCase):
    def test_m_bucket_is_power_of_two_and_caps_large_prefill(self):
        expected = {
            1: 1,
            2: 2,
            3: 4,
            8: 8,
            16: 16,
            32: 32,
            64: 64,
            128: 128,
            1025: 2048,
            2048: 2048,
            4096: 2048,
        }
        self.assertEqual(
            {tokens: _router_matmul_v2_m_bucket(tokens) for tokens in expected},
            expected,
        )

    def _select(self, env, *, use_mxfp8=False, has_quant_config=False):
        with mock.patch.dict(os.environ, env, clear=True):
            return _select_welm_router_linear_mode(
                use_mxfp8=use_mxfp8,
                has_quant_config=has_quant_config,
            )

    def test_v1_mode_keeps_fp32_gate(self):
        version, dtype, warning = self._select({})
        self.assertEqual(version, "v1")
        self.assertEqual(dtype, torch.float32)
        self.assertIsNone(warning)

    def test_v2_mode_uses_only_bf16_gate(self):
        version, dtype, warning = self._select({"WELM_USE_MMQ_ROUTER_LINEAR_V2": "1"})
        self.assertEqual(version, "v2")
        self.assertEqual(dtype, torch.bfloat16)
        self.assertIsNone(warning)

    def test_previous_precision_takes_priority(self):
        version, dtype, warning = self._select(
            {
                "WELM_USE_PREVIOUS_PRECISION": "1",
                "WELM_USE_MMQ_ROUTER_LINEAR_V2": "1",
            }
        )
        self.assertEqual(version, "previous")
        self.assertEqual(dtype, torch.float32)
        self.assertIn("takes priority", warning)

    def test_mxfp8_falls_back_to_v1(self):
        version, dtype, warning = self._select(
            {"WELM_USE_MMQ_ROUTER_LINEAR_V2": "1"}, use_mxfp8=True
        )
        self.assertEqual(version, "v1")
        self.assertEqual(dtype, torch.float32)
        self.assertIn("MXFP8", warning)

    def test_other_quantization_falls_back_to_v1(self):
        version, dtype, warning = self._select(
            {"WELM_USE_MMQ_ROUTER_LINEAR_V2": "1"}, has_quant_config=True
        )
        self.assertEqual(version, "v1")
        self.assertEqual(dtype, torch.float32)
        self.assertIn("quantized", warning)

    def test_checkpoint_copy_preserves_bf16_parameter_dtype(self):
        parameter = torch.nn.Parameter(
            torch.empty((8, 4), dtype=torch.bfloat16), requires_grad=False
        )
        checkpoint_weight = torch.randn((8, 4), dtype=torch.float32)
        default_weight_loader(parameter, checkpoint_weight)
        self.assertEqual(parameter.dtype, torch.bfloat16)
        torch.testing.assert_close(parameter, checkpoint_weight.to(torch.bfloat16))

    def test_router_gate_loader_preserves_selected_dtype(self):
        block = Qwen2MoeSparseMoeBlock.__new__(Qwen2MoeSparseMoeBlock)
        torch.nn.Module.__init__(block)
        block.router_linear_version = "v2"
        block.router_gate_dtype = torch.bfloat16
        parameter = torch.nn.Parameter(
            torch.empty((8, 4), dtype=torch.bfloat16), requires_grad=False
        )
        base_loader = mock.Mock(
            side_effect=lambda param, loaded: param.data.copy_(loaded)
        )
        block.gate = SimpleNamespace(weight_loader=base_loader)
        checkpoint_weight = torch.randn((8, 4), dtype=torch.float32)

        block._router_gate_weight_loader(parameter, checkpoint_weight)

        base_loader.assert_called_once_with(parameter, checkpoint_weight)
        self.assertEqual(parameter.dtype, torch.bfloat16)
        torch.testing.assert_close(parameter, checkpoint_weight.to(torch.bfloat16))

    def test_router_gate_loader_rejects_dtype_drift(self):
        block = Qwen2MoeSparseMoeBlock.__new__(Qwen2MoeSparseMoeBlock)
        torch.nn.Module.__init__(block)
        block.router_linear_version = "v2"
        block.router_gate_dtype = torch.bfloat16
        block.gate = SimpleNamespace(weight_loader=mock.Mock())
        parameter = torch.nn.Parameter(
            torch.empty((8, 4), dtype=torch.float32), requires_grad=False
        )

        with self.assertRaisesRegex(RuntimeError, "dtype changed before weight load"):
            block._router_gate_weight_loader(parameter, torch.randn((8, 4)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestWelmRouterLinearV2Cuda(unittest.TestCase):
    def test_v1_v2_logits_and_topk(self):
        torch.manual_seed(7)
        weight_bf16 = torch.randn((512, 4096), device="cuda", dtype=torch.bfloat16)
        weight_fp32 = weight_bf16.float()
        for tokens in (1, 32):
            with self.subTest(tokens=tokens):
                x = torch.randn((tokens, 4096), device="cuda", dtype=torch.bfloat16)
                v1 = mmq_style_router_linear(x, weight_fp32)
                v2 = mmq_style_router_linear_v2(x, weight_bf16)
                torch.testing.assert_close(v2, v1, atol=2e-2, rtol=2e-3)
                self.assertTrue(
                    torch.equal(
                        v1.topk(10, dim=-1).indices, v2.topk(10, dim=-1).indices
                    )
                )


if __name__ == "__main__":
    unittest.main()
