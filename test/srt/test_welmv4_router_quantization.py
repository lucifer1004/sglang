import unittest
from unittest import mock

import torch

from sglang.srt.layers.welmv4_op import mmq_style_router_linear
from sglang.srt.models.welmv4 import (
    expert_bias_routing,
    sigmoid_routing_function,
)


class TestWelmV4RouterQuantizationDispatch(unittest.TestCase):
    def setUp(self):
        self.hidden_states = torch.randn(3, 4)
        self.gating_output = torch.randn(3, 8)

    def test_expert_bias_topk_dtype_depends_on_mxfp8(self):
        expert_bias = torch.randn(8)

        bf16_scores, bf16_ids = expert_bias_routing(
            self.hidden_states,
            self.gating_output,
            topk=2,
            expert_bias=expert_bias,
            use_mxfp8=False,
        )
        mxfp8_scores, mxfp8_ids = expert_bias_routing(
            self.hidden_states,
            self.gating_output,
            topk=2,
            expert_bias=expert_bias,
            use_mxfp8=True,
        )

        self.assertEqual(bf16_ids.dtype, torch.int64)
        self.assertEqual(mxfp8_ids.dtype, torch.int32)
        torch.testing.assert_close(bf16_scores, mxfp8_scores)
        torch.testing.assert_close(bf16_ids, mxfp8_ids.to(torch.int64))

    def test_sigmoid_topk_dtype_depends_on_mxfp8(self):
        bf16_scores, bf16_ids = sigmoid_routing_function(
            self.hidden_states,
            self.gating_output,
            topk=2,
            renormalize=False,
            use_mxfp8=False,
        )
        mxfp8_scores, mxfp8_ids = sigmoid_routing_function(
            self.hidden_states,
            self.gating_output,
            topk=2,
            renormalize=False,
            use_mxfp8=True,
        )

        self.assertEqual(bf16_ids.dtype, torch.int64)
        self.assertEqual(mxfp8_ids.dtype, torch.int32)
        torch.testing.assert_close(bf16_scores, mxfp8_scores)
        torch.testing.assert_close(bf16_ids, mxfp8_ids.to(torch.int64))

    @mock.patch(
        "sglang.srt.layers.welmv4_op._mmq_style_router_linear_cublas"
    )
    @mock.patch(
        "sglang.srt.layers.welmv4_op._mmq_style_router_linear_triton"
    )
    def test_router_gemm_backend_depends_on_mxfp8(
        self, triton_router, cublas_router
    ):
        x = mock.Mock(dtype=torch.bfloat16, is_cuda=True, shape=(2, 4))
        x.dim.return_value = 2
        weight = mock.Mock(shape=(8, 4))
        weight.dim.return_value = 2
        triton_result = torch.randn(2, 8)
        cublas_result = torch.randn(2, 8)
        triton_router.return_value = triton_result
        cublas_router.return_value = cublas_result

        self.assertIs(
            mmq_style_router_linear(x, weight, use_mxfp8=False),
            triton_result,
        )
        self.assertIs(
            mmq_style_router_linear(x, weight, use_mxfp8=True),
            cublas_result,
        )

        triton_router.assert_called_once_with(x, weight)
        cublas_router.assert_called_once_with(x, weight)


if __name__ == "__main__":
    unittest.main()
