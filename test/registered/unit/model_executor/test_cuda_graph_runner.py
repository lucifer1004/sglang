import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _make_model_runner(*, attn_cp_mode="none"):
    server_args = SimpleNamespace(
        cuda_graph_bs=[1, 2, 4, 8, 12, 16],
        enable_two_batch_overlap=False,
        enable_torch_compile=False,
        torch_compile_max_bs=32,
        attn_cp_mode=attn_cp_mode,
    )
    return SimpleNamespace(
        server_args=server_args,
        req_to_token_pool=SimpleNamespace(size=2048),
    )


class TestCudaGraphBatchSizes(unittest.TestCase):
    @patch(
        "sglang.srt.model_executor.cuda_graph_runner.require_gathered_buffer",
        return_value=False,
    )
    @patch(
        "sglang.srt.model_executor.cuda_graph_runner.get_attention_tp_size",
        return_value=1,
    )
    @patch(
        "sglang.srt.model_executor.cuda_graph_runner.get_attention_cp_size",
        return_value=2,
    )
    def test_context_parallel_filters_odd_cuda_graph_bs(
        self, _mock_cp_size, _mock_tp_size, _mock_gathered
    ):
        from sglang.srt.model_executor.cuda_graph_runner import (
            get_batch_sizes_to_capture,
        )

        capture_bs, _ = get_batch_sizes_to_capture(_make_model_runner())

        self.assertEqual(capture_bs, [2, 4, 8, 12, 16])

    @patch(
        "sglang.srt.model_executor.cuda_graph_runner.require_gathered_buffer",
        return_value=False,
    )
    @patch(
        "sglang.srt.model_executor.cuda_graph_runner.get_attention_tp_size",
        return_value=1,
    )
    @patch(
        "sglang.srt.model_executor.cuda_graph_runner.get_attention_cp_size",
        return_value=2,
    )
    def test_sharded_kv_context_parallel_keeps_bs_one_cuda_graph(
        self, _mock_cp_size, _mock_tp_size, _mock_gathered
    ):
        from sglang.srt.model_executor.cuda_graph_runner import (
            get_batch_sizes_to_capture,
        )

        capture_bs, _ = get_batch_sizes_to_capture(
            _make_model_runner(attn_cp_mode="sharded-kv")
        )

        self.assertEqual(capture_bs, [1, 2, 4, 8, 12, 16])


if __name__ == "__main__":
    unittest.main()
