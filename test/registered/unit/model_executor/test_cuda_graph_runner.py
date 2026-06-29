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
    def test_context_parallel_filters_odd_cuda_graph_bs(self):
        import sglang.srt.model_executor.cuda_graph_runner as cgr

        with (
            patch.object(cgr, "require_gathered_buffer", return_value=False),
            patch.object(cgr, "get_attention_tp_size", return_value=1),
            patch.object(cgr, "get_attention_cp_size", return_value=2),
        ):
            capture_bs, _ = cgr.get_batch_sizes_to_capture(_make_model_runner())

        self.assertEqual(capture_bs, [2, 4, 8, 12, 16])

    def test_sharded_kv_context_parallel_keeps_bs_one_cuda_graph(self):
        import sglang.srt.model_executor.cuda_graph_runner as cgr

        with (
            patch.object(cgr, "require_gathered_buffer", return_value=False),
            patch.object(cgr, "get_attention_tp_size", return_value=1),
            patch.object(cgr, "get_attention_cp_size", return_value=2),
        ):
            capture_bs, _ = cgr.get_batch_sizes_to_capture(
                _make_model_runner(attn_cp_mode="sharded-kv")
            )

        self.assertEqual(capture_bs, [1, 2, 4, 8, 12, 16])


class TestAttnCPCudaGraphSeqBuckets(unittest.TestCase):
    def _make_runner(
        self,
        *,
        capture_bs=(1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 72),
        max_context_len=262144,
        max_total_num_tokens=1596282,
        max_prefill_tokens=16384,
        page_size=1,
        explicit=False,
        is_attn_cp_sharded_kv=True,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        runner = CudaGraphRunner.__new__(CudaGraphRunner)
        runner.dllm_config = None
        runner.capture_bs = list(capture_bs)
        runner.enable_seq_len_graph_buckets = True
        runner.disable_padding = False
        runner.attn_backend = SimpleNamespace(
            get_cuda_graph_seq_len_fill_value=lambda: max_context_len,
            max_context_len=max_context_len,
            page_size=page_size,
            cuda_graph_max_seq_len_is_explicit=explicit,
            is_attn_cp_sharded_kv=is_attn_cp_sharded_kv,
        )
        runner.model_runner = SimpleNamespace(
            max_total_num_tokens=max_total_num_tokens,
            server_args=SimpleNamespace(max_prefill_tokens=max_prefill_tokens),
        )
        return runner

    def _install_buckets(self, runner):
        buckets = runner._build_seq_len_fill_value_buckets_by_bs()
        runner.seq_len_fill_values_by_bs = buckets
        runner.seq_len_fill_value = max(max(values) for values in buckets.values())
        return buckets

    def test_sharded_kv_skewed_long_context_bucket_covers_target_concurrency(self):
        runner = self._make_runner()

        buckets = self._install_buckets(runner)

        self.assertIn(32, buckets)
        self.assertGreaterEqual(max(buckets[32]), 100000)
        self.assertIsNotNone(
            runner._select_seq_len_fill_value_for_bs(32, 100000),
            buckets[32],
        )

    def test_sharded_kv_skew_bucket_does_not_expand_large_batch_buckets(self):
        runner = self._make_runner(capture_bs=(64, 72))

        buckets = self._install_buckets(runner)

        self.assertGreater(max(buckets[64]), max(buckets[72]))

    def test_sharded_kv_medium_batch_covers_new_eval_skew_case(self):
        runner = self._make_runner(capture_bs=(32, 40, 72))

        buckets = self._install_buckets(runner)

        self.assertIn(40, buckets)
        self.assertGreaterEqual(max(buckets[40]), 150000)
        self.assertIsNotNone(
            runner._select_seq_len_fill_value_for_bs(40, 150000),
            buckets[40],
        )
        self.assertLess(max(buckets[72]), 150000)

    def test_seq_len_bucket_alignment_rounds_up_to_preserve_coverage(self):
        runner = self._make_runner(
            capture_bs=(1,),
            max_context_len=4097,
            max_total_num_tokens=4097,
            max_prefill_tokens=1025,
            page_size=16,
        )

        buckets = self._install_buckets(runner)

        self.assertGreaterEqual(buckets[1][0], 1025)
        self.assertIsNotNone(runner._select_seq_len_fill_value_for_bs(1, 1025))

    def test_non_sharded_kv_does_not_expand_skew_buckets(self):
        runner = self._make_runner(
            is_attn_cp_sharded_kv=False,
        )

        buckets = self._install_buckets(runner)

        self.assertLess(max(buckets[40]), 150000)


if __name__ == "__main__":
    unittest.main()
