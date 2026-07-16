import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.layers import dp_attention
from sglang.srt.layers import linear as linear_module
from sglang.srt.layers import logits_processor as logits_processor_module
from sglang.srt.layers.attention import flashattention_backend as fa_backend
from sglang.srt.layers.attention.cp_sharded_kv import (
    build_cp_sharded_kv_prefill_plan,
)
from sglang.srt.models import welmv4


class TestAttentionTPHelpers(unittest.TestCase):
    def test_sharded_kv_keeps_attention_tp_group(self):
        attn_tp_group = object()
        with (
            patch.object(
                dp_attention,
                "_is_sharded_kv_context_parallel",
                return_value=True,
            ),
            patch.object(dp_attention, "get_attn_tp_group", return_value=attn_tp_group),
            patch.object(dp_attention, "get_tp_group", return_value=object()),
            patch.object(
                dp_attention,
                "get_attn_tensor_model_parallel_rank",
                return_value=1,
            ),
            patch.object(
                dp_attention,
                "get_tensor_model_parallel_rank",
                return_value=5,
            ),
            patch.object(
                dp_attention,
                "get_attn_tensor_model_parallel_world_size",
                return_value=2,
            ),
            patch.object(
                dp_attention,
                "get_tensor_model_parallel_world_size",
                return_value=8,
            ),
        ):
            self.assertIs(dp_attention.get_attention_tp_group(), attn_tp_group)
            self.assertEqual(dp_attention.get_attention_tp_rank(), 1)
            self.assertEqual(dp_attention.get_attention_tp_size(), 2)

    def test_attention_world_info_for_tp8_cp4_attntp2(self):
        rank_info = [
            dp_attention.compute_dp_attention_world_info(
                enable_dp_attention=False,
                tp_rank=rank,
                tp_size=8,
                dp_size=1,
                attn_cp_size=4,
            )
            for rank in range(8)
        ]
        self.assertEqual(
            [(rank, size) for rank, size, _ in rank_info],
            [(0, 2), (1, 2), (0, 2), (1, 2), (0, 2), (1, 2), (0, 2), (1, 2)],
        )

    def test_sharded_kv_cp_groups_keep_attention_head_shards_separate(self):
        self.assertEqual(
            parallel_state.compute_sharded_kv_cp_group_ranks(
                tensor_model_parallel_size=8,
                attention_context_model_parallel_size=4,
                attention_data_parallel_size=1,
                num_tensor_model_parallel_groups=1,
            ),
            [[0, 2, 4, 6], [1, 3, 5, 7]],
        )
        self.assertEqual(
            parallel_state.compute_sharded_kv_cp_group_ranks(
                tensor_model_parallel_size=16,
                attention_context_model_parallel_size=4,
                attention_data_parallel_size=2,
                num_tensor_model_parallel_groups=1,
            ),
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
                [8, 10, 12, 14],
                [9, 11, 13, 15],
            ],
        )
        self.assertEqual(
            parallel_state.compute_sharded_kv_cp_group_ranks(
                tensor_model_parallel_size=8,
                attention_context_model_parallel_size=2,
                attention_data_parallel_size=1,
                num_tensor_model_parallel_groups=2,
            ),
            [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [8, 12],
                [9, 13],
                [10, 14],
                [11, 15],
            ],
        )

    def test_decode_sharded_kv_cp_groups_restore_contiguous_tp_layout(self):
        self.assertEqual(
            parallel_state.compute_sharded_kv_cp_group_ranks(
                tensor_model_parallel_size=4,
                attention_context_model_parallel_size=2,
                attention_data_parallel_size=1,
                num_tensor_model_parallel_groups=1,
                use_decode_layout=True,
            ),
            [[0, 1], [2, 3]],
        )
        self.assertEqual(
            parallel_state.compute_sharded_kv_cp_group_ranks(
                tensor_model_parallel_size=8,
                attention_context_model_parallel_size=4,
                attention_data_parallel_size=1,
                num_tensor_model_parallel_groups=1,
                use_decode_layout=True,
            ),
            [[0, 1, 2, 3], [4, 5, 6, 7]],
        )

    def test_welm_checkpoint_shard_uses_attention_tp_rank(self):
        with (
            patch.object(
                welmv4, "is_decode_cp_kv_sharded", return_value=False
            ),
            patch.object(welmv4, "get_attention_tp_rank", return_value=1),
            patch.object(welmv4, "get_attention_tp_size", return_value=2),
            patch.object(welmv4, "get_tensor_model_parallel_rank", return_value=5),
        ):
            self.assertEqual(welmv4._get_welm_head_shard_start(128, None), 128)

    def test_welm_decode_projection_restores_global_tp_shard(self):
        with (
            patch.object(
                welmv4, "is_decode_cp_kv_sharded", return_value=True
            ),
            patch.object(welmv4, "get_attention_tp_rank", return_value=1),
            patch.object(welmv4, "get_attention_tp_size", return_value=2),
            patch.object(welmv4, "get_tensor_model_parallel_rank", return_value=3),
            patch.object(
                welmv4, "get_tensor_model_parallel_world_size", return_value=4
            ),
        ):
            self.assertEqual(
                welmv4._get_welm_attention_tp_rank_and_size(), (3, 4)
            )

    def test_welm_prefill_projection_keeps_attention_tp_shard(self):
        with (
            patch.object(
                welmv4, "is_decode_cp_kv_sharded", return_value=False
            ),
            patch.object(welmv4, "get_attention_tp_rank", return_value=1),
            patch.object(welmv4, "get_attention_tp_size", return_value=2),
            patch.object(welmv4, "get_tensor_model_parallel_rank", return_value=3),
            patch.object(
                welmv4, "get_tensor_model_parallel_world_size", return_value=4
            ),
        ):
            self.assertEqual(
                welmv4._get_welm_attention_tp_rank_and_size(), (1, 2)
            )


class TestWelMAttentionTPProjection(unittest.TestCase):
    def test_prefill_plan_uses_cpu_lengths_without_reading_gpu_lengths(self):
        class CacheLensWithoutHostRead:
            dtype = torch.int32
            device = torch.device("cpu")

            def numel(self):
                return 1

            def reshape(self, *args, **kwargs):
                raise AssertionError("cache_seqlens must stay on device")

        cache_seqlens = CacheLensWithoutHostRead()
        plan = build_cp_sharded_kv_prefill_plan(
            logical_page_table=torch.tensor([[10, 11, 12]], dtype=torch.int32),
            cache_seqlens=cache_seqlens,
            prefix_lens=[4],
            seq_lens=[10],
            page_size=4,
        )

        self.assertIs(plan.full_cache_seqlens, cache_seqlens)
        self.assertEqual(plan.extend.cache_seqlens.tolist(), [6])

    def test_prefill_plan_is_initialized_once_per_forward(self):
        backend = object.__new__(fa_backend.FlashAttentionBackend)
        backend.page_size = 4
        metadata = fa_backend.FlashAttentionMetadata(
            page_table=torch.tensor([[10, 11, 12]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([10], dtype=torch.int32),
        )
        forward_batch = SimpleNamespace(
            extend_prefix_lens_cpu=[4],
            seq_lens_cpu=torch.tensor([10], dtype=torch.int64),
        )

        backend._set_sharded_kv_prefill_plan(metadata, forward_batch)
        first_plan = metadata.cp_sharded_kv_prefill_plan
        backend._set_sharded_kv_prefill_plan(metadata, forward_batch)

        self.assertIs(metadata.cp_sharded_kv_prefill_plan, first_plan)

    def test_prefill_split_reuses_the_forward_plan(self):
        backend = object.__new__(fa_backend.FlashAttentionBackend)
        backend.page_size = 4
        calls = []

        def fake_gather(page_table, cache_seqlens, key_cache, value_cache):
            calls.append((page_table.clone(), cache_seqlens.clone()))
            num_pages = page_table.shape[1]
            marker = float(len(calls))
            dense = torch.full((1, num_pages, 4, 1, 1), marker)
            return dense, dense + 10, torch.arange(num_pages).view(1, -1)

        plan = build_cp_sharded_kv_prefill_plan(
            logical_page_table=torch.tensor([[10, 11, 12]], dtype=torch.int32),
            cache_seqlens=torch.tensor([10], dtype=torch.int32),
            prefix_lens=[4],
            seq_lens=[10],
            page_size=4,
        )
        with patch.object(backend, "_gather_sharded_kv_dense", fake_gather):
            dense_k, dense_v, dense_page_table, dense_cache_seqlens = (
                backend._gather_sharded_kv_dense_prefill_split(
                    plan, torch.empty(0), torch.empty(0)
                )
            )

        self.assertEqual(
            [call[0].tolist() for call in calls], [[[10]], [[11, 12]]]
        )
        self.assertEqual([call[1].tolist() for call in calls], [[4], [6]])
        self.assertEqual(dense_k[:, :1].unique().tolist(), [1.0])
        self.assertEqual(dense_k[:, 1:].unique().tolist(), [2.0])
        self.assertEqual(dense_v[:, :1].unique().tolist(), [11.0])
        self.assertEqual(dense_v[:, 1:].unique().tolist(), [12.0])
        self.assertEqual(dense_page_table.tolist(), [[0, 1, 2]])
        self.assertEqual(dense_cache_seqlens.tolist(), [10])

    def test_fa3_local_head_counts_use_attention_tp_size(self):
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(num_attention_heads=24),
            get_num_kv_heads=lambda tp_size: 8 // tp_size,
        )

        with patch.object(fa_backend, "get_attention_tp_size", return_value=2):
            self.assertEqual(
                fa_backend._get_local_attention_head_counts(model_config),
                (12, 4),
            )

    def test_fa3_local_head_counts_follow_role_projection_tp_size(self):
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(num_attention_heads=24),
            get_num_kv_heads=lambda tp_size: max(1, 2 // tp_size),
        )

        self.assertEqual(
            fa_backend._get_local_attention_head_counts(model_config, tp_size=2),
            (12, 1),
        )
        self.assertEqual(
            fa_backend._get_local_attention_head_counts(model_config, tp_size=4),
            (6, 1),
        )

    def test_fa3_sharded_kv_gather_keeps_local_head_shard_on_cp_group(self):
        collective_inputs = []

        class FakeCPGroup:
            def all_reduce_coalesced(self, tensors):
                collective_inputs.append(tuple(tensor.clone() for tensor in tensors))
                return tensors

        backend = object.__new__(fa_backend.FlashAttentionBackend)
        backend.page_size = 4
        backend.enable_attn_cp_zero_dummy_slot = False
        backend.cp_sharded_page_table_resolver = SimpleNamespace(
            resolve_full=lambda page_table: page_table
        )
        key_cache = torch.arange(40, dtype=torch.float32).view(5, 4, 1, 2)
        value_cache = key_cache + 100

        with patch.object(
            fa_backend, "get_sharded_kv_cp_group", return_value=FakeCPGroup()
        ):
            full_k, full_v, dense_page_table = backend._gather_sharded_kv_dense(
                page_table=torch.tensor([[1, 0]], dtype=torch.int32),
                cache_seqlens=torch.tensor([8], dtype=torch.int32),
                key_cache=key_cache,
                value_cache=value_cache,
            )

        self.assertEqual(len(collective_inputs), 1)
        self.assertEqual(tuple(collective_inputs[0][0].shape), (1, 2, 4, 1, 2))
        self.assertEqual(tuple(collective_inputs[0][1].shape), (1, 2, 4, 1, 2))
        self.assertEqual(tuple(full_k.shape), (1, 2, 4, 1, 2))
        self.assertEqual(tuple(full_v.shape), (1, 2, 4, 1, 2))
        self.assertEqual(dense_page_table.tolist(), [[0, 1]])

    def test_qkv_attention_tp_shards_reconstruct_unsharded_projection(self):
        hidden_size = 4
        head_dim = 2
        total_num_heads = 4
        total_num_kv_heads = 2
        input_tensor = torch.arange(8, dtype=torch.float32).view(2, hidden_size)
        output_sizes = (
            total_num_heads * head_dim,
            total_num_kv_heads * head_dim,
            total_num_kv_heads * head_dim,
        )
        weight = torch.arange(
            sum(output_sizes) * hidden_size, dtype=torch.float32
        ).view(sum(output_sizes), hidden_size)
        bias = torch.arange(sum(output_sizes), dtype=torch.float32)

        local_outputs = []
        for attn_tp_rank in range(2):
            projection = welmv4.BaseWelmQkvProjection(
                hidden_size=hidden_size,
                head_dim=head_dim,
                total_num_heads=total_num_heads,
                total_num_kv_heads=total_num_kv_heads,
                qkv_bias=True,
                quant_config=None,
                prefix="test.qkv",
                tp_rank=attn_tp_rank,
                tp_size=2,
            )
            projection.weight_loader(projection.weight, weight)
            projection.weight_loader(projection.bias, bias)
            local_outputs.append(
                projection._apply_qkv(input_tensor).split(
                    [
                        projection.q_proj_shard_size,
                        projection.kv_proj_shard_size,
                        projection.kv_proj_shard_size,
                    ],
                    dim=-1,
                )
            )

        full_qkv = torch.nn.functional.linear(input_tensor, weight, bias)
        full_q, full_k, full_v = full_qkv.split(output_sizes, dim=-1)
        for shard_idx, expected in enumerate((full_q, full_k, full_v)):
            actual = torch.cat(
                [rank_output[shard_idx] for rank_output in local_outputs], dim=-1
            )
            torch.testing.assert_close(actual, expected)

    def test_row_parallel_attention_reduce_uses_attention_tp_group(self):
        attention_tp_group = SimpleNamespace(
            all_reduce=lambda tensor: tensor + 7,
        )
        fake_layer = SimpleNamespace(
            input_is_parallel=True,
            tp_size=2,
            tp_rank=1,
            quant_method=SimpleNamespace(
                apply=lambda layer, tensor, bias=None: tensor + 1
            ),
            bias=None,
            skip_bias_add=False,
            use_dp_attention_reduce=False,
            use_attention_tp_reduce=True,
            reduce_results=True,
        )
        input_tensor = torch.tensor([[1.0, 2.0]])

        with (
            patch.object(
                linear_module, "use_symmetric_memory", return_value=nullcontext()
            ),
            patch.object(
                linear_module,
                "get_attention_tp_group",
                return_value=attention_tp_group,
            ) as get_attention_group,
            patch.object(
                linear_module, "tensor_model_parallel_all_reduce"
            ) as global_all_reduce,
        ):
            output, output_bias = linear_module.RowParallelLinear.forward(
                fake_layer, input_tensor
            )

        torch.testing.assert_close(output, input_tensor + 8)
        self.assertIsNone(output_bias)
        get_attention_group.assert_called()
        global_all_reduce.assert_not_called()

    def test_o_projection_attention_tp_partials_reconstruct_unsharded_output(self):
        input_tensor = torch.arange(16, dtype=torch.float32).view(2, 8)
        weight = torch.arange(32, dtype=torch.float32).view(4, 8)
        bias = torch.arange(4, dtype=torch.float32)
        partial_outputs = []

        with (
            patch.object(
                linear_module, "use_symmetric_memory", return_value=nullcontext()
            ),
            patch.object(linear_module, "get_tp_group", return_value=object()),
        ):
            for attn_tp_rank, input_shard in enumerate(input_tensor.chunk(2, dim=-1)):
                fake_layer = SimpleNamespace(
                    input_is_parallel=True,
                    tp_size=2,
                    tp_rank=attn_tp_rank,
                    weight=weight[:, attn_tp_rank * 4 : (attn_tp_rank + 1) * 4],
                    quant_method=SimpleNamespace(
                        apply=lambda layer, tensor, bias=None: torch.nn.functional.linear(
                            tensor, layer.weight, bias
                        )
                    ),
                    bias=bias,
                    skip_bias_add=False,
                    use_dp_attention_reduce=False,
                    use_attention_tp_reduce=False,
                    reduce_results=False,
                )
                partial_output, _ = linear_module.RowParallelLinear.forward(
                    fake_layer, input_shard
                )
                partial_outputs.append(partial_output)

        expected = torch.nn.functional.linear(input_tensor, weight, bias)
        torch.testing.assert_close(sum(partial_outputs), expected)


class TestWelMAttentionTPBoundaries(unittest.TestCase):
    def test_moe_output_reduces_on_global_tp_not_attention_tp(self):
        fake_block = SimpleNamespace(
            layer_id=0,
            shared_expert=None,
            shared_expert_gate=None,
            gate=SimpleNamespace(weight=torch.empty((3, 2))),
            topk=lambda hidden_states, router_logits: object(),
            experts=lambda hidden_states, topk_output: hidden_states + 1,
            tp_size=8,
        )
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with (
            patch.object(welmv4, "_WELM_DUMP_ENABLED", False),
            patch.object(welmv4, "welm_use_previous_precision", return_value=False),
            patch.object(
                welmv4,
                "mmq_style_router_linear",
                return_value=torch.zeros((2, 3)),
            ),
            patch.object(
                welmv4,
                "tensor_model_parallel_all_reduce",
                side_effect=lambda tensor: tensor + 5,
            ) as global_all_reduce,
            patch.object(welmv4, "attn_tp_all_reduce") as attention_all_reduce,
        ):
            output = welmv4.Qwen2MoeSparseMoeBlock.forward(
                fake_block,
                hidden_states,
                hidden_states.float(),
                forward_batch=None,
            )

        torch.testing.assert_close(output, hidden_states + 6)
        global_all_reduce.assert_called_once()
        attention_all_reduce.assert_not_called()

    def test_logits_processor_keeps_global_vocab_group_without_dp_lm_head(self):
        server_args = SimpleNamespace(
            enable_dp_lm_head=False,
            enable_fp32_lm_head=False,
            enable_mis=False,
        )
        attention_tp_size = MagicMock(return_value=2)

        with (
            patch.object(
                logits_processor_module,
                "get_global_server_args",
                return_value=server_args,
            ),
            patch.object(
                logits_processor_module,
                "get_tensor_model_parallel_world_size",
                return_value=8,
            ),
            patch.object(
                logits_processor_module,
                "get_attention_dp_size",
                return_value=1,
            ),
            patch.object(
                logits_processor_module,
                "get_attention_tp_size",
                attention_tp_size,
            ),
        ):
            processor = logits_processor_module.LogitsProcessor(
                SimpleNamespace(vocab_size=128)
            )

        self.assertFalse(processor.use_attn_tp_group)
        self.assertTrue(processor.do_tensor_parallel_all_gather)
        self.assertFalse(processor.do_tensor_parallel_all_gather_dp_attn)
        attention_tp_size.assert_not_called()


if __name__ == "__main__":
    unittest.main()
