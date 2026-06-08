import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbeddingShardIndices
from sglang.srt.models.welm_perf_opt import (
    _can_use_specialized_welm_oe_prehashed_lookup_concat,
    _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512,
    WELM_OE_IMPL_ENV,
    WELM_OE_POST_PROJ_ALL_REDUCE_ENV,
    WELM_OE_TRITON_PREPROCESS_ENV,
    compute_welm_oe_concat_local_partials,
    compute_welm_oe_embedding,
    get_welm_oe_implementation,
    hash_and_localize_welm_oe_input_ids,
    hash_input_ids_vectorized,
    should_use_welm_oe_triton_preprocess,
)


class DummyOEContext:
    def __init__(self, grams=None):
        self.grams = grams or {}

    def get_gram(self, n: int):
        return self.grams.get(n)


class FakeProjModule(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(bias.clone(), requires_grad=False) if bias is not None else None
        )

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias), None


class FakeShardedEmbedding(nn.Module):
    def __init__(
        self,
        *,
        local_weight: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
        tp_size: int,
        padded_end: int | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(local_weight.clone(), requires_grad=False)
        self.tp_size = tp_size
        padded_end = padded_end or vocab_end
        self.shard_indices = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=vocab_start,
            padded_org_vocab_end_index=padded_end,
            padded_added_vocab_start_index=padded_end,
            padded_added_vocab_end_index=padded_end,
            org_vocab_start_index=vocab_start,
            org_vocab_end_index=vocab_end,
            added_vocab_start_index=vocab_end,
            added_vocab_end_index=vocab_end,
        )


class TestWelmV4OEEmbedding(unittest.TestCase):
    def setUp(self):
        self.input_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int64)
        self.forward_batch = SimpleNamespace(
            oe_context=DummyOEContext(
                {
                    2: torch.tensor([9, 8, 7, 6], dtype=torch.int64),
                    3: torch.tensor([4, 3, 2, 1], dtype=torch.int64),
                    4: torch.tensor([5, 6, 7, 8], dtype=torch.int64),
                }
            )
        )
        self.base_hidden_states = (
            torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10.0
        )
        self.vocab_size = 17
        self.oe_grams = [2, 4]
        self.oe_vocab_sizes = [11, 13]
        self.full_weight_0 = torch.arange(33, dtype=torch.float32).reshape(11, 3) / 7.0
        self.full_weight_1 = (
            torch.arange(39, dtype=torch.float32).reshape(13, 3) / 11.0
        )
        proj_weight = torch.arange(36, dtype=torch.float32).reshape(6, 6) / 13.0
        proj_bias = torch.arange(6, dtype=torch.float32) / 17.0
        self.proj_module = FakeProjModule(proj_weight, proj_bias)

    def test_missing_hash_inputs_rejects_materialized_oe_context(self):
        modules = [
            FakeShardedEmbedding(
                local_weight=self.full_weight_0,
                vocab_start=0,
                vocab_end=11,
                padded_end=11,
                tp_size=1,
            ),
            FakeShardedEmbedding(
                local_weight=self.full_weight_1,
                vocab_start=0,
                vocab_end=13,
                padded_end=13,
                tp_size=1,
            ),
        ]

        with self.assertRaisesRegex(RuntimeError, "Materialized n-gram fallback"):
            compute_welm_oe_concat_local_partials(
                input_ids=self.input_ids,
                forward_batch=self.forward_batch,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=self.oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=modules,
                use_triton_preprocess=False,
            )

    def test_compute_embedding_rejects_legacy_implementation(self):
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            compute_welm_oe_embedding(
                input_ids=self.input_ids,
                forward_batch=self.forward_batch,
                base_hidden_states=self.base_hidden_states,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=self.oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=[],
                oe_proj_module=self.proj_module,
                implementation="legacy",
            )

    def test_hash_and_localize_matches_reference_formula(self):
        input_ids = torch.tensor([1, 17, 33, 49, 65], dtype=torch.int64)
        hashed, local_idx, valid_mask = hash_and_localize_welm_oe_input_ids(
            input_ids,
            vocab_size=13,
            shard_start=5,
            shard_end=10,
        )

        expected_hashed = hash_input_ids_vectorized(input_ids.to(torch.int64)) % 13
        expected_mask = (expected_hashed >= 5) & (expected_hashed < 10)
        expected_local = torch.where(
            expected_mask, expected_hashed - 5, torch.zeros_like(expected_hashed)
        )

        torch.testing.assert_close(hashed, expected_hashed.to(torch.int64))
        torch.testing.assert_close(local_idx, expected_local.to(torch.int64))
        torch.testing.assert_close(valid_mask, expected_mask)

    def test_cached_hash_embedding_matches_manual_tp_sum(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the OE hash kernel path")

        torch.manual_seed(0)
        input_ids = self.input_ids.cuda()
        base_hidden_states = torch.randn(
            input_ids.shape[0], 6, device="cuda", dtype=torch.float32
        )
        oe_vocab_sizes = [11, 13]
        cached_hash = torch.stack(
            [
                torch.tensor([1, 6, 9, 10], device="cuda", dtype=torch.int64),
                torch.tensor([0, 7, 12, 5], device="cuda", dtype=torch.int64),
            ]
        )
        forward_batch = SimpleNamespace(
            oe_context=None,
            welm_oe_decode_hashed_inputs=cached_hash,
        )

        full_modules = [
            FakeShardedEmbedding(
                local_weight=self.full_weight_0.cuda(),
                vocab_start=0,
                vocab_end=11,
                padded_end=11,
                tp_size=1,
            ),
            FakeShardedEmbedding(
                local_weight=self.full_weight_1.cuda(),
                vocab_start=0,
                vocab_end=13,
                padded_end=13,
                tp_size=1,
            ),
        ]
        rank0_modules = [
            FakeShardedEmbedding(
                local_weight=self.full_weight_0[:6].cuda(),
                vocab_start=0,
                vocab_end=6,
                padded_end=6,
                tp_size=2,
            ),
            FakeShardedEmbedding(
                local_weight=self.full_weight_1[:7].cuda(),
                vocab_start=0,
                vocab_end=7,
                padded_end=7,
                tp_size=2,
            ),
        ]
        rank1_modules = [
            FakeShardedEmbedding(
                local_weight=self.full_weight_0[6:].cuda(),
                vocab_start=6,
                vocab_end=11,
                padded_end=11,
                tp_size=2,
            ),
            FakeShardedEmbedding(
                local_weight=self.full_weight_1[7:].cuda(),
                vocab_start=7,
                vocab_end=13,
                padded_end=13,
                tp_size=2,
            ),
        ]
        proj_module = FakeProjModule(
            self.proj_module.weight.cuda(),
            self.proj_module.bias.cuda(),
        )

        expected_concat = torch.cat(
            [
                F.embedding(cached_hash[0], full_modules[0].weight),
                F.embedding(cached_hash[1], full_modules[1].weight),
            ],
            dim=-1,
        )
        expected_proj = F.linear(expected_concat, proj_module.weight, proj_module.bias)
        expected = (base_hidden_states + expected_proj) / 2.0

        with patch.dict(os.environ, {WELM_OE_IMPL_ENV: "fused_ngram_hash"}, clear=False):
            rank1_concat = compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=forward_batch,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank1_modules,
                use_triton_preprocess=False,
            )
            actual = compute_welm_oe_embedding(
                input_ids=input_ids,
                forward_batch=forward_batch,
                base_hidden_states=base_hidden_states,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank0_modules,
                oe_proj_module=proj_module,
                use_triton_preprocess=False,
                all_reduce_fn=lambda x: x + rank1_concat,
            )

        torch.testing.assert_close(actual, expected)

        rank1_proj_no_bias = F.linear(rank1_concat, proj_module.weight, bias=None)
        with patch.dict(
            os.environ,
            {
                WELM_OE_IMPL_ENV: "fused_ngram_hash",
                WELM_OE_POST_PROJ_ALL_REDUCE_ENV: "1",
            },
            clear=False,
        ):
            actual_post_proj = compute_welm_oe_embedding(
                input_ids=input_ids,
                forward_batch=forward_batch,
                base_hidden_states=base_hidden_states,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank0_modules,
                oe_proj_module=proj_module,
                use_triton_preprocess=False,
                all_reduce_fn=lambda x: x + rank1_proj_no_bias,
            )
        torch.testing.assert_close(actual_post_proj, expected)

    def test_specialized_prehashed_lookup_concat_matches_generic_hash_path(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to validate specialized Triton dispatch")

        torch.manual_seed(1)
        input_ids = self.input_ids.cuda()
        oe_vocab_sizes = [11, 7, 13, 17]
        full_modules = [
            FakeShardedEmbedding(
                local_weight=torch.randn(vs, 512, device="cuda", dtype=torch.float32),
                vocab_start=0,
                vocab_end=vs,
                padded_end=vs,
                tp_size=1,
            )
            for vs in oe_vocab_sizes
        ]
        rank0_modules = [
            FakeShardedEmbedding(
                local_weight=module.weight[: (module.weight.shape[0] + 1) // 2],
                vocab_start=0,
                vocab_end=(module.weight.shape[0] + 1) // 2,
                padded_end=(module.weight.shape[0] + 1) // 2,
                tp_size=2,
            )
            for module in full_modules
        ]
        rank1_modules = [
            FakeShardedEmbedding(
                local_weight=module.weight[(module.weight.shape[0] + 1) // 2 :],
                vocab_start=(module.weight.shape[0] + 1) // 2,
                vocab_end=module.weight.shape[0],
                padded_end=module.weight.shape[0],
                tp_size=2,
            )
            for module in full_modules
        ]
        hashed_inputs = [
            torch.randint(0, vs, (input_ids.numel(),), device="cuda", dtype=torch.int64)
            for vs in oe_vocab_sizes
        ]
        cached_hash = torch.stack(hashed_inputs)
        forward_batch = SimpleNamespace(
            oe_context=None,
            welm_oe_decode_hashed_inputs=cached_hash,
        )

        self.assertTrue(
            _can_use_specialized_welm_oe_prehashed_lookup_concat(
                hashed_inputs,
                rank0_modules,
                use_triton_preprocess=True,
            )
        )

        with patch.dict(os.environ, {WELM_OE_IMPL_ENV: "fused_ngram_hash"}, clear=False):
            generic_rank0 = compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=forward_batch,
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank0_modules,
                use_triton_preprocess=False,
            )
            fused_rank0 = compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=forward_batch,
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank0_modules,
                use_triton_preprocess=True,
            )
            generic_rank1 = compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=forward_batch,
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank1_modules,
                use_triton_preprocess=False,
            )
            fused_rank1 = compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=forward_batch,
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank1_modules,
                use_triton_preprocess=True,
            )
            generic_full = compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=forward_batch,
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=full_modules,
                use_triton_preprocess=False,
            )

        direct_rank0 = _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512(
            hashed_inputs=hashed_inputs,
            oe_embed_modules=rank0_modules,
        )
        torch.testing.assert_close(fused_rank0, generic_rank0)
        torch.testing.assert_close(fused_rank1, generic_rank1)
        torch.testing.assert_close(direct_rank0, generic_rank0)
        torch.testing.assert_close(fused_rank0 + fused_rank1, generic_full)

    def test_env_var_defaults_to_hash_and_rejects_fallbacks(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(WELM_OE_IMPL_ENV, None)
            self.assertEqual(get_welm_oe_implementation(None), "fused_ngram_hash")

        for value in (
            "legacy",
            "reference",
            "old",
            "tp_fused",
            "fused",
            "new",
            "optimized",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "no longer supported"):
                    get_welm_oe_implementation(value)

        self.assertEqual(
            get_welm_oe_implementation("fused_ngram_hash"), "fused_ngram_hash"
        )
        with self.assertRaisesRegex(ValueError, "invalid"):
            get_welm_oe_implementation("bad-value")
        self.assertFalse(should_use_welm_oe_triton_preprocess(False))
        with patch.dict(
            os.environ, {WELM_OE_TRITON_PREPROCESS_ENV: "1"}, clear=False
        ):
            self.assertTrue(should_use_welm_oe_triton_preprocess())


if __name__ == "__main__":
    unittest.main()
