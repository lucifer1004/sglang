import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbeddingShardIndices
from sglang.srt.models.welm_perf_opt import (
    _can_use_specialized_welm_oe_hash_prepare,
    _can_use_specialized_welm_oe_lookup_concat,
    _compute_welm_oe_proj_reference,
    _compute_welm_oe_hashed_inputs_fused,
    _compute_welm_oe_concat_local_partials_specialized_2233,
    _compute_welm_oe_hashed_inputs_specialized_2233,
    compute_welm_oe_concat_local_partials,
    compute_welm_oe_embedding,
    get_welm_oe_implementation,
    should_use_welm_oe_triton_preprocess,
    hash_and_localize_welm_oe_input_ids,
    hash_input_ids_vectorized,
    WELM_OE_IMPL_ENV,
    WELM_OE_TRITON_PREPROCESS_ENV,
    WELM_OE_TRITON_LOOKUP_FUSION_ENV,
)
from sglang.srt.models.welmv4 import Qwen2MoeModel


class DummyOEContext:
    def __init__(self, grams):
        self.grams = grams

    def get_gram(self, n: int):
        return self.grams.get(n)


class RecordingEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.last_ids = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self.last_ids = ids.clone()
        return F.embedding(ids.long(), self.weight)


class FailingModule(nn.Module):
    def forward(self, *_args, **_kwargs):
        raise AssertionError("default OE modules should not be used in this test")


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


class ForwardableShardedEmbedding(FakeShardedEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_ids = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self.last_ids = ids.clone()
        return F.embedding(ids.long(), self.weight)


def manual_oe_reference(
    *,
    input_ids: torch.Tensor,
    forward_batch,
    base_hidden_states: torch.Tensor,
    vocab_size: int,
    oe_grams,
    oe_vocab_sizes,
    oe_embed_modules,
    oe_proj_module,
) -> torch.Tensor:
    input_ids_ngram = []
    input_ids_ngram_tmp = input_ids
    for g in range(1, max(oe_grams)):
        gram_tensor = forward_batch.oe_context.get_gram(g + 1)
        if gram_tensor is not None:
            input_ids_ngram_tmp = input_ids_ngram_tmp + gram_tensor * (vocab_size**g)
        input_ids_ngram.append(hash_input_ids_vectorized(input_ids_ngram_tmp))

    emb_ngram = []
    for i, vs in enumerate(oe_vocab_sizes):
        hashed = input_ids_ngram[oe_grams[i] - 2] % vs
        emb_ngram.append(oe_embed_modules[i](hashed))
    emb_new, _ = oe_proj_module(torch.cat(emb_ngram, dim=-1))
    return (base_hidden_states + emb_new) / 2.0


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
        self.embed_modules = nn.ModuleList(
            [
                RecordingEmbedding(self.full_weight_0),
                RecordingEmbedding(self.full_weight_1),
            ]
        )
        proj_weight = torch.arange(36, dtype=torch.float32).reshape(6, 6) / 13.0
        proj_bias = torch.arange(6, dtype=torch.float32) / 17.0
        self.proj_module = FakeProjModule(proj_weight, proj_bias)

    def test_compute_oe_embedding_hashes_selected_ngrams_and_averages_with_base(self):
        model = SimpleNamespace(
            vocab_size=self.vocab_size,
            oe_grams=self.oe_grams,
            oe_vocab_sizes=self.oe_vocab_sizes,
            oe_embed=self.embed_modules,
            oe_gate_up_proj=self.proj_module,
        )

        out = Qwen2MoeModel._compute_oe_embedding(
            model,
            self.input_ids,
            self.forward_batch,
            self.base_hidden_states,
        )
        expected = manual_oe_reference(
            input_ids=self.input_ids,
            forward_batch=self.forward_batch,
            base_hidden_states=self.base_hidden_states,
            vocab_size=self.vocab_size,
            oe_grams=self.oe_grams,
            oe_vocab_sizes=self.oe_vocab_sizes,
            oe_embed_modules=self.embed_modules,
            oe_proj_module=self.proj_module,
        )

        ngram2 = hash_input_ids_vectorized(
            self.input_ids + self.forward_batch.oe_context.get_gram(2) * self.vocab_size
        ) % self.oe_vocab_sizes[0]
        ngram4 = hash_input_ids_vectorized(
            self.input_ids
            + self.forward_batch.oe_context.get_gram(2) * self.vocab_size
            + self.forward_batch.oe_context.get_gram(3) * (self.vocab_size**2)
            + self.forward_batch.oe_context.get_gram(4) * (self.vocab_size**3)
        ) % self.oe_vocab_sizes[1]

        torch.testing.assert_close(self.embed_modules[0].last_ids, ngram2)
        torch.testing.assert_close(self.embed_modules[1].last_ids, ngram4)
        torch.testing.assert_close(out, expected)

    def test_compute_oe_embedding_uses_override_modules_for_scale_seq_path(self):
        override_embed_modules = nn.ModuleList(
            [
                RecordingEmbedding(self.full_weight_0 + 10),
                RecordingEmbedding(self.full_weight_1 + 20),
            ]
        )
        override_proj = FakeProjModule(self.proj_module.weight + 1.5, self.proj_module.bias)

        model = SimpleNamespace(
            vocab_size=self.vocab_size,
            oe_grams=self.oe_grams,
            oe_vocab_sizes=self.oe_vocab_sizes,
            oe_embed=nn.ModuleList([FailingModule(), FailingModule()]),
            oe_gate_up_proj=FailingModule(),
        )

        out = Qwen2MoeModel._compute_oe_embedding(
            model,
            self.input_ids,
            self.forward_batch,
            self.base_hidden_states,
            oe_embed_modules=override_embed_modules,
            oe_up_proj_module=override_proj,
        )
        expected = manual_oe_reference(
            input_ids=self.input_ids,
            forward_batch=self.forward_batch,
            base_hidden_states=self.base_hidden_states,
            vocab_size=self.vocab_size,
            oe_grams=self.oe_grams,
            oe_vocab_sizes=self.oe_vocab_sizes,
            oe_embed_modules=override_embed_modules,
            oe_proj_module=override_proj,
        )

        torch.testing.assert_close(out, expected)
        self.assertIsNotNone(override_embed_modules[0].last_ids)
        self.assertIsNotNone(override_embed_modules[1].last_ids)

    def test_tp_sharded_oe_embeddings_sum_to_reference_proj_output(self):
        rank0_modules = [
            FakeShardedEmbedding(
                local_weight=self.full_weight_0[:6],
                vocab_start=0,
                vocab_end=6,
                padded_end=6,
                tp_size=2,
            ),
            FakeShardedEmbedding(
                local_weight=self.full_weight_1[:7],
                vocab_start=0,
                vocab_end=7,
                padded_end=7,
                tp_size=2,
            ),
        ]
        rank1_modules = [
            FakeShardedEmbedding(
                local_weight=self.full_weight_0[6:],
                vocab_start=6,
                vocab_end=11,
                padded_end=11,
                tp_size=2,
            ),
            FakeShardedEmbedding(
                local_weight=self.full_weight_1[7:],
                vocab_start=7,
                vocab_end=13,
                padded_end=13,
                tp_size=2,
            ),
        ]
        full_modules = nn.ModuleList(
            [RecordingEmbedding(self.full_weight_0), RecordingEmbedding(self.full_weight_1)]
        )

        expected_proj = _compute_welm_oe_proj_reference(
            input_ids=self.input_ids,
            forward_batch=self.forward_batch,
            oe_grams=self.oe_grams,
            oe_vocab_sizes=self.oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=full_modules,
            oe_proj_module=self.proj_module,
        )
        expected_proj_no_bias = expected_proj - self.proj_module.bias

        concat_rank0 = compute_welm_oe_concat_local_partials(
            input_ids=self.input_ids,
            forward_batch=self.forward_batch,
            oe_grams=self.oe_grams,
            oe_vocab_sizes=self.oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=rank0_modules,
            use_triton_preprocess=False,
        )
        concat_rank1 = compute_welm_oe_concat_local_partials(
            input_ids=self.input_ids,
            forward_batch=self.forward_batch,
            oe_grams=self.oe_grams,
            oe_vocab_sizes=self.oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=rank1_modules,
            use_triton_preprocess=False,
        )
        expected_concat = torch.cat(
            [
                full_modules[0](
                    hash_input_ids_vectorized(
                        self.input_ids
                        + self.forward_batch.oe_context.get_gram(2) * self.vocab_size
                    )
                    % self.oe_vocab_sizes[0]
                ),
                full_modules[1](
                    hash_input_ids_vectorized(
                        self.input_ids
                        + self.forward_batch.oe_context.get_gram(2) * self.vocab_size
                        + self.forward_batch.oe_context.get_gram(3)
                        * (self.vocab_size**2)
                        + self.forward_batch.oe_context.get_gram(4)
                        * (self.vocab_size**3)
                    )
                    % self.oe_vocab_sizes[1]
                ),
            ],
            dim=-1,
        )
        torch.testing.assert_close(concat_rank0 + concat_rank1, expected_concat)
        torch.testing.assert_close(
            F.linear(concat_rank0 + concat_rank1, self.proj_module.weight, bias=None),
            expected_proj_no_bias,
        )

        expected = (self.base_hidden_states + expected_proj) / 2.0
        actual = compute_welm_oe_embedding(
            input_ids=self.input_ids,
            forward_batch=self.forward_batch,
            base_hidden_states=self.base_hidden_states,
            oe_grams=self.oe_grams,
            oe_vocab_sizes=self.oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=rank0_modules,
            oe_proj_module=self.proj_module,
            implementation="tp_fused",
            use_triton_preprocess=False,
            all_reduce_fn=lambda x: x + concat_rank1,
        )
        torch.testing.assert_close(actual, expected)

    def test_hash_and_localize_triton_matches_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to validate the Triton preprocess path")

        input_ids = torch.tensor([1, 17, 33, 49, 65], device="cuda", dtype=torch.int64)
        hashed_ref, local_ref, mask_ref = hash_and_localize_welm_oe_input_ids(
            input_ids,
            vocab_size=13,
            shard_start=5,
            shard_end=10,
            use_triton=False,
        )
        hashed_tri, local_tri, mask_tri = hash_and_localize_welm_oe_input_ids(
            input_ids,
            vocab_size=13,
            shard_start=5,
            shard_end=10,
            use_triton=True,
        )

        torch.testing.assert_close(hashed_tri.cpu(), hashed_ref.cpu())
        torch.testing.assert_close(local_tri.cpu(), local_ref.cpu())
        torch.testing.assert_close(mask_tri.cpu(), mask_ref.cpu())

    def test_specialized_hash_prepare_dispatch_for_2233_shape(self):
        rank0_modules = [
            FakeShardedEmbedding(
                local_weight=self.full_weight_0,
                vocab_start=0,
                vocab_end=11,
                padded_end=11,
                tp_size=1,
            )
            for _ in range(4)
        ]

        self.assertFalse(
            _can_use_specialized_welm_oe_hash_prepare(
                self.input_ids,
                [2, 2, 3, 3],
                [11, 13, 17, 19],
                rank0_modules,
                use_triton_preprocess=True,
            )
        )

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to validate specialized Triton dispatch")

        input_ids = self.input_ids.cuda()
        oe_context = DummyOEContext(
            {
                2: self.forward_batch.oe_context.get_gram(2).cuda(),
                3: self.forward_batch.oe_context.get_gram(3).cuda(),
            }
        )
        oe_vocab_sizes = [11, 7, 13, 17]
        modules = [
            FakeShardedEmbedding(
                local_weight=torch.randn(vs, 3, device="cuda"),
                vocab_start=0,
                vocab_end=vs,
                padded_end=vs,
                tp_size=1,
            )
            for vs in oe_vocab_sizes
        ]

        self.assertTrue(
            _can_use_specialized_welm_oe_hash_prepare(
                input_ids,
                [2, 2, 3, 3],
                oe_vocab_sizes,
                modules,
                use_triton_preprocess=True,
            )
        )

        generic = _compute_welm_oe_hashed_inputs_fused(
            input_ids=input_ids,
            oe_context=oe_context,
            oe_grams=[2, 2, 3, 3],
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=modules,
            use_triton_preprocess=False,
        )
        specialized = _compute_welm_oe_hashed_inputs_specialized_2233(
            input_ids=input_ids,
            oe_context=oe_context,
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=self.vocab_size,
        )

        for got, expected in zip(specialized, generic):
            torch.testing.assert_close(got.cpu().to(expected.dtype), expected.cpu())

    def test_specialized_hash_prepare_handles_missing_grams_like_generic(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to validate specialized Triton dispatch")

        input_ids = self.input_ids.cuda()
        oe_vocab_sizes = [11, 7, 13, 17]
        modules = [
            FakeShardedEmbedding(
                local_weight=torch.randn(vs, 3, device="cuda"),
                vocab_start=0,
                vocab_end=vs,
                padded_end=vs,
                tp_size=1,
            )
            for vs in oe_vocab_sizes
        ]
        oe_context = DummyOEContext({2: self.forward_batch.oe_context.get_gram(2).cuda()})

        generic = _compute_welm_oe_hashed_inputs_fused(
            input_ids=input_ids,
            oe_context=oe_context,
            oe_grams=[2, 2, 3, 3],
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=modules,
            use_triton_preprocess=False,
        )
        specialized = _compute_welm_oe_hashed_inputs_specialized_2233(
            input_ids=input_ids,
            oe_context=oe_context,
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=self.vocab_size,
        )

        for got, expected in zip(specialized, generic):
            torch.testing.assert_close(got.cpu().to(expected.dtype), expected.cpu())

    def test_specialized_hash_prepare_matches_generic_for_large_vocab_size(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to validate specialized Triton dispatch")

        vocab_size = 200000
        input_ids = torch.tensor([12345, 54321, 100001, 199999], device="cuda", dtype=torch.int64)
        oe_context = DummyOEContext(
            {
                2: torch.tensor([111, 222, 333, 444], device="cuda", dtype=torch.int64),
                3: torch.tensor([555, 666, 777, 888], device="cuda", dtype=torch.int64),
            }
        )
        oe_vocab_sizes = [16000008, 16000016, 16000024, 16000032]
        modules = [
            FakeShardedEmbedding(
                local_weight=torch.randn(min(vs, 4096), 512, device="cuda", dtype=torch.bfloat16),
                vocab_start=0,
                vocab_end=min(vs, 4096),
                padded_end=min(vs, 4096),
                tp_size=1,
            )
            for vs in oe_vocab_sizes
        ]

        generic = _compute_welm_oe_hashed_inputs_fused(
            input_ids=input_ids,
            oe_context=oe_context,
            oe_grams=[2, 2, 3, 3],
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=vocab_size,
            oe_embed_modules=modules,
            use_triton_preprocess=False,
        )
        specialized = _compute_welm_oe_hashed_inputs_specialized_2233(
            input_ids=input_ids,
            oe_context=oe_context,
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=vocab_size,
        )

        for got, expected in zip(specialized, generic):
            torch.testing.assert_close(got.cpu().to(expected.dtype), expected.cpu())

    def test_specialized_lookup_concat_2233_matches_generic_concat(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to validate specialized Triton dispatch")

        input_ids = self.input_ids.cuda()
        oe_context = DummyOEContext(
            {
                2: self.forward_batch.oe_context.get_gram(2).cuda(),
                3: self.forward_batch.oe_context.get_gram(3).cuda(),
            }
        )
        oe_vocab_sizes = [11, 7, 13, 17]
        modules = [
            FakeShardedEmbedding(
                local_weight=torch.randn(vs, 512, device="cuda", dtype=torch.bfloat16),
                vocab_start=0,
                vocab_end=vs,
                padded_end=vs,
                tp_size=1,
            )
            for vs in oe_vocab_sizes
        ]

        with patch.dict(
            os.environ, {WELM_OE_TRITON_LOOKUP_FUSION_ENV: "1"}, clear=False
        ):
            self.assertTrue(
                _can_use_specialized_welm_oe_lookup_concat(
                    input_ids,
                    [2, 2, 3, 3],
                    oe_vocab_sizes,
                    modules,
                    use_triton_preprocess=True,
                )
            )

            generic = compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=SimpleNamespace(oe_context=oe_context),
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=modules,
                use_triton_preprocess=False,
            )
            specialized = _compute_welm_oe_concat_local_partials_specialized_2233(
                input_ids=input_ids,
                oe_context=oe_context,
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=modules,
            )
        torch.testing.assert_close(specialized, generic)

    def test_specialized_lookup_concat_2233_handles_missing_gram3_like_generic(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to validate specialized Triton dispatch")

        input_ids = self.input_ids.cuda()
        oe_context = DummyOEContext({2: self.forward_batch.oe_context.get_gram(2).cuda()})
        oe_vocab_sizes = [11, 7, 13, 17]
        modules = [
            FakeShardedEmbedding(
                local_weight=torch.randn(vs, 512, device="cuda", dtype=torch.bfloat16),
                vocab_start=0,
                vocab_end=vs,
                padded_end=vs,
                tp_size=1,
            )
            for vs in oe_vocab_sizes
        ]

        with patch.dict(
            os.environ, {WELM_OE_TRITON_LOOKUP_FUSION_ENV: "1"}, clear=False
        ):
            generic = compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=SimpleNamespace(oe_context=oe_context),
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=modules,
                use_triton_preprocess=False,
            )
            specialized = _compute_welm_oe_concat_local_partials_specialized_2233(
                input_ids=input_ids,
                oe_context=oe_context,
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=modules,
            )
        torch.testing.assert_close(specialized, generic)

    def test_specialized_tp_fused_2233_matches_legacy_end_to_end(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to validate specialized Triton dispatch")

        input_ids = self.input_ids.cuda()
        base_hidden_states = torch.randn(
            input_ids.shape[0], 6, device="cuda", dtype=torch.float32
        )
        oe_context = DummyOEContext(
            {
                2: self.forward_batch.oe_context.get_gram(2).cuda(),
                3: self.forward_batch.oe_context.get_gram(3).cuda(),
            }
        )
        oe_vocab_sizes = [11, 7, 13, 17]
        modules = [
            ForwardableShardedEmbedding(
                local_weight=torch.randn(vs, 512, device="cuda", dtype=torch.float32),
                vocab_start=0,
                vocab_end=vs,
                padded_end=vs,
                tp_size=1,
            )
            for vs in oe_vocab_sizes
        ]
        proj_weight = torch.randn(6, 2048, device="cuda", dtype=torch.float32)
        proj_bias = torch.randn(6, device="cuda", dtype=torch.float32)
        proj_module = FakeProjModule(proj_weight, proj_bias)

        legacy = compute_welm_oe_embedding(
            input_ids=input_ids,
            forward_batch=SimpleNamespace(oe_context=oe_context),
            base_hidden_states=base_hidden_states,
            oe_grams=[2, 2, 3, 3],
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=modules,
            oe_proj_module=proj_module,
            implementation="legacy",
            use_triton_preprocess=False,
        )
        with patch.dict(
            os.environ, {WELM_OE_TRITON_LOOKUP_FUSION_ENV: "1"}, clear=False
        ):
            fused = compute_welm_oe_embedding(
                input_ids=input_ids,
                forward_batch=SimpleNamespace(oe_context=oe_context),
                base_hidden_states=base_hidden_states,
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=modules,
                oe_proj_module=proj_module,
                implementation="tp_fused",
                use_triton_preprocess=True,
            )
        torch.testing.assert_close(fused, legacy, atol=1e-3, rtol=1e-3)

    def test_specialized_tp_fused_2233_sharded_sum_matches_legacy(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to validate specialized Triton dispatch")

        input_ids = self.input_ids.cuda()
        base_hidden_states = torch.randn(
            input_ids.shape[0], 6, device="cuda", dtype=torch.float32
        )
        oe_context = DummyOEContext(
            {
                2: self.forward_batch.oe_context.get_gram(2).cuda(),
                3: self.forward_batch.oe_context.get_gram(3).cuda(),
            }
        )
        oe_vocab_sizes = [11, 7, 13, 17]
        full_modules = [
            ForwardableShardedEmbedding(
                local_weight=torch.randn(vs, 512, device="cuda", dtype=torch.float32),
                vocab_start=0,
                vocab_end=vs,
                padded_end=vs,
                tp_size=1,
            )
            for vs in oe_vocab_sizes
        ]
        rank0_modules = [
            ForwardableShardedEmbedding(
                local_weight=module.weight[: (module.weight.shape[0] + 1) // 2],
                vocab_start=0,
                vocab_end=(module.weight.shape[0] + 1) // 2,
                padded_end=(module.weight.shape[0] + 1) // 2,
                tp_size=2,
            )
            for module in full_modules
        ]
        rank1_modules = [
            ForwardableShardedEmbedding(
                local_weight=module.weight[(module.weight.shape[0] + 1) // 2 :],
                vocab_start=(module.weight.shape[0] + 1) // 2,
                vocab_end=module.weight.shape[0],
                padded_end=module.weight.shape[0],
                tp_size=2,
            )
            for module in full_modules
        ]
        proj_weight = torch.randn(6, 2048, device="cuda", dtype=torch.float32)
        proj_bias = torch.randn(6, device="cuda", dtype=torch.float32)
        proj_module = FakeProjModule(proj_weight, proj_bias)

        legacy = compute_welm_oe_embedding(
            input_ids=input_ids,
            forward_batch=SimpleNamespace(oe_context=oe_context),
            base_hidden_states=base_hidden_states,
            oe_grams=[2, 2, 3, 3],
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=full_modules,
            oe_proj_module=proj_module,
            implementation="legacy",
            use_triton_preprocess=False,
        )

        with patch.dict(
            os.environ, {WELM_OE_TRITON_LOOKUP_FUSION_ENV: "1"}, clear=False
        ):
            rank0_concat = compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=SimpleNamespace(oe_context=oe_context),
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank0_modules,
                use_triton_preprocess=True,
            )
            rank1_concat = compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=SimpleNamespace(oe_context=oe_context),
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank1_modules,
                use_triton_preprocess=True,
            )
            fused = compute_welm_oe_embedding(
                input_ids=input_ids,
                forward_batch=SimpleNamespace(oe_context=oe_context),
                base_hidden_states=base_hidden_states,
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank0_modules,
                oe_proj_module=proj_module,
                implementation="tp_fused",
                use_triton_preprocess=True,
                all_reduce_fn=lambda x: x + rank1_concat,
            )

        torch.testing.assert_close(
            rank0_concat + rank1_concat,
            compute_welm_oe_concat_local_partials(
                input_ids=input_ids,
                forward_batch=SimpleNamespace(oe_context=oe_context),
                oe_grams=[2, 2, 3, 3],
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=full_modules,
                use_triton_preprocess=False,
            ),
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(fused, legacy, atol=1e-3, rtol=1e-3)

    def test_concat_local_partials_supports_repeated_oe_grams(self):
        oe_grams = [2, 2, 4]
        oe_vocab_sizes = [11, 7, 13]
        embed_modules = [
            FakeShardedEmbedding(
                local_weight=self.full_weight_0,
                vocab_start=0,
                vocab_end=11,
                padded_end=11,
                tp_size=1,
            ),
            FakeShardedEmbedding(
                local_weight=torch.arange(21, dtype=torch.float32).reshape(7, 3) / 5.0,
                vocab_start=0,
                vocab_end=7,
                padded_end=7,
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

        actual = compute_welm_oe_concat_local_partials(
            input_ids=self.input_ids,
            forward_batch=self.forward_batch,
            oe_grams=oe_grams,
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=embed_modules,
            use_triton_preprocess=False,
        )

        ngram2 = hash_input_ids_vectorized(
            self.input_ids + self.forward_batch.oe_context.get_gram(2) * self.vocab_size
        )
        ngram4 = hash_input_ids_vectorized(
            self.input_ids
            + self.forward_batch.oe_context.get_gram(2) * self.vocab_size
            + self.forward_batch.oe_context.get_gram(3) * (self.vocab_size**2)
            + self.forward_batch.oe_context.get_gram(4) * (self.vocab_size**3)
        )
        expected = torch.cat(
            [
                F.embedding((ngram2 % oe_vocab_sizes[0]).long(), embed_modules[0].weight),
                F.embedding((ngram2 % oe_vocab_sizes[1]).long(), embed_modules[1].weight),
                F.embedding((ngram4 % oe_vocab_sizes[2]).long(), embed_modules[2].weight),
            ],
            dim=-1,
        )
        torch.testing.assert_close(actual, expected)

    def test_env_var_selects_legacy_and_tp_fused_paths(self):
        rank0_modules = [
            FakeShardedEmbedding(
                local_weight=self.full_weight_0[:6],
                vocab_start=0,
                vocab_end=6,
                padded_end=6,
                tp_size=2,
            ),
            FakeShardedEmbedding(
                local_weight=self.full_weight_1[:7],
                vocab_start=0,
                vocab_end=7,
                padded_end=7,
                tp_size=2,
            ),
        ]
        rank1_modules = [
            FakeShardedEmbedding(
                local_weight=self.full_weight_0[6:],
                vocab_start=6,
                vocab_end=11,
                padded_end=11,
                tp_size=2,
            ),
            FakeShardedEmbedding(
                local_weight=self.full_weight_1[7:],
                vocab_start=7,
                vocab_end=13,
                padded_end=13,
                tp_size=2,
            ),
        ]
        concat_rank1 = compute_welm_oe_concat_local_partials(
            input_ids=self.input_ids,
            forward_batch=self.forward_batch,
            oe_grams=self.oe_grams,
            oe_vocab_sizes=self.oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=rank1_modules,
            use_triton_preprocess=False,
        )

        with patch.dict(os.environ, {WELM_OE_IMPL_ENV: "legacy"}, clear=False):
            legacy_out = compute_welm_oe_embedding(
                input_ids=self.input_ids,
                forward_batch=self.forward_batch,
                base_hidden_states=self.base_hidden_states,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=self.oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=self.embed_modules,
                oe_proj_module=self.proj_module,
                use_triton_preprocess=False,
            )

        with patch.dict(os.environ, {WELM_OE_IMPL_ENV: "tp_fused"}, clear=False):
            fused_out = compute_welm_oe_embedding(
                input_ids=self.input_ids,
                forward_batch=self.forward_batch,
                base_hidden_states=self.base_hidden_states,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=self.oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank0_modules,
                oe_proj_module=self.proj_module,
                use_triton_preprocess=False,
                all_reduce_fn=lambda x: x + concat_rank1,
            )

        torch.testing.assert_close(fused_out, legacy_out)
        self.assertEqual(get_welm_oe_implementation("legacy"), "legacy")
        self.assertEqual(get_welm_oe_implementation("tp_fused"), "tp_fused")
        self.assertEqual(get_welm_oe_implementation("bad-value"), "legacy")
        self.assertFalse(should_use_welm_oe_triton_preprocess(False))
        with patch.dict(
            os.environ, {WELM_OE_TRITON_PREPROCESS_ENV: "1"}, clear=False
        ):
            self.assertTrue(should_use_welm_oe_triton_preprocess())


if __name__ == "__main__":
    unittest.main()
