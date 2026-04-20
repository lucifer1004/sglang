import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbeddingShardIndices
from sglang.srt.models.welm_perf_opt import (
    DEFAULT_SPECIALIZED_WELM_OE_EMBED_BLOCK_D,
    DEFAULT_SPECIALIZED_WELM_OE_EMBED_NUM_WARPS,
    _can_use_specialized_welm_oe_lookup_concat,
    _compute_welm_oe_proj_reference,
    _compute_welm_oe_concat_local_partials_specialized_2233,
    _welm_oe_lookup_concat_2233_kernel,
    compute_welm_oe_concat_local_partials,
    compute_welm_oe_embedding,
    get_welm_oe_implementation,
    should_use_welm_oe_triton_preprocess,
    hash_and_localize_welm_oe_input_ids,
    hash_input_ids_vectorized,
    WELM_OE_IMPL_ENV,
    WELM_OE_POST_PROJ_ALL_REDUCE_ENV,
    WELM_OE_TRITON_PREPROCESS_ENV,
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


def _get_welm_oe_dump_paths() -> list[Path]:
    dump_dir = Path(os.getenv("SGLANG_WELM_OE_DUMP_DIR", "/tmp/welm_oe_dumps"))
    return [
        dump_dir / "welm_oe_inputs_rank0_512.pt",
        dump_dir / "welm_oe_inputs_rank0_2581.pt",
    ]


def _make_lookup_bench_modules(*, rows: int, dim: int, device: str):
    return [
        FakeShardedEmbedding(
            local_weight=torch.randn(rows, dim, device=device, dtype=torch.bfloat16),
            vocab_start=0,
            vocab_end=rows,
            padded_end=rows,
            tp_size=1,
        )
        for _ in range(4)
    ]


def _warm_gpu_with_matmul(*, device: str, iters: int = 8):
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    for _ in range(iters):
        torch.matmul(a, b)


def _measure_lookup_concat_effective_bandwidth(
    dump_path: Path,
    *,
    rows: int = 65536,
    variants: int = 32,
    replays: int = 100,
) -> dict[str, float | tuple[int, ...]]:
    device = "cuda"
    payload = torch.load(dump_path, map_location="cpu")
    input_ids = payload["input_ids"].to(device)
    gram2 = payload["gram2"].to(device)
    gram3 = payload["gram3"].to(device)
    vocab_size = int(payload["vocab_size"])
    num_tokens = input_ids.numel()
    modules = _make_lookup_bench_modules(
        rows=rows, dim=512, device=device
    )

    workloads = []
    for idx in range(variants):
        delta = (idx + 1) * 104729
        variant_input = (input_ids + delta).remainder(vocab_size).contiguous()
        variant_gram2 = (gram2 + delta * 3).remainder(vocab_size).contiguous()
        variant_gram3 = (gram3 + delta * 7).remainder(vocab_size).contiguous()
        variant_output = torch.empty(
            (num_tokens, 2048), device=device, dtype=torch.bfloat16
        )
        workloads.append((variant_input, variant_gram2, variant_gram3, variant_output))

    _warm_gpu_with_matmul(device=device)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for variant_input, variant_gram2, variant_gram3, variant_output in workloads:
            _welm_oe_lookup_concat_2233_kernel[(num_tokens, 1)](
                variant_input,
                variant_gram2,
                variant_gram3,
                modules[0].weight,
                modules[1].weight,
                modules[2].weight,
                modules[3].weight,
                variant_output,
                num_tokens,
                vocab_size,
                vocab_size * vocab_size,
                rows,
                rows,
                rows,
                rows,
                0,
                0,
                0,
                0,
                rows,
                rows,
                rows,
                rows,
                modules[0].weight.stride(0),
                modules[1].weight.stride(0),
                modules[2].weight.stride(0),
                modules[3].weight.stride(0),
                variant_output.stride(0),
                BLOCK_D=DEFAULT_SPECIALIZED_WELM_OE_EMBED_BLOCK_D,
                EMBED_DIM=512,
                num_warps=DEFAULT_SPECIALIZED_WELM_OE_EMBED_NUM_WARPS,
            )

    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(replays):
        graph.replay()
    end_event.record()
    end_event.synchronize()

    avg_ms = start_event.elapsed_time(end_event) / replays
    dtype_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
    read_bytes = num_tokens * variants * 4 * 512 * dtype_bytes
    write_bytes = num_tokens * variants * 4 * 512 * dtype_bytes
    return {
        "input_shape": tuple(input_ids.shape),
        "output_shape": (num_tokens, 2048),
        "avg_ms": avg_ms,
        "read_tbps": read_bytes / (avg_ms / 1000.0) / 1e12,
        "effective_hbm_tbps": (read_bytes + write_bytes) / (avg_ms / 1000.0) / 1e12,
    }


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

        rank1_proj_no_bias = F.linear(concat_rank1, self.proj_module.weight, bias=None)
        with patch.dict(
            os.environ, {WELM_OE_POST_PROJ_ALL_REDUCE_ENV: "1"}, clear=False
        ):
            actual_post_proj = compute_welm_oe_embedding(
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
                all_reduce_fn=lambda x: x + rank1_proj_no_bias,
            )
        torch.testing.assert_close(actual_post_proj, expected)

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

    def test_specialized_lookup_concat_2233_requires_gram3(self):
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

        with self.assertRaisesRegex(
            AssertionError, r"oe_context\.get_gram\(3\)"
        ):
            _compute_welm_oe_concat_local_partials_specialized_2233(
                input_ids=input_ids,
                oe_context=oe_context,
                oe_vocab_sizes=oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=modules,
            )

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

        rank1_proj_no_bias = F.linear(rank1_concat, proj_module.weight, bias=None)
        with patch.dict(
            os.environ, {WELM_OE_POST_PROJ_ALL_REDUCE_ENV: "1"}, clear=False
        ):
            fused_post_proj = compute_welm_oe_embedding(
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
                all_reduce_fn=lambda x: x + rank1_proj_no_bias,
            )
        torch.testing.assert_close(fused_post_proj, legacy, atol=1e-3, rtol=1e-3)

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

    def test_online_dump_shapes_match_lookup_concat_output_shape(self):
        dump_paths = _get_welm_oe_dump_paths()
        for dump_path in dump_paths:
            self.assertTrue(dump_path.exists(), f"missing dump file: {dump_path}")
            payload = torch.load(dump_path, map_location="cpu")
            input_shape = tuple(payload["input_ids"].shape)
            output_shape = (payload["num_tokens"], 2048)
            print(
                f"[welm_oe_dump_shape] dump={dump_path.name} input_shape={input_shape} "
                f"output_shape={output_shape}"
            )
            self.assertEqual(input_shape, (payload["num_tokens"],))
            self.assertEqual(output_shape[1], 2048)

    def test_specialized_lookup_concat_hot_graph_reaches_near_3tbps_effective_hbm(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the lookup fusion performance benchmark")
        if os.getenv("SGLANG_RUN_WELM_OE_PERF_TEST", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            self.skipTest("Set SGLANG_RUN_WELM_OE_PERF_TEST=1 to run the heavy GPU perf test")

        hot_dump = _get_welm_oe_dump_paths()[1]
        self.assertTrue(hot_dump.exists(), f"missing dump file: {hot_dump}")
        result = _measure_lookup_concat_effective_bandwidth(hot_dump)
        print(
            "[welm_oe_lookup_perf] "
            f"dump={hot_dump.name} input_shape={result['input_shape']} "
            f"output_shape={result['output_shape']} avg_ms={result['avg_ms']:.4f} "
            f"read_tbps={result['read_tbps']:.4f} "
            f"effective_hbm_tbps={result['effective_hbm_tbps']:.4f}"
        )
        self.assertGreaterEqual(result["effective_hbm_tbps"], 2.8)

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
        rank1_proj_no_bias = F.linear(concat_rank1, self.proj_module.weight, bias=None)
        with patch.dict(
            os.environ,
            {
                WELM_OE_IMPL_ENV: "tp_fused",
                WELM_OE_POST_PROJ_ALL_REDUCE_ENV: "1",
            },
            clear=False,
        ):
            fused_out_post_proj = compute_welm_oe_embedding(
                input_ids=self.input_ids,
                forward_batch=self.forward_batch,
                base_hidden_states=self.base_hidden_states,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=self.oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=rank0_modules,
                oe_proj_module=self.proj_module,
                use_triton_preprocess=False,
                all_reduce_fn=lambda x: x + rank1_proj_no_bias,
            )
        torch.testing.assert_close(fused_out_post_proj, legacy_out)
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
