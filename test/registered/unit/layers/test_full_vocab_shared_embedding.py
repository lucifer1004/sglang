import gc
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.full_vocab_shared_embedding import FullVocabSharedEmbedding


class _FakeRegistry:
    def __init__(self, key, tensor):
        self.key = key
        self.tensor = tensor
        self.requests = []

    def get(self, key):
        self.requests.append(key)
        assert key == self.key
        return self.tensor


def test_full_vocab_module_uses_registry_tensor_without_private_allocation():
    key = "model.embed_tokens.weight"
    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    registry = _FakeRegistry(key, weight)

    module = FullVocabSharedEmbedding(
        key=key,
        num_embeddings=6,
        embedding_dim=4,
        registry=registry,
    )

    assert registry.requests == [key]
    assert module.weight.data_ptr() == weight.data_ptr()
    assert not module.weight.requires_grad
    assert module.weight_key == key
    assert module.tp_size == 1
    assert module.use_attn_tp_group is False
    assert module.is_full_vocab_shared is True
    assert module.requires_vocab_reduce is False
    assert module.shard_indices.org_vocab_start_index == 0
    assert module.shard_indices.org_vocab_end_index == 6
    assert module.shard_indices.num_org_vocab_padding == 0


def test_full_vocab_forward_matches_embedding_reference():
    key = "model.oe_embed.0.weight"
    weight = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    module = FullVocabSharedEmbedding(
        key=key,
        num_embeddings=7,
        embedding_dim=5,
        registry=_FakeRegistry(key, weight),
    )
    input_ids = torch.tensor([6, 0, 3, 3], dtype=torch.int64)

    torch.testing.assert_close(module(input_ids), F.embedding(input_ids, weight))


def test_full_vocab_module_rejects_registry_shape_mismatch():
    key = "model.embed_tokens.weight"

    with pytest.raises(ValueError, match="shape"):
        FullVocabSharedEmbedding(
            key=key,
            num_embeddings=6,
            embedding_dim=4,
            registry=_FakeRegistry(key, torch.empty((5, 4))),
        )


def test_full_vocab_weight_loader_rejects_mutation():
    key = "model.embed_tokens.weight"
    module = FullVocabSharedEmbedding(
        key=key,
        num_embeddings=2,
        embedding_dim=3,
        registry=_FakeRegistry(key, torch.empty((2, 3))),
    )

    with pytest.raises(RuntimeError, match="immutable"):
        module.weight_loader(module.weight, torch.zeros_like(module.weight))


def test_full_vocab_modules_never_require_vocab_reduce():
    from sglang.srt.models.welm_perf_opt import (
        _embedding_requires_vocab_reduce,
    )

    key = "model.embed_tokens.weight"
    module = FullVocabSharedEmbedding(
        key=key,
        num_embeddings=2,
        embedding_dim=3,
        registry=_FakeRegistry(key, torch.empty((2, 3))),
    )

    assert not _embedding_requires_vocab_reduce([module])
    assert _embedding_requires_vocab_reduce([SimpleNamespace(tp_size=2)])


def test_fused_decode_discovery_explicitly_rejects_full_vocab_shared():
    from sglang.srt.models.welm_perf_opt import (
        discover_welm_oe_fused_decode_modules,
    )

    base_key = "model.embed_tokens.weight"
    base = FullVocabSharedEmbedding(
        key=base_key,
        num_embeddings=2,
        embedding_dim=3,
        registry=_FakeRegistry(base_key, torch.empty((2, 3))),
    )
    oe_modules = []
    for index in range(4):
        key = f"model.oe_embed.{index}.weight"
        oe_modules.append(
            FullVocabSharedEmbedding(
                key=key,
                num_embeddings=5,
                embedding_dim=2,
                registry=_FakeRegistry(key, torch.empty((5, 2))),
            )
        )
    model = SimpleNamespace(
        embed_tokens=base,
        oe_embed=oe_modules,
        oe_gate_up_proj=SimpleNamespace(weight=torch.empty((3, 8)), bias=None),
    )

    assert discover_welm_oe_fused_decode_modules(model) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_specialized_oe_lookup_reads_read_only_uva_full_tables(tmp_path):
    from sglang.srt.models.welm_perf_opt import (
        _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512,
    )
    from sglang.srt.utils.shared_uva_tensor import (
        SharedTensorFileSpec,
        SharedUVATensorView,
    )

    source_weights = [
        torch.arange((7 + index) * 512, dtype=torch.float32)
        .reshape(7 + index, 512)
        .to(torch.bfloat16)
        for index in range(4)
    ]
    views = []
    modules = []
    try:
        for index, source in enumerate(source_weights):
            path = tmp_path / f"oe-{index}.bin"
            path.write_bytes(source.view(torch.uint16).numpy().tobytes())
            spec = SharedTensorFileSpec(
                key=f"model.oe_embed.{index}.weight",
                path=str(path),
                shape=tuple(source.shape),
                dtype="bfloat16",
                nbytes=source.numel() * source.element_size(),
                replica_id="bind-numa-0",
                numa_nodes=(0,),
                inode=path.stat().st_ino,
            )
            view = SharedUVATensorView.open(spec, 0, arena_root=tmp_path)
            views.append(view)
            modules.append(
                FullVocabSharedEmbedding(
                    key=spec.key,
                    num_embeddings=source.shape[0],
                    embedding_dim=512,
                    registry=SimpleNamespace(get=lambda _key, view=view: view.cuda_tensor),
                )
            )
        hashed_inputs = [
            torch.tensor([0, source.shape[0] - 1, 2], device="cuda:0")
            for source in source_weights
        ]

        actual = _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512(
            hashed_inputs=hashed_inputs,
            oe_embed_modules=modules,
        )
        expected = torch.cat(
            [
                F.embedding(ids, source.to(device="cuda:0"))
                for ids, source in zip(hashed_inputs, source_weights)
            ],
            dim=-1,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        modules.clear()
        gc.collect()
        for view in reversed(views):
            view.close()
