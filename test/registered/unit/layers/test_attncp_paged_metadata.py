from types import SimpleNamespace

import torch

import sglang.srt.layers.attention.flashattention_backend as fa_backend
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)


class _FakeCPGroup:
    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size


def _make_backend(page_size: int = 4, chunk_size: int = 8) -> FlashAttentionBackend:
    backend = object.__new__(FlashAttentionBackend)
    backend.page_size = page_size
    backend.attn_cp_kv_chunk_size = chunk_size
    backend.max_context_len = 128
    backend.num_splits = 0
    backend.fa_impl_ver = 3
    return backend


def test_sharded_kv_decode_metadata_compacts_paged_local_table(monkeypatch):
    monkeypatch.setattr(
        fa_backend, "get_sharded_kv_cp_group", lambda: _FakeCPGroup(1, 2)
    )
    backend = _make_backend()
    metadata = FlashAttentionMetadata()

    page_table = torch.tensor(
        [
            # logical pages: [0, 4, 8, 12, 16, 20, 24]
            # CP rank 1 owns pages starting at 8, 12, 24.
            [0, 0, 5, 6, 0, 0, 9],
            [0, 0, 7, 8, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    cache_seqlens = torch.tensor([25, 14], dtype=torch.int32)

    backend._set_sharded_kv_decode_metadata(metadata, page_table, cache_seqlens)

    assert metadata.cp_local_cache_seqlens_int32.tolist() == [9, 6]
    assert metadata.cp_local_page_table[:, :3].tolist() == [[5, 6, 9], [7, 8, 0]]


def test_sharded_kv_decode_metadata_uses_graph_buffers_for_paged_table(monkeypatch):
    monkeypatch.setattr(
        fa_backend, "get_sharded_kv_cp_group", lambda: _FakeCPGroup(0, 2)
    )
    backend = _make_backend()
    metadata = FlashAttentionMetadata()

    out_page_table = torch.full((1, 8), -1, dtype=torch.int32)
    out_cache_seqlens = torch.full((1,), -1, dtype=torch.int32)
    page_table = torch.tensor([[1, 2, 0, 0, 3, 4]], dtype=torch.int32)
    cache_seqlens = torch.tensor([21], dtype=torch.int32)

    backend._set_sharded_kv_decode_metadata(
        metadata,
        page_table,
        cache_seqlens,
        out_page_table=out_page_table,
        out_cache_seqlens=out_cache_seqlens,
    )

    assert metadata.cp_local_cache_seqlens_int32 is out_cache_seqlens
    assert metadata.cp_local_page_table.data_ptr() == out_page_table.data_ptr()
    assert out_cache_seqlens.tolist() == [13]
    assert out_page_table[0, :4].tolist() == [1, 2, 3, 4]


def test_sharded_kv_dense_drops_scheduler_metadata_for_translated_page_size(
    monkeypatch,
):
    backend = _make_backend(page_size=16)
    captured = {}

    def fake_gather(page_table, cache_seqlens, key_cache, value_cache):
        dense_k = torch.zeros((1, 4, 1, 2), dtype=torch.float32)
        dense_v = torch.zeros((1, 4, 1, 2), dtype=torch.float32)
        dense_page_table = torch.arange(4, dtype=torch.int32).view(1, 4)
        return dense_k, dense_v, dense_page_table

    def fake_flash_attn_with_kvcache(**kwargs):
        captured["scheduler_metadata"] = kwargs.get("scheduler_metadata")
        return torch.zeros((1, 1, 2), dtype=torch.float32)

    monkeypatch.setattr(backend, "_gather_sharded_kv_dense", fake_gather)
    monkeypatch.setattr(
        fa_backend, "flash_attn_with_kvcache", fake_flash_attn_with_kvcache
    )

    layer = SimpleNamespace(
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=1,
        head_dim=2,
        v_head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
    )
    scheduler_metadata = torch.ones((4,), dtype=torch.int32)

    backend._flash_attn_sharded_kv_dense(
        torch.zeros((1, 1, 2), dtype=torch.float32),
        layer,
        torch.zeros((1, 1), dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.empty(0),
        torch.empty(0),
        torch.tensor([0, 1], dtype=torch.int32),
        1,
        window_size=(-1, -1),
        causal=True,
        kwargs={},
        scheduler_metadata=scheduler_metadata,
    )

    assert captured["scheduler_metadata"] is None


def test_sharded_kv_dense_preserves_scheduler_metadata_for_paged_dense(
    monkeypatch,
):
    backend = _make_backend(page_size=16)
    captured = {}

    def fake_gather(page_table, cache_seqlens, key_cache, value_cache):
        dense_k = torch.zeros((1, 1, 16, 1, 2), dtype=torch.float32)
        dense_v = torch.zeros((1, 1, 16, 1, 2), dtype=torch.float32)
        dense_page_table = torch.zeros((1, 1), dtype=torch.int32)
        return dense_k, dense_v, dense_page_table

    def fake_flash_attn_with_kvcache(**kwargs):
        captured["scheduler_metadata"] = kwargs.get("scheduler_metadata")
        captured["k_cache_shape"] = tuple(kwargs["k_cache"].shape)
        return torch.zeros((1, 1, 2), dtype=torch.float32)

    monkeypatch.setattr(backend, "_gather_sharded_kv_dense", fake_gather)
    monkeypatch.setattr(
        fa_backend, "flash_attn_with_kvcache", fake_flash_attn_with_kvcache
    )

    layer = SimpleNamespace(
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=1,
        head_dim=2,
        v_head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
    )
    scheduler_metadata = torch.ones((4,), dtype=torch.int32)

    backend._flash_attn_sharded_kv_dense(
        torch.zeros((1, 1, 2), dtype=torch.float32),
        layer,
        torch.zeros((1, 1), dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.empty(0),
        torch.empty(0),
        torch.tensor([0, 1], dtype=torch.int32),
        1,
        window_size=(-1, -1),
        causal=True,
        kwargs={},
        scheduler_metadata=scheduler_metadata,
    )

    assert captured["scheduler_metadata"] is scheduler_metadata
    assert captured["k_cache_shape"] == (1, 16, 1, 2)


def test_fa_lse_normalization_accepts_batch_head_layouts():
    lse_batch_head = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    lse_head_batch = lse_batch_head.T.contiguous()

    normalized_batch_head = FlashAttentionBackend._normalize_fa_lse(
        lse_batch_head, batch_size=1, num_heads=3
    )
    normalized_head_batch = FlashAttentionBackend._normalize_fa_lse(
        lse_head_batch, batch_size=1, num_heads=3
    )

    assert normalized_batch_head.tolist() == [[1.0, 2.0, 3.0]]
    assert normalized_head_batch.tolist() == [[1.0, 2.0, 3.0]]
