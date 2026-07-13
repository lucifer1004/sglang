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


def test_decode_swa_metadata_compacts_tail_only_page_table():
    backend = _make_backend(page_size=16)
    backend.has_swa = True
    backend.sliding_window_size = 511
    metadata = FlashAttentionMetadata()

    # The decode allocator keeps only the aligned 512-token SWA tail.  The
    # first 32 full-cache pages therefore have no SWA peer, while pages 32..64
    # contain tokens 512..1024 (prompt tail plus the first decode token).
    sparse_page_table = torch.tensor(
        [[0] * 32 + list(range(101, 134))], dtype=torch.int32
    )
    cache_seqlens = torch.tensor([1025], dtype=torch.int32)

    backend._set_decode_swa_metadata(
        metadata,
        sparse_page_table,
        cache_seqlens,
    )

    assert metadata.swa_cache_seqlens_int32.tolist() == [513]
    assert metadata.swa_page_table.shape == (1, 33)
    assert metadata.swa_page_table[0].tolist() == list(range(101, 134))


def test_decode_swa_metadata_reuses_graph_buffers_and_clears_padding():
    backend = _make_backend(page_size=16)
    backend.has_swa = True
    backend.sliding_window_size = 511
    metadata = FlashAttentionMetadata()
    out_page_table = torch.full((2, 33), -1, dtype=torch.int32)
    out_cache_seqlens = torch.full((2,), -1, dtype=torch.int32)
    sparse_page_table = torch.tensor(
        [
            [0] * 32 + list(range(101, 134)),
            [7] + [0] * 64,
        ],
        dtype=torch.int32,
    )

    backend._set_decode_swa_metadata(
        metadata,
        sparse_page_table,
        torch.tensor([1025, 16], dtype=torch.int32),
        out_page_table=out_page_table,
        out_cache_seqlens=out_cache_seqlens,
    )

    assert metadata.swa_page_table.data_ptr() == out_page_table.data_ptr()
    assert metadata.swa_cache_seqlens_int32 is out_cache_seqlens
    assert out_cache_seqlens.tolist() == [513, 16]
    assert out_page_table[0].tolist() == list(range(101, 134))
    assert out_page_table[1, 0].item() == 7
    assert out_page_table[1, 1:].tolist() == [0] * 32


def test_decode_swa_forward_uses_compact_metadata_without_retranslation(monkeypatch):
    backend = _make_backend(page_size=16)
    backend.forward_metadata = FlashAttentionMetadata(
        cache_seqlens_int32=torch.tensor([1025], dtype=torch.int32),
        swa_cache_seqlens_int32=torch.tensor([513], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        page_table=torch.tensor([[201]], dtype=torch.int32),
        swa_page_table=torch.tensor(
            [list(range(101, 134))], dtype=torch.int32
        ),
    )
    backend.use_mla = False
    backend.kv_cache_dtype_str = "auto"
    backend.has_local_attention = False
    backend.use_sliding_window_kv_pool = True
    captured = {}

    class Pool:
        full_to_swa_index_mapping = torch.ones(1, dtype=torch.int32)

        @staticmethod
        def get_kv_buffer(layer_id):
            del layer_id
            shape = (33 * 16, 1, 2)
            return torch.zeros(shape), torch.zeros(shape)

        @staticmethod
        def is_swa_layer(layer_id):
            del layer_id
            return True

        @staticmethod
        def translate_loc_from_full_to_swa(page_table):
            del page_table
            raise AssertionError("pretranslated SWA page table must not be translated again")

    def fake_flash_attn_with_kvcache(**kwargs):
        captured["page_table"] = kwargs["page_table"]
        captured["cache_seqlens"] = kwargs["cache_seqlens"]
        return torch.zeros((1, 1, 2), dtype=torch.float32)

    monkeypatch.setattr(fa_backend, "is_cp_kv_sharded", lambda: False)
    monkeypatch.setattr(
        fa_backend, "flash_attn_with_kvcache", fake_flash_attn_with_kvcache
    )
    pool = Pool()
    forward_batch = SimpleNamespace(
        spec_info=None,
        token_to_kv_pool=pool,
        batch_size=1,
        _attn_output=None,
    )
    layer = SimpleNamespace(
        layer_id=0,
        scale_seq_attn_per_suffix=False,
        is_cross_attention=False,
        attn_type=None,
        sliding_window_size=511,
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=1,
        head_dim=2,
        v_head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
    )

    backend.forward_decode(
        q=torch.zeros((1, 1, 2), dtype=torch.float32),
        k=None,
        v=None,
        layer=layer,
        forward_batch=forward_batch,
        save_kv_cache=False,
    )

    assert captured["page_table"] is backend.forward_metadata.swa_page_table
    assert captured["cache_seqlens"] is backend.forward_metadata.swa_cache_seqlens_int32


def test_decode_full_pool_layer_ignores_swa_metadata(monkeypatch):
    backend = _make_backend(page_size=16)
    backend.forward_metadata = FlashAttentionMetadata(
        cache_seqlens_int32=torch.tensor([33], dtype=torch.int32),
        swa_cache_seqlens_int32=torch.tensor([33], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        page_table=torch.tensor([[201, 202, 203]], dtype=torch.int32),
        swa_page_table=torch.tensor([[101, 102, 103]], dtype=torch.int32),
    )
    backend.use_mla = False
    backend.kv_cache_dtype_str = "auto"
    backend.has_local_attention = False
    backend.use_sliding_window_kv_pool = True
    captured = {}

    class Pool:
        @staticmethod
        def get_kv_buffer(layer_id):
            del layer_id
            shape = (204 * 16, 1, 2)
            return torch.zeros(shape), torch.zeros(shape)

        @staticmethod
        def is_swa_layer(layer_id):
            del layer_id
            return False

    def fake_flash_attn_with_kvcache(**kwargs):
        captured.update(kwargs)
        return torch.zeros((1, 1, 2), dtype=torch.float32)

    monkeypatch.setattr(fa_backend, "is_cp_kv_sharded", lambda: False)
    monkeypatch.setattr(
        fa_backend, "flash_attn_with_kvcache", fake_flash_attn_with_kvcache
    )
    forward_batch = SimpleNamespace(
        spec_info=None,
        token_to_kv_pool=Pool(),
        batch_size=1,
        _attn_output=None,
    )
    layer = SimpleNamespace(
        layer_id=0,
        scale_seq_attn_per_suffix=False,
        is_cross_attention=False,
        attn_type=None,
        sliding_window_size=262144,
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=1,
        head_dim=2,
        v_head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
    )

    backend.forward_decode(
        q=torch.zeros((1, 1, 2), dtype=torch.float32),
        k=None,
        v=None,
        layer=layer,
        forward_batch=forward_batch,
        save_kv_cache=False,
    )

    assert captured["page_table"] is backend.forward_metadata.page_table
    assert captured["cache_seqlens"] is backend.forward_metadata.cache_seqlens_int32
    assert captured["window_size"] == (-1, -1)


def test_prefill_swa_forward_uses_pretranslated_page_table(monkeypatch):
    backend = _make_backend(page_size=16)
    backend.forward_metadata = FlashAttentionMetadata(
        cache_seqlens_int32=torch.tensor([32], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 32], dtype=torch.int32),
        max_seq_len_q=1,
        page_table=torch.tensor([[201, 202]], dtype=torch.int32),
        swa_page_table=torch.tensor([[101, 102]], dtype=torch.int32),
    )
    backend.use_mla = False
    backend.kv_cache_dtype_str = "auto"
    backend.has_local_attention = False
    backend.use_sliding_window_kv_pool = True
    backend.is_welm_v4_model = False
    backend.topk = 0
    backend.fa_skip_kv_cache = False
    backend.attn_cp_size = 1
    captured = {}

    class Pool:
        full_to_swa_index_mapping = torch.ones(1, dtype=torch.int32)

        @staticmethod
        def get_kv_buffer(layer_id):
            del layer_id
            shape = (2 * 16, 1, 2)
            return torch.zeros(shape), torch.zeros(shape)

        @staticmethod
        def is_swa_layer(layer_id):
            del layer_id
            return True

        @staticmethod
        def translate_loc_from_full_to_swa(page_table):
            del page_table
            raise AssertionError("pretranslated SWA page table must not be translated again")

    class ExtendMode:
        @staticmethod
        def is_context_parallel_extend():
            return False

        @staticmethod
        def is_target_verify():
            return False

    def fake_flash_attn_with_kvcache(**kwargs):
        captured["page_table"] = kwargs["page_table"]
        return torch.zeros((1, 1, 2), dtype=torch.float32)

    monkeypatch.setattr(fa_backend, "is_cp_kv_sharded", lambda: False)
    monkeypatch.setattr(
        fa_backend, "flash_attn_with_kvcache", fake_flash_attn_with_kvcache
    )
    pool = Pool()
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=None,
        forward_mode=ExtendMode(),
        attn_cp_metadata=None,
        token_to_kv_pool=pool,
        batch_size=1,
        _attn_output=None,
    )
    layer = SimpleNamespace(
        layer_id=0,
        scale_seq_attn_per_suffix=False,
        is_cross_attention=False,
        attn_type=None,
        sliding_window_size=511,
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=1,
        head_dim=2,
        v_head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
    )

    backend.forward_extend(
        q=torch.zeros((1, 1, 2), dtype=torch.float32),
        k=None,
        v=None,
        layer=layer,
        forward_batch=forward_batch,
        save_kv_cache=False,
    )

    assert captured["page_table"] is backend.forward_metadata.swa_page_table


def test_prefill_full_pool_layer_ignores_swa_metadata(monkeypatch):
    backend = _make_backend(page_size=16)
    backend.forward_metadata = FlashAttentionMetadata(
        cache_seqlens_int32=torch.tensor([32], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 32], dtype=torch.int32),
        max_seq_len_q=1,
        page_table=torch.tensor([[201, 202]], dtype=torch.int32),
        swa_page_table=torch.tensor([[101, 102]], dtype=torch.int32),
    )
    backend.use_mla = False
    backend.kv_cache_dtype_str = "auto"
    backend.has_local_attention = False
    backend.use_sliding_window_kv_pool = True
    backend.is_welm_v4_model = False
    backend.topk = 0
    backend.fa_skip_kv_cache = False
    backend.attn_cp_size = 1
    captured = {}

    class Pool:
        @staticmethod
        def get_kv_buffer(layer_id):
            del layer_id
            shape = (203 * 16, 1, 2)
            return torch.zeros(shape), torch.zeros(shape)

        @staticmethod
        def is_swa_layer(layer_id):
            del layer_id
            return False

    class ExtendMode:
        @staticmethod
        def is_context_parallel_extend():
            return False

        @staticmethod
        def is_target_verify():
            return False

    def fake_flash_attn_with_kvcache(**kwargs):
        captured.update(kwargs)
        return torch.zeros((1, 1, 2), dtype=torch.float32)

    monkeypatch.setattr(fa_backend, "is_cp_kv_sharded", lambda: False)
    monkeypatch.setattr(
        fa_backend, "flash_attn_with_kvcache", fake_flash_attn_with_kvcache
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=None,
        forward_mode=ExtendMode(),
        attn_cp_metadata=None,
        token_to_kv_pool=Pool(),
        batch_size=1,
        _attn_output=None,
    )
    layer = SimpleNamespace(
        layer_id=0,
        scale_seq_attn_per_suffix=False,
        is_cross_attention=False,
        attn_type=None,
        sliding_window_size=262144,
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=1,
        head_dim=2,
        v_head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
    )

    backend.forward_extend(
        q=torch.zeros((1, 1, 2), dtype=torch.float32),
        k=None,
        v=None,
        layer=layer,
        forward_batch=forward_batch,
        save_kv_cache=False,
    )

    assert captured["page_table"] is backend.forward_metadata.page_table
    assert captured["window_size"] == (-1, -1)


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
