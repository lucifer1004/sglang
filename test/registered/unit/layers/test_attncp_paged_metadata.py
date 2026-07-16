from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.attention.flashattention_backend as fa_backend
import sglang.srt.models.welmv4 as welmv4_model
from sglang.srt.context_parallel import (
    build_cp_prefill_split_spec,
    contract_cp_prefill_runtime_to_last_q,
    materialize_cp_prefill_runtime_layout,
)
from sglang.srt.layers.attention.cp_sharded_kv import (
    CPKVGatherSegmentPlan,
    CPPrefillKVGatherPlan,
    CPShardedKVPageTableResolver,
    build_cp_prefill_swa_gather_plan,
    build_cp_sharded_kv_prefill_plan,
)
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.welmv4 import _welm_should_dispatch_attention


class _FakeCPGroup:
    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size


class _FakeGatherCPGroup(_FakeCPGroup):
    def __init__(self, outputs, *, rank=0, world_size=2):
        super().__init__(rank=rank, world_size=world_size)
        self.outputs = list(outputs)
        self.calls = []

    def all_gatherv(self, tensors, sizes):
        self.calls.append((tensors, sizes))
        return self.outputs.pop(0)

    def gatherv_to_ranks(self, tensors, sizes, dst_ranks):
        if self.rank_in_group not in dst_ranks and sizes[self.rank_in_group] == 0:
            return None
        self.calls.append((tensors, sizes, dst_ranks))
        return self.outputs.pop(0) if self.rank_in_group in dst_ranks else None

    def all_to_allv(self, tensors, send_sizes, recv_sizes):
        self.calls.append(
            ("all_to_allv", tensors, tuple(send_sizes), tuple(recv_sizes))
        )
        return self.outputs.pop(0)


def _make_backend(page_size: int = 4, chunk_size: int = 8) -> FlashAttentionBackend:
    backend = object.__new__(FlashAttentionBackend)
    backend.page_size = page_size
    backend.token_to_kv_pool_allocator = None
    backend.cp_sharded_page_table_resolver = CPShardedKVPageTableResolver(None)
    backend.attn_cp_kv_chunk_size = chunk_size
    backend.max_context_len = 128
    backend.num_splits = 0
    backend.fa_impl_ver = 3
    return backend


@pytest.mark.parametrize(
    ("model_type", "cp_size", "cp_mode", "disaggregation_mode", "expected"),
    [
        ("welmv4_moe", 4, "sharded-kv", "prefill", True),
        ("welmv4_moe", 4, "sharded-kv", "null", True),
        ("welmv4_moe", 2, "sharded-kv", "prefill", False),
        ("welmv4_moe", 4, "none", "prefill", False),
        ("welmv4_moe", 4, "sharded-kv", "decode", False),
        ("qwen2", 4, "sharded-kv", "prefill", False),
    ],
)
def test_prefill_cp_kv_ipc_selection_is_automatic_for_welm_cp4_prefill(
    model_type,
    cp_size,
    cp_mode,
    disaggregation_mode,
    expected,
):
    model_runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type, architectures=None),
            hf_text_config=SimpleNamespace(model_type=model_type, architectures=None),
        ),
        attn_cp_size=cp_size,
        server_args=SimpleNamespace(
            attn_cp_mode=cp_mode,
            disaggregation_mode=disaggregation_mode,
        ),
    )

    assert fa_backend._should_use_prefill_cp_kv_ipc(model_runner) is expected


def test_decode_fused_flag_is_inactive_on_prefill_only_role():
    prefill_args = SimpleNamespace(
        attn_cp_decode_fused_q_fa=True,
        disaggregation_mode="prefill",
    )
    decode_args = SimpleNamespace(
        attn_cp_decode_fused_q_fa=True,
        disaggregation_mode="decode",
    )

    assert not fa_backend._attncp_decode_fused_q_fa_active(prefill_args)
    assert fa_backend._attncp_decode_fused_q_fa_active(decode_args)


def test_requested_fused_decode_rejects_unsupported_head_shape():
    with pytest.raises(
        RuntimeError,
        match=(
            r"--attn-cp-decode-fused-q-fa.*"
            r"local_q_heads=12.*local_kv_heads=1.*cp_size=2"
        ),
    ):
        FlashAttentionBackend._require_attncp_fused_q_fa_shape(
            local_q_heads=12,
            local_kv_heads=1,
            cp_world_size=2,
        )


def test_requested_fused_decode_accepts_restored_tp4_head_shape():
    FlashAttentionBackend._require_attncp_fused_q_fa_shape(
        local_q_heads=6,
        local_kv_heads=1,
        cp_world_size=2,
    )


def test_sharded_kv_prefill_split_rejects_batch_greater_than_one():
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        build_cp_sharded_kv_prefill_plan(
            logical_page_table=torch.ones((2, 2), dtype=torch.int32),
            cache_seqlens=torch.tensor([8, 8], dtype=torch.int32),
            prefix_lens=[4, 4],
            seq_lens=[8, 8],
            page_size=4,
        )


def test_sharded_kv_prefill_plan_is_backend_neutral():
    plan = build_cp_sharded_kv_prefill_plan(
        logical_page_table=torch.tensor([[10, 11, 12]], dtype=torch.int32),
        cache_seqlens=torch.tensor([10], dtype=torch.int32),
        prefix_lens=[4],
        seq_lens=[10],
        page_size=4,
    )

    assert plan.prefix.logical_page_table.tolist() == [[10]]
    assert plan.prefix.cache_seqlens.tolist() == [4]
    assert plan.extend.logical_page_table.tolist() == [[11, 12]]
    assert plan.extend.cache_seqlens.tolist() == [6]
    assert plan.total_page_columns == 3


def test_sharded_kv_prefill_plan_uses_cpu_lengths_without_reading_gpu_lengths():
    class CacheLensWithoutHostRead:
        dtype = torch.int32
        device = torch.device("cpu")

        def numel(self):
            return 1

        def reshape(self, *args, **kwargs):
            raise AssertionError("cache_seqlens must not be read on the layer path")

    cache_seqlens = CacheLensWithoutHostRead()
    plan = build_cp_sharded_kv_prefill_plan(
        logical_page_table=torch.tensor([[10, 11, 12]], dtype=torch.int32),
        cache_seqlens=cache_seqlens,
        prefix_lens=[4],
        seq_lens=[10],
        page_size=4,
    )

    assert plan.full_cache_seqlens is cache_seqlens
    assert plan.extend.cache_seqlens.tolist() == [6]


def test_sharded_kv_prefill_plan_is_initialized_once_per_forward():
    backend = _make_backend()
    metadata = FlashAttentionMetadata(
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

    assert metadata.cp_sharded_kv_prefill_plan is first_plan


def test_phase2_prefill_metadata_does_not_translate_logical_slots_as_swa_slots():
    backend = _make_backend()
    backend.is_attn_cp_sharded_kv = True
    backend.use_sliding_window_kv_pool = True
    backend.has_local_attention = False
    backend.topk = 0
    backend.device = torch.device("cpu")
    backend._attncp_forward_batch_requires_exact_logprob = lambda _batch: False
    backend.attncp_forward_batch_has_empty_cp_shard = lambda _batch: False

    class Pool:
        def translate_loc_from_full_to_swa(self, slots):
            raise AssertionError(
                f"logical CP slots must not enter the SWA physical mapping: {slots}"
            )

    backend.token_to_kv_pool = Pool()
    logical_slots = torch.arange(128, 136, dtype=torch.int64).view(1, -1)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_prefill_split_specs=[object()],
        attn_cp_prefill_runtime_layout=object(),
        seq_lens=torch.tensor([8], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
        batch_size=1,
        req_to_token_pool=SimpleNamespace(req_to_token=logical_slots),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        enable_welm_kv_mirror_opt=False,
        extend_prefix_lens_cpu=[0],
        extend_seq_lens=torch.tensor([8], dtype=torch.int32),
        extend_seq_lens_cpu=[8],
        encoder_lens=None,
        spec_info=None,
    )

    backend.init_forward_metadata(forward_batch)

    assert backend.forward_metadata.swa_page_table is None


def test_cp_decode_metadata_resolves_logical_pages_to_physical_swa_pages(
    monkeypatch,
):
    backend = _make_backend(page_size=4)
    backend.is_attn_cp_sharded_kv = True
    backend.enable_attn_cp_decode_local_merge = False
    backend.use_sliding_window_kv_pool = True
    backend.has_local_attention = False
    backend.topk = 0
    backend.device = torch.device("cpu")
    backend.use_mla = False
    backend._get_scheduler_metadata = None
    backend._disable_scheduler_metadata_precompute = False
    backend._attncp_forward_batch_requires_exact_logprob = lambda _batch: False
    backend.attncp_forward_batch_has_empty_cp_shard = lambda _batch: False

    class Pool:
        def translate_loc_from_full_to_swa(self, slots):
            raise AssertionError(
                f"logical CP slots must not enter the SWA physical mapping: {slots}"
            )

    resolver_calls = []

    class Resolver:
        def resolve_swa_pages(self, logical_pages):
            resolver_calls.append(logical_pages.clone())
            return logical_pages + 100

    backend.token_to_kv_pool = Pool()
    backend.cp_sharded_page_table_resolver = Resolver()
    logical_slots = torch.arange(128, 136, dtype=torch.int64).view(1, -1)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        attn_cp_prefill_split_specs=None,
        attn_cp_prefill_runtime_layout=None,
        seq_lens=torch.tensor([8], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
        batch_size=1,
        req_to_token_pool=SimpleNamespace(req_to_token=logical_slots),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        encoder_lens=None,
        spec_info=None,
    )
    monkeypatch.setattr(fa_backend, "is_cp_kv_sharded", lambda: True)

    backend.init_forward_metadata(forward_batch)

    assert resolver_calls[0].tolist() == [[32, 33]]
    assert backend.forward_metadata.page_table.tolist() == [[32, 33]]
    assert backend.forward_metadata.swa_page_table.tolist() == [[132, 133]]


def test_non_cp_decode_compacts_swa_metadata_once(monkeypatch):
    backend = _make_backend(page_size=4)
    backend.is_attn_cp_sharded_kv = False
    backend.use_sliding_window_kv_pool = True
    backend.has_local_attention = False
    backend.topk = 0
    backend.device = torch.device("cpu")
    backend.use_mla = False
    backend._get_scheduler_metadata = None
    backend._disable_scheduler_metadata_precompute = False
    backend._attncp_forward_batch_requires_exact_logprob = lambda _batch: False
    backend.attncp_forward_batch_has_empty_cp_shard = lambda _batch: False

    class Pool:
        @staticmethod
        def translate_loc_from_full_to_swa(slots):
            return slots + 100

    calls = []

    def compact_once(metadata, page_table, cache_seqlens):
        calls.append((page_table.clone(), cache_seqlens.clone()))
        metadata.swa_page_table = page_table[:, :1]

    backend.token_to_kv_pool = Pool()
    backend._set_decode_swa_metadata = compact_once
    logical_slots = torch.arange(128, 136, dtype=torch.int64).view(1, -1)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        attn_cp_prefill_split_specs=None,
        attn_cp_prefill_runtime_layout=None,
        seq_lens=torch.tensor([8], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
        batch_size=1,
        req_to_token_pool=SimpleNamespace(req_to_token=logical_slots),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        encoder_lens=None,
        spec_info=None,
    )
    monkeypatch.setattr(fa_backend, "is_cp_kv_sharded", lambda: False)

    backend.init_forward_metadata(forward_batch)

    assert len(calls) == 1


def test_sharded_kv_prefill_split_gathers_prefix_and_extend_separately(monkeypatch):
    backend = _make_backend()
    calls = []

    def fake_gather(page_table, cache_seqlens, key_cache, value_cache):
        calls.append((page_table.clone(), cache_seqlens.clone()))
        num_pages = page_table.shape[1]
        marker = float(len(calls))
        dense_k = torch.full((1, num_pages, 4, 1, 1), marker)
        dense_v = torch.full((1, num_pages, 4, 1, 1), marker + 10)
        dense_page_table = torch.arange(num_pages, dtype=torch.int32).view(1, -1)
        return dense_k, dense_v, dense_page_table

    monkeypatch.setattr(backend, "_gather_sharded_kv_dense", fake_gather)

    plan = build_cp_sharded_kv_prefill_plan(
        logical_page_table=torch.tensor([[10, 11, 12]], dtype=torch.int32),
        cache_seqlens=torch.tensor([10], dtype=torch.int32),
        prefix_lens=[4],
        seq_lens=[10],
        page_size=4,
    )

    result = backend._gather_sharded_kv_dense_prefill_split(
        plan=plan,
        key_cache=torch.empty(0),
        value_cache=torch.empty(0),
    )

    dense_k, dense_v, dense_page_table, dense_cache_seqlens = result
    assert len(calls) == 2
    assert calls[0][0].tolist() == [[10]]
    assert calls[0][1].tolist() == [4]
    assert calls[1][0].tolist() == [[11, 12]]
    assert calls[1][1].tolist() == [6]
    assert dense_k[:, :1].unique().tolist() == [1.0]
    assert dense_k[:, 1:].unique().tolist() == [2.0]
    assert dense_v[:, :1].unique().tolist() == [11.0]
    assert dense_v[:, 1:].unique().tolist() == [12.0]
    assert dense_page_table.tolist() == [[0, 1, 2]]
    assert dense_cache_seqlens.tolist() == [10]


def test_phase2_prefill_uses_compact_kv_and_explicit_q_blocks(monkeypatch):
    backend = _make_backend(page_size=4)
    backend.kv_cache_dtype_str = "auto"
    spec = build_cp_prefill_split_spec(
        extend_start=4,
        extend_len=16,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    out_cache_loc = torch.zeros(16, dtype=torch.int64)
    out_cache_loc[:4] = torch.arange(4) + 8
    out_cache_loc[12:] = torch.arange(4) + 12
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=0,
        input_ids=torch.arange(16),
        positions=torch.arange(4, 20, dtype=torch.int32),
        out_cache_loc=out_cache_loc,
    )
    prefix_segment = CPKVGatherSegmentPlan(
        sizes=(2, 2),
        local_physical_slots=torch.tensor([4, 6]),
        rank_packed_to_logical=torch.tensor([0, 2, 1, 3]),
        logical_token_count=4,
    )
    extend_map = torch.tensor(
        [
            logical_index - spec.extend_start
            for rank in range(2)
            for block in spec.local_blocks(rank)
            for logical_index in range(
                block.logical_start,
                block.logical_start + block.token_count,
            )
        ]
    )
    extend_segment = CPKVGatherSegmentPlan(
        sizes=spec.per_rank_tokens,
        local_physical_slots=runtime.local_out_cache_loc,
        rank_packed_to_logical=extend_map,
        logical_token_count=16,
    )
    gather_plan = CPPrefillKVGatherPlan(
        prefix=prefix_segment,
        extend=extend_segment,
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=runtime,
        attn_cp_prefill_kv_gather_plan=gather_plan,
    )
    layer = SimpleNamespace(
        is_cross_attention=False,
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=2,
        v_head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
    )
    key_cache = torch.zeros((4, 4, 1, 2), dtype=torch.float32)
    value_cache = torch.zeros_like(key_cache)
    prefix_k_logical = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    prefix_v_logical = prefix_k_logical + 100
    key_cache[1, 0] = prefix_k_logical[0]
    key_cache[1, 2] = prefix_k_logical[2]
    value_cache[1, 0] = prefix_v_logical[0]
    value_cache[1, 2] = prefix_v_logical[2]
    extend_k_logical = torch.arange(32, dtype=torch.float32).view(16, 1, 2) + 20
    extend_v_logical = extend_k_logical + 100
    local_extend_indices = runtime.local_extend_indices
    local_k = extend_k_logical.index_select(0, local_extend_indices)
    local_v = extend_v_logical.index_select(0, local_extend_indices)
    prefix_k_packed = prefix_k_logical.index_select(
        0, prefix_segment.rank_packed_to_logical
    )
    prefix_v_packed = prefix_v_logical.index_select(
        0, prefix_segment.rank_packed_to_logical
    )
    extend_k_packed = extend_k_logical.index_select(0, extend_map)
    extend_v_packed = extend_v_logical.index_select(0, extend_map)
    cp_group = _FakeGatherCPGroup(
        [(prefix_k_packed, prefix_v_packed), (extend_k_packed, extend_v_packed)]
    )
    monkeypatch.setattr(fa_backend, "get_sharded_kv_cp_group", lambda: cp_group)
    monkeypatch.setattr(
        backend,
        "_flash_attn_sharded_kv_dense",
        lambda *_args, **_kwargs: pytest.fail("Phase 2 must not use dense fallback"),
    )
    fa_calls = []

    def fake_flash_attn_varlen_func(**kwargs):
        fa_calls.append(kwargs)
        return kwargs["q"] + 1

    monkeypatch.setattr(
        fa_backend, "flash_attn_varlen_func", fake_flash_attn_varlen_func
    )
    q = torch.arange(16, dtype=torch.float32).view(8, 2)

    result = backend._forward_extend_sharded_kv(
        q,
        local_k,
        local_v,
        layer,
        forward_batch,
        FlashAttentionMetadata(),
        key_cache,
        value_cache,
        use_welm_custom_last_q=False,
        window_size=(-1, -1),
        causal=True,
        kwargs={},
    )

    assert torch.equal(result, q + 1)
    assert len(cp_group.calls) == 2
    assert torch.equal(cp_group.calls[0][0][0], prefix_k_logical[[0, 2]])
    assert torch.equal(cp_group.calls[0][0][1], prefix_v_logical[[0, 2]])
    assert cp_group.calls[0][1] == [2, 2]
    assert torch.equal(cp_group.calls[1][0][0], local_k)
    assert torch.equal(cp_group.calls[1][0][1], local_v)
    assert cp_group.calls[1][1] == list(spec.per_rank_tokens)
    assert [call["max_seqlen_q"] for call in fa_calls] == [4, 4]
    assert [call["max_seqlen_k"] for call in fa_calls] == [8, 20]
    assert [call["cu_seqlens_k"].tolist() for call in fa_calls] == [
        [0, 8],
        [0, 20],
    ]


def test_phase2_prefix_reader_zeroes_dummy_swa_slots():
    backend = _make_backend(page_size=4)
    key_cache = torch.arange(16, dtype=torch.float32).view(2, 4, 1, 2)
    value_cache = key_cache + 100

    key, value = backend._read_cp_prefill_prefix_kv(
        torch.tensor([0, 5]),
        key_cache,
        value_cache,
        page_size=4,
        zero_dummy_slots=True,
    )

    assert torch.equal(key[0], torch.zeros_like(key[0]))
    assert torch.equal(value[0], torch.zeros_like(value[0]))
    assert torch.equal(key[1], key_cache[1, 1])
    assert torch.equal(value[1], value_cache[1, 1])
    assert torch.equal(key_cache[0, 0], torch.zeros_like(key_cache[0, 0]))
    assert torch.equal(value_cache[0, 0], torch.zeros_like(value_cache[0, 0]))


def test_phase2_dense_prefix_reader_does_not_mask_slot_zero():
    backend = _make_backend(page_size=4)
    key_cache = torch.arange(16, dtype=torch.float32).view(2, 4, 1, 2)
    value_cache = key_cache + 100

    key, value = backend._read_cp_prefill_prefix_kv(
        torch.tensor([0, 5]),
        key_cache,
        value_cache,
        page_size=4,
        zero_dummy_slots=False,
    )

    assert torch.equal(key[0], key_cache[0, 0])
    assert torch.equal(value[0], value_cache[0, 0])


def test_phase2_swa_prefill_translates_prefix_and_windows_every_q_block(
    monkeypatch,
):
    backend = _make_backend(page_size=4)
    backend.kv_cache_dtype_str = "auto"
    backend.prefill_cp_kv_ipc_transport = SimpleNamespace(
        push=lambda **_kwargs: pytest.fail(
            "compact SWA must not enter the full-attention IPC transport"
        )
    )
    resolver_calls = []

    class Resolver:
        def resolve_swa_slots(self, slots):
            resolver_calls.append(slots.clone())
            assert slots.tolist() == [9]
            return torch.tensor([5], dtype=torch.int64)

    backend.cp_sharded_page_table_resolver = Resolver()
    spec = build_cp_prefill_split_spec(
        extend_start=4,
        extend_len=16,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    out_cache_loc = torch.zeros(16, dtype=torch.int64)
    out_cache_loc[:4] = torch.arange(4) + 8
    out_cache_loc[12:] = torch.arange(4) + 12
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=0,
        input_ids=torch.arange(16),
        positions=torch.arange(4, 20, dtype=torch.int32),
        out_cache_loc=out_cache_loc,
    )
    prefix_segment = CPKVGatherSegmentPlan(
        sizes=(2, 2),
        local_physical_slots=torch.tensor([8, 9]),
        rank_packed_to_logical=torch.tensor([0, 2, 1, 3]),
        logical_token_count=4,
    )
    extend_map = torch.tensor(
        [
            logical_index - spec.extend_start
            for rank in range(2)
            for block in spec.local_blocks(rank)
            for logical_index in range(
                block.logical_start,
                block.logical_start + block.token_count,
            )
        ]
    )
    gather_plan = CPPrefillKVGatherPlan(
        prefix=prefix_segment,
        extend=CPKVGatherSegmentPlan(
            sizes=spec.per_rank_tokens,
            local_physical_slots=runtime.local_out_cache_loc,
            rank_packed_to_logical=extend_map,
            logical_token_count=16,
        ),
    )
    swa_plan = build_cp_prefill_swa_gather_plan(
        plan=gather_plan,
        runtime_layout=runtime,
        window_left=3,
    )
    key_cache = torch.full((2, 4, 1, 2), 99.0)
    value_cache = key_cache + 100
    key_cache[1, 1] = torch.tensor([[5.0, 6.0]])
    value_cache[1, 1] = torch.tensor([[105.0, 106.0]])
    prefix_k_logical = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    prefix_v_logical = prefix_k_logical + 100
    prefix_k_logical[2] = key_cache[1, 1]
    prefix_v_logical[2] = value_cache[1, 1]
    extend_k_logical = torch.arange(32, dtype=torch.float32).view(16, 1, 2) + 20
    extend_v_logical = extend_k_logical + 100
    full_k = torch.cat((prefix_k_logical, extend_k_logical))
    full_v = torch.cat((prefix_v_logical, extend_v_logical))
    compact_indices = torch.cat(
        [
            torch.arange(
                max(0, block.logical_start - 3), block.visible_kv_end
            )
            for block in runtime.q_blocks
        ]
    )
    compact_k = full_k.index_select(0, compact_indices)
    compact_v = full_v.index_select(0, compact_indices)
    cp_group = _FakeGatherCPGroup(
        [
            (
                compact_k.index_select(
                    0, swa_plan.prefix.recv_packed_to_compact
                ),
                compact_v.index_select(
                    0, swa_plan.prefix.recv_packed_to_compact
                ),
            ),
            (
                compact_k.index_select(
                    0, swa_plan.extend.recv_packed_to_compact
                ),
                compact_v.index_select(
                    0, swa_plan.extend.recv_packed_to_compact
                ),
            ),
        ]
    )
    monkeypatch.setattr(fa_backend, "get_sharded_kv_cp_group", lambda: cp_group)
    fa_calls = []

    def fake_flash_attn_varlen_func(**kwargs):
        fa_calls.append(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(
        fa_backend, "flash_attn_varlen_func", fake_flash_attn_varlen_func
    )
    local_k = extend_k_logical.index_select(0, runtime.local_extend_indices)
    local_v = extend_v_logical.index_select(0, runtime.local_extend_indices)
    q = torch.arange(16, dtype=torch.float32).view(8, 2)

    result = backend._forward_extend_sharded_kv(
        q,
        local_k,
        local_v,
        SimpleNamespace(
            is_cross_attention=False,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=2,
            v_head_dim=2,
            scaling=1.0,
            logit_cap=0.0,
        ),
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=runtime,
            attn_cp_prefill_kv_gather_plan=gather_plan,
            attn_cp_prefill_swa_kv_gather_plans=None,
        ),
        FlashAttentionMetadata(),
        key_cache,
        value_cache,
        use_welm_custom_last_q=False,
        window_size=(3, 0),
        causal=True,
        kwargs={},
    )

    assert torch.equal(result, q)
    assert len(resolver_calls) == 1
    assert torch.equal(resolver_calls[0], torch.tensor([9]))
    assert len(cp_group.calls) == 2
    assert [call[0] for call in cp_group.calls] == [
        "all_to_allv",
        "all_to_allv",
    ]
    assert len(fa_calls) == 2
    for block_index, call in enumerate(fa_calls):
        compact_start = block_index * 7
        assert torch.equal(call["k"], compact_k[compact_start : compact_start + 7])
        assert torch.equal(call["v"], compact_v[compact_start : compact_start + 7])
        assert call["window_size"] == (3, 0)
        assert call["cu_seqlens_q"].tolist() == [0, 4]
        assert call["cu_seqlens_k"].tolist() == [0, 7]
        assert call["max_seqlen_q"] == 4
    assert [call["max_seqlen_k"] for call in fa_calls] == [8, 20]


def test_phase2_swa_plan_is_cached_per_runtime_layout(monkeypatch):
    backend = _make_backend()
    runtime = SimpleNamespace(
        q_is_contracted=False,
        active_per_rank_tokens=(4, 4),
    )
    gather_plan = object()
    batch = SimpleNamespace(attn_cp_prefill_swa_kv_gather_plans=None)
    built_plan = object()
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return built_plan

    monkeypatch.setattr(fa_backend, "build_cp_prefill_swa_gather_plan", fake_build)

    first = backend._get_cp_prefill_swa_gather_plan(
        batch, gather_plan, runtime, window_left=512
    )
    second = backend._get_cp_prefill_swa_gather_plan(
        batch, gather_plan, runtime, window_left=512
    )

    assert first is built_plan
    assert second is built_plan
    assert len(calls) == 1


def test_phase2_empty_local_q_without_owned_kv_does_not_communicate(monkeypatch):
    backend = _make_backend(page_size=4)
    backend.kv_cache_dtype_str = "auto"
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=1,
        cp_size=4,
        page_size=4,
        owner_rotation=0,
    )
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=3,
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0], dtype=torch.int32),
        out_cache_loc=torch.tensor([0]),
    )
    gather_plan = CPPrefillKVGatherPlan(
        prefix=CPKVGatherSegmentPlan(
            sizes=(0, 0, 0, 0),
            local_physical_slots=torch.empty((0,), dtype=torch.int64),
            rank_packed_to_logical=torch.empty((0,), dtype=torch.int64),
            logical_token_count=0,
        ),
        extend=CPKVGatherSegmentPlan(
            sizes=spec.per_rank_tokens,
            local_physical_slots=runtime.local_out_cache_loc,
            rank_packed_to_logical=torch.tensor([0]),
            logical_token_count=1,
        ),
    )
    cp_group = _FakeGatherCPGroup([], rank=3, world_size=4)
    monkeypatch.setattr(fa_backend, "get_sharded_kv_cp_group", lambda: cp_group)
    monkeypatch.setattr(
        fa_backend,
        "flash_attn_varlen_func",
        lambda **_kwargs: pytest.fail("empty local Q must not launch FA3"),
    )
    layer = SimpleNamespace(
        is_cross_attention=False,
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=2,
        v_head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
    )

    result = backend._forward_extend_sharded_kv(
        torch.empty((0, 2)),
        torch.empty((0, 1, 2)),
        torch.empty((0, 1, 2)),
        layer,
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=runtime,
            attn_cp_prefill_kv_gather_plan=gather_plan,
        ),
        FlashAttentionMetadata(),
        torch.empty((1, 4, 1, 2)),
        torch.empty((1, 4, 1, 2)),
        use_welm_custom_last_q=False,
        window_size=(-1, -1),
        causal=True,
        kwargs={},
    )

    assert result.shape == (0, 2)
    assert cp_group.calls == []


def test_phase2_full_attention_uses_ipc_source_push_and_releases_after_fa(
    monkeypatch,
):
    backend = _make_backend(page_size=4)
    backend.kv_cache_dtype_str = "auto"
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=4,
        cp_size=1,
        page_size=4,
        owner_rotation=0,
    )
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=0,
        input_ids=torch.arange(4),
        positions=torch.arange(4, dtype=torch.int32),
        out_cache_loc=torch.arange(4, dtype=torch.int64) + 4,
    )
    gather_plan = CPPrefillKVGatherPlan(
        prefix=CPKVGatherSegmentPlan(
            sizes=(0,),
            local_physical_slots=torch.empty((0,), dtype=torch.int64),
            rank_packed_to_logical=torch.empty((0,), dtype=torch.int64),
            logical_token_count=0,
        ),
        extend=CPKVGatherSegmentPlan(
            sizes=(4,),
            local_physical_slots=runtime.local_out_cache_loc,
            rank_packed_to_logical=torch.arange(4),
            logical_token_count=4,
        ),
    )
    full_k = torch.arange(8, dtype=torch.float32).view(4, 2)
    full_v = full_k + 100

    class _Lease:
        def __init__(self):
            self.key = full_k
            self.value = full_v
            self.released = False

        def release(self):
            self.released = True

    class _Transport:
        def __init__(self):
            self.calls = []
            self.lease = _Lease()

        def push(self, **kwargs):
            self.calls.append(kwargs)
            return self.lease

    transport = _Transport()
    backend.prefill_cp_kv_ipc_transport = transport
    monkeypatch.setattr(
        fa_backend,
        "gather_cp_prefill_kv",
        lambda **_kwargs: pytest.fail("IPC path must not call the NCCL gather"),
    )
    fa_calls = []

    def fake_flash_attn_varlen_func(**kwargs):
        fa_calls.append(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(
        fa_backend, "flash_attn_varlen_func", fake_flash_attn_varlen_func
    )
    key_cache = torch.empty((2, 4, 1, 2))
    value_cache = torch.empty_like(key_cache)
    local_k = full_k.view(4, 1, 2)
    local_v = full_v.view(4, 1, 2)
    q = torch.arange(8, dtype=torch.float32).view(4, 2)

    result = backend._forward_extend_sharded_kv_compact(
        q,
        local_k,
        local_v,
        SimpleNamespace(
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=2,
            v_head_dim=2,
            scaling=1.0,
            logit_cap=0.0,
        ),
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=runtime,
            attn_cp_prefill_kv_gather_plan=gather_plan,
            attn_cp_prefill_kv_source_push_plan=SimpleNamespace(
                logical_token_count=4
            ),
        ),
        key_cache,
        value_cache,
        use_welm_custom_last_q=False,
        window_size=(-1, -1),
        causal=True,
        kwargs={},
    )

    assert torch.equal(result, q)
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["prefix_key_rows"].data_ptr() == key_cache.data_ptr()
    assert call["prefix_value_rows"].data_ptr() == value_cache.data_ptr()
    assert torch.equal(call["extend_key_rows"], full_k)
    assert torch.equal(call["extend_value_rows"], full_v)
    assert call["destination_ranks"] == (0,)
    assert len(fa_calls) == 1
    assert torch.equal(fa_calls[0]["k"], local_k)
    assert torch.equal(fa_calls[0]["v"], local_v)
    assert transport.lease.released

    transport.lease = _Lease()

    def fail_flash_attn(**_kwargs):
        raise RuntimeError("FA failure")

    monkeypatch.setattr(
        fa_backend,
        "flash_attn_varlen_func",
        fail_flash_attn,
    )
    with pytest.raises(RuntimeError, match="FA failure"):
        backend._forward_extend_sharded_kv_compact(
            q,
            local_k,
            local_v,
            SimpleNamespace(
                tp_q_head_num=1,
                tp_k_head_num=1,
                tp_v_head_num=1,
                head_dim=2,
                v_head_dim=2,
                scaling=1.0,
                logit_cap=0.0,
            ),
            SimpleNamespace(
                attn_cp_prefill_runtime_layout=runtime,
                attn_cp_prefill_kv_gather_plan=gather_plan,
                attn_cp_prefill_kv_source_push_plan=SimpleNamespace(
                    logical_token_count=4
                ),
            ),
            key_cache,
            value_cache,
            use_welm_custom_last_q=False,
            window_size=(-1, -1),
            causal=True,
            kwargs={},
        )
    assert transport.lease.released


def test_phase2_ipc_missing_source_push_plan_fails_without_gather(monkeypatch):
    backend = _make_backend(page_size=4)
    backend.kv_cache_dtype_str = "auto"
    backend.prefill_cp_kv_ipc_transport = object()
    runtime = SimpleNamespace(
        q_is_contracted=False,
        active_local_tokens=0,
        kv_local_tokens=0,
        active_tokens_per_cp_rank=lambda: (0,),
    )
    monkeypatch.setattr(
        fa_backend,
        "gather_cp_prefill_kv",
        lambda **_kwargs: pytest.fail("IPC failure must not fall back to gather"),
    )

    with pytest.raises(RuntimeError, match="source-push plan"):
        backend._forward_extend_sharded_kv_compact(
            torch.empty((0, 2)),
            torch.empty((0, 1, 2)),
            torch.empty((0, 1, 2)),
            SimpleNamespace(
                tp_q_head_num=1,
                tp_k_head_num=1,
                tp_v_head_num=1,
                head_dim=2,
                v_head_dim=2,
            ),
            SimpleNamespace(
                attn_cp_prefill_runtime_layout=runtime,
                attn_cp_prefill_kv_gather_plan=object(),
                attn_cp_prefill_kv_source_push_plan=None,
            ),
            torch.empty((1, 4, 1, 2)),
            torch.empty((1, 4, 1, 2)),
            use_welm_custom_last_q=False,
            window_size=(-1, -1),
            causal=True,
            kwargs={},
        )


def test_phase2_source_push_plan_without_transport_fails_without_gather(monkeypatch):
    backend = _make_backend(page_size=4)
    backend.kv_cache_dtype_str = "auto"
    backend.prefill_cp_kv_ipc_transport = None
    runtime = SimpleNamespace(
        q_is_contracted=False,
        active_local_tokens=0,
        kv_local_tokens=0,
        active_tokens_per_cp_rank=lambda: (0,),
    )
    monkeypatch.setattr(
        fa_backend,
        "gather_cp_prefill_kv",
        lambda **_kwargs: pytest.fail(
            "selected IPC path must not fall back to the NCCL gather"
        ),
    )

    with pytest.raises(RuntimeError, match="transport"):
        backend._forward_extend_sharded_kv_compact(
            torch.empty((0, 2)),
            torch.empty((0, 1, 2)),
            torch.empty((0, 1, 2)),
            SimpleNamespace(
                tp_q_head_num=1,
                tp_k_head_num=1,
                tp_v_head_num=1,
                head_dim=2,
                v_head_dim=2,
            ),
            SimpleNamespace(
                attn_cp_prefill_runtime_layout=runtime,
                attn_cp_prefill_kv_gather_plan=object(),
                attn_cp_prefill_kv_source_push_plan=SimpleNamespace(
                    logical_token_count=0
                ),
            ),
            torch.empty((1, 4, 1, 2)),
            torch.empty((1, 4, 1, 2)),
            use_welm_custom_last_q=False,
            window_size=(-1, -1),
            causal=True,
            kwargs={},
        )


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_phase2_kv_mirror_contracts_q_but_exchanges_all_owner_kv(
    monkeypatch, cp_rank
):
    backend = _make_backend(page_size=4)
    backend.kv_cache_dtype_str = "auto"
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=8,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    full_cache_loc = torch.zeros(8, dtype=torch.int64)
    for block in spec.local_blocks(cp_rank):
        start = block.logical_start
        full_cache_loc[start : start + block.token_count] = (
            torch.arange(block.token_count) + 8 + start
        )
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=cp_rank,
        input_ids=torch.arange(8),
        positions=torch.arange(8, dtype=torch.int32),
        out_cache_loc=full_cache_loc,
    )
    kv_extend_indices = runtime.local_extend_indices.clone()
    runtime = contract_cp_prefill_runtime_to_last_q(runtime)
    last_owner = spec.blocks[-1].owner_rank
    extend_map = torch.tensor(
        [
            logical_index
            for rank in range(2)
            for block in spec.local_blocks(rank)
            for logical_index in range(
                block.logical_start,
                block.logical_start + block.token_count,
            )
        ]
    )
    gather_plan = CPPrefillKVGatherPlan(
        prefix=CPKVGatherSegmentPlan(
            sizes=(0, 0),
            local_physical_slots=torch.empty((0,), dtype=torch.int64),
            rank_packed_to_logical=torch.empty((0,), dtype=torch.int64),
            logical_token_count=0,
        ),
        extend=CPKVGatherSegmentPlan(
            sizes=spec.per_rank_tokens,
            local_physical_slots=runtime.local_out_cache_loc,
            rank_packed_to_logical=extend_map,
            logical_token_count=8,
        ),
    )
    full_k = torch.arange(16, dtype=torch.float32).view(8, 1, 2)
    full_v = full_k + 100
    packed_k = full_k.index_select(0, extend_map)
    packed_v = full_v.index_select(0, extend_map)
    owner_outputs = [
        (torch.empty((0, 1, 2)), torch.empty((0, 1, 2))),
        (packed_k, packed_v),
    ]
    cp_group = _FakeGatherCPGroup(
        owner_outputs if cp_rank == last_owner else [],
        rank=cp_rank,
        world_size=2,
    )
    monkeypatch.setattr(fa_backend, "get_sharded_kv_cp_group", lambda: cp_group)
    fa_calls = []

    def fake_flash_attn_varlen_func(**kwargs):
        fa_calls.append(kwargs)
        return kwargs["q"] + 1

    monkeypatch.setattr(
        fa_backend, "flash_attn_varlen_func", fake_flash_attn_varlen_func
    )
    layer = SimpleNamespace(
        is_cross_attention=False,
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=2,
        v_head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
    )
    local_k = full_k.index_select(0, kv_extend_indices)
    local_v = full_v.index_select(0, kv_extend_indices)
    q = (
        torch.tensor([[7.0, 8.0]])
        if cp_rank == last_owner
        else torch.empty((0, 2))
    )

    result = backend._forward_extend_sharded_kv(
        q,
        local_k,
        local_v,
        layer,
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=runtime,
            attn_cp_prefill_kv_gather_plan=gather_plan,
        ),
        FlashAttentionMetadata(),
        torch.empty((4, 4, 1, 2)),
        torch.empty((4, 4, 1, 2)),
        use_welm_custom_last_q=True,
        window_size=(-1, -1),
        causal=True,
        kwargs={},
    )

    assert len(cp_group.calls) == (2 if cp_rank == last_owner else 1)
    assert tuple(cp_group.calls[-1][2]) == (last_owner,)
    if cp_rank == last_owner:
        assert torch.equal(result, q + 1)
        assert len(fa_calls) == 1
        assert fa_calls[0]["max_seqlen_q"] == 1
        assert fa_calls[0]["max_seqlen_k"] == 8
    else:
        assert result.shape == (0, 2)
        assert fa_calls == []


def test_phase2_custom_last_q_requires_contracted_runtime(monkeypatch):
    backend = _make_backend(page_size=4)
    backend.kv_cache_dtype_str = "auto"
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=4,
        cp_size=1,
        page_size=4,
        owner_rotation=0,
    )
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=0,
        input_ids=torch.arange(4),
        positions=torch.arange(4, dtype=torch.int32),
        out_cache_loc=torch.arange(4, dtype=torch.int64) + 4,
    )
    gather_plan = CPPrefillKVGatherPlan(
        prefix=CPKVGatherSegmentPlan(
            sizes=(0,),
            local_physical_slots=torch.empty((0,), dtype=torch.int64),
            rank_packed_to_logical=torch.empty((0,), dtype=torch.int64),
            logical_token_count=0,
        ),
        extend=CPKVGatherSegmentPlan(
            sizes=(4,),
            local_physical_slots=runtime.local_out_cache_loc,
            rank_packed_to_logical=torch.arange(4),
            logical_token_count=4,
        ),
    )
    monkeypatch.setattr(
        fa_backend,
        "get_sharded_kv_cp_group",
        lambda: pytest.fail("layout validation must precede KV communication"),
    )

    with pytest.raises(RuntimeError, match="contracted Q runtime"):
        backend._forward_extend_sharded_kv(
            torch.empty((4, 2)),
            torch.empty((4, 1, 2)),
            torch.empty((4, 1, 2)),
            SimpleNamespace(
                is_cross_attention=False,
                tp_q_head_num=1,
                tp_k_head_num=1,
                tp_v_head_num=1,
                head_dim=2,
                v_head_dim=2,
                scaling=1.0,
                logit_cap=0.0,
            ),
            SimpleNamespace(
                attn_cp_prefill_runtime_layout=runtime,
                attn_cp_prefill_kv_gather_plan=gather_plan,
            ),
            FlashAttentionMetadata(),
            torch.empty((2, 4, 1, 2)),
            torch.empty((2, 4, 1, 2)),
            use_welm_custom_last_q=True,
            window_size=(-1, -1),
            causal=True,
            kwargs={},
        )


def test_phase2_kv_mirror_prefix_hit_uses_cached_prefix_and_local_extend(
    monkeypatch,
):
    backend = _make_backend(page_size=4)
    backend.kv_cache_dtype_str = "auto"
    spec = build_cp_prefill_split_spec(
        extend_start=4,
        extend_len=8,
        cp_size=2,
        page_size=4,
        owner_rotation=1,
    )
    last_owner = spec.blocks[-1].owner_rank
    full_cache_loc = torch.zeros(8, dtype=torch.int64)
    for block in spec.local_blocks(last_owner):
        relative_start = block.logical_start - spec.extend_start
        full_cache_loc[relative_start : relative_start + block.token_count] = (
            torch.arange(block.token_count) + 16 + block.logical_start
        )
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=last_owner,
        input_ids=torch.arange(8),
        positions=torch.arange(4, 12, dtype=torch.int32),
        out_cache_loc=full_cache_loc,
    )
    kv_extend_indices = runtime.local_extend_indices.clone()
    runtime = contract_cp_prefill_runtime_to_last_q(runtime)

    prefix_map = torch.tensor([0, 2, 1, 3])
    prefix_sizes = (2, 2)
    local_prefix_indices = prefix_map[
        sum(prefix_sizes[:last_owner]) : sum(prefix_sizes[: last_owner + 1])
    ]
    local_prefix_slots = torch.tensor([4, 6], dtype=torch.int64)
    extend_map = torch.tensor(
        [
            logical_index - spec.extend_start
            for rank in range(2)
            for block in spec.local_blocks(rank)
            for logical_index in range(
                block.logical_start,
                block.logical_start + block.token_count,
            )
        ]
    )
    gather_plan = CPPrefillKVGatherPlan(
        prefix=CPKVGatherSegmentPlan(
            sizes=prefix_sizes,
            local_physical_slots=local_prefix_slots,
            rank_packed_to_logical=prefix_map,
            logical_token_count=4,
        ),
        extend=CPKVGatherSegmentPlan(
            sizes=spec.per_rank_tokens,
            local_physical_slots=runtime.local_out_cache_loc,
            rank_packed_to_logical=extend_map,
            logical_token_count=8,
        ),
    )

    prefix_k = torch.arange(8, dtype=torch.float32).view(4, 1, 2) / 10
    prefix_v = prefix_k + 1
    extend_k = torch.arange(16, dtype=torch.float32).view(8, 1, 2) / 10 + 2
    extend_v = extend_k + 1
    key_cache = torch.zeros((8, 4, 1, 2), dtype=torch.float32)
    value_cache = torch.zeros_like(key_cache)
    pages = torch.div(local_prefix_slots, 4, rounding_mode="floor")
    offsets = local_prefix_slots % 4
    key_cache[pages, offsets] = prefix_k.index_select(0, local_prefix_indices)
    value_cache[pages, offsets] = prefix_v.index_select(0, local_prefix_indices)

    packed_prefix_k = prefix_k.index_select(0, prefix_map)
    packed_prefix_v = prefix_v.index_select(0, prefix_map)
    packed_extend_k = extend_k.index_select(0, extend_map)
    packed_extend_v = extend_v.index_select(0, extend_map)
    cp_group = _FakeGatherCPGroup(
        [
            (packed_prefix_k, packed_prefix_v),
            (packed_extend_k, packed_extend_v),
        ],
        rank=last_owner,
        world_size=2,
    )
    monkeypatch.setattr(fa_backend, "get_sharded_kv_cp_group", lambda: cp_group)
    fa_calls = []

    def reference_flash_attn(**kwargs):
        fa_calls.append(kwargs)
        q_rows = kwargs["q"][:, 0]
        k_rows = kwargs["k"][:, 0]
        v_rows = kwargs["v"][:, 0]
        weights = torch.softmax(q_rows @ k_rows.T * kwargs["softmax_scale"], dim=-1)
        return (weights @ v_rows).unsqueeze(1)

    monkeypatch.setattr(
        fa_backend, "flash_attn_varlen_func", reference_flash_attn
    )
    q = torch.tensor([[0.25, -0.5]])
    local_k = extend_k.index_select(0, kv_extend_indices)
    local_v = extend_v.index_select(0, kv_extend_indices)

    result = backend._forward_extend_sharded_kv(
        q,
        local_k,
        local_v,
        SimpleNamespace(
            is_cross_attention=False,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=2,
            v_head_dim=2,
            scaling=0.5,
            logit_cap=0.0,
        ),
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=runtime,
            attn_cp_prefill_kv_gather_plan=gather_plan,
        ),
        FlashAttentionMetadata(),
        key_cache,
        value_cache,
        use_welm_custom_last_q=True,
        window_size=(-1, -1),
        causal=True,
        kwargs={},
    )

    full_k = torch.cat((prefix_k, extend_k), dim=0)
    full_v = torch.cat((prefix_v, extend_v), dim=0)
    expected_weights = torch.softmax(q @ full_k[:, 0].T * 0.5, dim=-1)
    expected = expected_weights @ full_v[:, 0]
    assert torch.allclose(result, expected)
    assert len(fa_calls) == 1
    assert torch.equal(fa_calls[0]["k"], full_k)
    assert torch.equal(fa_calls[0]["v"], full_v)
    assert fa_calls[0]["max_seqlen_q"] == 1
    assert fa_calls[0]["max_seqlen_k"] == 12
    assert runtime.q_blocks[0].logical_start == 11
    assert torch.equal(
        cp_group.calls[0][0][0], prefix_k.index_select(0, local_prefix_indices)
    )
    assert torch.equal(cp_group.calls[1][0][0], local_k)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_welm_mirror_projection_selects_local_q_from_precontracted_runtime(
    monkeypatch, cp_rank
):
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=8,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    full_cache_loc = torch.zeros(8, dtype=torch.int64)
    for block in spec.local_blocks(cp_rank):
        full_cache_loc[
            block.logical_start : block.logical_start + block.token_count
        ] = torch.arange(block.token_count) + 8 + block.logical_start
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=cp_rank,
        input_ids=torch.arange(8),
        positions=torch.arange(8, dtype=torch.int32),
        out_cache_loc=full_cache_loc,
    )
    original_kv_rows = runtime.active_local_tokens
    hidden = torch.arange(original_kv_rows * 2, dtype=torch.float32).view(-1, 2)
    k = torch.arange(original_kv_rows * 2, dtype=torch.float32).view(-1, 2) + 50
    v = k + 100
    batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=runtime,
        welm_kv_mirror_last_q_indices=torch.tensor([7]),
        enable_welm_kv_mirror_opt=True,
        return_logprob=False,
        forward_mode=SimpleNamespace(
            is_extend_without_speculative=lambda: True,
            is_draft_extend=lambda include_v2=False: False,
        ),
    )
    assert welmv4_model._welm_init_kv_mirror_last_q_indices(batch)
    welmv4_model._welm_contract_cp_prefill_kv_mirror_layout(batch)
    projection = object.__new__(welmv4_model.MirrorQProjection)
    torch.nn.Module.__init__(projection)
    projection.imitated_layer_idx = 0
    projection.mirror_layer_idx = None
    monkeypatch.setattr(
        welmv4_model.MirrorQProjection,
        "_apply_qkv",
        lambda _self, selected: selected,
    )

    q, returned_k, returned_v, projected_hidden = projection.forward(
        SimpleNamespace(need_clear_kv_cache=False),
        hidden,
        batch,
        {0: (k, v)},
    )

    contracted = batch.attn_cp_prefill_runtime_layout
    last_owner = spec.blocks[-1].owner_rank
    expected_q_rows = 1 if cp_rank == last_owner else 0
    assert projected_hidden.shape[0] == expected_q_rows
    assert torch.equal(q, projected_hidden)
    assert torch.equal(returned_k, k)
    assert torch.equal(returned_v, v)
    assert contracted.q_is_contracted
    assert contracted.active_local_tokens == expected_q_rows
    assert contracted.kv_local_tokens == original_kv_rows
    assert contracted.local_out_cache_loc.numel() == original_kv_rows


@pytest.mark.parametrize("cp_rank", range(4))
def test_welm_kv_mirror_contracts_zero_row_ranks_before_dispatch(cp_rank):
    spec = build_cp_prefill_split_spec(
        extend_start=16,
        extend_len=1,
        cp_size=4,
        page_size=4,
        owner_rotation=0,
    )
    out_cache_loc = torch.zeros(1, dtype=torch.int64)
    if spec.per_rank_tokens[cp_rank] != 0:
        out_cache_loc[0] = 8
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=cp_rank,
        input_ids=torch.tensor([7]),
        positions=torch.tensor([16], dtype=torch.int32),
        out_cache_loc=out_cache_loc,
    )
    prefix_slots = (
        torch.tensor([4], dtype=torch.int64)
        if cp_rank == 1
        else torch.empty((0,), dtype=torch.int64)
    )
    batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=runtime,
        attn_cp_prefill_kv_gather_plan=SimpleNamespace(
            prefix=SimpleNamespace(local_physical_slots=prefix_slots)
        ),
        welm_kv_mirror_last_q_indices=torch.tensor([0]),
    )

    contracted = welmv4_model._welm_prepare_cp_prefill_kv_mirror_layout(batch)

    last_owner = spec.blocks[-1].owner_rank
    assert contracted.q_is_contracted
    assert contracted.active_tokens_per_cp_rank() == tuple(
        1 if rank == last_owner else 0 for rank in range(4)
    )
    assert contracted.active_local_tokens == (1 if cp_rank == last_owner else 0)
    assert contracted.kv_local_tokens == spec.per_rank_tokens[cp_rank]
    should_dispatch = _welm_should_dispatch_attention(
        contracted.active_local_tokens,
        batch,
        needs_empty_dp_collectives=False,
    )
    assert should_dispatch == (cp_rank in (last_owner, 1))
    assert welmv4_model._welm_cp_prefill_prefix_send_only(
        contracted.active_local_tokens, batch
    ) == (cp_rank == 1)


def test_welm_kv_mirror_middle_chunk_contracts_all_q_rows_to_zero():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=8,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=spec.blocks[-1].owner_rank,
        input_ids=torch.arange(8),
        positions=torch.arange(8, dtype=torch.int32),
        out_cache_loc=torch.arange(8, dtype=torch.int64) + 8,
    )
    batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=runtime,
        welm_kv_mirror_last_q_indices=torch.empty((0,), dtype=torch.long),
    )

    contracted = welmv4_model._welm_prepare_cp_prefill_kv_mirror_layout(batch)

    assert batch.custom_last_index.numel() == 0
    assert batch.kv_mirror_output_size == 0
    assert contracted.active_tokens_per_cp_rank() == (0, 0)
    assert contracted.active_local_tokens == 0
    assert contracted.kv_local_tokens == runtime.kv_local_tokens


def test_phase2_nextn_mirror_fails_before_projection():
    projection = object.__new__(welmv4_model.NextnMirrorQProjection)
    torch.nn.Module.__init__(projection)

    with pytest.raises(NotImplementedError, match="NextN"):
        projection.forward(
            SimpleNamespace(),
            torch.empty((1, 2)),
            SimpleNamespace(attn_cp_prefill_runtime_layout=object()),
            {},
        )


def test_phase2_missing_gather_plan_fails_instead_of_dense_fallback(monkeypatch):
    backend = _make_backend(page_size=4)
    backend.kv_cache_dtype_str = "auto"
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=1,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=0,
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0], dtype=torch.int32),
        out_cache_loc=torch.tensor([4]),
    )
    monkeypatch.setattr(
        backend,
        "_flash_attn_sharded_kv_dense",
        lambda *_args, **_kwargs: pytest.fail("dense fallback must not run"),
    )
    layer = SimpleNamespace(is_cross_attention=False)

    with pytest.raises(RuntimeError, match="dense fallback is disabled"):
        backend._forward_extend_sharded_kv(
            torch.empty((1, 2)),
            torch.empty((1, 1, 2)),
            torch.empty((1, 1, 2)),
            layer,
            SimpleNamespace(
                attn_cp_prefill_runtime_layout=runtime,
                attn_cp_prefill_kv_gather_plan=None,
            ),
            FlashAttentionMetadata(),
            torch.empty((2, 4, 1, 2)),
            torch.empty((2, 4, 1, 2)),
            use_welm_custom_last_q=False,
            window_size=(-1, -1),
            causal=True,
            kwargs={},
        )


def test_phase2_cache_write_skip_is_only_valid_for_prefix_send_owner():
    prefix_owner_plan = SimpleNamespace(
        prefix=SimpleNamespace(local_physical_slots=torch.tensor([4]))
    )
    empty_prefix_plan = SimpleNamespace(
        prefix=SimpleNamespace(
            local_physical_slots=torch.empty((0,), dtype=torch.int64)
        )
    )

    FlashAttentionBackend._validate_cp_prefill_kv_write_mode(
        SimpleNamespace(active_local_tokens=0),
        prefix_owner_plan,
        save_kv_cache=False,
    )
    with pytest.raises(RuntimeError, match="prefix-send-only"):
        FlashAttentionBackend._validate_cp_prefill_kv_write_mode(
            SimpleNamespace(active_local_tokens=1),
            prefix_owner_plan,
            save_kv_cache=False,
        )
    with pytest.raises(RuntimeError, match="prefix-send-only"):
        FlashAttentionBackend._validate_cp_prefill_kv_write_mode(
            SimpleNamespace(active_local_tokens=0),
            empty_prefix_plan,
            save_kv_cache=False,
        )


def test_phase2_suffix_parallel_fails_before_old_suffix_path(monkeypatch):
    backend = _make_backend()
    backend.kv_cache_dtype_str = "auto"
    monkeypatch.setattr(
        backend,
        "_per_suffix_scp_attn_compute",
        lambda **_kwargs: pytest.fail("Phase 2 must not enter the suffix path"),
    )
    layer = SimpleNamespace(
        is_cross_attention=False,
        sliding_window_size=-1,
        scale_seq_attn_per_suffix=True,
        suffix_parallel=True,
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=SimpleNamespace(active_local_tokens=1),
        attn_cp_prefill_kv_gather_plan=SimpleNamespace(
            prefix=SimpleNamespace(
                local_physical_slots=torch.empty((0,), dtype=torch.int64)
            )
        ),
    )

    with pytest.raises(NotImplementedError, match="suffix parallel"):
        backend.forward_extend(
            torch.empty((1, 2)),
            torch.empty((1, 1, 2)),
            torch.empty((1, 1, 2)),
            layer,
            forward_batch,
            save_kv_cache=True,
        )


def test_welm_dispatches_empty_q_only_for_owned_prefix_kv():
    empty_prefix = SimpleNamespace(
        local_physical_slots=torch.empty((0,), dtype=torch.int64)
    )
    owned_prefix = SimpleNamespace(local_physical_slots=torch.tensor([4]))
    assert not _welm_should_dispatch_attention(
        0,
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=object(),
            attn_cp_prefill_kv_gather_plan=SimpleNamespace(prefix=empty_prefix),
        ),
        needs_empty_dp_collectives=False,
    )
    assert _welm_should_dispatch_attention(
        0,
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=object(),
            attn_cp_prefill_kv_gather_plan=SimpleNamespace(prefix=owned_prefix),
        ),
        needs_empty_dp_collectives=False,
    )
    assert welmv4_model._welm_cp_prefill_prefix_send_only(
        0,
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=object(),
            attn_cp_prefill_kv_gather_plan=SimpleNamespace(prefix=owned_prefix),
        ),
    )
    assert not welmv4_model._welm_cp_prefill_prefix_send_only(
        1,
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=object(),
            attn_cp_prefill_kv_gather_plan=SimpleNamespace(prefix=owned_prefix),
        ),
    )

    mirror_kv_runtime = SimpleNamespace(
        active_local_tokens=0,
        kv_local_tokens=4,
    )
    mirror_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=mirror_kv_runtime,
        attn_cp_prefill_kv_gather_plan=SimpleNamespace(prefix=empty_prefix),
    )
    assert _welm_should_dispatch_attention(
        0, mirror_batch, needs_empty_dp_collectives=False
    )
    assert not welmv4_model._welm_cp_prefill_prefix_send_only(0, mirror_batch)


def test_welm_prefix_owner_bypasses_zero_row_qkv_projection():
    class FailQKVProjection:
        def forward(self, *_args, **_kwargs):
            pytest.fail("prefix-send-only rank must bypass QKV projection")

    class RecordingAttention:
        tp_q_head_num = 2
        tp_k_head_num = 1
        tp_v_head_num = 1
        head_dim = 4
        qk_head_dim = 4
        v_head_dim = 4

        def __init__(self):
            self.calls = []

        def __call__(self, q, k, v, forward_batch, save_kv_cache=True):
            self.calls.append((q, k, v, forward_batch, save_kv_cache))
            return q

    attention = object.__new__(welmv4_model.Qwen2MoeAttention)
    torch.nn.Module.__init__(attention)
    attention.layer_idx = 0
    attention.is_nextn = False
    attention.suffix_parallel = False
    attention.hidden_size = 8
    attention.qkv_proj = FailQKVProjection()
    attention.attn = RecordingAttention()
    owned_prefix = SimpleNamespace(local_physical_slots=torch.tensor([4]))
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=object(),
        attn_cp_prefill_kv_gather_plan=SimpleNamespace(prefix=owned_prefix),
    )

    output = attention.forward(
        positions=torch.empty((0,), dtype=torch.int32),
        hidden_states=torch.empty((0, 8)),
        forward_batch=forward_batch,
        kv_mirror_states={},
    )

    assert output.shape == (0, 8)
    assert len(attention.attn.calls) == 1
    q, k, v, passed_batch, save_kv_cache = attention.attn.calls[0]
    assert q.shape == (0, 8)
    assert k.shape == (0, 4)
    assert v.shape == (0, 4)
    assert passed_batch is forward_batch
    assert not save_kv_cache


def test_phase2_split_spec_without_runtime_fails_fast():
    with pytest.raises(RuntimeError, match="refusing dense fallback"):
        FlashAttentionBackend._validate_cp_prefill_runtime(
            SimpleNamespace(
                forward_mode=fa_backend.ForwardMode.EXTEND,
                attn_cp_prefill_split_specs=(object(),),
                attn_cp_prefill_runtime_layout=None,
            )
        )

    FlashAttentionBackend._validate_cp_prefill_runtime(
        SimpleNamespace(
            forward_mode=fa_backend.ForwardMode.DECODE,
            attn_cp_prefill_split_specs=(object(),),
            attn_cp_prefill_runtime_layout=None,
        )
    )
    assert not _welm_should_dispatch_attention(
        0,
        SimpleNamespace(attn_cp_prefill_runtime_layout=None),
        needs_empty_dp_collectives=False,
    )


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


def test_sharded_kv_decode_swa_metadata_keeps_physical_swa_pages():
    backend = _make_backend(page_size=4)
    backend.has_swa = True
    backend.sliding_window_size = 15

    class Resolver:
        def resolve_full(self, page_table):
            raise AssertionError(
                f"physical SWA pages must not be resolved as logical full pages: {page_table}"
            )

    backend.cp_sharded_page_table_resolver = Resolver()
    metadata = FlashAttentionMetadata()
    physical_swa_pages = torch.tensor([[101, 0, 102, 103]], dtype=torch.int32)

    backend._set_sharded_kv_decode_swa_metadata(
        metadata,
        physical_swa_pages,
        torch.tensor([16], dtype=torch.int32),
    )

    assert metadata.cp_swa_local_cache_seqlens_int32.tolist() == [12]
    assert metadata.cp_swa_local_page_table[0, :3].tolist() == [101, 102, 103]


def test_sharded_kv_dense_decode_window_keeps_physical_swa_pages(monkeypatch):
    backend = _make_backend(page_size=4)
    backend.forward_metadata = FlashAttentionMetadata(
        cache_seqlens_int32=torch.tensor([8], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        page_table=torch.tensor([[31, 32]], dtype=torch.int32),
        swa_page_table=torch.tensor([[101, 102]], dtype=torch.int32),
    )
    backend.use_mla = False
    backend.kv_cache_dtype_str = "auto"
    backend.has_local_attention = False
    backend.use_sliding_window_kv_pool = True
    backend.enable_attn_cp_decode_local_merge = False
    backend.enable_attn_cp_decode_local_merge_swa = False
    backend.enable_attn_cp_zero_dummy_slot = False
    backend.decode_cuda_graph_metadata = {}
    backend.attncp_dense_window_static_tensors = {}

    class Resolver:
        def resolve_full(self, page_table):
            raise AssertionError(
                f"physical SWA pages must not be resolved as logical full pages: {page_table}"
            )

    class CPGroup:
        @staticmethod
        def all_reduce_coalesced(tensors):
            return tensors

    backend.cp_sharded_page_table_resolver = Resolver()
    monkeypatch.setattr(fa_backend, "is_cp_kv_sharded", lambda: True)
    monkeypatch.setattr(fa_backend, "get_sharded_kv_cp_group", lambda: CPGroup())

    class Pool:
        def __init__(self):
            self.key_cache = torch.zeros((103 * 4, 1, 1), dtype=torch.float32)
            self.value_cache = torch.zeros_like(self.key_cache)
            self.key_cache[102 * 4 : 103 * 4, 0, 0] = torch.tensor(
                [1.0, 2.0, 3.0, 4.0]
            )
            self.value_cache[102 * 4 : 103 * 4, 0, 0] = torch.tensor(
                [11.0, 12.0, 13.0, 14.0]
            )

        def get_kv_buffer(self, layer_id):
            del layer_id
            return self.key_cache, self.value_cache

        @staticmethod
        def is_swa_layer(layer_id):
            del layer_id
            return True

    captured = {}

    def fake_flash_attn_with_kvcache(**kwargs):
        captured["k"] = kwargs["k_cache"][:, 0, 0, 0].clone()
        captured["v"] = kwargs["v_cache"][:, 0, 0, 0].clone()
        return torch.zeros((1, 1, 1), dtype=torch.float32)

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
        sliding_window_size=3,
        tp_k_head_num=1,
        tp_v_head_num=1,
        tp_q_head_num=1,
        head_dim=1,
        v_head_dim=1,
        scaling=1.0,
        logit_cap=0.0,
    )

    backend.forward_decode(
        q=torch.zeros((1, 1, 1), dtype=torch.float32),
        k=None,
        v=None,
        layer=layer,
        forward_batch=forward_batch,
        save_kv_cache=False,
    )

    assert captured["k"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert captured["v"].tolist() == [11.0, 12.0, 13.0, 14.0]


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


def test_phase2_compact_swa_prefill_does_not_require_generic_swa_page_table(
    monkeypatch,
):
    backend = _make_backend(page_size=16)
    backend.forward_metadata = FlashAttentionMetadata(
        cache_seqlens_int32=torch.tensor([32], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 32], dtype=torch.int32),
        max_seq_len_q=1,
        page_table=torch.tensor([[201, 202]], dtype=torch.int32),
        swa_page_table=None,
    )
    backend.use_mla = False
    backend.kv_cache_dtype_str = "auto"
    backend.has_local_attention = False
    backend.use_sliding_window_kv_pool = True
    backend.is_welm_v4_model = False
    backend.topk = 0
    backend.fa_skip_kv_cache = False
    backend.attn_cp_size = 2
    captured = {}

    class Pool:
        @staticmethod
        def get_kv_buffer(layer_id):
            del layer_id
            shape = (2 * 16, 1, 2)
            return torch.zeros(shape), torch.zeros(shape)

        @staticmethod
        def is_swa_layer(layer_id):
            del layer_id
            return True

    class ExtendMode:
        @staticmethod
        def is_context_parallel_extend():
            return False

        @staticmethod
        def is_target_verify():
            return False

    def fake_sharded_prefill(*args, **kwargs):
        captured["page_table"] = kwargs["page_table"]
        return torch.zeros((1, 2), dtype=torch.float32)

    monkeypatch.setattr(fa_backend, "is_cp_kv_sharded", lambda: True)
    monkeypatch.setattr(backend, "_forward_extend_sharded_kv", fake_sharded_prefill)
    runtime_layout = SimpleNamespace(active_local_tokens=1, kv_local_tokens=1)
    pool = Pool()
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=runtime_layout,
        attn_cp_prefill_kv_gather_plan=object(),
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
        save_kv_cache=True,
    )

    assert captured["page_table"] is backend.forward_metadata.page_table


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

    def fake_gather(
        page_table,
        cache_seqlens,
        key_cache,
        value_cache,
        *,
        page_table_is_physical=False,
    ):
        assert not page_table_is_physical
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

    def fake_gather(
        page_table,
        cache_seqlens,
        key_cache,
        value_cache,
        *,
        page_table_is_physical=False,
    ):
        assert not page_table_is_physical
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
