import inspect
import math

import pytest
import torch

from sgl_kernel import merge_state_v2
from sglang.srt.layers.attention import attncp_fused_ops as fused_ops
from sglang.srt.layers.attention.attncp_fused_ops import (
    attncp_cp2_fused_q_fa_decode,
    attncp_cp2_fused_q_fa_supports_shape,
    attncp_sharded_kv_local_cap,
)
from sglang.jit_kernel.flash_attention import (
    flash_attn_with_kvcache,
)


def _reference_cp2_local_decode(
    q_local,
    q_peer,
    key_cache,
    value_cache,
    page_table,
    cache_seqlens,
    *,
    cp_rank,
    softmax_scale,
    softcap,
    window_left,
    sinks,
):
    batch_size, local_q_heads, head_dim = q_local.shape
    full_q_heads = local_q_heads * 2
    num_kv_heads = key_cache.shape[2]
    q_heads_per_kv = full_q_heads // num_kv_heads
    q_full = torch.empty(
        batch_size,
        full_q_heads,
        head_dim,
        dtype=q_local.dtype,
        device=q_local.device,
    )
    if cp_rank == 0:
        q_full[:, :local_q_heads, :] = q_local
        q_full[:, local_q_heads:, :] = q_peer
    else:
        q_full[:, :local_q_heads, :] = q_peer
        q_full[:, local_q_heads:, :] = q_local

    out = torch.zeros_like(q_full)
    lse = torch.empty(
        batch_size, full_q_heads, dtype=torch.float32, device=q_local.device
    )
    for batch_idx in range(batch_size):
        seq_len = int(cache_seqlens[batch_idx].item())
        for q_head in range(full_q_heads):
            if seq_len == 0:
                lse[batch_idx, q_head] = (
                    sinks[q_head].float() if sinks is not None else -float("inf")
                )
                continue

            kv_head = q_head // q_heads_per_kv
            start = 0
            if window_left >= 0:
                start = max(seq_len - window_left - 1, 0)
            pages = page_table[batch_idx, start:seq_len].long()
            k = key_cache[pages, 0, kv_head, :].float()
            v = value_cache[pages, 0, kv_head, :].float()
            q = q_full[batch_idx, q_head, :].float()
            scores = torch.matmul(k, q) * softmax_scale
            if softcap > 0:
                scores = torch.tanh(scores / softcap) * softcap

            if sinks is None:
                probs = torch.softmax(scores, dim=-1)
                lse[batch_idx, q_head] = torch.logsumexp(scores, dim=-1)
            else:
                sink = sinks[q_head].float()
                max_score = torch.maximum(scores.max(), sink)
                denom = torch.exp(scores - max_score).sum() + torch.exp(
                    sink - max_score
                )
                probs = torch.exp(scores - max_score) / denom
                lse[batch_idx, q_head] = torch.log(denom) + max_score
            out[batch_idx, q_head, :] = torch.matmul(probs, v).to(out.dtype)
    return out, lse


def _normalize_fa_lse(lse, batch_size, num_heads):
    if lse.shape == (num_heads, batch_size):
        return lse.T.contiguous()
    if lse.shape == (batch_size, num_heads):
        return lse.contiguous()
    if lse.shape == (batch_size, num_heads, 1):
        return lse[:, :, 0].contiguous()
    if lse.shape == (num_heads, batch_size, 1):
        return lse[:, :, 0].T.contiguous()
    raise AssertionError(f"unexpected FA LSE shape: {tuple(lse.shape)}")


class _FakeTritonKernel:
    def __init__(self):
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)

        def _launch(*args, **kwargs):
            return None

        return _launch


def test_attncp_cp2_fused_q_fa_shape_guard_preserves_kv_stationary_contract():
    assert attncp_cp2_fused_q_fa_supports_shape(6, 1)
    assert attncp_cp2_fused_q_fa_supports_shape(6, 2)
    assert not attncp_cp2_fused_q_fa_supports_shape(17, 1)
    assert not attncp_cp2_fused_q_fa_supports_shape(6, 5)
    assert not attncp_cp2_fused_q_fa_supports_shape(6, 1, cp_world_size=4)


def test_attncp_cp2_fused_q_fa_kernel_source_stays_kv_stationary():
    for kernel in (
        fused_ops._attncp_cp2_fused_q_fa_decode_kernel,
        fused_ops._attncp_cp2_fused_q_fa_decode_split_kernel,
    ):
        source = inspect.getsource(kernel.fn)
        assert "q_head_idx = tl.program_id" not in source
        assert source.count("tl.load(key_cache") == 1
        assert source.count("tl.load(value_cache") == 1


@pytest.mark.parametrize(
    ("max_seq_len", "chunk_size", "cp_size", "expected"),
    [
        (0, 1024, 2, [0, 0]),
        (1, 1024, 2, [1, 1]),
        (1024, 1024, 2, [1024, 1]),
        (1025, 1024, 2, [1024, 1]),
        (2048, 1024, 2, [1024, 1024]),
        (2049, 1024, 2, [1025, 1024]),
        (40960, 1024, 2, [20480, 20480]),
        (5120, 1024, 4, [2048, 1024, 1024, 1024]),
        (5121, 1024, 4, [2048, 1025, 1024, 1024]),
    ],
)
def test_attncp_sharded_kv_local_cap_matches_chunk_owner_distribution(
    max_seq_len, chunk_size, cp_size, expected
):
    actual = [
        attncp_sharded_kv_local_cap(
            max_seq_len,
            cp_rank=cp_rank,
            cp_size=cp_size,
            chunk_size=chunk_size,
        )
        for cp_rank in range(cp_size)
    ]
    assert actual == expected


def test_attncp_sharded_kv_local_cap_rejects_invalid_rank():
    with pytest.raises(ValueError, match="Invalid cp_rank"):
        attncp_sharded_kv_local_cap(
            1024,
            cp_rank=2,
            cp_size=2,
            chunk_size=1024,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_attncp_cp2_fused_q_fa_decode_launch_grid_is_kv_stationary(monkeypatch):
    non_split_kernel = _FakeTritonKernel()
    split_kernel = _FakeTritonKernel()
    merge_kernel = _FakeTritonKernel()
    monkeypatch.setattr(
        fused_ops, "_attncp_cp2_fused_q_fa_decode_kernel", non_split_kernel
    )
    monkeypatch.setattr(
        fused_ops, "_attncp_cp2_fused_q_fa_decode_split_kernel", split_kernel
    )
    monkeypatch.setattr(fused_ops, "_attncp_cp2_merge_fa_splits_kernel", merge_kernel)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    local_q_heads = 6
    full_q_heads = local_q_heads * 2
    num_kv_heads = 1
    head_dim = 64
    q_local = torch.empty(batch_size, local_q_heads, head_dim, device=device, dtype=dtype)
    q_peer = torch.empty_like(q_local)
    out = torch.empty(batch_size, full_q_heads, head_dim, device=device, dtype=dtype)
    lse = torch.empty(batch_size, full_q_heads, device=device, dtype=torch.float32)
    cache_lens = torch.ones(batch_size, device=device, dtype=torch.int32)

    key_cache = torch.empty(32, 1, num_kv_heads, head_dim, device=device, dtype=dtype)
    value_cache = torch.empty_like(key_cache)
    page_table = torch.zeros(batch_size, 32, device=device, dtype=torch.int32)
    fused_ops.attncp_cp2_fused_q_fa_decode(
        q_local,
        q_peer,
        key_cache,
        value_cache,
        page_table,
        cache_lens,
        out,
        lse,
        cp_rank=0,
        softmax_scale=1.0,
        page_size=1,
    )
    assert non_split_kernel.grids == [(batch_size, num_kv_heads)]

    split_o = torch.empty(8, batch_size, full_q_heads, head_dim, device=device, dtype=dtype)
    split_lse = torch.empty(
        8, batch_size, full_q_heads, device=device, dtype=torch.float32
    )
    key_cache = torch.empty(8192, 1, num_kv_heads, head_dim, device=device, dtype=dtype)
    value_cache = torch.empty_like(key_cache)
    page_table = torch.zeros(batch_size, 8192, device=device, dtype=torch.int32)
    fused_ops.attncp_cp2_fused_q_fa_decode(
        q_local,
        q_peer,
        key_cache,
        value_cache,
        page_table,
        cache_lens,
        out,
        lse,
        cp_rank=0,
        softmax_scale=1.0,
        page_size=1,
        split_o=split_o,
        split_lse=split_lse,
        max_splits=8,
    )
    assert split_kernel.grids == [(batch_size, num_kv_heads, 2)]
    assert merge_kernel.grids == [(batch_size, full_q_heads)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_attncp_cp2_fused_q_fa_decode_rejects_q_head_split_shape():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 1
    local_q_heads = 17
    full_q_heads = local_q_heads * 2
    num_kv_heads = 1
    head_dim = 16

    q_local = torch.empty(batch_size, local_q_heads, head_dim, device=device, dtype=dtype)
    q_peer = torch.empty_like(q_local)
    key_cache = torch.empty(1, 1, num_kv_heads, head_dim, device=device, dtype=dtype)
    value_cache = torch.empty_like(key_cache)
    page_table = torch.zeros(batch_size, 1, device=device, dtype=torch.int32)
    cache_lens = torch.ones(batch_size, device=device, dtype=torch.int32)
    out = torch.empty(batch_size, full_q_heads, head_dim, device=device, dtype=dtype)
    lse = torch.empty(batch_size, full_q_heads, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="KV-stationary"):
        fused_ops.attncp_cp2_fused_q_fa_decode(
            q_local,
            q_peer,
            key_cache,
            value_cache,
            page_table,
            cache_lens,
            out,
            lse,
            cp_rank=0,
            softmax_scale=1.0,
            page_size=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("softcap", [0.0, 15.0])
def test_attncp_cp2_fused_q_fa_decode_matches_reference(
    cp_rank, use_sinks, softcap
):
    torch.manual_seed(3 + cp_rank)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 3
    local_q_heads = 3
    full_q_heads = local_q_heads * 2
    num_kv_heads = 1
    head_dim = 64
    max_seq_len = 17
    num_pages = batch_size * max_seq_len + 11

    q_local = torch.randn(
        batch_size, local_q_heads, head_dim, device=device, dtype=dtype
    )
    q_peer = torch.randn_like(q_local)
    key_cache = torch.randn(
        num_pages, 1, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    page_table = torch.empty(
        batch_size, max_seq_len, device=device, dtype=torch.int32
    )
    perm = torch.randperm(num_pages, device=device, dtype=torch.int32)
    for batch_idx in range(batch_size):
        start = batch_idx * max_seq_len
        page_table[batch_idx] = perm[start : start + max_seq_len]
    cache_seqlens = torch.tensor([0, 7, 17], device=device, dtype=torch.int32)
    sinks = (
        torch.randn(full_q_heads, device=device, dtype=dtype)
        if use_sinks
        else None
    )
    out = torch.empty(
        batch_size, full_q_heads, head_dim, device=device, dtype=dtype
    )
    lse = torch.empty(batch_size, full_q_heads, device=device, dtype=torch.float32)

    attncp_cp2_fused_q_fa_decode(
        q_local,
        q_peer,
        key_cache,
        value_cache,
        page_table,
        cache_seqlens,
        out,
        lse,
        cp_rank=cp_rank,
        softmax_scale=1.0 / math.sqrt(head_dim),
        softcap=softcap,
        sinks=sinks,
        page_size=1,
    )
    ref_out, ref_lse = _reference_cp2_local_decode(
        q_local,
        q_peer,
        key_cache,
        value_cache,
        page_table,
        cache_seqlens,
        cp_rank=cp_rank,
        softmax_scale=1.0 / math.sqrt(head_dim),
        softcap=softcap,
        window_left=-1,
        sinks=sinks,
    )
    torch.testing.assert_close(out, ref_out, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(lse, ref_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("window_left", [-1, 8])
def test_attncp_cp2_fused_q_fa_decode_matches_fa3(cp_rank, use_sinks, window_left):
    torch.manual_seed(11 + cp_rank)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 4
    local_q_heads = 3
    full_q_heads = local_q_heads * 2
    num_kv_heads = 1
    head_dim = 64
    max_seq_len = 96
    num_pages = batch_size * max_seq_len + 7

    q_local = torch.randn(
        batch_size, local_q_heads, head_dim, device=device, dtype=dtype
    )
    q_peer = torch.randn_like(q_local)
    if cp_rank == 0:
        q_full = torch.cat([q_local, q_peer], dim=1).contiguous()
    else:
        q_full = torch.cat([q_peer, q_local], dim=1).contiguous()
    key_cache = torch.randn(
        num_pages, 1, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    page_table = torch.empty(
        batch_size, max_seq_len, device=device, dtype=torch.int32
    )
    perm = torch.randperm(num_pages, device=device, dtype=torch.int32)
    for batch_idx in range(batch_size):
        start = batch_idx * max_seq_len
        page_table[batch_idx] = perm[start : start + max_seq_len]
    cache_seqlens = torch.tensor([1, 17, 63, 96], device=device, dtype=torch.int32)
    sinks = (
        torch.randn(full_q_heads, device=device, dtype=dtype)
        if use_sinks
        else None
    )

    out = torch.empty(
        batch_size, full_q_heads, head_dim, device=device, dtype=dtype
    )
    lse = torch.empty(batch_size, full_q_heads, device=device, dtype=torch.float32)
    fa_out = torch.empty(
        batch_size, 1, full_q_heads, head_dim, device=device, dtype=dtype
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)

    attncp_cp2_fused_q_fa_decode(
        q_local,
        q_peer,
        key_cache,
        value_cache,
        page_table,
        cache_seqlens,
        out,
        lse,
        cp_rank=cp_rank,
        softmax_scale=softmax_scale,
        window_left=window_left,
        sinks=sinks,
        page_size=1,
    )
    fa_result = flash_attn_with_kvcache(
        q=q_full.view(batch_size, 1, full_q_heads, head_dim),
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(window_left, 0) if window_left >= 0 else (-1, -1),
        return_softmax_lse=True,
        out=fa_out,
        sinks=sinks,
        ver=3,
    )
    fa_lse = _normalize_fa_lse(fa_result[1], batch_size, full_q_heads)
    torch.testing.assert_close(out, fa_out[:, 0], atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(lse, fa_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("window_left", [-1, 512])
def test_attncp_cp2_fused_q_fa_decode_split_matches_fa3(
    cp_rank, use_sinks, window_left
):
    torch.manual_seed(31 + cp_rank)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    local_q_heads = 3
    full_q_heads = local_q_heads * 2
    num_kv_heads = 1
    head_dim = 64
    max_seq_len = 4608
    num_pages = batch_size * max_seq_len + 17

    q_local = torch.randn(
        batch_size, local_q_heads, head_dim, device=device, dtype=dtype
    )
    q_peer = torch.randn_like(q_local)
    if cp_rank == 0:
        q_full = torch.cat([q_local, q_peer], dim=1).contiguous()
    else:
        q_full = torch.cat([q_peer, q_local], dim=1).contiguous()
    key_cache = torch.randn(
        num_pages, 1, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    page_table = torch.empty(
        batch_size, max_seq_len, device=device, dtype=torch.int32
    )
    perm = torch.randperm(num_pages, device=device, dtype=torch.int32)
    for batch_idx in range(batch_size):
        start = batch_idx * max_seq_len
        page_table[batch_idx] = perm[start : start + max_seq_len]
    cache_seqlens = torch.tensor([4097, 4608], device=device, dtype=torch.int32)
    sinks = (
        torch.randn(full_q_heads, device=device, dtype=dtype)
        if use_sinks
        else None
    )

    out = torch.empty(
        batch_size, full_q_heads, head_dim, device=device, dtype=dtype
    )
    lse = torch.empty(batch_size, full_q_heads, device=device, dtype=torch.float32)
    split_o = torch.empty(
        8, batch_size, full_q_heads, head_dim, device=device, dtype=dtype
    )
    split_lse = torch.empty(
        8, batch_size, full_q_heads, device=device, dtype=torch.float32
    )
    fa_out = torch.empty(
        batch_size, 1, full_q_heads, head_dim, device=device, dtype=dtype
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)

    attncp_cp2_fused_q_fa_decode(
        q_local,
        q_peer,
        key_cache,
        value_cache,
        page_table,
        cache_seqlens,
        out,
        lse,
        cp_rank=cp_rank,
        softmax_scale=softmax_scale,
        window_left=window_left,
        sinks=sinks,
        page_size=1,
        split_o=split_o,
        split_lse=split_lse,
        max_splits=8,
    )
    fa_result = flash_attn_with_kvcache(
        q=q_full.view(batch_size, 1, full_q_heads, head_dim),
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(window_left, 0) if window_left >= 0 else (-1, -1),
        return_softmax_lse=True,
        out=fa_out,
        sinks=sinks,
        ver=3,
    )
    fa_lse = _normalize_fa_lse(fa_result[1], batch_size, full_q_heads)
    torch.testing.assert_close(out, fa_out[:, 0], atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(lse, fa_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_attncp_cp2_fused_q_fa_decode_head_dim_256_split_matches_fa3(cp_rank):
    torch.manual_seed(61 + cp_rank)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 1
    local_q_heads = 6
    full_q_heads = local_q_heads * 2
    num_kv_heads = 1
    head_dim = 256
    max_seq_len = 4608
    num_pages = batch_size * max_seq_len + 11

    q_local = torch.randn(
        batch_size, local_q_heads, head_dim, device=device, dtype=dtype
    )
    q_peer = torch.randn_like(q_local)
    if cp_rank == 0:
        q_full = torch.cat([q_local, q_peer], dim=1).contiguous()
    else:
        q_full = torch.cat([q_peer, q_local], dim=1).contiguous()
    key_cache = torch.randn(
        num_pages, 1, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    page_table = torch.randperm(
        num_pages, device=device, dtype=torch.int32
    )[:max_seq_len].view(batch_size, max_seq_len)
    cache_seqlens = torch.tensor([max_seq_len], device=device, dtype=torch.int32)
    sinks = torch.randn(full_q_heads, device=device, dtype=dtype)

    out = torch.empty(
        batch_size, full_q_heads, head_dim, device=device, dtype=dtype
    )
    lse = torch.empty(batch_size, full_q_heads, device=device, dtype=torch.float32)
    split_o = torch.empty(
        4, batch_size, full_q_heads, head_dim, device=device, dtype=dtype
    )
    split_lse = torch.empty(
        4, batch_size, full_q_heads, device=device, dtype=torch.float32
    )
    fa_out = torch.empty(
        batch_size, 1, full_q_heads, head_dim, device=device, dtype=dtype
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)
    window_left = 512

    attncp_cp2_fused_q_fa_decode(
        q_local,
        q_peer,
        key_cache,
        value_cache,
        page_table,
        cache_seqlens,
        out,
        lse,
        cp_rank=cp_rank,
        softmax_scale=softmax_scale,
        window_left=window_left,
        sinks=sinks,
        page_size=1,
        split_o=split_o,
        split_lse=split_lse,
        max_splits=4,
    )
    fa_result = flash_attn_with_kvcache(
        q=q_full.view(batch_size, 1, full_q_heads, head_dim),
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(window_left, 0),
        return_softmax_lse=True,
        out=fa_out,
        sinks=sinks,
        ver=3,
    )
    fa_lse = _normalize_fa_lse(fa_result[1], batch_size, full_q_heads)
    torch.testing.assert_close(out, fa_out[:, 0], atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(lse, fa_lse, atol=8e-2, rtol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("max_splits", [1, 4])
def test_attncp_cp2_fused_q_fa_decode_matches_service_merge(max_splits):
    torch.manual_seed(89 + max_splits)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 3
    local_q_heads = 3
    full_q_heads = local_q_heads * 2
    num_kv_heads = 1
    head_dim = 256
    max_seq_len = 4608
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q0 = torch.randn(batch_size, local_q_heads, head_dim, device=device, dtype=dtype)
    q1 = torch.randn_like(q0)
    q_full = torch.cat([q0, q1], dim=1).contiguous()
    sinks_full = torch.randn(full_q_heads, device=device, dtype=dtype)
    sinks_disabled = torch.full_like(sinks_full, -float("inf"))

    k0 = torch.randn(
        batch_size * max_seq_len + 17,
        1,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v0 = torch.randn_like(k0)
    k1 = torch.randn_like(k0)
    v1 = torch.randn_like(k0)
    pt0 = torch.empty(batch_size, max_seq_len, device=device, dtype=torch.int32)
    pt1 = torch.empty_like(pt0)
    perm0 = torch.randperm(k0.shape[0], device=device, dtype=torch.int32)
    perm1 = torch.randperm(k1.shape[0], device=device, dtype=torch.int32)
    for batch_idx in range(batch_size):
        start = batch_idx * max_seq_len
        end = start + max_seq_len
        pt0[batch_idx] = perm0[start:end]
        pt1[batch_idx] = perm1[start:end]
    lens0 = torch.tensor([0, 257, 4608], device=device, dtype=torch.int32)
    lens1 = torch.tensor([9, 1024, 4097], device=device, dtype=torch.int32)

    fa_o0 = torch.empty(batch_size, 1, full_q_heads, head_dim, device=device, dtype=dtype)
    fa_o1 = torch.empty_like(fa_o0)
    fa0 = flash_attn_with_kvcache(
        q=q_full.view(batch_size, 1, full_q_heads, head_dim),
        k_cache=k0,
        v_cache=v0,
        page_table=pt0,
        cache_seqlens=lens0,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        return_softmax_lse=True,
        out=fa_o0,
        sinks=sinks_full,
        ver=3,
    )
    fa1 = flash_attn_with_kvcache(
        q=q_full.view(batch_size, 1, full_q_heads, head_dim),
        k_cache=k1,
        v_cache=v1,
        page_table=pt1,
        cache_seqlens=lens1,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        return_softmax_lse=True,
        out=fa_o1,
        sinks=sinks_disabled,
        ver=3,
    )
    fa_lse0 = _normalize_fa_lse(fa0[1], batch_size, full_q_heads)
    fa_lse1 = _normalize_fa_lse(fa1[1], batch_size, full_q_heads)
    empty0 = lens0.eq(0)
    empty1 = lens1.eq(0)
    fa_o0[:, 0].masked_fill_(empty0.view(-1, 1, 1), 0)
    fa_o1[:, 0].masked_fill_(empty1.view(-1, 1, 1), 0)
    fa_lse0 = torch.where(
        empty0.view(-1, 1),
        sinks_full.float().view(1, -1).expand_as(fa_lse0),
        fa_lse0,
    )
    fa_lse1 = torch.where(
        empty1.view(-1, 1),
        sinks_disabled.float().view(1, -1).expand_as(fa_lse1),
        fa_lse1,
    )
    merged_fa_o = torch.empty_like(fa_o0[:, 0])
    merged_fa_lse = torch.empty_like(fa_lse0)
    merge_state_v2(fa_o0[:, 0], fa_lse0, fa_o1[:, 0], fa_lse1, merged_fa_o, merged_fa_lse)

    fused_o0 = torch.empty(batch_size, full_q_heads, head_dim, device=device, dtype=dtype)
    fused_lse0 = torch.empty(batch_size, full_q_heads, device=device, dtype=torch.float32)
    fused_o1 = torch.empty_like(fused_o0)
    fused_lse1 = torch.empty_like(fused_lse0)
    split_o0 = torch.empty(
        4, batch_size, full_q_heads, head_dim, device=device, dtype=dtype
    )
    split_lse0 = torch.empty(4, batch_size, full_q_heads, device=device, dtype=torch.float32)
    split_o1 = torch.empty_like(split_o0)
    split_lse1 = torch.empty_like(split_lse0)
    attncp_cp2_fused_q_fa_decode(
        q0,
        q1,
        k0,
        v0,
        pt0,
        lens0,
        fused_o0,
        fused_lse0,
        cp_rank=0,
        softmax_scale=softmax_scale,
        sinks=sinks_full,
        page_size=1,
        split_o=split_o0,
        split_lse=split_lse0,
        max_splits=max_splits,
    )
    attncp_cp2_fused_q_fa_decode(
        q1,
        q0,
        k1,
        v1,
        pt1,
        lens1,
        fused_o1,
        fused_lse1,
        cp_rank=1,
        softmax_scale=softmax_scale,
        sinks=sinks_disabled,
        page_size=1,
        split_o=split_o1,
        split_lse=split_lse1,
        max_splits=max_splits,
    )
    merged_fused_o = torch.empty_like(fused_o0)
    merged_fused_lse = torch.empty_like(fused_lse0)
    merge_state_v2(
        fused_o0, fused_lse0, fused_o1, fused_lse1, merged_fused_o, merged_fused_lse
    )

    torch.testing.assert_close(merged_fused_o, merged_fa_o, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(merged_fused_lse, merged_fa_lse, atol=8e-2, rtol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("max_splits", [1, 8])
@pytest.mark.parametrize("page_table_cap", [16384, 40960])
def test_attncp_cp2_fused_q_fa_decode_welm_shape_strict_local(
    max_splits, page_table_cap
):
    torch.manual_seed(127 + max_splits)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    local_q_heads = 6
    full_q_heads = local_q_heads * 2
    num_kv_heads = 1
    head_dim = 256
    max_seq_len = 16384
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q0 = torch.randn(batch_size, local_q_heads, head_dim, device=device, dtype=dtype)
    q1 = torch.randn_like(q0)
    q_full = torch.cat([q0, q1], dim=1).contiguous()
    key_cache = torch.randn(
        batch_size * max_seq_len,
        1,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    page_table = torch.arange(
        batch_size * max_seq_len, device=device, dtype=torch.int32
    ).view(batch_size, max_seq_len)
    if page_table_cap > max_seq_len:
        page_table_padded = torch.zeros(
            batch_size, page_table_cap, device=device, dtype=torch.int32
        )
        page_table_padded[:, :max_seq_len] = page_table
        page_table = page_table_padded
    cache_seqlens = torch.full(
        (batch_size,), max_seq_len, device=device, dtype=torch.int32
    )
    sinks = torch.randn(full_q_heads, device=device, dtype=dtype)

    fused_o = torch.empty(
        batch_size, full_q_heads, head_dim, device=device, dtype=dtype
    )
    fused_lse = torch.empty(
        batch_size, full_q_heads, device=device, dtype=torch.float32
    )
    split_o = torch.empty(
        max_splits, batch_size, full_q_heads, head_dim, device=device, dtype=dtype
    )
    split_lse = torch.empty(
        max_splits, batch_size, full_q_heads, device=device, dtype=torch.float32
    )
    fa_o = torch.empty(
        batch_size, 1, full_q_heads, head_dim, device=device, dtype=dtype
    )

    attncp_cp2_fused_q_fa_decode(
        q0,
        q1,
        key_cache,
        value_cache,
        page_table,
        cache_seqlens,
        fused_o,
        fused_lse,
        cp_rank=0,
        softmax_scale=softmax_scale,
        sinks=sinks,
        page_size=1,
        split_o=split_o,
        split_lse=split_lse,
        max_splits=max_splits,
    )
    fa_result = flash_attn_with_kvcache(
        q=q_full.view(batch_size, 1, full_q_heads, head_dim),
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        return_softmax_lse=True,
        out=fa_o,
        sinks=sinks,
        ver=3,
        num_splits=0,
    )
    fa_lse = _normalize_fa_lse(fa_result[1], batch_size, full_q_heads)

    o_diff = (fused_o - fa_o[:, 0]).abs().float()
    lse_diff = (fused_lse - fa_lse).abs().float()
    assert o_diff.max().item() <= 5e-4
    assert o_diff.mean().item() <= 6e-5
    assert lse_diff.max().item() <= 2e-5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
