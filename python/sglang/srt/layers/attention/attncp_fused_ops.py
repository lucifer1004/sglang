from __future__ import annotations

import torch
import triton
import triton.language as tl


_FUSED_Q_FA_TARGET_SPLIT_SIZE = 4096
_FUSED_Q_FA_MAX_BLOCK_H = 16
_FUSED_Q_FA_MAX_AUTO_SPLITS = 64


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def attncp_sharded_kv_local_cap(
    max_seq_len: int,
    *,
    cp_rank: int,
    cp_size: int,
    chunk_size: int,
) -> int:
    """Return the compact local KV page-table capacity for a global seq cap."""
    max_seq_len = int(max_seq_len)
    cp_rank = int(cp_rank)
    cp_size = int(cp_size)
    chunk_size = max(1, int(chunk_size))

    _check(cp_size >= 1, f"Invalid cp_size={cp_size}")
    _check(0 <= cp_rank < cp_size, f"Invalid cp_rank={cp_rank} for cp_size={cp_size}")

    if max_seq_len <= 0 or cp_size <= 1:
        return max(0, max_seq_len)

    full_chunks, tail_tokens = divmod(max_seq_len, chunk_size)
    owned_full_chunks = full_chunks // cp_size
    if cp_rank < full_chunks % cp_size:
        owned_full_chunks += 1

    local_cap = owned_full_chunks * chunk_size
    if full_chunks % cp_size == cp_rank:
        local_cap += tail_tokens
    return max(1, local_cap)


def attncp_cp2_fused_q_fa_supports_shape(
    local_q_heads: int,
    num_kv_heads: int,
    *,
    cp_world_size: int = 2,
) -> bool:
    if cp_world_size != 2 or local_q_heads <= 0 or num_kv_heads <= 0:
        return False
    full_q_heads = local_q_heads * cp_world_size
    if full_q_heads % num_kv_heads != 0:
        return False
    q_heads_per_kv = full_q_heads // num_kv_heads
    return triton.next_power_of_2(q_heads_per_kv) <= _FUSED_Q_FA_MAX_BLOCK_H


def attncp_cp2_fused_q_fa_max_splits(max_seq_len: int) -> int:
    """Return the internal split workspace size for a graph sequence cap."""
    max_seq_len = max(1, int(max_seq_len))
    preferred_splits = triton.cdiv(max_seq_len, _FUSED_Q_FA_TARGET_SPLIT_SIZE)
    return max(1, min(int(preferred_splits), _FUSED_Q_FA_MAX_AUTO_SPLITS))


def _require_cp2_fused_q_fa_kv_stationary_shape(
    local_q_heads: int,
    num_kv_heads: int,
    *,
    cp_world_size: int = 2,
) -> None:
    if attncp_cp2_fused_q_fa_supports_shape(
        local_q_heads,
        num_kv_heads,
        cp_world_size=cp_world_size,
    ):
        return

    full_q_heads = local_q_heads * cp_world_size
    if cp_world_size != 2:
        reason = f"cp_world_size={cp_world_size}"
    elif local_q_heads <= 0 or num_kv_heads <= 0:
        reason = f"local_q_heads={local_q_heads}, num_kv_heads={num_kv_heads}"
    elif full_q_heads % num_kv_heads != 0:
        reason = (
            f"full_q_heads={full_q_heads} is not divisible by "
            f"num_kv_heads={num_kv_heads}"
        )
    else:
        q_heads_per_kv = full_q_heads // num_kv_heads
        reason = (
            f"q_heads_per_kv={q_heads_per_kv} exceeds "
            f"KV-stationary block_h={_FUSED_Q_FA_MAX_BLOCK_H}"
        )
    raise ValueError(
        "Unsupported AttnCP CP2 fused Q+FA shape. "
        "The fused path must keep one program per (batch, kv_head[, split]) "
        "so each resident KV tile is loaded once and reused by all CP Q heads; "
        f"fall back instead of splitting by Q head: {reason}."
    )


@triton.jit
def _attncp_cp2_merge_local_head_slice_kernel(
    gathered_o,
    gathered_lse,
    out_o,
    BATCH_SIZE: tl.constexpr,
    FULL_Q_HEADS: tl.constexpr,
    LOCAL_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_START: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    local_head_idx = tl.program_id(1)
    global_head_idx = HEAD_START + local_head_idx
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM

    head_offset = global_head_idx * HEAD_DIM + dim_offsets
    base_a = token_idx * FULL_Q_HEADS * HEAD_DIM + head_offset
    base_b = BATCH_SIZE * FULL_Q_HEADS * HEAD_DIM + base_a

    lse_a = tl.load(gathered_lse + token_idx * FULL_Q_HEADS + global_head_idx)
    lse_b = tl.load(
        gathered_lse
        + BATCH_SIZE * FULL_Q_HEADS
        + token_idx * FULL_Q_HEADS
        + global_head_idx
    )
    lse_a = tl.where(lse_a == float("inf"), -float("inf"), lse_a)
    lse_b = tl.where(lse_b == float("inf"), -float("inf"), lse_b)
    max_lse = tl.maximum(lse_a, lse_b)
    lse_a = lse_a - max_lse
    lse_b = lse_b - max_lse
    se_a = tl.exp(lse_a)
    se_b = tl.exp(lse_b)
    out_se = se_a + se_b
    scale_a = se_a / out_se
    scale_b = se_b / out_se

    value_a = tl.load(gathered_o + base_a, mask=dim_mask).to(tl.float32)
    value_b = tl.load(gathered_o + base_b, mask=dim_mask).to(tl.float32)
    merged = value_a * scale_a + value_b * scale_b

    out_offsets = (
        token_idx * LOCAL_Q_HEADS * HEAD_DIM
        + local_head_idx * HEAD_DIM
        + dim_offsets
    )
    tl.store(out_o + out_offsets, merged, mask=dim_mask)


def attncp_cp2_merge_local_head_slice(
    gathered_o: torch.Tensor,
    gathered_lse: torch.Tensor,
    out_o: torch.Tensor,
    *,
    batch_size: int,
    full_q_heads: int,
    local_q_heads: int,
    head_dim: int,
    head_start: int,
) -> torch.Tensor:
    assert gathered_o.is_cuda
    assert gathered_lse.is_cuda
    assert out_o.is_cuda
    assert gathered_o.is_contiguous()
    assert gathered_lse.is_contiguous()
    assert out_o.is_contiguous()
    assert gathered_o.dim() == 3
    assert gathered_lse.dim() == 2
    assert tuple(out_o.shape) == (batch_size, local_q_heads, head_dim)

    _attncp_cp2_merge_local_head_slice_kernel[(batch_size, local_q_heads)](
        gathered_o,
        gathered_lse,
        out_o,
        batch_size,
        full_q_heads,
        local_q_heads,
        head_dim,
        head_start,
        triton.next_power_of_2(head_dim),
    )
    return out_o


@triton.jit
def _attncp_cp2_pack_local_head_slice_kernel(
    local_o_full,
    local_lse_full,
    packed_o,
    packed_lse,
    FULL_Q_HEADS: tl.constexpr,
    LOCAL_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_START: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    local_head_idx = tl.program_id(1)
    global_head_idx = HEAD_START + local_head_idx
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM

    src_offsets = (
        token_idx * FULL_Q_HEADS * HEAD_DIM
        + global_head_idx * HEAD_DIM
        + dim_offsets
    )
    dst_offsets = (
        token_idx * LOCAL_Q_HEADS * HEAD_DIM
        + local_head_idx * HEAD_DIM
        + dim_offsets
    )
    values = tl.load(local_o_full + src_offsets, mask=dim_mask)
    tl.store(packed_o + dst_offsets, values, mask=dim_mask)

    lse = tl.load(local_lse_full + token_idx * FULL_Q_HEADS + global_head_idx)
    tl.store(packed_lse + token_idx * LOCAL_Q_HEADS + local_head_idx, lse)


def attncp_cp2_pack_local_head_slice(
    local_o_full: torch.Tensor,
    local_lse_full: torch.Tensor,
    packed_o: torch.Tensor,
    packed_lse: torch.Tensor,
    *,
    full_q_heads: int,
    local_q_heads: int,
    head_dim: int,
    head_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = local_o_full.shape[0]
    assert local_o_full.is_cuda
    assert local_lse_full.is_cuda
    assert packed_o.is_cuda
    assert packed_lse.is_cuda
    assert local_o_full.is_contiguous()
    assert local_lse_full.is_contiguous()
    assert packed_o.is_contiguous()
    assert packed_lse.is_contiguous()
    assert tuple(packed_o.shape) == (batch_size, local_q_heads, head_dim)
    assert tuple(packed_lse.shape) == (batch_size, local_q_heads)

    _attncp_cp2_pack_local_head_slice_kernel[(batch_size, local_q_heads)](
        local_o_full,
        local_lse_full,
        packed_o,
        packed_lse,
        full_q_heads,
        local_q_heads,
        head_dim,
        head_start,
        triton.next_power_of_2(head_dim),
    )
    return packed_o, packed_lse


@triton.jit
def _attncp_cp2_merge_local_remote_head_slice_kernel(
    local_o_full,
    local_lse_full,
    remote_o,
    remote_lse,
    out_o,
    FULL_Q_HEADS: tl.constexpr,
    LOCAL_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_IS_CP0: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    local_head_idx = tl.program_id(1)
    global_head_idx = HEAD_START + local_head_idx
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM

    local_offsets = (
        token_idx * FULL_Q_HEADS * HEAD_DIM
        + global_head_idx * HEAD_DIM
        + dim_offsets
    )
    remote_offsets = (
        token_idx * LOCAL_Q_HEADS * HEAD_DIM
        + local_head_idx * HEAD_DIM
        + dim_offsets
    )
    local_lse = tl.load(local_lse_full + token_idx * FULL_Q_HEADS + global_head_idx)
    peer_lse = tl.load(remote_lse + token_idx * LOCAL_Q_HEADS + local_head_idx)

    if LOCAL_IS_CP0:
        lse_a = local_lse
        lse_b = peer_lse
        value_a = tl.load(local_o_full + local_offsets, mask=dim_mask).to(tl.float32)
        value_b = tl.load(remote_o + remote_offsets, mask=dim_mask).to(tl.float32)
    else:
        lse_a = peer_lse
        lse_b = local_lse
        value_a = tl.load(remote_o + remote_offsets, mask=dim_mask).to(tl.float32)
        value_b = tl.load(local_o_full + local_offsets, mask=dim_mask).to(tl.float32)

    lse_a = tl.where(lse_a == float("inf"), -float("inf"), lse_a)
    lse_b = tl.where(lse_b == float("inf"), -float("inf"), lse_b)
    max_lse = tl.maximum(lse_a, lse_b)
    lse_a = lse_a - max_lse
    lse_b = lse_b - max_lse
    se_a = tl.exp(lse_a)
    se_b = tl.exp(lse_b)
    out_se = se_a + se_b
    scale_a = se_a / out_se
    scale_b = se_b / out_se
    merged = value_a * scale_a + value_b * scale_b

    out_offsets = (
        token_idx * LOCAL_Q_HEADS * HEAD_DIM
        + local_head_idx * HEAD_DIM
        + dim_offsets
    )
    tl.store(out_o + out_offsets, merged, mask=dim_mask)


def attncp_cp2_merge_local_remote_head_slice(
    local_o_full: torch.Tensor,
    local_lse_full: torch.Tensor,
    remote_o: torch.Tensor,
    remote_lse: torch.Tensor,
    out_o: torch.Tensor,
    *,
    full_q_heads: int,
    local_q_heads: int,
    head_dim: int,
    head_start: int,
    local_is_cp0: bool,
) -> torch.Tensor:
    batch_size = local_o_full.shape[0]
    assert local_o_full.is_cuda
    assert local_lse_full.is_cuda
    assert remote_o.is_cuda
    assert remote_lse.is_cuda
    assert out_o.is_cuda
    assert local_o_full.is_contiguous()
    assert local_lse_full.is_contiguous()
    assert remote_o.is_contiguous()
    assert remote_lse.is_contiguous()
    assert out_o.is_contiguous()
    assert tuple(remote_o.shape) == (batch_size, local_q_heads, head_dim)
    assert tuple(remote_lse.shape) == (batch_size, local_q_heads)
    assert tuple(out_o.shape) == (batch_size, local_q_heads, head_dim)

    _attncp_cp2_merge_local_remote_head_slice_kernel[(batch_size, local_q_heads)](
        local_o_full,
        local_lse_full,
        remote_o,
        remote_lse,
        out_o,
        full_q_heads,
        local_q_heads,
        head_dim,
        head_start,
        local_is_cp0,
        triton.next_power_of_2(head_dim),
    )
    return out_o


@triton.jit
def _attncp_cp2_fused_q_fa_decode_kernel(
    q_local,
    q_peer,
    key_cache,
    value_cache,
    page_table,
    cache_seqlens,
    sinks,
    out_o,
    out_lse,
    PAGE_TABLE_STRIDE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    LOCAL_Q_HEADS: tl.constexpr,
    FULL_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CP_RANK: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    qh_per_kv = FULL_Q_HEADS // NUM_KV_HEADS
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    token_offsets = tl.arange(0, BLOCK_N)

    global_q_heads = kv_head_idx * qh_per_kv + head_offsets
    head_mask = head_offsets < qh_per_kv
    dim_mask = dim_offsets < HEAD_DIM

    local_head_start = CP_RANK * LOCAL_Q_HEADS
    peer_head_start = (1 - CP_RANK) * LOCAL_Q_HEADS
    is_local_q = (global_q_heads >= local_head_start) & (
        global_q_heads < local_head_start + LOCAL_Q_HEADS
    )
    is_peer_q = (global_q_heads >= peer_head_start) & (
        global_q_heads < peer_head_start + LOCAL_Q_HEADS
    )

    local_head_offsets = global_q_heads - local_head_start
    peer_head_offsets = global_q_heads - peer_head_start
    q_local_offsets = (
        batch_idx * LOCAL_Q_HEADS * HEAD_DIM
        + local_head_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    q_peer_offsets = (
        batch_idx * LOCAL_Q_HEADS * HEAD_DIM
        + peer_head_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    q_local_vals = tl.load(
        q_local + q_local_offsets,
        mask=(head_mask & is_local_q)[:, None] & dim_mask[None, :],
        other=0.0,
    )
    q_peer_vals = tl.load(
        q_peer + q_peer_offsets,
        mask=(head_mask & is_peer_q)[:, None] & dim_mask[None, :],
        other=0.0,
    )
    q_vals = q_local_vals + q_peer_vals

    if HAS_SINKS:
        sink_vals = tl.load(sinks + global_q_heads, mask=head_mask, other=-float("inf"))
        m_i = sink_vals.to(tl.float32)
        l_i = tl.where(sink_vals == -float("inf"), 0.0, 1.0).to(tl.float32)
    else:
        m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    seq_len = tl.load(cache_seqlens + batch_idx)
    window_start = 0
    if WINDOW_LEFT >= 0:
        window_start = tl.maximum(seq_len - WINDOW_LEFT - 1, 0)
    for start_n in tl.range(0, MAX_SEQ_LEN, BLOCK_N):
        kv_positions = start_n + token_offsets
        kv_mask = kv_positions < seq_len
        if WINDOW_LEFT >= 0:
            kv_mask = kv_mask & (kv_positions >= window_start)
        page_indices = tl.load(
            page_table + batch_idx * PAGE_TABLE_STRIDE + kv_positions // PAGE_SIZE,
            mask=kv_mask,
            other=0,
        )
        offsets_in_page = kv_positions - (kv_positions // PAGE_SIZE) * PAGE_SIZE
        kv_base = (
            ((page_indices * PAGE_SIZE + offsets_in_page) * NUM_KV_HEADS + kv_head_idx)
            * HEAD_DIM
        )
        kv_offsets = kv_base[:, None] + dim_offsets[None, :]
        kv_load_mask = kv_mask[:, None] & dim_mask[None, :]
        # The resident K/V tile is loaded once and reused for every CP Q head
        # mapped to this KV head. Do not add a Q-head program dimension here.
        k_vals = tl.load(key_cache + kv_offsets, mask=kv_load_mask, other=0.0)
        v_vals = tl.load(value_cache + kv_offsets, mask=kv_load_mask, other=0.0)

        scores = tl.dot(q_vals, tl.trans(k_vals)) * SOFTMAX_SCALE
        if SOFTCAP > 0.0:
            scores = (2.0 * tl.sigmoid(2.0 * scores / SOFTCAP) - 1.0) * SOFTCAP
        score_mask = head_mask[:, None] & kv_mask[None, :]
        scores = tl.where(score_mask, scores, -float("inf"))

        tile_m = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, tile_m)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.where(
            score_mask,
            tl.exp2((scores - m_new_safe[:, None]) * 1.4426950408889634),
            0.0,
        )
        alpha = tl.where(
            m_i == -float("inf"),
            0.0,
            tl.exp2((m_i - m_new_safe) * 1.4426950408889634),
        )
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_vals.dtype), v_vals)
        m_i = m_new
        l_i = l_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out_vals = acc / l_safe[:, None]
    out_vals = tl.where(l_i[:, None] > 0.0, out_vals, 0.0)
    lse_vals = tl.where(
        l_i > 0.0, tl.log2(l_i) * 0.6931471805599453 + m_i, -float("inf")
    )

    out_offsets = (
        batch_idx * FULL_Q_HEADS * HEAD_DIM
        + global_q_heads[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    tl.store(out_o + out_offsets, out_vals, mask=head_mask[:, None] & dim_mask[None, :])
    lse_offsets = batch_idx * FULL_Q_HEADS + global_q_heads
    tl.store(out_lse + lse_offsets, lse_vals, mask=head_mask)


@triton.jit
def _attncp_cp2_fused_q_fa_decode_split_kernel(
    q_local,
    q_peer,
    key_cache,
    value_cache,
    page_table,
    cache_seqlens,
    sinks,
    split_o,
    split_lse,
    PAGE_TABLE_STRIDE: tl.constexpr,
    SPLIT_O_STRIDE_SPLIT: tl.constexpr,
    SPLIT_O_STRIDE_BATCH: tl.constexpr,
    SPLIT_O_STRIDE_HEAD: tl.constexpr,
    SPLIT_LSE_STRIDE_SPLIT: tl.constexpr,
    SPLIT_LSE_STRIDE_BATCH: tl.constexpr,
    SPLIT_LSE_STRIDE_HEAD: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    LOCAL_Q_HEADS: tl.constexpr,
    FULL_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CP_RANK: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    qh_per_kv = FULL_Q_HEADS // NUM_KV_HEADS
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    token_offsets = tl.arange(0, BLOCK_N)

    global_q_heads = kv_head_idx * qh_per_kv + head_offsets
    head_mask = head_offsets < qh_per_kv
    dim_mask = dim_offsets < HEAD_DIM

    local_head_start = CP_RANK * LOCAL_Q_HEADS
    peer_head_start = (1 - CP_RANK) * LOCAL_Q_HEADS
    is_local_q = (global_q_heads >= local_head_start) & (
        global_q_heads < local_head_start + LOCAL_Q_HEADS
    )
    is_peer_q = (global_q_heads >= peer_head_start) & (
        global_q_heads < peer_head_start + LOCAL_Q_HEADS
    )

    local_head_offsets = global_q_heads - local_head_start
    peer_head_offsets = global_q_heads - peer_head_start
    q_local_offsets = (
        batch_idx * LOCAL_Q_HEADS * HEAD_DIM
        + local_head_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    q_peer_offsets = (
        batch_idx * LOCAL_Q_HEADS * HEAD_DIM
        + peer_head_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    q_local_vals = tl.load(
        q_local + q_local_offsets,
        mask=(head_mask & is_local_q)[:, None] & dim_mask[None, :],
        other=0.0,
    )
    q_peer_vals = tl.load(
        q_peer + q_peer_offsets,
        mask=(head_mask & is_peer_q)[:, None] & dim_mask[None, :],
        other=0.0,
    )
    q_vals = q_local_vals + q_peer_vals

    if HAS_SINKS:
        sink_vals = tl.load(sinks + global_q_heads, mask=head_mask, other=-float("inf"))
        use_sink = split_idx == 0
        m_i = tl.where(use_sink, sink_vals.to(tl.float32), -float("inf"))
        l_i = tl.where(use_sink & (sink_vals != -float("inf")), 1.0, 0.0).to(
            tl.float32
        )
    else:
        m_i = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    seq_len = tl.load(cache_seqlens + batch_idx)
    window_start = 0
    if WINDOW_LEFT >= 0:
        window_start = tl.maximum(seq_len - WINDOW_LEFT - 1, 0)
    split_start = split_idx * SPLIT_SIZE
    split_end = split_start + SPLIT_SIZE
    for rel_start in tl.range(0, SPLIT_SIZE, BLOCK_N):
        kv_positions = split_start + rel_start + token_offsets
        kv_mask = (kv_positions < seq_len) & (kv_positions < split_end)
        if WINDOW_LEFT >= 0:
            kv_mask = kv_mask & (kv_positions >= window_start)
        page_indices = tl.load(
            page_table + batch_idx * PAGE_TABLE_STRIDE + kv_positions // PAGE_SIZE,
            mask=kv_mask,
            other=0,
        )
        offsets_in_page = kv_positions - (kv_positions // PAGE_SIZE) * PAGE_SIZE
        kv_base = (
            ((page_indices * PAGE_SIZE + offsets_in_page) * NUM_KV_HEADS + kv_head_idx)
            * HEAD_DIM
        )
        kv_offsets = kv_base[:, None] + dim_offsets[None, :]
        kv_load_mask = kv_mask[:, None] & dim_mask[None, :]
        # The split owns a disjoint KV range; each loaded K/V tile is shared by
        # all CP Q heads for this KV head before advancing to the next tile.
        k_vals = tl.load(key_cache + kv_offsets, mask=kv_load_mask, other=0.0)
        v_vals = tl.load(value_cache + kv_offsets, mask=kv_load_mask, other=0.0)

        scores = tl.dot(q_vals, tl.trans(k_vals)) * SOFTMAX_SCALE
        if SOFTCAP > 0.0:
            scores = (2.0 * tl.sigmoid(2.0 * scores / SOFTCAP) - 1.0) * SOFTCAP
        score_mask = head_mask[:, None] & kv_mask[None, :]
        scores = tl.where(score_mask, scores, -float("inf"))

        tile_m = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, tile_m)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.where(
            score_mask,
            tl.exp2((scores - m_new_safe[:, None]) * 1.4426950408889634),
            0.0,
        )
        alpha = tl.where(
            m_i == -float("inf"),
            0.0,
            tl.exp2((m_i - m_new_safe) * 1.4426950408889634),
        )
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_vals.dtype), v_vals)
        m_i = m_new
        l_i = l_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out_vals = acc / l_safe[:, None]
    out_vals = tl.where(l_i[:, None] > 0.0, out_vals, 0.0)
    lse_vals = tl.where(
        l_i > 0.0, tl.log2(l_i) * 0.6931471805599453 + m_i, -float("inf")
    )

    split_o_offsets = (
        split_idx * SPLIT_O_STRIDE_SPLIT
        + batch_idx * SPLIT_O_STRIDE_BATCH
        + global_q_heads[:, None] * SPLIT_O_STRIDE_HEAD
        + dim_offsets[None, :]
    )
    split_lse_offsets = (
        split_idx * SPLIT_LSE_STRIDE_SPLIT
        + batch_idx * SPLIT_LSE_STRIDE_BATCH
        + global_q_heads * SPLIT_LSE_STRIDE_HEAD
    )
    tl.store(
        split_o + split_o_offsets,
        out_vals,
        mask=head_mask[:, None] & dim_mask[None, :],
    )
    tl.store(split_lse + split_lse_offsets, lse_vals, mask=head_mask)


@triton.jit
def _attncp_cp2_merge_fa_splits_kernel(
    split_o,
    split_lse,
    out_o,
    out_lse,
    SPLIT_O_STRIDE_SPLIT: tl.constexpr,
    SPLIT_O_STRIDE_BATCH: tl.constexpr,
    SPLIT_O_STRIDE_HEAD: tl.constexpr,
    SPLIT_LSE_STRIDE_SPLIT: tl.constexpr,
    SPLIT_LSE_STRIDE_BATCH: tl.constexpr,
    SPLIT_LSE_STRIDE_HEAD: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    FULL_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM

    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    base = batch_idx * FULL_Q_HEADS + q_head_idx
    for split_idx in tl.range(0, NUM_SPLITS):
        lse = tl.load(
            split_lse
            + split_idx * SPLIT_LSE_STRIDE_SPLIT
            + batch_idx * SPLIT_LSE_STRIDE_BATCH
            + q_head_idx * SPLIT_LSE_STRIDE_HEAD
        )
        m_new = tl.maximum(m_i, lse)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.where(
            m_i == -float("inf"),
            0.0,
            tl.exp(m_i - m_new_safe),
        )
        beta = tl.where(
            lse == -float("inf"),
            0.0,
            tl.exp(lse - m_new_safe),
        )
        o_offsets = (
            split_idx * SPLIT_O_STRIDE_SPLIT
            + batch_idx * SPLIT_O_STRIDE_BATCH
            + q_head_idx * SPLIT_O_STRIDE_HEAD
            + dim_offsets
        )
        o = tl.load(split_o + o_offsets, mask=dim_mask, other=0.0).to(tl.float32)
        acc = acc * alpha + o * beta
        l_i = l_i * alpha + beta
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out_vals = acc / l_safe
    out_vals = tl.where(l_i > 0.0, out_vals, 0.0)
    final_lse = tl.where(l_i > 0.0, tl.log(l_i) + m_i, -float("inf"))
    out_offsets = base * HEAD_DIM + dim_offsets
    tl.store(out_o + out_offsets, out_vals, mask=dim_mask)
    tl.store(out_lse + base, final_lse)


def attncp_cp2_fused_q_fa_decode(
    q_local: torch.Tensor,
    q_peer: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    out_o: torch.Tensor,
    out_lse: torch.Tensor,
    *,
    cp_rank: int,
    softmax_scale: float,
    softcap: float = 0.0,
    window_left: int = -1,
    sinks: torch.Tensor | None = None,
    page_size: int = 1,
    block_n: int = 256,
    split_o: torch.Tensor | None = None,
    split_lse: torch.Tensor | None = None,
    max_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute CP2 sharded-KV decode attention for all CP Q heads.

    The Triton kernels are intentionally KV-stationary: each program owns one
    ``(batch, kv_head[, split])`` and evaluates every logical CP Q head mapped
    to that KV head while the K/V tile is resident. Do not split this path by Q
    head, or the same resident KV shard will be read multiple times.
    """
    _check(cp_rank in (0, 1), f"cp_rank must be 0 or 1, got {cp_rank}")
    _check(q_local.is_cuda and q_peer.is_cuda, "Q tensors must be CUDA tensors")
    _check(
        key_cache.is_cuda and value_cache.is_cuda,
        "K/V cache tensors must be CUDA tensors",
    )
    _check(
        page_table.is_cuda and cache_seqlens.is_cuda,
        "page_table and cache_seqlens must be CUDA tensors",
    )
    _check(out_o.is_cuda and out_lse.is_cuda, "output tensors must be CUDA tensors")
    _check(
        q_local.is_contiguous() and q_peer.is_contiguous(),
        "Q tensors must be contiguous",
    )
    _check(
        key_cache.is_contiguous() and value_cache.is_contiguous(),
        "K/V cache tensors must be contiguous",
    )
    _check(cache_seqlens.is_contiguous(), "cache_seqlens must be contiguous")
    _check(
        out_o.is_contiguous() and out_lse.is_contiguous(),
        "output tensors must be contiguous",
    )
    _check(
        q_local.dim() == 3 and q_peer.shape == q_local.shape,
        "q_local and q_peer must have the same [batch, local_q_heads, head_dim] shape",
    )
    _check(
        key_cache.dim() == 4 and value_cache.shape == key_cache.shape,
        "key_cache and value_cache must have the same 4D paged-cache shape",
    )
    _check(page_size >= 1, f"page_size must be >= 1, got {page_size}")
    _check(
        page_size == key_cache.shape[1],
        f"page_size={page_size} does not match key_cache.shape[1]={key_cache.shape[1]}",
    )
    batch_size, local_q_heads, head_dim = q_local.shape
    full_q_heads = local_q_heads * 2
    num_kv_heads = key_cache.shape[2]
    _require_cp2_fused_q_fa_kv_stationary_shape(local_q_heads, num_kv_heads)
    _check(
        tuple(out_o.shape) == (batch_size, full_q_heads, head_dim),
        "out_o must have shape [batch, full_q_heads, head_dim]",
    )
    _check(
        tuple(out_lse.shape) == (batch_size, full_q_heads),
        "out_lse must have shape [batch, full_q_heads]",
    )
    _check(page_table.shape[0] == batch_size, "page_table batch size mismatch")
    _check(cache_seqlens.shape[0] == batch_size, "cache_seqlens batch size mismatch")
    _check(
        head_dim == key_cache.shape[3],
        f"Q head_dim={head_dim} does not match key_cache head_dim={key_cache.shape[3]}",
    )
    if sinks is not None:
        _check(
            sinks.is_cuda and sinks.is_contiguous(),
            "sinks must be a contiguous CUDA tensor",
        )
        _check(
            sinks.numel() == full_q_heads,
            f"sinks must have {full_q_heads} elements, got {sinks.numel()}",
        )

    if head_dim >= 256:
        block_n = min(int(block_n), 128)

    block_h = triton.next_power_of_2(full_q_heads // num_kv_heads)
    block_d = triton.next_power_of_2(head_dim)
    max_seq_len = page_table.shape[1] * page_size
    if split_o is not None or split_lse is not None:
        _check(
            split_o is not None and split_lse is not None,
            "split_o and split_lse must be provided together",
        )
        _check(
            split_o.is_cuda and split_lse.is_cuda,
            "split workspaces must be CUDA tensors",
        )
        _check(
            split_o.is_contiguous() and split_lse.is_contiguous(),
            "split workspaces must be contiguous",
        )
        _check(split_o.dim() == 4, "split_o must be 4D")
        _check(split_lse.dim() == 3, "split_lse must be 3D")
        _check(split_o.shape[1] >= batch_size, "split_o batch workspace too small")
        _check(
            split_o.shape[2:] == (full_q_heads, head_dim),
            "split_o must have trailing shape [full_q_heads, head_dim]",
        )
        _check(split_lse.shape[1] >= batch_size, "split_lse batch workspace too small")
        _check(
            split_lse.shape[2:] == (full_q_heads,),
            "split_lse must have trailing shape [full_q_heads]",
        )
        max_splits = min(int(max_splits), int(split_o.shape[0]))
    else:
        max_splits = 1

    if max_seq_len <= _FUSED_Q_FA_TARGET_SPLIT_SIZE or max_splits <= 1:
        num_splits = 1
    else:
        preferred_splits = triton.cdiv(max_seq_len, _FUSED_Q_FA_TARGET_SPLIT_SIZE)
        num_splits = max(1, min(int(max_splits), preferred_splits))
    if num_splits > 1:
        if num_splits * _FUSED_Q_FA_TARGET_SPLIT_SIZE >= max_seq_len:
            split_size = _FUSED_Q_FA_TARGET_SPLIT_SIZE
        else:
            split_size = triton.cdiv(max_seq_len, num_splits)
        # Keep the launch KV-stationary. Splitting by Q head would reload the
        # same local KV shard and break the AttnCP decode bandwidth invariant.
        _attncp_cp2_fused_q_fa_decode_split_kernel[
            (batch_size, num_kv_heads, num_splits)
        ](
            q_local,
            q_peer,
            key_cache,
            value_cache,
            page_table,
            cache_seqlens,
            sinks if sinks is not None else out_lse,
            split_o[:num_splits],
            split_lse[:num_splits],
            page_table.stride(0),
            split_o.stride(0),
            split_o.stride(1),
            split_o.stride(2),
            split_lse.stride(0),
            split_lse.stride(1),
            split_lse.stride(2),
            split_size,
            num_kv_heads,
            local_q_heads,
            full_q_heads,
            head_dim,
            page_size,
            cp_rank,
            float(softmax_scale),
            float(softcap or 0.0),
            int(window_left),
            sinks is not None,
            block_h,
            block_n,
            block_d,
            num_warps=4,
            num_stages=3,
        )
        _attncp_cp2_merge_fa_splits_kernel[(batch_size, full_q_heads)](
            split_o[:num_splits],
            split_lse[:num_splits],
            out_o,
            out_lse,
            split_o.stride(0),
            split_o.stride(1),
            split_o.stride(2),
            split_lse.stride(0),
            split_lse.stride(1),
            split_lse.stride(2),
            num_splits,
            full_q_heads,
            head_dim,
            block_d,
            num_warps=4,
            num_stages=3,
        )
        return out_o, out_lse

    # Keep the launch KV-stationary. A Q-head grid dimension is not allowed for
    # this fused path because it would read the same K/V tile multiple times.
    _attncp_cp2_fused_q_fa_decode_kernel[(batch_size, num_kv_heads)](
        q_local,
        q_peer,
        key_cache,
        value_cache,
        page_table,
        cache_seqlens,
        sinks if sinks is not None else out_lse,
        out_o,
        out_lse,
        page_table.stride(0),
        max_seq_len,
        num_kv_heads,
        local_q_heads,
        full_q_heads,
        head_dim,
        page_size,
        cp_rank,
        float(softmax_scale),
        float(softcap or 0.0),
        int(window_left),
        sinks is not None,
        block_h,
        block_n,
        block_d,
        num_warps=4,
        num_stages=3,
    )
    return out_o, out_lse
