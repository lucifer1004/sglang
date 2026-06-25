from __future__ import annotations

import torch
import triton
import triton.language as tl


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
