# Copyright (c) 2026, SGLang Team.
"""SM120 FA4 path for BF16 Q with per-tensor-scaled FP8 paged K/V."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.ops.attention.fa4_sm120.runtime import sm120_forward_host
from sglang.kernels.ops.attention.flash_attn.cute.cute_dsl_utils import (
    to_cute_tensor,
)
from sglang.kernels.ops.attention.flash_attn.cute.interface import (
    _fwd_combine_compile_key,
    _flash_attn_fwd_combine,
)
from sglang.kernels.ops.attention.flash_attn.cute.utils import AuxData


_COMPILE_CACHE_CAPACITY = 128
_WORKSPACE_CACHE_CAPACITY = 64
_COMBINE_CACHE_CAPACITY = 128
_FP8_DMA_THREADS = 64

_compile_cache: OrderedDict[tuple, object] = OrderedDict()
_workspace_cache: OrderedDict[
    tuple, tuple[torch.Tensor, torch.Tensor]
] = OrderedDict()
_combine_cache: OrderedDict[tuple, object] = OrderedDict()


def _cache_get(cache: OrderedDict, key: tuple):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(
    cache: OrderedDict,
    key: tuple,
    value: object,
    capacity: int,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > capacity:
        cache.popitem(last=False)


def _to_cute_descale(tensor: torch.Tensor):
    """Preserve scalar-expand stride-0 scale layouts in the kernel ABI."""
    return from_dlpack(tensor.detach(), assumed_align=4, enable_tvm_ffi=True)


def _num_splits_heuristic(
    total_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    if num_n_blocks <= 4 or total_mblocks == 0:
        return 1
    return min(num_sms // total_mblocks, max_splits, num_n_blocks)


def flash_attn_fp8_kv_sm120(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    num_splits: int = 1,
    pack_gqa: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run fused FP8-KV attention for the supported SM120 contract.

    The SGLang-facing wrapper owns dispatch to this internal entry point.  The
    generic FA4 interface remains unchanged.
    """
    if q.device.type != "cuda":
        raise ValueError("SM120 FP8 KV attention requires CUDA tensors")
    if torch.cuda.get_device_capability(q.device)[0] != 12:
        raise ValueError("SM120 FP8 KV attention requires compute capability 12.x")
    if q.dtype != torch.bfloat16:
        raise TypeError("q must be bfloat16")
    if k_cache.dtype != torch.float8_e4m3fn or v_cache.dtype != k_cache.dtype:
        raise TypeError("k_cache and v_cache must be float8_e4m3fn")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("expected q rank 3 and paged K/V rank 4")
    if k_cache.shape != v_cache.shape:
        raise ValueError("SM120 FP8 KV requires identical K/V shapes")
    tensors = (k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError(
            "Q, K/V, page table, and sequence tensors must share a device"
        )
    if any(tensor.stride(-1) != 1 for tensor in (q, k_cache, v_cache)):
        raise ValueError("Q and K/V must be contiguous in their last dimension")
    if page_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise TypeError("page_table and cache_seqlens must be int32")
    if cu_seqlens_q.dtype != torch.int32:
        raise TypeError("cu_seqlens_q must be int32")
    if page_table.ndim != 2 or cache_seqlens.ndim != 1 or cu_seqlens_q.ndim != 1:
        raise ValueError("page table and sequence tensors have invalid ranks")
    if max_seqlen_q <= 0 or max_seqlen_q > 8 or max_seqlen_k <= 0:
        raise ValueError("SM120 FP8 KV requires 1 <= max_seqlen_q <= 8 and K > 0")
    if not causal:
        raise NotImplementedError(
            "SM120 FP8 KV currently supports causal attention only"
        )
    if num_splits < 0:
        raise ValueError("num_splits must be non-negative")
    if not pack_gqa:
        raise NotImplementedError("SM120 FP8 KV currently requires packed GQA")

    batch_size = cu_seqlens_q.numel() - 1
    num_head = q.shape[1]
    num_head_kv = k_cache.shape[2]
    head_dim = q.shape[2]
    head_dim_v = v_cache.shape[3]
    if k_cache.shape[3] != head_dim:
        raise ValueError("K head dimension must match Q")
    if num_head % num_head_kv:
        raise ValueError("Q head count must be divisible by KV head count")
    qhead_per_kvhead = num_head // num_head_kv
    expected_descale_shape = (batch_size, num_head_kv)
    for name, descale in (("k_descale", k_descale), ("v_descale", v_descale)):
        if descale.shape != expected_descale_shape:
            raise ValueError(
                f"{name} shape {descale.shape} != {expected_descale_shape}"
            )
        if descale.dtype != torch.float32 or descale.device != q.device:
            raise TypeError(f"{name} must be float32 on {q.device}")
    if page_table.shape[0] != batch_size:
        raise ValueError("page_table batch dimension does not match cu_seqlens_q")
    if max_seqlen_k > page_table.shape[1] * k_cache.shape[1]:
        raise ValueError("max_seqlen_k exceeds the page-table capacity")
    if cache_seqlens.shape != (batch_size,):
        raise ValueError("cache_seqlens must have shape (batch_size,)")
    if out is None:
        out = torch.empty(
            (*q.shape[:-1], head_dim_v),
            dtype=torch.bfloat16,
            device=q.device,
        )
    elif out.shape != (*q.shape[:-1], head_dim_v) or out.dtype != torch.bfloat16:
        raise ValueError("out must be contiguous BF16 with the expected shape")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")

    device = q.device
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    config = sm120_forward_host.select_config(
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        tile_mn=None,
        has_bias=False,
        total_q_rows=q.shape[0] * num_head,
        num_sms=num_sms,
        num_batch=batch_size,
        seqlen_q=max_seqlen_q,
        seqlen_k=max_seqlen_k,
        num_head_kv=num_head_kv,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=causal,
        is_local=False,
        window_size_left=None,
        window_size_right=None,
        pack_gqa=pack_gqa,
        paged_kv=True,
    )
    packed_q_rows = max_seqlen_q * qhead_per_kvhead
    num_m_blocks = math.ceil(packed_q_rows / config.tile_m)
    total_mblocks = batch_size * num_head_kv * num_m_blocks
    num_n_blocks = math.ceil(max_seqlen_k / config.tile_n)
    plan = sm120_forward_host.resolve_plan(
        requested_num_splits=num_splits,
        generic_num_n_blocks=num_n_blocks,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        batch_size=batch_size,
        num_head_kv=num_head_kv,
        paged_kv=True,
        page_size=k_cache.shape[1],
        k=k_cache,
        v=v_cache,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        pack_gqa=pack_gqa,
        compute_dtype=q.dtype,
        element_size=k_cache.element_size(),
        packed_q_rows=packed_q_rows,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        num_m_blocks=num_m_blocks,
        total_mblocks=total_mblocks,
        num_sms=num_sms,
        total_q=q.shape[0],
        has_cu_seqlens_q=True,
        has_seqused_q=False,
        has_seqused_k=True,
        is_causal=causal,
        is_local=False,
        window_size_left=None,
        window_size_right=None,
        has_score_or_mask_mod=False,
        is_stream_capturing=torch.cuda.is_current_stream_capturing(),
        device=device,
        fake_mode=False,
        generic_heuristic=_num_splits_heuristic,
    )
    if plan.transpose_qk_pv:
        raise NotImplementedError("SM120 FP8 KV does not support transpose decode")
    actual_num_splits = plan.num_splits
    is_split_kv = actual_num_splits > 1
    if is_split_kv:
        workspace_key = (
            device,
            torch.cuda.current_stream(device).cuda_stream,
            actual_num_splits,
            tuple(q.shape),
            head_dim_v,
        )
        workspace = _cache_get(_workspace_cache, workspace_key)
        out_partial_shape = (actual_num_splits, *q.shape[:-1], head_dim_v)
        lse_partial_shape = (actual_num_splits, num_head, q.shape[0])
        if workspace is None:
            workspace = (
                torch.empty(out_partial_shape, dtype=torch.float32, device=device),
                torch.empty(lse_partial_shape, dtype=torch.float32, device=device),
            )
            _cache_put(
                _workspace_cache,
                workspace_key,
                workspace,
                _WORKSPACE_CACHE_CAPACITY,
            )
        out_partial, lse_partial = workspace
        kernel_out = out_partial
        kernel_lse = lse_partial
    else:
        out_partial = None
        lse_partial = None
        kernel_out = out
        kernel_lse = None
    kernel = sm120_forward_host.make_kernel(
        dtype=cutlass.BFloat16,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=causal,
        is_local=False,
        pack_gqa=pack_gqa,
        config=config,
        paged_kv=True,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors=True,
        is_split_kv=is_split_kv,
        has_bias=False,
        bias_block_size=64,
        rel_extent_padded=128,
        plan=plan,
        fp8_kv=True,
        fp8_dma_threads=_FP8_DMA_THREADS,
    )

    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    key = (
        q.device,
        tuple(q.shape),
        tuple(q.stride()),
        tuple(k_cache.shape),
        tuple(k_cache.stride()),
        tuple(v_cache.stride()),
        tuple(page_table.shape),
        tuple(page_table.stride()),
        tuple(cache_seqlens.shape),
        tuple(cache_seqlens.stride()),
        tuple(cu_seqlens_q.shape),
        tuple(cu_seqlens_q.stride()),
        tuple(kernel_out.shape),
        tuple(kernel_out.stride()),
        tuple(k_descale.shape),
        tuple(k_descale.stride()),
        tuple(v_descale.shape),
        tuple(v_descale.stride()),
        config.compile_key,
        plan.compile_key,
        actual_num_splits,
    )
    compiled = _cache_get(_compile_cache, key)
    if compiled is None:
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_args = [
            kernel,
            to_cute_tensor(q),
            to_cute_tensor(k_cache),
            to_cute_tensor(v_cache),
            to_cute_tensor(kernel_out),
            to_cute_tensor(kernel_lse, assumed_align=4),
            scale,
            to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0),
            None,
            None,
            to_cute_tensor(cache_seqlens, assumed_align=4, leading_dim=0),
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1),
            None,
            None,
            None,
            None,
            AuxData(
                [
                    _to_cute_descale(k_descale),
                    _to_cute_descale(v_descale),
                ],
                None,
            ),
            None,
            Int32(0),
            stream,
        ]
        compiled = cute.compile(*compile_args, options="--enable-tvm-ffi")
        _cache_put(_compile_cache, key, compiled, _COMPILE_CACHE_CAPACITY)

    compiled(
        q.detach(),
        k_cache.detach().view(torch.uint8),
        v_cache.detach().view(torch.uint8),
        kernel_out.detach(),
        kernel_lse,
        scale,
        cu_seqlens_q,
        None,
        None,
        cache_seqlens,
        page_table,
        None,
        None,
        None,
        None,
        AuxData([k_descale, v_descale], None),
        None,
        *sm120_forward_host.runtime_arguments(plan),
    )
    if is_split_kv:
        combine_cache_key = (
            device,
            actual_num_splits,
            tuple(out_partial.shape),
            tuple(out.shape),
            tuple(cu_seqlens_q.shape),
        )
        compiled_combine = _cache_get(_combine_cache, combine_cache_key)
        lse_partial_transposed = lse_partial.transpose(-1, -2)
        if compiled_combine is None:
            _flash_attn_fwd_combine(
                out_partial,
                lse_partial_transposed,
                out,
                None,
                cu_seqlens_q,
                None,
            )
            combine_key = _fwd_combine_compile_key(
                out_partial,
                out,
                None,
                cu_seqlens_q,
                None,
                None,
            )
            _cache_put(
                _combine_cache,
                combine_cache_key,
                _flash_attn_fwd_combine.compile_cache[combine_key],
                _COMBINE_CACHE_CAPACITY,
            )
            return out
        compiled_combine(
            out_partial,
            lse_partial_transposed,
            out,
            None,
            cu_seqlens_q,
            None,
            None,
            None,
            None,
        )
    return out
