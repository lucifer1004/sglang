# Copyright (c) 2026, SGLang Team.
"""SGLang-facing FlashAttention-4 APIs specialized for SM12x."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.attention.flash_attention_v4 import (
    _flash_attn_import_error,
    _flash_attn_varlen_func,
    _maybe_contiguous,
    _pad_mla_q_heads,
    _unpad_mla_result,
)

if os.environ.get("SGLANG_INKLING_FA4_USE_PIP") == "1":
    # The pip escape hatch deliberately bypasses SGLang-owned SM12x kernels.
    _flash_attn_fp8_kv_sm120 = None
    get_forward_arch = None
    resolve_runtime_policy = None
    try_cached_paged_decode = None
else:
    from sglang.kernels.ops.attention.fa4_sm120.dispatch import (
        get_forward_arch,
        resolve_runtime_policy,
        try_cached_paged_decode,
    )
    from sglang.kernels.ops.attention.fa4_sm120.fp8_kv import (
        flash_attn_fp8_kv_sm120 as _flash_attn_fp8_kv_sm120,
    )


@dataclass(frozen=True)
class FlashAttentionV4SM120RuntimePolicy:
    num_splits: int
    decode_num_splits: int
    decode_uses_static_max_seqlen_k: bool


def get_flash_attention_v4_sm120_runtime_policy(
    *,
    device_capability: tuple[int, int],
    deterministic: bool,
) -> FlashAttentionV4SM120RuntimePolicy:
    """Resolve the SM12x FA4 launch policy exposed to SGLang."""
    if resolve_runtime_policy is None:
        num_splits = 1 if deterministic or device_capability < (9, 0) else 0
        return FlashAttentionV4SM120RuntimePolicy(
            num_splits=num_splits,
            decode_num_splits=num_splits,
            decode_uses_static_max_seqlen_k=False,
        )
    num_splits, decode_num_splits, decode_uses_static_max_seqlen_k = (
        resolve_runtime_policy(
            device_capability=device_capability,
            deterministic=deterministic,
        )
    )
    return FlashAttentionV4SM120RuntimePolicy(
        num_splits=num_splits,
        decode_num_splits=decode_num_splits,
        decode_uses_static_max_seqlen_k=decode_uses_static_max_seqlen_k,
    )


def _validate_out_contract(out: Optional[torch.Tensor]) -> None:
    if out is None:
        return
    if out.requires_grad:
        raise ValueError("out must not require gradients")
    if out.stride(-1) != 1:
        raise ValueError("out must have stride 1 in the last dimension")


def _can_use_fused_fp8_kv(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    causal: bool,
    window_size: Tuple[int, int],
    attention_chunk: Optional[int],
    softcap: float,
    num_splits: int,
    pack_gqa: Optional[bool],
    sinks: Optional[torch.Tensor],
    score_mod: Optional[Callable],
    aux_tensors: Optional[list],
    sfq: Optional[torch.Tensor],
    sfk: Optional[torch.Tensor],
    sfv: Optional[torch.Tensor],
    rel_bias: Optional[torch.Tensor],
    rel_bias_prep_cache: Optional[dict],
    return_softmax_lse: bool,
    out: Optional[torch.Tensor],
) -> bool:
    """Recognize the deliberately narrow SM120 BF16-compute FP8-KV path."""
    return (
        _flash_attn_fp8_kv_sm120 is not None
        and q.is_cuda
        and torch.cuda.get_device_capability(q.device)[0] == 12
        and q.dtype == torch.bfloat16
        and k_cache.dtype == torch.float8_e4m3fn
        and v_cache.dtype == torch.float8_e4m3fn
        and q.device == k_cache.device == v_cache.device
        and q.ndim == 3
        and k_cache.ndim == 4
        and k_cache.shape == v_cache.shape
        and q.shape[-1] == k_cache.shape[-1] == v_cache.shape[-1] == 128
        and page_table is not None
        and cache_seqlens is not None
        and cu_seqlens_q is not None
        and max_seqlen_q is not None
        and 0 < max_seqlen_q <= 8
        and max_seqlen_k is not None
        and q_descale is None
        and k_descale is not None
        and v_descale is not None
        and causal
        and window_size == (-1, -1)
        and attention_chunk is None
        and softcap in (None, 0.0)
        and num_splits >= 0
        and (
            pack_gqa is True
            or (pack_gqa is None and q.shape[1] > k_cache.shape[2])
        )
        and sinks is None
        and score_mod is None
        and aux_tensors is None
        and sfq is None
        and sfk is None
        and sfv is None
        and rel_bias is None
        and rel_bias_prep_cache is None
        and not return_softmax_lse
        and (out is None or out.is_contiguous())
    )


@debug_kernel_api
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (-1, -1),
    learnable_sink: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    rel_bias: Optional[torch.Tensor] = None,
    rel_bias_prep_cache: Optional[dict] = None,
    return_softmax_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    **_: object,
):
    if _flash_attn_varlen_func is None:  # pragma: no cover
        raise ImportError(
            "FlashAttention-4 CUTE is not available. Install flash-attn-4 with "
            "its CUDA/CUTE dependencies, or run from a source tree where the "
            "vendored FA4 package is importable."
        ) from _flash_attn_import_error

    _validate_out_contract(out)
    q, k, v, qv = [_maybe_contiguous(t) for t in (q, k, v, qv)]
    if qv is None and q.shape[-1] == 256 and k.shape[-1] == 256 and v.shape[-1] == 256:
        # The vendored hd256 kernel assumes dense Q/K/V strides.
        q, k, v = [t.contiguous() for t in (q, k, v)]
    q, qv, mla_head_padding = _pad_mla_q_heads(q, qv, v, pack_gqa)
    if qv is not None and num_splits < 1:
        # FA4 MLA does not implement split-KV; auto mode must use one split.
        num_splits = 1
    cu_seqlens_q, cu_seqlens_k = [
        _maybe_contiguous(t) for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    seqused_q, seqused_k = [_maybe_contiguous(t) for t in (seqused_q, seqused_k)]
    page_table = _maybe_contiguous(page_table)

    if learnable_sink is None and sinks is not None:
        learnable_sink = sinks
    if window_size == (-1, -1):
        window_size = (None, None)

    sf_kwargs = {}
    if sfq is not None:
        sf_kwargs["sfq"] = sfq
    if sfk is not None:
        sf_kwargs["sfk"] = sfk
    if sfv is not None:
        sf_kwargs["sfv"] = sfv

    descale_kwargs = {}
    if q_descale is not None:
        descale_kwargs["q_descale"] = q_descale
    if k_descale is not None:
        descale_kwargs["k_descale"] = k_descale
    if v_descale is not None:
        descale_kwargs["v_descale"] = v_descale

    rel_bias_kwargs = {}
    if rel_bias is not None:
        rel_bias_kwargs["rel_bias"] = rel_bias
    if rel_bias_prep_cache is not None:
        rel_bias_kwargs["rel_bias_prep_cache"] = rel_bias_prep_cache

    result = _flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        qv=qv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size=window_size,
        learnable_sink=learnable_sink,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_lse=return_softmax_lse,
        out=out,
        **sf_kwargs,
        **descale_kwargs,
        **rel_bias_kwargs,
    )
    result = _unpad_mla_result(result, mla_head_padding)

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result


@debug_kernel_api
def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: Optional[int] = None,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata=None,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    sinks: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    rel_bias: Optional[torch.Tensor] = None,
    rel_bias_prep_cache: Optional[dict] = None,
    return_softmax_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
    **_: object,
):
    _validate_out_contract(out)
    if k is not None or v is not None:
        raise NotImplementedError("FA4 does not support updating KV cache in-place.")
    if rotary_cos is not None or rotary_sin is not None or rotary_seqlens is not None:
        raise NotImplementedError("FA4 path does not support rotary embedding.")
    if cache_batch_idx is not None or cache_leftpad is not None:
        raise NotImplementedError(
            "FA4 path does not support non-consecutive batch indices or left padding."
        )
    if isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )

    if _can_use_fused_fp8_kv(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sinks=sinks,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        rel_bias=rel_bias,
        rel_bias_prep_cache=rel_bias_prep_cache,
        return_softmax_lse=return_softmax_lse,
        out=out,
    ):
        q, k_cache, v_cache = [_maybe_contiguous(t) for t in (q, k_cache, v_cache)]
        cu_seqlens_q, cache_seqlens, page_table = [
            _maybe_contiguous(t) for t in (cu_seqlens_q, cache_seqlens, page_table)
        ]
        return _flash_attn_fp8_kv_sm120(
            q,
            k_cache,
            v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=softmax_scale,
            causal=causal,
            num_splits=num_splits,
            pack_gqa=(q.shape[1] > k_cache.shape[2] if pack_gqa is None else pack_gqa),
            out=out,
        )

    forward_arch = get_forward_arch(q.device) if get_forward_arch is not None else None
    if (
        forward_arch is not None
        and not return_softmax_lse
        and softcap in (None, 0.0)
        and all(
            value is None
            for value in (
                qv,
                score_mod,
                aux_tensors,
                q_descale,
                k_descale,
                v_descale,
                sfq,
                sfk,
                sfv,
                rel_bias,
                rel_bias_prep_cache,
            )
        )
    ):
        q, k_cache, v_cache = [_maybe_contiguous(t) for t in (q, k_cache, v_cache)]
        cu_seqlens_q, cache_seqlens, page_table = [
            _maybe_contiguous(t) for t in (cu_seqlens_q, cache_seqlens, page_table)
        ]
        fast_result = try_cached_paged_decode(
            arch=forward_arch,
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=cache_seqlens,
            page_table=page_table,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=sinks,
            requested_num_splits=num_splits,
            pack_gqa=pack_gqa,
            out=out,
        )
        if fast_result is not None:
            return fast_result[0]

    result = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        qv=qv,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap if softcap != 0.0 else None,
        window_size=window_size,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        learnable_sink=sinks,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        rel_bias=rel_bias,
        rel_bias_prep_cache=rel_bias_prep_cache,
        return_softmax_lse=return_softmax_lse if forward_arch is not None else True,
        out=out,
    )

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result
