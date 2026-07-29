# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
"""Host-side policy and launch state for the SM120 forward kernel.

The generic FA4 interface owns argument normalization, compilation, and
architecture dispatch. This module owns the SM120-specific decisions that
must remain consistent across those phases:

* tile, stage, and warp configuration;
* SplitKV sizing for paged decode;
* the direct single-wave specialization;
* cached TVM-FFI launch plans and their temporary workspaces.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import torch

from sglang.kernels.ops.attention.flash_attn.cute import fa_logging
from sglang.kernels.ops.attention.flash_attn.cute.flash_fwd_sm120 import (
    FlashAttentionForwardSm120,
)
from sglang.kernels.ops.attention.flash_attn.cute.utils import AuxData

_LAUNCH_PLAN_CAPACITY = 4096
_EMPTY_AUX_DATA = AuxData(None, None)


def supports_sm120_paged_decode(
    device_capability: tuple[int, int],
    head_dim: int,
) -> bool:
    """Return whether the qualified SM120 paged-decode path applies."""
    return device_capability[0] == 12 and head_dim == 256


@dataclass(frozen=True)
class Sm120ForwardConfig:
    tile_m: int
    tile_n: int
    num_stages: int
    num_threads: int


@dataclass(frozen=True)
class _VarlenLaunchPlan:
    compiled_fn: Callable
    compile_key: tuple


@dataclass(frozen=True)
class _PagedDecodeLaunchPlan:
    compiled_fn: Callable
    compile_key: tuple
    actual_num_splits: int
    compiled_combine: Optional[Callable]


@lru_cache(maxsize=None)
def _get_device_memory_bus_width(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).memory_bus_width


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    return (
        type(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.requires_grad,
    )


def _resolve_causal_local_window(
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
) -> tuple[bool, Optional[int], Optional[int]]:
    if causal:
        window_size_right = 0
    if (
        window_size_left is not None
        and window_size_right is not None
        and window_size_left + window_size_right < 0
    ):
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal = True
            window_size_right = None
        else:
            causal = False
    return causal, window_size_left, window_size_right


class Sm120ForwardHost:
    """SM120 forward configuration, scheduling, and direct-launch cache."""

    def __init__(self) -> None:
        self._varlen_plans: OrderedDict[tuple, _VarlenLaunchPlan] = OrderedDict()
        self._paged_plans: OrderedDict[tuple, _PagedDecodeLaunchPlan] = OrderedDict()
        self._paged_plan_tiles: dict[tuple, tuple[int, int]] = {}
        self._paged_workspaces: dict[
            tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._paged_workspace_views: dict[
            tuple[torch.device, int, int, tuple[int, ...], int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    @staticmethod
    def supports_arch(arch: int) -> bool:
        return arch // 10 == 12

    @staticmethod
    def implementation_token() -> tuple:
        return (
            FlashAttentionForwardSm120,
            FlashAttentionForwardSm120.get_fwd_tile_size,
        )

    @staticmethod
    def select_config(
        *,
        head_dim: int,
        head_dim_v: int,
        tile_mn: Optional[tuple[int, int]],
        total_q_rows: int,
        num_sms: Optional[int],
        num_batch: int,
        seqlen_q: Optional[int],
        seqlen_k: Optional[int],
        num_head_kv: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        is_local: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        pack_gqa: bool,
        paged_kv: bool,
    ) -> Sm120ForwardConfig:
        if tile_mn is None:
            tile_m, tile_n = FlashAttentionForwardSm120.get_fwd_tile_size(
                head_dim,
                head_dim_v,
                total_q_rows=total_q_rows,
                num_sms=num_sms,
                num_batch=num_batch,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                num_head_kv=num_head_kv,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                is_local=is_local,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                pack_gqa=pack_gqa,
            )
        else:
            tile_m, tile_n = tile_mn
        return Sm120ForwardConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=FlashAttentionForwardSm120.get_fwd_num_stages(
                head_dim, head_dim_v, tile_m, tile_n
            ),
            num_threads=FlashAttentionForwardSm120.get_fwd_num_threads(
                head_dim,
                head_dim_v,
                tile_m,
                tile_n,
                paged_kv=paged_kv,
            ),
        )

    @staticmethod
    def select_num_splits(
        *,
        requested_num_splits: int,
        head_dim: int,
        head_dim_v: int,
        paged_kv: bool,
        max_seqlen_q: int,
        pack_gqa: bool,
        total_mblocks: int,
        num_sms: int,
        num_n_blocks: int,
        device: torch.device,
        fake_mode: bool,
        generic_heuristic: Callable[[int, int, int, int], int],
    ) -> int:
        num_splits = requested_num_splits
        if num_splits < 1:
            is_hd256_paged_decode = (
                head_dim == 256
                and head_dim_v == 256
                and paged_kv
                and max_seqlen_q == 1
                and pack_gqa
            )
            if is_hd256_paged_decode:
                # Three CTAs per 32-bit memory channel is the measured
                # saturation point on the qualified SM120 SKUs.
                saturation_ctas = (
                    max(1, 3 * _get_device_memory_bus_width(device) // 32)
                    if not fake_mode
                    else max(1, num_sms // 4)
                )
                # Keep the B1 graph/reduction specialization power-of-two sized
                # without assuming that every SM120 SKU has 188 SMs.
                max_splits = min(128, 1 << (num_sms.bit_length() - 1))
                if num_n_blocks < 12 or total_mblocks >= saturation_ctas:
                    num_splits = 1
                else:
                    num_splits = generic_heuristic(
                        total_mblocks,
                        num_sms,
                        num_n_blocks,
                        max_splits,
                    )
            else:
                num_splits = generic_heuristic(
                    total_mblocks,
                    num_sms,
                    num_n_blocks,
                    128,
                )

        if num_splits <= 1 or num_n_blocks == 0:
            return 1

        # BlockInfo assigns a uniform ceil-div chunk to every split. Normalize
        # the count so no SM120 CTA owns an empty tail chunk.
        requested_splits = min(num_splits, num_n_blocks)
        blocks_per_split = (num_n_blocks + requested_splits - 1) // requested_splits
        return (num_n_blocks + blocks_per_split - 1) // blocks_per_split

    @staticmethod
    def use_direct_single_wave(
        *,
        batch_size: int,
        pack_gqa: bool,
        has_cu_seqlens_q: bool,
        has_seqused_q: bool,
        total_q: int,
        max_seqlen_q: int,
        total_mblocks: int,
        num_sms: int,
    ) -> bool:
        return (
            batch_size == 1
            and pack_gqa
            and has_cu_seqlens_q
            and not has_seqused_q
            and total_q == max_seqlen_q
            and total_mblocks <= num_sms
        )

    @staticmethod
    def make_kernel(
        *,
        dtype,
        head_dim: int,
        head_dim_v: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        is_local: bool,
        pack_gqa: bool,
        config: Sm120ForwardConfig,
        paged_kv: bool,
        score_mod: Optional[Callable],
        mask_mod: Optional[Callable],
        has_aux_tensors: bool,
        is_split_kv: bool,
        direct_single_wave: bool,
    ) -> FlashAttentionForwardSm120:
        if not FlashAttentionForwardSm120.can_implement(
            dtype,
            head_dim,
            head_dim_v,
            config.tile_m,
            config.tile_n,
            num_stages=config.num_stages,
            num_threads=config.num_threads,
            is_causal=is_causal,
            Q_in_regs=False,
            paged_kv=paged_kv,
        ):
            raise ValueError(
                "The requested FlashAttention forward configuration exceeds "
                "SM120 kernel constraints or shared-memory capacity"
            )
        return FlashAttentionForwardSm120(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=is_causal,
            is_local=is_local,
            pack_gqa=pack_gqa,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            num_stages=config.num_stages,
            num_threads=config.num_threads,
            Q_in_regs=False,
            score_mod=score_mod,
            mask_mod=mask_mod,
            has_aux_tensors=has_aux_tensors,
            is_split_kv=is_split_kv,
            direct_single_wave=direct_single_wave,
        )

    def _varlen_key(
        self,
        *,
        arch: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        learnable_sink: Optional[torch.Tensor],
        pack_gqa: bool,
    ) -> tuple:
        sink_signature = (
            None if learnable_sink is None else _tensor_signature(learnable_sink)
        )
        return (
            "basic-varlen",
            arch,
            tuple(_tensor_signature(t) for t in (q, k, v, cu_seqlens_q, cu_seqlens_k)),
            sink_signature,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            window_size_left,
            window_size_right,
            pack_gqa,
            self.implementation_token(),
            fa_logging.get_fa_log_level(),
        )

    def try_varlen(
        self,
        *,
        arch: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
        softmax_scale: Optional[float],
        causal: bool,
        window_size: tuple[Optional[int], Optional[int]],
        learnable_sink: Optional[torch.Tensor],
        pack_gqa: Optional[bool],
        out: Optional[torch.Tensor],
    ) -> Optional[tuple[torch.Tensor, None]]:
        if (
            not self.supports_arch(arch)
            or q.ndim != 3
            or k.ndim != 3
            or k.shape[1] == 0
        ):
            return None
        expected_out_shape = (*q.shape[:-1], v.shape[-1])
        if out is not None and (
            out.shape != expected_out_shape
            or out.dtype != q.dtype
            or out.device != q.device
            or not out.is_contiguous()
        ):
            return None
        actual_pack_gqa = q.shape[1] // k.shape[1] > 1 if pack_gqa is None else pack_gqa
        actual_max_seqlen_q = q.shape[0] if max_seqlen_q is None else max_seqlen_q
        actual_max_seqlen_k = k.shape[0] if max_seqlen_k is None else max_seqlen_k
        causal, window_size_left, window_size_right = _resolve_causal_local_window(
            causal,
            window_size[0],
            window_size[1],
        )
        key = self._varlen_key(
            arch=arch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=actual_max_seqlen_q,
            max_seqlen_k=actual_max_seqlen_k,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            learnable_sink=learnable_sink,
            pack_gqa=actual_pack_gqa,
        )
        plan = self._varlen_plans.get(key)
        if plan is None:
            return None
        self._varlen_plans.move_to_end(key)
        if out is None:
            out = torch.empty(
                expected_out_shape,
                dtype=q.dtype,
                device=q.device,
            )
        scale = 1.0 / math.sqrt(q.shape[-1]) if softmax_scale is None else softmax_scale
        plan.compiled_fn(
            q,
            k,
            v,
            out,
            None,
            scale,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
            window_size_left,
            window_size_right,
            learnable_sink,
            None,
            _EMPTY_AUX_DATA,
        )
        return out, None

    def register_varlen(
        self,
        *,
        arch: int,
        compiled_fn: Callable,
        compile_key: tuple,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        learnable_sink: Optional[torch.Tensor],
        pack_gqa: bool,
    ) -> None:
        key = self._varlen_key(
            arch=arch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            learnable_sink=learnable_sink,
            pack_gqa=pack_gqa,
        )
        self._varlen_plans[key] = _VarlenLaunchPlan(compiled_fn, compile_key)
        self._varlen_plans.move_to_end(key)
        while len(self._varlen_plans) > _LAUNCH_PLAN_CAPACITY:
            self._varlen_plans.popitem(last=False)

    def _paged_base_key(
        self,
        *,
        arch: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        page_table: torch.Tensor,
        max_seqlen_q: int,
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        learnable_sink: Optional[torch.Tensor],
        pack_gqa: bool,
        requested_num_splits: int,
    ) -> tuple:
        sink_signature = (
            None if learnable_sink is None else _tensor_signature(learnable_sink)
        )
        return (
            "paged-forward",
            arch,
            tuple(
                _tensor_signature(t)
                for t in (q, k, v, cu_seqlens_q, seqused_k, page_table)
            ),
            sink_signature,
            max_seqlen_q,
            causal,
            window_size_left,
            window_size_right,
            pack_gqa,
            requested_num_splits,
            self.implementation_token(),
            fa_logging.get_fa_log_level(),
        )

    @staticmethod
    def _paged_selection_key(
        base_key: tuple,
        tile_m: int,
        tile_n: int,
        max_seqlen_k: int,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
    ) -> tuple:
        is_local = window_size_left is not None or window_size_right is not None
        seqlen_k_loaded = (
            max_seqlen_k
            if not is_local
            else max(
                0,
                min(
                    max_seqlen_k,
                    (window_size_right or max_seqlen_k)
                    + (window_size_left or max_seqlen_k)
                    + 1
                    + tile_m,
                ),
            )
        )
        num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
        return (*base_key, (tile_m, tile_n, num_n_blocks))

    def _paged_workspace(
        self,
        plan: _PagedDecodeLaunchPlan,
        q: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stream_key = torch.cuda.current_stream(q.device).cuda_stream
        key = (q.device, stream_key)
        view_key = (
            q.device,
            stream_key,
            plan.actual_num_splits,
            tuple(q.shape[:-1]),
            v.shape[-1],
        )
        cached_view = self._paged_workspace_views.get(view_key)
        if cached_view is not None:
            return cached_view
        out_numel = plan.actual_num_splits * math.prod(q.shape[:-1]) * v.shape[-1]
        lse_numel = plan.actual_num_splits * q.shape[-2] * q.shape[0]
        workspace = self._paged_workspaces.get(key)
        if (
            workspace is None
            or workspace[0].numel() < out_numel
            or workspace[1].numel() < lse_numel
        ):
            self._drop_workspace_views(key)
            workspace = (
                torch.empty(out_numel, dtype=torch.float32, device=q.device),
                torch.empty(lse_numel, dtype=torch.float32, device=q.device),
            )
            self._paged_workspaces[key] = workspace
        out_partial = workspace[0][:out_numel].view(
            plan.actual_num_splits,
            *q.shape[:-1],
            v.shape[-1],
        )
        lse_partial = workspace[1][:lse_numel].view(
            plan.actual_num_splits,
            q.shape[-2],
            q.shape[0],
        )
        result = (out_partial, lse_partial, lse_partial.transpose(-1, -2))
        self._paged_workspace_views[view_key] = result
        return result

    def try_paged_decode(
        self,
        *,
        arch: int,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_q: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
        softmax_scale: Optional[float],
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        learnable_sink: Optional[torch.Tensor],
        requested_num_splits: int,
        pack_gqa: Optional[bool],
        out: Optional[torch.Tensor],
    ) -> Optional[tuple[torch.Tensor, None]]:
        if (
            not self.supports_arch(arch)
            or q is None
            or k is None
            or q.ndim != 3
            or k.ndim != 4
            or cu_seqlens_q is None
            or cu_seqlens_k is not None
            or seqused_q is not None
            or seqused_k is None
            or page_table is None
            or torch.cuda.is_current_stream_capturing()
            or any(t.requires_grad for t in (q, k, v))
            or (learnable_sink is not None and learnable_sink.requires_grad)
        ):
            return None
        expected_out_shape = (*q.shape[:-1], v.shape[-1])
        if out is not None and (
            out.shape != expected_out_shape
            or out.dtype != q.dtype
            or out.device != q.device
            or not out.is_contiguous()
        ):
            return None
        actual_pack_gqa = (
            q.shape[1] // k.shape[-2] > 1 if pack_gqa is None else pack_gqa
        )
        actual_max_seqlen_q = q.shape[0] if max_seqlen_q is None else max_seqlen_q
        actual_max_seqlen_k = (
            k.shape[0] * k.shape[1] if max_seqlen_k is None else max_seqlen_k
        )
        if (
            q.shape[-1] != 256
            or v.shape[-1] != 256
            or not actual_pack_gqa
            or actual_max_seqlen_q * (q.shape[-2] // v.shape[-2]) > 16
        ):
            return None
        causal, window_size_left, window_size_right = _resolve_causal_local_window(
            causal,
            window_size_left,
            window_size_right,
        )
        base_key = self._paged_base_key(
            arch=arch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            page_table=page_table,
            max_seqlen_q=actual_max_seqlen_q,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            learnable_sink=learnable_sink,
            pack_gqa=actual_pack_gqa,
            requested_num_splits=requested_num_splits,
        )
        tile_mn = self._paged_plan_tiles.get(base_key)
        if tile_mn is None:
            return None
        selection_key = self._paged_selection_key(
            base_key,
            *tile_mn,
            actual_max_seqlen_k,
            window_size_left,
            window_size_right,
        )
        plan = self._paged_plans.get(selection_key)
        if plan is None:
            return None
        self._paged_plans.move_to_end(selection_key)
        if out is None:
            out = torch.empty(
                expected_out_shape,
                dtype=q.dtype,
                device=q.device,
            )
        scale = 1.0 / math.sqrt(q.shape[-1]) if softmax_scale is None else softmax_scale
        kernel_out = out
        kernel_lse = None
        lse_partial_transposed = None
        if plan.actual_num_splits > 1:
            if plan.compiled_combine is None:
                return None
            kernel_out, kernel_lse, lse_partial_transposed = self._paged_workspace(
                plan, q, v
            )
        plan.compiled_fn(
            q,
            k,
            v,
            kernel_out,
            kernel_lse,
            scale,
            cu_seqlens_q,
            None,
            None,
            seqused_k,
            page_table,
            window_size_left,
            window_size_right,
            learnable_sink,
            None,
            _EMPTY_AUX_DATA,
        )
        if plan.actual_num_splits > 1:
            plan.compiled_combine(
                kernel_out,
                lse_partial_transposed,
                out,
                None,
                cu_seqlens_q,
                None,
                None,
                None,
                None,
            )
        return out, None

    def register_paged_decode(
        self,
        *,
        arch: int,
        compiled_fn: Callable,
        compile_key: tuple,
        compiled_combine: Optional[Callable],
        actual_num_splits: int,
        tile_m: int,
        tile_n: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        page_table: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        learnable_sink: Optional[torch.Tensor],
        pack_gqa: bool,
        requested_num_splits: int,
        out_partial: Optional[torch.Tensor],
        lse_partial: Optional[torch.Tensor],
    ) -> None:
        base_key = self._paged_base_key(
            arch=arch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            page_table=page_table,
            max_seqlen_q=max_seqlen_q,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            learnable_sink=learnable_sink,
            pack_gqa=pack_gqa,
            requested_num_splits=requested_num_splits,
        )
        selection_key = self._paged_selection_key(
            base_key,
            tile_m,
            tile_n,
            max_seqlen_k,
            window_size_left,
            window_size_right,
        )
        self._paged_plan_tiles[base_key] = (tile_m, tile_n)
        self._paged_plans[selection_key] = _PagedDecodeLaunchPlan(
            compiled_fn=compiled_fn,
            compile_key=compile_key,
            actual_num_splits=actual_num_splits,
            compiled_combine=compiled_combine,
        )
        self._paged_plans.move_to_end(selection_key)
        while len(self._paged_plans) > _LAUNCH_PLAN_CAPACITY:
            self._paged_plans.popitem(last=False)

        if out_partial is None or lse_partial is None:
            return
        stream_key = torch.cuda.current_stream(q.device).cuda_stream
        workspace_key = (q.device, stream_key)
        current_workspace = self._paged_workspaces.get(workspace_key)
        if (
            current_workspace is None
            or current_workspace[0].numel() < out_partial.numel()
            or current_workspace[1].numel() < lse_partial.numel()
        ):
            self._drop_workspace_views(workspace_key)
            self._paged_workspaces[workspace_key] = (
                out_partial.view(-1),
                lse_partial.view(-1),
            )
        view_key = (
            q.device,
            stream_key,
            actual_num_splits,
            tuple(q.shape[:-1]),
            v.shape[-1],
        )
        workspace = self._paged_workspaces[workspace_key]
        out_partial_view = workspace[0][: out_partial.numel()].view(out_partial.shape)
        lse_partial_view = workspace[1][: lse_partial.numel()].view(lse_partial.shape)
        self._paged_workspace_views[view_key] = (
            out_partial_view,
            lse_partial_view,
            lse_partial_view.transpose(-1, -2),
        )

    def _drop_workspace_views(self, workspace_key: tuple[torch.device, int]) -> None:
        stale_keys = [
            key for key in self._paged_workspace_views if key[:2] == workspace_key
        ]
        for key in stale_keys:
            del self._paged_workspace_views[key]

    def clear_launch_plans(self) -> None:
        self._varlen_plans.clear()
        self._paged_plans.clear()
        self._paged_plan_tiles.clear()
        self._paged_workspaces.clear()
        self._paged_workspace_views.clear()


sm120_forward_host = Sm120ForwardHost()
