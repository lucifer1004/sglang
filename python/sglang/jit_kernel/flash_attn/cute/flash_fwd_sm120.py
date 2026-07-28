# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.
#
# SM120 uses the same SM80-era MMA instructions (mma.sync.aligned.m16n8k16) but has
# a smaller shared memory capacity (99 KB vs 163 KB on SM80). Keep SM120-specific
# feature support and tile selection here even though the instruction-level kernel
# is shared with SM80.

import math

import cutlass
import cutlass.utils as utils_basic

from sglang.jit_kernel.flash_attn.cute.flash_fwd import (
    FlashAttentionForwardSm80,
)


class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
    supports_learnable_sink = True

    @staticmethod
    def _smem_usage_in_bytes(
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        Q_in_regs=False,
    ) -> int:
        """Return SMEM usage after padding head dimensions to kernel alignment."""
        head_dim = math.ceil(head_dim / 16) * 16
        head_dim_v = math.ceil(head_dim_v / 16) * 16
        element_size = 2
        smem_usage_Q = tile_m * head_dim * element_size
        smem_usage_K = tile_n * head_dim * num_stages * element_size
        smem_usage_V = tile_n * head_dim_v * num_stages * element_size
        smem_usage_QV = (
            smem_usage_Q + smem_usage_V
            if not Q_in_regs
            else max(smem_usage_Q, smem_usage_V)
        )
        return smem_usage_QV + smem_usage_K

    @staticmethod
    def get_fwd_tile_size(head_dim: int, head_dim_v: int) -> tuple[int, int]:
        """Select an SM120 tile that fits the architecture's 99 KB SMEM."""
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        candidates = (
            ((128, 128),) if max(head_dim, head_dim_v) <= 64 else ()
        ) + ((128, 64), (64, 64))
        for tile_m, tile_n in candidates:
            if (
                FlashAttentionForwardSm120._smem_usage_in_bytes(
                    head_dim,
                    head_dim_v,
                    tile_m,
                    tile_n,
                    num_stages=1,
                )
                <= smem_capacity
            ):
                return tile_m, tile_n
        raise ValueError(
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) exceeds "
            f"SM120 shared-memory capacity ({smem_capacity} bytes)"
        )

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
    ) -> bool:
        """Check if the kernel can be implemented on SM120.

        Same logic as SM80 but uses SM120's shared memory capacity (99 KB).
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        smem_usage = FlashAttentionForwardSm120._smem_usage_in_bytes(
            head_dim,
            head_dim_v,
            tile_m,
            tile_n,
            num_stages,
            Q_in_regs,
        )
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        if smem_usage > smem_capacity:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True
