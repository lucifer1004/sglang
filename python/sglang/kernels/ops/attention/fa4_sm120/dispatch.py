# Copyright (c) 2026, SGLang Team.
"""Lightweight dispatch bridge for SGLang-owned SM120 FA4 kernels.

The vendored FA4 interface dispatches through this module so the SM120
implementation and its launch state remain outside ``flash_attn/cute``.
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def get_forward_host(arch: int):
    """Return the optional architecture-owned forward host."""
    if arch // 10 == 12:
        from sglang.kernels.ops.attention.fa4_sm120.runtime import (
            sm120_forward_host,
        )

        return sm120_forward_host
    return None


def try_cached_paged_decode(*, arch: int, **kwargs):
    """Try an architecture-owned paged-decode launch plan."""
    host = get_forward_host(arch)
    return None if host is None else host.try_paged_decode(arch=arch, **kwargs)


def try_cached_varlen(*, arch: int, **kwargs):
    """Try an architecture-owned varlen launch plan."""
    host = get_forward_host(arch)
    return None if host is None else host.try_varlen(arch=arch, **kwargs)


def clear_forward_host_caches() -> None:
    """Clear host state coupled to compiled forward functions."""
    for arch in (120,):
        host = get_forward_host(arch)
        if host is not None:
            host.clear_launch_plans()
