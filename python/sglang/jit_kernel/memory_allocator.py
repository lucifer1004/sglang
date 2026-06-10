"""Custom memory allocator using CUDA pinned memory.

Provides ``custom_empty`` which allocates tensors via ``cudaMallocHost``
(pinned host memory presented as CUDA device tensors). This enables
zero-copy GPU access under Unified Virtual Addressing (UVA).

Migrated from prc_custom_ops to eliminate the external dependency.
Uses ctypes to call the CUDA runtime directly — no JIT compilation needed.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from typing import Tuple, Union

import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DTYPE_SIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.bool: 1,
}

_COMPACT_MEMORY_PATH = "/proc/sys/vm/compact_memory"
_CUDA_MALLOC_HOST_RETRY_LOCK = "/tmp/sglang_cuda_malloc_host_retry.lock"


def _element_size(dtype: torch.dtype) -> int:
    if dtype in _DTYPE_SIZE:
        return _DTYPE_SIZE[dtype]
    return torch.tensor([], dtype=dtype).element_size()


def _numel(sizes) -> int:
    n = 1
    for s in sizes:
        n *= s
    return n


def _cuda_malloc_host(cudart, total_bytes: int):
    host_ptr = ctypes.c_void_p(0)
    err = cudart.cudaMallocHost(ctypes.byref(host_ptr), ctypes.c_size_t(total_bytes))
    return err, host_ptr


def _compact_memory() -> None:
    with open(_COMPACT_MEMORY_PATH, "w") as f:
        f.write("1\n")


# Memory compaction is a host-wide operation. Serialize fallback retries so
# multiple TP workers do not compact memory concurrently after the same failure.
def _retry_cuda_malloc_host_after_compaction(cudart, total_bytes: int):
    import fcntl

    os.makedirs(os.path.dirname(_CUDA_MALLOC_HOST_RETRY_LOCK), exist_ok=True)
    with open(_CUDA_MALLOC_HOST_RETRY_LOCK, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            err, host_ptr = _cuda_malloc_host(cudart, total_bytes)
            if err == 0:
                return err, host_ptr, None

            compact_error = None
            try:
                _compact_memory()
            except OSError as e:
                compact_error = e

            err, host_ptr = _cuda_malloc_host(cudart, total_bytes)
            return err, host_ptr, compact_error
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


@lru_cache()
def _get_cudart():
    """Return a ctypes handle to the CUDA runtime library."""
    try:
        return ctypes.CDLL("libcudart.so")
    except OSError:
        return ctypes.CDLL("libcudart.so.12")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def custom_empty(
    sizes: Union[Tuple[int, ...], list],
    dtype: torch.dtype = torch.float32,
    device_id: int = 0,
) -> torch.Tensor:
    """Create an empty tensor backed by ``cudaMallocHost`` (pinned host memory).

    The tensor is presented as a CUDA device tensor. Under Unified Virtual
    Addressing (UVA), the pinned host pointer is directly usable as a CUDA
    device pointer, enabling zero-copy GPU access.

    Args:
        sizes: Shape of the tensor.
        dtype: Data type (default: ``torch.float32``).
        device_id: CUDA device ID (default: 0).

    Returns:
        A ``torch.Tensor`` on the specified CUDA device backed by pinned host memory.
    """
    sizes = tuple(sizes)
    total_bytes = _numel(sizes) * _element_size(dtype)

    if total_bytes == 0:
        return torch.empty(sizes, dtype=dtype, device=torch.device("cuda", device_id))

    cudart = _get_cudart()
    torch.cuda.set_device(device_id)

    err, host_ptr = _cuda_malloc_host(cudart, total_bytes)
    if err != 0:
        first_err = err
        try:
            err, host_ptr, compact_error = _retry_cuda_malloc_host_after_compaction(
                cudart, total_bytes
            )
        except Exception as e:
            raise RuntimeError(
                f"cudaMallocHost failed with error code {first_err} "
                f"(requested {total_bytes} bytes); retry after memory "
                f"compaction could not run: {e}"
            ) from e
        if err != 0:
            error_msg = (
                f"cudaMallocHost failed with error code {first_err} "
                f"(requested {total_bytes} bytes); retry after memory "
                f"compaction failed with error code {err}"
            )
            if compact_error is not None:
                error_msg += f"; memory compaction failed: {compact_error}"
            raise RuntimeError(error_msg)
    ptr = host_ptr.value

    def _free_host(p, _cudart=cudart, _c_void_p=ctypes.c_void_p):
        _cudart.cudaFreeHost(_c_void_p(p))

    tensor = _make_tensor_from_ptr(
        ptr, sizes, dtype, torch.device("cuda", device_id), _free_host
    )
    return tensor


# ---------------------------------------------------------------------------
# Tensor construction from raw pointer
# ---------------------------------------------------------------------------


def _make_tensor_from_ptr(
    ptr: int,
    sizes: tuple,
    dtype: torch.dtype,
    device: torch.device,
    free_fn,
) -> torch.Tensor:
    """Create a torch.Tensor from a raw memory pointer with a custom destructor.

    The destructor guard is attached to the *storage* object (not the tensor)
    because ``nn.Parameter(tensor)`` does not inherit custom Python attributes
    from the wrapped tensor. The storage is shared, so the guard survives.
    """
    numel = _numel(sizes)
    element_size = _element_size(dtype)
    total_bytes = numel * element_size

    storage = torch._C._construct_storage_from_data_pointer(ptr, device, total_bytes)

    ndim = len(sizes)
    strides: tuple = ()
    if ndim > 0:
        strides_list = [0] * ndim
        strides_list[-1] = 1
        for i in range(ndim - 2, -1, -1):
            strides_list[i] = strides_list[i + 1] * sizes[i + 1]
        strides = tuple(strides_list)

    tensor = torch.tensor([], dtype=dtype, device=device).set_(
        storage, 0, sizes, strides
    )

    # prevent premature deallocation
    class _prevent_free:
        __slots__ = ("_ptr", "_free", "_freed")

        def __init__(self, p, f):
            self._ptr = p
            self._free = f
            self._freed = False

        def __del__(self):
            if self._freed:
                return
            self._freed = True
            try:
                self._free(self._ptr)
            except Exception:
                pass

    storage._prevent_free = _prevent_free(ptr, free_fn)
    return tensor
