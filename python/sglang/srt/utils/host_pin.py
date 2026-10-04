"""Pin large host buffers for CUDA in chunks, never in one call.

Linux kernels with torvalds/linux@53ba78de064b but not its fix 94efde1d1539
cap one long-term pin below 2 GiB, and a failed pin leaks the pages it had
already pinned until reboot. Every pin also kmallocs an array of 8 bytes per
page, so 64 MiB chunks keep that allocation at 128 KiB; 1 GiB chunks need
2 MiB allocations, which fail intermittently under fragmentation.

A buffer registered in chunks is device-visible through UVA as one range, so
kernels may read across chunk boundaries; a cudaMemcpy whose host range spans
two registrations fails, so these buffers are for direct (UVA) reads.
"""

from __future__ import annotations

import math
import mmap
from typing import Sequence

import torch

HOST_PIN_CHUNK_BYTES = 64 << 20

# Anonymous mappings pinned for the life of the process.
_PROCESS_LIFETIME_MAPPINGS: list[mmap.mmap] = []


def cuda_host_register_chunked(
    ptr: int, nbytes: int, chunk_bytes: int = HOST_PIN_CHUNK_BYTES
) -> None:
    """``cudaHostRegister`` ``[ptr, ptr + nbytes)`` in ``chunk_bytes`` pieces.

    On failure, unregisters the chunks already registered and raises.
    """
    cudart = torch.cuda.cudart()
    registered: list[int] = []
    try:
        for offset in range(0, nbytes, chunk_bytes):
            size = min(chunk_bytes, nbytes - offset)
            rc = int(cudart.cudaHostRegister(ptr + offset, size, 0))
            if rc != 0:
                raise RuntimeError(
                    f"cudaHostRegister of {size} bytes at offset {offset} of a "
                    f"{nbytes}-byte host buffer failed: "
                    f"{cudart.cudaGetErrorString(rc)} ({rc})"
                )
            registered.append(ptr + offset)
    except BaseException:
        for chunk in reversed(registered):
            cudart.cudaHostUnregister(chunk)
        raise


def empty_pinned_host(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """An uninitialized host tensor, pinned in chunks for the life of the process."""
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes == 0:
        return torch.empty(tuple(shape), dtype=dtype, device="cpu")
    mapping = mmap.mmap(
        -1,
        nbytes,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    raw = torch.frombuffer(mapping, dtype=torch.uint8)
    cuda_host_register_chunked(raw.data_ptr(), nbytes)
    _PROCESS_LIFETIME_MAPPINGS.append(mapping)
    return raw.view(dtype).view(*shape)
