from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

DUMMY_SLOT = 0


def unwrap_cp_sharded_allocator(allocator):
    return (
        allocator.base_allocator
        if isinstance(allocator, CPShardedKVPoolAllocator)
        else allocator
    )


def get_cp_owner(
    positions: torch.Tensor, cp_size: int, cp_kv_chunk_size: int
) -> torch.Tensor:
    if cp_size <= 0:
        raise ValueError("cp_size must be positive")
    if cp_kv_chunk_size <= 0:
        raise ValueError("cp_kv_chunk_size must be positive")
    return torch.div(positions, cp_kv_chunk_size, rounding_mode="floor") % cp_size


def filter_dummy_slots(slots: torch.Tensor) -> torch.Tensor:
    return slots[slots != DUMMY_SLOT]


def build_extend_positions(
    prefix_lens_cpu: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    if prefix_lens_cpu.numel() != extend_lens_cpu.numel():
        raise ValueError("prefix_lens_cpu and extend_lens_cpu must have same length")

    pieces = []
    for prefix_len, extend_len in zip(prefix_lens_cpu.tolist(), extend_lens_cpu.tolist()):
        if extend_len <= 0:
            continue
        pieces.append(
            torch.arange(
                int(prefix_len),
                int(prefix_len + extend_len),
                dtype=torch.int64,
                device=device,
            )
        )
    if not pieces:
        return torch.empty((0,), dtype=torch.int64, device=device)
    return torch.cat(pieces)


class CPShardedKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Position-aware wrapper for CP sharded KV residency.

    Slot 0 is a permanent DUMMY slot. Only positions owned by ``cp_rank`` are
    allocated from the wrapped allocator; all other positions map to slot 0.
    """

    def __init__(
        self,
        base_allocator: BaseTokenToKVPoolAllocator,
        *,
        cp_rank: int,
        cp_size: int,
        cp_kv_chunk_size: int,
        logical_size: Optional[int] = None,
        logical_full_size: Optional[int] = None,
        logical_swa_size: Optional[int] = None,
    ):
        if cp_rank < 0 or cp_rank >= cp_size:
            raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
        if cp_kv_chunk_size <= 0:
            raise ValueError("cp_kv_chunk_size must be positive")

        self.base_allocator = base_allocator
        self.size = (
            int(logical_size)
            if logical_size is not None
            else int(base_allocator.size) * cp_size
        )
        self.full_size = (
            int(logical_full_size)
            if logical_full_size is not None
            else self.size
        )
        self.swa_size = (
            int(logical_swa_size) if logical_swa_size is not None else self.size
        )
        self.page_size = base_allocator.page_size
        self.dtype = base_allocator.dtype
        self.device = base_allocator.device
        self.need_sort = base_allocator.need_sort
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.cp_kv_chunk_size = cp_kv_chunk_size

    def __getattr__(self, name: str):
        return getattr(self.base_allocator, name)

    def debug_print(self) -> str:
        return self.base_allocator.debug_print()

    def _logical_available(self, physical_available: int, logical_size: int) -> int:
        return min(int(logical_size), int(physical_available) * self.cp_size)

    def available_size(self):
        return self._logical_available(self.base_allocator.available_size(), self.size)

    def full_available_size(self):
        if hasattr(self.base_allocator, "full_available_size"):
            return self._logical_available(
                self.base_allocator.full_available_size(), self.full_size
            )
        return self.available_size()

    def swa_available_size(self):
        if hasattr(self.base_allocator, "swa_available_size"):
            return self._logical_available(
                self.base_allocator.swa_available_size(), self.swa_size
            )
        return self.available_size()

    def get_kvcache(self):
        return self.base_allocator.get_kvcache()

    def backup_state(self):
        return self.base_allocator.backup_state()

    def restore_state(self, state):
        self.base_allocator.restore_state(state)

    def free_group_begin(self):
        self.base_allocator.free_group_begin()

    def free_group_end(self):
        self.base_allocator.free_group_end()

    def clear(self):
        self.base_allocator.clear()

    def alloc(self, need_size: int):
        return self.base_allocator.alloc(need_size)

    def alloc_for_positions(self, positions: torch.Tensor) -> Optional[torch.Tensor]:
        positions = positions.to(dtype=torch.int64)
        owner_mask = (
            get_cp_owner(positions, self.cp_size, self.cp_kv_chunk_size) == self.cp_rank
        )
        local_count = int(owner_mask.sum().item())
        out_slots = torch.full_like(positions, DUMMY_SLOT, dtype=torch.int64)
        if local_count == 0:
            return out_slots

        local_slots = self.base_allocator.alloc(local_count)
        if local_slots is None:
            return None
        out_slots[owner_mask] = local_slots.to(device=out_slots.device)
        return out_slots

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.page_size != 1:
            raise NotImplementedError("CP sharded KV currently requires page_size=1")
        extend_lens_cpu = seq_lens_cpu - prefix_lens_cpu
        positions = build_extend_positions(
            prefix_lens_cpu, extend_lens_cpu, device=prefix_lens.device
        )
        if positions.numel() != extend_num_tokens:
            raise RuntimeError(
                "CP sharded extend position count mismatch: "
                f"{positions.numel()} vs {extend_num_tokens}"
            )
        return self.alloc_for_positions(positions)

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.page_size != 1:
            raise NotImplementedError("CP sharded KV currently requires page_size=1")
        return self.alloc_for_positions(seq_lens.to(dtype=torch.int64) - 1)

    def free(self, free_index: torch.Tensor):
        self.base_allocator.free(filter_dummy_slots(free_index))

    def free_swa(self, free_index: torch.Tensor):
        self.base_allocator.free_swa(filter_dummy_slots(free_index))
