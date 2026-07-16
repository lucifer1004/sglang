from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.context_parallel import CPPrefillSplitSpec
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.cp_sharded_residency import (
    CPLogicalOwnerPlan,
    CPShardedKVLoadSnapshot,
    CPShardedKVResidencyLedger,
)

DUMMY_SLOT = 0


@dataclass(frozen=True)
class CPShardedKVAllocation:
    logical_slots: torch.Tensor
    physical_write_slots: torch.Tensor


def unwrap_cp_sharded_allocator(allocator):
    return (
        allocator.base_allocator
        if isinstance(allocator, CPShardedKVPoolAllocator)
        else allocator
    )


def get_cp_owner(
    positions: torch.Tensor,
    cp_size: int,
    cp_kv_chunk_size: int,
    owner_rotation: int | torch.Tensor = 0,
) -> torch.Tensor:
    if cp_size <= 0:
        raise ValueError("cp_size must be positive")
    if cp_kv_chunk_size <= 0:
        raise ValueError("cp_kv_chunk_size must be positive")
    chunk_idx = torch.div(positions, cp_kv_chunk_size, rounding_mode="floor")
    if cp_size == 1:
        return torch.zeros_like(chunk_idx)
    period = cp_size * 2
    phase = chunk_idx % period
    base_owner = torch.where(phase < cp_size, phase, period - phase - 1)
    return (base_owner + owner_rotation) % cp_size


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
    for prefix_len, extend_len in zip(
        prefix_lens_cpu.tolist(), extend_lens_cpu.tolist()
    ):
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
        use_decode_owner_layout: bool = False,
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
        self.use_decode_owner_layout = bool(use_decode_owner_layout)
        self.is_not_in_free_group = True
        self.free_group: list[torch.Tensor] = []
        self._init_logical_residency()
        if self.page_size > 1 and self.cp_kv_chunk_size % self.page_size != 0:
            raise ValueError(
                "cp_kv_chunk_size must be divisible by page_size for CP "
                "sharded KV with paged allocation"
            )

    def __getattr__(self, name: str):
        return getattr(self.base_allocator, name)

    def debug_print(self) -> str:
        return self.base_allocator.debug_print()

    def _logical_pool_available(
        self,
        logical_size: int,
        ledger: CPShardedKVResidencyLedger,
    ) -> int:
        snapshot = ledger.snapshot()
        allocated_tokens = sum(snapshot.allocated_tokens)
        pool_capacity = min(
            int(logical_size),
            self._logical_capacity_tokens,
            sum(snapshot.capacity_tokens),
        )
        return max(0, pool_capacity - allocated_tokens)

    def available_size(self):
        full_available = self.full_available_size()
        if self.swa_residency_ledger is None:
            return full_available
        return min(full_available, self.swa_available_size())

    def physical_available_size(self) -> int:
        return int(self.base_allocator.available_size())

    def full_available_size(self):
        return self._logical_pool_available(self.full_size, self.residency_ledger)

    def swa_available_size(self):
        if self.swa_residency_ledger is None:
            return self.full_available_size()
        return self._logical_pool_available(
            self.swa_size, self.swa_residency_ledger
        )

    def get_kvcache(self):
        return self.base_allocator.get_kvcache()

    def get_cpu_copy(self, indices, mamba_indices=None):
        physical_indices = self.logical_slots_to_physical(indices)
        return self.base_allocator.get_cpu_copy(
            physical_indices, mamba_indices=mamba_indices
        )

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        physical_indices = self.logical_slots_to_physical(indices)
        self.base_allocator.load_cpu_copy(
            kv_cache_cpu,
            physical_indices,
            mamba_indices=mamba_indices,
        )

    def restore_state(self, state):
        if isinstance(state, tuple) and len(state) in (6, 7):
            (
                base_state,
                logical_free,
                logical_to_physical,
                logical_page_free,
                logical_page_to_physical,
                residency_state,
                *optional_swa_state,
            ) = state
            self.base_allocator.restore_state(base_state)
            self.logical_free_slots = logical_free.clone()
            self.logical_to_physical_slot.copy_(logical_to_physical)
            self.logical_free_pages = logical_page_free.clone()
            self.logical_to_physical_page.copy_(logical_page_to_physical)
            self.residency_ledger.restore_state(residency_state)
            if optional_swa_state and optional_swa_state[0] is not None:
                if self.swa_residency_ledger is None:
                    raise ValueError("SWA residency state requires an SWA allocator")
                self.swa_residency_ledger.restore_state(optional_swa_state[0])
            return
        self.base_allocator.restore_state(state)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self._free_compacted(torch.cat(self.free_group))
            self.free_group = []

    def clear(self):
        self.is_not_in_free_group = True
        self.free_group = []
        self.base_allocator.clear()
        self._init_logical_residency()

    def _reset_graph_visible_mapping(self, name: str, size: int) -> None:
        current = self.__dict__.get(name)
        if current is None:
            setattr(
                self,
                name,
                torch.zeros(int(size), dtype=torch.int64, device=self.device),
            )
            return
        if current.numel() != int(size):
            raise RuntimeError(
                f"Cannot resize CUDA-graph-visible CP mapping {name}: "
                f"current={current.numel()}, expected={int(size)}"
            )
        current.zero_()

    def alloc(self, need_size: int):
        return self.base_allocator.alloc(need_size)

    def _init_logical_residency(self):
        logical_capacity = max(0, int(self.size))
        is_hybrid_swa = all(
            callable(getattr(self.base_allocator, name, None))
            for name in ("full_available_size", "swa_available_size")
        )
        full_capacity_per_rank = int(
            self.base_allocator.full_available_size()
            if is_hybrid_swa
            else self.base_allocator.available_size()
        )
        swa_capacity_per_rank = (
            int(self.base_allocator.swa_available_size())
            if is_hybrid_swa
            else None
        )
        if self.page_size == 1:
            self._logical_capacity_tokens = logical_capacity
            self.logical_free_slots = torch.arange(
                1, logical_capacity + 1, dtype=torch.int64, device=self.device
            )
            self._reset_graph_visible_mapping(
                "logical_to_physical_slot", logical_capacity + 1
            )
            self.logical_free_pages = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
            self._reset_graph_visible_mapping("logical_to_physical_page", 1)
            logical_unit_capacity = logical_capacity
            allocation_unit_tokens = 1
        else:
            logical_page_capacity = logical_capacity // self.page_size
            self._logical_capacity_tokens = logical_page_capacity * self.page_size
            self.logical_free_slots = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
            self._reset_graph_visible_mapping("logical_to_physical_slot", 1)
            self.logical_free_pages = torch.arange(
                1, logical_page_capacity + 1, dtype=torch.int64, device=self.device
            )
            self._reset_graph_visible_mapping(
                "logical_to_physical_page", logical_page_capacity + 1
            )
            logical_unit_capacity = logical_page_capacity
            allocation_unit_tokens = self.page_size

        self.residency_ledger = CPShardedKVResidencyLedger(
            cp_size=self.cp_size,
            logical_unit_capacity=logical_unit_capacity,
            allocation_unit_tokens=allocation_unit_tokens,
            physical_capacity_tokens_per_rank=full_capacity_per_rank,
            device=self.device,
        )
        self.swa_residency_ledger = (
            CPShardedKVResidencyLedger(
                cp_size=self.cp_size,
                logical_unit_capacity=logical_unit_capacity,
                allocation_unit_tokens=allocation_unit_tokens,
                physical_capacity_tokens_per_rank=swa_capacity_per_rank,
                device=self.device,
            )
            if swa_capacity_per_rank is not None
            else None
        )

    def backup_state(self):
        return (
            self.base_allocator.backup_state(),
            self.logical_free_slots.clone(),
            self.logical_to_physical_slot.clone(),
            self.logical_free_pages.clone(),
            self.logical_to_physical_page.clone(),
            self.residency_ledger.backup_state(),
            (
                self.swa_residency_ledger.backup_state()
                if self.swa_residency_ledger is not None
                else None
            ),
        )

    def physical_load_snapshot(self) -> CPShardedKVLoadSnapshot:
        full_snapshot = self.residency_ledger.snapshot()
        if self.swa_residency_ledger is None:
            return full_snapshot
        swa_snapshot = self.swa_residency_ledger.snapshot()
        return CPShardedKVLoadSnapshot(
            capacity_tokens=full_snapshot.capacity_tokens,
            allocated_tokens=full_snapshot.allocated_tokens,
            available_tokens_override=tuple(
                min(full_available, swa_available)
                for full_available, swa_available in zip(
                    full_snapshot.available_tokens,
                    swa_snapshot.available_tokens,
                )
            ),
        )

    def owner_plan_for_logical_pages(
        self, logical_pages: torch.Tensor
    ) -> CPLogicalOwnerPlan:
        if self.page_size == 1:
            raise ValueError("logical page owner plans require paged allocation")
        return self.residency_ledger.owner_plan_for_logical_units(logical_pages)

    def owner_plan_for_logical_slots(
        self, logical_slots: torch.Tensor
    ) -> CPLogicalOwnerPlan:
        logical_slots = logical_slots.reshape(-1).to(
            device=self.device, dtype=torch.int64
        )
        logical_units = (
            logical_slots
            if self.page_size == 1
            else torch.div(
                logical_slots, self.page_size, rounding_mode="floor"
            )
        )
        return self.residency_ledger.owner_plan_for_logical_units(logical_units)

    def _assign_residency(self, logical_units, owners) -> None:
        self.residency_ledger.assign(logical_units, owners)
        if self.swa_residency_ledger is not None:
            self.swa_residency_ledger.assign(logical_units, owners)

    def alloc_size_per_rank_for_range(
        self, start: int, length: int, owner_rotation: int = 0
    ) -> tuple[int, ...]:
        start = int(start)
        end = start + int(length)
        counts = [0] * self.cp_size
        if end <= start:
            return tuple(counts)

        if self.page_size == 1:
            first_alloc = start
        else:
            first_alloc = (
                (start + self.page_size - 1) // self.page_size * self.page_size
            )
        if first_alloc >= end:
            return tuple(counts)

        first_chunk = first_alloc // self.cp_kv_chunk_size
        last_chunk = (end - 1) // self.cp_kv_chunk_size
        for chunk_idx in range(first_chunk, last_chunk + 1):
            chunk_start = chunk_idx * self.cp_kv_chunk_size
            segment_start = max(first_alloc, chunk_start)
            segment_end = min(end, chunk_start + self.cp_kv_chunk_size)
            if self.page_size == 1:
                alloc_tokens = segment_end - segment_start
            else:
                alloc_tokens = (
                    (segment_end - segment_start + self.page_size - 1)
                    // self.page_size
                    * self.page_size
                )
            owner = self._owner_for_position(chunk_start, owner_rotation)
            counts[owner] += alloc_tokens
        return tuple(counts)

    def _zigzag_period(self) -> int:
        return max(1, self.cp_size * 2)

    def owner_period(self) -> int:
        return self.cp_size

    def _owner_for_position_tensor(
        self, positions: torch.Tensor, owner_rotation: int | torch.Tensor = 0
    ) -> torch.Tensor:
        if self.use_decode_owner_layout:
            chunk_idx = torch.div(
                positions, self.cp_kv_chunk_size, rounding_mode="floor"
            )
            return chunk_idx % self.cp_size
        return get_cp_owner(
            positions, self.cp_size, self.cp_kv_chunk_size, owner_rotation
        )

    def _owner_for_position(self, position: int, owner_rotation: int = 0) -> int:
        chunk_idx = int(position) // self.cp_kv_chunk_size
        if self.use_decode_owner_layout:
            return chunk_idx % self.cp_size
        period = self._zigzag_period()
        phase = chunk_idx % period
        base_owner = phase if phase < self.cp_size else period - phase - 1
        return (base_owner + int(owner_rotation)) % self.cp_size

    def _owns_page_start(self, page_start: int, owner_rotation: int = 0) -> bool:
        return self._owner_for_position(page_start, owner_rotation) == self.cp_rank

    def _range_local_alloc_size_for_rank(
        self, start: int, length: int, cp_rank: int, owner_rotation: int = 0
    ) -> int:
        return self.alloc_size_per_rank_for_range(
            start, length, owner_rotation
        )[cp_rank]

    def max_local_alloc_size_for_range(
        self, start: int, length: int, owner_rotation: int = 0
    ) -> int:
        return max(
            self.alloc_size_per_rank_for_range(
                start, length, owner_rotation
            ),
            default=0,
        )

    def local_alloc_size_for_range(
        self, start: int, length: int, owner_rotation: int = 0
    ) -> int:
        return self._range_local_alloc_size_for_rank(
            start, length, self.cp_rank, owner_rotation
        )

    def _alloc_logical_slots(self, count: int) -> Optional[torch.Tensor]:
        count = int(count)
        if count <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        if count > int(self.logical_free_slots.numel()):
            return None
        slots = self.logical_free_slots[:count].clone()
        self.logical_free_slots = self.logical_free_slots[count:]
        return slots

    def _alloc_logical_pages(self, count: int) -> Optional[torch.Tensor]:
        count = int(count)
        if count <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        if count > int(self.logical_free_pages.numel()):
            return None
        pages = self.logical_free_pages[:count].clone()
        self.logical_free_pages = self.logical_free_pages[count:]
        return pages

    def _rollback_logical_pages(self, pages: torch.Tensor) -> None:
        if pages.numel() == 0:
            return
        self.logical_free_pages = torch.cat(
            (pages.to(device=self.device), self.logical_free_pages)
        )

    def _free_logical_slots(self, slots: torch.Tensor):
        slots = filter_dummy_slots(slots.reshape(-1).to(dtype=torch.int64))
        if slots.numel() == 0:
            return
        slots = torch.unique(slots.to(device=self.device))
        slots = self.residency_ledger.release(slots)
        if self.swa_residency_ledger is not None:
            self.swa_residency_ledger.release(slots)
        physical_slots = self.logical_to_physical_slot[slots]
        owned_physical = filter_dummy_slots(physical_slots)
        if owned_physical.numel() > 0:
            self.base_allocator.free(owned_physical)
        self.logical_to_physical_slot[slots] = 0
        self.logical_free_slots = torch.cat((self.logical_free_slots, slots))

    def _free_logical_pages_from_slots(self, logical_slots: torch.Tensor):
        logical_slots = filter_dummy_slots(
            logical_slots.reshape(-1).to(dtype=torch.int64)
        )
        if logical_slots.numel() == 0:
            return
        logical_pages = torch.unique(
            logical_slots.to(device=self.device) // self.page_size
        )
        logical_pages = logical_pages[logical_pages != 0]
        if logical_pages.numel() == 0:
            return
        logical_pages = self.residency_ledger.release(logical_pages)
        if self.swa_residency_ledger is not None:
            self.swa_residency_ledger.release(logical_pages)
        physical_pages = self.logical_to_physical_page[logical_pages]
        owned_physical_pages = filter_dummy_slots(physical_pages)
        if owned_physical_pages.numel() > 0:
            self._free_base_pages(owned_physical_pages)
        self.logical_to_physical_page[logical_pages] = 0
        self.logical_free_pages = torch.cat((self.logical_free_pages, logical_pages))

    def logical_page_table_to_physical(self, page_table: torch.Tensor) -> torch.Tensor:
        logical = torch.clamp(
            page_table.to(device=self.device, dtype=torch.long), min=0
        )
        if self.page_size == 1:
            physical = self.logical_to_physical_slot[logical]
        else:
            physical = self.logical_to_physical_page[logical]
        return physical.to(device=page_table.device, dtype=page_table.dtype)

    def logical_slots_to_physical(self, slots: torch.Tensor) -> torch.Tensor:
        logical_slots = torch.clamp(
            slots.to(device=self.device, dtype=torch.int64), min=0
        )
        if self.page_size == 1:
            physical_slots = self.logical_to_physical_slot[logical_slots]
        else:
            logical_pages = torch.div(
                logical_slots, self.page_size, rounding_mode="floor"
            )
            page_offsets = logical_slots % self.page_size
            physical_pages = self.logical_to_physical_page[logical_pages]
            physical_slots = torch.where(
                physical_pages.ne(0),
                physical_pages * self.page_size + page_offsets,
                torch.zeros_like(physical_pages),
            )
        return physical_slots.to(device=slots.device, dtype=slots.dtype)

    def logical_swa_page_table_to_physical(
        self, logical_page_table: torch.Tensor
    ) -> torch.Tensor:
        if self.swa_residency_ledger is None:
            raise ValueError("SWA page translation requires a paired SWA allocator")
        full_pages = self.logical_page_table_to_physical(logical_page_table).to(
            device=self.device, dtype=torch.int64
        )
        owned_mask = full_pages.ne(0)
        swa_pages = torch.zeros_like(full_pages)
        full_slots = full_pages[owned_mask] * self.page_size
        if full_slots.numel() > 0:
            swa_slots = self.base_allocator.translate_loc_from_full_to_swa(full_slots)
            swa_pages[owned_mask] = torch.div(
                swa_slots.to(dtype=swa_pages.dtype),
                self.page_size,
                rounding_mode="floor",
            )
        return swa_pages.to(
            device=logical_page_table.device, dtype=logical_page_table.dtype
        )

    def _logical_slots_to_physical_slots(self, slots: torch.Tensor) -> torch.Tensor:
        physical_slots = self.logical_slots_to_physical(
            slots.reshape(-1).to(device=self.device, dtype=torch.int64)
        )
        return filter_dummy_slots(physical_slots)

    def _alloc_page_count(self, count: int) -> Optional[torch.Tensor]:
        if count <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        local_slots = self._alloc_page_slots(count)
        if local_slots is None:
            return None
        return local_slots[:: self.page_size]

    @staticmethod
    def _merge_allocator_released_pages_without_sort(allocator):
        release_pages = allocator.release_pages
        if release_pages is None or len(release_pages) == 0:
            return
        allocator.free_pages = torch.cat((allocator.free_pages, release_pages))
        allocator.release_pages = torch.empty(
            (0,), dtype=release_pages.dtype, device=release_pages.device
        )

    def _merge_released_pages_without_sort(self):
        self._merge_allocator_released_pages_without_sort(self.base_allocator)

    def _has_paired_swa_allocator(self) -> bool:
        return (
            self.page_size > 1
            and hasattr(self.base_allocator, "full_attn_allocator")
            and hasattr(self.base_allocator, "swa_attn_allocator")
            and hasattr(self.base_allocator, "set_full_to_swa_mapping")
            and hasattr(self.base_allocator, "full_to_swa_index_mapping")
        )

    def _alloc_paired_swa_page_slots(self, count: int) -> Optional[torch.Tensor]:
        full_allocator = self.base_allocator.full_attn_allocator
        swa_allocator = self.base_allocator.swa_attn_allocator
        for allocator in (full_allocator, swa_allocator):
            if allocator.need_sort and int(count) > len(allocator.free_pages):
                self._merge_allocator_released_pages_without_sort(allocator)

        if int(count) > len(full_allocator.free_pages):
            return None
        if int(count) > len(swa_allocator.free_pages):
            return None

        need_size = int(count) * self.page_size
        full_slots = full_allocator.alloc(need_size)
        if full_slots is None:
            return None
        swa_slots = swa_allocator.alloc(need_size)
        if swa_slots is None:
            full_allocator.free(full_slots)
            return None

        self.base_allocator.set_full_to_swa_mapping(full_slots, swa_slots)
        return full_slots

    def _alloc_page_slots(self, count: int) -> Optional[torch.Tensor]:
        if count <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        if self._has_paired_swa_allocator():
            return self._alloc_paired_swa_page_slots(count)
        if (
            self.base_allocator.need_sort
            and int(count) > len(self.base_allocator.free_pages)
        ):
            # CP sharded residency does not require globally sorted physical page
            # reuse. Sorting the release list on the decode hot path stalls high
            # concurrency PD decode, so merge released pages directly.
            self._merge_released_pages_without_sort()
        alloc_contiguous = getattr(self.base_allocator, "alloc_contiguous", None)
        if alloc_contiguous is not None:
            contiguous_slots = alloc_contiguous(int(count))
            if contiguous_slots is not None:
                return contiguous_slots
        return self.base_allocator.alloc(int(count) * self.page_size)

    def alloc_for_positions(
        self,
        positions: torch.Tensor,
        *,
        positions_cpu: Optional[torch.Tensor] = None,
        owner_rotations_cpu: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        result = self.alloc_for_positions_with_logical(
            positions,
            positions_cpu=positions_cpu,
            owner_rotations_cpu=owner_rotations_cpu,
        )
        if result is None:
            return None
        return result.physical_write_slots

    def alloc_for_positions_with_logical(
        self,
        positions: torch.Tensor,
        *,
        positions_cpu: Optional[torch.Tensor] = None,
        owner_rotations_cpu: Optional[torch.Tensor] = None,
    ) -> Optional[CPShardedKVAllocation]:
        positions = positions.to(dtype=torch.int64)
        owner_positions = positions
        if positions_cpu is not None:
            if tuple(positions_cpu.shape) != tuple(positions.shape):
                raise ValueError(
                    "positions_cpu must have the same shape as positions, "
                    f"got {tuple(positions_cpu.shape)} vs {tuple(positions.shape)}"
                )
            owner_positions = positions_cpu.to(dtype=torch.int64)
        if owner_rotations_cpu is None:
            owner_rotations = torch.zeros_like(owner_positions, dtype=torch.int64)
        else:
            if tuple(owner_rotations_cpu.shape) != tuple(positions.shape):
                raise ValueError(
                    "owner_rotations_cpu must have the same shape as positions, "
                    f"got {tuple(owner_rotations_cpu.shape)} vs {tuple(positions.shape)}"
                )
            owner_rotations = owner_rotations_cpu.to(dtype=torch.int64)
        owner_ranks = self._owner_for_position_tensor(
            owner_positions, owner_rotations
        )
        owner_mask = owner_ranks == self.cp_rank
        if self.page_size > 1:
            page_owner_positions = (
                torch.div(owner_positions, self.page_size, rounding_mode="floor")
                * self.page_size
            )
            page_owner_ranks = self._owner_for_position_tensor(
                page_owner_positions, owner_rotations
            )
            owner_mask = page_owner_ranks == self.cp_rank
            page_keys: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for page_start, rotation in zip(
                page_owner_positions.reshape(-1).to("cpu").tolist(),
                owner_rotations.reshape(-1).to("cpu").tolist(),
            ):
                key = (int(page_start), int(rotation))
                if key in seen:
                    continue
                seen.add(key)
                page_keys.append(key)

            logical_pages = self._alloc_logical_pages(len(page_keys))
            if logical_pages is None:
                return None
            logical_page_map = {
                key: int(page)
                for key, page in zip(page_keys, logical_pages.to("cpu").tolist())
            }
            page_owners = [
                self._owner_for_position(page_start, owner_rotation=rotation)
                for page_start, rotation in page_keys
            ]

            owned_page_keys = [
                key
                for key, owner in zip(page_keys, page_owners)
                if owner == self.cp_rank
            ]
            page_base_slots = self._alloc_page_count(len(owned_page_keys))
            if page_base_slots is None:
                self.logical_free_pages = torch.cat(
                    (self.logical_free_pages, logical_pages)
                )
                return None
            owned_page_key_set = set(owned_page_keys)
            owned_page_indices = [
                idx for idx, key in enumerate(page_keys) if key in owned_page_key_set
            ]
            owned_logical_pages = logical_pages[owned_page_indices]
            owned_physical_pages = page_base_slots.to(
                dtype=self.logical_to_physical_page.dtype
            ) // self.page_size
            if owned_logical_pages.numel() > 0:
                self.logical_to_physical_page[owned_logical_pages] = (
                    owned_physical_pages
                )
            self._assign_residency(logical_pages, page_owners)

            flat_positions_cpu = positions.to("cpu").reshape(-1).tolist()
            flat_page_starts = page_owner_positions.reshape(-1).to("cpu").tolist()
            flat_rotations = owner_rotations.reshape(-1).to("cpu").tolist()
            logical_values = []
            for pos, page_start, rotation in zip(
                flat_positions_cpu, flat_page_starts, flat_rotations
            ):
                key = (int(page_start), int(rotation))
                logical_page = logical_page_map[key]
                page_start = (int(pos) // self.page_size) * self.page_size
                page_offset = int(pos) - page_start
                logical_values.append(logical_page * self.page_size + page_offset)

            logical_out = torch.tensor(
                logical_values, dtype=torch.int64, device=positions.device
            ).view_as(positions)
            logical_page_ids = torch.div(
                logical_out, self.page_size, rounding_mode="floor"
            )
            physical_pages = self.logical_to_physical_page[
                logical_page_ids.to(device=self.device)
            ].to(device=positions.device)
            page_offsets = logical_out % self.page_size
            physical_out = torch.where(
                owner_mask.to(device=positions.device),
                physical_pages * self.page_size + page_offsets,
                torch.full_like(logical_out, DUMMY_SLOT),
            )
            return CPShardedKVAllocation(
                logical_slots=logical_out,
                physical_write_slots=physical_out,
            )

        logical_slots = self._alloc_logical_slots(positions.numel())
        if logical_slots is None:
            return None
        logical_slots = logical_slots.to(device=positions.device).view_as(positions)
        local_count = int(owner_mask.to(device="cpu").sum())
        if local_count == 0:
            self._assign_residency(
                logical_slots.reshape(-1), owner_ranks.reshape(-1)
            )
            return CPShardedKVAllocation(
                logical_slots=logical_slots,
                physical_write_slots=torch.full_like(
                    positions, DUMMY_SLOT, dtype=torch.int64
                ),
            )

        local_slots = self.base_allocator.alloc(local_count)
        if local_slots is None:
            self.logical_free_slots = torch.cat(
                (self.logical_free_slots, logical_slots.reshape(-1).to(self.device))
            )
            return None
        if local_count == positions.numel():
            physical_slots = local_slots.to(device=positions.device).view_as(positions)
            self.logical_to_physical_slot[
                logical_slots.reshape(-1).to(self.device)
            ] = physical_slots.reshape(-1).to(self.device)
            self._assign_residency(
                logical_slots.reshape(-1), owner_ranks.reshape(-1)
            )
            return CPShardedKVAllocation(
                logical_slots=logical_slots,
                physical_write_slots=physical_slots,
            )

        out_slots = torch.full_like(positions, DUMMY_SLOT, dtype=torch.int64)
        owner_indices = owner_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        if owner_indices.device != out_slots.device:
            owner_indices = owner_indices.to(device=out_slots.device, non_blocking=True)
        out_slots.view(-1)[owner_indices] = local_slots.to(device=out_slots.device)
        self.logical_to_physical_slot[
            logical_slots.reshape(-1).to(self.device)[owner_indices.to(self.device)]
        ] = local_slots.to(self.device)
        self._assign_residency(
            logical_slots.reshape(-1), owner_ranks.reshape(-1)
        )
        return CPShardedKVAllocation(
            logical_slots=logical_slots,
            physical_write_slots=out_slots,
        )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        owner_rotations_cpu: Optional[torch.Tensor] = None,
        split_spec: Optional[CPPrefillSplitSpec] = None,
    ):
        result = self.alloc_extend_with_logical(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            owner_rotations_cpu=owner_rotations_cpu,
            split_spec=split_spec,
        )
        if result is None:
            return None
        return result.physical_write_slots

    def alloc_extend_with_logical(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        owner_rotations_cpu: Optional[torch.Tensor] = None,
        split_spec: Optional[CPPrefillSplitSpec] = None,
    ) -> Optional[CPShardedKVAllocation]:
        if self.use_decode_owner_layout and split_spec is not None:
            raise ValueError(
                "Decode CP allocation cannot consume a Prefill split spec"
            )
        if self.page_size != 1:
            return self._alloc_extend_paged(
                prefix_lens_cpu,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                device=prefix_lens.device,
                owner_rotations_cpu=owner_rotations_cpu,
                split_spec=split_spec,
            )
        if split_spec is not None:
            raise ValueError("CP prefill split specs require paged allocation")
        extend_lens_cpu = seq_lens_cpu - prefix_lens_cpu
        positions = build_extend_positions(
            prefix_lens_cpu, extend_lens_cpu, device=prefix_lens.device
        )
        if positions.numel() != extend_num_tokens:
            raise RuntimeError(
                "CP sharded extend position count mismatch: "
                f"{positions.numel()} vs {extend_num_tokens}"
            )
        if owner_rotations_cpu is None:
            owner_rotations_cpu = torch.zeros_like(prefix_lens_cpu, dtype=torch.int64)
        rotation_pieces = []
        for rotation, extend_len in zip(
            owner_rotations_cpu.to(dtype=torch.int64).tolist(),
            extend_lens_cpu.to(dtype=torch.int64).tolist(),
        ):
            if int(extend_len) <= 0:
                continue
            rotation_pieces.append(
                torch.full((int(extend_len),), int(rotation), dtype=torch.int64)
            )
        position_rotations_cpu = (
            torch.cat(rotation_pieces)
            if rotation_pieces
            else torch.empty((0,), dtype=torch.int64)
        )
        return self.alloc_for_positions_with_logical(
            positions,
            positions_cpu=positions.to("cpu"),
            owner_rotations_cpu=position_rotations_cpu,
        )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        owner_rotations_cpu: Optional[torch.Tensor] = None,
    ):
        result = self.alloc_decode_with_logical(
            seq_lens,
            seq_lens_cpu,
            last_loc,
            owner_rotations_cpu=owner_rotations_cpu,
        )
        if result is None:
            return None
        return result.physical_write_slots

    def alloc_decode_with_logical(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        owner_rotations_cpu: Optional[torch.Tensor] = None,
    ) -> Optional[CPShardedKVAllocation]:
        if self.page_size != 1:
            return self._alloc_decode_paged(
                seq_lens,
                seq_lens_cpu,
                last_loc,
                device=seq_lens.device,
                owner_rotations_cpu=owner_rotations_cpu,
            )
        positions = seq_lens.to(dtype=torch.int64) - 1
        if owner_rotations_cpu is None:
            owner_rotations_cpu = torch.zeros_like(seq_lens_cpu, dtype=torch.int64)
        return self.alloc_for_positions_with_logical(
            positions,
            positions_cpu=positions.to("cpu"),
            owner_rotations_cpu=owner_rotations_cpu,
        )

    def _alloc_extend_paged_single_empty(
        self,
        seq_len: int,
        extend_num_tokens: int,
        *,
        device: torch.device | str,
        owner_rotation: int = 0,
    ) -> Optional[CPShardedKVAllocation]:
        if seq_len != extend_num_tokens:
            raise RuntimeError(
                "CP sharded paged empty-prefix allocation expects seq_len to "
                f"match extend_num_tokens, got {seq_len} vs {extend_num_tokens}"
            )

        num_pages = (int(seq_len) + self.page_size - 1) // self.page_size
        logical_pages = self._alloc_logical_pages(num_pages)
        if logical_pages is None:
            return None
        page_owners = [
            self._owner_for_position(
                idx * self.page_size, owner_rotation=owner_rotation
            )
            for idx in range(num_pages)
        ]
        owned_ord = [
            idx for idx, owner in enumerate(page_owners) if owner == self.cp_rank
        ]
        num_owned_pages = len(owned_ord)
        local_slots = self._alloc_page_slots(num_owned_pages)
        if local_slots is None:
            self.logical_free_pages = torch.cat(
                (self.logical_free_pages, logical_pages)
            )
            return None

        physical_out = torch.full(
            (seq_len,), DUMMY_SLOT, dtype=torch.int64, device=device
        )
        logical_out = torch.empty((seq_len,), dtype=torch.int64, device=device)
        self._assign_residency(logical_pages, page_owners)
        if seq_len == 0 or num_owned_pages == 0:
            if seq_len > 0:
                positions = torch.arange(seq_len, dtype=torch.int64, device=device)
                logical_page_ord = torch.div(
                    positions, self.page_size, rounding_mode="floor"
                )
                page_offset = positions - logical_page_ord * self.page_size
                logical_out.copy_(
                    logical_pages.to(device=device)[logical_page_ord] * self.page_size
                    + page_offset
                )
            return CPShardedKVAllocation(
                logical_slots=logical_out,
                physical_write_slots=physical_out,
            )

        owned_logical_pages = logical_pages[owned_ord]
        owned_physical_pages = local_slots[:: self.page_size] // self.page_size
        if owned_logical_pages.numel() > 0:
            self.logical_to_physical_page[owned_logical_pages.to(self.device)] = (
                owned_physical_pages.to(self.device)
            )

        positions = torch.arange(seq_len, dtype=torch.int64, device=device)
        logical_page = torch.div(positions, self.page_size, rounding_mode="floor")
        logical_page_ids = logical_pages.to(device=device)[logical_page]
        page_offset = positions - logical_page * self.page_size
        logical_out.copy_(logical_page_ids * self.page_size + page_offset)

        page_starts = logical_page * self.page_size
        owner_mask = (
            self._owner_for_position_tensor(page_starts, owner_rotation)
            == self.cp_rank
        )
        physical_pages = self.logical_to_physical_page[
            logical_page_ids.to(device=self.device)
        ].to(device=device)
        physical_slots = physical_pages * self.page_size + page_offset
        physical_out.copy_(torch.where(owner_mask, physical_slots, physical_out))
        return CPShardedKVAllocation(
            logical_slots=logical_out,
            physical_write_slots=physical_out,
        )

    def _alloc_extend_paged(
        self,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        *,
        device: torch.device | str,
        owner_rotations_cpu: Optional[torch.Tensor] = None,
        split_spec: Optional[CPPrefillSplitSpec] = None,
    ) -> Optional[CPShardedKVAllocation]:
        if split_spec is not None:
            return self._alloc_extend_paged_explicit(
                prefix_lens_cpu,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                split_spec=split_spec,
                device=device,
                owner_rotations_cpu=owner_rotations_cpu,
            )

        prefix_lens_list = [int(x) for x in prefix_lens_cpu.tolist()]
        seq_lens_list = [int(x) for x in seq_lens_cpu.tolist()]
        if owner_rotations_cpu is None:
            owner_rotations_list = [0 for _ in prefix_lens_list]
        else:
            owner_rotations_list = [int(x) for x in owner_rotations_cpu.tolist()]

        if len(prefix_lens_list) == 1 and prefix_lens_list[0] == 0:
            return self._alloc_extend_paged_single_empty(
                seq_lens_list[0],
                extend_num_tokens,
                device=device,
                owner_rotation=owner_rotations_list[0],
            )

        last_loc_device = last_loc.to(device=device, dtype=torch.int64).reshape(-1)

        page_keys: list[tuple[int, int]] = []
        page_owners: list[int] = []
        owned_page_keys: list[tuple[int, int]] = []
        for req_idx, (prefix_len, seq_len, owner_rotation) in enumerate(
            zip(prefix_lens_list, seq_lens_list, owner_rotations_list)
        ):
            first_new_page = (
                (prefix_len + self.page_size - 1) // self.page_size
            ) * self.page_size
            for page_start in range(first_new_page, seq_len, self.page_size):
                key = (req_idx, page_start)
                page_keys.append(key)
                owner = self._owner_for_position(page_start, owner_rotation)
                page_owners.append(owner)
                if owner == self.cp_rank:
                    owned_page_keys.append(key)

        logical_pages = self._alloc_logical_pages(len(page_keys))
        if logical_pages is None:
            return None

        page_base_slots = self._alloc_page_count(len(owned_page_keys))
        if page_base_slots is None:
            self.logical_free_pages = torch.cat(
                (self.logical_free_pages, logical_pages)
            )
            return None
        owned_page_indices = [
            idx for idx, owner in enumerate(page_owners) if owner == self.cp_rank
        ]
        owned_logical_pages = logical_pages[owned_page_indices]
        owned_physical_pages = page_base_slots.to(
            dtype=self.logical_to_physical_page.dtype
        ) // self.page_size
        if owned_logical_pages.numel() > 0:
            self.logical_to_physical_page[owned_logical_pages] = (
                owned_physical_pages
            )
        self._assign_residency(logical_pages, page_owners)

        logical_chunks = []
        physical_chunks = []
        page_cursor = 0
        for req_idx, (prefix_len, seq_len, owner_rotation) in enumerate(
            zip(prefix_lens_list, seq_lens_list, owner_rotations_list)
        ):
            if seq_len <= prefix_len:
                continue

            positions = torch.arange(
                prefix_len, seq_len, dtype=torch.int64, device=device
            )
            page_starts = torch.div(
                positions, self.page_size, rounding_mode="floor"
            ) * self.page_size
            first_new_page = (
                (prefix_len + self.page_size - 1) // self.page_size
            ) * self.page_size
            new_page_count = max(
                0, (seq_len - first_new_page + self.page_size - 1) // self.page_size
            )
            new_page_ids = logical_pages[page_cursor : page_cursor + new_page_count]
            page_cursor += new_page_count
            page_ord = torch.div(
                torch.clamp(page_starts - first_new_page, min=0),
                self.page_size,
                rounding_mode="floor",
            )
            if new_page_ids.numel() > 0:
                new_logical_slots = (
                    new_page_ids[page_ord] * self.page_size
                    + positions
                    - page_starts
                )
            else:
                new_logical_slots = torch.zeros_like(positions)
            logical_slots = torch.where(
                positions < first_new_page,
                last_loc_device[req_idx]
                + positions
                - prefix_len
                + 1,
                new_logical_slots,
            )
            logical_page_ids = torch.div(
                logical_slots, self.page_size, rounding_mode="floor"
            )

            owner_mask = (
                self._owner_for_position_tensor(
                    page_starts, owner_rotation=owner_rotation
                )
                == self.cp_rank
            )
            physical_pages = self.logical_to_physical_page[
                logical_page_ids.to(device=self.device)
            ].to(device=device)
            physical_slots = physical_pages * self.page_size + (
                logical_slots % self.page_size
            )

            logical_chunks.append(logical_slots)
            physical_chunks.append(
                torch.where(
                    owner_mask,
                    physical_slots,
                    torch.full_like(physical_slots, DUMMY_SLOT),
                )
            )

        logical_out = (
            torch.cat(logical_chunks)
            if logical_chunks
            else torch.empty((0,), dtype=torch.int64, device=device)
        )
        physical_out = (
            torch.cat(physical_chunks)
            if physical_chunks
            else torch.empty((0,), dtype=torch.int64, device=device)
        )
        return CPShardedKVAllocation(
            logical_slots=logical_out,
            physical_write_slots=physical_out,
        )

    def _alloc_extend_paged_explicit(
        self,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        *,
        split_spec: CPPrefillSplitSpec,
        device: torch.device | str,
        owner_rotations_cpu: Optional[torch.Tensor] = None,
    ) -> Optional[CPShardedKVAllocation]:
        if prefix_lens_cpu.numel() != 1 or seq_lens_cpu.numel() != 1:
            raise ValueError("one CP prefill split spec requires batch_size=1")

        prefix_len = int(prefix_lens_cpu.reshape(-1).tolist()[0])
        seq_len = int(seq_lens_cpu.reshape(-1).tolist()[0])
        if seq_len < prefix_len:
            raise ValueError("sequence length must not be smaller than prefix length")
        if split_spec.extend_start != prefix_len:
            raise ValueError("CP prefill split start must match prefix length")
        if split_spec.extend_len != seq_len - prefix_len:
            raise ValueError("CP prefill split length must match the extend interval")
        if split_spec.extend_len != int(extend_num_tokens):
            raise ValueError("CP prefill split length must match extend_num_tokens")
        split_spec.validate(cp_size=self.cp_size, page_size=self.page_size)

        if owner_rotations_cpu is not None:
            if owner_rotations_cpu.numel() != 1:
                raise ValueError("one CP prefill split spec requires one rotation")
            owner_rotation = int(owner_rotations_cpu.reshape(-1).tolist()[0])
            if owner_rotation != split_spec.owner_rotation:
                raise ValueError("CP prefill split rotation does not match request")

        last_loc_device = last_loc.to(
            device=self.device, dtype=torch.int64
        ).reshape(-1)
        if last_loc_device.numel() != 1:
            raise ValueError("one CP prefill split spec requires one last_loc")
        if split_spec.extend_len and prefix_len % self.page_size:
            leading_owner = split_spec.blocks[0].owner_rank
            leading_logical_page = torch.div(
                last_loc_device,
                self.page_size,
                rounding_mode="floor",
            )
            recorded_owner = self.residency_ledger.owner_ranks_for_logical_units(
                leading_logical_page
            )
            last_offset_cpu, recorded_owner_cpu = (
                torch.cat((last_loc_device % self.page_size, recorded_owner))
                .detach()
                .cpu()
                .tolist()
            )
            if last_offset_cpu != (prefix_len - 1) % self.page_size:
                raise ValueError(
                    "CP prefill split last_loc offset does not match prefix length"
                )
            if recorded_owner_cpu != leading_owner:
                raise ValueError(
                    "CP prefill split leading page owner does not match residency"
                )

        first_new_page = (
            (prefix_len + self.page_size - 1) // self.page_size
        ) * self.page_size
        new_page_starts = torch.arange(
            first_new_page,
            seq_len,
            self.page_size,
            dtype=torch.int64,
        )
        if new_page_starts.numel() > 0:
            block_ends = torch.tensor(
                [
                    block.logical_start + block.token_count
                    for block in split_spec.blocks
                ],
                dtype=torch.int64,
            )
            block_owners = torch.tensor(
                [block.owner_rank for block in split_spec.blocks],
                dtype=torch.int64,
            )
            block_indices = torch.bucketize(
                new_page_starts, block_ends, right=True
            )
            page_owners_cpu = block_owners[block_indices]
        else:
            page_owners_cpu = torch.empty((0,), dtype=torch.int64)

        page_counts = tuple(
            int(count)
            for count in torch.bincount(
                page_owners_cpu, minlength=self.cp_size
            ).tolist()
        )
        if page_counts != split_spec.page_demand(self.page_size):
            raise ValueError("CP prefill split page demand does not match its blocks")

        logical_pages = self._alloc_logical_pages(new_page_starts.numel())
        if logical_pages is None:
            return None

        owned_page_indices_cpu = torch.nonzero(
            page_owners_cpu == self.cp_rank, as_tuple=False
        ).reshape(-1)
        page_base_slots = self._alloc_page_count(owned_page_indices_cpu.numel())
        if page_base_slots is None:
            self._rollback_logical_pages(logical_pages)
            return None

        owned_page_indices = owned_page_indices_cpu.to(device=self.device)
        owned_logical_pages = logical_pages[owned_page_indices]
        owned_physical_pages = page_base_slots.to(
            dtype=self.logical_to_physical_page.dtype
        ) // self.page_size
        if owned_logical_pages.numel() > 0:
            self.logical_to_physical_page[owned_logical_pages] = (
                owned_physical_pages
            )
        self._assign_residency(logical_pages, page_owners_cpu)

        positions = torch.arange(
            prefix_len, seq_len, dtype=torch.int64, device=device
        )
        page_starts = torch.div(
            positions, self.page_size, rounding_mode="floor"
        ) * self.page_size
        page_ord = torch.div(
            torch.clamp(page_starts - first_new_page, min=0),
            self.page_size,
            rounding_mode="floor",
        )
        if logical_pages.numel() > 0:
            new_logical_slots = (
                logical_pages.to(device=device)[page_ord] * self.page_size
                + positions
                - page_starts
            )
        else:
            new_logical_slots = torch.zeros_like(positions)
        logical_slots = torch.where(
            positions < first_new_page,
            last_loc_device.to(device=device) + positions - prefix_len + 1,
            new_logical_slots,
        )

        owner_mask_parts = [
            torch.full(
                (block.token_count,),
                block.owner_rank == self.cp_rank,
                dtype=torch.bool,
                device=device,
            )
            for block in split_spec.blocks
        ]
        owner_mask = (
            torch.cat(owner_mask_parts)
            if owner_mask_parts
            else torch.empty((0,), dtype=torch.bool, device=device)
        )
        logical_page_ids = torch.div(
            logical_slots, self.page_size, rounding_mode="floor"
        )
        physical_pages = self.logical_to_physical_page[
            logical_page_ids.to(device=self.device)
        ].to(device=device)
        physical_slots = physical_pages * self.page_size + (
            logical_slots % self.page_size
        )
        physical_out = torch.where(
            owner_mask,
            physical_slots,
            torch.full_like(physical_slots, DUMMY_SLOT),
        )
        return CPShardedKVAllocation(
            logical_slots=logical_slots,
            physical_write_slots=physical_out,
        )

    def _alloc_decode_paged(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        *,
        device: torch.device | str,
        owner_rotations_cpu: Optional[torch.Tensor] = None,
    ) -> Optional[CPShardedKVAllocation]:
        if owner_rotations_cpu is None:
            owner_rotations_list = [0 for _ in seq_lens_cpu.tolist()]
        else:
            owner_rotations_list = [int(x) for x in owner_rotations_cpu.tolist()]

        new_page_req_indices: list[int] = []
        new_page_owners: list[int] = []
        owned_new_page_req_indices: list[int] = []
        for req_idx, (seq_len, owner_rotation) in enumerate(
            zip(seq_lens_cpu.tolist(), owner_rotations_list)
        ):
            pos = seq_len - 1
            if pos < 0 or pos % self.page_size != 0:
                continue
            page_start = (pos // self.page_size) * self.page_size
            new_page_req_indices.append(req_idx)
            owner = self._owner_for_position(page_start, owner_rotation)
            new_page_owners.append(owner)
            if owner == self.cp_rank:
                owned_new_page_req_indices.append(req_idx)

        logical_pages = self._alloc_logical_pages(len(new_page_req_indices))
        if logical_pages is None:
            return None
        local_slots = self._alloc_page_slots(len(owned_new_page_req_indices))
        if local_slots is None:
            self.logical_free_pages = torch.cat(
                (self.logical_free_pages, logical_pages)
            )
            return None
        new_page_req_indices_tensor = torch.tensor(
            new_page_req_indices, dtype=torch.long, device=device
        )
        owned_page_ord = torch.tensor(
            [new_page_req_indices.index(req_idx) for req_idx in owned_new_page_req_indices],
            dtype=torch.long,
            device=self.device,
        )
        owned_logical_pages = logical_pages[owned_page_ord]
        owned_physical_pages = (
            local_slots.to(device=self.device)[:: self.page_size] // self.page_size
        )
        if owned_logical_pages.numel() > 0:
            self.logical_to_physical_page[owned_logical_pages] = (
                owned_physical_pages
            )
        self._assign_residency(logical_pages, new_page_owners)

        seq_lens = seq_lens.to(dtype=torch.int64, device=device)
        last_loc = last_loc.to(dtype=torch.int64, device=device)
        pos = seq_lens - 1
        valid_mask = pos >= 0
        if owner_rotations_cpu is None:
            owner_rotations = torch.zeros_like(seq_lens, dtype=torch.int64)
        else:
            owner_rotations = owner_rotations_cpu.to(
                dtype=torch.int64, device=device
            )
        logical_page = torch.div(
            torch.clamp(pos, min=0), self.page_size, rounding_mode="floor"
        )
        page_start = logical_page * self.page_size
        owner_mask = valid_mask & (
            self._owner_for_position_tensor(page_start, owner_rotations)
            == self.cp_rank
        )
        logical_out = last_loc + 1
        logical_out[new_page_req_indices_tensor] = (
            logical_pages.to(device=device) * self.page_size
        )

        physical_out = torch.full_like(seq_lens, DUMMY_SLOT)
        logical_pages_for_out = torch.div(
            torch.clamp(logical_out, min=0), self.page_size, rounding_mode="floor"
        )
        physical_pages = self.logical_to_physical_page[
            logical_pages_for_out.to(device=self.device)
        ].to(device=device)
        physical_slots = physical_pages * self.page_size + (logical_out % self.page_size)
        physical_out = torch.where(owner_mask, physical_slots, physical_out)

        return CPShardedKVAllocation(
            logical_slots=logical_out,
            physical_write_slots=physical_out,
        )

    def _free_base_pages(self, page_indices: torch.Tensor):
        if page_indices.numel() == 0:
            return

        if self._has_paired_swa_allocator():
            page_offsets = torch.arange(
                self.page_size, dtype=torch.int64, device=page_indices.device
            )
            full_slots = (
                page_indices.to(dtype=torch.int64)[:, None] * self.page_size
                + page_offsets[None, :]
            ).reshape(-1)
            self.base_allocator.free(full_slots)
            return

        if self.base_allocator.need_sort:
            self.base_allocator.release_pages = torch.cat(
                (page_indices, self.base_allocator.release_pages)
            )
        else:
            self.base_allocator.free_pages = torch.cat(
                (page_indices, self.base_allocator.free_pages)
            )

    def _free_compacted(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.page_size == 1:
            self._free_logical_slots(free_index)
            return

        self._free_logical_pages_from_slots(free_index)

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self._free_compacted(free_index)
        else:
            self.free_group.append(free_index)

    def free_swa(self, free_index: torch.Tensor):
        physical_slots = self._logical_slots_to_physical_slots(free_index)
        if physical_slots.numel() > 0:
            self.base_allocator.free_swa(physical_slots)
        if self.swa_residency_ledger is None:
            return
        logical_units = filter_dummy_slots(
            free_index.reshape(-1).to(device=self.device, dtype=torch.int64)
        )
        if self.page_size > 1:
            logical_units = torch.div(
                logical_units, self.page_size, rounding_mode="floor"
            )
            logical_units = logical_units[logical_units != 0]
        if logical_units.numel() > 0:
            self.swa_residency_ledger.release(torch.unique(logical_units))
