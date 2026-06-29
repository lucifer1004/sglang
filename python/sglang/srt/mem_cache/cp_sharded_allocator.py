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
        self.is_not_in_free_group = True
        self.free_group: list[torch.Tensor] = []
        if self.page_size > 1 and self.cp_kv_chunk_size % self.page_size != 0:
            raise ValueError(
                "cp_kv_chunk_size must be divisible by page_size for CP "
                "sharded KV with paged allocation"
            )

    def __getattr__(self, name: str):
        return getattr(self.base_allocator, name)

    def debug_print(self) -> str:
        return self.base_allocator.debug_print()

    def _logical_available(self, physical_available: int, logical_size: int) -> int:
        return min(int(logical_size), int(physical_available) * self.cp_size)

    def available_size(self):
        return self._logical_available(self.base_allocator.available_size(), self.size)

    def physical_available_size(self) -> int:
        return int(self.base_allocator.available_size())

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
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            if self.page_size == 1:
                self._free_compacted(torch.cat(self.free_group))
            else:
                page_indices = [
                    self._ordered_page_indices_from_slots(free_index)
                    for free_index in self.free_group
                ]
                page_indices = [x for x in page_indices if x.numel() > 0]
                if page_indices:
                    self._free_base_pages(torch.cat(page_indices))
            self.free_group = []

    def clear(self):
        self.is_not_in_free_group = True
        self.free_group = []
        self.base_allocator.clear()

    def alloc(self, need_size: int):
        return self.base_allocator.alloc(need_size)

    def _owns_page_start(self, page_start: int) -> bool:
        return (int(page_start) // self.cp_kv_chunk_size) % self.cp_size == self.cp_rank

    def _range_local_alloc_size_for_rank(
        self, start: int, length: int, cp_rank: int
    ) -> int:
        start = int(start)
        length = int(length)
        if length <= 0:
            return 0
        end = start + length
        if self.page_size == 1:
            first_chunk = start // self.cp_kv_chunk_size
            last_chunk = (end - 1) // self.cp_kv_chunk_size
            count = 0
            for chunk_idx in range(first_chunk, last_chunk + 1):
                if chunk_idx % self.cp_size != cp_rank:
                    continue
                chunk_start = chunk_idx * self.cp_kv_chunk_size
                chunk_end = chunk_start + self.cp_kv_chunk_size
                count += max(0, min(end, chunk_end) - max(start, chunk_start))
            return count

        first_new_page = ((start + self.page_size - 1) // self.page_size) * self.page_size
        if first_new_page >= end:
            return 0
        count = 0
        for page_start in range(first_new_page, end, self.page_size):
            owner = (page_start // self.cp_kv_chunk_size) % self.cp_size
            if owner == cp_rank:
                count += self.page_size
        return count

    def max_local_alloc_size_for_range(self, start: int, length: int) -> int:
        return max(
            self._range_local_alloc_size_for_rank(start, length, rank)
            for rank in range(self.cp_size)
        )

    def local_alloc_size_for_range(self, start: int, length: int) -> int:
        return self._range_local_alloc_size_for_rank(start, length, self.cp_rank)

    def _alloc_page_starts(self, page_starts: list[int]) -> Optional[dict[int, int]]:
        """Allocate one physical page for each logical page start."""
        if not page_starts:
            return {}

        page_base_slots = self._alloc_page_count(len(page_starts))
        if page_base_slots is None:
            return None
        return {
            int(page_start): int(slot)
            for page_start, slot in zip(page_starts, page_base_slots)
        }

    def _alloc_page_count(self, count: int) -> Optional[list[int]]:
        if count <= 0:
            return []
        local_slots = self._alloc_page_slots(count)
        if local_slots is None:
            return None
        return [int(slot) for slot in local_slots[:: self.page_size].to("cpu").tolist()]

    def _merge_released_pages_without_sort(self):
        release_pages = self.base_allocator.release_pages
        if release_pages is None or len(release_pages) == 0:
            return
        self.base_allocator.free_pages = torch.cat(
            (self.base_allocator.free_pages, release_pages)
        )
        self.base_allocator.release_pages = torch.empty(
            (0,), dtype=release_pages.dtype, device=release_pages.device
        )

    def _alloc_page_slots(self, count: int) -> Optional[torch.Tensor]:
        if count <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        if (
            self.base_allocator.need_sort
            and int(count) > len(self.base_allocator.free_pages)
        ):
            # CP sharded residency does not require globally sorted physical page
            # reuse. Sorting the release list on the decode hot path stalls high
            # concurrency PD decode, so merge released pages directly.
            self._merge_released_pages_without_sort()
        return self.base_allocator.alloc(int(count) * self.page_size)

    def _ordered_owned_page_starts(self, positions: torch.Tensor) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for pos in positions.to("cpu").tolist():
            page_start = (int(pos) // self.page_size) * self.page_size
            if page_start in seen or not self._owns_page_start(page_start):
                continue
            seen.add(page_start)
            out.append(page_start)
        return out

    def alloc_for_positions(
        self,
        positions: torch.Tensor,
        *,
        positions_cpu: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        positions = positions.to(dtype=torch.int64)
        owner_positions = positions
        if positions_cpu is not None:
            if tuple(positions_cpu.shape) != tuple(positions.shape):
                raise ValueError(
                    "positions_cpu must have the same shape as positions, "
                    f"got {tuple(positions_cpu.shape)} vs {tuple(positions.shape)}"
                )
            owner_positions = positions_cpu.to(dtype=torch.int64)
        owner_mask = (
            get_cp_owner(owner_positions, self.cp_size, self.cp_kv_chunk_size)
            == self.cp_rank
        )
        if self.page_size > 1:
            page_owner_positions = (
                torch.div(owner_positions, self.page_size, rounding_mode="floor")
                * self.page_size
            )
            owner_mask = (
                get_cp_owner(page_owner_positions, self.cp_size, self.cp_kv_chunk_size)
                == self.cp_rank
            )
            page_starts = self._ordered_owned_page_starts(page_owner_positions)
            page_to_slot = self._alloc_page_starts(page_starts)
            if page_to_slot is None:
                return None

            out_slots = torch.full_like(positions, DUMMY_SLOT, dtype=torch.int64)
            flat_out = out_slots.view(-1)
            flat_owner_mask = owner_mask.reshape(-1)
            flat_positions_cpu = positions.to("cpu").reshape(-1).tolist()
            values = []
            indices = []
            for idx, pos in enumerate(flat_positions_cpu):
                if not bool(flat_owner_mask[idx].item()):
                    continue
                page_start = (int(pos) // self.page_size) * self.page_size
                indices.append(idx)
                values.append(page_to_slot[page_start] + (int(pos) - page_start))
            if indices:
                flat_out[
                    torch.tensor(indices, dtype=torch.long, device=positions.device)
                ] = torch.tensor(values, dtype=torch.int64, device=positions.device)
            return out_slots

        local_count = int(owner_mask.sum().item())
        if local_count == 0:
            return torch.full_like(positions, DUMMY_SLOT, dtype=torch.int64)

        local_slots = self.base_allocator.alloc(local_count)
        if local_slots is None:
            return None
        if local_count == positions.numel():
            return local_slots.to(device=positions.device).view_as(positions)

        out_slots = torch.full_like(positions, DUMMY_SLOT, dtype=torch.int64)
        owner_indices = owner_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        if owner_indices.device != out_slots.device:
            owner_indices = owner_indices.to(device=out_slots.device, non_blocking=True)
        out_slots.view(-1)[owner_indices] = local_slots.to(device=out_slots.device)
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
            return self._alloc_extend_paged(
                prefix_lens_cpu,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                device=prefix_lens.device,
            )
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
            return self._alloc_decode_paged(
                seq_lens, seq_lens_cpu, last_loc, device=seq_lens.device
            )
        return self.alloc_for_positions(seq_lens.to(dtype=torch.int64) - 1)

    def _alloc_extend_paged_single_empty(
        self,
        seq_len: int,
        extend_num_tokens: int,
        *,
        device: torch.device | str,
    ) -> Optional[torch.Tensor]:
        if seq_len != extend_num_tokens:
            raise RuntimeError(
                "CP sharded paged empty-prefix allocation expects seq_len to "
                f"match extend_num_tokens, got {seq_len} vs {extend_num_tokens}"
            )

        num_owned_pages = self._owned_page_count_for_length(seq_len)
        local_slots = self._alloc_page_slots(num_owned_pages)
        if local_slots is None:
            return None

        out = torch.full((seq_len,), DUMMY_SLOT, dtype=torch.int64, device=device)
        if seq_len == 0 or num_owned_pages == 0:
            return out

        positions = torch.arange(seq_len, dtype=torch.int64, device=device)
        pages_per_chunk = self.cp_kv_chunk_size // self.page_size
        logical_page = torch.div(positions, self.page_size, rounding_mode="floor")
        chunk_idx = torch.div(logical_page, pages_per_chunk, rounding_mode="floor")
        page_in_chunk = logical_page - chunk_idx * pages_per_chunk
        owner_mask = chunk_idx % self.cp_size == self.cp_rank
        local_page_ord = (
            torch.div(chunk_idx, self.cp_size, rounding_mode="floor") * pages_per_chunk
            + page_in_chunk
        )
        page_offset = positions - logical_page * self.page_size
        page_slots = local_slots.view(num_owned_pages, self.page_size)
        safe_page_ord = torch.clamp(local_page_ord, max=num_owned_pages - 1)
        owned_slots = page_slots[safe_page_ord, page_offset]
        return torch.where(owner_mask, owned_slots, out)

    def _alloc_extend_paged(
        self,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        *,
        device: torch.device | str,
    ) -> Optional[torch.Tensor]:
        prefix_lens_list = [int(x) for x in prefix_lens_cpu.tolist()]
        seq_lens_list = [int(x) for x in seq_lens_cpu.tolist()]

        if len(prefix_lens_list) == 1 and prefix_lens_list[0] == 0:
            return self._alloc_extend_paged_single_empty(
                seq_lens_list[0], extend_num_tokens, device=device
            )

        last_loc_list = [int(x) for x in last_loc.to("cpu").tolist()]

        page_keys: list[tuple[int, int]] = []
        for req_idx, (prefix_len, seq_len) in enumerate(
            zip(prefix_lens_list, seq_lens_list)
        ):
            first_new_page = (
                (prefix_len + self.page_size - 1) // self.page_size
            ) * self.page_size
            for page_start in range(first_new_page, seq_len, self.page_size):
                if not self._owns_page_start(page_start):
                    continue
                page_keys.append((req_idx, page_start))

        page_base_slots = self._alloc_page_count(len(page_keys))
        if page_base_slots is None:
            return None
        page_to_slot = {
            page_key: page_base_slots[idx] for idx, page_key in enumerate(page_keys)
        }

        out = torch.full(
            (extend_num_tokens,), DUMMY_SLOT, dtype=torch.int64, device=device
        )
        values: list[int] = []
        write_offsets: list[int] = []
        out_offset = 0
        for req_idx, (prefix_len, seq_len, prev_loc) in enumerate(
            zip(prefix_lens_list, seq_lens_list, last_loc_list)
        ):
            first_new_page = (
                (prefix_len + self.page_size - 1) // self.page_size
            ) * self.page_size
            for pos in range(prefix_len, seq_len):
                page_start = (pos // self.page_size) * self.page_size
                if self._owns_page_start(page_start):
                    if pos < first_new_page:
                        slot = prev_loc + (pos - prefix_len + 1)
                    else:
                        slot = page_to_slot[(req_idx, page_start)] + (
                            pos - page_start
                        )
                    write_offsets.append(out_offset)
                    values.append(slot)
                out_offset += 1

        if values:
            out[torch.tensor(write_offsets, dtype=torch.long, device=device)] = (
                torch.tensor(values, dtype=torch.int64, device=device)
            )
        return out

    def _alloc_decode_paged(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> Optional[torch.Tensor]:
        num_new_pages = 0
        for seq_len in seq_lens_cpu.tolist():
            pos = seq_len - 1
            if pos < 0 or pos % self.page_size != 0:
                continue
            page_start = (pos // self.page_size) * self.page_size
            if self._owns_page_start(page_start):
                num_new_pages += 1

        local_slots = self._alloc_page_slots(num_new_pages)
        if local_slots is None:
            return None

        seq_lens = seq_lens.to(dtype=torch.int64, device=device)
        last_loc = last_loc.to(dtype=torch.int64, device=device)
        pos = seq_lens - 1
        valid_mask = pos >= 0
        logical_page = torch.div(
            torch.clamp(pos, min=0), self.page_size, rounding_mode="floor"
        )
        page_start = logical_page * self.page_size
        owner_mask = valid_mask & (
            get_cp_owner(page_start, self.cp_size, self.cp_kv_chunk_size) == self.cp_rank
        )
        new_page_mask = owner_mask & (pos % self.page_size == 0)
        out = torch.full_like(seq_lens, DUMMY_SLOT)

        if num_new_pages > 0:
            page_slots = local_slots.view(num_new_pages, self.page_size)[:, 0]
            new_page_ord = torch.cumsum(new_page_mask.to(torch.int64), dim=0) - 1
            new_page_values = page_slots[torch.clamp(new_page_ord, min=0)]
            owner_slots = torch.where(new_page_mask, new_page_values, last_loc + 1)
        else:
            owner_slots = last_loc + 1

        return torch.where(owner_mask, owner_slots, out)

    def _free_base_pages(self, page_indices: torch.Tensor):
        if page_indices.numel() == 0:
            return

        if self.base_allocator.need_sort:
            self.base_allocator.release_pages = torch.cat(
                (page_indices, self.base_allocator.release_pages)
            )
        else:
            self.base_allocator.free_pages = torch.cat(
                (page_indices, self.base_allocator.free_pages)
            )

    def _owned_page_count_for_length(self, num_tokens: int) -> int:
        num_pages = (int(num_tokens) + self.page_size - 1) // self.page_size
        pages_per_chunk = self.cp_kv_chunk_size // self.page_size
        count = 0
        chunk_idx = self.cp_rank
        while chunk_idx * pages_per_chunk < num_pages:
            chunk_start = chunk_idx * pages_per_chunk
            count += min(pages_per_chunk, num_pages - chunk_start)
            chunk_idx += self.cp_size
        return count

    def _ordered_page_indices_from_slots(self, free_index: torch.Tensor) -> torch.Tensor:
        free_index = free_index.reshape(-1)
        if free_index.numel() == 0:
            return free_index

        num_owned_pages = self._owned_page_count_for_length(free_index.numel())
        if num_owned_pages == 0:
            return torch.empty((0,), dtype=torch.int64, device=free_index.device)

        # req_to_token stores slots in request-position order. For paged CP
        # sharding, page ownership is determined by the logical page start. Build
        # the owned page offsets directly on the target device from the CP chunk
        # formula so release does not run torch.unique, boolean filtering, or
        # CPU->GPU offset copies on the hot path.
        pages_per_chunk = self.cp_kv_chunk_size // self.page_size
        local_page_ord = torch.arange(
            num_owned_pages, dtype=torch.int64, device=free_index.device
        )
        owned_chunk_ord = torch.div(
            local_page_ord, pages_per_chunk, rounding_mode="floor"
        )
        page_in_chunk = local_page_ord - owned_chunk_ord * pages_per_chunk
        logical_page = (
            self.cp_rank + owned_chunk_ord * self.cp_size
        ) * pages_per_chunk + page_in_chunk
        return free_index[logical_page * self.page_size] // self.page_size

    def _free_compacted(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.page_size == 1:
            self.base_allocator.free(filter_dummy_slots(free_index))
            return

        self._free_base_pages(self._ordered_page_indices_from_slots(free_index))

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self._free_compacted(free_index)
        else:
            self.free_group.append(free_index)

    def free_swa(self, free_index: torch.Tensor):
        self.base_allocator.free_swa(filter_dummy_slots(free_index))
