import unittest

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.cp_sharded_allocator import (
    DUMMY_SLOT,
    CPShardedKVPoolAllocator,
    build_extend_positions,
    filter_dummy_slots,
    get_cp_owner,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _DummySWAKVPool(BaseSWAKVPool):
    def __init__(self):
        self.full_kv_pool = None
        self.swa_kv_pool = None
        self.full_to_swa_index_mapping = None
        self.invalidate_count = 0

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def invalidate_loc_cache(self) -> None:
        self.invalidate_count += 1

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        return self.full_to_swa_index_mapping[kv_indices]

    def set_swa_loc(self, loc: torch.Tensor) -> None:
        pass

    def get_state_buf_infos(self):
        return [], [], []

    def get_key_buffer(self, layer_id: int):
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int):
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int):
        raise NotImplementedError()

    def set_kv_buffer(self, layer, loc, cache_k, cache_v) -> None:
        raise NotImplementedError()


class TestCPShardedKVPoolAllocator(unittest.TestCase):
    def test_owner_uses_chunk_granularity(self):
        positions = torch.arange(0, 10, dtype=torch.int64)
        owner = get_cp_owner(positions, cp_size=2, cp_kv_chunk_size=4)

        self.assertEqual(owner.tolist(), [0, 0, 0, 0, 1, 1, 1, 1, 0, 0])

    def test_build_extend_positions(self):
        positions = build_extend_positions(
            torch.tensor([0, 3], dtype=torch.int64),
            torch.tensor([2, 3], dtype=torch.int64),
            device="cpu",
        )

        self.assertEqual(positions.tolist(), [0, 1, 3, 4, 5])

    def test_alloc_for_positions_uses_dummy_for_non_owner(self):
        base = TokenToKVPoolAllocator(
            size=10,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=1,
            cp_size=2,
            cp_kv_chunk_size=4,
        )

        slots = allocator.alloc_for_positions(torch.arange(0, 9, dtype=torch.int64))

        self.assertEqual(
            slots.tolist(),
            [
                DUMMY_SLOT,
                DUMMY_SLOT,
                DUMMY_SLOT,
                DUMMY_SLOT,
                1,
                2,
                3,
                4,
                DUMMY_SLOT,
            ],
        )
        self.assertEqual(base.available_size(), 6)

    def test_available_size_reports_logical_capacity(self):
        base = TokenToKVPoolAllocator(
            size=5,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=4,
            logical_size=9,
        )

        self.assertEqual(allocator.available_size(), 9)
        allocator.alloc_for_positions(torch.arange(0, 8, dtype=torch.int64))
        self.assertEqual(base.available_size(), 1)
        self.assertEqual(allocator.available_size(), 2)

    def test_local_alloc_size_for_range_uses_worst_cp_rank(self):
        base = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=4,
        )

        self.assertEqual(allocator.local_alloc_size_for_range(3, 6), 2)
        self.assertEqual(allocator.max_local_alloc_size_for_range(3, 6), 4)

    def test_paged_local_alloc_size_counts_owned_new_pages(self):
        base = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
        )

        self.assertEqual(allocator.local_alloc_size_for_range(6, 10), 0)
        self.assertEqual(allocator.max_local_alloc_size_for_range(6, 10), 8)

    def test_free_filters_dummy_slot(self):
        base = TokenToKVPoolAllocator(
            size=10,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=4,
        )

        slots = allocator.alloc_for_positions(torch.arange(0, 4, dtype=torch.int64))
        allocator.free(torch.tensor([DUMMY_SLOT, int(slots[0]), DUMMY_SLOT]))

        self.assertEqual(base.available_size(), 7)
        self.assertEqual(
            filter_dummy_slots(torch.tensor([0, 3, 0, 4], dtype=torch.int64)).tolist(),
            [3, 4],
        )

    def test_paged_alloc_extend_allocates_per_request_owned_pages(self):
        base = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=1,
            cp_size=2,
            cp_kv_chunk_size=8,
        )

        slots = allocator.alloc_extend(
            prefix_lens=torch.tensor([8, 8], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([8, 8], dtype=torch.int64),
            seq_lens=torch.tensor([10, 10], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 10], dtype=torch.int64),
            last_loc=torch.tensor([DUMMY_SLOT, DUMMY_SLOT], dtype=torch.int64),
            extend_num_tokens=4,
        )

        self.assertEqual(slots.tolist(), [4, 5, 8, 9])
        self.assertEqual(base.available_size(), 56)

    def test_paged_alloc_extend_reuses_existing_owner_page(self):
        base = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=1,
            cp_size=2,
            cp_kv_chunk_size=8,
        )

        slots = allocator.alloc_extend(
            prefix_lens=torch.tensor([9], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([9], dtype=torch.int64),
            seq_lens=torch.tensor([11], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([11], dtype=torch.int64),
            last_loc=torch.tensor([4], dtype=torch.int64),
            extend_num_tokens=2,
        )

        self.assertEqual(slots.tolist(), [5, 6])
        self.assertEqual(base.available_size(), 64)

    def test_paged_alloc_decode_allocates_per_request_owned_pages(self):
        base = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
        )

        slots = allocator.alloc_decode(
            seq_lens=torch.tensor([1, 1], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int64),
            last_loc=torch.tensor([DUMMY_SLOT, DUMMY_SLOT], dtype=torch.int64),
        )

        self.assertEqual(slots.tolist(), [4, 8])
        self.assertEqual(base.available_size(), 56)

    def test_paged_free_group_frees_ordered_pages_with_dummy_slots(self):
        base = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
        )

        slots_a = allocator.alloc_for_positions(torch.arange(0, 20, dtype=torch.int64))
        slots_b = allocator.alloc_for_positions(torch.arange(0, 12, dtype=torch.int64))
        self.assertEqual(base.available_size(), 44)

        allocator.free_group_begin()
        allocator.free(slots_a)
        allocator.free(slots_b)
        allocator.free_group_end()

        self.assertEqual(base.available_size(), 64)

    def test_paged_swa_allocates_and_frees_paired_pages(self):
        kvcache = _DummySWAKVPool()
        base = SWATokenToKVPoolAllocator(
            size=64,
            size_swa=64,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
        )

        slots = allocator.alloc_for_positions(torch.arange(0, 4, dtype=torch.int64))

        self.assertEqual(slots.tolist(), [4, 5, 6, 7])
        self.assertEqual(base.full_attn_allocator.available_size(), 60)
        self.assertEqual(base.swa_attn_allocator.available_size(), 60)
        self.assertEqual(
            kvcache.full_to_swa_index_mapping[slots].tolist(), [4, 5, 6, 7]
        )

        allocator.free(slots)

        self.assertEqual(base.full_attn_allocator.available_size(), 64)
        self.assertEqual(base.swa_attn_allocator.available_size(), 64)
        self.assertEqual(kvcache.full_to_swa_index_mapping[slots].tolist(), [0] * 4)

    def test_paged_alloc_rejects_unaligned_chunk_size(self):
        base = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )

        with self.assertRaisesRegex(ValueError, "divisible by page_size"):
            CPShardedKVPoolAllocator(
                base,
                cp_rank=0,
                cp_size=2,
                cp_kv_chunk_size=6,
            )

if __name__ == "__main__":
    unittest.main()
