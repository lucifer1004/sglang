import unittest

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cp_sharded_allocator import (
    DUMMY_SLOT,
    CPShardedKVPoolAllocator,
    build_extend_positions,
    filter_dummy_slots,
    get_cp_owner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


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

if __name__ == "__main__":
    unittest.main()
