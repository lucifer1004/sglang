import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.context_parallel import build_cp_prefill_split_spec
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.common import alloc_for_extend
from sglang.srt.mem_cache.cp_sharded_allocator import (
    DUMMY_SLOT,
    CPShardedKVAllocation,
    CPShardedKVPoolAllocator,
    build_extend_positions,
    filter_dummy_slots,
    get_cp_owner,
)
from sglang.srt.mem_cache.cp_sharded_capacity import (
    ensure_cp_sharded_kv_capacity,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
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
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

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


class _EvictingTree:
    def __init__(self, allocator, victims):
        self.allocator = allocator
        self.victims = list(victims)
        self.evict_calls = []

    def evict(self, params: EvictParams) -> EvictResult:
        self.evict_calls.append(params.num_tokens)
        if not self.victims:
            return EvictResult()
        victim = self.victims.pop(0)
        self.allocator.free(victim)
        return EvictResult(num_tokens_evicted=len(victim))


class _BatchEvictingTree:
    def __init__(self, allocator, victims):
        self.allocator = allocator
        self.victims = list(victims)

    def evict(self, _params: EvictParams) -> EvictResult:
        num_tokens_evicted = 0
        for victim in self.victims:
            self.allocator.free(victim)
            num_tokens_evicted += len(victim)
        self.victims = []
        return EvictResult(num_tokens_evicted=num_tokens_evicted)


class TestCPShardedKVPoolAllocator(unittest.TestCase):
    def _make_allocator_for_mapping_lifecycle(self, page_size: int):
        if page_size == 1:
            base = TokenToKVPoolAllocator(
                size=32,
                dtype=torch.float32,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )
        else:
            base = PagedTokenToKVPoolAllocator(
                size=32,
                page_size=page_size,
                dtype=torch.float32,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )
        return CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
            logical_size=64,
        )

    def test_clear_preserves_graph_visible_mapping_storage(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                allocator = self._make_allocator_for_mapping_lifecycle(page_size)
                allocation = allocator.alloc_for_positions_with_logical(
                    torch.arange(0, 8, dtype=torch.int64)
                )
                self.assertIsNotNone(allocation)
                mapping_name = (
                    "logical_to_physical_slot"
                    if page_size == 1
                    else "logical_to_physical_page"
                )
                captured_mapping = getattr(allocator, mapping_name)
                self.assertGreater(torch.count_nonzero(captured_mapping).item(), 0)

                allocator.clear()

                self.assertIs(captured_mapping, getattr(allocator, mapping_name))
                self.assertEqual(torch.count_nonzero(captured_mapping).item(), 0)

    def test_restore_state_preserves_graph_visible_mapping_storage(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                allocator = self._make_allocator_for_mapping_lifecycle(page_size)
                allocation = allocator.alloc_for_positions_with_logical(
                    torch.arange(0, 8, dtype=torch.int64)
                )
                self.assertIsNotNone(allocation)
                mapping_name = (
                    "logical_to_physical_slot"
                    if page_size == 1
                    else "logical_to_physical_page"
                )
                captured_mapping = getattr(allocator, mapping_name)
                expected_mapping = captured_mapping.clone()
                state = allocator.backup_state()

                captured_mapping.zero_()
                allocator.restore_state(state)

                self.assertIs(captured_mapping, getattr(allocator, mapping_name))
                self.assertTrue(torch.equal(captured_mapping, expected_mapping))

    def test_owner_uses_chunk_granularity(self):
        positions = torch.arange(0, 10, dtype=torch.int64)
        owner = get_cp_owner(positions, cp_size=2, cp_kv_chunk_size=4)

        self.assertEqual(owner.tolist(), [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    def test_owner_uses_balanced_zigzag_with_rank_rotation(self):
        for cp_size in (2, 4, 8):
            positions = torch.arange(2 * cp_size, dtype=torch.int64)
            base = list(range(cp_size)) + list(reversed(range(cp_size)))
            for rotation in range(cp_size):
                with self.subTest(cp_size=cp_size, rotation=rotation):
                    owner = get_cp_owner(
                        positions,
                        cp_size=cp_size,
                        cp_kv_chunk_size=1,
                        owner_rotation=rotation,
                    )
                    expected = [
                        (rank + rotation) % cp_size for rank in base
                    ]
                    self.assertEqual(owner.tolist(), expected)
                    self.assertEqual(
                        torch.bincount(owner, minlength=cp_size).tolist(),
                        [2] * cp_size,
                    )

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

        slots = allocator.alloc_for_positions(torch.arange(0, 8, dtype=torch.int64))

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
        self.assertEqual(allocator.available_size(), 1)

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

        self.assertEqual(allocator.local_alloc_size_for_range(3, 6), 1)
        self.assertEqual(allocator.max_local_alloc_size_for_range(3, 6), 5)

    def test_alloc_size_vector_is_balanced_and_rotated(self):
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
            cp_size=4,
            cp_kv_chunk_size=4,
        )

        self.assertEqual(
            allocator.alloc_size_per_rank_for_range(0, 32, owner_rotation=0),
            (8, 8, 8, 8),
        )
        self.assertEqual(
            allocator.alloc_size_per_rank_for_range(0, 4, owner_rotation=2),
            (0, 0, 4, 0),
        )

    def test_decode_alloc_size_vector_uses_independent_round_robin(self):
        prefill_base = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        prefill_allocator = CPShardedKVPoolAllocator(
            prefill_base,
            cp_rank=0,
            cp_size=4,
            cp_kv_chunk_size=4,
        )
        decode_base = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        decode_allocator = CPShardedKVPoolAllocator(
            decode_base,
            cp_rank=0,
            cp_size=4,
            cp_kv_chunk_size=4,
            use_decode_owner_layout=True,
        )

        self.assertEqual(
            prefill_allocator.alloc_size_per_rank_for_range(16, 8),
            (0, 0, 4, 4),
        )
        self.assertEqual(
            decode_allocator.alloc_size_per_rank_for_range(16, 8),
            (4, 4, 0, 0),
        )

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
        self.assertEqual(
            allocator.alloc_size_per_rank_for_range(6, 2),
            (0, 0),
        )

    def test_paged_cp_allocation_prefers_contiguous_physical_pages(self):
        base = PagedTokenToKVPoolAllocator(
            size=32,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        base.free_pages = torch.tensor([1, 3, 4, 7], dtype=torch.int64)
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=1,
            cp_kv_chunk_size=8,
        )

        slots = allocator._alloc_page_slots(2)

        self.assertEqual(slots.tolist(), list(range(12, 20)))
        self.assertEqual(base.free_pages.tolist(), [1, 7])

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

        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 4, dtype=torch.int64)
        )
        allocator.free(
            torch.tensor(
                [DUMMY_SLOT, int(allocation.logical_slots[0]), DUMMY_SLOT]
            )
        )

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

        allocation = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([8, 8], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([8, 8], dtype=torch.int64),
            seq_lens=torch.tensor([10, 10], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 10], dtype=torch.int64),
            last_loc=torch.tensor([DUMMY_SLOT, DUMMY_SLOT], dtype=torch.int64),
            extend_num_tokens=4,
        )

        self.assertEqual(allocation.physical_write_slots.tolist(), [4, 5, 8, 9])
        self.assertEqual(allocation.logical_slots.tolist(), [4, 5, 8, 9])
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

        prefix = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([9], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([9], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=9,
        )
        allocation = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([9], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([9], dtype=torch.int64),
            seq_lens=torch.tensor([11], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([11], dtype=torch.int64),
            last_loc=prefix.logical_slots[-1:],
            extend_num_tokens=2,
        )

        self.assertEqual(allocation.physical_write_slots.tolist(), [5, 6])
        self.assertEqual(allocation.logical_slots.tolist(), [13, 14])
        self.assertEqual(base.available_size(), 60)

    def test_paged_alloc_extend_uses_explicit_split_owners(self):
        base = PagedTokenToKVPoolAllocator(
            size=32,
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
            logical_size=64,
        )
        split_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=8,
            cp_size=2,
            page_size=4,
            owner_rotation=1,
        )

        # The legacy chunk formula assigns both pages to rank 1. The explicit
        # plan assigns page 0 to rank 1 and page 1 to rank 0.
        self.assertEqual(
            allocator.alloc_size_per_rank_for_range(0, 8, owner_rotation=1),
            (0, 8),
        )
        allocation = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=8,
            owner_rotations_cpu=torch.tensor([1], dtype=torch.int64),
            split_spec=split_spec,
        )

        self.assertEqual(allocation.logical_slots.tolist(), list(range(4, 12)))
        self.assertEqual(
            allocation.physical_write_slots.tolist(),
            [DUMMY_SLOT] * 4 + [4, 5, 6, 7],
        )
        self.assertEqual(base.available_size(), 28)
        self.assertEqual(
            allocator.physical_load_snapshot().allocated_tokens,
            (4, 4),
        )

    def test_explicit_split_reuses_recorded_leading_partial_page_owner(self):
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
            logical_size=128,
        )
        prefix_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=6,
            cp_size=2,
            page_size=4,
            owner_rotation=0,
        )
        prefix = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([6], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([6], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=6,
            split_spec=prefix_spec,
        )
        available_before = base.available_size()
        continuation_spec = build_cp_prefill_split_spec(
            extend_start=6,
            extend_len=2,
            cp_size=2,
            page_size=4,
            owner_rotation=1,
            leading_page_owner=1,
        )

        continuation = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([6], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([6], dtype=torch.int64),
            seq_lens=torch.tensor([8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
            last_loc=prefix.logical_slots[-1:],
            extend_num_tokens=2,
            split_spec=continuation_spec,
        )

        self.assertEqual(continuation.logical_slots.tolist(), [10, 11])
        self.assertEqual(continuation.physical_write_slots.tolist(), [6, 7])
        self.assertEqual(base.available_size(), available_before)
        self.assertEqual(continuation_spec.page_demand(4), (0, 0))

    def test_explicit_split_rejects_wrong_leading_page_owner_before_mutation(self):
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
            logical_size=128,
        )
        prefix_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=6,
            cp_size=2,
            page_size=4,
            owner_rotation=0,
        )
        prefix = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([6], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([6], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=6,
            split_spec=prefix_spec,
        )
        invalid_spec = build_cp_prefill_split_spec(
            extend_start=6,
            extend_len=2,
            cp_size=2,
            page_size=4,
            owner_rotation=0,
            leading_page_owner=0,
        )
        logical_free_before = allocator.logical_free_pages.clone()
        load_before = allocator.physical_load_snapshot()
        available_before = base.available_size()

        with self.assertRaisesRegex(ValueError, "leading page owner"):
            allocator.alloc_extend_with_logical(
                prefix_lens=torch.tensor([6], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([6], dtype=torch.int64),
                seq_lens=torch.tensor([8], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
                last_loc=prefix.logical_slots[-1:],
                extend_num_tokens=2,
                split_spec=invalid_spec,
            )

        self.assertTrue(torch.equal(allocator.logical_free_pages, logical_free_before))
        self.assertEqual(allocator.physical_load_snapshot(), load_before)
        self.assertEqual(base.available_size(), available_before)

    def test_explicit_split_rejects_wrong_leading_last_loc_offset(self):
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
            logical_size=128,
        )
        prefix_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=6,
            cp_size=2,
            page_size=4,
            owner_rotation=0,
        )
        prefix = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([6], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([6], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=6,
            split_spec=prefix_spec,
        )
        continuation_spec = build_cp_prefill_split_spec(
            extend_start=6,
            extend_len=2,
            cp_size=2,
            page_size=4,
            owner_rotation=0,
            leading_page_owner=1,
        )
        logical_free_before = allocator.logical_free_pages.clone()
        load_before = allocator.physical_load_snapshot()

        with self.assertRaisesRegex(ValueError, "last_loc"):
            allocator.alloc_extend_with_logical(
                prefix_lens=torch.tensor([6], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([6], dtype=torch.int64),
                seq_lens=torch.tensor([8], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
                last_loc=prefix.logical_slots[-2:-1],
                extend_num_tokens=2,
                split_spec=continuation_spec,
            )

        self.assertTrue(torch.equal(allocator.logical_free_pages, logical_free_before))
        self.assertEqual(allocator.physical_load_snapshot(), load_before)

    def test_explicit_split_owner_plan_is_rank_packed_and_counted(self):
        base = PagedTokenToKVPoolAllocator(
            size=32,
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
            logical_size=64,
        )
        split_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=12,
            cp_size=2,
            page_size=4,
            owner_rotation=1,
        )
        allocation = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([12], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([12], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=12,
            split_spec=split_spec,
        )
        logical_pages = torch.unique(
            allocation.logical_slots // 4, sorted=True
        )

        owner_plan = allocator.owner_plan_for_logical_pages(logical_pages)

        self.assertEqual(owner_plan.owner_ranks.tolist(), [1, 0, 0])
        self.assertEqual(owner_plan.per_rank_counts, (2, 1))
        self.assertEqual(owner_plan.rank_packed_to_logical.tolist(), [1, 2, 0])

    def test_attntp_peers_keep_explicit_fragmented_physical_layout(self):
        allocators = []
        for _attn_tp_rank in range(2):
            base = PagedTokenToKVPoolAllocator(
                size=32,
                page_size=4,
                dtype=torch.float32,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )
            base.free_pages = torch.tensor([1, 3, 5, 7], dtype=torch.int64)
            allocators.append(
                CPShardedKVPoolAllocator(
                    base,
                    cp_rank=0,
                    cp_size=2,
                    cp_kv_chunk_size=8,
                    logical_size=64,
                )
            )
        split_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=16,
            cp_size=2,
            page_size=4,
            owner_rotation=0,
        )

        allocations = [
            allocator.alloc_extend_with_logical(
                prefix_lens=torch.tensor([0], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([16], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([16], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64),
                extend_num_tokens=16,
                split_spec=split_spec,
            )
            for allocator in allocators
        ]

        self.assertEqual(
            allocations[0].logical_slots.tolist(),
            allocations[1].logical_slots.tolist(),
        )
        self.assertEqual(
            allocations[0].physical_write_slots.tolist(),
            allocations[1].physical_write_slots.tolist(),
        )
        logical_pages = torch.unique(allocations[0].logical_slots // 4, sorted=True)
        self.assertEqual(
            allocators[0].logical_page_table_to_physical(logical_pages).tolist(),
            allocators[1].logical_page_table_to_physical(logical_pages).tolist(),
        )

    def test_explicit_split_oom_rolls_back_before_residency_mutation(self):
        base = PagedTokenToKVPoolAllocator(
            size=4,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        base.alloc(4)
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
            logical_size=16,
        )
        split_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=4,
            cp_size=2,
            page_size=4,
            owner_rotation=0,
        )
        logical_free_before = allocator.logical_free_pages.clone()
        page_map_before = allocator.logical_to_physical_page.clone()
        load_before = allocator.physical_load_snapshot()

        allocation = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([4], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=4,
            split_spec=split_spec,
        )

        self.assertIsNone(allocation)
        self.assertTrue(
            torch.equal(allocator.logical_free_pages, logical_free_before)
        )
        self.assertTrue(
            torch.equal(allocator.logical_to_physical_page, page_map_before)
        )
        self.assertEqual(allocator.physical_load_snapshot(), load_before)

    def test_explicit_split_grouped_free_restores_owner_load_without_collective(self):
        base = PagedTokenToKVPoolAllocator(
            size=32,
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
            logical_size=64,
        )
        split_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=8,
            cp_size=2,
            page_size=4,
            owner_rotation=1,
        )

        with patch.object(torch.distributed, "all_reduce") as all_reduce:
            allocation = allocator.alloc_extend_with_logical(
                prefix_lens=torch.tensor([0], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([8], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64),
                extend_num_tokens=8,
                split_spec=split_spec,
            )
            allocator.free_group_begin()
            allocator.free(allocation.logical_slots[:4])
            allocator.free(allocation.logical_slots[4:])
            allocator.free_group_end()

        self.assertEqual(
            allocator.physical_load_snapshot().allocated_tokens,
            (0, 0),
        )
        self.assertEqual(base.available_size(), 32)
        all_reduce.assert_not_called()

    def test_explicit_split_hot_path_does_not_reconstruct_owners_or_call_item(self):
        source = inspect.getsource(
            CPShardedKVPoolAllocator._alloc_extend_paged_explicit
        )

        self.assertNotIn(".item(", source)
        self.assertNotIn("get_cp_owner", source)
        self.assertNotIn("_owner_for_position", source)

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

        allocation = allocator.alloc_decode_with_logical(
            seq_lens=torch.tensor([1, 1], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int64),
            last_loc=torch.tensor([DUMMY_SLOT, DUMMY_SLOT], dtype=torch.int64),
        )

        self.assertEqual(allocation.physical_write_slots.tolist(), [4, 8])
        self.assertEqual(allocation.logical_slots.tolist(), [4, 8])
        self.assertEqual(base.available_size(), 56)

    def test_paged_decode_new_page_uses_round_robin_not_prefill_zigzag(self):
        physical_slots = []
        for cp_rank in range(2):
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
                cp_rank=cp_rank,
                cp_size=2,
                cp_kv_chunk_size=8,
                use_decode_owner_layout=True,
            )

            allocation = allocator.alloc_decode_with_logical(
                seq_lens=torch.tensor([17], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([17], dtype=torch.int64),
                last_loc=torch.tensor([DUMMY_SLOT], dtype=torch.int64),
            )
            physical_slots.append(int(allocation.physical_write_slots.item()))

        # Position 16 is in owner chunk 2. Decode round-robin maps chunk 2 to
        # rank 0, while Prefill zigzag maps it to rank 1.
        self.assertGreater(physical_slots[0], DUMMY_SLOT)
        self.assertEqual(physical_slots[1], DUMMY_SLOT)

    def test_default_decode_keeps_prefill_owner_rotation(self):
        physical_slots = []
        for cp_rank in range(2):
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
                cp_rank=cp_rank,
                cp_size=2,
                cp_kv_chunk_size=8,
            )

            allocation = allocator.alloc_decode_with_logical(
                seq_lens=torch.tensor([1], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([1], dtype=torch.int64),
                last_loc=torch.tensor([DUMMY_SLOT], dtype=torch.int64),
                owner_rotations_cpu=torch.tensor([1], dtype=torch.int64),
            )
            physical_slots.append(int(allocation.physical_write_slots.item()))

        self.assertEqual(physical_slots[0], DUMMY_SLOT)
        self.assertGreater(physical_slots[1], DUMMY_SLOT)

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

        allocation_a = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 20, dtype=torch.int64)
        )
        allocation_b = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 12, dtype=torch.int64)
        )
        self.assertEqual(base.available_size(), 48)

        allocator.free_group_begin()
        allocator.free(allocation_a.logical_slots)
        allocator.free(allocation_b.logical_slots)
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

        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 4, dtype=torch.int64)
        )
        slots = allocation.physical_write_slots

        self.assertEqual(slots.tolist(), [4, 5, 6, 7])
        self.assertEqual(base.full_attn_allocator.available_size(), 60)
        self.assertEqual(base.swa_attn_allocator.available_size(), 60)
        self.assertEqual(
            kvcache.full_to_swa_index_mapping[slots].tolist(), [4, 5, 6, 7]
        )

        allocator.free(allocation.logical_slots)

        self.assertEqual(base.full_attn_allocator.available_size(), 64)
        self.assertEqual(base.swa_attn_allocator.available_size(), 64)
        self.assertEqual(kvcache.full_to_swa_index_mapping[slots].tolist(), [0] * 4)

    def test_explicit_split_allocates_and_frees_paired_swa_pages(self):
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
            logical_size=128,
        )
        split_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=8,
            cp_size=2,
            page_size=4,
            owner_rotation=1,
        )

        allocation = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=8,
            split_spec=split_spec,
        )
        owned_slots = filter_dummy_slots(allocation.physical_write_slots)

        self.assertEqual(owned_slots.tolist(), [4, 5, 6, 7])
        self.assertEqual(base.full_attn_allocator.available_size(), 60)
        self.assertEqual(base.swa_attn_allocator.available_size(), 60)
        self.assertEqual(
            allocator.residency_ledger.snapshot().allocated_tokens,
            (4, 4),
        )
        self.assertEqual(
            allocator.swa_residency_ledger.snapshot().allocated_tokens,
            (4, 4),
        )
        self.assertEqual(
            kvcache.full_to_swa_index_mapping[owned_slots].tolist(),
            [4, 5, 6, 7],
        )

        allocator.free(allocation.logical_slots)

        self.assertEqual(base.full_attn_allocator.available_size(), 64)
        self.assertEqual(base.swa_attn_allocator.available_size(), 64)
        self.assertEqual(
            allocator.residency_ledger.snapshot().allocated_tokens,
            (0, 0),
        )
        self.assertEqual(
            allocator.swa_residency_ledger.snapshot().allocated_tokens,
            (0, 0),
        )

    def test_paged_swa_resolves_owner_local_transfer_indices(self):
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
            logical_size=128,
        )
        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 32, dtype=torch.int64)
        )
        logical_pages = allocation.logical_slots[::4] // 4

        self.assertEqual(
            allocator.logical_page_table_to_physical(logical_pages).tolist(),
            [1, 2, 0, 0, 0, 0, 3, 4],
        )
        self.assertEqual(
            allocator.logical_swa_page_table_to_physical(logical_pages).tolist(),
            [1, 2, 0, 0, 0, 0, 3, 4],
        )
        self.assertEqual(
            allocator.logical_slots_to_physical(
                allocation.logical_slots
            ).tolist(),
            allocation.physical_write_slots.tolist(),
        )

    def test_swa_reports_separate_full_and_window_pool_capacity(self):
        kvcache = _DummySWAKVPool()
        base = SWATokenToKVPoolAllocator(
            size=64,
            size_swa=48,
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
            logical_size=128,
            logical_full_size=128,
            logical_swa_size=96,
        )

        self.assertEqual(allocator.available_size(), 96)
        self.assertEqual(allocator.full_available_size(), 128)
        self.assertEqual(allocator.swa_available_size(), 96)

        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 4, dtype=torch.int64)
        )
        self.assertEqual(allocator.available_size(), 92)
        self.assertEqual(allocator.full_available_size(), 124)
        self.assertEqual(allocator.swa_available_size(), 92)

        allocator.free_swa(allocation.logical_slots)
        self.assertEqual(allocator.available_size(), 96)
        self.assertEqual(allocator.full_available_size(), 124)
        self.assertEqual(allocator.swa_available_size(), 96)

        allocator.free(allocation.logical_slots)
        self.assertEqual(allocator.full_available_size(), 128)
        self.assertEqual(allocator.swa_available_size(), 96)

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

    def test_logical_ids_are_identical_across_ranks(self):
        allocators = []
        for cp_rank in range(2):
            base = PagedTokenToKVPoolAllocator(
                size=32,
                page_size=4,
                dtype=torch.float32,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )
            allocators.append(
                CPShardedKVPoolAllocator(
                    base,
                    cp_rank=cp_rank,
                    cp_size=2,
                    cp_kv_chunk_size=4,
                    logical_size=64,
                )
            )

        allocations = [
            allocator.alloc_for_positions_with_logical(
                torch.arange(0, 8, dtype=torch.int64)
            )
            for allocator in allocators
        ]

        self.assertEqual(
            allocations[0].logical_slots.tolist(),
            allocations[1].logical_slots.tolist(),
        )
        self.assertEqual(
            allocations[0].physical_write_slots.tolist(), [4, 5, 6, 7, 0, 0, 0, 0]
        )
        self.assertEqual(
            allocations[1].physical_write_slots.tolist(), [0, 0, 0, 0, 4, 5, 6, 7]
        )
        self.assertEqual(
            allocators[0].physical_load_snapshot(),
            allocators[1].physical_load_snapshot(),
        )
        self.assertEqual(
            allocators[0].physical_load_snapshot().allocated_tokens,
            (4, 4),
        )

    def test_fragmented_physical_pages_do_not_change_logical_ids(self):
        base = PagedTokenToKVPoolAllocator(
            size=32,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        reserved = base.alloc(8)
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=4,
            logical_size=64,
        )

        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 4, dtype=torch.int64)
        )

        self.assertEqual(allocation.logical_slots.tolist(), [4, 5, 6, 7])
        self.assertEqual(allocation.physical_write_slots.tolist(), [12, 13, 14, 15])
        self.assertEqual(
            allocator.logical_page_table_to_physical(torch.tensor([[1]])).tolist(),
            [[3]],
        )
        allocator.free(allocation.logical_slots)
        self.assertEqual(base.available_size(), 24)
        base.free(reserved)
        self.assertEqual(base.available_size(), 32)

    def test_attention_tp_peers_keep_identical_fragmented_physical_layout(self):
        allocators = []
        for _attn_tp_rank in range(2):
            base = PagedTokenToKVPoolAllocator(
                size=48,
                page_size=4,
                dtype=torch.float32,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )
            base.free_pages = torch.tensor(
                [1, 3, 4, 7, 8, 10, 11], dtype=torch.int64
            )
            allocators.append(
                CPShardedKVPoolAllocator(
                    base,
                    cp_rank=2,
                    cp_size=4,
                    cp_kv_chunk_size=8,
                    logical_size=192,
                )
            )

        positions = torch.arange(0, 64, dtype=torch.int64)
        allocations = [
            allocator.alloc_for_positions_with_logical(
                positions,
                owner_rotations_cpu=torch.full_like(positions, 1),
            )
            for allocator in allocators
        ]

        self.assertEqual(
            allocations[0].logical_slots.tolist(),
            allocations[1].logical_slots.tolist(),
        )
        self.assertEqual(
            allocations[0].physical_write_slots.tolist(),
            allocations[1].physical_write_slots.tolist(),
        )
        logical_pages = torch.unique(allocations[0].logical_slots // 4)
        self.assertEqual(
            allocators[0].logical_page_table_to_physical(logical_pages).tolist(),
            allocators[1].logical_page_table_to_physical(logical_pages).tolist(),
        )
        self.assertEqual(
            allocators[0].physical_load_snapshot(),
            allocators[1].physical_load_snapshot(),
        )

    def test_failed_physical_allocation_rolls_back_logical_pages(self):
        base = PagedTokenToKVPoolAllocator(
            size=4,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        base.alloc(4)
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=4,
            logical_size=8,
        )
        before = allocator.logical_free_pages.clone()

        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 4, dtype=torch.int64)
        )

        self.assertIsNone(allocation)
        self.assertEqual(
            sorted(allocator.logical_free_pages.tolist()), sorted(before.tolist())
        )
        self.assertEqual(
            allocator.physical_load_snapshot().allocated_tokens,
            (0, 0),
        )

    def test_residency_load_is_replicated_for_all_owner_ranks(self):
        base = TokenToKVPoolAllocator(
            size=8,
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
            logical_size=16,
        )

        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 8, dtype=torch.int64)
        )

        self.assertEqual(
            allocator.physical_load_snapshot().allocated_tokens,
            (4, 4),
        )
        allocator.free(allocation.logical_slots[4:])
        self.assertEqual(
            allocator.physical_load_snapshot().allocated_tokens,
            (4, 0),
        )
        self.assertEqual(base.available_size(), 4)

    def test_residency_snapshot_is_reused_until_owner_load_changes(self):
        base = TokenToKVPoolAllocator(
            size=8,
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
            logical_size=16,
        )

        empty_snapshot = allocator.physical_load_snapshot()
        self.assertIs(empty_snapshot, allocator.physical_load_snapshot())

        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 8, dtype=torch.int64)
        )
        allocated_snapshot = allocator.physical_load_snapshot()
        self.assertIsNot(allocated_snapshot, empty_snapshot)
        self.assertIs(allocated_snapshot, allocator.physical_load_snapshot())
        self.assertEqual(allocated_snapshot.allocated_tokens, (4, 4))

        allocator.free(allocation.logical_slots[4:])
        released_snapshot = allocator.physical_load_snapshot()
        self.assertIsNot(released_snapshot, allocated_snapshot)
        self.assertIs(released_snapshot, allocator.physical_load_snapshot())
        self.assertEqual(released_snapshot.allocated_tokens, (4, 0))

    def test_logical_free_is_idempotent_for_radix_lifecycle_safety(self):
        base = TokenToKVPoolAllocator(
            size=8,
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
            logical_size=16,
        )
        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 4, dtype=torch.int64)
        )

        allocator.free(allocation.logical_slots)
        allocator.free(allocation.logical_slots)

        self.assertEqual(base.available_size(), 8)
        self.assertEqual(allocator.available_size(), 16)
        self.assertEqual(
            allocator.physical_load_snapshot().allocated_tokens,
            (0, 0),
        )

    def test_capacity_check_does_not_touch_radix_without_rank_deficit(self):
        base = TokenToKVPoolAllocator(
            size=4,
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
            logical_size=8,
        )
        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 4, dtype=torch.int64)
        )
        tree = _EvictingTree(allocator, [allocation.logical_slots])

        result = ensure_cp_sharded_kv_capacity(
            allocator=allocator,
            tree_cache=tree,
            demand_tokens=(0, 4),
        )

        self.assertTrue(result)
        self.assertEqual(tree.evict_calls, [])

    def test_capacity_check_evicts_for_logical_deficit(self):
        base = TokenToKVPoolAllocator(
            size=8,
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
            logical_size=8,
        )
        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 8, dtype=torch.int64)
        )
        tree = _EvictingTree(allocator, [allocation.logical_slots[:4]])

        result = ensure_cp_sharded_kv_capacity(
            allocator=allocator,
            tree_cache=tree,
            demand_tokens=(1, 0),
        )

        self.assertTrue(result)
        self.assertEqual(tree.evict_calls, [1])

    def test_capacity_check_retries_when_first_victim_frees_wrong_owner(self):
        base = TokenToKVPoolAllocator(
            size=4,
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
            logical_size=8,
        )
        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 8, dtype=torch.int64)
        )
        owner_zero = allocation.logical_slots[:4]
        owner_one = allocation.logical_slots[4:]
        tree = _EvictingTree(allocator, [owner_one, owner_zero])

        with patch.object(torch.distributed, "all_reduce") as all_reduce:
            result = ensure_cp_sharded_kv_capacity(
                allocator=allocator,
                tree_cache=tree,
                demand_tokens=(4, 0),
        )

        self.assertTrue(result)
        self.assertEqual(tree.evict_calls, [8, 8])
        all_reduce.assert_not_called()

    def test_capacity_check_batches_allocator_frees_per_radix_eviction(self):
        base = TokenToKVPoolAllocator(
            size=4,
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
            logical_size=8,
        )
        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 8, dtype=torch.int64)
        )
        tree = _BatchEvictingTree(
            allocator,
            [allocation.logical_slots[4:], allocation.logical_slots[:4]],
        )

        with patch.object(
            allocator,
            "_free_compacted",
            wraps=allocator._free_compacted,
        ) as free_compacted:
            result = ensure_cp_sharded_kv_capacity(
                allocator=allocator,
                tree_cache=tree,
                demand_tokens=(4, 0),
            )

        self.assertTrue(result)
        free_compacted.assert_called_once()

    def test_capacity_check_stops_when_radix_cannot_make_progress(self):
        base = TokenToKVPoolAllocator(
            size=4,
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
            logical_size=8,
        )
        allocator.alloc_for_positions_with_logical(
            torch.arange(0, 4, dtype=torch.int64)
        )
        tree = _EvictingTree(allocator, [])

        result = ensure_cp_sharded_kv_capacity(
            allocator=allocator,
            tree_cache=tree,
            demand_tokens=(1, 0),
        )

        self.assertFalse(result)
        self.assertEqual(tree.evict_calls, [2])

    def test_prefill_allocation_rejects_batch_greater_than_one_before_mutation(self):
        base = PagedTokenToKVPoolAllocator(
            size=32,
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
            cp_kv_chunk_size=4,
            logical_size=64,
        )
        maybe_evict_swa = MagicMock()
        batch = SimpleNamespace(
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            reqs=[object(), object()],
            maybe_evict_swa=maybe_evict_swa,
        )

        with self.assertRaisesRegex(NotImplementedError, "batch_size=1"):
            alloc_for_extend(batch)

        maybe_evict_swa.assert_not_called()

    def test_prefill_allocation_forwards_current_split_spec(self):
        base = PagedTokenToKVPoolAllocator(
            size=32,
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
            logical_size=64,
        )
        split_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=8,
            cp_size=2,
            page_size=4,
            owner_rotation=1,
        )
        allocation = CPShardedKVAllocation(
            logical_slots=torch.arange(4, 12, dtype=torch.int64),
            physical_write_slots=torch.arange(4, 12, dtype=torch.int64),
        )
        allocator.alloc_extend_with_logical = MagicMock(return_value=allocation)
        batch = SimpleNamespace(
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            reqs=[
                SimpleNamespace(
                    prefix_indices=torch.empty((0,), dtype=torch.int64),
                    attn_cp_owner_rotation=1,
                )
            ],
            maybe_evict_swa=MagicMock(),
            scale_seq_factor=1,
            prefix_lens=[0],
            extend_lens=[8],
            device="cpu",
            req_to_token_pool=object(),
            seq_lens=torch.tensor([8], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
            extend_num_tokens=8,
            attn_cp_prefill_split_specs=(split_spec,),
        )

        with patch(
            "sglang.srt.mem_cache.common.alloc_req_slots", return_value=[0]
        ), patch("sglang.srt.mem_cache.common.write_cache_indices"):
            out_cache_loc, _, _ = alloc_for_extend(batch)

        self.assertTrue(torch.equal(out_cache_loc, allocation.physical_write_slots))
        self.assertIs(
            allocator.alloc_extend_with_logical.call_args.kwargs["split_spec"],
            split_spec,
        )

    def test_radix_stores_logical_ids_and_evicts_owned_physical_pages(self):
        base = PagedTokenToKVPoolAllocator(
            size=32,
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
            cp_kv_chunk_size=4,
            logical_size=64,
        )
        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 8, dtype=torch.int64)
        )
        tree = RadixCache.create_simulated(
            mock_allocator=allocator,
            page_size=4,
        )
        token_ids = list(range(8))

        tree.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=allocation.logical_slots,
            )
        )
        match = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))

        self.assertEqual(
            match.device_indices.tolist(), allocation.logical_slots.tolist()
        )
        self.assertNotEqual(
            match.device_indices.tolist(), allocation.physical_write_slots.tolist()
        )
        self.assertEqual(base.available_size(), 28)
        tree.evict(EvictParams(num_tokens=8))
        self.assertEqual(base.available_size(), 32)
        self.assertEqual(allocator.available_size(), 64)

if __name__ == "__main__":
    unittest.main()
