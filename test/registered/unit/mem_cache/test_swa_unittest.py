import unittest

import torch

from sglang.srt.disaggregation.kv_events import BlockRemoved, BlockStored
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import available_and_evictable_str
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="stage-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class _DummyReq:
    def __init__(self):
        self._kv_committed_len = 0
        self.swa_prefix_lock_released = False

    def pop_committed_kv_cache(self):
        return self._kv_committed_len


def _build_swa_tree(
    is_eagle: bool,
    page_size: int = 1,
    req_size: int = 8,
    max_context_len: int = 64,
    kv_size: int = 64,
    kv_size_swa: int = 32,
    sliding_window_size: int = 4,
    enable_kv_cache_events: bool = False,
):
    head_num = 8
    head_dim = 128
    num_layers = 24
    global_interval = 4
    dtype = torch.bfloat16
    device = get_device()
    full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
    full_attention_layer_ids_set = set(full_attention_layer_ids)
    swa_attention_layer_ids = [
        i for i in range(num_layers) if i not in full_attention_layer_ids_set
    ]

    req_to_token_pool = ReqToTokenPool(
        size=req_size,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
    )
    kv_pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        enable_kvcache_transpose=False,
        device=device,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    tree = SWARadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            disable=False,
            is_eagle=is_eagle,
            sliding_window_size=sliding_window_size,
            enable_kv_cache_events=enable_kv_cache_events,
        ),
    )
    return tree, allocator, req_to_token_pool


def _swa_alloc(allocator, need_size):
    """SWA-pool alloc that also works for page_size > 1 (built-in alloc asserts page_size == 1)."""
    if allocator.page_size == 1:
        return allocator.alloc(need_size)

    assert need_size % allocator.page_size == 0
    full_indices = allocator.full_attn_allocator.alloc(need_size)
    swa_indices = allocator.swa_attn_allocator.alloc(need_size)
    assert full_indices is not None and swa_indices is not None
    allocator.full_to_swa_index_mapping[full_indices] = swa_indices
    return full_indices


def _insert(tree, allocator, token_ids):
    indices = _swa_alloc(allocator, len(token_ids))
    assert indices is not None
    tree.insert(InsertParams(key=RadixKey(token_ids), value=indices))


def _insert_chain(tree, allocator, token_ids):
    _insert(tree, allocator, token_ids)
    match = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
    return match.last_device_node


def _expected_tail_size(window: int, page_size: int) -> int:
    """Mirror of _maybe_split_leaf_for_swa_lock's tail_size formula."""
    return (window + page_size - 1) // page_size * page_size


class TestSWA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_swa_radix_cache_kv_events(self):
        tree, allocator, _ = _build_swa_tree(
            is_eagle=False, enable_kv_cache_events=True
        )
        tree.take_events()  # Clear the reset event.

        _insert(tree, allocator, [1, 2, 3, 4])
        first_insert_events = [
            e for e in tree.take_events() if isinstance(e, BlockStored)
        ]
        self.assertEqual(len(first_insert_events), 4)
        self.assertEqual([e.token_ids[0] for e in first_insert_events], [1, 2, 3, 4])

        _insert(tree, allocator, [1, 2, 3, 4, 5, 6])
        second_insert_events = [
            e for e in tree.take_events() if isinstance(e, BlockStored)
        ]
        self.assertEqual(len(second_insert_events), 2)
        self.assertEqual([e.token_ids[0] for e in second_insert_events], [5, 6])

        stored_hashes = [
            e.block_hashes[0] for e in first_insert_events + second_insert_events
        ]

        # Evicting only SWA tokens tombstones nodes but keeps full KV blocks.
        result = tree.evict(EvictParams(num_tokens=0, swa_num_tokens=1))
        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertGreaterEqual(result.swa_num_tokens_evicted, 1)
        self.assertEqual(
            [e for e in tree.take_events() if isinstance(e, BlockRemoved)], []
        )

        result = tree.evict(EvictParams(num_tokens=1, swa_num_tokens=0))
        self.assertGreaterEqual(result.num_tokens_evicted, 1)
        removed_hashes = [
            e.block_hashes[0] for e in tree.take_events() if isinstance(e, BlockRemoved)
        ]
        self.assertCountEqual(removed_hashes, stored_hashes)

    def test_swa_radix_cache_kv_events_split_hash(self):
        tree, allocator, _ = _build_swa_tree(
            is_eagle=False, enable_kv_cache_events=True
        )
        tree.take_events()  # Clear the reset event.

        _insert(tree, allocator, [1, 2, 3, 4])
        first_insert_events = [
            e for e in tree.take_events() if isinstance(e, BlockStored)
        ]
        self.assertEqual(len(first_insert_events), 4)
        split_parent_hash = first_insert_events[1].block_hashes[0]

        _insert(tree, allocator, [1, 2, 5, 6])
        second_insert_events = [
            e for e in tree.take_events() if isinstance(e, BlockStored)
        ]
        self.assertEqual(len(second_insert_events), 2)
        self.assertEqual(second_insert_events[0].token_ids, [5])
        self.assertEqual(second_insert_events[0].parent_block_hash, split_parent_hash)

    def test_swa_memory_pool(self):
        size = 16
        size_swa = 16
        page_size = 1
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        pool = SWAKVPool(
            size=size,
            size_swa=size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
        )
        alloc = SWATokenToKVPoolAllocator(
            size=size,
            size_swa=size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        self.assertEqual(
            alloc.full_available_size() + alloc.swa_available_size(), size + size_swa
        )
        index = alloc.alloc(1)
        self.assertEqual(
            alloc.full_available_size() + alloc.swa_available_size(),
            size_swa + size_swa - 2,
        )
        alloc.free_swa(index)
        result = alloc.translate_loc_from_full_to_swa(index)
        print(result)

    def _build_swa_pool_alloc(self, size=16, size_swa=16, page_size=1):
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        pool = SWAKVPool(
            size=size,
            size_swa=size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
        )
        alloc = SWATokenToKVPoolAllocator(
            size=size,
            size_swa=size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        return alloc

    def test_swa_backup_pending_no_pending_frees_immediately(self):
        """Default (no pending refcount) ``free_swa`` is a fast direct free.

        Backwards-compatibility check: any code path that frees SWA slots
        without going through add_backup_pending must continue to free
        immediately, with available_size restored on the same call.
        """
        alloc = self._build_swa_pool_alloc()
        before = alloc.swa_available_size()
        idx = alloc.alloc(1)
        self.assertLess(alloc.swa_available_size(), before)
        alloc.free_swa(idx)
        self.assertEqual(alloc.swa_available_size(), before)
        self.assertEqual(len(alloc._deferred_swa_free), 0)

    def test_swa_backup_pending_defers_then_frees_on_dec(self):
        """``free_swa`` defers physical free for slots with pending refcount.

        Lifecycle the offload manager will rely on:
          add_backup_pending(swa_loc)   # offload starts D->H
          ... (D->H copy in flight)
          free_swa(full_loc)            # _evict_swa fires before backup done
          ... slot is on _deferred_swa_free, NOT freed yet
          dec_backup_pending(swa_loc)   # backup completes
          ... slot now freed, available_size restored
        """
        alloc = self._build_swa_pool_alloc()
        before = alloc.swa_available_size()
        full_idx = alloc.alloc(1)
        swa_idx = alloc.translate_loc_from_full_to_swa(full_idx)
        self.assertGreater(int(swa_idx[0].item()), 0)

        # Mark backup in-flight on the SWA slot.
        alloc.add_backup_pending(swa_idx)
        self.assertEqual(int(alloc.backup_pending_ref[swa_idx][0].item()), 1)

        # Eviction tries to free the full slot — SWA side must defer.
        post_alloc_avail = alloc.swa_available_size()
        alloc.free_swa(full_idx)
        self.assertEqual(
            alloc.swa_available_size(),
            post_alloc_avail,
            "free_swa must defer when backup_pending_ref > 0",
        )
        self.assertEqual(len(alloc._deferred_swa_free), 1)
        self.assertEqual(alloc._deferred_swa_free[0], int(swa_idx[0].item()))

        # Backup completes — slot now flushed to free list.
        alloc.dec_backup_pending(swa_idx)
        self.assertEqual(alloc.swa_available_size(), before)
        self.assertEqual(len(alloc._deferred_swa_free), 0)
        self.assertEqual(int(alloc.backup_pending_ref[swa_idx][0].item()), 0)

    def test_swa_backup_pending_refcount_supports_overlapping_backups(self):
        """Refcount semantics: same slot can be added/dec'd multiple times.

        If two concurrent backup operations target the same slot
        (D->H followed by H->L3, both keep a refcount), free_swa must
        wait until the LAST dec_backup_pending."""
        alloc = self._build_swa_pool_alloc()
        before = alloc.swa_available_size()
        full_idx = alloc.alloc(1)
        swa_idx = alloc.translate_loc_from_full_to_swa(full_idx)

        alloc.add_backup_pending(swa_idx)
        alloc.add_backup_pending(swa_idx)  # second concurrent backup
        self.assertEqual(int(alloc.backup_pending_ref[swa_idx][0].item()), 2)

        alloc.free_swa(full_idx)
        post_first_dec = alloc.swa_available_size()

        alloc.dec_backup_pending(swa_idx)
        # Refcount still > 0 — slot must NOT be freed yet.
        self.assertEqual(
            alloc.swa_available_size(),
            post_first_dec,
            "free deferred until refcount reaches 0",
        )
        self.assertEqual(int(alloc.backup_pending_ref[swa_idx][0].item()), 1)

        alloc.dec_backup_pending(swa_idx)
        self.assertEqual(alloc.swa_available_size(), before)

    def test_swa_backup_pending_refcount_covers_paged_padding_slots(self):
        """Paged SWA indices can legally land in the final padding page."""
        page_size = 16
        alloc = self._build_swa_pool_alloc(
            size=32, size_swa=32, page_size=page_size
        )
        self.assertEqual(alloc.backup_pending_ref.numel(), 32 + page_size)

        full_idx = alloc.full_attn_allocator.alloc(32)
        swa_idx = alloc.swa_attn_allocator.alloc(32)
        self.assertIsNotNone(full_idx)
        self.assertIsNotNone(swa_idx)
        alloc.full_to_swa_index_mapping[full_idx] = swa_idx

        tail_full = full_idx[-page_size:]
        tail_swa = swa_idx[-page_size:]
        self.assertEqual(int(tail_swa[-1].item()), 32 + page_size - 1)

        alloc.add_backup_pending(tail_swa)
        self.assertTrue(bool((alloc.backup_pending_ref[tail_swa] == 1).all().item()))

        post_alloc_avail = alloc.swa_available_size()
        alloc.free_swa(tail_full)
        self.assertEqual(alloc.swa_available_size(), post_alloc_avail)
        self.assertEqual(len(alloc._deferred_swa_free), page_size)

        alloc.dec_backup_pending(tail_swa)
        self.assertEqual(alloc.swa_available_size(), post_alloc_avail + page_size)
        self.assertEqual(len(alloc._deferred_swa_free), 0)

    def test_swa_backup_pending_zero_indices_noop(self):
        """SWA index 0 is the 'no pair' sentinel — must never be tracked."""
        alloc = self._build_swa_pool_alloc()
        zero = torch.tensor([0], dtype=torch.int64, device=alloc.device)
        # No-op: no exception, no refcount change.
        alloc.add_backup_pending(zero)
        alloc.dec_backup_pending(zero)
        self.assertEqual(int(alloc.backup_pending_ref[0].item()), 0)

    def test_swa_backup_pending_dec_underflow_clamps_not_asserts(self):
        """``dec_backup_pending`` must not assert on underflow.

        Abort/restart paths can decrement without a matching add (e.g. the
        request was cancelled before alloc_decode populated the SWA pair).
        The pool clamps the refcount back to 0 and logs, so a single
        dropped pair doesn't take the scheduler down.
        """
        alloc = self._build_swa_pool_alloc()
        full_idx = alloc.alloc(1)
        swa_idx = alloc.translate_loc_from_full_to_swa(full_idx)
        self.assertEqual(int(alloc.backup_pending_ref[swa_idx][0].item()), 0)

        # Decrement without a matching add — must not raise.
        alloc.dec_backup_pending(swa_idx)
        # Refcount clamped to 0, slot remains usable.
        self.assertEqual(int(alloc.backup_pending_ref[swa_idx][0].item()), 0)
        self.assertEqual(len(alloc._deferred_swa_free), 0)

    def test_swa_trim_to_trailing_window(self):
        """`_trim_swa_to_trailing_window` slices host_value + hash list to
        the last `n_window_pages` pages.

        A radix node carries one hash per page in `hash_value` and one
        host slot per token in `host_value` (page_size tokens per page).
        BACKUP_STORAGE only needs the trailing window because PREFETCH
        will never read older pages back. The trim must produce a
        consistent (host_value, hash_value) pair where lengths satisfy
        `len(host_value) == len(hash_value) * page_size`.
        """
        from sglang.srt.mem_cache.unified_cache_components.swa_component import (
            _trim_swa_to_trailing_window,
        )
        page_size = 64
        # Long node: 6 pages of hash + 6*64 host slots
        host = torch.arange(6 * page_size)
        hashes = [f"h{i}" for i in range(6)]

        # Window = 2 → keep last 2 pages.
        trimmed_host, trimmed_keys = _trim_swa_to_trailing_window(
            host, hashes, n_window_pages=2, page_size=page_size
        )
        self.assertEqual(trimmed_keys, ["h4", "h5"])
        self.assertEqual(len(trimmed_host), 2 * page_size)
        # Verify slice semantics: trimmed_host should match host[-128:].
        self.assertTrue(torch.equal(trimmed_host, host[-2 * page_size:]))

        # Window >= total → no trim, returns full list (defensive).
        full_host, full_keys = _trim_swa_to_trailing_window(
            host, hashes, n_window_pages=10, page_size=page_size
        )
        self.assertEqual(full_keys, hashes)
        self.assertTrue(torch.equal(full_host, host))

        # Single-page node + window=2 → keep the single page (n_keep=1).
        small_host = torch.arange(page_size)
        small_hashes = ["solo"]
        sh, sk = _trim_swa_to_trailing_window(
            small_host, small_hashes, n_window_pages=2, page_size=page_size
        )
        self.assertEqual(sk, ["solo"])
        self.assertEqual(len(sh), page_size)

    def test_swa_bounded_bfs_descendant_leaf_window(self):
        """Bounded BFS keeps only nodes whose descendant leaf can terminate
        within the trailing SWA window.

        With a 2-page window and a one-page chain A->B->C->D, only C and D
        can contribute SWA KV to the trailing window of the D prefix. A and B
        are too old and may be skipped by the optional bounded-BFS optimizer.
        """
        from sglang.srt.mem_cache.unified_cache_components.swa_component import (
            _swa_node_has_descendant_leaf_within_window,
        )

        page_size = 1

        class _Node:
            def __init__(self, name, children=()):
                self.key = [name]
                self.hash_value = [name]
                self.children = {child.key[0]: child for child in children}

        d = _Node("D")
        c = _Node("C", [d])
        b = _Node("B", [c])
        a = _Node("A", [b])

        self.assertFalse(
            _swa_node_has_descendant_leaf_within_window(
                a, n_window_pages=2, page_size=page_size
            )
        )
        self.assertFalse(
            _swa_node_has_descendant_leaf_within_window(
                b, n_window_pages=2, page_size=page_size
            )
        )
        self.assertTrue(
            _swa_node_has_descendant_leaf_within_window(
                c, n_window_pages=2, page_size=page_size
            )
        )
        self.assertTrue(
            _swa_node_has_descendant_leaf_within_window(
                d, n_window_pages=2, page_size=page_size
            )
        )

        # Branching is conservative: if any descendant leaf is close enough,
        # keep the node so short-prefix traffic can still hit SWA L3.
        short_leaf = _Node("S")
        branched = _Node("A2", [b, short_leaf])
        self.assertTrue(
            _swa_node_has_descendant_leaf_within_window(
                branched, n_window_pages=2, page_size=page_size
            )
        )

    def test_swa_storage_tail_trim_and_bounded_bfs_flags_are_orthogonal(self):
        """SWA L3 write-reduction knobs can be enabled independently.

        Tail trim changes how many pages a kept node writes. Bounded BFS
        decides whether to skip the node entirely based on descendant-leaf
        distance. The two decisions are independent.
        """
        from sglang.srt.mem_cache.unified_cache_components.swa_component import (
            SWAComponent,
        )

        page_size = 4

        class _Cache:
            page_size = 4

        class _CD:
            def __init__(self, host_value):
                self.host_value = host_value
                self.value = None

        class _Node:
            def __init__(self, name, pages, children=()):
                self.key = list(range(pages * page_size))
                self.hash_value = [f"{name}{i}" for i in range(pages)]
                self.children = {child.hash_value[0]: child for child in children}
                self.component_data = {
                    ComponentType.SWA: _CD(torch.arange(pages * page_size))
                }

        leaf = _Node("L", pages=1)
        node = _Node("N", pages=3, children=[leaf])

        comp = SWAComponent.__new__(SWAComponent)
        comp.component_type = ComponentType.SWA
        comp.cache = _Cache()
        comp.sliding_window_size = 2 * page_size
        comp._swa_kv_pool_host = object()

        def build(*, tail_trim: bool, bounded_bfs: bool, target=node):
            with envs.SGLANG_HICACHE_SWA_STORAGE_ENABLE.override(True):
                with envs.SGLANG_HICACHE_SWA_STORAGE_TAIL_TRIM_ENABLE.override(
                    tail_trim
                ):
                    with envs.SGLANG_HICACHE_SWA_STORAGE_BOUNDED_BFS_ENABLE.override(
                        bounded_bfs
                    ):
                        return comp.build_hicache_transfers(
                            target, CacheTransferPhase.BACKUP_STORAGE
                        )

        # No optimization: all node pages are emitted.
        transfers = build(tail_trim=False, bounded_bfs=False)
        self.assertEqual(transfers[0].name, PoolName.SWA)
        self.assertEqual(transfers[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)
        self.assertEqual(transfers[0].keys, ["N0", "N1", "N2"])
        self.assertEqual(transfers[0].host_indices.tolist(), list(range(12)))

        # Tail trim only: keep the node, but write only its trailing window
        # plus one guard page for page-aligned replay prefixes.
        transfers = build(tail_trim=True, bounded_bfs=False)
        self.assertEqual(transfers[0].keys, ["N0", "N1", "N2"])
        self.assertEqual(transfers[0].host_indices.tolist(), list(range(12)))

        # Bounded BFS only: the node is kept because its child leaf is within
        # the 2-page window; with tail trim off it still writes all pages.
        transfers = build(tail_trim=False, bounded_bfs=True)
        self.assertEqual(transfers[0].keys, ["N0", "N1", "N2"])
        self.assertEqual(transfers[0].host_indices.tolist(), list(range(12)))

        # Both on: bounded BFS keeps this node, tail trim then trims it.
        transfers = build(tail_trim=True, bounded_bfs=True)
        self.assertEqual(transfers[0].keys, ["N0", "N1", "N2"])
        self.assertEqual(transfers[0].host_indices.tolist(), list(range(12)))

        # A node whose nearest descendant leaf is a full window away is skipped
        # only when bounded BFS is enabled.
        deep_leaf = _Node("D", pages=1)
        mid = _Node("M", pages=1, children=[deep_leaf])
        old = _Node("O", pages=1, children=[mid])
        self.assertIsNotNone(build(tail_trim=True, bounded_bfs=False, target=old))
        self.assertIsNone(build(tail_trim=True, bounded_bfs=True, target=old))

    def test_swa_storage_keys_align_to_partial_host_pages_when_trim_off(self):
        """A prefetch-loaded host node may cover only the trailing pages.

        Disabling tail trim should not pair that partial host_value with the
        full node hash list; storage writes still need one key per host page.
        """
        from sglang.srt.mem_cache.unified_cache_components.swa_component import (
            SWAComponent,
        )

        page_size = 4

        class _Cache:
            page_size = 4

        class _CD:
            value = None

            def __init__(self):
                # Two host pages loaded for a four-page node.
                self.host_value = torch.arange(2 * page_size)

        class _Node:
            key = list(range(4 * page_size))
            hash_value = ["h0", "h1", "h2", "h3"]
            children = {}
            component_data = {ComponentType.SWA: _CD()}

        comp = SWAComponent.__new__(SWAComponent)
        comp.component_type = ComponentType.SWA
        comp.cache = _Cache()
        comp.sliding_window_size = 4 * page_size
        comp._swa_kv_pool_host = object()

        with envs.SGLANG_HICACHE_SWA_STORAGE_ENABLE.override(True):
            with envs.SGLANG_HICACHE_SWA_STORAGE_TAIL_TRIM_ENABLE.override(False):
                with envs.SGLANG_HICACHE_SWA_STORAGE_BOUNDED_BFS_ENABLE.override(
                    False
                ):
                    transfers = comp.build_hicache_transfers(
                        _Node, CacheTransferPhase.BACKUP_STORAGE
                    )

        xfer = transfers[0]
        self.assertEqual(xfer.keys, ["h2", "h3"])
        self.assertEqual(int(xfer.host_indices.numel()), 2 * page_size)

    def test_swa_storage_key_alignment_keeps_partial_page(self):
        """A sub-page SWA host_value still maps to the trailing page key."""
        from sglang.srt.mem_cache.unified_cache_components.swa_component import (
            _align_swa_keys_to_host_pages,
        )

        host = torch.arange(3)
        aligned_host, keys = _align_swa_keys_to_host_pages(
            host, ["h0", "h1"], page_size=4
        )
        self.assertEqual(keys, ["h1"])
        self.assertTrue(torch.equal(aligned_host, host))

    def test_swa_trailing_host_value_split_keeps_tail_alignment(self):
        """Splitting a PREFETCH-loaded SWA host node must preserve tail offsets."""
        from sglang.srt.mem_cache.unified_cache_components.swa_component import (
            _split_swa_trailing_device_value_on_node_split,
            _split_swa_trailing_host_value_on_node_split,
        )

        # Full coverage behaves like an ordinary tensor split.
        host = torch.arange(12)
        parent_host, child_host = _split_swa_trailing_host_value_on_node_split(
            host, old_node_len=12, split_len=5
        )
        self.assertEqual(parent_host.tolist(), list(range(5)))
        self.assertEqual(child_host.tolist(), list(range(5, 12)))

        # Tail-only host coverage for old [0, 12) with host covering [8, 12).
        # Split at 5: parent [0, 5) has no SWA host data; child [5, 12)
        # carries the full tail coverage.
        host = torch.arange(100, 104)
        parent_host, child_host = _split_swa_trailing_host_value_on_node_split(
            host, old_node_len=12, split_len=5
        )
        self.assertIsNone(parent_host)
        self.assertEqual(child_host.tolist(), [100, 101, 102, 103])

        # Split inside the covered suffix. Parent gets the covered overlap
        # [8, 10), child gets [10, 12).
        parent_host, child_host = _split_swa_trailing_host_value_on_node_split(
            host, old_node_len=12, split_len=10
        )
        self.assertEqual(parent_host.tolist(), [100, 101])
        self.assertEqual(child_host.tolist(), [102, 103])

        # Split after the covered suffix gives all SWA host data to parent.
        parent_host, child_host = _split_swa_trailing_host_value_on_node_split(
            host, old_node_len=12, split_len=12
        )
        self.assertEqual(parent_host.tolist(), [100, 101, 102, 103])
        self.assertIsNone(child_host)

        # Device values restored from L3 use the same tail-alignment rule.
        parent_value, child_value = _split_swa_trailing_device_value_on_node_split(
            torch.arange(200, 204), old_node_len=12, split_len=10
        )
        self.assertEqual(parent_value.tolist(), [200, 201])
        self.assertEqual(child_value.tolist(), [202, 203])

    def test_swa_redistribute_split_preserves_partial_host_tail(self):
        """Node split must not attach tail-only SWA host data to old prefixes."""
        from sglang.srt.mem_cache.unified_cache_components.swa_component import (
            SWAComponent,
        )
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            ComponentData,
        )

        class _LRU:
            def __init__(self):
                self.nodes = set()

            def in_list(self, node):
                return node.id in self.nodes

            def insert_mru(self, node):
                self.nodes.add(node.id)

            def remove_node(self, node):
                self.nodes.remove(node.id)

        class _Cache:
            def __init__(self, lru):
                self.host_lru_lists = {ComponentType.SWA: lru}

        class _Node:
            def __init__(self, node_id, key_len):
                self.id = node_id
                self.key = list(range(key_len))
                self.component_data = [ComponentData() for _ in range(3)]

        def run_split(split_len, child_len, host_values):
            lru = _LRU()
            comp = SWAComponent.__new__(SWAComponent)
            comp.cache = _Cache(lru)
            comp.component_type = ComponentType.SWA
            parent = _Node(1, split_len)
            child = _Node(2, child_len)
            child.component_data[ComponentType.SWA].host_value = torch.tensor(
                host_values
            )
            child.component_data[ComponentType.SWA].metadata["uuid"] = "u"
            lru.insert_mru(child)
            comp.redistribute_on_node_split(parent, child)
            return parent, child, lru

        # Old node length 12, SWA host covers old [8, 12), split at 5.
        # Parent [0, 5) must not receive tail KV.
        parent, child, lru = run_split(5, 7, [100, 101, 102, 103])
        self.assertIsNone(parent.component_data[ComponentType.SWA].host_value)
        self.assertEqual(
            child.component_data[ComponentType.SWA].host_value.tolist(),
            [100, 101, 102, 103],
        )
        self.assertFalse(lru.in_list(parent))
        self.assertTrue(lru.in_list(child))

        # Split inside the covered suffix: parent gets [8, 10), child [10, 12).
        parent, child, lru = run_split(10, 2, [100, 101, 102, 103])
        self.assertEqual(
            parent.component_data[ComponentType.SWA].host_value.tolist(), [100, 101]
        )
        self.assertEqual(
            child.component_data[ComponentType.SWA].host_value.tolist(), [102, 103]
        )
        self.assertTrue(lru.in_list(parent))
        self.assertTrue(lru.in_list(child))

    def test_swa_validator_counts_host_value_for_partial_trailing_coverage(self):
        """SWA validator must count host_value length, not node.key length.

        After PREFETCH from L3, a single multi-page node can hold a
        host_value that covers only the trailing window — not the full
        node key. The validator must therefore add `len(host_value)` to
        its in-window accumulator, not `len(node.key)`. Counting the
        full key length would over-claim coverage and let SWA layer
        attention read positions whose KV is missing.
        """
        from sglang.srt.mem_cache.unified_cache_components.swa_component import (
            SWAComponent,
        )
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            ComponentType,
        )

        # Minimal stand-in: a closure capturing only the fields the
        # validator actually reads (sliding_window_size, component_type,
        # node.component_data[ct].{value,host_value}, node.key).
        page_size = 64
        sliding_window_size = 128

        class _NodeStub:
            def __init__(self, key_len, host_value_len, has_device_value=False):
                self.key = list(range(key_len))
                cd = type(
                    "_CD",
                    (),
                    {
                        "value": (
                            torch.arange(host_value_len)
                            if has_device_value
                            else None
                        ),
                        "host_value": (
                            torch.arange(host_value_len)
                            if host_value_len > 0
                            else None
                        ),
                    },
                )()
                self.component_data = {ComponentType.SWA: cd}

        # Make a fake SWAComponent — bypass __init__ since it pulls in
        # full UnifiedRadixCache / allocator state we don't need here.
        comp = SWAComponent.__new__(SWAComponent)
        comp.sliding_window_size = sliding_window_size
        comp.component_type = ComponentType.SWA

        # Case 1: host-only node, host_value covers full window → match.
        validator = comp.create_match_validator(match_device_only=False)
        node_full_window = _NodeStub(
            key_len=384, host_value_len=128, has_device_value=False
        )
        self.assertTrue(validator(node_full_window))

        # Case 2: host-only node, host_value < window → must NOT match
        # (under-coverage would corrupt SWA-layer attention).
        validator = comp.create_match_validator(match_device_only=False)
        node_partial = _NodeStub(
            key_len=384, host_value_len=64, has_device_value=False
        )
        self.assertFalse(validator(node_partial))

        # Case 3: short host-only prefix is valid when host_value covers the
        # whole prefix, even though it is shorter than sliding_window_size.
        validator = comp.create_match_validator(match_device_only=False)
        node_short_full = _NodeStub(
            key_len=64, host_value_len=64, has_device_value=False
        )
        self.assertTrue(validator(node_short_full))

        validator = comp.create_match_validator(match_device_only=False)
        node_short_partial = _NodeStub(
            key_len=64, host_value_len=32, has_device_value=False
        )
        self.assertFalse(validator(node_short_partial))

        # Case 4: device-resident node uses len(node.key) (full coverage).
        validator = comp.create_match_validator(match_device_only=False)
        node_device = _NodeStub(
            key_len=128, host_value_len=128, has_device_value=True
        )
        self.assertTrue(validator(node_device))

        # Case 5: matching is cumulative along the radix path. A missing old
        # node can become irrelevant once later nodes cover the full trailing
        # sliding window, but not before.
        validator = comp.create_match_validator(match_device_only=False)
        node_missing_old = _NodeStub(
            key_len=64, host_value_len=0, has_device_value=False
        )
        node_new_tail_1 = _NodeStub(
            key_len=64, host_value_len=64, has_device_value=False
        )
        node_new_tail_2 = _NodeStub(
            key_len=64, host_value_len=64, has_device_value=False
        )
        self.assertFalse(validator(node_missing_old))
        self.assertFalse(validator(node_new_tail_1))
        self.assertTrue(validator(node_new_tail_2))

    def test_swa_radix_cache_1(self):
        # args
        req_size = 10
        max_context_len = 128
        kv_size = 128
        kv_size_swa = 64
        page_size = 1
        sliding_window_size = 4
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        # setup req to token pool
        req_to_token_pool = ReqToTokenPool(
            size=req_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        # setup kv pool
        kv_pool = SWAKVPool(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
        )
        # setup token to kv pool allocator
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        # setup radix cache
        tree = SWARadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                disable=False,
                page_size=page_size,
                sliding_window_size=sliding_window_size,
            ),
        )

        # test
        print(
            f"[Start] allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req1_token_ids, req1_kv_indices = [1, 2, 3], allocator.alloc(3)
        self.assertEqual(len(req1_token_ids), len(req1_kv_indices))
        print(
            f"req1: inserting, req1_token_ids: {req1_token_ids}, req1_kv_indices: {req1_kv_indices}"
        )
        key = RadixKey(req1_token_ids)
        result = tree.insert(InsertParams(key=key, value=req1_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req1: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        self.assertEqual(len(req2_token_ids), len(req2_kv_indices))
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        key = RadixKey(req2_token_ids)
        result = tree.insert(InsertParams(key=key, value=req2_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req2: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        self.assertEqual(len(req3_token_ids), len(req3_kv_indices))
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        key = RadixKey(req3_token_ids)
        result = tree.insert(InsertParams(key=key, value=req3_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req3: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        self.assertEqual(len(req4_token_ids), len(req4_kv_indices))
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        key = RadixKey(req4_token_ids)
        result = tree.insert(InsertParams(key=key, value=req4_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        print(
            f"req4: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )

        tree.pretty_print()
        full_num_tokens, swa_num_tokens = 1, 0
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 0, 1
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 1, 2
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req5_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 0)

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req6_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 7)
        self.assertEqual(len(last_node.key), 2)
        self.assertEqual(last_node.key.token_ids[0], 60)
        self.assertEqual(last_node.key.token_ids[1], 70)

        print(tree.available_and_evictable_str())
        print(available_and_evictable_str(tree))
        tree.sanity_check()

    def test_swa_radix_cache_eagle(self):
        # args
        req_size = 10
        max_context_len = 128
        kv_size = 128
        kv_size_swa = 64
        page_size = 1
        sliding_window_size = 4
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]
        # setup req to token pool
        req_to_token_pool = ReqToTokenPool(
            size=req_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        # setup kv pool
        kv_pool = SWAKVPool(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
        )
        # setup token to kv pool allocator
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        # setup radix cache
        tree = SWARadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                disable=False,
                is_eagle=True,
                sliding_window_size=sliding_window_size,
            ),
        )

        # test
        print(
            f"[Start] allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req1_token_ids, req1_kv_indices = [1, 2, 3], allocator.alloc(3)
        self.assertEqual(len(req1_token_ids), len(req1_kv_indices))
        print(
            f"req1: inserting, req1_token_ids: {req1_token_ids}, req1_kv_indices: {req1_kv_indices}"
        )
        key = RadixKey(req1_token_ids)
        result = tree.insert(InsertParams(key=key, value=req1_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 0)
        print(
            f"req1: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        self.assertEqual(len(req2_token_ids), len(req2_kv_indices))
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        key = RadixKey(req2_token_ids)
        result = tree.insert(InsertParams(key=key, value=req2_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 2)
        print(
            f"req2: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        self.assertEqual(len(req3_token_ids), len(req3_kv_indices))
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        key = RadixKey(req3_token_ids)
        result = tree.insert(InsertParams(key=key, value=req3_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 0)
        print(
            f"req3: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        self.assertEqual(len(req4_token_ids), len(req4_kv_indices))
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        key = RadixKey(req4_token_ids)
        result = tree.insert(InsertParams(key=key, value=req4_kv_indices[: len(key)]))
        prefix_len = result.prefix_len
        self.assertEqual(prefix_len, 4)
        print(
            f"req4: prefix_len: {prefix_len}, allocator swa available size: {allocator.swa_available_size()}, full available size: {allocator.full_available_size()}"
        )

        tree.pretty_print()
        full_num_tokens, swa_num_tokens = 1, 0
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        evict_result = tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        assert isinstance(evict_result, EvictResult)
        assert (
            evict_result.num_tokens_evicted >= full_num_tokens
        )  # May evict more due to node granularity
        print(
            f"evicted {evict_result.num_tokens_evicted} full tokens, {evict_result.swa_num_tokens_evicted} swa tokens"
        )
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 0, 1
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        evict_result = tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        assert isinstance(evict_result, EvictResult)
        assert (
            evict_result.swa_num_tokens_evicted >= swa_num_tokens
        ), f"evicted {evict_result.swa_num_tokens_evicted} swa tokens, expected {swa_num_tokens}"
        tree.pretty_print()

        full_num_tokens, swa_num_tokens = 1, 2
        print(f"evicting {full_num_tokens} full token and {swa_num_tokens} swa token")
        evict_result = tree.evict(
            EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
        )
        assert isinstance(evict_result, EvictResult)
        assert (
            evict_result.num_tokens_evicted >= full_num_tokens
        ), f"evicted {evict_result.num_tokens_evicted} full tokens, expected {full_num_tokens}"
        assert (
            evict_result.swa_num_tokens_evicted >= swa_num_tokens
        ), f"evicted {evict_result.swa_num_tokens_evicted} swa tokens, expected {swa_num_tokens}"
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req5_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 0)  # no swa prefix matched

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req6_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        self.assertEqual(len(kv_indices), 6)
        self.assertEqual(len(last_node.key), 2)
        # Bigram view: token_ids holds raw tokens; iteration yields bigram tuples.
        self.assertTrue(last_node.key.is_bigram)
        self.assertEqual(list(last_node.key), [(5, 60), (60, 70)])

    def test_swa_cache_finished_req_eagle_uses_cache_protected_len_and_bigram_key(self):
        tree, allocator, req_to_token_pool = _build_swa_tree(is_eagle=True)

        # Case 1: is_insert=True should pass bigram key and use cache_protected_len.
        req = _DummyReq()
        req.req_pool_idx = 0
        req.origin_input_ids = [1, 2, 3, 4, 5, 6]
        req.output_ids = []
        req._kv_committed_len = len(req.origin_input_ids)
        kv_indices = allocator.alloc(req._kv_committed_len)
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, req._kv_committed_len)), kv_indices
        )
        req.extra_key = None
        req.last_node = tree.root_node
        req.swa_uuid_for_lock = None
        req.swa_evicted_seqlen = 0
        req.cache_protected_len = 1
        # Intentionally mismatch to ensure code does not use len(prefix_indices).
        req.prefix_indices = torch.tensor([7, 8, 9, 10, 11], device=tree.device)

        captured = {}
        original_insert = tree.insert

        def wrapped_insert(params):
            captured["prev_prefix_len"] = params.prev_prefix_len
            captured["is_bigram"] = params.key.is_bigram
            captured["key_len"] = len(params.key)
            return original_insert(params)

        tree.insert = wrapped_insert
        tree.cache_finished_req(req, is_insert=True)

        self.assertEqual(captured["prev_prefix_len"], req.cache_protected_len)
        self.assertTrue(captured["is_bigram"])
        self.assertEqual(captured["key_len"], len(req.origin_input_ids) - 1)

        # Case 2: is_insert=False should free [cache_protected_len:page_aligned_len]
        # even when len(prefix_indices) is intentionally larger.
        req2 = _DummyReq()
        req2.req_pool_idx = 1
        req2.origin_input_ids = [11, 12, 13, 14, 15, 16]
        req2.output_ids = []
        req2._kv_committed_len = len(req2.origin_input_ids)
        kv_indices2 = allocator.alloc(req2._kv_committed_len)
        req_to_token_pool.write(
            (req2.req_pool_idx, slice(0, req2._kv_committed_len)), kv_indices2
        )
        req2.extra_key = None
        req2.last_node = tree.root_node
        req2.swa_uuid_for_lock = None
        req2.swa_evicted_seqlen = 0
        req2.cache_protected_len = 1
        req2.prefix_indices = torch.tensor([21, 22, 23, 24, 25], device=tree.device)

        freed_lens = []
        original_free = allocator.free

        def wrapped_free(indices):
            freed_lens.append(int(indices.numel()))
            return original_free(indices)

        allocator.free = wrapped_free
        tree.cache_finished_req(req2, is_insert=False)

        # EAGLE + page_size=1 => page_aligned_len = committed_len - 1 = 5
        # Expected frees:
        #   overlap range [1:5] -> 4
        #   tail range [5:]     -> 1
        self.assertEqual(freed_lens, [4, 1])


# Optimization: SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.
# Splits a freshly-inserted leaf at the (page-aligned) sliding-window
# boundary so a future inc_lock_ref protects only ~sliding_window_size SWA
# tokens instead of the whole chunked-prefill chain.
class TestSWASplitLeafOnInsert(CustomTestCase):
    def _insert_and_lock(self, *, window, page_size, leaf_len, flag_on):
        tree, allocator, _ = _build_swa_tree(
            is_eagle=False,
            kv_size=128,
            kv_size_swa=64,
            sliding_window_size=window,
            page_size=page_size,
        )
        token_ids = list(range(leaf_len))
        with envs.SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.override(flag_on):
            leaf = _insert_chain(tree, allocator, token_ids)
        result = tree.inc_lock_ref(leaf)
        return tree, leaf, result

    def test_flag_off_protects_full_leaf(self):
        tree, leaf, _ = self._insert_and_lock(
            window=4, page_size=1, leaf_len=12, flag_on=False
        )
        self.assertEqual(len(leaf.value), 12)
        self.assertEqual(tree.swa_protected_size_, 12)

    def test_flag_on_caps_protection_at_window(self):
        # (window, page_size, leaf_len, expected_tail_size); leaf_len picked
        # > tail_size and page-aligned for page_size > 1.
        cases = [
            (4, 1, 12, 4),
            (4, 1, 5, 4),
            (1, 1, 5, 1),
            (4, 2, 12, 4),
            (8, 2, 12, 8),
            (4, 4, 12, 4),
            # window NOT page-aligned -> tail rounds up to page boundary.
            (3, 2, 12, 4),
            (5, 4, 12, 8),
            (3, 4, 12, 4),
        ]
        for window, page_size, leaf_len, expected_tail in cases:
            with self.subTest(window=window, page_size=page_size, leaf_len=leaf_len):
                self.assertEqual(_expected_tail_size(window, page_size), expected_tail)
                tree, leaf, _ = self._insert_and_lock(
                    window=window,
                    page_size=page_size,
                    leaf_len=leaf_len,
                    flag_on=True,
                )
                self.assertEqual(len(leaf.value), expected_tail)
                self.assertEqual(tree.swa_protected_size_, expected_tail)

    def test_flag_on_no_split_when_leaf_within_window(self):
        # leaf_len <= tail_size: split must no-op.
        cases = [
            (4, 1, 4),
            (4, 1, 3),
            (4, 2, 4),
            (3, 2, 4),
            (8, 2, 4),
            (4, 4, 4),
        ]
        for window, page_size, leaf_len in cases:
            with self.subTest(window=window, page_size=page_size, leaf_len=leaf_len):
                tree, leaf, _ = self._insert_and_lock(
                    window=window,
                    page_size=page_size,
                    leaf_len=leaf_len,
                    flag_on=True,
                )
                self.assertEqual(len(leaf.value), leaf_len)
                self.assertEqual(tree.swa_protected_size_, leaf_len)

    def test_match_prefix_returns_full_chain_after_split(self):
        tree, allocator, _ = _build_swa_tree(
            is_eagle=False,
            kv_size=128,
            kv_size_swa=64,
            sliding_window_size=4,
            page_size=1,
        )
        token_ids = list(range(12))
        with envs.SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT.override(True):
            inserted_leaf = _insert_chain(tree, allocator, token_ids)
        self.assertEqual(len(inserted_leaf.value), 4)
        match = tree.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(match.device_indices.shape[0], 12)
        self.assertIs(match.last_device_node, inserted_leaf)

    def test_dec_lock_ref_after_split_balances_to_zero(self):
        tree, leaf, result = self._insert_and_lock(
            window=4, page_size=1, leaf_len=12, flag_on=True
        )
        self.assertEqual(tree.swa_protected_size_, 4)
        self.assertEqual(tree.full_protected_size_, 12)

        tree.dec_lock_ref(
            leaf,
            params=DecLockRefParams(swa_uuid_for_lock=result.swa_uuid_for_lock),
        )

        self.assertEqual(tree.swa_protected_size_, 0)
        self.assertEqual(tree.full_protected_size_, 0)
        tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
