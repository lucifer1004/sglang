from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    CPHiCacheTransferPlan,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.unified_cache_components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
    next_component_uuid,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


logger = logging.getLogger(__name__)


def _trim_swa_to_trailing_window(
    host_value: torch.Tensor,
    hash_value,
    n_window_pages: int,
    page_size: int,
):
    """Restrict the SWA host_value/hash_value to the last `n_window_pages`
    pages of the node, mirroring what PREFETCH will request later.

    Each radix node holds a multi-page chunk of KV: ``hash_value`` is a
    per-page hash list (length = num_pages_for_this_node) and
    ``host_value`` is one host slot per token (length = num_pages * page_size).
    SWA layers only attend to the last ``sliding_window_size`` tokens, so
    only the trailing window pages will ever be read back from L3 (PREFETCH
    asks for exactly that many via ``TRAILING_PAGES``). Persisting earlier
    pages wastes L3 write bandwidth and storage.

    Returns ``(trimmed_host_value, trimmed_hash_list)`` where the keep
    count is ``min(n_window_pages, len(hash_value))``.
    """
    n_total = len(hash_value)
    n_keep = min(n_total, max(1, n_window_pages))
    if n_keep >= n_total:
        return host_value, list(hash_value)
    keep_tokens = n_keep * page_size
    return host_value[-keep_tokens:], list(hash_value)[-n_keep:]


def _align_swa_keys_to_host_pages(
    host_value: torch.Tensor,
    hash_value,
    page_size: int,
):
    """Pair SWA host slots with their trailing page hashes.

    A PREFETCH-loaded SWA host node can cover only the trailing portion of a
    multi-page Full node, while ``node.hash_value`` still describes the full
    node. Storage writes must pass one key per host page, so align to the tail
    of the hash chain before optional trimming.
    """
    if page_size <= 0:
        return host_value, list(hash_value)
    n_host_pages = (len(host_value) + page_size - 1) // page_size
    if n_host_pages <= 0:
        return host_value[:0], []
    if n_host_pages >= len(hash_value):
        return host_value, list(hash_value)
    keep_tokens = n_host_pages * page_size
    return host_value[-keep_tokens:], list(hash_value)[-n_host_pages:]


def _page_align_trailing_value(value: torch.Tensor, page_size: int) -> torch.Tensor:
    if page_size <= 1:
        return value
    aligned_len = (len(value) // page_size) * page_size
    if aligned_len <= 0:
        return value[:0]
    return value[-aligned_len:]


def _split_swa_trailing_host_value_on_node_split(
    host_value: torch.Tensor,
    old_node_len: int,
    split_len: int,
):
    """Split SWA host slots when a radix node is split.

    SWA host data may cover only the trailing suffix of a node after L3
    PREFETCH.  In that case, splitting the tensor from offset 0 would attach
    tail KV to the wrong prefix node.  Treat ``host_value`` as covering
    ``[old_node_len - len(host_value), old_node_len)`` and split by interval
    intersection with the new parent/child key ranges.
    """
    if host_value is None or len(host_value) == 0:
        return None, None

    host_len = min(len(host_value), old_node_len)
    if host_len <= 0:
        return None, None
    if host_len < len(host_value):
        host_value = host_value[-host_len:]

    covered_start = old_node_len - host_len
    split_len = max(0, min(split_len, old_node_len))

    parent_len = max(0, split_len - covered_start)
    parent_host = host_value[:parent_len].clone() if parent_len > 0 else None

    child_start = max(split_len, covered_start)
    child_offset = child_start - covered_start
    child_len = old_node_len - child_start
    child_host = (
        host_value[child_offset : child_offset + child_len].clone()
        if child_len > 0
        else None
    )

    return parent_host, child_host


def _split_swa_trailing_device_value_on_node_split(
    value: torch.Tensor,
    old_node_len: int,
    split_len: int,
):
    """Split SWA device slots, preserving tail-only coverage.

    Normal in-tree SWA device values cover the full node and behave like a
    plain prefix split. L3 LOAD_BACK can restore only a node's trailing SWA
    window, so the same interval split used for host_value must be applied to
    avoid attaching tail slots to older prefix nodes.
    """
    return _split_swa_trailing_host_value_on_node_split(value, old_node_len, split_len)


def _node_page_count(node, page_size: int) -> int:
    if node.hash_value is not None:
        return len(node.hash_value)
    key_len = len(node.key) if node.key is not None else 0
    return (key_len + page_size - 1) // page_size


def _swa_node_has_descendant_leaf_within_window(
    node,
    n_window_pages: int,
    page_size: int,
) -> bool:
    """Return whether this node can contribute SWA pages for a descendant leaf.

    The SWA storage prefetch path reads only the trailing window pages of the
    matched prefix. A node can contribute to that trailing window only if some
    descendant leaf ends fewer than ``n_window_pages`` pages after the current
    node. If every descendant leaf is at least a full SWA window deeper, the
    current node's SWA pages are too old for that leaf and can be skipped as an
    optional write-reduction optimization.

    This is intentionally conservative: leaf nodes are always considered useful,
    and traversal is bounded by page distance rather than unbounded tree depth.
    """
    if n_window_pages <= 0:
        return True
    children = getattr(node, "children", None)
    if not children:
        return True

    queue = []
    for child in children.values():
        pages_after = _node_page_count(child, page_size)
        if pages_after < n_window_pages:
            queue.append((child, pages_after))

    cursor = 0
    while cursor < len(queue):
        cur, pages_after = queue[cursor]
        cursor += 1
        cur_children = getattr(cur, "children", None)
        if not cur_children:
            return True

        for child in cur_children.values():
            next_pages = pages_after + _node_page_count(child, page_size)
            if next_pages < n_window_pages:
                queue.append((child, next_pages))

    return False


class SWAComponent(TreeComponent):
    """Sliding window attention component.

    Each SWA node stores translated SWA pool indices as its component
    value, independent of the full attention indices on the same tree node.
    When SWA data is evicted from an internal node the node is tombstoned
    — its SWA component value becomes None while the full attention
    value stays intact.
    """

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        from sglang.srt.mem_cache.cp_sharded_allocator import (
            unwrap_cp_sharded_allocator,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

        assert isinstance(
            unwrap_cp_sharded_allocator(cache.token_to_kv_pool_allocator),
            SWATokenToKVPoolAllocator,
        ), f"SWAComponent requires SWATokenToKVPoolAllocator, got {type(cache.token_to_kv_pool_allocator)}"
        super().__init__(cache, params)
        self.sliding_window_size = params.sliding_window_size
        # HiCache state: set to host SWA pool when HiCache enabled
        self._swa_kv_pool_host = None

    component_type = ComponentType.SWA
    HICACHE_OFFSETS_KEY = "swa_hicache_offsets"

    def _full_cp_meta(self, node: UnifiedTreeNode) -> Optional[dict]:
        try:
            full_data = node.component_data[BASE_COMPONENT_TYPE]
        except (IndexError, KeyError):
            return None
        return getattr(full_data, "metadata", {}).get(FullComponent.CP_HICACHE_META_KEY)

    def _logical_len(self, node: UnifiedTreeNode) -> int:
        meta = self._full_cp_meta(node)
        if meta is not None:
            return int(meta["logical_len"])
        cd = node.component_data[self.component_type]
        value = cd.value if cd.value is not None else cd.host_value
        return len(value) if value is not None else 0

    def _host_covered_len(self, node: UnifiedTreeNode) -> int:
        cd = node.component_data[self.component_type]
        if cd.host_value is None:
            return 0
        valid_offsets = getattr(cd, "metadata", {}).get(self.HICACHE_OFFSETS_KEY)
        if valid_offsets is not None:
            node_len = min(self._logical_len(node), len(node.key))
            owned_offsets = self._owned_offsets(node)
            if owned_offsets is None:
                owned_offsets = valid_offsets
            owned_offsets = owned_offsets.to(device="cpu", dtype=torch.int64)
            valid_offsets = valid_offsets.to(device="cpu", dtype=torch.int64)
            if owned_offsets.numel() == 0:
                return node_len

            missing = owned_offsets[
                ~torch.isin(owned_offsets, valid_offsets, assume_unique=False)
            ]
            if missing.numel() == 0:
                return node_len
            return max(0, node_len - int(missing.max().item()) - 1)
        return min(len(cd.host_value), len(node.key))

    def _owned_offsets(self, node: UnifiedTreeNode) -> Optional[torch.Tensor]:
        meta = self._full_cp_meta(node)
        if meta is None:
            return None
        return meta["owned_logical_offsets"].to(device="cpu", dtype=torch.int64)

    def _compact_swa_value(
        self,
        node: UnifiedTreeNode,
        value: torch.Tensor,
        cp_plan: Optional[CPHiCacheTransferPlan] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = (
            cp_plan.owned_logical_offsets
            if cp_plan is not None
            else self._owned_offsets(node)
        )
        if offsets is None:
            flat = value.reshape(-1).to(torch.int64)
            valid = flat > 0
            valid_offsets = (
                valid.nonzero(as_tuple=False)
                .flatten()
                .to(device="cpu", dtype=torch.int64)
            )
            return flat[valid], valid_offsets
        if offsets.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=value.device),
                torch.empty((0,), dtype=torch.int64),
            )
        compact_offsets = offsets.to(device=value.device)
        compact = value.reshape(-1)[compact_offsets].to(torch.int64)
        valid = compact > 0
        valid_offsets = offsets[valid.to(device=offsets.device)].to(
            device="cpu", dtype=torch.int64
        )
        return compact[valid], valid_offsets

    def _translate_full_to_swa(self, full_indices: torch.Tensor) -> torch.Tensor:
        return self.cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            full_indices
        )

    def _restore_device_value(self, node: UnifiedTreeNode, value: torch.Tensor) -> None:
        ct = self.component_type
        node.component_data[ct].value = value
        host_lru = self.cache.host_lru_lists[ct]
        if host_lru.in_list(node):
            host_lru.remove_node(node)
        self.cache.lru_lists[ct].insert_mru(node)
        self.cache.component_evictable_size_[ct] += len(value)

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        sliding_window_size = self.sliding_window_size
        ct = self.component_type
        state = {"prefix_len": 0, "covered_tail_len": 0}

        def validator(node: UnifiedTreeNode) -> bool:
            cd = node.component_data[ct]
            node_len = len(node.key)
            state["prefix_len"] += node_len

            # HiCache: a host-only tombstone is a valid match boundary too
            # — load_back will restore SWA from host before use.
            if cd.value is None and (match_device_only or cd.host_value is None):
                state["covered_tail_len"] = 0
                return False

            # Device-resident SWA covers the whole node key. Host-only SWA
            # may cover only the trailing window (PREFETCH from L3 stores
            # at most `sliding_window_size` tokens per node — earlier
            # tokens of a multi-page node have no SWA KV). Track only the
            # contiguous SWA-covered suffix of the matched prefix. A boundary
            # is valid when that suffix covers the full shorter prefix, or the
            # full sliding window for longer prefixes.
            if cd.value is not None:
                covered_len = node_len
            else:
                covered_len = self._host_covered_len(node)

            if covered_len >= node_len:
                state["covered_tail_len"] += node_len
            else:
                state["covered_tail_len"] = covered_len
            state["covered_tail_len"] = min(
                state["covered_tail_len"], state["prefix_len"]
            )
            required_len = min(state["prefix_len"], sliding_window_size)
            return state["covered_tail_len"] >= required_len

        return validator

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        # NOTE: this method only signals "presence of host-only ancestors"
        # via host_hit_length=max(prev, 1) — a sentinel, not a count.
        # The actual host-token count must be set by FullComponent's
        # finalize_match_result, which runs *before* this one because the
        # tree iterates components in ComponentType order (FULL=0, SWA=1
        # — see _components_tuple in UnifiedRadixCache). Reordering
        # components in that tuple would silently reduce host_hit_length
        # to 1 here and break init_load_back's prefix promotion. If you
        # ever change component ordering, update this method to compute
        # an actual count instead of relying on the upstream value.
        # TODO(ispobock): refactor host_hit_length usage
        ct = self.component_type
        n_swa = 0
        node = result.best_match_node
        root = self.cache.root_node
        while node is not root and n_swa < self.sliding_window_size:
            cd = node.component_data[ct]
            if cd.value is None and cd.host_value is not None:
                return result._replace(host_hit_length=max(result.host_hit_length, 1))
            if cd.value is not None:
                n_swa += len(cd.value)
            elif cd.host_value is not None:
                n_swa += len(cd.host_value)
            else:
                break
            node = node.parent
        return result

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> int:
        if params.prev_prefix_len >= total_prefix_len + prefix_len:
            return prefix_len

        is_tombstone = node.component_data[self.component_type].value is None
        if not is_tombstone:
            return prefix_len

        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            node.component_data[self.component_type].lock_ref == 0
        ), f"tombstone {self.component_type} lock_ref should be 0, node {node.id}"
        assert (
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{self.component_type}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        if swa_evicted_seqlen <= total_prefix_len:
            # Branch 1: entire value_slice is within SWA window — recover
            self.cache.token_to_kv_pool_allocator.free(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[BASE_COMPONENT_TYPE].value = value_slice.clone()
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            self._restore_device_value(node, swa_value)
            return 0
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            # Branch 2: value_slice[start_idx:] is within SWA window — partial recover
            start_idx = swa_evicted_seqlen - total_prefix_len
            self.cache.token_to_kv_pool_allocator.free(
                node.component_data[BASE_COMPONENT_TYPE].value[start_idx:]
            )
            self.cache._split_node(node.key, node, start_idx)
            node.component_data[BASE_COMPONENT_TYPE].value = value_slice[
                start_idx:
            ].clone()
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            self._restore_device_value(node, swa_value)
            return start_idx
        else:
            # Branch 3: entire value_slice is outside SWA window — not consumed
            return prefix_len

    def should_skip_leaf_creation(
        self, total_prefix_len: int, key_len: int, params: InsertParams
    ) -> bool:
        return params.swa_evicted_seqlen >= total_prefix_len + key_len

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
    ) -> None:
        # _unevict_node_on_insert already wrote the request's fresh KV slice
        # into the base value. We just need to rebuild SWA from that slice for
        # the in-window portion. There is no old SWA slot to free here.
        ct = self.component_type
        if node.component_data[ct].value is not None:
            return
        assert (
            node.component_data[ct].lock_ref == 0
        ), f"tombstone {ct} lock_ref should be 0 on unevict, node {node.id}"
        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{ct}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        full_value = node.component_data[BASE_COMPONENT_TYPE].value
        if swa_evicted_seqlen <= total_prefix_len:
            swa_value = self._translate_full_to_swa(full_value)
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            start_idx = swa_evicted_seqlen - total_prefix_len
            self.cache._split_node(node.key, node, start_idx)
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            swa_value = self._translate_full_to_swa(full_value)
        else:
            return
        self._restore_device_value(node, swa_value)

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        if not is_new_leaf:
            return

        node_start = result.prefix_len
        split_pos = params.swa_evicted_seqlen - node_start

        if split_pos <= 0:
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)
        elif split_pos < len(node.key):
            # Node straddles the SWA eviction boundary
            # Split into parent (tombstone, no SWA) and child (with SWA)
            # After _split_node, `node` becomes the child
            self.cache._split_node(node.key, node, split_pos)
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        new_parent.component_data[self.component_type].lock_ref = child.component_data[
            self.component_type
        ].lock_ref

        child_swa_value = child.component_data[self.component_type].value
        if child_swa_value is not None:
            split_len = len(new_parent.key)
            old_node_len = split_len + len(child.key)
            parent_value, child_value = _split_swa_trailing_device_value_on_node_split(
                child_swa_value, old_node_len, split_len
            )
            new_parent.component_data[self.component_type].value = parent_value
            child.component_data[self.component_type].value = child_value
        else:
            new_parent.component_data[self.component_type].value = None

        child_swa_host_value = child.component_data[self.component_type].host_value
        if child_swa_host_value is not None:
            child_host_lock_ref = child.component_data[
                self.component_type
            ].host_lock_ref
            split_len = len(new_parent.key)
            child_offsets_meta = child.component_data[self.component_type].metadata.get(
                self.HICACHE_OFFSETS_KEY
            )
            if child_offsets_meta is not None:
                child_offsets_meta = child_offsets_meta.to(
                    device="cpu", dtype=torch.int64
                )
                parent_mask = child_offsets_meta < split_len
                child_mask = ~parent_mask
                parent_len = int(parent_mask.sum().item())
                child_len = int(child_mask.sum().item())
                parent_host_value = child_swa_host_value[:parent_len].clone()
                child_host_value = child_swa_host_value[
                    parent_len : parent_len + child_len
                ].clone()
                padded_tail = child_swa_host_value[parent_len + child_len :]
                if padded_tail.numel() > 0 and self._swa_kv_pool_host is not None:
                    self._swa_kv_pool_host.free(padded_tail)
                new_parent.component_data[self.component_type].host_value = (
                    parent_host_value
                )
                child.component_data[self.component_type].host_value = child_host_value
                new_parent.component_data[self.component_type].metadata[
                    self.HICACHE_OFFSETS_KEY
                ] = child_offsets_meta[parent_mask].clone()
                child.component_data[self.component_type].metadata[
                    self.HICACHE_OFFSETS_KEY
                ] = (child_offsets_meta[child_mask] - split_len).clone()
            else:
                old_node_len = split_len + len(child.key)
                parent_host_value, child_host_value = (
                    _split_swa_trailing_host_value_on_node_split(
                        child_swa_host_value, old_node_len, split_len
                    )
                )
                new_parent.component_data[self.component_type].host_value = (
                    parent_host_value
                )
                child.component_data[self.component_type].host_value = child_host_value
            new_parent.component_data[self.component_type].host_lock_ref = (
                child_host_lock_ref if parent_host_value is not None else 0
            )
            child.component_data[self.component_type].host_lock_ref = (
                child_host_lock_ref if child_host_value is not None else 0
            )
            host_lru = self.cache.host_lru_lists[self.component_type]
            if (
                parent_host_value is not None
                and new_parent.component_data[self.component_type].value is None
                and new_parent.component_data[self.component_type].host_lock_ref == 0
            ):
                host_lru.insert_mru(new_parent)
            if (
                child_host_value is not None
                and child.component_data[self.component_type].value is None
                and child.component_data[self.component_type].host_lock_ref == 0
                and not host_lru.in_list(child)
            ):
                host_lru.insert_mru(child)
            if (
                child_host_value is None
                or child.component_data[self.component_type].host_lock_ref > 0
            ) and host_lru.in_list(child):
                host_lru.remove_node(child)

        # parent inherits the swa_uuid from child for swa lock ref
        new_parent.component_data[self.component_type].metadata["uuid"] = (
            child.component_data[self.component_type].metadata.get("uuid")
        )
        child.component_data[self.component_type].metadata.pop("uuid", None)
        host_uuid = child.component_data[self.component_type].metadata.get("host_uuid")
        if host_uuid is not None:
            if new_parent.component_data[self.component_type].host_lock_ref > 0:
                new_parent.component_data[self.component_type].metadata[
                    "host_uuid"
                ] = host_uuid
                child.component_data[self.component_type].metadata.pop(
                    "host_uuid", None
                )
            elif child.component_data[self.component_type].host_lock_ref == 0:
                child.component_data[self.component_type].metadata.pop(
                    "host_uuid", None
                )

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        ct = self.component_type
        cd = node.component_data[ct]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            # Pass full indices to free_swa so slots with no SWA pair are
            # skipped. Freeing swa_value directly would double free those
            # entries since they all map to the same sentinel slot.
            self.cache.token_to_kv_pool_allocator.free_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            freed = len(cd.value)
            self.cache.component_evictable_size_[ct] -= freed
            cd.value = None

        # Host layer
        host_lru = self.cache.host_lru_lists[ct]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            if self._swa_kv_pool_host is not None:
                self._swa_kv_pool_host.free(cd.host_value)
            cd.host_value = None
            if host_lru.in_list(node):
                host_lru.remove_node(node)

        # After device tombstone: if host_value remains, move into host LRU
        if (
            target is EvictLayer.DEVICE
            and cd.value is None
            and cd.host_value is not None
        ):
            if not host_lru.in_list(node):
                host_lru.insert_mru(node)

        return freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 1

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.swa_num_tokens
        ct = self.component_type
        lru = self.cache.lru_lists[ct]
        x = lru.get_lru_no_lock()
        while tracker[ct] < request and x is not None and lru.in_list(x):
            assert x.component_data[ct].value is not None
            if x in self.cache.evictable_device_leaves:
                # D-leaf: atomic eviction of all components
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_device_leaf(x, tracker)
                if not lru.in_list(x_next):
                    x_next = lru.get_lru_no_lock()
                x = x_next
            else:
                # Internal: tombstone SWA + cascade
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.DEVICE, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type
        root = self.cache.root_node
        sliding_window_size = self.sliding_window_size
        swa_lock_size = 0
        swa_uuid_for_lock = None
        host_lru = self.cache.host_lru_lists[ct] if lock_host else None

        # Tombstoned nodes (cd.value is None) have no SWA chunk to protect
        # skip them and keep walking up. This path is hit when HiCache
        # backs up a FULL present internal node whose SWA was already evicted.
        cur = node
        while cur != root and swa_lock_size < sliding_window_size:
            comp = cur.component_data[ct]
            value = comp.host_value if lock_host else comp.value
            if value is None:
                result.skip_lock_node_ids.setdefault(ct, set()).add(cur.id)
                cur = cur.parent
                continue

            if lock_host:
                if comp.host_lock_ref == 0 and host_lru.in_list(cur):
                    host_lru.remove_node(cur)
                comp.host_lock_ref += 1
                self.cache._update_evictable_leaf_sets(cur)
                swa_lock_size += min(len(value), len(cur.key))
            else:
                if comp.lock_ref == 0:
                    key_len = len(comp.value)
                    self.cache.component_evictable_size_[ct] -= key_len
                    self.cache.component_protected_size_[ct] += key_len
                comp.lock_ref += 1
                swa_lock_size += len(cur.key)

            if swa_lock_size >= sliding_window_size:
                metadata_key = "host_uuid" if lock_host else "uuid"
                if comp.metadata.get(metadata_key) is None:
                    comp.metadata[metadata_key] = next_component_uuid()
                swa_uuid_for_lock = comp.metadata[metadata_key]
            cur = cur.parent

        if lock_host:
            result.swa_uuid_for_host_lock = swa_uuid_for_lock
        else:
            result.swa_uuid_for_lock = swa_uuid_for_lock
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        root = self.cache.root_node
        swa_uuid_for_lock = (
            params.swa_uuid_for_host_lock
            if lock_host and params
            else params.swa_uuid_for_lock if params else None
        )
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        dec_swa = True
        host_lru = self.cache.host_lru_lists[ct] if lock_host else None

        # A node in skip_lock_node_ids was a tombstone when this lock was acquired.
        cur = node
        while cur != root and dec_swa:
            comp = cur.component_data[ct]
            if cur.id in skip_lock_node_ids:
                cur = cur.parent
                continue

            if lock_host:
                if comp.host_lock_ref == 0:
                    cur = cur.parent
                    continue
                comp.host_lock_ref -= 1
                if (
                    comp.host_lock_ref == 0
                    and comp.value is None
                    and comp.host_value is not None
                    and not host_lru.in_list(cur)
                ):
                    host_lru.insert_mru(cur)
                self.cache._update_evictable_leaf_sets(cur)
                metadata_key = "host_uuid"
            else:
                if comp.lock_ref == 0:
                    cur = cur.parent
                    continue
                if comp.lock_ref == 1:
                    key_len = len(comp.value)
                    self.cache.component_evictable_size_[ct] += key_len
                    self.cache.component_protected_size_[ct] -= key_len
                comp.lock_ref -= 1
                metadata_key = "uuid"

            if (
                swa_uuid_for_lock
                and comp.metadata.get(metadata_key) == swa_uuid_for_lock
            ):
                dec_swa = False
            cur = cur.parent

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        if is_finished:
            insert_params.swa_evicted_seqlen = req.swa_evicted_seqlen
        return None

    # ---- HiCache Hooks ----

    def build_hicache_transfers(
        self, node: UnifiedTreeNode, phase: CacheTransferPhase, **kw
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            cd = node.component_data[ct]
            if cd.value is None:
                return None
            cp_plan = kw.get("cp_plan")
            # cd.value already holds SWA-pool indices (translated at insert time).
            if cp_plan is not None or self._owned_offsets(node) is not None:
                compact_value, valid_offsets = self._compact_swa_value(
                    node, cd.value, cp_plan=cp_plan
                )
                device_indices = compact_value
                allocator = getattr(self.cache, "token_to_kv_pool_allocator", None)
                if hasattr(allocator, "_pad_transfer_slots"):
                    device_indices = allocator._pad_transfer_slots(compact_value)
                transfer_cp_plan = CPHiCacheTransferPlan(
                    logical_start=(
                        int(cp_plan.logical_start) if cp_plan is not None else 0
                    ),
                    logical_len=self._logical_len(node),
                    full_device_indices=cd.value,
                    owned_device_indices=device_indices,
                    owned_logical_offsets=valid_offsets,
                )
            else:
                # Host pools allocate whole pages. SWA values can cover only a
                # trailing suffix after chunked generation, so keep the newest
                # page-aligned suffix instead of padding missing device slots.
                device_indices = _page_align_trailing_value(
                    cd.value, self.cache.page_size
                ).to(torch.int64)
                if len(device_indices) == 0:
                    return None
                transfer_cp_plan = None
            # Host pool indexing wants int64.
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=device_indices,
                    cp_plan=transfer_cp_plan,
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:
            # `node` is best_match_node; the SWA validator guarantees every
            # ancestor within `sliding_window_size` has value or host_value.
            n_swa = 0
            backed_up: list[torch.Tensor] = []
            nodes: list = []
            cur = node
            while cur is not self.cache.root_node and n_swa < self.sliding_window_size:
                cd = cur.component_data[ct]
                if cd.value is not None:
                    # device exists, skip it
                    n_swa += self._logical_len(cur)
                elif cd.host_value is not None:
                    # host only, collect it
                    backed_up.append(cd.host_value)
                    nodes.append(cur)
                    n_swa += self._host_covered_len(cur)
                else:
                    assert self._full_cp_meta(node) is not None
                    break
                cur = cur.parent

            if not backed_up:
                return None

            backed_up.reverse()
            nodes.reverse()

            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.cat(backed_up),
                    device_indices=None,
                    nodes_to_load=nodes,
                )
            ]

        if phase == CacheTransferPhase.BACKUP_STORAGE:
            # Persist this node's SWA host indices to L3, keyed by the
            # node's per-page hashes. Only nodes with live host_value
            # (i.e. still in the host LRU window) reach here; nodes
            # whose SWA host data was already evicted have host_value
            # cleared and short-circuit above.
            if not envs.SGLANG_HICACHE_SWA_STORAGE_ENABLE.get():
                return None
            if self._swa_kv_pool_host is None:
                return None
            cd = node.component_data[ct]
            if cd.host_value is None or not node.hash_value:
                return None
            page_size = self.cache.page_size
            n_window_pages = (self.sliding_window_size + page_size - 1) // page_size
            # Store one guard page beyond the active SWA window. The request
            # match path uses input_len - 1 and then page-aligns down; when a
            # prompt length is exactly page-aligned, replay prefetches a prefix
            # that ends one page before the backed-up node. Without this guard,
            # that prefix is missing the older of its two trailing SWA pages.
            n_storage_pages = n_window_pages + 1
            if (
                envs.SGLANG_HICACHE_SWA_STORAGE_BOUNDED_BFS_ENABLE.get()
                and not _swa_node_has_descendant_leaf_within_window(
                    node, n_storage_pages, page_size
                )
            ):
                return None
            # Trailing-window pruning: a node carries SWA host_value for
            # all of its pages, but PREFETCH later only ever asks for the
            # trailing `sliding_window_size` tokens (TRAILING_PAGES policy
            # in PREFETCH's build_hicache_transfers). Pages older than
            # that will never be read back, so persisting them to L3 is
            # pure waste. Slice host_value + hash_value to the last
            # `n_storage_pages` so the controller writes one file per page
            # for at most the active window plus the guard page — independent
            # of the node's total page count.
            host_indices, keys = _align_swa_keys_to_host_pages(
                cd.host_value, node.hash_value, page_size
            )
            if envs.SGLANG_HICACHE_SWA_STORAGE_TAIL_TRIM_ENABLE.get():
                host_indices, keys = _trim_swa_to_trailing_window(
                    host_indices, keys, n_storage_pages, page_size
                )
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=host_indices,
                    keys=keys,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        if phase == CacheTransferPhase.PREFETCH:
            # Allocate host slots for the trailing window pages of the
            # prefetch range, request them from L3 via TRAILING_PAGES
            # policy. Only the trailing `sliding_window_size` tokens
            # are useful for SWA — earlier pages will never be read by
            # any new token's SWA attention.
            if not envs.SGLANG_HICACHE_SWA_STORAGE_ENABLE.get():
                return None
            if self._swa_kv_pool_host is None:
                return None
            page_size = self.cache.page_size
            prefetch_tokens = int(kw.get("prefetch_tokens", 0))
            if prefetch_tokens <= 0 or page_size <= 0:
                return None
            # Ceil division: ensure we cover the full window, not (window-1)
            # tokens. With sliding_window_size=127 (e.g. GPT-OSS, where
            # `get_attention_sliding_window_size = config.sliding_window - 1`)
            # and page_size=64, floor division would give 1 page = 64 tokens,
            # which leaves 63 in-window positions without SWA KV after a
            # restart-and-prefetch — the validator (which requires enough
            # contiguous trailing SWA coverage) would correctly reject this as
            # insufficient coverage and we'd take a cold prefill.
            n_window_pages = (self.sliding_window_size + page_size - 1) // page_size
            n_pages = min(prefetch_tokens // page_size, n_window_pages)
            if n_pages <= 0:
                return None
            swa_host_size = n_pages * page_size
            host_indices = self._swa_kv_pool_host.alloc(swa_host_size)
            if host_indices is None:
                self.cache.evict_host(swa_host_size, ct)
                host_indices = self._swa_kv_pool_host.alloc(swa_host_size)
            if host_indices is None:
                # Signal alloc failure to caller (sentinel: empty list).
                return []
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=host_indices,
                    keys=["__placeholder__"] * n_pages,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        **kw,
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                xfer = transfers[0]
                cd = node.component_data[ct]
                if cd.host_value is None:
                    cd.host_value = xfer.host_indices.clone()
                    cp_plan = xfer.cp_plan
                    if cp_plan is not None:
                        cd.metadata[self.HICACHE_OFFSETS_KEY] = (
                            cp_plan.owned_logical_offsets.to(
                                device="cpu", dtype=torch.int64
                            ).clone()
                        )
            return

        if phase == CacheTransferPhase.LOAD_BACK:
            assert transfers and transfers[0].device_indices is not None
            xfer = transfers[0]
            device_indices = xfer.device_indices
            allocator = self.cache.token_to_kv_pool_allocator

            offset = 0
            for original_node in xfer.nodes_to_load or []:
                n = original_node
                cd_n = n.component_data[ct]
                cd_full_n = n.component_data[BASE_COMPONENT_TYPE]
                valid_offsets = cd_n.metadata.get(self.HICACHE_OFFSETS_KEY)
                compact = None
                if valid_offsets is not None:
                    valid_offsets = valid_offsets.to(device="cpu", dtype=torch.int64)
                    host_len = len(cd_n.host_value)
                    n_tokens = int(valid_offsets.numel())
                    compact = device_indices[offset : offset + host_len]
                    swa_chunk = torch.zeros_like(cd_full_n.value)
                    if n_tokens > 0:
                        swa_chunk[valid_offsets.to(device=swa_chunk.device)] = compact[
                            :n_tokens
                        ].to(device=swa_chunk.device)
                    offset += host_len
                else:
                    offsets = self._owned_offsets(n)
                    if offsets is None:
                        n_tokens = len(cd_n.host_value)
                        compact = device_indices[offset : offset + n_tokens]
                        swa_chunk = compact.clone()
                        if n_tokens < len(n.key):
                            split_len = len(n.key) - n_tokens
                            self.cache._split_node(n.key, n, split_len)
                            cd_n = n.component_data[ct]
                            cd_full_n = n.component_data[BASE_COMPONENT_TYPE]
                            assert len(n.key) == n_tokens
                        offset += n_tokens
                    else:
                        n_tokens = int(offsets.numel())
                        host_len = len(cd_n.host_value)
                        compact = device_indices[offset : offset + host_len]
                        swa_chunk = torch.zeros_like(cd_full_n.value)
                        if n_tokens > 0:
                            swa_chunk[offsets.to(device=swa_chunk.device)] = compact[
                                :n_tokens
                            ].to(device=swa_chunk.device)
                        offset += host_len
                self._restore_device_value(n, swa_chunk)
                assert compact is not None
                assert cd_full_n.value is not None
                assert len(cd_full_n.value) == len(swa_chunk)
                allocator.set_full_to_swa_mapping(cd_full_n.value, swa_chunk)
            assert offset == len(xfer.host_indices)
            return

        if phase == CacheTransferPhase.BACKUP_STORAGE:
            # No tree state to update — backup just persists existing host data.
            return

        if phase == CacheTransferPhase.PREFETCH:
            if not transfers:
                return
            transfer = transfers[0]
            host_indices = transfer.host_indices
            insert_result = kw.get("insert_result")
            pool_storage_result = kw.get("pool_storage_result")
            loaded_pages = (
                pool_storage_result.extra_pool_hit_pages.get(PoolName.SWA, 0)
                if pool_storage_result is not None
                else 0
            )
            target_node = (
                insert_result.inserted_host_node if insert_result is not None else None
            )
            page_size = self.cache.page_size
            loaded_tokens = loaded_pages * page_size

            # Bail conditions: alloc failed, no insert target, no pages
            # loaded, target already has SWA host data, the loaded range
            # exceeds the target node, OR L3 returned fewer SWA tokens than
            # the shorter of the matched prefix and the sliding window. The
            # window-coverage requirement is correctness-critical: if attached,
            # the validator marks this node as a valid SWA boundary. A partial
            # trailing fill leaves some in-window positions without SWA KV,
            # which would silently corrupt SWA-layer attention. Better to bail
            # and let the request recompute the SWA layer than to attach an
            # under-covered host_value.
            #
            # `loaded_tokens > len(target_node.key)` is a structural
            # impossibility (we requested at most `n_window_pages * page_size`
            # and target_node was sized to the full prefetch_key). It is
            # kept as a defensive assertion-equivalent.
            required_tokens = (
                min(len(target_node.key), self.sliding_window_size)
                if target_node is not None
                else self.sliding_window_size
            )
            under_covered = loaded_pages > 0 and loaded_tokens < required_tokens
            if loaded_pages <= 0:
                logger.warning(
                    "SWA prefetch found no storage pages: target_node=%s "
                    "target_len=%s requested_host_tokens=%s sliding_window=%d",
                    getattr(target_node, "id", None),
                    len(target_node.key) if target_node is not None else None,
                    host_indices.numel() if host_indices is not None else None,
                    self.sliding_window_size,
                )
            if under_covered:
                # Window-coverage shortfall is silent in metrics — surface it
                # so ops can correlate cold-prefill spikes with L3 partial
                # eviction or BACKUP_STORAGE drops.
                logger.warning(
                    "SWA prefetch under-covered window: loaded_tokens=%d "
                    "required_tokens=%d sliding_window=%d - bailing to cold "
                    "prefill for this node",
                    loaded_tokens,
                    required_tokens,
                    self.sliding_window_size,
                )
            if (
                host_indices is None
                or target_node is None
                or loaded_pages <= 0
                or target_node.component_data[ct].host_value is not None
                or loaded_tokens > len(target_node.key)
                or under_covered
            ):
                self.cache.cache_controller.append_host_mem_release(
                    extra_pools=[transfer]
                )
                return

            target_node.component_data[ct].host_value = host_indices[
                :loaded_tokens
            ].clone()
            logger.info(
                "SWA prefetch attached host pages: target_node=%s "
                "loaded_pages=%d loaded_tokens=%d required_tokens=%d",
                getattr(target_node, "id", None),
                loaded_pages,
                loaded_tokens,
                required_tokens,
            )
            if target_node.component_data[ct].value is None:
                host_lru = self.cache.host_lru_lists[ct]
                if not host_lru.in_list(target_node):
                    host_lru.insert_mru(target_node)
            return

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict SWA host resources.
        Internal nodes: private tombstone (free SWA host only).
        Host leaves: atomic eviction via _evict_host_leaf."""
        ct = self.component_type
        host_lru = self.cache.host_lru_lists[ct]
        x = host_lru.get_lru_no_lock()
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):
            x_next = host_lru.get_prev_no_lock(x)
            cd = x.component_data[ct]
            if x in self.cache.evictable_host_leaves:
                self.cache._evict_host_leaf(x, tracker)
            else:
                assert cd.host_value is not None
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.HOST, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker, target=EvictLayer.HOST)
            x = x_next
