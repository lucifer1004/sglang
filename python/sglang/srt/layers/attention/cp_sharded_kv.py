from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.context_parallel import CPPrefillRuntimeLayout
from sglang.srt.mem_cache.cp_sharded_allocator import CPShardedKVPoolAllocator


@dataclass(frozen=True)
class CPShardedKVRegion:
    logical_page_table: torch.Tensor
    cache_seqlens: torch.Tensor


@dataclass(frozen=True)
class CPShardedKVPrefillPlan:
    prefix: CPShardedKVRegion
    extend: CPShardedKVRegion
    full_cache_seqlens: torch.Tensor
    total_page_columns: int


@dataclass(frozen=True)
class CPKVGatherSegmentPlan:
    sizes: tuple[int, ...]
    local_physical_slots: torch.Tensor
    rank_packed_to_logical: torch.Tensor
    logical_token_count: int


@dataclass(frozen=True)
class CPPrefillKVGatherPlan:
    prefix: CPKVGatherSegmentPlan
    extend: CPKVGatherSegmentPlan


@dataclass(frozen=True)
class CPKVSourcePushSegmentPlan:
    source_rows: torch.Tensor
    destination_rows: torch.Tensor


@dataclass(frozen=True)
class CPPrefillKVSourcePushPlan:
    prefix: CPKVSourcePushSegmentPlan
    extend: CPKVSourcePushSegmentPlan
    source_mask: int
    logical_token_count: int


@dataclass(frozen=True)
class CPCompactKVExchangePlan:
    local_send_indices: torch.Tensor
    send_sizes: tuple[int, ...]
    recv_sizes: tuple[int, ...]
    recv_packed_to_compact: torch.Tensor


@dataclass(frozen=True)
class CPPrefillSWAKVGatherPlan:
    prefix: CPCompactKVExchangePlan
    extend: CPCompactKVExchangePlan
    compact_token_count: int
    block_cu_seqlens_k: torch.Tensor
    block_k_lengths: tuple[int, ...]
    window_left: int


@dataclass(frozen=True)
class _CPCompactKVTargetInterval:
    destination: int
    logical_start: int
    token_count: int
    compact_start: int

    @property
    def logical_end(self) -> int:
        return self.logical_start + self.token_count


def should_use_full_cp_kv_collective(
    runtime_layout: CPPrefillRuntimeLayout,
    *,
    page_size: int,
) -> bool:
    """Allow bounded full participation when a short split leaves empty owners."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    owner_tokens_per_rank = tuple(
        int(count) for count in runtime_layout.spec.per_rank_tokens
    )
    cp_size = len(owner_tokens_per_rank)
    if cp_size == 0:
        raise ValueError("split ownership must contain one entry per CP rank")
    if any(count < 0 for count in owner_tokens_per_rank):
        raise ValueError("split ownership counts must be non-negative")
    if cp_size <= 1 or not any(count == 0 for count in owner_tokens_per_rank):
        return False

    logical_kv_tokens = (
        runtime_layout.spec.extend_start + runtime_layout.spec.extend_len
    )
    return logical_kv_tokens <= page_size * cp_size


class CPShardedKVPageTableResolver:
    """Resolve logical CP-sharded page tables for an attention backend."""

    def __init__(self, allocator):
        self._allocator = (
            allocator if isinstance(allocator, CPShardedKVPoolAllocator) else None
        )

    @property
    def enabled(self) -> bool:
        return self._allocator is not None

    def resolve_full(self, logical_page_table: torch.Tensor) -> torch.Tensor:
        if self._allocator is None:
            return logical_page_table
        return self._allocator.logical_page_table_to_physical(logical_page_table)

    def resolve_swa_pages(self, logical_full_pages: torch.Tensor) -> torch.Tensor:
        """Resolve logical full-cache pages directly to physical SWA pages."""
        if self._allocator is None:
            raise RuntimeError(
                "SWA page translation requires a CP-sharded KV allocator"
            )
        return self._allocator.logical_swa_page_table_to_physical(
            logical_full_pages
        )

    def resolve_swa_slots(self, full_physical_slots: torch.Tensor) -> torch.Tensor:
        if self._allocator is None:
            raise RuntimeError(
                "SWA slot translation requires a CP-sharded KV allocator"
            )
        resolved = self._allocator.translate_loc_from_full_to_swa(
            full_physical_slots
        )
        return torch.where(
            full_physical_slots.ne(0),
            resolved.to(device=full_physical_slots.device),
            torch.zeros_like(resolved, device=full_physical_slots.device),
        )


def build_cp_prefill_kv_gather_plan(
    *,
    prefix_logical_slots: torch.Tensor,
    runtime_layout: CPPrefillRuntimeLayout,
    allocator,
) -> CPPrefillKVGatherPlan:
    """Build compact prefix and extend gather metadata once per prefill."""
    cp_size = len(runtime_layout.spec.per_rank_tokens)
    if getattr(allocator, "cp_size", None) != cp_size:
        raise ValueError("KV gather allocator CP size does not match the split spec")
    if getattr(allocator, "cp_rank", None) != runtime_layout.cp_rank:
        raise ValueError("KV gather allocator CP rank does not match runtime layout")

    prefix_logical_slots = prefix_logical_slots.reshape(-1).to(
        device=runtime_layout.local_extend_indices.device,
        dtype=torch.int64,
    )
    prefix_owner_plan = allocator.owner_plan_for_logical_slots(
        prefix_logical_slots
    )
    local_prefix_mask = prefix_owner_plan.owner_ranks == runtime_layout.cp_rank
    local_prefix_logical_slots = prefix_logical_slots[local_prefix_mask]
    local_prefix_physical_slots = allocator.logical_slots_to_physical(
        local_prefix_logical_slots
    )
    _assert_no_dummy_physical_slot(
        local_prefix_physical_slots,
        "CP-local cached prefix resolved to a dummy physical slot",
    )
    prefix = CPKVGatherSegmentPlan(
        sizes=prefix_owner_plan.per_rank_counts,
        local_physical_slots=local_prefix_physical_slots,
        rank_packed_to_logical=prefix_owner_plan.rank_packed_to_logical,
        logical_token_count=prefix_logical_slots.numel(),
    )

    extend_packed_parts = tuple(
        torch.arange(
            block.logical_start - runtime_layout.spec.extend_start,
            block.logical_start
            - runtime_layout.spec.extend_start
            + block.token_count,
            dtype=torch.int64,
            device=runtime_layout.local_extend_indices.device,
        )
        for rank in range(cp_size)
        for block in runtime_layout.spec.local_blocks(rank)
    )
    extend_packed_to_logical = (
        torch.cat(extend_packed_parts)
        if extend_packed_parts
        else torch.empty_like(runtime_layout.local_extend_indices)
    )
    extend = CPKVGatherSegmentPlan(
        sizes=runtime_layout.spec.per_rank_tokens,
        local_physical_slots=runtime_layout.local_out_cache_loc,
        rank_packed_to_logical=extend_packed_to_logical,
        logical_token_count=runtime_layout.spec.extend_len,
    )
    return CPPrefillKVGatherPlan(prefix=prefix, extend=extend)


def _to_source_push_indices(name: str, indices: torch.Tensor) -> torch.Tensor:
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must use an integer row dtype")
    return indices.to(dtype=torch.int32).contiguous()


def _local_rank_packed_rows(
    segment: CPKVGatherSegmentPlan, cp_rank: int
) -> torch.Tensor:
    cp_size = len(segment.sizes)
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError("CP rank is outside the source-push plan")
    if sum(segment.sizes) != segment.logical_token_count:
        raise ValueError("source-push segment sizes do not cover its logical rows")
    if segment.rank_packed_to_logical.numel() != segment.logical_token_count:
        raise ValueError("source-push reorder map does not cover its logical rows")
    local_offset = sum(segment.sizes[:cp_rank])
    local_count = segment.sizes[cp_rank]
    return segment.rank_packed_to_logical.narrow(0, local_offset, local_count)


def build_cp_prefill_kv_source_push_plan(
    *,
    gather_plan: CPPrefillKVGatherPlan,
    cp_rank: int,
) -> CPPrefillKVSourcePushPlan:
    """Build owner-local indexed rows for direct final-logical K/V writes."""
    prefix_logical_rows = _local_rank_packed_rows(gather_plan.prefix, cp_rank)
    prefix_source_rows = gather_plan.prefix.local_physical_slots.reshape(-1)
    if prefix_source_rows.numel() != prefix_logical_rows.numel():
        raise ValueError("local Prefix physical rows do not match its owner slice")

    extend_logical_rows = _local_rank_packed_rows(gather_plan.extend, cp_rank)
    extend_count = extend_logical_rows.numel()
    extend_source_rows = torch.arange(
        extend_count,
        dtype=torch.int32,
        device=extend_logical_rows.device,
    )
    prefix_count = gather_plan.prefix.logical_token_count
    logical_token_count = prefix_count + gather_plan.extend.logical_token_count
    if logical_token_count > torch.iinfo(torch.int32).max:
        raise ValueError("source-push logical rows exceed int32 capacity")

    cp_size = len(gather_plan.prefix.sizes)
    if len(gather_plan.extend.sizes) != cp_size:
        raise ValueError("Prefix and Extend source-push CP sizes differ")
    source_mask = 0
    for rank in range(cp_size):
        if gather_plan.prefix.sizes[rank] + gather_plan.extend.sizes[rank] > 0:
            source_mask |= 1 << rank
    if source_mask == 0:
        raise ValueError("source-push plan must contain at least one K/V source")

    return CPPrefillKVSourcePushPlan(
        prefix=CPKVSourcePushSegmentPlan(
            source_rows=_to_source_push_indices(
                "Prefix source rows", prefix_source_rows
            ),
            destination_rows=_to_source_push_indices(
                "Prefix destination rows", prefix_logical_rows
            ),
        ),
        extend=CPKVSourcePushSegmentPlan(
            source_rows=extend_source_rows,
            destination_rows=_to_source_push_indices(
                "Extend destination rows", extend_logical_rows + prefix_count
            ),
        ),
        source_mask=source_mask,
        logical_token_count=logical_token_count,
    )


def _empty_compact_exchange_plan(
    *, device: torch.device, cp_size: int
) -> CPCompactKVExchangePlan:
    empty = torch.empty((0,), dtype=torch.int64, device=device)
    return CPCompactKVExchangePlan(
        local_send_indices=empty,
        send_sizes=(0,) * cp_size,
        recv_sizes=(0,) * cp_size,
        recv_packed_to_compact=empty,
    )


def _slice_compact_targets_to_segment(
    *,
    targets: Sequence[_CPCompactKVTargetInterval],
    segment_start: int,
    segment_len: int,
) -> tuple[_CPCompactKVTargetInterval, ...]:
    segment_end = segment_start + segment_len
    sliced = []
    for target in targets:
        overlap_start = max(segment_start, target.logical_start)
        overlap_end = min(segment_end, target.logical_end)
        if overlap_start >= overlap_end:
            continue
        sliced.append(
            _CPCompactKVTargetInterval(
                destination=target.destination,
                logical_start=overlap_start - segment_start,
                token_count=overlap_end - overlap_start,
                compact_start=(
                    target.compact_start
                    + overlap_start
                    - target.logical_start
                ),
            )
        )
    return tuple(sliced)


def _cat_index_parts(
    parts: list[torch.Tensor], *, device: torch.device
) -> torch.Tensor:
    if parts:
        return torch.cat(parts)
    return torch.empty((0,), dtype=torch.int64, device=device)


def _build_compact_exchange_for_rank_packed_segment(
    *,
    segment: CPKVGatherSegmentPlan,
    targets: Sequence[_CPCompactKVTargetInterval],
    cp_rank: int,
    cp_size: int,
) -> CPCompactKVExchangePlan:
    """Build routes for an arbitrary cached-prefix owner layout.

    Rows inside each source's rank-packed slice are logically sorted. Query only
    the SWA target interval boundaries, then transfer those O(cp_size * blocks)
    bounds to CPU once for the NCCL counts. No full-prefix owner tensor is built.
    """
    device = segment.rank_packed_to_logical.device
    if not targets:
        return _empty_compact_exchange_plan(device=device, cp_size=cp_size)
    if sum(segment.sizes) != segment.logical_token_count:
        raise ValueError("segment sizes do not cover its logical rows")
    if segment.rank_packed_to_logical.numel() != segment.logical_token_count:
        raise ValueError("segment reorder map does not cover its logical rows")

    query_bounds = torch.tensor(
        [
            bound
            for target in targets
            for bound in (target.logical_start, target.logical_end)
        ],
        dtype=torch.int64,
        device=device,
    )
    source_rows = []
    source_bounds = []
    source_offset = 0
    for source, source_size in enumerate(segment.sizes):
        rows = segment.rank_packed_to_logical.narrow(
            0, source_offset, source_size
        )
        source_rows.append(rows)
        source_bounds.append(torch.searchsorted(rows, query_bounds))
        source_offset += source_size
    bounds_cpu = torch.stack(source_bounds).detach().cpu().tolist()

    route_counts = [[0] * cp_size for _ in range(cp_size)]
    for target_index, target in enumerate(targets):
        covered = 0
        for source in range(cp_size):
            start = int(bounds_cpu[source][2 * target_index])
            end = int(bounds_cpu[source][2 * target_index + 1])
            count = end - start
            route_counts[source][target.destination] += count
            covered += count
        if covered != target.token_count:
            raise RuntimeError("cached-prefix owners do not cover an SWA interval")

    local_send_parts = []
    for target_index, target in enumerate(targets):
        start = int(bounds_cpu[cp_rank][2 * target_index])
        end = int(bounds_cpu[cp_rank][2 * target_index + 1])
        if start != end:
            local_send_parts.append(
                torch.arange(start, end, dtype=torch.int64, device=device)
            )

    recv_reorder_parts = []
    for source in range(cp_size):
        for target_index, target in enumerate(targets):
            if target.destination != cp_rank:
                continue
            start = int(bounds_cpu[source][2 * target_index])
            end = int(bounds_cpu[source][2 * target_index + 1])
            if start == end:
                continue
            logical_rows = source_rows[source].narrow(0, start, end - start)
            recv_reorder_parts.append(
                logical_rows - target.logical_start + target.compact_start
            )

    send_sizes = tuple(route_counts[cp_rank])
    recv_sizes = tuple(
        route_counts[source][cp_rank] for source in range(cp_size)
    )
    local_send_indices = _cat_index_parts(local_send_parts, device=device)
    recv_packed_to_compact = _cat_index_parts(
        recv_reorder_parts, device=device
    )
    return CPCompactKVExchangePlan(
        local_send_indices=local_send_indices,
        send_sizes=send_sizes,
        recv_sizes=recv_sizes,
        recv_packed_to_compact=recv_packed_to_compact,
    )


def _build_compact_exchange_for_extend(
    *,
    segment: CPKVGatherSegmentPlan,
    runtime_layout: CPPrefillRuntimeLayout,
    targets: Sequence[_CPCompactKVTargetInterval],
) -> CPCompactKVExchangePlan:
    """Build extend routes directly from the page-aligned zigzag blocks."""
    device = segment.rank_packed_to_logical.device
    cp_size = len(runtime_layout.spec.per_rank_tokens)
    cp_rank = runtime_layout.cp_rank
    if not targets:
        return _empty_compact_exchange_plan(device=device, cp_size=cp_size)

    source_runs = []
    for source in range(cp_size):
        local_offset = 0
        runs = []
        for block in runtime_layout.spec.local_blocks(source):
            runs.append(
                (
                    block.logical_start - runtime_layout.spec.extend_start,
                    block.token_count,
                    local_offset,
                )
            )
            local_offset += block.token_count
        if local_offset != segment.sizes[source]:
            raise RuntimeError("extend owner blocks do not match gather sizes")
        source_runs.append(runs)

    route_counts = [[0] * cp_size for _ in range(cp_size)]
    for target in targets:
        covered = 0
        for source, runs in enumerate(source_runs):
            for run_start, run_count, _ in runs:
                overlap_start = max(target.logical_start, run_start)
                overlap_end = min(target.logical_end, run_start + run_count)
                count = max(0, overlap_end - overlap_start)
                route_counts[source][target.destination] += count
                covered += count
        if covered != target.token_count:
            raise RuntimeError("extend owners do not cover an SWA interval")

    local_send_parts = []
    for target in targets:
        for run_start, run_count, local_offset in source_runs[cp_rank]:
            overlap_start = max(target.logical_start, run_start)
            overlap_end = min(target.logical_end, run_start + run_count)
            if overlap_start < overlap_end:
                local_start = local_offset + overlap_start - run_start
                local_send_parts.append(
                    torch.arange(
                        local_start,
                        local_start + overlap_end - overlap_start,
                        dtype=torch.int64,
                        device=device,
                    )
                )

    recv_reorder_parts = []
    for source, runs in enumerate(source_runs):
        for target in targets:
            if target.destination != cp_rank:
                continue
            for run_start, run_count, _ in runs:
                overlap_start = max(target.logical_start, run_start)
                overlap_end = min(target.logical_end, run_start + run_count)
                if overlap_start < overlap_end:
                    compact_start = (
                        target.compact_start
                        + overlap_start
                        - target.logical_start
                    )
                    recv_reorder_parts.append(
                        torch.arange(
                            compact_start,
                            compact_start + overlap_end - overlap_start,
                            dtype=torch.int64,
                            device=device,
                        )
                    )

    return CPCompactKVExchangePlan(
        local_send_indices=_cat_index_parts(local_send_parts, device=device),
        send_sizes=tuple(route_counts[cp_rank]),
        recv_sizes=tuple(
            route_counts[source][cp_rank] for source in range(cp_size)
        ),
        recv_packed_to_compact=_cat_index_parts(
            recv_reorder_parts, device=device
        ),
    )


def build_cp_prefill_swa_gather_plan(
    *,
    plan: CPPrefillKVGatherPlan,
    runtime_layout: CPPrefillRuntimeLayout,
    window_left: int,
) -> CPPrefillSWAKVGatherPlan:
    """Plan destination-specific SWA K/V rows for one CP runtime layout."""
    window_left = int(window_left)
    if window_left <= 0:
        raise ValueError("SWA compact gather requires a positive left window")
    cp_size = len(runtime_layout.spec.per_rank_tokens)
    cp_rank = runtime_layout.cp_rank
    if len(plan.prefix.sizes) != cp_size or len(plan.extend.sizes) != cp_size:
        raise ValueError("SWA gather plan CP size does not match runtime layout")
    if plan.prefix.logical_token_count != runtime_layout.spec.extend_start:
        raise ValueError("SWA gather prefix length does not match extend_start")
    if plan.extend.logical_token_count != runtime_layout.spec.extend_len:
        raise ValueError("SWA gather extend length does not match split spec")

    device = runtime_layout.local_extend_indices.device
    per_destination_offset = [0] * cp_size

    if runtime_layout.q_is_contracted:
        active_counts = runtime_layout.active_tokens_per_cp_rank()
        if sum(active_counts) not in (0, 1):
            raise ValueError("contracted SWA runtime must contain at most one Q row")
        global_q_blocks = []
        if sum(active_counts) == 1:
            destination = active_counts.index(1)
            last_logical = (
                runtime_layout.spec.extend_start
                + runtime_layout.spec.extend_len
                - 1
            )
            global_q_blocks.append((destination, last_logical, 1))
    else:
        global_q_blocks = [
            (destination, block.logical_start, block.token_count)
            for destination in range(cp_size)
            for block in runtime_layout.spec.local_blocks(destination)
        ]

    global_targets = []
    for destination, q_start, q_count in global_q_blocks:
        kv_start = max(0, q_start - window_left)
        kv_end = q_start + q_count
        kv_count = kv_end - kv_start
        global_targets.append(
            _CPCompactKVTargetInterval(
                destination=destination,
                logical_start=kv_start,
                token_count=kv_count,
                compact_start=per_destination_offset[destination],
            )
        )
        per_destination_offset[destination] += kv_count

    prefix_targets = _slice_compact_targets_to_segment(
        targets=global_targets,
        segment_start=0,
        segment_len=plan.prefix.logical_token_count,
    )
    extend_targets = _slice_compact_targets_to_segment(
        targets=global_targets,
        segment_start=runtime_layout.spec.extend_start,
        segment_len=plan.extend.logical_token_count,
    )
    prefix = _build_compact_exchange_for_rank_packed_segment(
        segment=plan.prefix,
        targets=prefix_targets,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )
    extend = _build_compact_exchange_for_extend(
        segment=plan.extend,
        runtime_layout=runtime_layout,
        targets=extend_targets,
    )

    k_lengths = [
        block.visible_kv_end
        - max(0, block.logical_start - window_left)
        for block in runtime_layout.q_blocks
    ]
    block_cu_seqlens_k = torch.tensor(
        [[0, k_length] for k_length in k_lengths],
        dtype=torch.int32,
        device=device,
    ).reshape(-1, 2)
    compact_token_count = sum(k_lengths)
    if sum(prefix.recv_sizes) + sum(extend.recv_sizes) != compact_token_count:
        raise RuntimeError("compact receive counts do not cover local SWA K/V")
    return CPPrefillSWAKVGatherPlan(
        prefix=prefix,
        extend=extend,
        compact_token_count=compact_token_count,
        block_cu_seqlens_k=block_cu_seqlens_k,
        block_k_lengths=tuple(k_lengths),
        window_left=window_left,
    )


def restore_rank_packed_rows(
    packed_rows: torch.Tensor,
    segment: CPKVGatherSegmentPlan,
) -> torch.Tensor:
    if packed_rows.shape[0] != segment.logical_token_count:
        raise ValueError("rank-packed row count does not match the gather plan")
    if segment.rank_packed_to_logical.numel() != segment.logical_token_count:
        raise ValueError("rank-packed reorder length does not match the gather plan")
    logical_rows = packed_rows.new_empty(
        (segment.logical_token_count, *packed_rows.shape[1:])
    )
    logical_rows.index_copy_(
        0,
        segment.rank_packed_to_logical.to(device=packed_rows.device),
        packed_rows,
    )
    return logical_rows


def gather_cp_prefill_kv(
    *,
    plan: CPPrefillKVGatherPlan,
    local_prefix_k: torch.Tensor,
    local_prefix_v: torch.Tensor,
    local_extend_k: torch.Tensor,
    local_extend_v: torch.Tensor,
    cp_group,
    destination_ranks: Sequence[int],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Exchange prefix and extend K/V only among owners and Q consumers."""
    destinations = tuple(int(rank) for rank in destination_ranks)

    def exchange_segment(segment, local_k, local_v):
        tensors = [local_k.contiguous(), local_v.contiguous()]
        if len(destinations) == cp_group.world_size:
            return cp_group.all_gatherv(tensors, sizes=list(segment.sizes))
        return cp_group.gatherv_to_ranks(
            tensors,
            sizes=list(segment.sizes),
            dst_ranks=list(destinations),
        )

    prefix_packed = exchange_segment(
        plan.prefix, local_prefix_k, local_prefix_v
    )
    extend_packed = exchange_segment(
        plan.extend, local_extend_k, local_extend_v
    )
    if cp_group.rank_in_group not in destinations:
        if prefix_packed is not None or extend_packed is not None:
            raise RuntimeError("non-consumer CP rank received gathered K/V")
        return None, None
    if prefix_packed is None or extend_packed is None:
        raise RuntimeError("Q consumer did not receive complete gathered K/V")

    prefix_k_packed, prefix_v_packed = prefix_packed
    extend_k_packed, extend_v_packed = extend_packed
    prefix_k = restore_rank_packed_rows(prefix_k_packed, plan.prefix)
    prefix_v = restore_rank_packed_rows(prefix_v_packed, plan.prefix)
    extend_k = restore_rank_packed_rows(extend_k_packed, plan.extend)
    extend_v = restore_rank_packed_rows(extend_v_packed, plan.extend)
    return (
        torch.cat((prefix_k, extend_k), dim=0).contiguous(),
        torch.cat((prefix_v, extend_v), dim=0).contiguous(),
    )


def gather_cp_prefill_swa_kv(
    *,
    plan: CPPrefillSWAKVGatherPlan,
    packed_prefix_k: torch.Tensor,
    packed_prefix_v: torch.Tensor,
    packed_extend_k: torch.Tensor,
    packed_extend_v: torch.Tensor,
    cp_group,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Exchange only the logical K/V intervals visible to local SWA Q blocks."""
    if packed_prefix_k.shape[0] != packed_prefix_v.shape[0]:
        raise ValueError("packed prefix K/V row counts must match")
    if packed_extend_k.shape[0] != packed_extend_v.shape[0]:
        raise ValueError("packed extend K/V row counts must match")

    compact_k = (
        packed_extend_k.new_empty(
            (plan.compact_token_count, *packed_extend_k.shape[1:])
        )
        if plan.compact_token_count
        else None
    )
    compact_v = (
        packed_extend_v.new_empty(
            (plan.compact_token_count, *packed_extend_v.shape[1:])
        )
        if plan.compact_token_count
        else None
    )

    def exchange_segment(
        segment: CPCompactKVExchangePlan,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
    ) -> None:
        if sum(segment.send_sizes) == 0 and sum(segment.recv_sizes) == 0:
            return
        if local_k.shape[0] != sum(segment.send_sizes):
            raise ValueError("packed K rows do not match compact send sizes")
        recv_k, recv_v = cp_group.all_to_allv(
            [local_k, local_v],
            send_sizes=list(segment.send_sizes),
            recv_sizes=list(segment.recv_sizes),
        )
        recv_rows = sum(segment.recv_sizes)
        if recv_rows == 0:
            return
        if compact_k is None or compact_v is None:
            raise RuntimeError("non-Q CP rank unexpectedly received compact K/V")
        reorder = segment.recv_packed_to_compact.to(device=compact_k.device)
        compact_k.index_copy_(0, reorder, recv_k)
        compact_v.index_copy_(0, reorder, recv_v)

    exchange_segment(plan.prefix, packed_prefix_k, packed_prefix_v)
    exchange_segment(plan.extend, packed_extend_k, packed_extend_v)
    return compact_k, compact_v


def _assert_no_dummy_physical_slot(slots: torch.Tensor, message: str) -> None:
    if slots.numel() == 0:
        return
    has_dummy = torch.any(slots == 0)
    if slots.device.type == "cpu":
        if bool(has_dummy):
            raise ValueError(message)
        return
    torch._assert_async(torch.logical_not(has_dummy), message)


def _lens_to_list(lens: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(lens, torch.Tensor):
        return [int(x) for x in lens.to("cpu").tolist()]
    return [int(x) for x in lens]


def build_cp_sharded_kv_prefill_plan(
    *,
    logical_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    prefix_lens: Sequence[int] | torch.Tensor,
    seq_lens: Optional[Sequence[int] | torch.Tensor] = None,
    page_size: int,
) -> Optional[CPShardedKVPrefillPlan]:
    """Build the backend-neutral Phase 1 prefix/extend split plan.

    Phase 1 intentionally supports one prefill request per batch. A page-aligned
    cached prefix can be represented as two page-table slices without copying.
    """

    if logical_page_table is None:
        return None
    prefix_lens_list = _lens_to_list(prefix_lens)
    batch_size = int(cache_seqlens.numel())
    if len(prefix_lens_list) != batch_size:
        raise ValueError(
            "prefix_lens and cache_seqlens must describe the same batch size"
        )
    seq_lens_list = _lens_to_list(seq_lens) if seq_lens is not None else None
    if seq_lens_list is not None and len(seq_lens_list) != batch_size:
        raise ValueError(
            "seq_lens and cache_seqlens must describe the same batch size"
        )
    if logical_page_table.shape[0] != batch_size:
        raise ValueError(
            "logical_page_table and cache_seqlens must describe the same batch size"
        )
    if batch_size != 1:
        raise NotImplementedError(
            "CP sharded-KV prefill currently requires batch_size=1"
        )
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    prefix_len = prefix_lens_list[0]
    seq_len = (
        seq_lens_list[0]
        if seq_lens_list is not None
        else int(cache_seqlens.reshape(-1)[0].item())
    )
    extend_len = seq_len - prefix_len
    if prefix_len <= 0 or extend_len <= 0:
        return None
    if page_size > 1 and prefix_len % page_size != 0:
        return None

    if page_size == 1:
        prefix_columns = prefix_len
        total_columns = seq_len
    else:
        prefix_columns = prefix_len // page_size
        total_columns = (seq_len + page_size - 1) // page_size
    extend_columns = total_columns - prefix_columns
    if (
        prefix_columns <= 0
        or extend_columns <= 0
        or logical_page_table.shape[1] < total_columns
    ):
        return None

    prefix_cache_seqlens = torch.tensor(
        [prefix_len], dtype=cache_seqlens.dtype, device=cache_seqlens.device
    )
    extend_cache_seqlens = torch.tensor(
        [extend_len], dtype=cache_seqlens.dtype, device=cache_seqlens.device
    )
    return CPShardedKVPrefillPlan(
        prefix=CPShardedKVRegion(
            logical_page_table=logical_page_table[:, :prefix_columns],
            cache_seqlens=prefix_cache_seqlens,
        ),
        extend=CPShardedKVRegion(
            logical_page_table=logical_page_table[:, prefix_columns:total_columns],
            cache_seqlens=extend_cache_seqlens,
        ),
        full_cache_seqlens=cache_seqlens,
        total_page_columns=total_columns,
    )
