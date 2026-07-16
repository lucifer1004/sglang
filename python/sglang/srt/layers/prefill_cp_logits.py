"""Hidden-row routing for persistent-token context-parallel prefill logits."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.distributed.parallel_state import get_attn_tp_group, get_tp_group


@dataclass(frozen=True)
class CPLogitsRouteSegment:
    logical_start: int
    token_count: int


@dataclass(frozen=True)
class CPPrefillLogitsRoutePlan:
    logical_start: int
    token_count: int
    per_cp_rank_segments: tuple[tuple[CPLogitsRouteSegment, ...], ...]
    per_cp_rank_tokens: tuple[int, ...]
    global_tp_sizes: tuple[int, ...]


def build_cp_prefill_logits_route_plan(
    runtime_layout,
    *,
    logical_start: int,
    token_count: int,
) -> CPPrefillLogitsRoutePlan:
    logical_start = int(logical_start)
    token_count = int(token_count)
    if token_count <= 0:
        raise ValueError("prefill CP logits route must contain at least one token")

    spec = runtime_layout.spec
    logical_end = logical_start + token_count
    extend_end = spec.extend_start + spec.extend_len
    if logical_start < spec.extend_start or logical_end > extend_end:
        raise ValueError("prefill CP logits route is outside the scheduled extend")

    cp_size = len(spec.per_rank_tokens)
    segments_by_rank: list[tuple[CPLogitsRouteSegment, ...]] = []
    if runtime_layout.q_is_contracted:
        final_logical = extend_end - 1
        final_owner = spec.blocks[-1].owner_rank
        for cp_rank in range(cp_size):
            if cp_rank == final_owner and logical_start <= final_logical < logical_end:
                segments = (CPLogitsRouteSegment(final_logical, 1),)
            else:
                segments = ()
            segments_by_rank.append(segments)
    else:
        for cp_rank in range(cp_size):
            segments = []
            for block in spec.local_blocks(cp_rank):
                start = max(block.logical_start, logical_start)
                end = min(block.logical_start + block.token_count, logical_end)
                if start < end:
                    segments.append(CPLogitsRouteSegment(start, end - start))
            segments_by_rank.append(tuple(segments))

    per_cp_rank_tokens = tuple(
        sum(segment.token_count for segment in segments)
        for segments in segments_by_rank
    )
    if sum(per_cp_rank_tokens) != token_count:
        raise RuntimeError(
            "prefill CP logits route is not fully represented by active hidden rows"
        )
    global_tp_sizes = tuple(
        lane_count
        for cp_count in per_cp_rank_tokens
        for lane_count in (cp_count, 0)
    )
    return CPPrefillLogitsRoutePlan(
        logical_start=logical_start,
        token_count=token_count,
        per_cp_rank_segments=tuple(segments_by_rank),
        per_cp_rank_tokens=per_cp_rank_tokens,
        global_tp_sizes=global_tp_sizes,
    )


def _validate_route_topology(runtime_layout, global_tp_group, attn_tp_group) -> int:
    cp_size = len(runtime_layout.active_tokens_per_cp_rank())
    if cp_size <= 1:
        raise RuntimeError("prefill CP logits routing requires CP size greater than one")
    if attn_tp_group.world_size != 2:
        raise RuntimeError("prefill CP logits routing requires AttnTP2")
    if global_tp_group.world_size != 2 * cp_size:
        raise RuntimeError(
            "prefill CP logits global-TP size must be twice the CP size"
        )
    lane = attn_tp_group.rank_in_group
    if lane not in (0, 1):
        raise RuntimeError("prefill CP logits AttnTP lane must be 0 or 1")
    expected_global_rank = 2 * runtime_layout.cp_rank + lane
    if global_tp_group.rank_in_group != expected_global_rank:
        raise RuntimeError("prefill CP logits global-TP rank order does not match CP/lane")
    return lane


def _local_route_indices(hidden_states, runtime_layout, plan):
    parts = []
    logical_end = plan.logical_start + plan.token_count
    for block in runtime_layout.q_blocks:
        start = max(block.logical_start, plan.logical_start)
        end = min(block.logical_start + block.token_count, logical_end)
        if start < end:
            local_start = block.local_start + start - block.logical_start
            parts.append(
                torch.arange(
                    local_start,
                    local_start + end - start,
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
            )
    if parts:
        return torch.cat(parts)
    return torch.empty((0,), dtype=torch.int64, device=hidden_states.device)


def _rank_packed_to_logical_indices(hidden_states, plan):
    parts = []
    for segments in plan.per_cp_rank_segments:
        for segment in segments:
            start = segment.logical_start - plan.logical_start
            parts.append(
                torch.arange(
                    start,
                    start + segment.token_count,
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
            )
    if parts:
        return torch.cat(parts)
    return torch.empty((0,), dtype=torch.int64, device=hidden_states.device)


def route_cp_prefill_hidden_states(
    hidden_states: torch.Tensor,
    runtime_layout,
    *,
    logical_start: int,
    token_count: int,
    global_tp_group=None,
    attn_tp_group=None,
) -> torch.Tensor:
    if hidden_states.shape[0] != runtime_layout.active_local_tokens:
        raise RuntimeError("prefill CP logits hidden rows do not match runtime layout")
    global_tp_group = global_tp_group or get_tp_group()
    attn_tp_group = attn_tp_group or get_attn_tp_group()
    lane = _validate_route_topology(
        runtime_layout, global_tp_group, attn_tp_group
    )
    plan = build_cp_prefill_logits_route_plan(
        runtime_layout,
        logical_start=logical_start,
        token_count=token_count,
    )

    local_indices = _local_route_indices(hidden_states, runtime_layout, plan)
    expected_local_rows = plan.per_cp_rank_tokens[runtime_layout.cp_rank]
    if local_indices.numel() != expected_local_rows:
        raise RuntimeError("prefill CP logits local route does not match its plan")
    local_rows = (
        hidden_states.index_select(0, local_indices)
        if lane == 0
        else hidden_states.new_empty((0, *hidden_states.shape[1:]))
    )

    gathered = global_tp_group.all_gatherv(
        [local_rows], sizes=list(plan.global_tp_sizes)
    )
    if not isinstance(gathered, list) or len(gathered) != 1:
        raise RuntimeError("prefill CP logits gather returned an invalid result")
    packed_rows = gathered[0]
    if packed_rows.shape[0] != plan.token_count:
        raise RuntimeError("prefill CP logits gather returned the wrong row count")

    packed_to_logical = _rank_packed_to_logical_indices(hidden_states, plan)
    routed = packed_rows.new_empty((plan.token_count, *packed_rows.shape[1:]))
    routed.index_copy_(0, packed_to_logical, packed_rows)
    return routed
