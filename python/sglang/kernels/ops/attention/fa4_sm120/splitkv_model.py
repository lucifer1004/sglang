# Copyright (c) 2026, SGLang Team.
"""Pure analytical model for SM120 FA4 SplitKV routing.

The model uses KV tiles per split CTA as its optimization primitive.  The
effective split count is derived from that grain, matching the runtime's
uniform ceil-div partitioning.  Main-kernel work is priced with the exact
list-scheduling makespan of full and tail split CTAs; the combine kernel and
partial workspace are modeled separately.

This module deliberately has no CUDA or Torch dependency.  Device-local
calibration supplies the constants, while runtime policy can test and memoize
the pure functions before CUDA Graph capture.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Optional


def ceil_div(dividend: int, divisor: int) -> int:
    """Return ``ceil(dividend / divisor)`` for non-negative integers."""
    if dividend < 0 or divisor <= 0:
        raise ValueError("ceil_div requires dividend >= 0 and divisor > 0")
    return -(-dividend // divisor)


def normalize_num_splits(num_splits: int, num_n_blocks: int) -> int:
    """Remove empty tail splits using the scheduler's uniform chunk rule."""
    if num_splits <= 1 or num_n_blocks <= 1:
        return 1
    requested_splits = min(num_splits, num_n_blocks)
    blocks_per_split = ceil_div(num_n_blocks, requested_splits)
    return ceil_div(num_n_blocks, blocks_per_split)


def effective_partitions(
    num_n_blocks: int,
    *,
    max_splits: int,
) -> tuple[tuple[int, int], ...]:
    """Return every legal ``(tiles_per_cta, splits)`` partition.

    Enumerating the grain instead of a power-of-two split list retains the
    discrete full/tail scheduling teeth.  Multiple grains may derive the same
    split count but different tail geometry (for example, 16 tiles can be
    partitioned as 6+6+4 or 7+7+2); they are distinct algorithmic candidates.
    """
    if num_n_blocks <= 0 or max_splits <= 0:
        return ((0, 1),)
    return tuple(
        (tiles_per_cta, ceil_div(num_n_blocks, tiles_per_cta))
        for tiles_per_cta in range(1, num_n_blocks + 1)
        if ceil_div(num_n_blocks, tiles_per_cta) <= max_splits
    )


def list_schedule_makespan(
    num_slots: int,
    batches: Iterable[tuple[int, float]],
) -> float:
    """Return the exact Graham list-scheduling makespan.

    ``batches`` contains ``(job_count, job_duration_s)`` pairs in CUDA launch
    order.  Jobs within one batch are identical.  Equal-availability slots are
    advanced together, avoiding a per-CTA heap walk for large grids.
    """
    if num_slots <= 0:
        raise ValueError("num_slots must be positive")
    available = [0.0] * num_slots
    for job_count, duration_s in batches:
        if job_count <= 0 or duration_s <= 0:
            continue
        remaining = job_count
        while remaining > 0:
            available.sort()
            earliest = available[0]
            tied = bisect.bisect_right(available, earliest)
            next_level = available[tied] if tied < num_slots else math.inf
            rounds_to_exhaust = ceil_div(remaining, tied)
            rounds_to_next = (
                math.inf
                if math.isinf(next_level)
                else max(
                    1,
                    math.ceil((next_level - earliest) / duration_s - 1e-9),
                )
            )
            if rounds_to_exhaust <= rounds_to_next:
                complete_rounds, extra_jobs = divmod(remaining, tied)
                for slot in range(tied):
                    available[slot] += complete_rounds * duration_s
                for slot in range(extra_jobs):
                    available[slot] += duration_s
                remaining = 0
            else:
                rounds = int(rounds_to_next)
                for slot in range(tied):
                    available[slot] += rounds * duration_s
                remaining -= tied * rounds
    return max(available)


@dataclass(frozen=True)
class SplitKvCalibration:
    """Device- and route-family constants consumed by the pure model."""

    sm_slots: int
    main_inv_bandwidth_s_per_byte: float
    main_inv_single_sm_s_per_byte: float
    main_fixed_s: float
    combine_inv_bandwidth_s_per_byte: float
    combine_inv_single_sm_s_per_byte: float
    combine_fixed_s: float
    combine_cta_fixed_s: float
    main_first_two_tile_scale: float = 1.0
    l2_cache_bytes: int = 0

    def __post_init__(self) -> None:
        if self.sm_slots <= 0:
            raise ValueError("sm_slots must be positive")
        rates = (
            self.main_inv_bandwidth_s_per_byte,
            self.main_inv_single_sm_s_per_byte,
            self.main_fixed_s,
            self.combine_inv_bandwidth_s_per_byte,
            self.combine_inv_single_sm_s_per_byte,
            self.combine_fixed_s,
            self.combine_cta_fixed_s,
            self.main_first_two_tile_scale,
        )
        if any(value < 0 or not math.isfinite(value) for value in rates):
            raise ValueError("calibration constants must be finite and non-negative")
        if self.l2_cache_bytes < 0:
            raise ValueError("l2_cache_bytes must be non-negative")


@dataclass(frozen=True)
class SplitKvWorkload:
    """Algorithm-level workload facts shared by every route candidate."""

    total_mblocks: int
    num_n_blocks: int
    main_bytes_per_kv_tile: int
    output_rows: int
    head_dim_v: int
    max_splits: int = 128
    partial_element_size: int = 4
    output_element_size: int = 2
    lse_element_size: int = 4
    combine_rows_per_cta: int = 8
    combine_cols_per_cta: int = 128
    max_workspace_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        positive = (
            self.total_mblocks,
            self.main_bytes_per_kv_tile,
            self.head_dim_v,
            self.max_splits,
            self.partial_element_size,
            self.output_element_size,
            self.lse_element_size,
            self.combine_rows_per_cta,
            self.combine_cols_per_cta,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("workload dimensions and element sizes must be positive")
        if self.num_n_blocks < 0 or self.output_rows < 0:
            raise ValueError("num_n_blocks and output_rows must be non-negative")
        if self.max_workspace_bytes is not None and self.max_workspace_bytes < 0:
            raise ValueError("max_workspace_bytes must be non-negative")


@dataclass(frozen=True)
class SplitKvPrediction:
    """Predicted cost and structure for one effective partition."""

    kv_tiles_per_cta: int
    num_splits: int
    main_ctas: int
    main_s: float
    combine_s: float
    total_s: float
    workspace_bytes: int
    concurrent_main_footprint_bytes: int


def _combine_job_batches(
    workload: SplitKvWorkload,
    *,
    num_splits: int,
    byte_time_s: float,
    cta_fixed_s: float,
) -> tuple[tuple[int, float], ...]:
    row_tiles = ceil_div(workload.output_rows, workload.combine_rows_per_cta)
    col_tiles = ceil_div(workload.head_dim_v, workload.combine_cols_per_cta)
    batches = []
    for row_tile in range(row_tiles):
        rows = min(
            workload.combine_rows_per_cta,
            workload.output_rows - row_tile * workload.combine_rows_per_cta,
        )
        for col_tile in range(col_tiles):
            cols = min(
                workload.combine_cols_per_cta,
                workload.head_dim_v - col_tile * workload.combine_cols_per_cta,
            )
            partial_bytes = rows * cols * num_splits * workload.partial_element_size
            output_bytes = rows * cols * workload.output_element_size
            lse_bytes = rows * num_splits * workload.lse_element_size
            duration_s = cta_fixed_s + (
                partial_bytes + output_bytes + lse_bytes
            ) * byte_time_s
            batches.append((1, duration_s))
    return tuple(batches)


def predict_partition(
    workload: SplitKvWorkload,
    calibration: SplitKvCalibration,
    *,
    kv_tiles_per_cta: int,
) -> SplitKvPrediction:
    """Predict one canonical KV-grain partition."""
    if workload.num_n_blocks == 0:
        return SplitKvPrediction(0, 1, 0, 0.0, 0.0, 0.0, 0, 0)
    if kv_tiles_per_cta <= 0:
        raise ValueError("kv_tiles_per_cta must be positive")

    num_splits = ceil_div(workload.num_n_blocks, kv_tiles_per_cta)
    if num_splits > workload.max_splits:
        raise ValueError("partition exceeds workload.max_splits")
    tail_tiles = workload.num_n_blocks - (num_splits - 1) * kv_tiles_per_cta
    main_ctas = workload.total_mblocks * num_splits
    main_concurrency = min(main_ctas, calibration.sm_slots)
    main_byte_time_s = max(
        workload.main_bytes_per_kv_tile
        * main_concurrency
        * calibration.main_inv_bandwidth_s_per_byte,
        workload.main_bytes_per_kv_tile
        * calibration.main_inv_single_sm_s_per_byte,
    )
    def main_job_s(tiles: int) -> float:
        short_tiles = min(tiles, 2)
        steady_tiles = max(tiles - short_tiles, 0)
        return calibration.main_fixed_s + main_byte_time_s * (
            steady_tiles
            + short_tiles * calibration.main_first_two_tile_scale
        )

    main_s = list_schedule_makespan(
        calibration.sm_slots,
        (
            (
                workload.total_mblocks * (num_splits - 1),
                main_job_s(kv_tiles_per_cta),
            ),
            (
                workload.total_mblocks,
                main_job_s(tail_tiles),
            ),
        ),
    )
    footprint_bytes = (
        main_concurrency
        * kv_tiles_per_cta
        * workload.main_bytes_per_kv_tile
    )

    workspace_bytes = 0
    combine_s = 0.0
    if num_splits > 1:
        workspace_bytes = num_splits * workload.output_rows * (
            workload.head_dim_v * workload.partial_element_size
            + workload.lse_element_size
        )
        combine_ctas = ceil_div(
            workload.output_rows, workload.combine_rows_per_cta
        ) * ceil_div(workload.head_dim_v, workload.combine_cols_per_cta)
        combine_concurrency = min(combine_ctas, calibration.sm_slots)
        combine_byte_time_s = max(
            combine_concurrency * calibration.combine_inv_bandwidth_s_per_byte,
            calibration.combine_inv_single_sm_s_per_byte,
        )
        combine_s = calibration.combine_fixed_s + list_schedule_makespan(
            calibration.sm_slots,
            _combine_job_batches(
                workload,
                num_splits=num_splits,
                byte_time_s=combine_byte_time_s,
                cta_fixed_s=calibration.combine_cta_fixed_s,
            ),
        )
    return SplitKvPrediction(
        kv_tiles_per_cta=kv_tiles_per_cta,
        num_splits=num_splits,
        main_ctas=main_ctas,
        main_s=main_s,
        combine_s=combine_s,
        total_s=main_s + combine_s,
        workspace_bytes=workspace_bytes,
        concurrent_main_footprint_bytes=footprint_bytes,
    )


def predict_partitions(
    workload: SplitKvWorkload,
    calibration: SplitKvCalibration,
) -> tuple[SplitKvPrediction, ...]:
    """Predict all legal effective partitions after resource guard rails."""
    if workload.num_n_blocks == 0:
        return (predict_partition(workload, calibration, kv_tiles_per_cta=1),)
    predictions = tuple(
        predict_partition(workload, calibration, kv_tiles_per_cta=grain)
        for grain, _ in effective_partitions(
            workload.num_n_blocks,
            max_splits=workload.max_splits,
        )
    )
    workspace_valid = tuple(
        prediction
        for prediction in predictions
        if workload.max_workspace_bytes is None
        or prediction.workspace_bytes <= workload.max_workspace_bytes
    )
    if not workspace_valid:
        workspace_valid = tuple(
            prediction for prediction in predictions if prediction.num_splits == 1
        )
    if not calibration.l2_cache_bytes:
        return workspace_valid
    l2_valid = tuple(
        prediction
        for prediction in workspace_valid
        if prediction.num_splits == 1
        or prediction.concurrent_main_footprint_bytes
        <= calibration.l2_cache_bytes
    )
    return l2_valid or workspace_valid


def select_partition(
    workload: SplitKvWorkload,
    calibration: SplitKvCalibration,
) -> SplitKvPrediction:
    """Return the minimum predicted route; exact ties prefer fewer splits."""
    return min(
        predict_partitions(workload, calibration),
        key=lambda prediction: (prediction.total_s, prediction.num_splits),
    )


def near_optimal_partitions(
    workload: SplitKvWorkload,
    calibration: SplitKvCalibration,
    *,
    relative_tolerance: float = 0.10,
) -> tuple[SplitKvPrediction, ...]:
    """Return candidates inside the model's relative regret envelope."""
    if relative_tolerance < 0 or not math.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be finite and non-negative")
    predictions = predict_partitions(workload, calibration)
    best = min(prediction.total_s for prediction in predictions)
    limit = best * (1.0 + relative_tolerance)
    return tuple(
        prediction for prediction in predictions if prediction.total_s <= limit
    )


def partition_selection_is_ambiguous(
    workload: SplitKvWorkload,
    calibration: SplitKvCalibration,
    *,
    relative_tolerance: float = 0.10,
) -> bool:
    """Return whether multiple candidates lie inside the regret envelope."""
    return (
        len(
            near_optimal_partitions(
                workload,
                calibration,
                relative_tolerance=relative_tolerance,
            )
        )
        > 1
    )
