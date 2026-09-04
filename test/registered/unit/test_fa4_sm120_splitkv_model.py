"""CPU tests for the pure SM120 FA4 SplitKV routing model."""

import heapq
import random

import pytest

from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
    SplitKvCalibration,
    SplitKvWorkload,
    effective_partitions,
    list_schedule_makespan,
    near_optimal_partitions,
    partition_selection_is_ambiguous,
    predict_partition,
    predict_partitions,
    select_partition,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _brute_makespan(num_slots, batches):
    available = [0.0] * num_slots
    heapq.heapify(available)
    for count, duration in batches:
        for _ in range(count):
            heapq.heapreplace(available, available[0] + duration)
    return max(available)


def _calibration(**overrides):
    values = {
        "sm_slots": 188,
        "main_inv_bandwidth_s_per_byte": 5e-13,
        "main_inv_single_sm_s_per_byte": 1.2e-10,
        "main_fixed_s": 6.4e-6,
        "combine_inv_bandwidth_s_per_byte": 5e-13,
        "combine_inv_single_sm_s_per_byte": 1e-11,
        "combine_fixed_s": 3e-6,
        "combine_cta_fixed_s": 1e-6,
        "l2_cache_bytes": 0,
    }
    values.update(overrides)
    return SplitKvCalibration(**values)


def _workload(**overrides):
    values = {
        "total_mblocks": 1,
        "num_n_blocks": 128,
        "main_bytes_per_kv_tile": 64 * 128 * 2 * 2,
        "output_rows": 12,
        "head_dim_v": 128,
    }
    values.update(overrides)
    return SplitKvWorkload(**values)


def test_effective_partitions_cover_non_power_of_two_grains():
    assert effective_partitions(10, max_splits=10) == (
        (1, 10),
        (2, 5),
        (3, 4),
        (4, 3),
        (5, 2),
        (6, 2),
        (7, 2),
        (8, 2),
        (9, 2),
        (10, 1),
    )


def test_compressed_list_schedule_matches_brute_force():
    generator = random.Random(20260903)
    for _ in range(200):
        num_slots = generator.randint(1, 32)
        batches = tuple(
            (generator.randint(0, 80), generator.randint(1, 20) / 10.0)
            for _ in range(generator.randint(1, 5))
        )
        assert list_schedule_makespan(
            num_slots, batches
        ) == pytest.approx(_brute_makespan(num_slots, batches))


def test_tail_partition_retains_list_scheduling_sawtooth():
    workload = _workload(
        total_mblocks=64,
        num_n_blocks=16,
        main_bytes_per_kv_tile=37376,
        output_rows=0,
    )
    calibration = _calibration(
        combine_fixed_s=0.0,
        combine_cta_fixed_s=0.0,
        combine_inv_bandwidth_s_per_byte=0.0,
        combine_inv_single_sm_s_per_byte=0.0,
    )
    grain_7 = predict_partition(workload, calibration, kv_tiles_per_cta=7)
    grain_8 = predict_partition(workload, calibration, kv_tiles_per_cta=8)
    assert grain_7.num_splits == 3
    assert grain_8.num_splits == 2
    assert grain_7.main_s < grain_8.main_s


def test_excessive_splitting_pays_separate_combine_cost():
    workload = _workload()
    calibration = _calibration()
    split_64 = predict_partition(workload, calibration, kv_tiles_per_cta=2)
    split_128 = predict_partition(workload, calibration, kv_tiles_per_cta=1)
    assert split_128.main_s <= split_64.main_s
    assert split_128.combine_s > split_64.combine_s


def test_reduced_first_tile_cost_prevents_false_short_kv_split():
    workload = _workload(num_n_blocks=2, output_rows=12)
    calibration = _calibration(main_first_two_tile_scale=0.0)
    split = predict_partition(workload, calibration, kv_tiles_per_cta=1)
    unsplit = predict_partition(workload, calibration, kv_tiles_per_cta=2)
    assert split.main_s == unsplit.main_s
    assert split.total_s > unsplit.total_s


def test_workspace_cap_filters_large_split_counts():
    workload = _workload(max_workspace_bytes=64 * 12 * (128 * 4 + 4))
    predictions = predict_partitions(workload, _calibration())
    assert predictions
    assert max(prediction.num_splits for prediction in predictions) <= 64
    assert all(
        prediction.workspace_bytes <= workload.max_workspace_bytes
        for prediction in predictions
    )


def test_l2_guard_filters_oversized_concurrent_streaming_windows():
    workload = _workload()
    calibration = _calibration(l2_cache_bytes=8 * workload.main_bytes_per_kv_tile)
    predictions = predict_partitions(workload, calibration)
    assert predictions
    assert all(
        prediction.num_splits == 1
        or prediction.concurrent_main_footprint_bytes
        <= calibration.l2_cache_bytes
        for prediction in predictions
    )


def test_exact_tie_prefers_fewer_splits():
    calibration = _calibration(
        main_inv_bandwidth_s_per_byte=0.0,
        main_inv_single_sm_s_per_byte=0.0,
        main_fixed_s=0.0,
        combine_inv_bandwidth_s_per_byte=0.0,
        combine_inv_single_sm_s_per_byte=0.0,
        combine_fixed_s=0.0,
        combine_cta_fixed_s=0.0,
    )
    selected = select_partition(_workload(output_rows=0), calibration)
    assert selected.num_splits == 1


def test_empty_kv_selects_unsplit_zero_cost():
    selected = select_partition(_workload(num_n_blocks=0), _calibration())
    assert selected.num_splits == 1
    assert selected.total_s == 0.0
    assert selected.workspace_bytes == 0


def test_ambiguity_uses_relative_regret_envelope():
    workload = _workload(num_n_blocks=4, output_rows=12)
    calibration = _calibration()
    best = select_partition(workload, calibration)
    exact = near_optimal_partitions(
        workload,
        calibration,
        relative_tolerance=0.0,
    )
    assert exact == (best,)
    assert partition_selection_is_ambiguous(
        workload,
        calibration,
        relative_tolerance=1.0,
    )
