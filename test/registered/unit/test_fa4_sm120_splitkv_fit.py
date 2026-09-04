"""CPU tests for fitting SM120 FA4 SplitKV model constants."""

import pytest

from sglang.kernels.ops.attention.fa4_sm120.splitkv_fit import (
    SplitKvCalibrationFitError,
    SplitKvObservation,
    fit_splitkv_calibration,
)
from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
    SplitKvCalibration,
    SplitKvWorkload,
    predict_partition,
    select_partition,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_TRUE = SplitKvCalibration(
    sm_slots=188,
    main_inv_bandwidth_s_per_byte=1e-12,
    main_inv_single_sm_s_per_byte=1.2e-10,
    main_fixed_s=6.4e-6,
    combine_inv_bandwidth_s_per_byte=8e-13,
    combine_inv_single_sm_s_per_byte=2e-11,
    combine_fixed_s=3.1e-6,
    combine_cta_fixed_s=8e-7,
    main_first_two_tile_scale=0.25,
    l2_cache_bytes=0,
)


def _workload(m, n, rows, *, tile_bytes=32768, head_dim_v=128):
    return SplitKvWorkload(
        total_mblocks=m,
        num_n_blocks=n,
        main_bytes_per_kv_tile=tile_bytes,
        output_rows=rows,
        head_dim_v=head_dim_v,
    )


def _synthetic_observations():
    main_specs = (
        (_workload(1, 2, 0), 1),
        (_workload(1, 2, 0), 2),
        (_workload(1, 4, 0), 1),
        (_workload(1, 4, 0), 4),
        (_workload(1, 128, 0), 128),
        (_workload(1, 128, 0), 2),
        (_workload(8, 128, 0), 8),
        (_workload(64, 16, 0, tile_bytes=37376), 7),
        (_workload(64, 16, 0, tile_bytes=37376), 8),
        (_workload(188, 64, 0, tile_bytes=65536), 64),
    )
    combine_specs = (
        (_workload(1, 128, 8), 16),
        (_workload(1, 128, 12), 4),
        (_workload(1, 128, 64), 2),
        (_workload(2, 64, 128, head_dim_v=256), 2),
        (_workload(2, 64, 256, head_dim_v=256), 1),
    )
    observations = []
    for workload, grain in main_specs:
        elapsed = predict_partition(
            workload, _TRUE, kv_tiles_per_cta=grain
        ).main_s
        observations.append(SplitKvObservation("main", workload, grain, elapsed))
    for workload, grain in combine_specs:
        elapsed = predict_partition(
            workload, _TRUE, kv_tiles_per_cta=grain
        ).combine_s
        observations.append(SplitKvObservation("combine", workload, grain, elapsed))
    for workload, grain in main_specs[:4] + main_specs[4:6]:
        elapsed = predict_partition(
            workload, _TRUE, kv_tiles_per_cta=grain
        ).total_s
        observations.append(SplitKvObservation("total", workload, grain, elapsed))
    return observations


def test_fit_reconstructs_observed_curves_and_route_choices():
    fit = fit_splitkv_calibration(
        _synthetic_observations(),
        sm_slots=188,
        l2_cache_bytes=0,
        max_relative_rms=1e-4,
    )
    assert fit.main_relative_rms < 1e-5
    assert fit.combine_relative_rms < 1e-5

    validation = (
        _workload(1, 128, 12),
        _workload(2, 64, 80),
        _workload(32, 32, 128, tile_bytes=65536),
    )
    for workload in validation:
        expected = select_partition(workload, _TRUE)
        actual = select_partition(workload, fit.constants)
        assert actual.kv_tiles_per_cta == expected.kv_tiles_per_cta
        assert actual.num_splits == expected.num_splits


def test_fit_requires_identifiable_component_observations():
    observations = _synthetic_observations()
    with pytest.raises(SplitKvCalibrationFitError, match="four main"):
        fit_splitkv_calibration(
            observations[:3] + observations[10:],
            sm_slots=188,
            l2_cache_bytes=0,
        )


def test_observation_rejects_invalid_measurements():
    workload = _workload(1, 128, 12)
    with pytest.raises(ValueError, match="finite and positive"):
        SplitKvObservation("main", workload, 4, 0.0)
    with pytest.raises(ValueError, match="component"):
        SplitKvObservation("other", workload, 4, 1e-5)
