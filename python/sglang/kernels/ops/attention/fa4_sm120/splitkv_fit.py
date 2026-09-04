# Copyright (c) 2026, SGLang Team.
"""Fit SM120 FA4 SplitKV model constants from component observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
    SplitKvCalibration,
    SplitKvWorkload,
    predict_partition,
)

# Route selection depends on candidate deltas, while component and absolute
# residuals regularize the decomposition.  One residual unit is the accepted
# 10-percent oracle-regret envelope relative to the workload's unsplit time.
_ROUTE_DELTA_TOLERANCE = 0.10
_COMPONENT_RESIDUAL_TOLERANCE = 0.25


class SplitKvCalibrationFitError(RuntimeError):
    """Raised when observations cannot identify a plausible model."""


@dataclass(frozen=True)
class SplitKvObservation:
    """One measured main-only or combine-only route component."""

    component: Literal["main", "combine", "total"]
    workload: SplitKvWorkload
    kv_tiles_per_cta: int
    elapsed_s: float

    def __post_init__(self) -> None:
        if self.component not in ("main", "combine", "total"):
            raise ValueError("component must be 'main', 'combine', or 'total'")
        if self.kv_tiles_per_cta <= 0:
            raise ValueError("kv_tiles_per_cta must be positive")
        if self.elapsed_s <= 0 or not math.isfinite(self.elapsed_s):
            raise ValueError("elapsed_s must be finite and positive")


@dataclass(frozen=True)
class SplitKvCalibrationFit:
    """Fitted constants plus relative-residual diagnostics."""

    constants: SplitKvCalibration
    main_relative_rms: float
    main_relative_max: float
    combine_relative_rms: float
    combine_relative_max: float


def _relative_diagnostics(residuals) -> tuple[float, float]:
    import numpy as np

    residuals = np.asarray(residuals, dtype=np.float64)
    return (
        float(np.sqrt(np.mean(residuals * residuals))),
        float(np.max(np.abs(residuals))),
    )


def fit_splitkv_calibration(
    observations: Sequence[SplitKvObservation],
    *,
    sm_slots: int,
    l2_cache_bytes: int,
    initial: Optional[SplitKvCalibration] = None,
    max_relative_rms: float = 0.25,
) -> SplitKvCalibrationFit:
    """Fit component constants, then resolve route-relevant degeneracy.

    Main and combine start from independent positive-log fits so neither can
    absorb the other's scheduling error.  When total observations are present,
    alternating route-aware fits select among component-equivalent solutions
    using normalized candidate deltas without abandoning the component gates.
    """
    import numpy as np
    from scipy.optimize import least_squares

    if sm_slots <= 0 or l2_cache_bytes < 0:
        raise ValueError("invalid device geometry")
    if max_relative_rms <= 0:
        raise ValueError("max_relative_rms must be positive")
    main_observations = tuple(
        observation
        for observation in observations
        if observation.component == "main"
    )
    combine_observations = tuple(
        observation
        for observation in observations
        if observation.component == "combine"
    )
    total_observations = tuple(
        observation
        for observation in observations
        if observation.component == "total"
    )
    if len(main_observations) < 4 or len(combine_observations) < 4:
        raise SplitKvCalibrationFitError(
            "at least four main and four combine observations are required"
        )

    if initial is None:
        initial = SplitKvCalibration(
            sm_slots=sm_slots,
            main_inv_bandwidth_s_per_byte=1e-12,
            main_inv_single_sm_s_per_byte=1e-10,
            main_fixed_s=5e-6,
            combine_inv_bandwidth_s_per_byte=1e-12,
            combine_inv_single_sm_s_per_byte=1e-11,
            combine_fixed_s=3e-6,
            combine_cta_fixed_s=1e-6,
            main_first_two_tile_scale=0.25,
            l2_cache_bytes=l2_cache_bytes,
        )
    if initial.sm_slots != sm_slots:
        raise ValueError("initial calibration sm_slots does not match")

    def constants_with(
        main_values: Sequence[float],
        combine_values: Sequence[float],
    ) -> SplitKvCalibration:
        return SplitKvCalibration(
            sm_slots=sm_slots,
            main_inv_bandwidth_s_per_byte=float(main_values[0]),
            main_inv_single_sm_s_per_byte=float(main_values[1]),
            main_fixed_s=float(main_values[2]),
            combine_inv_bandwidth_s_per_byte=float(combine_values[0]),
            combine_inv_single_sm_s_per_byte=float(combine_values[1]),
            combine_fixed_s=float(combine_values[2]),
            combine_cta_fixed_s=float(combine_values[3]),
            main_first_two_tile_scale=float(main_values[3]),
            l2_cache_bytes=l2_cache_bytes,
        )

    initial_main = np.array(
        [
            initial.main_inv_bandwidth_s_per_byte,
            initial.main_inv_single_sm_s_per_byte,
            initial.main_fixed_s,
            max(initial.main_first_two_tile_scale, 1e-4),
        ],
        dtype=np.float64,
    )
    initial_combine = np.array(
        [
            initial.combine_inv_bandwidth_s_per_byte,
            initial.combine_inv_single_sm_s_per_byte,
            initial.combine_fixed_s,
            initial.combine_cta_fixed_s,
        ],
        dtype=np.float64,
    )

    rate_bounds = (1e-16, 1e-6)
    fixed_bounds = (1e-9, 1e-2)
    main_lower = np.log(
        [rate_bounds[0], rate_bounds[0], fixed_bounds[0], 1e-4]
    )
    main_upper = np.log(
        [rate_bounds[1], rate_bounds[1], fixed_bounds[1], 4.0]
    )
    combine_lower = np.log(
        [rate_bounds[0], rate_bounds[0], fixed_bounds[0], fixed_bounds[0]]
    )
    combine_upper = np.log(
        [rate_bounds[1], rate_bounds[1], fixed_bounds[1], fixed_bounds[1]]
    )

    def main_residuals(theta):
        main_values = np.exp(theta)
        constants = constants_with(main_values, initial_combine)
        return np.array(
            [
                (
                    predict_partition(
                        observation.workload,
                        constants,
                        kv_tiles_per_cta=observation.kv_tiles_per_cta,
                    ).main_s
                    - observation.elapsed_s
                )
                / observation.elapsed_s
                for observation in main_observations
            ]
        )

    main_result = least_squares(
        main_residuals,
        np.log(initial_main),
        bounds=(main_lower, main_upper),
        max_nfev=512,
    )
    fitted_main = np.exp(main_result.x)

    def combine_residuals(theta):
        combine_values = np.exp(theta)
        constants = constants_with(fitted_main, combine_values)
        return np.array(
            [
                (
                    predict_partition(
                        observation.workload,
                        constants,
                        kv_tiles_per_cta=observation.kv_tiles_per_cta,
                    ).combine_s
                    - observation.elapsed_s
                )
                / observation.elapsed_s
                for observation in combine_observations
            ]
        )

    combine_result = least_squares(
        combine_residuals,
        np.log(initial_combine),
        bounds=(combine_lower, combine_upper),
        max_nfev=512,
    )
    fitted_main_theta = main_result.x
    fitted_combine_theta = combine_result.x

    if total_observations:
        total_by_workload = {}
        for observation in total_observations:
            total_by_workload.setdefault(observation.workload, []).append(
                observation
            )

        def total_and_route_residuals(constants):
            total_residuals = np.array(
                [
                    (
                        predict_partition(
                            observation.workload,
                            constants,
                            kv_tiles_per_cta=observation.kv_tiles_per_cta,
                        ).total_s
                        - observation.elapsed_s
                    )
                    / observation.elapsed_s
                    for observation in total_observations
                ]
            )
            route_delta_residuals = []
            for workload, workload_observations in total_by_workload.items():
                unsplit = next(
                    observation
                    for observation in workload_observations
                    if observation.kv_tiles_per_cta == workload.num_n_blocks
                )
                predicted_unsplit = predict_partition(
                    workload,
                    constants,
                    kv_tiles_per_cta=unsplit.kv_tiles_per_cta,
                ).total_s
                for observation in workload_observations:
                    predicted = predict_partition(
                        workload,
                        constants,
                        kv_tiles_per_cta=observation.kv_tiles_per_cta,
                    ).total_s
                    route_delta_residuals.append(
                        (
                            (predicted - predicted_unsplit)
                            - (observation.elapsed_s - unsplit.elapsed_s)
                        )
                        / (_ROUTE_DELTA_TOLERANCE * unsplit.elapsed_s)
                    )
            return np.concatenate(
                (
                    total_residuals / _COMPONENT_RESIDUAL_TOLERANCE,
                    np.asarray(route_delta_residuals),
                )
            )

        def route_aware_combine_residuals(combine_theta):
            constants = constants_with(
                fitted_main,
                np.exp(combine_theta),
            )
            return np.concatenate(
                (
                    combine_residuals(combine_theta)
                    / _COMPONENT_RESIDUAL_TOLERANCE,
                    total_and_route_residuals(constants),
                )
            )

        route_combine_result = least_squares(
            route_aware_combine_residuals,
            fitted_combine_theta,
            bounds=(combine_lower, combine_upper),
            max_nfev=1024,
        )
        fitted_combine_theta = route_combine_result.x
        fitted_combine = np.exp(fitted_combine_theta)

        def route_aware_main_residuals(main_theta):
            constants = constants_with(np.exp(main_theta), fitted_combine)
            return np.concatenate(
                (
                    main_residuals(main_theta)
                    / _COMPONENT_RESIDUAL_TOLERANCE,
                    total_and_route_residuals(constants),
                )
            )

        route_main_result = least_squares(
            route_aware_main_residuals,
            fitted_main_theta,
            bounds=(main_lower, main_upper),
            max_nfev=1024,
        )
        fitted_main_theta = route_main_result.x
        route_aware_success = (
            route_combine_result.success and route_main_result.success
        )
    else:
        fitted_combine = np.exp(fitted_combine_theta)
        route_aware_success = True

    fitted_main = np.exp(fitted_main_theta)
    main_rms, main_max = _relative_diagnostics(
        main_residuals(fitted_main_theta)
    )
    combine_rms, combine_max = _relative_diagnostics(
        combine_residuals(fitted_combine_theta)
    )
    if (
        not main_result.success
        or not combine_result.success
        or not route_aware_success
        or main_rms > max_relative_rms
        or combine_rms > max_relative_rms
    ):
        raise SplitKvCalibrationFitError(
            "SplitKV calibration fit is unusable: "
            f"main_rms={main_rms:.3f}, combine_rms={combine_rms:.3f}"
        )
    return SplitKvCalibrationFit(
        constants=constants_with(fitted_main, fitted_combine),
        main_relative_rms=main_rms,
        main_relative_max=main_max,
        combine_relative_rms=combine_rms,
        combine_relative_max=combine_max,
    )
