"""CPU tests for SM120 FA4 calibrated SplitKV route consumption."""

import pytest
import torch

from sglang.kernels.ops.attention.fa4_sm120 import splitkv_router
from sglang.kernels.ops.attention.fa4_sm120.splitkv_calibration import (
    SplitKvCalibrationCache,
    SplitKvCalibrationEntry,
    SplitKvCalibrationKey,
)
from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
    SplitKvCalibration,
    SplitKvWorkload,
)
from sglang.kernels.ops.attention.fa4_sm120.splitkv_router import (
    SplitKvCalibrationMode,
    SplitKvCalibrationRegistry,
    SplitKvProbeSpec,
    SplitKvRouteSpec,
    splitkv_calibration_session,
    splitkv_workload_key,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _constants():
    return SplitKvCalibration(
        sm_slots=188,
        main_inv_bandwidth_s_per_byte=5e-13,
        main_inv_single_sm_s_per_byte=1.2e-10,
        main_fixed_s=6.4e-6,
        combine_inv_bandwidth_s_per_byte=8e-13,
        combine_inv_single_sm_s_per_byte=2e-11,
        combine_fixed_s=3.1e-6,
        combine_cta_fixed_s=8e-7,
        l2_cache_bytes=0,
    )


def _route():
    return SplitKvRouteSpec(
        kv_storage="bf16",
        compute="bf16",
        head_dim=128,
        head_dim_v=128,
        tile_m=64,
        tile_n=64,
        page_size=64,
        direct_uniform_batch=True,
        split_qk_n=False,
    )


def _workload():
    return SplitKvWorkload(
        total_mblocks=1,
        num_n_blocks=128,
        main_bytes_per_kv_tile=32768,
        output_rows=12,
        head_dim_v=128,
    )


def _patch_identity(monkeypatch):
    monkeypatch.setattr(
        splitkv_router,
        "splitkv_device_identity",
        lambda _device: ("gpu", {"sm_count": 188}),
    )
    monkeypatch.setattr(
        splitkv_router, "splitkv_implementation_identity", lambda: "implementation"
    )


def _probe():
    return SplitKvProbeSpec(1, 1, 6, 2, 8192, 64, True)


def test_route_family_is_algorithmic_not_exact_shape():
    assert _route().family == (
        "bf16-to-bf16-hd128-m64n64-paged64-gather-uniform-singleqk"
    )
    assert "num_n_blocks=128" in splitkv_workload_key(_workload())


def test_registry_loads_constants_and_honors_refinement(monkeypatch, tmp_path):
    _patch_identity(monkeypatch)
    cache = SplitKvCalibrationCache(tmp_path / "calibration.json")
    key = SplitKvCalibrationKey("gpu", "implementation", _route().family)
    cache.save_constants(key, _constants())
    cache.save_refinement(key, splitkv_workload_key(_workload()), 7)
    registry = SplitKvCalibrationRegistry(cache)

    with splitkv_calibration_session(SplitKvCalibrationMode.LOAD):
        decision = registry.resolve(
            route=_route(),
            workload=_workload(),
            device=torch.device("cpu"),
            is_stream_capturing=False,
        )

    assert decision is not None
    assert decision.refined
    assert decision.prediction.kv_tiles_per_cta == 7
    assert decision.prediction.num_splits == 19


def test_capture_never_loads_disk(monkeypatch, tmp_path):
    _patch_identity(monkeypatch)
    cache = SplitKvCalibrationCache(tmp_path / "calibration.json")
    key = SplitKvCalibrationKey("gpu", "implementation", _route().family)
    cache.save_constants(key, _constants())
    registry = SplitKvCalibrationRegistry(cache)

    with splitkv_calibration_session(SplitKvCalibrationMode.LOAD):
        assert (
            registry.resolve(
                route=_route(),
                workload=_workload(),
                device=torch.device("cpu"),
                is_stream_capturing=True,
            )
            is None
        )

        warm = registry.resolve(
            route=_route(),
            workload=_workload(),
            device=torch.device("cpu"),
            is_stream_capturing=False,
        )
        replay = registry.resolve(
            route=_route(),
            workload=_workload(),
            device=torch.device("cpu"),
            is_stream_capturing=True,
        )

    assert warm == replay


def test_off_mode_ignores_a_published_entry(monkeypatch, tmp_path):
    _patch_identity(monkeypatch)
    registry = SplitKvCalibrationRegistry(
        SplitKvCalibrationCache(tmp_path / "calibration.json")
    )
    key = SplitKvCalibrationKey("gpu", "implementation", _route().family)
    registry.publish(key, SplitKvCalibrationEntry(_constants(), {}))

    with splitkv_calibration_session(SplitKvCalibrationMode.OFF):
        assert (
            registry.resolve(
                route=_route(),
                workload=_workload(),
                device=torch.device("cpu"),
                is_stream_capturing=False,
            )
            is None
        )


def test_tune_mode_fills_family_and_exact_workload(monkeypatch, tmp_path):
    from sglang.kernels.ops.attention.fa4_sm120 import splitkv_tuner

    _patch_identity(monkeypatch)
    calls = []
    monkeypatch.setattr(
        splitkv_tuner,
        "calibrate_route_family",
        lambda **_kwargs: calls.append("calibrate") or _constants(),
    )
    monkeypatch.setattr(
        splitkv_tuner,
        "refine_route_workload",
        lambda **_kwargs: calls.append("refine") or 7,
    )
    cache = SplitKvCalibrationCache(tmp_path / "calibration.json")
    registry = SplitKvCalibrationRegistry(cache)

    with splitkv_calibration_session(
        SplitKvCalibrationMode.TUNE, allow_tuning=True
    ):
        decision = registry.resolve(
            route=_route(),
            workload=_workload(),
            probe=_probe(),
            device=torch.device("cpu"),
            is_stream_capturing=False,
        )
        repeated = registry.resolve(
            route=_route(),
            workload=_workload(),
            probe=_probe(),
            device=torch.device("cpu"),
            is_stream_capturing=False,
        )

    assert calls == ["calibrate", "refine"]
    assert decision == repeated
    assert decision is not None and decision.refined
    assert decision.prediction.kv_tiles_per_cta == 7


def test_tuner_rejects_cuda_graph_capture_before_allocating(monkeypatch):
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_tuner import (
        SplitKvTuningError,
        calibrate_route_family,
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(SplitKvTuningError, match="during CUDA Graph capture"):
        calibrate_route_family(
            route=_route(),
            workload=_workload(),
            probe=_probe(),
            device=torch.device("cpu"),
        )
