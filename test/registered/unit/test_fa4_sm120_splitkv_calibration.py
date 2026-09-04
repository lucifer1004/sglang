"""CPU tests for SM120 FA4 SplitKV calibration persistence."""

import json

import pytest

from sglang.kernels.ops.attention.fa4_sm120.splitkv_calibration import (
    SCHEMA_VERSION,
    SplitKvCalibrationCache,
    SplitKvCalibrationKey,
    default_calibration_path,
)
from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
    SplitKvCalibration,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _constants(sm_slots=188):
    return SplitKvCalibration(
        sm_slots=sm_slots,
        main_inv_bandwidth_s_per_byte=5e-13,
        main_inv_single_sm_s_per_byte=1.2e-10,
        main_fixed_s=6.4e-6,
        combine_inv_bandwidth_s_per_byte=8e-13,
        combine_inv_single_sm_s_per_byte=2e-11,
        combine_fixed_s=3.1e-6,
        combine_cta_fixed_s=8e-7,
        l2_cache_bytes=96 << 20,
    )


def _key(device="sm120-0", implementation="source-a", family="bf16-m64n128"):
    return SplitKvCalibrationKey(device, implementation, family)


def test_default_path_tracks_existing_sglang_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_CACHE_DIR", str(tmp_path))
    assert default_calibration_path() == (
        tmp_path / "fa4_sm120_splitkv_calibration.json"
    )


def test_default_path_accepts_explicit_override(monkeypatch, tmp_path):
    path = tmp_path / "custom.json"
    monkeypatch.setenv("SGLANG_FA4_SPLITKV_CALIBRATION_CACHE", str(path))
    assert default_calibration_path() == path


def test_constants_and_refinement_round_trip(tmp_path):
    path = tmp_path / "calibration.json"
    cache = SplitKvCalibrationCache(path)
    key = _key()
    saved = cache.save_constants(key, _constants())
    assert saved.constants == _constants()
    assert saved.refinements == {}

    refined = cache.save_refinement(key, "M1-N128-R12", 2)
    assert refined.constants == _constants()
    assert refined.refinements == {"M1-N128-R12": 2}

    reloaded = SplitKvCalibrationCache(path).get(key)
    assert reloaded == refined


def test_updates_merge_sibling_identities(tmp_path):
    path = tmp_path / "calibration.json"
    cache = SplitKvCalibrationCache(path)
    keys = (
        _key(),
        _key(family="bf16-m64n64"),
        _key(implementation="source-b"),
        _key(device="sm120-1"),
    )
    for index, key in enumerate(keys):
        cache.save_constants(key, _constants(sm_slots=188 - index))

    cold = SplitKvCalibrationCache(path)
    assert [cold.get(key).constants.sm_slots for key in keys] == [188, 187, 186, 185]


@pytest.mark.parametrize(
    "payload",
    (
        "not-json",
        json.dumps({"schema_version": SCHEMA_VERSION + 1, "devices": {}}),
        json.dumps({"schema_version": SCHEMA_VERSION, "devices": []}),
    ),
)
def test_corrupt_or_mismatched_documents_count_as_absent(tmp_path, payload):
    path = tmp_path / "calibration.json"
    path.write_text(payload, encoding="utf-8")
    assert SplitKvCalibrationCache(path).get(_key()) is None


def test_invalid_entry_does_not_publish_partial_state(tmp_path):
    path = tmp_path / "calibration.json"
    cache = SplitKvCalibrationCache(path)
    key = _key()
    cache.save_constants(key, _constants())
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["devices"][key.device]["implementations"][key.implementation][
        "route_families"
    ][key.route_family]
    raw["constants"]["main_fixed_s"] = -1
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert SplitKvCalibrationCache(path).get(key) is None


def test_refinement_requires_valid_constants(tmp_path):
    cache = SplitKvCalibrationCache(tmp_path / "calibration.json")
    with pytest.raises(ValueError, match="constants must exist"):
        cache.save_refinement(_key(), "shape", 4)


def test_mtime_refresh_observes_another_writer(tmp_path):
    path = tmp_path / "calibration.json"
    reader = SplitKvCalibrationCache(path)
    writer = SplitKvCalibrationCache(path)
    key = _key()
    assert reader.get(key) is None
    writer.save_constants(key, _constants())
    assert reader.get(key).constants == _constants()


def test_atomic_save_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "calibration.json"
    SplitKvCalibrationCache(path).save_constants(_key(), _constants())
    assert (
        json.loads(path.read_text(encoding="utf-8"))["schema_version"]
        == SCHEMA_VERSION
    )
    assert not list(tmp_path.glob("*.tmp"))
