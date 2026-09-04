# Copyright (c) 2026, SGLang Team.
"""Persistent device-local calibration state for SM120 FA4 SplitKV routing."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
    SplitKvCalibration,
)


SCHEMA_VERSION = 2
_CACHE_FILENAME = "fa4_sm120_splitkv_calibration.json"
_PROCESS_LOCKS: dict[Path, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


@lru_cache(maxsize=1)
def splitkv_implementation_identity() -> str:
    """Hash every source file that can change SplitKV timing or selection.

    Resolve paths without importing the runtime modules.  This keeps the
    identity helper safe to call from ``runtime.py`` while that module owns the
    launch policy importing this cache module.
    """
    fa4_dir = Path(__file__).resolve().parent
    attention_dir = fa4_dir.parent
    sources = (
        fa4_dir / "flash_fwd.py",
        fa4_dir / "flash_fwd_decode.py",
        fa4_dir / "paged_kv.py",
        fa4_dir / "fp8_kv.py",
        fa4_dir / "runtime.py",
        fa4_dir / "splitkv_router.py",
        fa4_dir / "splitkv_tuner.py",
        fa4_dir / "splitkv_model.py",
        fa4_dir / "splitkv_fit.py",
        attention_dir / "flash_attn" / "cute" / "flash_fwd_combine.py",
    )
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            # An absent optional specialization must have a stable identity
            # distinct from a present empty file.
            digest.update(b"<absent>")
    return digest.hexdigest()


def splitkv_device_identity(device) -> tuple[str, dict[str, object]]:
    """Return the stable hardware key and inspectable facts for ``device``."""
    import torch

    device = torch.device(device)
    properties = torch.cuda.get_device_properties(device)
    facts = {
        "name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": properties.multi_processor_count,
        "l2_cache_bytes": int(getattr(properties, "L2_cache_size", 0) or 0),
        "memory_bus_width": properties.memory_bus_width,
    }
    encoded = json.dumps(facts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest(), facts


def default_calibration_path() -> Path:
    """Return the local calibration path under the existing SGLang cache."""
    override = os.getenv("SGLANG_FA4_SPLITKV_CALIBRATION_CACHE")
    if override:
        return Path(override).expanduser()
    cache_root = os.getenv("SGLANG_CACHE_DIR")
    if cache_root:
        root = Path(cache_root).expanduser()
    else:
        root = Path.home() / ".cache" / "sglang"
    return root / _CACHE_FILENAME


@dataclass(frozen=True)
class SplitKvCalibrationKey:
    """Stable cache identity for one device and compiled route family."""

    device: str
    implementation: str
    route_family: str

    def __post_init__(self) -> None:
        if not self.device or not self.implementation or not self.route_family:
            raise ValueError("calibration key fields must be non-empty")


@dataclass(frozen=True)
class SplitKvCalibrationEntry:
    """Calibrated constants and optional exact-workload grain overrides."""

    constants: SplitKvCalibration
    refinements: dict[str, int]

    def __post_init__(self) -> None:
        if any(not key or grain <= 0 for key, grain in self.refinements.items()):
            raise ValueError("refinement keys must be non-empty and grains positive")


def _empty_payload() -> dict:
    return {"schema_version": SCHEMA_VERSION, "devices": {}}


def _process_lock(path: Path) -> threading.Lock:
    resolved = path.expanduser().resolve()
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(resolved, threading.Lock())


@contextlib.contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _empty_payload()
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("devices"), dict)
    ):
        return _empty_payload()
    return payload


def _raw_entry(payload: dict, key: SplitKvCalibrationKey) -> Optional[dict]:
    try:
        entry = payload["devices"][key.device]["implementations"][
            key.implementation
        ]["route_families"][key.route_family]
    except (KeyError, TypeError):
        return None
    return entry if isinstance(entry, dict) else None


def _parse_entry(raw: Optional[dict]) -> Optional[SplitKvCalibrationEntry]:
    if raw is None:
        return None
    try:
        constants = SplitKvCalibration(**raw["constants"])
        refinements_raw = raw.get("refinements", {})
        if not isinstance(refinements_raw, dict):
            return None
        refinements = {str(key): int(value) for key, value in refinements_raw.items()}
        return SplitKvCalibrationEntry(constants, refinements)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _entry_for_update(payload: dict, key: SplitKvCalibrationKey) -> dict:
    devices = payload.setdefault("devices", {})
    device = devices.setdefault(key.device, {})
    implementations = device.setdefault("implementations", {})
    implementation = implementations.setdefault(key.implementation, {})
    route_families = implementation.setdefault("route_families", {})
    entry = route_families.setdefault(key.route_family, {})
    if not isinstance(entry, dict):
        entry = route_families[key.route_family] = {}
    return entry


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


class SplitKvCalibrationCache:
    """Mtime-aware process cache backed by a locked atomic JSON document."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = (
            default_calibration_path() if path is None else Path(path).expanduser()
        )
        self._mtime_ns = -1
        self._payload = _empty_payload()

    def _refresh(self) -> None:
        try:
            mtime_ns = self.path.stat().st_mtime_ns
        except OSError:
            mtime_ns = -1
        if mtime_ns == self._mtime_ns:
            return
        self._payload = _read_payload(self.path)
        self._mtime_ns = mtime_ns

    def get(self, key: SplitKvCalibrationKey) -> Optional[SplitKvCalibrationEntry]:
        """Return a valid entry, or ``None`` for absent or unusable state."""
        self._refresh()
        return _parse_entry(_raw_entry(self._payload, key))

    def save_constants(
        self,
        key: SplitKvCalibrationKey,
        constants: SplitKvCalibration,
    ) -> SplitKvCalibrationEntry:
        """Merge constants without discarding existing exact-shape refinements."""
        with _process_lock(self.path), _exclusive_file_lock(self.path):
            payload = _read_payload(self.path)
            entry = _entry_for_update(payload, key)
            entry["constants"] = asdict(constants)
            if not isinstance(entry.get("refinements"), dict):
                entry["refinements"] = {}
            _atomic_write(self.path, payload)
        self._mtime_ns = -1
        loaded = self.get(key)
        if loaded is None:
            raise RuntimeError("saved SplitKV calibration could not be reloaded")
        return loaded

    def save_refinement(
        self,
        key: SplitKvCalibrationKey,
        workload_key: str,
        kv_tiles_per_cta: int,
    ) -> SplitKvCalibrationEntry:
        """Merge one measured exact-workload override into an existing entry."""
        if not workload_key or kv_tiles_per_cta <= 0:
            raise ValueError("refinement key must be non-empty and grain positive")
        with _process_lock(self.path), _exclusive_file_lock(self.path):
            payload = _read_payload(self.path)
            entry = _entry_for_update(payload, key)
            if _parse_entry(entry) is None:
                raise ValueError("calibrated constants must exist before refinement")
            refinements = entry.setdefault("refinements", {})
            refinements[workload_key] = int(kv_tiles_per_cta)
            _atomic_write(self.path, payload)
        self._mtime_ns = -1
        loaded = self.get(key)
        if loaded is None:
            raise RuntimeError("saved SplitKV refinement could not be reloaded")
        return loaded
