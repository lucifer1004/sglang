# Copyright (c) 2026, SGLang Team.
"""Process-local calibrated routing for SM120 FA4 SplitKV.

The hot path consumes a frozen in-memory snapshot.  Disk discovery is allowed
only before CUDA Graph recording; a first lookup made under capture is treated
as an ordinary cache miss and retains the architecture-owned safe heuristic.
"""

from __future__ import annotations

import contextlib
import contextvars
import enum
import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from sglang.kernels.ops.attention.fa4_sm120.splitkv_calibration import (
    SplitKvCalibrationCache,
    SplitKvCalibrationEntry,
    SplitKvCalibrationKey,
    splitkv_device_identity,
    splitkv_implementation_identity,
)
from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
    SplitKvPrediction,
    SplitKvWorkload,
    partition_selection_is_ambiguous,
    select_partition,
)

logger = logging.getLogger(__name__)


class SplitKvCalibrationMode(enum.Enum):
    """Production behavior for calibrated routing."""

    OFF = "off"
    LOAD = "load"
    TUNE = "tune"
    FORCE = "force"

    @classmethod
    def from_value(cls, value: Optional[str]) -> "SplitKvCalibrationMode":
        normalized = "load" if value is None else value.strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"invalid SM120 FA4 SplitKV calibration mode {value!r}; "
                f"expected one of: {choices}"
            ) from error


def configured_calibration_mode() -> SplitKvCalibrationMode:
    return SplitKvCalibrationMode.from_value(
        os.getenv("SGLANG_FA4_SPLITKV_CALIBRATION")
    )


def configured_refinement_limit() -> int:
    raw = os.getenv("SGLANG_FA4_SPLITKV_MAX_REFINEMENTS", "8")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "SGLANG_FA4_SPLITKV_MAX_REFINEMENTS must be an integer"
        ) from error
    if value < 0:
        raise ValueError("SGLANG_FA4_SPLITKV_MAX_REFINEMENTS must be non-negative")
    return value


@dataclass(frozen=True)
class SplitKvRouteSpec:
    """Kernel/dataflow facts shared by one calibrated route family."""

    kv_storage: str
    compute: str
    head_dim: int
    head_dim_v: int
    tile_m: int
    tile_n: int
    page_size: int
    direct_uniform_batch: bool
    split_qk_n: bool

    def __post_init__(self) -> None:
        if not self.kv_storage or not self.compute:
            raise ValueError("route storage and compute identities must be non-empty")
        if min(
            self.head_dim,
            self.head_dim_v,
            self.tile_m,
            self.tile_n,
            self.page_size,
        ) <= 0:
            raise ValueError("route dimensions must be positive")

    @property
    def family(self) -> str:
        hdim = (
            f"hd{self.head_dim}"
            if self.head_dim == self.head_dim_v
            else f"hd{self.head_dim}x{self.head_dim_v}"
        )
        scheduler = "uniform" if self.direct_uniform_batch else "varlen"
        qk = "splitqk" if self.split_qk_n else "singleqk"
        return (
            f"{self.kv_storage}-to-{self.compute}-{hdim}-"
            f"m{self.tile_m}n{self.tile_n}-paged{self.page_size}-"
            f"gather-{scheduler}-{qk}"
        )


@dataclass(frozen=True)
class SplitKvRouteDecision:
    """Calibrated selection plus the identity used to obtain it."""

    key: SplitKvCalibrationKey
    workload_key: str
    prediction: SplitKvPrediction
    refined: bool


@dataclass(frozen=True)
class SplitKvProbeSpec:
    """Concrete paged call geometry used only by startup tuning."""

    batch_size: int
    num_head_kv: int
    qhead_per_kvhead: int
    max_seqlen_q: int
    max_seqlen_k: int
    page_size: int
    causal: bool

    def __post_init__(self) -> None:
        positive = (
            self.batch_size,
            self.num_head_kv,
            self.qhead_per_kvhead,
            self.max_seqlen_q,
            self.max_seqlen_k,
            self.page_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("probe dimensions must be positive")


def splitkv_workload_key(workload: SplitKvWorkload) -> str:
    """Return an exact, readable key for refinement overrides."""
    values = asdict(workload)
    return "|".join(f"{name}={values[name]}" for name in sorted(values))


def kv_storage_identity(dtype: torch.dtype) -> Optional[str]:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float8_e4m3fn:
        return "fp8e4m3"
    return None


@dataclass(frozen=True)
class _CalibrationSession:
    mode: SplitKvCalibrationMode
    process_group: object = None
    allow_tuning: bool = False


_SESSION = contextvars.ContextVar(
    "sm120_fa4_splitkv_calibration_session",
    default=_CalibrationSession(configured_calibration_mode()),
)


@contextlib.contextmanager
def splitkv_calibration_session(
    mode: SplitKvCalibrationMode | str | None = None,
    *,
    process_group=None,
    allow_tuning: bool = True,
):
    """Set the startup calibration contract for nested FA4 forwards."""
    resolved = (
        configured_calibration_mode()
        if mode is None
        else mode
        if isinstance(mode, SplitKvCalibrationMode)
        else SplitKvCalibrationMode.from_value(mode)
    )
    token = _SESSION.set(_CalibrationSession(resolved, process_group, allow_tuning))
    try:
        yield
    finally:
        _SESSION.reset(token)


def current_calibration_session() -> _CalibrationSession:
    return _SESSION.get()


class SplitKvCalibrationRegistry:
    """Frozen process snapshot used by route resolution and graph capture."""

    def __init__(self, cache: Optional[SplitKvCalibrationCache] = None) -> None:
        self.cache = SplitKvCalibrationCache() if cache is None else cache
        self._entries: dict[
            SplitKvCalibrationKey, Optional[SplitKvCalibrationEntry]
        ] = {}
        self._decisions: dict[
            tuple[SplitKvCalibrationKey, str], Optional[SplitKvRouteDecision]
        ] = {}
        self._capture_decisions: dict[
            tuple[str, str, str], Optional[SplitKvRouteDecision]
        ] = {}
        self._lock = threading.RLock()
        self._forced_families: set[SplitKvCalibrationKey] = set()
        self._forced_workloads: set[tuple[SplitKvCalibrationKey, str]] = set()
        self._failed_families: set[SplitKvCalibrationKey] = set()
        self._failed_workloads: set[tuple[SplitKvCalibrationKey, str]] = set()
        self._group_device_consensus: dict[int, bool] = {}
        self._refinement_attempts: dict[SplitKvCalibrationKey, int] = {}
        self._version = 0

    @property
    def version(self) -> int:
        """Monotonic process version for launch-plan cache invalidation."""
        with self._lock:
            return self._version

    def reset(self) -> None:
        """Clear process snapshots; intended for tests and explicit recapture."""
        with self._lock:
            self._entries.clear()
            self._decisions.clear()
            self._capture_decisions.clear()
            self._forced_families.clear()
            self._forced_workloads.clear()
            self._failed_families.clear()
            self._failed_workloads.clear()
            self._group_device_consensus.clear()
            self._refinement_attempts.clear()
            self._version += 1

    def publish(
        self,
        key: SplitKvCalibrationKey,
        entry: Optional[SplitKvCalibrationEntry],
    ) -> None:
        """Publish one startup result and invalidate its memoized decisions."""
        with self._lock:
            self._entries[key] = entry
            self._version += 1
            stale = [
                decision_key
                for decision_key in self._decisions
                if decision_key[0] == key
            ]
            for decision_key in stale:
                del self._decisions[decision_key]
            stale_capture = [
                capture_key
                for capture_key in self._capture_decisions
                if capture_key[1] == key.route_family
            ]
            for capture_key in stale_capture:
                del self._capture_decisions[capture_key]

    def entry(
        self,
        key: SplitKvCalibrationKey,
        *,
        is_stream_capturing: bool,
    ) -> Optional[SplitKvCalibrationEntry]:
        with self._lock:
            if key in self._entries:
                return self._entries[key]
            if is_stream_capturing:
                # Loading JSON, hashing mtimes, and taking locks are outside the
                # graph contract.  The subsequent graph warmup can publish it.
                return None
            entry = self.cache.get(key)
            self._entries[key] = entry
            return entry

    def resolve(
        self,
        *,
        route: SplitKvRouteSpec,
        workload: SplitKvWorkload,
        device: torch.device,
        is_stream_capturing: bool,
        probe: Optional[SplitKvProbeSpec] = None,
    ) -> Optional[SplitKvRouteDecision]:
        session = current_calibration_session()
        if session.mode is SplitKvCalibrationMode.OFF:
            return None
        workload_key = splitkv_workload_key(workload)
        capture_key = (str(device), route.family, workload_key)
        if is_stream_capturing:
            # The matching eager graph warmup must have published this exact
            # decision.  A miss stays on the safe heuristic without device
            # queries, source hashing, cache reads, locks, or tuning.
            with self._lock:
                return self._capture_decisions.get(capture_key)
        device_key, device_facts = splitkv_device_identity(device)
        key = SplitKvCalibrationKey(
            device=device_key,
            implementation=splitkv_implementation_identity(),
            route_family=route.family,
        )
        decision_key = (key, workload_key)
        with self._lock:
            if (
                decision_key in self._decisions
                and (
                    is_stream_capturing
                    or session.mode
                    not in (
                        SplitKvCalibrationMode.TUNE,
                        SplitKvCalibrationMode.FORCE,
                    )
                )
            ):
                return self._decisions[decision_key]
        if (
            not is_stream_capturing
            and session.allow_tuning
            and session.mode
            in (SplitKvCalibrationMode.TUNE, SplitKvCalibrationMode.FORCE)
            and session.process_group is not None
        ):
            if not self._homogeneous_group(session.process_group, device_key):
                logger.warning(
                    "SM120 FA4 SplitKV startup tuning requires a homogeneous "
                    "TP device group; retaining the safe heuristic"
                )
                return None
            with self._lock:
                entry_is_synced = key in self._entries
                entry = self._entries.get(key)
            if not entry_is_synced:
                entry = self._run_on_group_leader(
                    session.process_group, lambda: self.cache.get(key)
                )
                self.publish(key, entry)
        else:
            entry = self.entry(key, is_stream_capturing=is_stream_capturing)
        if (
            not is_stream_capturing
            and probe is not None
            and session.allow_tuning
            and session.mode
            in (SplitKvCalibrationMode.TUNE, SplitKvCalibrationMode.FORCE)
        ):
            force_family = (
                session.mode is SplitKvCalibrationMode.FORCE
                and key not in self._forced_families
            )
            family_probe_is_useful = workload.total_mblocks < int(
                device_facts["sm_count"]
            )
            if (
                entry is None
                and key not in self._failed_families
                and family_probe_is_useful
            ) or force_family:
                entry = self._tune_family(
                    key=key,
                    route=route,
                    workload=workload,
                    probe=probe,
                    device=device,
                    process_group=session.process_group,
                )
                if entry is None:
                    self._failed_families.add(key)
                if session.mode is SplitKvCalibrationMode.FORCE:
                    self._forced_families.add(key)
            workload_identity = (key, workload_key)
            force_workload = (
                session.mode is SplitKvCalibrationMode.FORCE
                and workload_identity not in self._forced_workloads
            )
            proposal = (
                None
                if entry is None
                else select_partition(workload, entry.constants)
            )
            refinement_is_useful = (
                proposal is not None
                and partition_selection_is_ambiguous(
                    workload,
                    entry.constants,
                )
                and self._refinement_attempts.get(key, 0)
                < configured_refinement_limit()
            )
            if entry is not None and refinement_is_useful and (
                (
                    workload_key not in entry.refinements
                    and workload_identity not in self._failed_workloads
                )
                or force_workload
            ):
                before = entry
                self._refinement_attempts[key] = (
                    self._refinement_attempts.get(key, 0) + 1
                )
                entry = self._refine_workload(
                    key=key,
                    entry=entry,
                    route=route,
                    workload=workload,
                    workload_key=workload_key,
                    probe=probe,
                    device=device,
                    process_group=session.process_group,
                )
                if entry is before:
                    self._failed_workloads.add(workload_identity)
                if session.mode is SplitKvCalibrationMode.FORCE:
                    self._forced_workloads.add(workload_identity)
        if entry is None:
            decision = None
        else:
            refined_grain = entry.refinements.get(workload_key)
            if refined_grain is None:
                prediction = select_partition(workload, entry.constants)
            else:
                from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
                    predict_partition,
                )

                # Keep validation in the pure model: a stale or impossible
                # override is an unusable entry, not an instruction to launch
                # an empty partition.
                try:
                    prediction = predict_partition(
                        workload,
                        entry.constants,
                        kv_tiles_per_cta=refined_grain,
                    )
                except ValueError:
                    prediction = None
            decision = (
                None
                if prediction is None
                else SplitKvRouteDecision(
                    key=key,
                    workload_key=workload_key,
                    prediction=prediction,
                    refined=refined_grain is not None,
                )
            )
        with self._lock:
            self._decisions[decision_key] = decision
            self._capture_decisions[capture_key] = decision
        return decision

    def _homogeneous_group(self, process_group, device_key: str) -> bool:
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(process_group) > 1
        ):
            return True
        group_id = id(process_group)
        with self._lock:
            cached = self._group_device_consensus.get(group_id)
        if cached is not None:
            return cached
        gathered = [None] * torch.distributed.get_world_size(process_group)
        torch.distributed.all_gather_object(
            gathered, device_key, group=process_group
        )
        homogeneous = len(set(gathered)) == 1
        with self._lock:
            self._group_device_consensus[group_id] = homogeneous
        return homogeneous

    @staticmethod
    def _run_on_group_leader(process_group, operation):
        """Run a fallible tuning operation once and broadcast its result."""
        distributed = (
            process_group is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(process_group) > 1
        )
        rank = torch.distributed.get_rank(process_group) if distributed else 0
        payload = [None]
        if rank == 0:
            try:
                payload[0] = (True, operation())
            except (OSError, RuntimeError, ValueError, MemoryError) as error:
                payload[0] = (False, f"{type(error).__name__}: {error}")
        if distributed:
            torch.distributed.broadcast_object_list(
                payload, group=process_group, group_src=0
            )
        succeeded, result = payload[0]
        if not succeeded:
            logger.warning("SM120 FA4 SplitKV startup tuning failed: %s", result)
            return None
        return result

    def _tune_family(
        self,
        *,
        key: SplitKvCalibrationKey,
        route: SplitKvRouteSpec,
        workload: SplitKvWorkload,
        probe: SplitKvProbeSpec,
        device: torch.device,
        process_group,
    ) -> Optional[SplitKvCalibrationEntry]:
        def operation():
            from sglang.kernels.ops.attention.fa4_sm120.splitkv_tuner import (
                calibrate_route_family,
            )

            logger.info(
                "Calibrating SM120 FA4 SplitKV route family %s on %s",
                route.family,
                device,
            )
            try:
                constants = calibrate_route_family(
                    route=route,
                    workload=workload,
                    probe=probe,
                    device=device,
                )
            finally:
                torch.cuda.empty_cache()
            logger.info(
                "Calibrated SM120 FA4 SplitKV route family %s", route.family
            )
            try:
                return self.cache.save_constants(key, constants)
            except (OSError, ValueError) as error:
                logger.warning(
                    "SM120 FA4 SplitKV constants could not be persisted to %s "
                    "(%s); using them in this process only.",
                    self.cache.path,
                    error,
                )
                return SplitKvCalibrationEntry(constants, {})

        entry = self._run_on_group_leader(process_group, operation)
        self.publish(key, entry)
        return entry

    def _refine_workload(
        self,
        *,
        key: SplitKvCalibrationKey,
        entry: SplitKvCalibrationEntry,
        route: SplitKvRouteSpec,
        workload: SplitKvWorkload,
        workload_key: str,
        probe: SplitKvProbeSpec,
        device: torch.device,
        process_group,
    ) -> Optional[SplitKvCalibrationEntry]:
        def operation():
            from sglang.kernels.ops.attention.fa4_sm120.splitkv_tuner import (
                refine_route_workload,
            )

            logger.info(
                "Refining SM120 FA4 SplitKV route %s for %s",
                route.family,
                workload_key,
            )
            try:
                grain = refine_route_workload(
                    route=route,
                    workload=workload,
                    probe=probe,
                    constants=entry.constants,
                    device=device,
                )
            finally:
                torch.cuda.empty_cache()
            logger.info(
                "Refined SM120 FA4 SplitKV route %s to %d KV tiles per CTA",
                route.family,
                grain,
            )
            try:
                return self.cache.save_refinement(key, workload_key, grain)
            except (OSError, ValueError) as error:
                logger.warning(
                    "SM120 FA4 SplitKV refinement could not be persisted to %s "
                    "(%s); using it in this process only.",
                    self.cache.path,
                    error,
                )
                refinements = dict(entry.refinements)
                refinements[workload_key] = grain
                return SplitKvCalibrationEntry(entry.constants, refinements)

        refined = self._run_on_group_leader(process_group, operation)
        if refined is None:
            return entry
        self.publish(key, refined)
        return refined


splitkv_calibration_registry = SplitKvCalibrationRegistry()
