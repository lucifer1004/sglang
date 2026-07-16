from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import logging
import math
import mmap
import multiprocessing as mp
import os
import shutil
import signal
import stat
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.utils.shared_uva_tensor import SharedTensorFileSpec


_SHARED_EMBEDDING_KEYS = (
    "model.embed_tokens.weight",
    "model.oe_embed.0.weight",
    "model.oe_embed.1.weight",
    "model.oe_embed.2.weight",
    "model.oe_embed.3.weight",
)
_MANIFEST_SCHEMA_VERSION = 1
_NUMA_SAMPLE_COUNT = 127
_COPY_CHUNK_BYTES = 256 * 1024 * 1024
_ARENA_PREFIX = "sglang-welm-embedding-"
_LEASE_DIRECTORY_NAME = ".sglang-welm-embedding-leases"
_GC_LOCK_FILE_NAME = ".sglang-welm-embedding-gc.lock"
_OWNER_FILE_NAME = ".sglang-welm-embedding-owner.json"
_ARENA_PARENT = Path("/dev/shm")
logger = logging.getLogger(__name__)


def get_welm_shared_embedding_weight_names() -> frozenset[str]:
    return frozenset(_SHARED_EMBEDDING_KEYS)


class WeLMSharedEmbeddingPolicy(str, Enum):
    DISABLED = "disabled"
    BIND = "bind"
    INTERLEAVE = "interleave"
    REPLICATE_NUMA = "replicate-numa"


class NumaPlacementMode(str, Enum):
    BIND = "bind"
    INTERLEAVE = "interleave"


@dataclass(frozen=True)
class WeLMEmbeddingReplicaPlan:
    replica_id: str
    numa_nodes: tuple[int, ...]
    consumer_local_ranks: tuple[int, ...]


@dataclass(frozen=True)
class WeLMEmbeddingByteCounts:
    base_bytes: int
    oe_bytes: int
    logical_bytes: int
    physical_bytes: int


@dataclass(frozen=True)
class WeLMEmbeddingReplicaManifest:
    replica_id: str
    numa_nodes: tuple[int, ...]
    tensors: tuple[SharedTensorFileSpec, ...]


@dataclass(frozen=True)
class WeLMEmbeddingArenaManifest:
    schema_version: int
    arena_id: str
    checkpoint_identity: str
    policy: str
    logical_weight_bytes: int
    physical_weight_bytes: int
    replicas: tuple[WeLMEmbeddingReplicaManifest, ...]
    manager_pid: int


@dataclass(frozen=True)
class WeLMStaleArenaCleanupReport:
    cleaned: tuple[str, ...]
    active: tuple[str, ...]
    skipped: tuple[str, ...]


class WeLMEmbeddingArenaManagerStillAliveError(RuntimeError):
    def __init__(self, process) -> None:
        self.process = process
        super().__init__(
            f"WeLM embedding arena manager {process.pid} survived SIGKILL"
        )


@dataclass(frozen=True)
class _WeLMEmbeddingLeaseMetadata:
    ownership_id: str
    manager_pid: int | None
    manager_start_time: int | None


@dataclass
class WeLMEmbeddingArenaLease:
    arena_root: Path
    lease_path: Path
    run_id: str
    ownership_id: str
    fd: int
    owner_pid: int
    owner_start_time: int | None
    created_ns: int
    manager_pid: int | None = None
    manager_start_time: int | None = None
    _closed: bool = False

    def update_manager_pid(self, manager_pid: int) -> None:
        if self._closed:
            raise RuntimeError("cannot update a closed WeLM embedding arena lease")
        if manager_pid <= 0:
            raise ValueError("arena manager PID must be positive")
        self.manager_pid = manager_pid
        self.manager_start_time = _read_process_start_time(manager_pid)
        _write_welm_embedding_lease_metadata(self)

    def close(self, *, remove_file: bool = True) -> None:
        if self._closed:
            return
        errors = []
        if remove_file:
            try:
                _unlink_locked_file_if_same(self.lease_path, self.fd)
            except BaseException as exc:
                errors.append(exc)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        except BaseException as exc:
            errors.append(exc)
        try:
            os.close(self.fd)
        except BaseException as exc:
            errors.append(exc)
        self._closed = True
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(
                "WeLM embedding arena lease cleanup failed", errors
            )


_QUARANTINED_WELM_ARENA_LEASES: list[WeLMEmbeddingArenaLease] = []


def _quarantine_welm_embedding_arena_lease(
    lease: WeLMEmbeddingArenaLease | None,
) -> None:
    if lease is not None and not any(
        existing is lease for existing in _QUARANTINED_WELM_ARENA_LEASES
    ):
        _QUARANTINED_WELM_ARENA_LEASES.append(lease)


def _forget_quarantined_welm_embedding_arena_lease(
    lease: WeLMEmbeddingArenaLease,
) -> None:
    _QUARANTINED_WELM_ARENA_LEASES[:] = [
        existing
        for existing in _QUARANTINED_WELM_ARENA_LEASES
        if existing is not lease
    ]


def _validate_welm_embedding_run_id(run_id: str) -> None:
    if not run_id or any(
        char
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in run_id
    ):
        raise ValueError(
            "WeLM embedding arena run ID must contain only ASCII letters, "
            "digits, hyphen, and underscore"
        )


def _welm_embedding_lease_directory(arena_parent: Path) -> Path:
    lease_directory = arena_parent / _LEASE_DIRECTORY_NAME
    lease_directory.mkdir(mode=0o700, exist_ok=True)
    directory_stat = lease_directory.lstat()
    if (
        lease_directory.is_symlink()
        or not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_uid != os.geteuid()
        or stat.S_IMODE(directory_stat.st_mode) & 0o077
    ):
        raise RuntimeError(
            f"invalid WeLM embedding lease directory: {lease_directory}"
        )
    return lease_directory


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])


def _unlink_locked_file_if_same(path: Path, fd: int) -> bool:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return True
    fd_stat = os.fstat(fd)
    if (
        stat.S_ISLNK(path_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or (path_stat.st_dev, path_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino)
    ):
        logger.warning(
            "Refusing to unlink replaced WeLM embedding lease: %s",
            path,
        )
        return False
    path.unlink()
    return True


def _write_welm_embedding_lease_metadata(
    lease: WeLMEmbeddingArenaLease,
) -> None:
    payload = json.dumps(
        {
            "run_id": lease.run_id,
            "arena_root": str(lease.arena_root),
            "ownership_id": lease.ownership_id,
            "owner_pid": lease.owner_pid,
            "owner_start_time": lease.owner_start_time,
            "created_ns": lease.created_ns,
            "manager_pid": lease.manager_pid,
            "manager_start_time": lease.manager_start_time,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    os.lseek(lease.fd, 0, os.SEEK_SET)
    os.ftruncate(lease.fd, 0)
    _write_all(lease.fd, payload)
    os.fsync(lease.fd)


def _open_and_lock_file(
    path: Path, *, blocking: bool, create: bool = True
) -> int | None:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if create:
        flags |= os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    file_stat = os.fstat(fd)
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_nlink != 1
        or file_stat.st_uid != os.geteuid()
        or stat.S_IMODE(file_stat.st_mode) & 0o077
    ):
        os.close(fd)
        raise RuntimeError(f"unsafe WeLM embedding lock file: {path}")
    lock_flags = fcntl.LOCK_EX
    if not blocking:
        lock_flags |= fcntl.LOCK_NB
    try:
        fcntl.flock(fd, lock_flags)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    return fd


def _validate_welm_embedding_ownership_id(ownership_id: str) -> None:
    if len(ownership_id) != 32 or any(
        char not in "0123456789abcdef" for char in ownership_id
    ):
        raise ValueError("WeLM embedding ownership ID must be 32 lowercase hex digits")


def _read_locked_welm_embedding_lease_metadata(
    fd: int, *, expected_run_id: str, expected_arena_root: Path
) -> _WeLMEmbeddingLeaseMetadata | None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        payload_bytes = bytearray()
        while len(payload_bytes) <= 64 * 1024:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            payload_bytes.extend(chunk)
        if len(payload_bytes) > 64 * 1024:
            return None
        payload = json.loads(payload_bytes)
        ownership_id = payload["ownership_id"]
        _validate_welm_embedding_ownership_id(ownership_id)
        if payload["run_id"] != expected_run_id:
            return None
        if Path(payload["arena_root"]) != expected_arena_root:
            return None
        manager_pid = payload.get("manager_pid")
        manager_start_time = payload.get("manager_start_time")
        if manager_pid is not None and (
            isinstance(manager_pid, bool)
            or not isinstance(manager_pid, int)
            or manager_pid <= 0
        ):
            return None
        if manager_start_time is not None and (
            isinstance(manager_start_time, bool)
            or not isinstance(manager_start_time, int)
            or manager_start_time <= 0
        ):
            return None
        if manager_pid is None and manager_start_time is not None:
            return None
        return _WeLMEmbeddingLeaseMetadata(
            ownership_id=ownership_id,
            manager_pid=manager_pid,
            manager_start_time=manager_start_time,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None


def _is_registered_welm_embedding_manager_alive(
    metadata: _WeLMEmbeddingLeaseMetadata,
) -> bool:
    if metadata.manager_pid is None or not _is_process_alive(metadata.manager_pid):
        return False
    if metadata.manager_start_time is None:
        return True
    actual_start_time = _read_process_start_time(metadata.manager_pid)
    return (
        actual_start_time is None
        or actual_start_time == metadata.manager_start_time
    )


def _write_welm_embedding_arena_owner(
    arena_root: Path, ownership_id: str
) -> None:
    _validate_welm_embedding_ownership_id(ownership_id)
    owner_path = arena_root / _OWNER_FILE_NAME
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(owner_path, flags, 0o600)
    try:
        _write_all(
            fd,
            json.dumps(
                {"ownership_id": ownership_id},
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_welm_embedding_arena_owner(arena_root: Path) -> str | None:
    owner_path = arena_root / _OWNER_FILE_NAME
    fd = None
    try:
        fd = os.open(
            owner_path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        owner_stat = os.fstat(fd)
        if (
            not stat.S_ISREG(owner_stat.st_mode)
            or owner_stat.st_nlink != 1
            or owner_stat.st_uid != os.geteuid()
            or stat.S_IMODE(owner_stat.st_mode) & 0o077
            or owner_stat.st_size > 4096
        ):
            return None
        payload = bytearray()
        while len(payload) <= 4096:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > 4096:
            return None
        ownership_id = json.loads(payload)["ownership_id"]
        _validate_welm_embedding_ownership_id(ownership_id)
        return ownership_id
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None
    finally:
        if fd is not None:
            os.close(fd)


def _remove_matching_welm_embedding_arena_root_locked(
    arena_root: Path, *, expected_ownership_id: str
) -> bool:
    if not arena_root.exists():
        return True
    try:
        arena_stat = arena_root.lstat()
    except FileNotFoundError:
        return True
    except OSError as exc:
        logger.warning(
            "Refusing to remove unreadable WeLM embedding arena %s: %s",
            arena_root,
            exc,
        )
        return False
    if (
        arena_root.is_symlink()
        or not stat.S_ISDIR(arena_stat.st_mode)
        or arena_stat.st_uid != os.geteuid()
        or stat.S_IMODE(arena_stat.st_mode) & 0o077
        or _read_welm_embedding_arena_owner(arena_root) != expected_ownership_id
    ):
        logger.warning(
            "Refusing to remove WeLM embedding arena with mismatched owner: %s",
            arena_root,
        )
        return False
    return _try_remove_stale_arena_root(arena_root)


def _remove_matching_welm_embedding_arena_root(
    arena_root: Path,
    *,
    expected_ownership_id: str,
    gc_lock_held: bool = False,
) -> bool:
    arena_root = Path(arena_root).absolute()
    if gc_lock_held:
        return _remove_matching_welm_embedding_arena_root_locked(
            arena_root,
            expected_ownership_id=expected_ownership_id,
        )
    arena_parent = arena_root.parent.resolve(strict=True)
    gc_lock_fd = _open_and_lock_file(
        arena_parent / _GC_LOCK_FILE_NAME,
        blocking=True,
    )
    assert gc_lock_fd is not None
    try:
        return _remove_matching_welm_embedding_arena_root_locked(
            arena_root,
            expected_ownership_id=expected_ownership_id,
        )
    finally:
        fcntl.flock(gc_lock_fd, fcntl.LOCK_UN)
        os.close(gc_lock_fd)


def _prepare_welm_embedding_arena_root(
    arena_root: Path, *, ownership_id: str
) -> None:
    _validate_welm_embedding_ownership_id(ownership_id)
    arena_root = Path(arena_root).absolute()
    arena_parent = arena_root.parent.resolve(strict=True)
    gc_lock_fd = _open_and_lock_file(
        arena_parent / _GC_LOCK_FILE_NAME,
        blocking=True,
    )
    assert gc_lock_fd is not None
    try:
        if arena_root.exists() or arena_root.is_symlink():
            raise FileExistsError(
                f"WeLM embedding arena root already exists: {arena_root}"
            )
        os.mkdir(arena_root, mode=0o700)
        try:
            _write_welm_embedding_arena_owner(arena_root, ownership_id)
        except BaseException:
            shutil.rmtree(arena_root)
            raise
    finally:
        fcntl.flock(gc_lock_fd, fcntl.LOCK_UN)
        os.close(gc_lock_fd)


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_process_start_time(pid: int) -> int | None:
    if sys.platform != "linux" or pid <= 0:
        return None
    try:
        process_stat = Path(f"/proc/{pid}/stat").read_text()
        command_end = process_stat.rindex(")")
        fields_after_command = process_stat[command_end + 2 :].split()
        return int(fields_after_command[19])
    except (OSError, ValueError, IndexError):
        return None


def _try_remove_stale_arena_root(arena_root: Path) -> bool:
    try:
        shutil.rmtree(arena_root)
    except FileNotFoundError:
        return True
    except OSError as exc:
        logger.warning(
            "Failed to remove stale WeLM embedding arena %s: %s",
            arena_root,
            exc,
        )
        return False
    return True


def _classify_legacy_welm_embedding_arena(arena_root: Path) -> str:
    owner_path = arena_root / _OWNER_FILE_NAME
    try:
        owner_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning(
            "Skipping WeLM embedding arena with unreadable owner marker %s: %s",
            arena_root,
            exc,
        )
        return "skipped"
    else:
        logger.warning(
            "Skipping new-protocol WeLM embedding arena without a matching lease: %s",
            arena_root,
        )
        return "skipped"
    manifest_path = arena_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        logger.warning(
            "Skipping ambiguous legacy WeLM embedding arena: %s",
            arena_root,
        )
        return "skipped"
    try:
        manifest = load_welm_embedding_arena_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Skipping invalid legacy WeLM embedding arena %s: %s",
            arena_root,
            exc,
        )
        return "skipped"
    if _is_process_alive(manifest.manager_pid):
        return "active"
    if _try_remove_stale_arena_root(arena_root):
        return "cleaned"
    return "skipped"


def _cleanup_stale_welm_embedding_arenas_locked(
    arena_parent: Path,
) -> WeLMStaleArenaCleanupReport:
    lease_directory = _welm_embedding_lease_directory(arena_parent)
    cleaned = []
    active = []
    skipped = []
    for arena_root in sorted(arena_parent.iterdir(), key=lambda path: path.name):
        if not arena_root.name.startswith(_ARENA_PREFIX):
            continue
        run_id = arena_root.name[len(_ARENA_PREFIX) :]
        try:
            _validate_welm_embedding_run_id(run_id)
        except ValueError:
            skipped.append(str(arena_root))
            continue
        try:
            arena_stat = arena_root.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning(
                "Skipping unreadable WeLM embedding arena %s: %s",
                arena_root,
                exc,
            )
            skipped.append(str(arena_root))
            continue
        if (
            arena_root.is_symlink()
            or not stat.S_ISDIR(arena_stat.st_mode)
            or arena_stat.st_uid != os.geteuid()
            or stat.S_IMODE(arena_stat.st_mode) & 0o077
        ):
            skipped.append(str(arena_root))
            continue
        lease_path = lease_directory / f"{run_id}.lock"
        if lease_path.is_symlink():
            logger.warning(
                "Skipping WeLM embedding arena with unsafe lease symlink: %s",
                arena_root,
            )
            skipped.append(str(arena_root))
            continue
        if not lease_path.is_file():
            classification = _classify_legacy_welm_embedding_arena(arena_root)
            {"active": active, "cleaned": cleaned, "skipped": skipped}[
                classification
            ].append(str(arena_root))
            continue
        try:
            lease_fd = _open_and_lock_file(
                lease_path, blocking=False, create=False
            )
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            logger.warning(
                "Skipping WeLM embedding arena with unsafe lease %s: %s",
                arena_root,
                exc,
            )
            skipped.append(str(arena_root))
            continue
        if lease_fd is None:
            active.append(str(arena_root))
            continue
        try:
            lease_metadata = _read_locked_welm_embedding_lease_metadata(
                lease_fd,
                expected_run_id=run_id,
                expected_arena_root=arena_root,
            )
            if (
                lease_metadata is None
                or _read_welm_embedding_arena_owner(arena_root)
                != lease_metadata.ownership_id
            ):
                classification = _classify_legacy_welm_embedding_arena(arena_root)
                {"active": active, "cleaned": cleaned, "skipped": skipped}[
                    classification
                ].append(str(arena_root))
                continue
            if _is_registered_welm_embedding_manager_alive(lease_metadata):
                active.append(str(arena_root))
                continue
            if _remove_matching_welm_embedding_arena_root(
                arena_root,
                expected_ownership_id=lease_metadata.ownership_id,
                gc_lock_held=True,
            ):
                try:
                    _unlink_locked_file_if_same(lease_path, lease_fd)
                except OSError as exc:
                    logger.warning(
                        "Removed stale WeLM embedding arena but could not remove "
                        "lease %s: %s",
                        lease_path,
                        exc,
                    )
                cleaned.append(str(arena_root))
            else:
                skipped.append(str(arena_root))
        finally:
            fcntl.flock(lease_fd, fcntl.LOCK_UN)
            os.close(lease_fd)

    for lease_path in sorted(lease_directory.iterdir(), key=lambda path: path.name):
        if lease_path.is_symlink() or not lease_path.name.endswith(".lock"):
            continue
        run_id = lease_path.name[: -len(".lock")]
        try:
            _validate_welm_embedding_run_id(run_id)
        except ValueError:
            continue
        arena_root = arena_parent / f"{_ARENA_PREFIX}{run_id}"
        if arena_root.exists() or arena_root.is_symlink():
            continue
        try:
            lease_fd = _open_and_lock_file(
                lease_path, blocking=False, create=False
            )
        except (FileNotFoundError, OSError, RuntimeError):
            continue
        if lease_fd is None:
            continue
        try:
            lease_metadata = _read_locked_welm_embedding_lease_metadata(
                lease_fd,
                expected_run_id=run_id,
                expected_arena_root=arena_root,
            )
            if lease_metadata is None or _is_registered_welm_embedding_manager_alive(
                lease_metadata
            ):
                continue
            _unlink_locked_file_if_same(lease_path, lease_fd)
        except OSError as exc:
            logger.warning(
                "Failed to remove orphan WeLM embedding lease %s: %s",
                lease_path,
                exc,
            )
        finally:
            fcntl.flock(lease_fd, fcntl.LOCK_UN)
            os.close(lease_fd)
    return WeLMStaleArenaCleanupReport(
        cleaned=tuple(cleaned),
        active=tuple(active),
        skipped=tuple(skipped),
    )


def cleanup_stale_welm_embedding_arenas(
    arena_parent: Path,
) -> WeLMStaleArenaCleanupReport:
    arena_parent = Path(arena_parent).resolve(strict=True)
    gc_lock_path = arena_parent / _GC_LOCK_FILE_NAME
    gc_lock_fd = _open_and_lock_file(gc_lock_path, blocking=True)
    assert gc_lock_fd is not None
    try:
        return _cleanup_stale_welm_embedding_arenas_locked(arena_parent)
    finally:
        fcntl.flock(gc_lock_fd, fcntl.LOCK_UN)
        os.close(gc_lock_fd)


def _reserve_welm_embedding_arena_lease(
    arena_parent: Path, run_id: str
) -> WeLMEmbeddingArenaLease:
    _validate_welm_embedding_run_id(run_id)
    arena_parent = Path(arena_parent).resolve(strict=True)
    gc_lock_path = arena_parent / _GC_LOCK_FILE_NAME
    gc_lock_fd = _open_and_lock_file(gc_lock_path, blocking=True)
    assert gc_lock_fd is not None
    try:
        cleanup_report = _cleanup_stale_welm_embedding_arenas_locked(arena_parent)
        if cleanup_report.cleaned or cleanup_report.active or cleanup_report.skipped:
            logger.info(
                "WeLM embedding arena startup scan cleaned=%d active=%d skipped=%d",
                len(cleanup_report.cleaned),
                len(cleanup_report.active),
                len(cleanup_report.skipped),
            )
        lease_directory = _welm_embedding_lease_directory(arena_parent)
        lease_path = lease_directory / f"{run_id}.lock"
        lease_preexisted = lease_path.exists() or lease_path.is_symlink()
        lease_fd = _open_and_lock_file(lease_path, blocking=False)
        if lease_fd is None:
            raise RuntimeError(
                f"WeLM embedding arena lease is already owned: {lease_path}"
            )
        arena_root = arena_parent / f"{_ARENA_PREFIX}{run_id}"
        if lease_preexisted:
            previous_metadata = _read_locked_welm_embedding_lease_metadata(
                lease_fd,
                expected_run_id=run_id,
                expected_arena_root=arena_root,
            )
            manager_is_alive = (
                previous_metadata is not None
                and _is_registered_welm_embedding_manager_alive(previous_metadata)
            )
            fcntl.flock(lease_fd, fcntl.LOCK_UN)
            os.close(lease_fd)
            if manager_is_alive:
                raise RuntimeError(
                    f"WeLM embedding arena manager is still exiting: {arena_root}"
                )
            raise RuntimeError(
                f"WeLM embedding arena lease requires manual cleanup: {lease_path}"
            )
        if arena_root.exists() or arena_root.is_symlink():
            _unlink_locked_file_if_same(lease_path, lease_fd)
            fcntl.flock(lease_fd, fcntl.LOCK_UN)
            os.close(lease_fd)
            raise RuntimeError(
                f"WeLM embedding arena root requires manual cleanup: {arena_root}"
            )
        lease = WeLMEmbeddingArenaLease(
            arena_root=arena_root,
            lease_path=lease_path,
            run_id=run_id,
            ownership_id=uuid.uuid4().hex,
            fd=lease_fd,
            owner_pid=os.getpid(),
            owner_start_time=_read_process_start_time(os.getpid()),
            created_ns=time.time_ns(),
        )
        try:
            _write_welm_embedding_lease_metadata(lease)
        except BaseException:
            lease.close()
            raise
        return lease
    finally:
        fcntl.flock(gc_lock_fd, fcntl.LOCK_UN)
        os.close(gc_lock_fd)


@dataclass
class WeLMEmbeddingArenaProcessHandle:
    process: mp.Process
    manifest_path: str
    arena_root: str
    arena_id: str
    stop_event: object
    cuda_initialized: bool
    ownership_id: str
    lease: WeLMEmbeddingArenaLease | None = None
    _closed: bool = False

    def close(self, timeout: float = 30.0) -> None:
        if self._closed:
            return
        if timeout <= 0:
            raise ValueError("arena manager close timeout must be positive")
        try:
            _cancel_arena_manager_startup(
                self.process,
                self.stop_event,
                timeout=timeout,
            )
        except WeLMEmbeddingArenaManagerStillAliveError:
            _quarantine_welm_embedding_arena_lease(self.lease)
            raise
        errors = []
        try:
            _remove_owned_arena_root(
                Path(self.arena_root),
                manifest_path=Path(self.manifest_path),
                expected_arena_id=self.arena_id,
                expected_ownership_id=self.ownership_id,
            )
        except BaseException as exc:
            errors.append(exc)
        if self.lease is not None:
            try:
                self.lease.close()
            except BaseException as exc:
                errors.append(exc)
            else:
                _forget_quarantined_welm_embedding_arena_lease(self.lease)
        self._closed = True
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(
                "WeLM embedding arena process cleanup failed", errors
            )


class WeLMSharedEmbeddingRegistry:
    def __init__(
        self,
        *,
        manifest: WeLMEmbeddingArenaManifest,
        replica: WeLMEmbeddingReplicaManifest,
        gpu_id: int,
        views: dict[str, object],
    ) -> None:
        self._manifest = manifest
        self._replica = replica
        self._gpu_id = gpu_id
        self._views = views
        self._closed = False

    @classmethod
    def from_manifest(
        cls,
        path: str,
        *,
        gpu_id: int,
        gpu_numa_node: int,
        expected_checkpoint_identity: str | None = None,
        view_opener: Callable[..., object] | None = None,
    ) -> "WeLMSharedEmbeddingRegistry":
        if gpu_id < 0 or gpu_numa_node < 0:
            raise ValueError("GPU ID and NUMA node must be non-negative")
        manifest_path = Path(path).resolve(strict=True)
        manifest = load_welm_embedding_arena_manifest(manifest_path)
        if (
            expected_checkpoint_identity is not None
            and manifest.checkpoint_identity != expected_checkpoint_identity
        ):
            raise ValueError("shared embedding checkpoint identity does not match")
        policy = WeLMSharedEmbeddingPolicy(manifest.policy)
        if policy is WeLMSharedEmbeddingPolicy.REPLICATE_NUMA:
            candidates = [
                replica
                for replica in manifest.replicas
                if replica.numa_nodes == (gpu_numa_node,)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"replicate-numa has no unique local replica for NUMA node "
                    f"{gpu_numa_node}"
                )
            replica = candidates[0]
        else:
            replica = manifest.replicas[0]

        if view_opener is None:
            from sglang.srt.utils.shared_uva_tensor import SharedUVATensorView

            view_opener = SharedUVATensorView.open
        views: dict[str, object] = {}
        try:
            for tensor_spec in replica.tensors:
                views[tensor_spec.key] = view_opener(
                    tensor_spec,
                    gpu_id,
                    arena_root=manifest_path.parent,
                )
        except BaseException as original_error:
            cleanup_errors = []
            for view in reversed(tuple(views.values())):
                try:
                    view.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "shared embedding registry attachment cleanup failed",
                    [original_error, *cleanup_errors],
                )
            raise
        logger.info(
            "Attached WeLM shared embedding arena=%s policy=%s replica=%s "
            "gpu=%s numa=%s mapped_gib=%.5f",
            manifest.arena_id,
            manifest.policy,
            replica.replica_id,
            gpu_id,
            replica.numa_nodes,
            sum(tensor.nbytes for tensor in replica.tensors) / (1024**3),
        )
        return cls(
            manifest=manifest,
            replica=replica,
            gpu_id=gpu_id,
            views=views,
        )

    def get(self, key: str):
        if self._closed:
            raise RuntimeError("shared embedding registry is closed")
        try:
            view = self._views[key]
        except KeyError as exc:
            raise KeyError(f"shared embedding tensor is not registered: {key}") from exc
        return view.cuda_tensor

    def externally_owned_names(self) -> frozenset[str]:
        return frozenset(self._views)

    def diagnostics(self) -> dict[str, object]:
        return {
            "policy": self._manifest.policy,
            "arena_id": self._manifest.arena_id,
            "replica_id": self._replica.replica_id,
            "numa_nodes": self._replica.numa_nodes,
            "gpu_id": self._gpu_id,
            "mapped_bytes": sum(tensor.nbytes for tensor in self._replica.tensors),
            "paths": tuple(tensor.path for tensor in self._replica.tensors),
            "sampled_numa_nodes": {
                tensor.key: tensor.sampled_numa_nodes
                for tensor in self._replica.tensors
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        errors = []
        for key in reversed(tuple(self._views)):
            view = self._views[key]
            try:
                view.close()
            except BaseException as exc:
                errors.append(exc)
            else:
                del self._views[key]
        if errors:
            raise BaseExceptionGroup(
                "shared embedding registry cleanup failed", errors
            )
        self._closed = True


_PROCESS_SHARED_EMBEDDING_REGISTRY: WeLMSharedEmbeddingRegistry | None = None


def install_welm_shared_embedding_registry(
    manifest_path: str,
    *,
    gpu_id: int,
    gpu_numa_node: int,
    expected_checkpoint_identity: str | None = None,
    view_opener: Callable[..., object] | None = None,
) -> WeLMSharedEmbeddingRegistry:
    global _PROCESS_SHARED_EMBEDDING_REGISTRY
    if _PROCESS_SHARED_EMBEDDING_REGISTRY is not None:
        raise RuntimeError("WeLM shared embedding registry is already installed")
    registry = WeLMSharedEmbeddingRegistry.from_manifest(
        manifest_path,
        gpu_id=gpu_id,
        gpu_numa_node=gpu_numa_node,
        expected_checkpoint_identity=expected_checkpoint_identity,
        view_opener=view_opener,
    )
    _PROCESS_SHARED_EMBEDDING_REGISTRY = registry
    return registry


def get_welm_shared_embedding_registry(
    *, required: bool = True
) -> WeLMSharedEmbeddingRegistry | None:
    registry = _PROCESS_SHARED_EMBEDDING_REGISTRY
    if required and registry is None:
        raise RuntimeError("WeLM shared embedding registry is not installed")
    return registry


def close_welm_shared_embedding_registry() -> None:
    global _PROCESS_SHARED_EMBEDDING_REGISTRY
    registry = _PROCESS_SHARED_EMBEDDING_REGISTRY
    if registry is None:
        return
    registry.close()
    _PROCESS_SHARED_EMBEDDING_REGISTRY = None


@dataclass(frozen=True)
class WeLMSharedEmbeddingCheckpointTensorSpec:
    key: str
    path: Path
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class _ThreadNumaPolicy:
    mode: int
    maxnode: int
    nodemask: tuple[int, ...]


class NumaPlacementAdapter(Protocol):
    def apply(
        self,
        *,
        data_ptr: int,
        nbytes: int,
        mode: NumaPlacementMode,
        numa_nodes: tuple[int, ...],
    ) -> object: ...

    def reset(self, previous_policy: object) -> None: ...

    def sample(
        self, *, data_ptr: int, nbytes: int, max_samples: int
    ) -> dict[int, int]: ...


class LinuxNumaPlacementAdapter:
    def __init__(self) -> None:
        self._libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)

    def apply(
        self,
        *,
        data_ptr: int,
        nbytes: int,
        mode: NumaPlacementMode,
        numa_nodes: tuple[int, ...],
    ) -> _ThreadNumaPolicy:
        if data_ptr <= 0 or nbytes <= 0:
            raise ValueError("data_ptr and nbytes must be positive")
        numa_max_node = self._libnuma.numa_max_node
        numa_max_node.argtypes = []
        numa_max_node.restype = ctypes.c_int
        # The Linux NUMA syscall ABI decrements maxnode before consuming the mask.
        maxnode = max(numa_max_node() + 2, max(numa_nodes) + 2)
        word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        word_count = max(1, math.ceil(maxnode / word_bits))
        previous_mode = ctypes.c_int()
        previous_mask = (ctypes.c_ulong * word_count)()
        get_mempolicy = self._libnuma.get_mempolicy
        get_mempolicy.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        get_mempolicy.restype = ctypes.c_long
        result = get_mempolicy(
            ctypes.byref(previous_mode), previous_mask, maxnode, None, 0
        )
        if result == -1:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

        mask = (ctypes.c_ulong * word_count)()
        for node in numa_nodes:
            mask[node // word_bits] |= 1 << (node % word_bits)
        set_mempolicy = self._libnuma.set_mempolicy
        set_mempolicy.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
        ]
        set_mempolicy.restype = ctypes.c_long
        mode_value = 2 if mode is NumaPlacementMode.BIND else 3
        result = set_mempolicy(mode_value, mask, maxnode)
        if result == -1:
            error = ctypes.get_errno()
            if error in (errno.EACCES, errno.EPERM):
                raise PermissionError(
                    error, "set_mempolicy NUMA policy is not permitted"
                )
            raise OSError(error, os.strerror(error))
        return _ThreadNumaPolicy(
            mode=previous_mode.value,
            maxnode=maxnode,
            nodemask=tuple(previous_mask),
        )

    def reset(self, previous_policy: object) -> None:
        if not isinstance(previous_policy, _ThreadNumaPolicy):
            raise TypeError("invalid previous NUMA policy token")
        set_mempolicy = self._libnuma.set_mempolicy
        if previous_policy.mode == 0:
            result = set_mempolicy(0, None, 0)
        else:
            mask = (ctypes.c_ulong * len(previous_policy.nodemask))(
                *previous_policy.nodemask
            )
            result = set_mempolicy(
                previous_policy.mode, mask, previous_policy.maxnode
            )
        if result == -1:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    def sample(
        self, *, data_ptr: int, nbytes: int, max_samples: int
    ) -> dict[int, int]:
        if data_ptr <= 0 or nbytes <= 0 or max_samples <= 0:
            raise ValueError("data_ptr, nbytes, and max_samples must be positive")
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = max(1, math.ceil(nbytes / page_size))
        sample_count = min(page_count, max_samples)
        page_indices = sorted(
            {
                min(page_count - 1, index * page_count // sample_count)
                for index in range(sample_count)
            }
        )
        addresses = (ctypes.c_void_p * len(page_indices))(
            *(data_ptr + index * page_size for index in page_indices)
        )
        status = (ctypes.c_int * len(page_indices))()
        move_pages = self._libnuma.move_pages
        move_pages.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        move_pages.restype = ctypes.c_long
        result = move_pages(0, len(page_indices), addresses, None, status, 0)
        if result == -1:
            error = ctypes.get_errno()
            if error in (errno.EACCES, errno.EPERM):
                raise PermissionError(
                    error, "move_pages placement query is not permitted"
                )
            raise OSError(error, os.strerror(error))

        placement: dict[int, int] = {}
        for node in status:
            if node < 0:
                raise OSError(-node, os.strerror(-node))
            placement[node] = placement.get(node, 0) + 1
        return placement


def plan_welm_embedding_replicas(
    policy: WeLMSharedEmbeddingPolicy,
    gpu_numa_nodes: tuple[int, ...],
    bind_node: int | None,
) -> tuple[WeLMEmbeddingReplicaPlan, ...]:
    policy = WeLMSharedEmbeddingPolicy(policy)
    if policy is WeLMSharedEmbeddingPolicy.DISABLED:
        if bind_node is not None:
            raise ValueError(
                "--welm-shared-embedding-numa-node is only valid with bind policy"
            )
        return ()

    if not gpu_numa_nodes:
        raise ValueError("GPU NUMA topology must contain at least one local GPU")
    if any(node < 0 for node in gpu_numa_nodes):
        raise ValueError("GPU NUMA topology contains a negative NUMA node")

    active_nodes = tuple(sorted(set(gpu_numa_nodes)))
    all_consumers = tuple(range(len(gpu_numa_nodes)))

    if policy is WeLMSharedEmbeddingPolicy.BIND:
        if bind_node is None:
            raise ValueError("bind policy requires --welm-shared-embedding-numa-node")
        if bind_node not in active_nodes:
            raise ValueError(
                f"bind NUMA node {bind_node} has no consumer GPU; "
                f"active nodes are {active_nodes}"
            )
        return (
            WeLMEmbeddingReplicaPlan(
                replica_id=f"bind-numa-{bind_node}",
                numa_nodes=(bind_node,),
                consumer_local_ranks=all_consumers,
            ),
        )

    if bind_node is not None:
        raise ValueError(
            "--welm-shared-embedding-numa-node is only valid with bind policy"
        )

    if policy is WeLMSharedEmbeddingPolicy.INTERLEAVE:
        node_suffix = "-".join(str(node) for node in active_nodes)
        return (
            WeLMEmbeddingReplicaPlan(
                replica_id=f"interleave-numa-{node_suffix}",
                numa_nodes=active_nodes,
                consumer_local_ranks=all_consumers,
            ),
        )

    return tuple(
        WeLMEmbeddingReplicaPlan(
            replica_id=f"replica-numa-{node}",
            numa_nodes=(node,),
            consumer_local_ranks=tuple(
                rank
                for rank, consumer_node in enumerate(gpu_numa_nodes)
                if consumer_node == node
            ),
        )
        for node in active_nodes
    )


def calculate_welm_embedding_byte_counts(
    *,
    base_shape: Sequence[int],
    oe_shapes: Sequence[Sequence[int]],
    element_size: int,
    replica_count: int,
) -> WeLMEmbeddingByteCounts:
    if element_size <= 0:
        raise ValueError("element_size must be positive")
    if replica_count <= 0:
        raise ValueError("replica_count must be positive")
    if not oe_shapes:
        raise ValueError("at least one OE shape is required")

    def tensor_bytes(name: str, shape: Sequence[int]) -> int:
        if len(shape) != 2 or any(dim <= 0 for dim in shape):
            raise ValueError(f"{name} must be a two-dimensional positive shape")
        return math.prod(shape) * element_size

    base_bytes = tensor_bytes("base_shape", base_shape)
    oe_bytes = sum(
        tensor_bytes(f"oe_shapes[{index}]", shape)
        for index, shape in enumerate(oe_shapes)
    )
    logical_bytes = base_bytes + oe_bytes
    return WeLMEmbeddingByteCounts(
        base_bytes=base_bytes,
        oe_bytes=oe_bytes,
        logical_bytes=logical_bytes,
        physical_bytes=logical_bytes * replica_count,
    )


def discover_welm_shared_embedding_checkpoint_tensors(
    checkpoint: Path,
) -> tuple[tuple[WeLMSharedEmbeddingCheckpointTensorSpec, ...], str]:
    checkpoint = checkpoint.resolve(strict=True)
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"missing safetensors index: {index_path}")
    index_bytes = index_path.read_bytes()
    try:
        index = json.loads(index_bytes)
        weight_map = index["weight_map"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid safetensors index: {index_path}") from exc
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors weight_map must be an object")

    missing = [key for key in _SHARED_EMBEDDING_KEYS if key not in weight_map]
    if missing:
        raise ValueError(f"missing shared embedding weights: {', '.join(missing)}")
    expected_oe = set(_SHARED_EMBEDDING_KEYS[1:])
    extra_oe = sorted(
        key
        for key in weight_map
        if key.startswith("model.oe_embed.")
        and key.endswith(".weight")
        and key not in expected_oe
    )
    if extra_oe:
        raise ValueError(f"unexpected OE embedding weights: {', '.join(extra_oe)}")

    from safetensors import safe_open

    specs = []
    identity_tensors = []
    for key in _SHARED_EMBEDDING_KEYS:
        relative_path = weight_map[key]
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"invalid shard path for {key}")
        path = (checkpoint / relative_path).resolve(strict=True)
        try:
            path.relative_to(checkpoint)
        except ValueError as exc:
            raise ValueError(f"checkpoint shard for {key} escapes checkpoint root") from exc
        if not path.is_file():
            raise ValueError(f"checkpoint shard is not a file: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                raise ValueError(f"safetensors shard {path.name} does not contain {key}")
            tensor_slice = handle.get_slice(key)
            shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            dtype = str(tensor_slice.get_dtype())
        if len(shape) != 2 or any(dim <= 0 for dim in shape):
            raise ValueError(f"shared embedding {key} must be a positive 2D tensor")
        if dtype != "BF16":
            raise ValueError(f"shared embedding {key} must use BF16, found {dtype}")
        nbytes = math.prod(shape) * 2
        specs.append(
            WeLMSharedEmbeddingCheckpointTensorSpec(
                key=key,
                path=path,
                shape=shape,
                dtype="bfloat16",
                nbytes=nbytes,
            )
        )
        identity_tensors.append(
            {
                "key": key,
                "shard": str(path.relative_to(checkpoint)),
                "shape": shape,
                "dtype": dtype,
                "shard_size": path.stat().st_size,
            }
        )

    oe_widths = {spec.shape[1] for spec in specs[1:]}
    if len(oe_widths) != 1:
        raise ValueError("all OE embedding widths must match")

    identity_payload = {
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "tensors": identity_tensors,
    }
    checkpoint_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return tuple(specs), checkpoint_identity


def _available_bytes(path: Path) -> int:
    usage = os.statvfs(path)
    return usage.f_bavail * usage.f_frsize


def build_welm_shared_embedding_dry_run_report(
    *, checkpoint: Path, gpu_numa_nodes: tuple[int, ...]
) -> dict[str, object]:
    if not gpu_numa_nodes:
        raise ValueError("GPU NUMA topology must not be empty")
    sources, checkpoint_identity = discover_welm_shared_embedding_checkpoint_tensors(
        Path(checkpoint)
    )
    base_bytes = sources[0].nbytes
    oe_bytes = sum(source.nbytes for source in sources[1:])
    logical_bytes = base_bytes + oe_bytes
    active_numa_nodes = len(set(gpu_numa_nodes))
    return {
        "checkpoint_identity": checkpoint_identity,
        "base_weight_bytes": base_bytes,
        "oe_weight_bytes": oe_bytes,
        "logical_weight_bytes": logical_bytes,
        "topologies": {
            "cp1-attntp8": {"current_private_bytes": logical_bytes},
            "cp2-attntp4": {"current_private_bytes": logical_bytes * 2},
            "cp4-attntp2": {"current_private_bytes": logical_bytes * 4},
        },
        "policy_physical_bytes": {
            "bind": logical_bytes,
            "interleave": logical_bytes,
            "replicate-numa": logical_bytes * active_numa_nodes,
        },
    }


def _infer_policy(
    plans: tuple[WeLMEmbeddingReplicaPlan, ...],
) -> WeLMSharedEmbeddingPolicy:
    if not plans:
        raise ValueError("enabled shared embedding arena requires at least one replica")
    replica_ids = [plan.replica_id for plan in plans]
    if len(set(replica_ids)) != len(replica_ids):
        raise ValueError("shared embedding replica IDs must be unique")
    if len(plans) > 1:
        if any(len(plan.numa_nodes) != 1 for plan in plans):
            raise ValueError("replicate-numa requires one NUMA node per replica")
        return WeLMSharedEmbeddingPolicy.REPLICATE_NUMA
    if len(plans[0].numa_nodes) > 1:
        return WeLMSharedEmbeddingPolicy.INTERLEAVE
    return WeLMSharedEmbeddingPolicy.BIND


def _placement_mode(
    policy: WeLMSharedEmbeddingPolicy,
) -> NumaPlacementMode:
    if policy is WeLMSharedEmbeddingPolicy.INTERLEAVE:
        return NumaPlacementMode.INTERLEAVE
    return NumaPlacementMode.BIND


def _copy_tensor_rows(
    *,
    source: WeLMSharedEmbeddingCheckpointTensorSpec,
    mapping: mmap.mmap,
    compute_digest: bool,
) -> str | None:
    import torch
    from safetensors import safe_open

    row_nbytes = source.shape[1] * 2
    rows_per_chunk = max(1, _COPY_CHUNK_BYTES // row_nbytes)
    hasher = hashlib.sha256() if compute_digest else None
    with safe_open(source.path, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(source.key)
        for start in range(0, source.shape[0], rows_per_chunk):
            end = min(source.shape[0], start + rows_per_chunk)
            chunk = tensor_slice[start:end]
            byte_view = chunk.contiguous().view(torch.uint8).numpy()
            raw = memoryview(byte_view).cast("B")
            try:
                byte_start = start * row_nbytes
                mapping[byte_start : byte_start + raw.nbytes] = raw
                if hasher is not None:
                    hasher.update(raw)
            finally:
                raw.release()
                del raw, byte_view, chunk
    return hasher.hexdigest() if hasher is not None else None


def _validate_sampled_placement(
    *,
    placement: dict[int, int],
    requested_nodes: tuple[int, ...],
    mode: NumaPlacementMode,
) -> None:
    if not placement or any(count <= 0 for count in placement.values()):
        raise RuntimeError("NUMA placement sampling returned no resident pages")
    actual_nodes = set(placement)
    requested = set(requested_nodes)
    if not actual_nodes.issubset(requested):
        raise RuntimeError(
            f"NUMA placement {sorted(actual_nodes)} violates requested nodes "
            f"{sorted(requested)}"
        )
    if mode is NumaPlacementMode.BIND and actual_nodes != requested:
        raise RuntimeError(
            f"NUMA placement {sorted(actual_nodes)} does not match bound node "
            f"{sorted(requested)}"
        )
    if mode is NumaPlacementMode.INTERLEAVE and sum(placement.values()) >= len(
        requested
    ):
        if actual_nodes != requested:
            raise RuntimeError(
                f"NUMA placement {sorted(actual_nodes)} does not cover interleave "
                f"nodes {sorted(requested)}"
            )


def _tensor_file_name(index: int, key: str) -> str:
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"tensor-{index:02d}-{digest}.bin"


def _manifest_to_dict(manifest: WeLMEmbeddingArenaManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "arena_id": manifest.arena_id,
        "checkpoint_identity": manifest.checkpoint_identity,
        "policy": manifest.policy,
        "logical_weight_bytes": manifest.logical_weight_bytes,
        "physical_weight_bytes": manifest.physical_weight_bytes,
        "manager_pid": manifest.manager_pid,
        "replicas": [
            {
                "replica_id": replica.replica_id,
                "numa_nodes": list(replica.numa_nodes),
                "tensors": [
                    {
                        "key": tensor.key,
                        "path": tensor.path,
                        "shape": list(tensor.shape),
                        "dtype": tensor.dtype,
                        "nbytes": tensor.nbytes,
                        "replica_id": tensor.replica_id,
                        "numa_nodes": list(tensor.numa_nodes),
                        "inode": tensor.inode,
                        "sampled_numa_nodes": [
                            list(item) for item in tensor.sampled_numa_nodes
                        ],
                    }
                    for tensor in replica.tensors
                ],
            }
            for replica in manifest.replicas
        ],
    }


def _publish_manifest_atomically(
    *, root: Path, manifest: WeLMEmbeddingArenaManifest
) -> None:
    temporary_path = root / "manifest.json.tmp"
    final_path = root / "manifest.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(temporary_path, flags, 0o600)
    try:
        payload = json.dumps(
            _manifest_to_dict(manifest), sort_keys=True, separators=(",", ":")
        ).encode()
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary_path, final_path)
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_welm_embedding_arena_manifest(
    path: str | os.PathLike[str],
) -> WeLMEmbeddingArenaManifest:
    from sglang.srt.utils.shared_uva_tensor import SharedTensorFileSpec

    manifest_path = Path(path).resolve(strict=True)
    try:
        payload = json.loads(manifest_path.read_text())
        replicas = tuple(
            WeLMEmbeddingReplicaManifest(
                replica_id=replica["replica_id"],
                numa_nodes=tuple(replica["numa_nodes"]),
                tensors=tuple(
                    SharedTensorFileSpec(
                        key=tensor["key"],
                        path=tensor["path"],
                        shape=tuple(tensor["shape"]),
                        dtype=tensor["dtype"],
                        nbytes=tensor["nbytes"],
                        replica_id=tensor["replica_id"],
                        numa_nodes=tuple(tensor["numa_nodes"]),
                        inode=tensor["inode"],
                        sampled_numa_nodes=tuple(
                            tuple(item) for item in tensor["sampled_numa_nodes"]
                        ),
                    )
                    for tensor in replica["tensors"]
                ),
            )
            for replica in payload["replicas"]
        )
        manifest = WeLMEmbeddingArenaManifest(
            schema_version=payload["schema_version"],
            arena_id=payload["arena_id"],
            checkpoint_identity=payload["checkpoint_identity"],
            policy=payload["policy"],
            logical_weight_bytes=payload["logical_weight_bytes"],
            physical_weight_bytes=payload["physical_weight_bytes"],
            replicas=replicas,
            manager_pid=payload["manager_pid"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid WeLM shared embedding manifest: {manifest_path}") from exc
    if manifest.schema_version != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported WeLM shared embedding manifest schema "
            f"{manifest.schema_version}"
        )
    _validate_welm_embedding_arena_manifest(
        manifest, arena_root=manifest_path.parent
    )
    return manifest


def _validate_welm_embedding_arena_manifest(
    manifest: WeLMEmbeddingArenaManifest, *, arena_root: Path
) -> None:
    from sglang.srt.utils.shared_uva_tensor import (
        validate_shared_tensor_file_spec,
    )

    try:
        policy = WeLMSharedEmbeddingPolicy(manifest.policy)
    except ValueError as exc:
        raise ValueError(f"invalid shared embedding policy {manifest.policy!r}") from exc
    if policy is WeLMSharedEmbeddingPolicy.DISABLED:
        raise ValueError("disabled policy cannot publish a shared embedding arena")
    if not manifest.arena_id:
        raise ValueError("shared embedding arena_id must not be empty")
    if (
        len(manifest.checkpoint_identity) != 64
        or any(char not in "0123456789abcdef" for char in manifest.checkpoint_identity)
    ):
        raise ValueError("shared embedding checkpoint identity must be SHA-256")
    if manifest.manager_pid <= 0:
        raise ValueError("shared embedding manager PID must be positive")
    if not manifest.replicas:
        raise ValueError("shared embedding manifest must contain replicas")

    replica_ids = [replica.replica_id for replica in manifest.replicas]
    if len(replica_ids) != len(set(replica_ids)):
        raise ValueError("shared embedding replica IDs must be unique")
    if policy is WeLMSharedEmbeddingPolicy.BIND:
        if len(manifest.replicas) != 1 or len(manifest.replicas[0].numa_nodes) != 1:
            raise ValueError("bind policy requires one single-node replica")
    elif policy is WeLMSharedEmbeddingPolicy.INTERLEAVE:
        if len(manifest.replicas) != 1 or len(manifest.replicas[0].numa_nodes) < 2:
            raise ValueError("interleave policy requires one multi-node replica")
    elif any(len(replica.numa_nodes) != 1 for replica in manifest.replicas):
        raise ValueError("replicate-numa policy requires single-node replicas")

    reference_layout = None
    seen_paths: set[Path] = set()
    seen_inodes: set[int] = set()
    for replica in manifest.replicas:
        if not replica.replica_id or not replica.numa_nodes:
            raise ValueError("shared embedding replica metadata is incomplete")
        if len(set(replica.numa_nodes)) != len(replica.numa_nodes):
            raise ValueError("shared embedding replica NUMA nodes must be unique")
        keys = tuple(tensor.key for tensor in replica.tensors)
        if keys != _SHARED_EMBEDDING_KEYS:
            raise ValueError(
                "shared embedding replica must contain the complete shared tensor set"
            )
        layout = tuple(
            (tensor.key, tensor.shape, tensor.dtype, tensor.nbytes)
            for tensor in replica.tensors
        )
        if reference_layout is None:
            reference_layout = layout
        elif layout != reference_layout:
            raise ValueError("shared embedding replica tensor layouts must match")

        for tensor in replica.tensors:
            if tensor.replica_id != replica.replica_id:
                raise ValueError("shared tensor replica ID is inconsistent")
            if tensor.numa_nodes != replica.numa_nodes:
                raise ValueError("shared tensor NUMA nodes are inconsistent")
            try:
                resolved_path = Path(tensor.path).resolve(strict=True)
            except OSError as exc:
                raise ValueError(f"shared tensor path is unavailable: {tensor.path}") from exc
            if resolved_path in seen_paths or tensor.inode in seen_inodes:
                raise ValueError("distinct shared tensor keys must not alias a path/inode")
            seen_paths.add(resolved_path)
            seen_inodes.add(tensor.inode)
            try:
                path = validate_shared_tensor_file_spec(tensor, arena_root)
            except (OSError, ValueError) as exc:
                raise ValueError(str(exc)) from exc
            if path.stat().st_size != tensor.nbytes:
                raise ValueError("shared tensor file size does not match manifest")
            try:
                _validate_sampled_placement(
                    placement=dict(tensor.sampled_numa_nodes),
                    requested_nodes=replica.numa_nodes,
                    mode=(
                        NumaPlacementMode.INTERLEAVE
                        if policy is WeLMSharedEmbeddingPolicy.INTERLEAVE
                        else NumaPlacementMode.BIND
                    ),
                )
            except RuntimeError as exc:
                raise ValueError(f"sampled NUMA placement invalid: {exc}") from exc

    assert reference_layout is not None
    logical_bytes = sum(item[3] for item in reference_layout)
    physical_bytes = logical_bytes * len(manifest.replicas)
    if (
        manifest.logical_weight_bytes != logical_bytes
        or manifest.physical_weight_bytes != physical_bytes
    ):
        raise ValueError("shared embedding byte accounting is inconsistent")


def _close_mapping_and_fd(mapping: mmap.mmap | None, fd: int) -> None:
    errors = []
    if mapping is not None:
        try:
            mapping.close()
        except BaseException as exc:
            errors.append(exc)
    try:
        os.close(fd)
    except BaseException as exc:
        errors.append(exc)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("arena tensor mapping cleanup failed", errors)


def create_welm_embedding_arena(
    *,
    checkpoint: Path,
    root: Path,
    plans: tuple[WeLMEmbeddingReplicaPlan, ...],
    numa_adapter: NumaPlacementAdapter | None = None,
    ownership_id: str | None = None,
) -> WeLMEmbeddingArenaManifest:
    from sglang.srt.utils.shared_uva_tensor import SharedTensorFileSpec

    root = Path(root).absolute()
    parent = root.parent.resolve(strict=True)
    policy = _infer_policy(plans)
    sources, checkpoint_metadata_identity = (
        discover_welm_shared_embedding_checkpoint_tensors(Path(checkpoint))
    )
    logical_bytes = sum(source.nbytes for source in sources)
    physical_bytes = logical_bytes * len(plans)
    available_bytes = _available_bytes(parent)
    if physical_bytes > available_bytes:
        raise OSError(
            f"insufficient capacity for WeLM shared embedding arena: "
            f"need {physical_bytes} bytes, have {available_bytes} bytes"
        )
    adapter = numa_adapter or LinuxNumaPlacementAdapter()
    mode = _placement_mode(policy)
    created_root = False
    try:
        if ownership_id is None:
            os.mkdir(root, mode=0o700)
        else:
            _validate_welm_embedding_ownership_id(ownership_id)
            root_stat = root.lstat()
            if (
                root.is_symlink()
                or not stat.S_ISDIR(root_stat.st_mode)
                or root_stat.st_uid != os.geteuid()
                or stat.S_IMODE(root_stat.st_mode) & 0o077
                or _read_welm_embedding_arena_owner(root) != ownership_id
            ):
                raise RuntimeError(
                    f"invalid pre-created WeLM embedding arena root: {root}"
                )
        created_root = True
        os.chmod(root, 0o700)
        replica_manifests = []
        content_digests: dict[str, str] = {}
        for replica_index, plan in enumerate(plans):
            tensor_specs = []
            for index, source in enumerate(sources):
                path = root / f"{plan.replica_id}-{_tensor_file_name(index, source.key)}"
                flags = (
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                fd = os.open(path, flags, 0o600)
                mapping = None
                try:
                    os.fchmod(fd, 0o600)
                    os.ftruncate(fd, source.nbytes)
                    mapping = mmap.mmap(
                        fd,
                        source.nbytes,
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                    data_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                    previous_policy = adapter.apply(
                        data_ptr=data_ptr,
                        nbytes=source.nbytes,
                        mode=mode,
                        numa_nodes=plan.numa_nodes,
                    )
                    try:
                        content_digest = _copy_tensor_rows(
                            source=source,
                            mapping=mapping,
                            compute_digest=replica_index == 0,
                        )
                    finally:
                        adapter.reset(previous_policy)
                    if content_digest is not None:
                        content_digests[source.key] = content_digest
                    mapping.flush()
                    placement = adapter.sample(
                        data_ptr=data_ptr,
                        nbytes=source.nbytes,
                        max_samples=_NUMA_SAMPLE_COUNT,
                    )
                    _validate_sampled_placement(
                        placement=placement,
                        requested_nodes=plan.numa_nodes,
                        mode=mode,
                    )
                    os.fsync(fd)
                    file_stat = os.fstat(fd)
                    if not stat.S_ISREG(file_stat.st_mode):
                        raise RuntimeError(f"arena tensor is not a regular file: {path}")
                    tensor_specs.append(
                        SharedTensorFileSpec(
                            key=source.key,
                            path=str(path),
                            shape=source.shape,
                            dtype=source.dtype,
                            nbytes=source.nbytes,
                            replica_id=plan.replica_id,
                            numa_nodes=plan.numa_nodes,
                            inode=file_stat.st_ino,
                            sampled_numa_nodes=tuple(sorted(placement.items())),
                        )
                    )
                finally:
                    _close_mapping_and_fd(mapping, fd)
            replica_manifests.append(
                WeLMEmbeddingReplicaManifest(
                    replica_id=plan.replica_id,
                    numa_nodes=plan.numa_nodes,
                    tensors=tuple(tensor_specs),
                )
            )

        checkpoint_identity = hashlib.sha256(
            json.dumps(
                {
                    "metadata": checkpoint_metadata_identity,
                    "content": [
                        (key, content_digests[key]) for key in _SHARED_EMBEDDING_KEYS
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        manifest = WeLMEmbeddingArenaManifest(
            schema_version=_MANIFEST_SCHEMA_VERSION,
            arena_id=uuid.uuid4().hex,
            checkpoint_identity=checkpoint_identity,
            policy=policy.value,
            logical_weight_bytes=logical_bytes,
            physical_weight_bytes=physical_bytes,
            replicas=tuple(replica_manifests),
            manager_pid=os.getpid(),
        )
        _publish_manifest_atomically(root=root, manifest=manifest)
        return manifest
    except BaseException as original_error:
        if created_root:
            try:
                if ownership_id is None:
                    shutil.rmtree(root)
                elif not _remove_matching_welm_embedding_arena_root(
                    root, expected_ownership_id=ownership_id
                ):
                    raise RuntimeError(
                        "refusing to clean a replacement WeLM embedding arena"
                    )
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "WeLM shared embedding arena cleanup failed",
                    [original_error, cleanup_error],
                )
        raise


def _remove_owned_arena_root(
    root: Path,
    *,
    manifest_path: Path,
    expected_arena_id: str,
    expected_ownership_id: str,
) -> None:
    root = Path(root).absolute()
    root = root.parent.resolve(strict=True) / root.name
    gc_lock_fd = _open_and_lock_file(
        root.parent / _GC_LOCK_FILE_NAME,
        blocking=True,
    )
    assert gc_lock_fd is not None
    try:
        if not root.exists() and not root.is_symlink():
            return
        root_stat = root.lstat()
        if (
            root.is_symlink()
            or not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != os.geteuid()
            or stat.S_IMODE(root_stat.st_mode) & 0o077
        ):
            raise RuntimeError(f"refusing to clean unsafe arena root: {root}")
        if _read_welm_embedding_arena_owner(root) != expected_ownership_id:
            raise RuntimeError(
                f"refusing to clean arena {root}: ownership generation changed"
            )
        manifest_path = Path(manifest_path).absolute()
        expected_manifest_path = root / "manifest.json"
        if manifest_path != expected_manifest_path or manifest_path.is_symlink():
            raise RuntimeError(
                f"refusing to clean arena with unsafe manifest: {manifest_path}"
            )
        manifest = load_welm_embedding_arena_manifest(manifest_path)
        if manifest.arena_id != expected_arena_id:
            raise RuntimeError(
                f"refusing to clean arena {root}: expected arena ID "
                f"{expected_arena_id}, found {manifest.arena_id}"
            )
        if not _remove_matching_welm_embedding_arena_root_locked(
            root,
            expected_ownership_id=expected_ownership_id,
        ):
            raise RuntimeError(
                f"refusing to clean arena {root}: ownership generation changed"
            )
    finally:
        fcntl.flock(gc_lock_fd, fcntl.LOCK_UN)
        os.close(gc_lock_fd)


def _install_parent_death_signal(
    expected_parent_pid: int,
    *,
    prctl: Callable[[int, int], int] | None = None,
    get_parent_pid: Callable[[], int] = os.getppid,
) -> None:
    if expected_parent_pid <= 0:
        raise ValueError("expected parent PID must be positive")
    if sys.platform != "linux":
        logger.warning("parent-death signaling is only supported on Linux")
        return
    if prctl is None:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        prctl = libc.prctl
    result = prctl(1, signal.SIGTERM)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    actual_parent_pid = get_parent_pid()
    if actual_parent_pid != expected_parent_pid:
        raise RuntimeError(
            f"parent process {expected_parent_pid} exited before the WeLM "
            f"embedding arena manager initialized; current parent is "
            f"{actual_parent_pid}"
        )


def _raise_on_arena_manager_sigterm(signum, _frame) -> None:
    raise SystemExit(128 + signum)


def _run_welm_embedding_arena_manager(
    *,
    checkpoint: str,
    root: str,
    plans: tuple[WeLMEmbeddingReplicaPlan, ...],
    ready_writer,
    stop_event,
    start_event,
    numa_adapter_factory: Callable[[], NumaPlacementAdapter],
    expected_parent_pid: int,
    ownership_id: str,
) -> None:
    root_path = Path(root)
    try:
        signal.signal(signal.SIGTERM, _raise_on_arena_manager_sigterm)
        _install_parent_death_signal(expected_parent_pid)
        start_event.wait()
        if stop_event.is_set():
            return
        manifest = create_welm_embedding_arena(
            checkpoint=Path(checkpoint),
            root=root_path,
            plans=plans,
            numa_adapter=numa_adapter_factory(),
            ownership_id=ownership_id,
        )
        logger.info(
            "Published WeLM shared embedding arena=%s policy=%s replicas=%d "
            "logical_gib=%.5f physical_gib=%.5f manifest=%s",
            manifest.arena_id,
            manifest.policy,
            len(manifest.replicas),
            manifest.logical_weight_bytes / (1024**3),
            manifest.physical_weight_bytes / (1024**3),
            Path(root) / "manifest.json",
        )
        for replica in manifest.replicas:
            for tensor in replica.tensors:
                logger.info(
                    "WeLM shared tensor replica=%s key=%s inode=%d bytes=%d "
                    "sampled_numa=%s",
                    replica.replica_id,
                    tensor.key,
                    tensor.inode,
                    tensor.nbytes,
                    tensor.sampled_numa_nodes,
                )
        ready_writer.send(
            {
                "status": "ready",
                "manifest_path": str(Path(root) / "manifest.json"),
                "arena_id": manifest.arena_id,
            }
        )
        stop_event.wait()
    except BaseException:
        try:
            ready_writer.send(
                {
                    "status": "error",
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            _remove_matching_welm_embedding_arena_root(
                root_path,
                expected_ownership_id=ownership_id,
            )
        except BaseException:
            logger.exception(
                "Failed to remove WeLM embedding arena during manager exit: %s",
                root_path,
            )
        finally:
            ready_writer.close()


def _wait_for_arena_manager_ready(
    reader,
    process: mp.Process,
    *,
    timeout: float,
) -> dict[str, object]:
    if timeout <= 0:
        raise ValueError("arena manager startup timeout must be positive")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out waiting for WeLM embedding arena manager "
                f"{process.pid} after {timeout:.1f}s"
            )
        if reader.poll(min(0.1, remaining)):
            message = reader.recv()
            if not isinstance(message, dict) or "status" not in message:
                raise RuntimeError("invalid WeLM embedding arena manager response")
            return message
        if not process.is_alive():
            process.join(timeout=1)
            raise RuntimeError(
                f"WeLM embedding arena manager {process.pid} exited before ready "
                f"with code {process.exitcode}"
            )


def _cancel_arena_manager_startup(
    process: mp.Process,
    stop_event,
    *,
    timeout: float = 5.0,
    start_event=None,
) -> None:
    if process.is_alive():
        stop_event.set()
        if start_event is not None:
            start_event.set()
        process.join(timeout=timeout)
    if process.is_alive():
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        process.join(timeout=timeout)
    if process.is_alive():
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.join(timeout=timeout)
    if process.is_alive():
        raise WeLMEmbeddingArenaManagerStillAliveError(process)


def launch_welm_embedding_arena_process(
    *,
    checkpoint: Path,
    root: Path,
    plans: tuple[WeLMEmbeddingReplicaPlan, ...],
    numa_adapter_factory: Callable[[], NumaPlacementAdapter] = (
        LinuxNumaPlacementAdapter
    ),
    timeout: float = 3600.0,
    ownership_id: str | None = None,
    process_started_callback: Callable[[int], None] | None = None,
) -> WeLMEmbeddingArenaProcessHandle:
    root = Path(root).absolute()
    root = root.parent.resolve(strict=True) / root.name
    if ownership_id is None:
        ownership_id = uuid.uuid4().hex
    _prepare_welm_embedding_arena_root(
        root,
        ownership_id=ownership_id,
    )
    reader = None
    writer = None
    try:
        context = mp.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        stop_event = context.Event()
        start_event = context.Event()
        process = context.Process(
            target=_run_welm_embedding_arena_manager,
            kwargs={
                "checkpoint": str(Path(checkpoint).absolute()),
                "root": str(root),
                "plans": plans,
                "ready_writer": writer,
                "stop_event": stop_event,
                "start_event": start_event,
                "numa_adapter_factory": numa_adapter_factory,
                "expected_parent_pid": os.getpid(),
                "ownership_id": ownership_id,
            },
            name="sglang-welm-embedding-arena",
        )
    except BaseException:
        if reader is not None:
            reader.close()
        if writer is not None:
            writer.close()
        _remove_matching_welm_embedding_arena_root(
            root,
            expected_ownership_id=ownership_id,
        )
        raise
    previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        try:
            process.start()
            if process_started_callback is not None:
                process_started_callback(process.pid)
            start_event.set()
        except BaseException:
            try:
                if process.pid is not None:
                    _cancel_arena_manager_startup(
                        process,
                        stop_event,
                        start_event=start_event,
                    )
            except WeLMEmbeddingArenaManagerStillAliveError:
                reader.close()
                writer.close()
                raise
            reader.close()
            writer.close()
            _remove_matching_welm_embedding_arena_root(
                root,
                expected_ownership_id=ownership_id,
            )
            raise
    finally:
        if previous_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices
    writer.close()
    try:
        message = _wait_for_arena_manager_ready(reader, process, timeout=timeout)
    except BaseException:
        _cancel_arena_manager_startup(process, stop_event)
        _remove_matching_welm_embedding_arena_root(
            root,
            expected_ownership_id=ownership_id,
        )
        raise
    finally:
        reader.close()

    if message["status"] != "ready":
        _cancel_arena_manager_startup(process, stop_event)
        _remove_matching_welm_embedding_arena_root(
            root,
            expected_ownership_id=ownership_id,
        )
        detail = message.get("traceback", "unknown manager failure")
        raise RuntimeError(f"WeLM embedding arena manager failed:\n{detail}")

    try:
        manifest_path = Path(str(message["manifest_path"])).resolve(strict=True)
        if manifest_path != root.resolve(strict=True) / "manifest.json":
            raise RuntimeError("arena manager returned an unexpected manifest path")
        manifest = load_welm_embedding_arena_manifest(manifest_path)
        if manifest.arena_id != message.get("arena_id"):
            raise RuntimeError("arena manager returned an inconsistent arena ID")
    except BaseException:
        _cancel_arena_manager_startup(process, stop_event)
        _remove_matching_welm_embedding_arena_root(
            root,
            expected_ownership_id=ownership_id,
        )
        raise
    return WeLMEmbeddingArenaProcessHandle(
        process=process,
        manifest_path=str(manifest_path),
        arena_root=str(root),
        arena_id=manifest.arena_id,
        stop_event=stop_event,
        cuda_initialized=False,
        ownership_id=ownership_id,
    )


def _query_local_gpu_numa_nodes(server_args) -> tuple[int, ...]:
    import torch

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("shared WeLM embeddings require at least one CUDA device")
    configured_nodes = getattr(server_args, "numa_node", None)
    if configured_nodes is not None:
        if len(configured_nodes) < device_count:
            raise ValueError(
                "--numa-node must provide one NUMA node for every visible GPU"
            )
        return tuple(int(node) for node in configured_nodes[:device_count])

    from sglang.srt.utils.numa_utils import _query_numa_node_for_gpu

    result = []
    for device in range(device_count):
        nodes = _query_numa_node_for_gpu(device)
        if len(nodes) != 1:
            raise RuntimeError(
                f"expected exactly one NUMA node for visible GPU {device}, "
                f"found {nodes}"
            )
        result.append(nodes[0])
    return tuple(result)


def validate_welm_shared_embedding_checkpoint_config(checkpoint: Path) -> None:
    config_path = Path(checkpoint) / "config.json"
    if not config_path.is_file():
        return
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid WeLM checkpoint config: {config_path}") from exc
    text_config = config.get("text_config") or config
    scale_seq_times = int(
        text_config.get("scale_seq_times", config.get("scale_seq_times", 0)) or 0
    )
    if scale_seq_times != 0:
        raise ValueError(
            "shared WeLM host embeddings require scale_seq_times == 0"
        )
    oe_vocab_sizes = text_config.get(
        "oe_vocab_sizes", config.get("oe_vocab_sizes")
    )
    if oe_vocab_sizes is not None and len(oe_vocab_sizes) != 4:
        raise ValueError("shared WeLM host embeddings require exactly four OE tables")


def launch_welm_embedding_arena_manager(server_args):
    policy = WeLMSharedEmbeddingPolicy(server_args.welm_shared_embedding_policy)
    if policy is WeLMSharedEmbeddingPolicy.DISABLED:
        return None
    run_id = os.environ.get("SGLANG_RUN_ID")
    if run_id is None:
        raise RuntimeError("SGLANG_RUN_ID is missing for arena ownership")
    try:
        _validate_welm_embedding_run_id(run_id)
    except ValueError as exc:
        raise RuntimeError("SGLANG_RUN_ID is unsafe for arena ownership") from exc
    validate_welm_shared_embedding_checkpoint_config(Path(server_args.model_path))
    gpu_numa_nodes = _query_local_gpu_numa_nodes(server_args)
    plans = plan_welm_embedding_replicas(
        policy,
        gpu_numa_nodes,
        server_args.welm_shared_embedding_numa_node,
    )
    lease = _reserve_welm_embedding_arena_lease(_ARENA_PARENT, run_id)
    try:
        handle = launch_welm_embedding_arena_process(
            checkpoint=Path(server_args.model_path),
            root=lease.arena_root,
            plans=plans,
            ownership_id=lease.ownership_id,
            process_started_callback=lease.update_manager_pid,
        )
    except WeLMEmbeddingArenaManagerStillAliveError:
        _quarantine_welm_embedding_arena_lease(lease)
        raise
    except BaseException:
        lease.close()
        raise
    handle.lease = lease
    try:
        if lease.manager_pid != handle.process.pid:
            lease.update_manager_pid(handle.process.pid)
    except BaseException:
        handle.close()
        raise
    server_args.welm_shared_embedding_manifest_path = handle.manifest_path
    return handle
