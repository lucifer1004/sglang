from __future__ import annotations

import gc
import mmap
import os
import stat
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "int64": torch.int64,
}


@dataclass(frozen=True)
class SharedTensorFileSpec:
    key: str
    path: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    replica_id: str
    numa_nodes: tuple[int, ...]
    inode: int
    sampled_numa_nodes: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("shared tensor key must not be empty")
        if not Path(self.path).is_absolute():
            raise ValueError("shared tensor path must be absolute")
        if not self.replica_id:
            raise ValueError("shared tensor replica_id must not be empty")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError("shared tensor shape must contain positive dimensions")
        if self.dtype not in _DTYPES:
            raise ValueError(f"unsupported shared tensor dtype: {self.dtype!r}")
        if not self.numa_nodes or any(node < 0 for node in self.numa_nodes):
            raise ValueError("shared tensor NUMA nodes must be non-negative")
        if self.inode <= 0:
            raise ValueError("shared tensor inode must be positive")
        if any(node < 0 or count <= 0 for node, count in self.sampled_numa_nodes):
            raise ValueError("sampled NUMA nodes and page counts must be positive")

        expected_nbytes = _numel(self.shape) * _dtype(self.dtype).itemsize
        if self.nbytes != expected_nbytes:
            raise ValueError(
                f"shared tensor nbytes {self.nbytes} does not match "
                f"shape/dtype size {expected_nbytes}"
            )


def validate_shared_tensor_file_spec(
    spec: SharedTensorFileSpec, arena_root: str | os.PathLike[str]
) -> Path:
    root = Path(arena_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"arena root is not a directory: {root}")
    path = Path(spec.path).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"shared tensor path {path} is outside arena root {root}"
        ) from exc
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"shared tensor path is not a regular file: {path}")
    if file_stat.st_ino != spec.inode:
        raise ValueError(
            f"shared tensor inode {file_stat.st_ino} does not match "
            f"manifest inode {spec.inode}"
        )
    return path


def _dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        value *= dim
    return value


class _SharedMappingLease:
    def __init__(
        self,
        *,
        fd: int,
        mapping: mmap.mmap,
        cpu_tensor: torch.Tensor,
    ) -> None:
        self.fd = fd
        self.mapping = mapping
        self.cpu_tensor: torch.Tensor | None = cpu_tensor
        self.host_ptr: int | None = None
        self.registered = False
        self.storage_released = False
        self.close_requested = False
        self.closed = False
        self.cleanup_error: BaseException | None = None
        self._lock = threading.RLock()

    def mark_registered(self, host_ptr: int) -> None:
        with self._lock:
            self.host_ptr = host_ptr
            self.registered = True

    def release_cuda_storage(self, _device_ptr: int) -> None:
        with self._lock:
            self.storage_released = True

    def request_close(self, device: int) -> None:
        with self._lock:
            if self.closed:
                return
            _synchronize_cuda_device(device)
            self.close_requested = True

    def finish_close(self) -> None:
        with self._lock:
            if self.closed:
                return
            if not self.storage_released:
                raise RuntimeError(
                    "cannot close shared UVA mapping with outstanding CUDA tensor aliases"
                )
            self._cleanup_locked()

    def abort_open(self) -> None:
        with self._lock:
            self.close_requested = True
            self.storage_released = True
            self._cleanup_locked()

    def _cleanup_locked(self) -> None:
        if self.closed:
            return
        if self.registered:
            assert self.host_ptr is not None
            _unregister_host_mapping(self.host_ptr)
            self.registered = False

        self.cpu_tensor = None
        gc.collect()
        first_error: BaseException | None = None
        try:
            self.mapping.close()
        except BaseException as exc:
            first_error = exc
        try:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
        except BaseException as exc:
            if first_error is None:
                first_error = exc

        self.closed = self.mapping.closed and self.fd < 0
        self.cleanup_error = first_error
        if first_error is not None:
            raise first_error


_QUARANTINED_FAILED_ATTACHMENTS: list[_SharedMappingLease] = []
_QUARANTINE_LOCK = threading.Lock()


def _quarantine_failed_attachment(lease: _SharedMappingLease) -> None:
    with _QUARANTINE_LOCK:
        if not any(item is lease for item in _QUARANTINED_FAILED_ATTACHMENTS):
            _QUARANTINED_FAILED_ATTACHMENTS.append(lease)


def _release_quarantined_attachment(lease: _SharedMappingLease) -> None:
    with _QUARANTINE_LOCK:
        _QUARANTINED_FAILED_ATTACHMENTS[:] = [
            item for item in _QUARANTINED_FAILED_ATTACHMENTS if item is not lease
        ]


def _is_quarantined(lease: _SharedMappingLease) -> bool:
    with _QUARANTINE_LOCK:
        return any(item is lease for item in _QUARANTINED_FAILED_ATTACHMENTS)


def _register_cuda_mapping(
    cpu_tensor: torch.Tensor,
    device: int,
    release_callback: Callable[[int], None],
    registered_callback: Callable[[int], None],
) -> tuple[torch.Tensor, int]:
    import cuda.bindings.runtime as cuda_rt

    from sglang.jit_kernel.memory_allocator import _make_tensor_from_ptr
    from sglang.srt.utils import check_cuda_result

    torch.cuda.set_device(device)
    host_ptr = int(cpu_tensor.data_ptr())
    nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
    read_only_flag = getattr(cuda_rt, "cudaHostRegisterReadOnly", None)
    if read_only_flag is None:
        raise RuntimeError("CUDA runtime does not support read-only host registration")
    flags = (
        cuda_rt.cudaHostRegisterPortable
        | cuda_rt.cudaHostRegisterMapped
        | read_only_flag
    )
    check_cuda_result(cuda_rt.cudaHostRegister(host_ptr, nbytes, flags))
    registered_callback(host_ptr)
    device_ptr = int(
        check_cuda_result(cuda_rt.cudaHostGetDevicePointer(host_ptr, 0))[0]
    )
    cuda_tensor = _make_tensor_from_ptr(
        device_ptr,
        tuple(cpu_tensor.shape),
        cpu_tensor.dtype,
        torch.device("cuda", device),
        release_callback,
    )
    return cuda_tensor, host_ptr


def _synchronize_cuda_device(device: int) -> None:
    torch.cuda.synchronize(device)


def _unregister_host_mapping(host_ptr: int) -> None:
    import cuda.bindings.runtime as cuda_rt

    from sglang.srt.utils import check_cuda_result

    check_cuda_result(cuda_rt.cudaHostUnregister(host_ptr))


class SharedUVATensorView:
    def __init__(
        self,
        *,
        spec: SharedTensorFileSpec,
        device: int,
        lease: _SharedMappingLease,
        cuda_tensor: torch.Tensor,
        file_inode: int,
    ) -> None:
        self.spec = spec
        self.device = device
        self.file_inode = file_inode
        self._lease = lease
        self._fd = lease.fd
        self._mapping = lease.mapping
        self._cuda_tensor: torch.Tensor | None = cuda_tensor
        self._closed = False

    @classmethod
    def open(
        cls,
        spec: SharedTensorFileSpec,
        device: int,
        *,
        arena_root: str | os.PathLike[str],
    ) -> "SharedUVATensorView":
        path = validate_shared_tensor_file_spec(spec, arena_root)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = os.open(path, flags)
        mapping = None
        lease = None
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"shared tensor path is not a regular file: {path}")
            if file_stat.st_ino != spec.inode:
                raise ValueError(
                    f"opened shared tensor inode {file_stat.st_ino} does not match "
                    f"manifest inode {spec.inode}"
                )
            if file_stat.st_size != spec.nbytes:
                raise ValueError(
                    f"shared tensor file size {file_stat.st_size} does not match "
                    f"manifest nbytes {spec.nbytes}"
                )
            mapping = mmap.mmap(
                fd,
                spec.nbytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given buffer is not writable",
                    category=UserWarning,
                )
                cpu_tensor = torch.frombuffer(
                    mapping, dtype=_dtype(spec.dtype)
                ).view(spec.shape)
            lease = _SharedMappingLease(fd=fd, mapping=mapping, cpu_tensor=cpu_tensor)
            cuda_tensor, _ = _register_cuda_mapping(
                cpu_tensor,
                device,
                lease.release_cuda_storage,
                lease.mark_registered,
            )
            return cls(
                spec=spec,
                device=device,
                lease=lease,
                cuda_tensor=cuda_tensor,
                file_inode=file_stat.st_ino,
            )
        except BaseException as original_error:
            if lease is not None:
                try:
                    lease.abort_open()
                except BaseException as cleanup_error:
                    _quarantine_failed_attachment(lease)
                    raise RuntimeError(
                        "shared UVA attachment failed and CUDA registration "
                        "cleanup could not be confirmed; mapping quarantined"
                    ) from cleanup_error
            else:
                first_cleanup_error = None
                if mapping is not None:
                    try:
                        mapping.close()
                    except BaseException as exc:
                        first_cleanup_error = exc
                try:
                    os.close(fd)
                except BaseException as exc:
                    if first_cleanup_error is None:
                        first_cleanup_error = exc
                if first_cleanup_error is not None:
                    raise RuntimeError(
                        "shared UVA attachment failed and file cleanup also failed"
                    ) from first_cleanup_error
            raise original_error

    @property
    def cuda_tensor(self) -> torch.Tensor:
        if self._closed or self._cuda_tensor is None:
            raise RuntimeError("shared UVA tensor view is closed")
        return self._cuda_tensor

    def close(self) -> None:
        if self._closed:
            return

        try:
            self._lease.request_close(self.device)
            self._cuda_tensor = None
            gc.collect()
            self._lease.finish_close()
        except BaseException:
            _quarantine_failed_attachment(self._lease)
            raise
        else:
            self._closed = True
            _release_quarantined_attachment(self._lease)

    def __enter__(self) -> "SharedUVATensorView":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            lease = getattr(self, "_lease", None)
            if lease is not None and not lease.closed:
                _quarantine_failed_attachment(lease)
