import gc
import multiprocessing as mp
import os
import struct
from pathlib import Path

import pytest
import torch

import sglang.srt.utils.shared_uva_tensor as shared_uva
from sglang.srt.utils.shared_uva_tensor import (
    SharedTensorFileSpec,
    SharedUVATensorView,
    validate_shared_tensor_file_spec,
)


def _write_float32_file(path: Path, values: tuple[float, ...]) -> None:
    path.write_bytes(struct.pack(f"{len(values)}f", *values))


def _spec(path: Path, *, shape=(2, 2), nbytes=16) -> SharedTensorFileSpec:
    return SharedTensorFileSpec(
        key="model.embed_tokens.weight",
        path=str(path),
        shape=shape,
        dtype="float32",
        nbytes=nbytes,
        replica_id="interleave-numa-0-1",
        numa_nodes=(0, 1),
        inode=path.stat().st_ino,
    )


@pytest.fixture
def fake_cuda(monkeypatch):
    events = []

    def register(cpu_tensor, device, release_callback, registered_callback):
        from sglang.jit_kernel.memory_allocator import _make_tensor_from_ptr

        events.append(("register", device, cpu_tensor.data_ptr()))
        registered_callback(cpu_tensor.data_ptr())
        cuda_tensor = _make_tensor_from_ptr(
            cpu_tensor.data_ptr(),
            tuple(cpu_tensor.shape),
            cpu_tensor.dtype,
            torch.device("cpu"),
            release_callback,
        )
        return cuda_tensor, cpu_tensor.data_ptr()

    monkeypatch.setattr(shared_uva, "_register_cuda_mapping", register)
    monkeypatch.setattr(
        shared_uva,
        "_synchronize_cuda_device",
        lambda device: events.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        shared_uva,
        "_unregister_host_mapping",
        lambda host_ptr: events.append(("unregister", host_ptr)),
    )
    return events


def test_reconstructs_dtype_and_shape_from_shared_file(tmp_path, fake_cuda):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))

    with SharedUVATensorView.open(_spec(path), device=0, arena_root=tmp_path) as view:
        assert view.cuda_tensor.dtype == torch.float32
        assert view.cuda_tensor.shape == (2, 2)
        torch.testing.assert_close(
            view.cuda_tensor,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )


def test_rejects_file_with_wrong_exact_size(tmp_path, fake_cuda):
    path = tmp_path / "weight.bin"
    path.write_bytes(b"short")

    with pytest.raises(ValueError, match="file size"):
        SharedUVATensorView.open(_spec(path), device=0, arena_root=tmp_path)

    assert fake_cuda == []


def test_close_is_idempotent_and_orders_cuda_cleanup(tmp_path, fake_cuda):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))
    view = SharedUVATensorView.open(_spec(path), device=3, arena_root=tmp_path)

    view.close()
    view.close()

    assert [event[0] for event in fake_cuda] == [
        "register",
        "synchronize",
        "unregister",
    ]
    with pytest.raises(RuntimeError, match="closed"):
        _ = view.cuda_tensor


def test_close_retains_mapping_and_fd_when_unregister_fails_until_retry(
    tmp_path, fake_cuda, monkeypatch
):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))
    view = SharedUVATensorView.open(_spec(path), device=0, arena_root=tmp_path)
    fd = view._fd
    mapping = view._mapping

    def fail_unregister(_host_ptr):
        raise RuntimeError("unregister failed")

    monkeypatch.setattr(shared_uva, "_unregister_host_mapping", fail_unregister)

    with pytest.raises(RuntimeError, match="unregister failed"):
        view.close()

    assert not mapping.closed
    assert os.fstat(fd).st_size == 16
    assert shared_uva._is_quarantined(view._lease)

    monkeypatch.setattr(
        shared_uva,
        "_unregister_host_mapping",
        lambda host_ptr: fake_cuda.append(("unregister-retry", host_ptr)),
    )
    view.close()
    assert mapping.closed
    assert not shared_uva._is_quarantined(view._lease)
    with pytest.raises(OSError):
        os.fstat(fd)


def test_close_rejects_outstanding_cuda_tensor_alias(tmp_path, fake_cuda):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))
    view = SharedUVATensorView.open(_spec(path), device=0, arena_root=tmp_path)
    alias = view.cuda_tensor.view(-1)

    with pytest.raises(RuntimeError, match="outstanding CUDA tensor aliases"):
        view.close()

    assert not view._mapping.closed
    del alias
    gc.collect()
    assert not view._mapping.closed
    view.close()
    assert view._mapping.closed
    assert [event[0] for event in fake_cuda].count("synchronize") == 2


def test_open_after_unlink_fails(tmp_path, fake_cuda):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))
    spec = _spec(path)
    path.unlink()

    with pytest.raises(FileNotFoundError):
        SharedUVATensorView.open(spec, device=0, arena_root=tmp_path)


def test_rejects_relative_manifest_path():
    with pytest.raises(ValueError, match="absolute"):
        SharedTensorFileSpec(
            key="weight",
            path="relative/weight.bin",
            shape=(2, 2),
            dtype="float32",
            nbytes=16,
            replica_id="replica",
            numa_nodes=(0,),
            inode=1,
        )


def test_rejects_manifest_path_outside_arena_root(tmp_path):
    arena_root = tmp_path / "arena"
    arena_root.mkdir()
    outside = tmp_path / "outside.bin"
    _write_float32_file(outside, (1.0, 2.0, 3.0, 4.0))

    with pytest.raises(ValueError, match="outside arena root"):
        validate_shared_tensor_file_spec(_spec(outside), arena_root)


def test_rejects_manifest_nbytes_inconsistent_with_shape(tmp_path):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="nbytes"):
        _spec(path, nbytes=8)


def test_open_rejects_inode_changed_after_manifest_validation(tmp_path, fake_cuda):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))
    spec = _spec(path)
    with path.open("rb") as old_file:
        path.unlink()
        _write_float32_file(path, (5.0, 6.0, 7.0, 8.0))
        assert os.fstat(old_file.fileno()).st_ino != path.stat().st_ino

        with pytest.raises(ValueError, match="inode"):
            SharedUVATensorView.open(spec, device=0, arena_root=tmp_path)


def test_published_mapping_is_opened_read_only(
    tmp_path, fake_cuda, monkeypatch
):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))
    spec = _spec(path)
    open_flags = []
    mmap_protections = []
    real_open = shared_uva.os.open
    real_mmap = shared_uva.mmap.mmap

    def record_open(path, flags, *args, **kwargs):
        open_flags.append(flags)
        return real_open(path, flags, *args, **kwargs)

    def record_mmap(fd, length, *, flags, prot):
        mmap_protections.append(prot)
        return real_mmap(fd, length, flags=flags, prot=prot)

    monkeypatch.setattr(shared_uva.os, "open", record_open)
    monkeypatch.setattr(shared_uva.mmap, "mmap", record_mmap)

    with SharedUVATensorView.open(
        spec, device=0, arena_root=tmp_path
    ) as view:
        assert view.cuda_tensor[0, 0].item() == 1.0

    assert open_flags
    assert all(flags & os.O_ACCMODE == os.O_RDONLY for flags in open_flags)
    assert mmap_protections == [shared_uva.mmap.PROT_READ]


def test_open_failure_after_registration_unregisters_and_closes_file(
    tmp_path, fake_cuda, monkeypatch
):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))
    opened_fds = []
    mappings = []
    real_open = os.open
    real_mmap = shared_uva.mmap.mmap

    def record_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened_fds.append(fd)
        return fd

    def record_mmap(*args, **kwargs):
        mapping = real_mmap(*args, **kwargs)
        mappings.append(mapping)
        return mapping

    def fail_after_register(cpu_tensor, device, release_callback, on_registered):
        host_ptr = cpu_tensor.data_ptr()
        on_registered(host_ptr)
        raise RuntimeError("post-register failure")

    monkeypatch.setattr(shared_uva.os, "open", record_open)
    monkeypatch.setattr(shared_uva.mmap, "mmap", record_mmap)
    monkeypatch.setattr(shared_uva, "_register_cuda_mapping", fail_after_register)

    with pytest.raises(RuntimeError, match="post-register failure"):
        SharedUVATensorView.open(_spec(path), device=0, arena_root=tmp_path)

    assert mappings[0].closed
    with pytest.raises(OSError):
        os.fstat(opened_fds[0])
    assert any(event[0] == "unregister" for event in fake_cuda)


def test_open_quarantines_mapping_when_abort_unregister_fails(
    tmp_path, fake_cuda, monkeypatch
):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))
    before = set(id(lease) for lease in shared_uva._QUARANTINED_FAILED_ATTACHMENTS)

    def fail_after_register(cpu_tensor, device, release_callback, on_registered):
        on_registered(cpu_tensor.data_ptr())
        raise RuntimeError("post-register failure")

    monkeypatch.setattr(shared_uva, "_register_cuda_mapping", fail_after_register)
    monkeypatch.setattr(
        shared_uva,
        "_unregister_host_mapping",
        lambda _host_ptr: (_ for _ in ()).throw(RuntimeError("unregister failed")),
    )

    with pytest.raises(RuntimeError, match="mapping quarantined"):
        SharedUVATensorView.open(_spec(path), device=0, arena_root=tmp_path)

    quarantined = [
        lease
        for lease in shared_uva._QUARANTINED_FAILED_ATTACHMENTS
        if id(lease) not in before
    ]
    assert len(quarantined) == 1
    lease = quarantined[0]
    assert not lease.mapping.closed
    assert os.fstat(lease.fd).st_size == 16

    monkeypatch.setattr(shared_uva, "_unregister_host_mapping", lambda _host_ptr: None)
    lease.abort_open()
    shared_uva._release_quarantined_attachment(lease)


def _cuda_shared_mapping_worker(spec, arena_root, device, queue):
    try:
        torch.cuda.set_device(device)
        with SharedUVATensorView.open(
            spec, device=device, arena_root=arena_root
        ) as view:
            indices = torch.tensor([1, 0], dtype=torch.long, device=f"cuda:{device}")
            rows_tensor = view.cuda_tensor.index_select(0, indices).cpu()
            rows = rows_tensor.tolist()
            checksum = float(rows_tensor.sum().item())
            inode = view.file_inode
        queue.put((device, inode, rows, checksum, None))
    except BaseException as exc:
        queue.put((device, None, None, None, repr(exc)))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 5,
    reason="requires CUDA devices 0 and 4",
)
def test_two_gpu_processes_map_the_same_inode(tmp_path):
    path = tmp_path / "weight.bin"
    _write_float32_file(path, (1.0, 2.0, 3.0, 4.0))
    spec = _spec(path)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_cuda_shared_mapping_worker,
            args=(spec, str(tmp_path), device, queue),
        )
        for device in (0, 4)
    ]

    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=60)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)

    assert [result[4] for result in results] == [None, None]
    assert len({result[1] for result in results}) == 1
    assert [result[2] for result in results] == [
        [[3.0, 4.0], [1.0, 2.0]],
        [[3.0, 4.0], [1.0, 2.0]],
    ]
    assert [result[3] for result in results] == [10.0, 10.0]
