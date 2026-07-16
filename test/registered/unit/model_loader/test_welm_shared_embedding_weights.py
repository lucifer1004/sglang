import ctypes
import json
import os
import signal
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.srt.model_loader.welm_shared_embedding_weights import (
    LinuxNumaPlacementAdapter,
    NumaPlacementMode,
    WeLMSharedEmbeddingPolicy,
    calculate_welm_embedding_byte_counts,
    build_welm_shared_embedding_dry_run_report,
    create_welm_embedding_arena,
    discover_welm_shared_embedding_checkpoint_tensors,
    load_welm_embedding_arena_manifest,
    plan_welm_embedding_replicas,
)


GPU_NUMA_NODES = (0, 0, 0, 0, 1, 1, 1, 1)


def _create_leased_arena_root(lease) -> None:
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _write_welm_embedding_arena_owner,
    )

    lease.arena_root.mkdir(mode=0o700)
    _write_welm_embedding_arena_owner(lease.arena_root, lease.ownership_id)


def test_stale_arena_cleanup_preserves_locked_live_lease(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        cleanup_stale_welm_embedding_arenas,
    )

    lease = _reserve_welm_embedding_arena_lease(tmp_path, "live-run")
    _create_leased_arena_root(lease)
    (lease.arena_root / "sentinel").write_text("live")
    try:
        report = cleanup_stale_welm_embedding_arenas(tmp_path)

        assert report.active == (str(lease.arena_root),)
        assert report.cleaned == ()
        assert lease.arena_root.is_dir()
    finally:
        lease.close()


def test_sigkill_owner_releases_lease_for_next_startup(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        cleanup_stale_welm_embedding_arenas,
    )

    ready_path = tmp_path / "ready"
    script = f"""
import time
from pathlib import Path
from sglang.srt.model_loader.welm_shared_embedding_weights import (
    _reserve_welm_embedding_arena_lease,
    _write_welm_embedding_arena_owner,
)
lease = _reserve_welm_embedding_arena_lease(Path({str(tmp_path)!r}), "killed-owner")
lease.arena_root.mkdir(mode=0o700)
_write_welm_embedding_arena_owner(lease.arena_root, lease.ownership_id)
Path({str(ready_path)!r}).write_text("ready")
time.sleep(60)
"""
    process = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.monotonic() + 10
        while not ready_path.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"lease owner exited early: {process.returncode}")
            time.sleep(0.01)
        assert ready_path.is_file()
        active = cleanup_stale_welm_embedding_arenas(tmp_path)
        assert active.active == (
            str(tmp_path / "sglang-welm-embedding-killed-owner"),
        )

        process.kill()
        process.wait(timeout=10)

        cleaned = cleanup_stale_welm_embedding_arenas(tmp_path)
        assert cleaned.cleaned == (
            str(tmp_path / "sglang-welm-embedding-killed-owner"),
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_stale_arena_cleanup_removes_unlocked_lease_and_arena(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        cleanup_stale_welm_embedding_arenas,
    )

    lease = _reserve_welm_embedding_arena_lease(tmp_path, "stale-run")
    _create_leased_arena_root(lease)
    (lease.arena_root / "sentinel").write_text("stale")
    lease_path = lease.lease_path
    arena_root = lease.arena_root
    lease.close(remove_file=False)

    report = cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.cleaned == (str(arena_root),)
    assert report.active == ()
    assert not arena_root.exists()
    assert not lease_path.exists()


def test_stale_arena_cleanup_keeps_multiple_live_instances_isolated(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        cleanup_stale_welm_embedding_arenas,
    )

    first = _reserve_welm_embedding_arena_lease(tmp_path, "first-run")
    _create_leased_arena_root(first)
    second = _reserve_welm_embedding_arena_lease(tmp_path, "second-run")
    _create_leased_arena_root(second)
    try:
        report = cleanup_stale_welm_embedding_arenas(tmp_path)

        assert report.active == tuple(
            sorted((str(first.arena_root), str(second.arena_root)))
        )
        assert report.cleaned == ()
        assert first.arena_root.is_dir()
        assert second.arena_root.is_dir()
    finally:
        first.close()
        second.close()


def test_live_lease_is_authoritative_over_dead_manager_pid_metadata(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        cleanup_stale_welm_embedding_arenas,
    )

    lease = _reserve_welm_embedding_arena_lease(tmp_path, "pid-reuse-run")
    _create_leased_arena_root(lease)
    lease.update_manager_pid(999_999_999)
    try:
        metadata = json.loads(lease.lease_path.read_text())
        assert metadata["owner_start_time"] > 0
        report = cleanup_stale_welm_embedding_arenas(tmp_path)

        assert report.active == (str(lease.arena_root),)
        assert report.cleaned == ()
    finally:
        lease.close()


def test_unlocked_lease_waits_for_registered_manager_exit(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        cleanup_stale_welm_embedding_arenas,
    )

    lease = _reserve_welm_embedding_arena_lease(tmp_path, "retiring-manager")
    _create_leased_arena_root(lease)
    lease.update_manager_pid(os.getpid())
    arena_root = lease.arena_root
    lease_path = lease.lease_path
    lease.close(remove_file=False)
    try:
        report = cleanup_stale_welm_embedding_arenas(tmp_path)

        assert report.active == (str(arena_root),)
        assert report.cleaned == ()
        assert arena_root.is_dir()
        assert lease_path.is_file()
        with pytest.raises(RuntimeError, match="manager is still exiting"):
            _reserve_welm_embedding_arena_lease(tmp_path, "retiring-manager")
    finally:
        if arena_root.exists():
            shutil.rmtree(arena_root)
        lease_path.unlink(missing_ok=True)


def test_manager_pid_reuse_does_not_preserve_stale_arena(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        cleanup_stale_welm_embedding_arenas,
    )

    lease = _reserve_welm_embedding_arena_lease(tmp_path, "reused-manager-pid")
    _create_leased_arena_root(lease)
    lease.update_manager_pid(os.getpid())
    arena_root = lease.arena_root
    lease_path = lease.lease_path
    lease.close(remove_file=False)
    metadata = json.loads(lease_path.read_text())
    metadata["manager_start_time"] += 1
    lease_path.write_text(json.dumps(metadata))

    report = cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.cleaned == (str(arena_root),)
    assert report.active == ()
    assert not arena_root.exists()
    assert not lease_path.exists()


def test_lease_directory_must_be_private_and_owned_by_current_user(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
    )

    lease_directory = tmp_path / ".sglang-welm-embedding-leases"
    lease_directory.mkdir(mode=0o700)
    lease_directory.chmod(0o777)

    with pytest.raises(RuntimeError, match="lease directory"):
        _reserve_welm_embedding_arena_lease(tmp_path, "unsafe-directory")


def test_old_lease_close_cannot_unlink_replacement_file(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
    )

    lease = _reserve_welm_embedding_arena_lease(tmp_path, "replaced-lease")
    lease.lease_path.unlink()
    lease.lease_path.write_text("replacement")
    lease.lease_path.chmod(0o600)

    lease.close()

    assert lease.lease_path.read_text() == "replacement"


def test_reserve_never_overwrites_malformed_orphan_lease(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
    )

    lease_directory = tmp_path / ".sglang-welm-embedding-leases"
    lease_directory.mkdir(mode=0o700)
    lease_path = lease_directory / "malformed-orphan.lock"
    lease_path.write_text("{not-json")
    lease_path.chmod(0o600)

    with pytest.raises(RuntimeError, match="requires manual cleanup"):
        _reserve_welm_embedding_arena_lease(tmp_path, "malformed-orphan")

    assert lease_path.read_text() == "{not-json"


def test_stale_arena_cleanup_removes_legacy_arena_with_dead_manager(
    tmp_path, monkeypatch
):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "sglang-welm-embedding-legacy-dead"
    create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=arena_root,
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    monkeypatch.setattr(shared_weights, "_is_process_alive", lambda _pid: False)

    report = shared_weights.cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.cleaned == (str(arena_root),)
    assert not arena_root.exists()


def test_stale_arena_cleanup_preserves_legacy_arena_with_live_manager(
    tmp_path, monkeypatch
):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "sglang-welm-embedding-legacy-live"
    create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=arena_root,
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    monkeypatch.setattr(shared_weights, "_is_process_alive", lambda _pid: True)

    report = shared_weights.cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.active == (str(arena_root),)
    assert report.cleaned == ()
    assert arena_root.is_dir()


@pytest.mark.parametrize("lease_contents", [None, "{not-json"])
def test_new_protocol_arena_without_valid_lease_is_never_legacy_deleted(
    tmp_path, monkeypatch, lease_contents
):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "sglang-welm-embedding-unbound-new-protocol"
    shared_weights.create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=arena_root,
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    shared_weights._write_welm_embedding_arena_owner(arena_root, "a" * 32)
    if lease_contents is not None:
        lease_directory = tmp_path / ".sglang-welm-embedding-leases"
        lease_directory.mkdir(mode=0o700)
        lease_path = lease_directory / "unbound-new-protocol.lock"
        lease_path.write_text(lease_contents)
        lease_path.chmod(0o600)
    monkeypatch.setattr(shared_weights, "_is_process_alive", lambda _pid: False)

    report = shared_weights.cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.skipped == (str(arena_root),)
    assert report.cleaned == ()
    assert arena_root.is_dir()


@pytest.mark.parametrize("manifest_contents", [None, "{not-json"])
def test_stale_arena_cleanup_preserves_ambiguous_legacy_arena(
    tmp_path, manifest_contents
):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        cleanup_stale_welm_embedding_arenas,
    )

    arena_root = tmp_path / "sglang-welm-embedding-legacy-ambiguous"
    arena_root.mkdir()
    (arena_root / "partial.bin").write_bytes(b"partial")
    if manifest_contents is not None:
        (arena_root / "manifest.json").write_text(manifest_contents)

    report = cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.skipped == (str(arena_root),)
    assert arena_root.is_dir()


def test_stale_arena_cleanup_never_follows_arena_symlink(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        cleanup_stale_welm_embedding_arenas,
    )

    target = tmp_path / "outside"
    target.mkdir()
    (target / "sentinel").write_text("keep")
    arena_link = tmp_path / "sglang-welm-embedding-symlink-run"
    arena_link.symlink_to(target, target_is_directory=True)

    report = cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.skipped == (str(arena_link),)
    assert (target / "sentinel").read_text() == "keep"


def test_stale_arena_cleanup_never_follows_lease_symlink(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        cleanup_stale_welm_embedding_arenas,
    )

    arena_root = tmp_path / "sglang-welm-embedding-hostile-lease"
    arena_root.mkdir()
    lease_directory = tmp_path / ".sglang-welm-embedding-leases"
    lease_directory.mkdir(mode=0o700)
    outside = tmp_path / "outside.lock"
    outside.write_text("keep")
    (lease_directory / "hostile-lease.lock").symlink_to(outside)

    report = cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.skipped == (str(arena_root),)
    assert arena_root.is_dir()
    assert outside.read_text() == "keep"


def test_unlocked_fabricated_lease_cannot_delete_live_legacy_arena(
    tmp_path, monkeypatch
):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "sglang-welm-embedding-fabricated-lease"
    create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=arena_root,
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    lease_directory = tmp_path / ".sglang-welm-embedding-leases"
    lease_directory.mkdir(mode=0o700)
    lease_path = lease_directory / "fabricated-lease.lock"
    lease_path.write_text("{}")
    lease_path.chmod(0o600)
    monkeypatch.setattr(shared_weights, "_is_process_alive", lambda _pid: True)

    report = shared_weights.cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.active == (str(arena_root),)
    assert report.cleaned == ()
    assert arena_root.is_dir()


def test_manager_generation_cannot_delete_replacement_arena(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _remove_matching_welm_embedding_arena_root,
        _write_welm_embedding_arena_owner,
    )

    arena_root = tmp_path / "sglang-welm-embedding-replacement"
    arena_root.mkdir()
    _write_welm_embedding_arena_owner(arena_root, "b" * 32)
    (arena_root / "sentinel").write_text("replacement")

    removed = _remove_matching_welm_embedding_arena_root(
        arena_root, expected_ownership_id="a" * 32
    )

    assert removed is False
    assert (arena_root / "sentinel").read_text() == "replacement"


def test_owned_arena_cleanup_never_follows_root_symlink(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _remove_owned_arena_root,
        _write_welm_embedding_arena_owner,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    target = tmp_path / "target-arena"
    manifest = create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=target,
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    ownership_id = "a" * 32
    _write_welm_embedding_arena_owner(target, ownership_id)
    arena_link = tmp_path / "arena-link"
    arena_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="unsafe arena root"):
        _remove_owned_arena_root(
            arena_link,
            manifest_path=arena_link / "manifest.json",
            expected_arena_id=manifest.arena_id,
            expected_ownership_id=ownership_id,
        )

    assert (target / "manifest.json").is_file()


def test_stale_cleanup_rejects_lease_generation_mismatch(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        cleanup_stale_welm_embedding_arenas,
    )

    lease = _reserve_welm_embedding_arena_lease(tmp_path, "generation-mismatch")
    _create_leased_arena_root(lease)
    (lease.arena_root / "sentinel").write_text("replacement")
    lease_path = lease.lease_path
    arena_root = lease.arena_root
    lease.close(remove_file=False)
    metadata = json.loads(lease_path.read_text())
    metadata["ownership_id"] = "b" * 32
    lease_path.write_text(json.dumps(metadata))

    report = cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.skipped == (str(arena_root),)
    assert arena_root.is_dir()
    assert lease_path.is_file()


def test_stale_arena_cleanup_is_idempotent(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        cleanup_stale_welm_embedding_arenas,
    )

    lease = _reserve_welm_embedding_arena_lease(tmp_path, "idempotent-run")
    _create_leased_arena_root(lease)
    lease.close(remove_file=False)

    first = cleanup_stale_welm_embedding_arenas(tmp_path)
    second = cleanup_stale_welm_embedding_arenas(tmp_path)

    assert first.cleaned == (str(lease.arena_root),)
    assert second == type(second)(cleaned=(), active=(), skipped=())


def test_stale_arena_cleanup_removes_unlocked_orphan_lease(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        cleanup_stale_welm_embedding_arenas,
    )

    lease = _reserve_welm_embedding_arena_lease(tmp_path, "orphan-lease")
    lease_path = lease.lease_path
    lease.close(remove_file=False)

    cleanup_stale_welm_embedding_arenas(tmp_path)

    assert not lease_path.exists()


def test_stale_arena_cleanup_preserves_active_lease_before_root_creation(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        cleanup_stale_welm_embedding_arenas,
    )

    lease = _reserve_welm_embedding_arena_lease(tmp_path, "active-without-root")
    try:
        cleanup_stale_welm_embedding_arenas(tmp_path)

        assert lease.lease_path.is_file()
    finally:
        lease.close()


def test_stale_arena_cleanup_continues_after_one_removal_fails(
    tmp_path, monkeypatch
):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    blocked = shared_weights._reserve_welm_embedding_arena_lease(
        tmp_path, "blocked-stale"
    )
    _create_leased_arena_root(blocked)
    removable = shared_weights._reserve_welm_embedding_arena_lease(
        tmp_path, "removable-stale"
    )
    _create_leased_arena_root(removable)
    blocked.close(remove_file=False)
    removable.close(remove_file=False)
    real_rmtree = shared_weights.shutil.rmtree

    def selective_rmtree(path):
        if Path(path) == blocked.arena_root:
            raise PermissionError("blocked")
        return real_rmtree(path)

    monkeypatch.setattr(shared_weights.shutil, "rmtree", selective_rmtree)

    report = shared_weights.cleanup_stale_welm_embedding_arenas(tmp_path)

    assert report.skipped == (str(blocked.arena_root),)
    assert report.cleaned == (str(removable.arena_root),)
    assert blocked.arena_root.is_dir()
    assert blocked.lease_path.is_file()
    assert not removable.arena_root.exists()
    assert not removable.lease_path.exists()


def test_parent_death_signal_is_installed_before_parent_identity_recheck():
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _install_parent_death_signal,
    )

    events = []

    def prctl(option, death_signal):
        events.append(("prctl", option, death_signal))
        return 0

    def get_parent_pid():
        events.append(("getppid",))
        return 41

    _install_parent_death_signal(
        41,
        prctl=prctl,
        get_parent_pid=get_parent_pid,
    )

    assert events == [("prctl", 1, signal.SIGTERM), ("getppid",)]


def test_parent_death_signal_rejects_parent_that_already_exited():
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _install_parent_death_signal,
    )

    with pytest.raises(RuntimeError, match="parent process 41 exited"):
        _install_parent_death_signal(
            41,
            prctl=lambda *_args: 0,
            get_parent_pid=lambda: 1,
        )


def test_cancel_arena_manager_escalates_and_rejects_survivor():
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        WeLMEmbeddingArenaManagerStillAliveError,
        _cancel_arena_manager_startup,
    )

    events = []

    class StubbornProcess:
        pid = 41

        @staticmethod
        def is_alive():
            return True

        @staticmethod
        def join(timeout):
            events.append(("join", timeout))

        @staticmethod
        def terminate():
            events.append(("terminate",))

        @staticmethod
        def kill():
            events.append(("kill",))

    stop_event = SimpleNamespace(set=lambda: events.append(("stop",)))

    with pytest.raises(WeLMEmbeddingArenaManagerStillAliveError):
        _cancel_arena_manager_startup(
            StubbornProcess(),
            stop_event,
            timeout=0.01,
        )

    assert events == [
        ("stop",),
        ("join", 0.01),
        ("terminate",),
        ("join", 0.01),
        ("kill",),
        ("join", 0.01),
    ]


def test_process_handle_quarantines_lease_when_manager_survives_close(tmp_path):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    lease = shared_weights._reserve_welm_embedding_arena_lease(
        tmp_path, "close-survivor"
    )
    quarantine_size = len(shared_weights._QUARANTINED_WELM_ARENA_LEASES)

    class StubbornProcess:
        pid = 41

        @staticmethod
        def is_alive():
            return True

        @staticmethod
        def join(timeout):
            assert timeout == 0.01
            pass

        @staticmethod
        def terminate():
            pass

        @staticmethod
        def kill():
            pass

    handle = shared_weights.WeLMEmbeddingArenaProcessHandle(
        process=StubbornProcess(),
        manifest_path=str(lease.arena_root / "manifest.json"),
        arena_root=str(lease.arena_root),
        arena_id="a" * 32,
        stop_event=SimpleNamespace(set=lambda: None),
        cuda_initialized=False,
        ownership_id=lease.ownership_id,
        lease=lease,
    )

    try:
        with pytest.raises(
            shared_weights.WeLMEmbeddingArenaManagerStillAliveError
        ):
            handle.close(timeout=0.01)

        assert shared_weights._QUARANTINED_WELM_ARENA_LEASES[
            quarantine_size:
        ] == [lease]
        probe_fd = shared_weights._open_and_lock_file(
            lease.lease_path, blocking=False, create=False
        )
        try:
            assert probe_fd is None
        finally:
            if probe_fd is not None:
                os.close(probe_fd)
    finally:
        while len(shared_weights._QUARANTINED_WELM_ARENA_LEASES) > quarantine_size:
            shared_weights._QUARANTINED_WELM_ARENA_LEASES.pop().close()


def test_interleave_builds_one_replica_for_every_local_rank():
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.INTERLEAVE,
        GPU_NUMA_NODES,
        bind_node=None,
    )

    assert len(plans) == 1
    assert plans[0].numa_nodes == (0, 1)
    assert plans[0].consumer_local_ranks == tuple(range(8))


def test_bind_requires_a_node_used_by_a_consumer_gpu():
    with pytest.raises(ValueError, match="consumer GPU"):
        plan_welm_embedding_replicas(
            WeLMSharedEmbeddingPolicy.BIND,
            GPU_NUMA_NODES,
            bind_node=2,
        )


def test_replicate_numa_builds_one_local_replica_per_node():
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.REPLICATE_NUMA,
        GPU_NUMA_NODES,
        bind_node=None,
    )

    assert [plan.numa_nodes for plan in plans] == [(0,), (1,)]
    assert plans[0].consumer_local_ranks == (0, 1, 2, 3)
    assert plans[1].consumer_local_ranks == (4, 5, 6, 7)


@pytest.mark.parametrize(
    ("cp_size", "attn_tp_size"),
    [(1, 8), (2, 4), (4, 2)],
)
def test_all_cp_attn_tp_topologies_cover_every_local_rank(cp_size, attn_tp_size):
    assert cp_size * attn_tp_size == len(GPU_NUMA_NODES)

    interleave = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.INTERLEAVE,
        GPU_NUMA_NODES,
        bind_node=None,
    )
    replicated = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.REPLICATE_NUMA,
        GPU_NUMA_NODES,
        bind_node=None,
    )

    assert interleave[0].consumer_local_ranks == tuple(range(8))
    assert sorted(
        rank for replica in replicated for rank in replica.consumer_local_ranks
    ) == list(range(8))


def test_disabled_builds_no_replicas():
    assert (
        plan_welm_embedding_replicas(
            WeLMSharedEmbeddingPolicy.DISABLED,
            GPU_NUMA_NODES,
            bind_node=None,
        )
        == ()
    )


def test_rejects_empty_gpu_topology_for_enabled_policy():
    with pytest.raises(ValueError, match="GPU NUMA topology"):
        plan_welm_embedding_replicas(
            WeLMSharedEmbeddingPolicy.INTERLEAVE,
            (),
            bind_node=None,
        )


@pytest.mark.parametrize("policy", ["interleave", "replicate-numa"])
def test_non_bind_policy_rejects_bind_node(policy):
    with pytest.raises(ValueError, match="only valid with bind"):
        plan_welm_embedding_replicas(
            WeLMSharedEmbeddingPolicy(policy),
            GPU_NUMA_NODES,
            bind_node=0,
        )


def test_current_80b_embedding_byte_counts():
    counts = calculate_welm_embedding_byte_counts(
        base_shape=(155_648, 2_048),
        oe_shapes=(
            (16_000_008, 512),
            (16_000_016, 512),
            (16_000_024, 512),
            (16_000_032, 512),
        ),
        element_size=2,
        replica_count=2,
    )

    assert counts.base_bytes == 637_534_208
    assert counts.oe_bytes == 65_536_081_920
    assert counts.logical_bytes == 66_173_616_128
    assert counts.physical_bytes == 132_347_232_256


def test_dry_run_reports_private_and_shared_bytes_without_arena(tmp_path):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")

    report = build_welm_shared_embedding_dry_run_report(
        checkpoint=checkpoint,
        gpu_numa_nodes=(0, 0, 0, 0, 1, 1, 1, 1),
    )

    assert report["logical_weight_bytes"] > 0
    assert report["base_weight_bytes"] > 0
    assert report["oe_weight_bytes"] > report["base_weight_bytes"]
    logical = report["logical_weight_bytes"]
    assert report["topologies"]["cp1-attntp8"]["current_private_bytes"] == logical
    assert report["topologies"]["cp2-attntp4"]["current_private_bytes"] == 2 * logical
    assert report["topologies"]["cp4-attntp2"]["current_private_bytes"] == 4 * logical
    assert report["policy_physical_bytes"] == {
        "bind": logical,
        "interleave": logical,
        "replicate-numa": 2 * logical,
    }
    assert not (tmp_path / "arena").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_shape": (0, 2)}, "positive"),
        ({"oe_shapes": ()}, "OE shape"),
        ({"element_size": 0}, "element_size"),
        ({"replica_count": 0}, "replica_count"),
    ],
)
def test_embedding_byte_counts_reject_invalid_inputs(kwargs, message):
    defaults = dict(
        base_shape=(2, 2),
        oe_shapes=((2, 2),),
        element_size=2,
        replica_count=1,
    )
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        calculate_welm_embedding_byte_counts(**defaults)


SHARED_KEYS = (
    "model.embed_tokens.weight",
    "model.oe_embed.0.weight",
    "model.oe_embed.1.weight",
    "model.oe_embed.2.weight",
    "model.oe_embed.3.weight",
)


def _write_tiny_checkpoint(
    root: Path,
    *,
    missing: str | None = None,
    extra_oe: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    mismatched_oe_shape: bool = False,
    value_offset: int = 0,
) -> Path:
    root.mkdir()
    tensors = {
        SHARED_KEYS[0]: torch.arange(24, dtype=torch.float32)
        .reshape(6, 4)
        .add(value_offset)
        .to(dtype),
    }
    for index, key in enumerate(SHARED_KEYS[1:]):
        width = 3 if mismatched_oe_shape and index == 3 else 2
        tensors[key] = torch.full(
            (7 + index, width), index + 1 + value_offset, dtype=dtype
        )
    if missing is not None:
        tensors.pop(missing)
    if extra_oe:
        tensors["model.oe_embed.4.weight"] = torch.ones((11, 2), dtype=dtype)
    tensors["model.layers.0.input_layernorm.weight"] = torch.ones(4, dtype=dtype)

    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, root / shard)
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {key: shard for key in tensors},
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return root


def test_checkpoint_discovery_returns_canonical_shared_tensor_specs(tmp_path):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")

    specs, checkpoint_identity = discover_welm_shared_embedding_checkpoint_tensors(
        checkpoint
    )

    assert tuple(spec.key for spec in specs) == SHARED_KEYS
    assert specs[0].shape == (6, 4)
    assert tuple(spec.shape for spec in specs[1:]) == (
        (7, 2),
        (8, 2),
        (9, 2),
        (10, 2),
    )
    assert all(spec.dtype == "bfloat16" for spec in specs)
    assert len(checkpoint_identity) == 64


class _FakeNumaAdapter:
    def __init__(self):
        self.events = []

    def apply(self, *, data_ptr, nbytes, mode, numa_nodes):
        assert data_ptr > 0
        assert nbytes > 0
        self.events.append(("apply", mode, numa_nodes))
        return ("previous-policy",)

    def reset(self, previous_policy):
        assert previous_policy == ("previous-policy",)
        self.events.append(("reset",))

    def sample(self, *, data_ptr, nbytes, max_samples):
        assert max_samples == 127
        mode, nodes = next(
            event[1:] for event in reversed(self.events) if event[0] == "apply"
        )
        self.events.append(("sample", mode, nodes))
        if mode is NumaPlacementMode.BIND:
            return {nodes[0]: min(max_samples, 2)}
        return {node: 1 for node in nodes}


class _FakeCFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeLibNuma:
    def __init__(self):
        self.set_calls = []
        self.numa_max_node = _FakeCFunction(lambda: 1)
        self.get_mempolicy = _FakeCFunction(self._get_mempolicy)
        self.set_mempolicy = _FakeCFunction(self._set_mempolicy)

    @staticmethod
    def _get_mempolicy(mode, mask, maxnode, address, flags):
        ctypes.cast(mode, ctypes.POINTER(ctypes.c_int))[0] = 0
        for index in range(max(1, (maxnode + 63) // 64)):
            mask[index] = 0
        return 0

    def _set_mempolicy(self, mode, mask, maxnode):
        words = () if mask is None else tuple(mask)
        self.set_calls.append((mode, words, maxnode))
        return 0


def test_linux_numa_adapter_kernel_mask_includes_highest_requested_node():
    libnuma = _FakeLibNuma()
    adapter = LinuxNumaPlacementAdapter.__new__(LinuxNumaPlacementAdapter)
    adapter._libnuma = libnuma

    adapter.apply(
        data_ptr=1,
        nbytes=4096,
        mode=NumaPlacementMode.INTERLEAVE,
        numa_nodes=(0, 1),
    )

    mode, mask, maxnode = libnuma.set_calls[-1]
    assert mode == 3
    assert mask[0] & 0b11 == 0b11
    assert maxnode >= 3


@pytest.mark.parametrize(
    ("policy", "plans", "expected_modes"),
    [
        (
            WeLMSharedEmbeddingPolicy.BIND,
            ((0,),),
            (NumaPlacementMode.BIND,),
        ),
        (
            WeLMSharedEmbeddingPolicy.INTERLEAVE,
            ((0, 1),),
            (NumaPlacementMode.INTERLEAVE,),
        ),
        (
            WeLMSharedEmbeddingPolicy.REPLICATE_NUMA,
            ((0,), (1,)),
            (NumaPlacementMode.BIND, NumaPlacementMode.BIND),
        ),
    ],
)
def test_create_arena_applies_numa_before_copy_and_publishes_atomically(
    tmp_path, policy, plans, expected_modes, monkeypatch
):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    gpu_nodes = tuple(node for nodes in plans for node in nodes)
    if policy is WeLMSharedEmbeddingPolicy.REPLICATE_NUMA:
        gpu_nodes = (0, 1)
    replica_plans = plan_welm_embedding_replicas(
        policy,
        gpu_nodes,
        bind_node=plans[0][0] if policy is WeLMSharedEmbeddingPolicy.BIND else None,
    )
    adapter = _FakeNumaAdapter()
    arena_root = tmp_path / "arena"
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    copy_tensor_rows = shared_weights._copy_tensor_rows

    def record_copy(**kwargs):
        adapter.events.append(("copy",))
        return copy_tensor_rows(**kwargs)

    monkeypatch.setattr(shared_weights, "_copy_tensor_rows", record_copy)

    manifest = create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=arena_root,
        plans=replica_plans,
        numa_adapter=adapter,
    )

    assert (arena_root / "manifest.json").is_file()
    assert not (arena_root / "manifest.json.tmp").exists()
    assert stat.S_IMODE(arena_root.stat().st_mode) == 0o700
    assert manifest.policy == policy.value
    assert len(manifest.replicas) == len(plans)
    assert [event[1] for event in adapter.events if event[0] == "apply"] == [
        mode for mode in expected_modes for _ in SHARED_KEYS
    ]
    assert [event[0] for event in adapter.events] == [
        event
        for _ in range(len(plans) * len(SHARED_KEYS))
        for event in ("apply", "copy", "reset", "sample")
    ]
    for replica in manifest.replicas:
        assert tuple(spec.key for spec in replica.tensors) == SHARED_KEYS
        for spec in replica.tensors:
            assert stat.S_IMODE(Path(spec.path).stat().st_mode) == 0o600
            assert Path(spec.path).stat().st_ino == spec.inode
            assert spec.sampled_numa_nodes

    loaded = load_welm_embedding_arena_manifest(arena_root / "manifest.json")
    assert loaded == manifest


def test_arena_checkpoint_identity_is_deterministic(tmp_path):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )

    first = create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=tmp_path / "arena-a",
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    second = create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=tmp_path / "arena-b",
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )

    assert first.checkpoint_identity == second.checkpoint_identity
    assert first.arena_id != second.arena_id
    assert first.logical_weight_bytes == second.logical_weight_bytes


def test_arena_checkpoint_identity_includes_tensor_contents(tmp_path):
    first_checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint-a")
    second_checkpoint = _write_tiny_checkpoint(
        tmp_path / "checkpoint-b", value_offset=7
    )
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )

    first = create_welm_embedding_arena(
        checkpoint=first_checkpoint,
        root=tmp_path / "arena-a",
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    second = create_welm_embedding_arena(
        checkpoint=second_checkpoint,
        root=tmp_path / "arena-b",
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )

    assert first.checkpoint_identity != second.checkpoint_identity


def test_create_arena_copies_exact_values_in_bounded_row_chunks(
    tmp_path, monkeypatch
):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    monkeypatch.setattr(
        "sglang.srt.model_loader.welm_shared_embedding_weights._COPY_CHUNK_BYTES",
        8,
    )

    manifest = create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=tmp_path / "arena",
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )

    from safetensors import safe_open

    for spec in manifest.replicas[0].tensors:
        with safe_open(
            checkpoint / "model-00001-of-00001.safetensors",
            framework="pt",
            device="cpu",
        ) as handle:
            expected = handle.get_tensor(spec.key)
        raw = bytearray(Path(spec.path).read_bytes())
        actual = torch.frombuffer(raw, dtype=torch.bfloat16).view(spec.shape)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_create_arena_faults_destination_pages_without_torch_target_ops(
    tmp_path, monkeypatch
):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.INTERLEAVE, (0, 1), bind_node=None
    )

    def reject_torch_target(*args, **kwargs):
        raise AssertionError("destination mmap must be written by the manager thread")

    monkeypatch.setattr(torch, "frombuffer", reject_torch_target)

    manifest = create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=tmp_path / "arena",
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )

    assert len(manifest.replicas[0].tensors) == 5


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("policy", "policy"),
        ("missing-key", "complete shared tensor set"),
        ("duplicate-key", "complete shared tensor set"),
        ("outside-path", "outside arena root"),
        ("alias-path", "alias"),
        ("byte-accounting", "byte accounting"),
        ("sampled-node", "sampled NUMA"),
    ],
)
def test_manifest_loader_rejects_corrupt_or_untrusted_layout(
    tmp_path, corruption, message
):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "arena"
    create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=arena_root,
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    manifest_path = arena_root / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    tensors = payload["replicas"][0]["tensors"]
    if corruption == "policy":
        payload["policy"] = "disabled"
    elif corruption == "missing-key":
        tensors.pop()
    elif corruption == "duplicate-key":
        tensors[-1] = dict(tensors[0])
    elif corruption == "outside-path":
        outside = tmp_path / "outside.bin"
        outside.write_bytes(Path(tensors[0]["path"]).read_bytes())
        tensors[0]["path"] = str(outside)
        tensors[0]["inode"] = outside.stat().st_ino
    elif corruption == "alias-path":
        tensors[1]["path"] = tensors[0]["path"]
        tensors[1]["inode"] = tensors[0]["inode"]
    elif corruption == "byte-accounting":
        payload["logical_weight_bytes"] += 2
    elif corruption == "sampled-node":
        tensors[0]["sampled_numa_nodes"] = [[7, 1]]
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_welm_embedding_arena_manifest(manifest_path)


def test_manifest_loader_rechecks_interleave_coverage(tmp_path):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.INTERLEAVE, (0, 1), bind_node=None
    )
    arena_root = tmp_path / "arena"
    create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=arena_root,
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    manifest_path = arena_root / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    for tensor in payload["replicas"][0]["tensors"]:
        tensor["sampled_numa_nodes"] = [[0, 127]]
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not cover interleave"):
        load_welm_embedding_arena_manifest(manifest_path)


@pytest.mark.parametrize(
    ("checkpoint_kwargs", "message"),
    [
        ({"missing": SHARED_KEYS[0]}, "missing shared embedding"),
        ({"missing": SHARED_KEYS[-1]}, "missing shared embedding"),
        ({"extra_oe": True}, "unexpected OE embedding"),
        ({"dtype": torch.float16}, "BF16"),
        ({"mismatched_oe_shape": True}, "OE embedding widths"),
    ],
)
def test_create_arena_rejects_invalid_checkpoint(
    tmp_path, checkpoint_kwargs, message
):
    checkpoint = _write_tiny_checkpoint(
        tmp_path / "checkpoint", **checkpoint_kwargs
    )
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )

    with pytest.raises(ValueError, match=message):
        create_welm_embedding_arena(
            checkpoint=checkpoint,
            root=tmp_path / "arena",
            plans=plans,
            numa_adapter=_FakeNumaAdapter(),
        )

    assert not (tmp_path / "arena" / "manifest.json").exists()


def test_create_arena_checks_capacity_before_creating_files(tmp_path, monkeypatch):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    monkeypatch.setattr(
        "sglang.srt.model_loader.welm_shared_embedding_weights._available_bytes",
        lambda _path: 1,
    )

    with pytest.raises(OSError, match="insufficient capacity"):
        create_welm_embedding_arena(
            checkpoint=checkpoint,
            root=tmp_path / "arena",
            plans=plans,
            numa_adapter=_FakeNumaAdapter(),
        )

    assert not (tmp_path / "arena").exists()


def test_create_arena_rejects_placement_outside_requested_nodes(tmp_path):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    adapter = _FakeNumaAdapter()
    adapter.sample = lambda **_kwargs: {1: 1}

    with pytest.raises(RuntimeError, match="NUMA placement"):
        create_welm_embedding_arena(
            checkpoint=checkpoint,
            root=tmp_path / "arena",
            plans=plans,
            numa_adapter=adapter,
        )

    assert not (tmp_path / "arena").exists()


def test_create_arena_rejects_incomplete_interleave_placement(tmp_path):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.INTERLEAVE, (0, 1), bind_node=None
    )
    adapter = _FakeNumaAdapter()
    adapter.sample = lambda **_kwargs: {0: 127}

    with pytest.raises(RuntimeError, match="does not cover interleave"):
        create_welm_embedding_arena(
            checkpoint=checkpoint,
            root=tmp_path / "arena",
            plans=plans,
            numa_adapter=adapter,
        )

    assert not (tmp_path / "arena").exists()


@pytest.mark.parametrize("failure_point", ["copy", "publish"])
def test_create_arena_cleans_partial_files_on_failure(
    tmp_path, monkeypatch, failure_point
):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    if failure_point == "copy":
        monkeypatch.setattr(
            "sglang.srt.model_loader.welm_shared_embedding_weights._copy_tensor_rows",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("copy")),
        )
    else:
        monkeypatch.setattr(
            "sglang.srt.model_loader.welm_shared_embedding_weights.os.replace",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("publish")),
        )

    adapter = _FakeNumaAdapter()
    with pytest.raises(RuntimeError, match=failure_point):
        create_welm_embedding_arena(
            checkpoint=checkpoint,
            root=tmp_path / "arena",
            plans=plans,
            numa_adapter=adapter,
        )

    assert not (tmp_path / "arena").exists()
    if failure_point == "copy":
        assert [event[0] for event in adapter.events] == ["apply", "reset"]


def test_create_arena_reports_cleanup_failure(tmp_path, monkeypatch):
    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    monkeypatch.setattr(
        "sglang.srt.model_loader.welm_shared_embedding_weights._copy_tensor_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("copy")),
    )
    monkeypatch.setattr(
        "sglang.srt.model_loader.welm_shared_embedding_weights.shutil.rmtree",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("cleanup")),
    )

    with pytest.raises(ExceptionGroup, match="cleanup failed"):
        create_welm_embedding_arena(
            checkpoint=checkpoint,
            root=tmp_path / "arena",
            plans=plans,
            numa_adapter=_FakeNumaAdapter(),
        )


def test_arena_manager_process_publishes_ready_and_closes(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        launch_welm_embedding_arena_process,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "arena"

    handle = launch_welm_embedding_arena_process(
        checkpoint=checkpoint,
        root=arena_root,
        plans=plans,
        numa_adapter_factory=_FakeNumaAdapter,
        timeout=60,
    )
    try:
        assert handle.process.is_alive()
        assert Path(handle.manifest_path).is_file()
        assert handle.cuda_initialized is False
        assert load_welm_embedding_arena_manifest(handle.manifest_path).arena_id
    finally:
        handle.close(timeout=30)

    assert not handle.process.is_alive()
    assert not arena_root.exists()
    handle.close(timeout=30)


def test_arena_manager_waits_for_parent_registration_before_copy(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        launch_welm_embedding_arena_process,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "arena"
    registered_pids = []

    def register_manager(pid):
        assert arena_root.is_dir()
        assert not (arena_root / "manifest.json").exists()
        registered_pids.append(pid)

    handle = launch_welm_embedding_arena_process(
        checkpoint=checkpoint,
        root=arena_root,
        plans=plans,
        numa_adapter_factory=_FakeNumaAdapter,
        timeout=60,
        process_started_callback=register_manager,
    )
    try:
        assert registered_pids == [handle.process.pid]
        assert Path(handle.manifest_path).is_file()
    finally:
        handle.close(timeout=30)


def test_arena_manager_registration_failure_cancels_before_copy(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        launch_welm_embedding_arena_process,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "arena"

    def fail_registration(_pid):
        assert not (arena_root / "manifest.json").exists()
        raise RuntimeError("manager registration")

    with pytest.raises(RuntimeError, match="manager registration"):
        launch_welm_embedding_arena_process(
            checkpoint=checkpoint,
            root=arena_root,
            plans=plans,
            numa_adapter_factory=_FakeNumaAdapter,
            timeout=60,
            process_started_callback=fail_registration,
        )

    assert not arena_root.exists()


def test_arena_manager_propagates_loader_traceback(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        launch_welm_embedding_arena_process,
    )

    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "arena"

    with pytest.raises(RuntimeError, match="missing-checkpoint"):
        launch_welm_embedding_arena_process(
            checkpoint=tmp_path / "missing-checkpoint",
            root=arena_root,
            plans=plans,
            numa_adapter_factory=_FakeNumaAdapter,
            timeout=60,
        )

    assert not arena_root.exists()


def test_arena_manager_construction_failure_cleans_precreated_root(
    tmp_path, monkeypatch
):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "arena"

    class BrokenContext:
        @staticmethod
        def Pipe(*, duplex):
            assert duplex is False
            raise RuntimeError("pipe construction")

    monkeypatch.setattr(shared_weights.mp, "get_context", lambda _mode: BrokenContext())

    with pytest.raises(RuntimeError, match="pipe construction"):
        shared_weights.launch_welm_embedding_arena_process(
            checkpoint=tmp_path / "unused-checkpoint",
            root=arena_root,
            plans=plans,
        )

    assert not arena_root.exists()


def test_arena_manager_crash_after_ready_is_cleaned_by_owner(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        launch_welm_embedding_arena_process,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "arena"
    handle = launch_welm_embedding_arena_process(
        checkpoint=checkpoint,
        root=arena_root,
        plans=plans,
        numa_adapter_factory=_FakeNumaAdapter,
        timeout=60,
    )

    handle.process.terminate()
    handle.process.join(timeout=30)
    assert handle.process.exitcode is not None
    deadline = time.monotonic() + 5
    while arena_root.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not arena_root.exists()
    handle.close(timeout=30)

    assert not arena_root.exists()


def test_old_manager_cannot_remove_same_path_replacement(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _write_welm_embedding_arena_owner,
        launch_welm_embedding_arena_process,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    arena_root = tmp_path / "arena"
    detached_root = tmp_path / "detached-old-arena"
    handle = launch_welm_embedding_arena_process(
        checkpoint=checkpoint,
        root=arena_root,
        plans=plans,
        numa_adapter_factory=_FakeNumaAdapter,
        timeout=60,
    )
    try:
        arena_root.rename(detached_root)
        arena_root.mkdir(mode=0o700)
        _write_welm_embedding_arena_owner(arena_root, "b" * 32)
        (arena_root / "sentinel").write_text("replacement")

        handle.process.terminate()
        handle.process.join(timeout=30)

        assert not handle.process.is_alive()
        assert (arena_root / "sentinel").read_text() == "replacement"
    finally:
        if handle.process.is_alive():
            handle.process.kill()
            handle.process.join(timeout=30)
        for path in (arena_root, detached_root):
            if path.exists():
                shutil.rmtree(path)


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux PDEATHSIG")
def test_parent_sigkill_releases_lease_and_manager_removes_arena(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _read_process_start_time,
        cleanup_stale_welm_embedding_arenas,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    ready_path = tmp_path / "manager-ready.json"
    arena_root = tmp_path / "sglang-welm-embedding-parent-killed"
    lease_path = (
        tmp_path
        / ".sglang-welm-embedding-leases"
        / "parent-killed.lock"
    )
    (tmp_path / "fake_numa_adapter.py").write_text(
        """
class FakeNumaAdapter:
    def apply(self, *, data_ptr, nbytes, mode, numa_nodes):
        return None

    def reset(self, previous_policy):
        assert previous_policy is None

    def sample(self, *, data_ptr, nbytes, max_samples):
        return {0: 1}
"""
    )
    script = f"""
import json
import time
from pathlib import Path
from fake_numa_adapter import FakeNumaAdapter
from sglang.srt.model_loader.welm_shared_embedding_weights import (
    WeLMEmbeddingReplicaPlan,
    _read_process_start_time,
    _reserve_welm_embedding_arena_lease,
    launch_welm_embedding_arena_process,
)
parent = Path({str(tmp_path)!r})
lease = _reserve_welm_embedding_arena_lease(parent, "parent-killed")
handle = launch_welm_embedding_arena_process(
    checkpoint=Path({str(checkpoint)!r}),
    root=lease.arena_root,
    plans=(WeLMEmbeddingReplicaPlan("shared", (0,), (0,)),),
    numa_adapter_factory=FakeNumaAdapter,
    timeout=60,
    ownership_id=lease.ownership_id,
    process_started_callback=lease.update_manager_pid,
)
Path({str(ready_path)!r}).write_text(json.dumps({{
    "manager_pid": handle.process.pid,
    "manager_start_time": _read_process_start_time(handle.process.pid),
}}))
time.sleep(60)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(tmp_path), env.get("PYTHONPATH")))
    )
    parent = subprocess.Popen([sys.executable, "-c", script], env=env)
    manager_pid = None
    manager_start_time = None
    try:
        deadline = time.monotonic() + 30
        while not ready_path.exists() and time.monotonic() < deadline:
            if parent.poll() is not None:
                raise AssertionError(
                    f"arena owner exited before ready: {parent.returncode}"
                )
            time.sleep(0.02)
        assert ready_path.is_file()
        manager_metadata = json.loads(ready_path.read_text())
        manager_pid = manager_metadata["manager_pid"]
        manager_start_time = manager_metadata["manager_start_time"]
        assert arena_root.is_dir()
        assert lease_path.is_file()

        parent.kill()
        parent.wait(timeout=10)

        deadline = time.monotonic() + 30
        while arena_root.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not arena_root.exists()

        deadline = time.monotonic() + 30
        while (
            _read_process_start_time(manager_pid) == manager_start_time
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert _read_process_start_time(manager_pid) != manager_start_time

        cleanup_stale_welm_embedding_arenas(tmp_path)
        assert not lease_path.exists()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        if (
            manager_pid is not None
            and _read_process_start_time(manager_pid) == manager_start_time
        ):
            try:
                os.kill(manager_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_concurrent_arena_cleanup_never_removes_another_run(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        launch_welm_embedding_arena_process,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    first_root = tmp_path / "arena-first"
    second_root = tmp_path / "arena-second"
    first = launch_welm_embedding_arena_process(
        checkpoint=checkpoint,
        root=first_root,
        plans=plans,
        numa_adapter_factory=_FakeNumaAdapter,
        timeout=60,
    )
    second = launch_welm_embedding_arena_process(
        checkpoint=checkpoint,
        root=second_root,
        plans=plans,
        numa_adapter_factory=_FakeNumaAdapter,
        timeout=60,
    )
    try:
        first_manifest = load_welm_embedding_arena_manifest(first.manifest_path)
        second_manifest = load_welm_embedding_arena_manifest(second.manifest_path)
        assert first_manifest.arena_id != second_manifest.arena_id

        first.close(timeout=30)

        assert not first_root.exists()
        assert second.process.is_alive()
        assert Path(second.manifest_path).is_file()
        assert (
            load_welm_embedding_arena_manifest(second.manifest_path).arena_id
            == second_manifest.arena_id
        )
    finally:
        first.close(timeout=30)
        second.close(timeout=30)

    assert not second_root.exists()


def test_production_launcher_cleans_stale_arena_and_holds_live_lease(
    tmp_path, monkeypatch
):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    stale = shared_weights._reserve_welm_embedding_arena_lease(
        tmp_path, "stale-before-launch"
    )
    _create_leased_arena_root(stale)
    (stale.arena_root / "sentinel").write_text("stale")
    stale.close(remove_file=False)
    launched = {}

    def fake_launch(*, checkpoint, root, plans, **_kwargs):
        assert root.parent == tmp_path
        root.mkdir(mode=0o700)
        launched.update(checkpoint=checkpoint, root=root, plans=plans)
        return SimpleNamespace(
            process=SimpleNamespace(pid=4321),
            manifest_path=str(root / "manifest.json"),
            lease=None,
        )

    monkeypatch.setattr(shared_weights, "_ARENA_PARENT", tmp_path, raising=False)
    monkeypatch.setattr(
        shared_weights, "_query_local_gpu_numa_nodes", lambda _args: (0,)
    )
    monkeypatch.setattr(
        shared_weights, "launch_welm_embedding_arena_process", fake_launch
    )
    monkeypatch.setenv("SGLANG_RUN_ID", "new-live-run")
    server_args = SimpleNamespace(
        welm_shared_embedding_policy="bind",
        welm_shared_embedding_numa_node=0,
        welm_shared_embedding_manifest_path=None,
        model_path=str(tmp_path / "checkpoint"),
    )

    handle = shared_weights.launch_welm_embedding_arena_manager(server_args)
    try:
        assert not stale.arena_root.exists()
        assert handle.lease is not None
        assert handle.lease.manager_pid == 4321
        assert launched["root"] == handle.lease.arena_root
        report = shared_weights.cleanup_stale_welm_embedding_arenas(tmp_path)
        assert report.active == (str(handle.lease.arena_root),)
        assert server_args.welm_shared_embedding_manifest_path == handle.manifest_path
    finally:
        handle.lease.arena_root.rmdir()
        handle.lease.close()


def test_production_launcher_releases_lease_when_manager_startup_fails(
    tmp_path, monkeypatch
):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    lease_path = tmp_path / ".sglang-welm-embedding-leases" / "failed-run.lock"

    def fail_after_observing_locked_lease(**_kwargs):
        assert lease_path.is_file()
        assert (
            shared_weights._open_and_lock_file(
                lease_path, blocking=False, create=False
            )
            is None
        )
        raise RuntimeError("manager startup")

    monkeypatch.setattr(shared_weights, "_ARENA_PARENT", tmp_path, raising=False)
    monkeypatch.setattr(
        shared_weights, "_query_local_gpu_numa_nodes", lambda _args: (0,)
    )
    monkeypatch.setattr(
        shared_weights,
        "launch_welm_embedding_arena_process",
        fail_after_observing_locked_lease,
    )
    monkeypatch.setenv("SGLANG_RUN_ID", "failed-run")
    server_args = SimpleNamespace(
        welm_shared_embedding_policy="bind",
        welm_shared_embedding_numa_node=0,
        welm_shared_embedding_manifest_path=None,
        model_path=str(tmp_path / "checkpoint"),
    )

    with pytest.raises(RuntimeError, match="manager startup"):
        shared_weights.launch_welm_embedding_arena_manager(server_args)

    assert not lease_path.exists()


def test_production_launcher_keeps_lease_when_manager_survives_cancellation(
    tmp_path, monkeypatch
):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    process = SimpleNamespace(pid=4321)
    lease_path = tmp_path / ".sglang-welm-embedding-leases" / "survivor-run.lock"
    quarantine_size = len(shared_weights._QUARANTINED_WELM_ARENA_LEASES)

    def fail_with_surviving_manager(**_kwargs):
        raise shared_weights.WeLMEmbeddingArenaManagerStillAliveError(process)

    monkeypatch.setattr(shared_weights, "_ARENA_PARENT", tmp_path, raising=False)
    monkeypatch.setattr(
        shared_weights, "_query_local_gpu_numa_nodes", lambda _args: (0,)
    )
    monkeypatch.setattr(
        shared_weights,
        "launch_welm_embedding_arena_process",
        fail_with_surviving_manager,
    )
    monkeypatch.setenv("SGLANG_RUN_ID", "survivor-run")
    server_args = SimpleNamespace(
        welm_shared_embedding_policy="bind",
        welm_shared_embedding_numa_node=0,
        welm_shared_embedding_manifest_path=None,
        model_path=str(tmp_path / "checkpoint"),
    )

    try:
        with pytest.raises(
            shared_weights.WeLMEmbeddingArenaManagerStillAliveError
        ):
            shared_weights.launch_welm_embedding_arena_manager(server_args)

        assert len(shared_weights._QUARANTINED_WELM_ARENA_LEASES) == (
            quarantine_size + 1
        )
        probe_fd = shared_weights._open_and_lock_file(
            lease_path, blocking=False, create=False
        )
        try:
            assert probe_fd is None
        finally:
            if probe_fd is not None:
                os.close(probe_fd)
    finally:
        while len(shared_weights._QUARANTINED_WELM_ARENA_LEASES) > quarantine_size:
            shared_weights._QUARANTINED_WELM_ARENA_LEASES.pop().close()


def test_arena_process_handle_close_releases_attached_lease(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        _reserve_welm_embedding_arena_lease,
        launch_welm_embedding_arena_process,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    lease = _reserve_welm_embedding_arena_lease(tmp_path, "attached-lease")
    handle = launch_welm_embedding_arena_process(
        checkpoint=checkpoint,
        root=lease.arena_root,
        plans=plans,
        numa_adapter_factory=_FakeNumaAdapter,
        timeout=60,
    )
    handle.lease = lease
    lease.update_manager_pid(handle.process.pid)

    handle.close(timeout=30)

    assert not lease.arena_root.exists()
    assert not lease.lease_path.exists()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"scale_seq_times": 1, "oe_vocab_sizes": [1, 2, 3, 4]}, "scale_seq"),
        ({"scale_seq_times": 0, "oe_vocab_sizes": [1, 2, 3]}, "four OE"),
    ],
)
def test_shared_checkpoint_config_is_rejected_before_arena_creation(
    tmp_path, config, message
):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        validate_welm_shared_embedding_checkpoint_config,
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        validate_welm_shared_embedding_checkpoint_config(checkpoint)


class _FakeSharedView:
    opened = []
    closed = []

    def __init__(self, spec, device):
        self.spec = spec
        self.device = device
        self.cuda_tensor = torch.zeros(spec.shape, dtype=torch.bfloat16)

    @classmethod
    def open(cls, spec, device, *, arena_root):
        assert Path(spec.path).is_relative_to(Path(arena_root))
        view = cls(spec, device)
        cls.opened.append((spec.replica_id, spec.key, device))
        return view

    def close(self):
        self.closed.append((self.spec.replica_id, self.spec.key, self.device))


@pytest.mark.parametrize(
    ("policy", "gpu_nodes", "gpu_numa_node", "expected_replica"),
    [
        (WeLMSharedEmbeddingPolicy.BIND, (0, 1), 1, "bind-numa-0"),
        (
            WeLMSharedEmbeddingPolicy.INTERLEAVE,
            (0, 1),
            1,
            "interleave-numa-0-1",
        ),
        (
            WeLMSharedEmbeddingPolicy.REPLICATE_NUMA,
            (0, 1),
            1,
            "replica-numa-1",
        ),
    ],
)
def test_registry_maps_only_selected_replica(
    tmp_path, policy, gpu_nodes, gpu_numa_node, expected_replica
):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        WeLMSharedEmbeddingRegistry,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        policy,
        gpu_nodes,
        bind_node=0 if policy is WeLMSharedEmbeddingPolicy.BIND else None,
    )
    manifest = create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=tmp_path / "arena",
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    _FakeSharedView.opened.clear()
    _FakeSharedView.closed.clear()

    registry = WeLMSharedEmbeddingRegistry.from_manifest(
        str(tmp_path / "arena" / "manifest.json"),
        gpu_id=3,
        gpu_numa_node=gpu_numa_node,
        view_opener=_FakeSharedView.open,
    )
    try:
        assert {item[0] for item in _FakeSharedView.opened} == {expected_replica}
        assert len(_FakeSharedView.opened) == len(SHARED_KEYS)
        assert registry.externally_owned_names() == frozenset(SHARED_KEYS)
        assert registry.get(SHARED_KEYS[0]).shape == manifest.replicas[0].tensors[0].shape
        diagnostics = registry.diagnostics()
        assert diagnostics["replica_id"] == expected_replica
        assert diagnostics["mapped_bytes"] > 0
        assert not any(isinstance(value, torch.Tensor) for value in diagnostics.values())
    finally:
        registry.close()

    assert len(_FakeSharedView.closed) == len(SHARED_KEYS)
    registry.close()


def test_registry_replicate_numa_requires_local_replica(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        WeLMSharedEmbeddingRegistry,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.REPLICATE_NUMA, (0, 1), bind_node=None
    )
    create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=tmp_path / "arena",
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )

    with pytest.raises(ValueError, match="local replica"):
        WeLMSharedEmbeddingRegistry.from_manifest(
            str(tmp_path / "arena" / "manifest.json"),
            gpu_id=0,
            gpu_numa_node=2,
            view_opener=_FakeSharedView.open,
        )


def test_registry_closes_partial_attachments_when_open_fails(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        WeLMSharedEmbeddingRegistry,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=tmp_path / "arena",
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )
    opened = []

    def fail_second(spec, device, *, arena_root):
        if opened:
            raise RuntimeError("attach failed")
        view = _FakeSharedView.open(spec, device, arena_root=arena_root)
        opened.append(view)
        return view

    with pytest.raises(RuntimeError, match="attach failed"):
        WeLMSharedEmbeddingRegistry.from_manifest(
            str(tmp_path / "arena" / "manifest.json"),
            gpu_id=0,
            gpu_numa_node=0,
            view_opener=fail_second,
        )

    assert len(opened) == 1
    assert _FakeSharedView.closed[-1][1] == SHARED_KEYS[0]


def test_process_registry_singleton_installs_gets_and_closes(tmp_path):
    from sglang.srt.model_loader.welm_shared_embedding_weights import (
        close_welm_shared_embedding_registry,
        get_welm_shared_embedding_registry,
        install_welm_shared_embedding_registry,
    )

    checkpoint = _write_tiny_checkpoint(tmp_path / "checkpoint")
    plans = plan_welm_embedding_replicas(
        WeLMSharedEmbeddingPolicy.BIND, (0,), bind_node=0
    )
    create_welm_embedding_arena(
        checkpoint=checkpoint,
        root=tmp_path / "arena",
        plans=plans,
        numa_adapter=_FakeNumaAdapter(),
    )

    registry = install_welm_shared_embedding_registry(
        str(tmp_path / "arena" / "manifest.json"),
        gpu_id=0,
        gpu_numa_node=0,
        view_opener=_FakeSharedView.open,
    )
    assert get_welm_shared_embedding_registry() is registry
    with pytest.raises(RuntimeError, match="already installed"):
        install_welm_shared_embedding_registry(
            str(tmp_path / "arena" / "manifest.json"),
            gpu_id=0,
            gpu_numa_node=0,
            view_opener=_FakeSharedView.open,
        )

    close_welm_shared_embedding_registry()
    assert get_welm_shared_embedding_registry(required=False) is None
    close_welm_shared_embedding_registry()


def test_process_registry_close_failure_remains_retryable(monkeypatch):
    from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights

    class FlakyRegistry:
        def __init__(self):
            self.attempts = 0

        def close(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("unregister failed")

    registry = FlakyRegistry()
    monkeypatch.setattr(
        shared_weights, "_PROCESS_SHARED_EMBEDDING_REGISTRY", registry
    )

    with pytest.raises(RuntimeError, match="unregister failed"):
        shared_weights.close_welm_shared_embedding_registry()
    assert shared_weights.get_welm_shared_embedding_registry() is registry

    shared_weights.close_welm_shared_embedding_registry()
    assert registry.attempts == 2
    assert shared_weights.get_welm_shared_embedding_registry(required=False) is None
