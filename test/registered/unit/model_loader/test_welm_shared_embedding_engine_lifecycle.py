import re
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints import engine as engine_module
from sglang.srt.managers import tokenizer_manager as tokenizer_manager_module


class _FakeArenaHandle:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_sglang_run_id_is_safe_for_shared_embedding_arena_paths():
    run_id = engine_module._generate_sglang_run_id()

    assert re.fullmatch(r"[A-Za-z0-9_-]+", run_id)


def _launch_result():
    return (
        None,
        None,
        None,
        engine_module.SchedulerInitResult(scheduler_infos=[]),
        None,
    )


def test_launch_wrapper_transfers_arena_handle_to_scheduler_result(monkeypatch):
    server_args = SimpleNamespace()
    handle = _FakeArenaHandle()

    def launch_impl(cls, **kwargs):
        engine_module._WELM_ARENA_STARTUP_HANDLES[id(server_args)] = handle
        return _launch_result()

    monkeypatch.setattr(
        engine_module.Engine,
        "_launch_subprocesses_impl",
        classmethod(launch_impl),
    )
    result = engine_module.Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=lambda *_args: None,
        run_scheduler_process_func=lambda *_args: None,
        run_detokenizer_process_func=lambda *_args: None,
    )

    assert result[3].welm_embedding_arena_handle is handle
    assert engine_module._WELM_ARENA_STARTUP_HANDLES[id(server_args)] is handle
    engine_module._forget_welm_arena_handle(handle)


def test_arena_exit_cleanup_stops_children_before_removing_files(monkeypatch):
    handle = _FakeArenaHandle()
    events = []
    monkeypatch.setattr(
        engine_module,
        "kill_process_tree",
        lambda *args, **kwargs: events.append("kill-children"),
    )
    original_close = handle.close

    def record_close():
        events.append("close-arena")
        original_close()

    handle.close = record_close

    engine_module._shutdown_welm_arena_handle(handle)

    assert events == ["kill-children", "close-arena"]


def test_arena_exit_cleanup_closes_handle_when_child_kill_raises(monkeypatch):
    handle = _FakeArenaHandle()
    monkeypatch.setattr(
        engine_module,
        "kill_process_tree",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kill failed")),
    )

    with pytest.raises(RuntimeError, match="kill failed"):
        engine_module._shutdown_welm_arena_handle(handle)

    assert handle.closed


def test_launch_wrapper_stops_children_before_cleaning_arena_on_failure(
    monkeypatch,
):
    server_args = SimpleNamespace()
    handle = _FakeArenaHandle()
    events = []

    def launch_impl(cls, **kwargs):
        engine_module._WELM_ARENA_STARTUP_HANDLES[id(server_args)] = handle
        raise RuntimeError("scheduler startup failed")

    monkeypatch.setattr(
        engine_module.Engine,
        "_launch_subprocesses_impl",
        classmethod(launch_impl),
    )
    monkeypatch.setattr(
        engine_module,
        "kill_process_tree",
        lambda *args, **kwargs: events.append("kill-children"),
    )

    original_close = handle.close

    def record_close():
        events.append("close-arena")
        original_close()

    handle.close = record_close

    with pytest.raises(RuntimeError, match="scheduler startup failed"):
        engine_module.Engine._launch_subprocesses(
            server_args=server_args,
            init_tokenizer_manager_func=lambda *_args: None,
            run_scheduler_process_func=lambda *_args: None,
            run_detokenizer_process_func=lambda *_args: None,
        )

    assert events == ["kill-children", "close-arena"]
    assert handle.closed
    assert id(server_args) not in engine_module._WELM_ARENA_STARTUP_HANDLES


def test_running_sigquit_cleans_arena_after_children(monkeypatch):
    events = []
    handle = _FakeArenaHandle()
    original_close = handle.close

    def record_close():
        events.append("close-arena")
        original_close()

    handle.close = record_close
    tokenizer_manager = SimpleNamespace(
        _subprocess_watchdog=SimpleNamespace(
            stop=lambda: events.append("stop-watchdog")
        ),
        _welm_embedding_arena_handle=handle,
        dump_requests_before_crash=lambda: events.append("dump-requests"),
    )
    monkeypatch.setattr(
        tokenizer_manager_module,
        "kill_process_tree",
        lambda *args, **kwargs: events.append("kill-children"),
    )

    with pytest.raises(SystemExit):
        tokenizer_manager_module.SignalHandler(
            tokenizer_manager
        ).running_phase_sigquit_handler()

    assert events == [
        "stop-watchdog",
        "dump-requests",
        "kill-children",
        "close-arena",
    ]
    assert tokenizer_manager._welm_embedding_arena_handle is None
