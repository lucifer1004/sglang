from types import SimpleNamespace

import pytest

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.model_loader import welm_shared_embedding_weights as shared_weights


class _PipeWriter:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


@pytest.mark.parametrize("pp_rank", [0, 1])
def test_scheduler_registry_lifecycle_is_scoped_to_first_pp_stage(
    monkeypatch, pp_rank
):
    events = []
    server_args = SimpleNamespace(
        welm_shared_embedding_manifest_path="/tmp/manifest.json",
        numa_node=[0],
        enable_trace=False,
    )

    monkeypatch.setattr(scheduler_module, "load_plugins", lambda: None)
    monkeypatch.setattr(
        scheduler_module,
        "configure_scheduler_process",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        scheduler_module.psutil,
        "Process",
        lambda: SimpleNamespace(parent=lambda: SimpleNamespace(send_signal=lambda *_: None)),
    )
    monkeypatch.setattr(
        shared_weights,
        "install_welm_shared_embedding_registry",
        lambda *args, **kwargs: events.append("install-registry"),
    )
    monkeypatch.setattr(
        shared_weights,
        "close_welm_shared_embedding_registry",
        lambda: events.append("close-registry"),
    )
    monkeypatch.setattr(scheduler_module.gc, "collect", lambda: events.append("gc"))

    class FakeScheduler:
        def __init__(self, *args, **kwargs):
            events.append("scheduler-init")

        def get_init_info(self):
            return {"ready": True}

        def run_event_loop(self):
            events.append("event-loop")

        def _shutdown_fpm(self):
            events.append("shutdown-fpm")

        def __del__(self):
            events.append("scheduler-del")

    monkeypatch.setattr(scheduler_module, "Scheduler", FakeScheduler)
    pipe_writer = _PipeWriter()

    scheduler_module.run_scheduler_process(
        server_args=server_args,
        port_args=SimpleNamespace(),
        gpu_id=0,
        tp_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        moe_ep_rank=0,
        pp_rank=pp_rank,
        dp_rank=None,
        pipe_writer=pipe_writer,
    )

    if pp_rank == 0:
        assert events == [
            "install-registry",
            "scheduler-init",
            "event-loop",
            "shutdown-fpm",
            "scheduler-del",
            "gc",
            "close-registry",
        ]
    else:
        assert events == [
            "scheduler-init",
            "event-loop",
            "shutdown-fpm",
            "scheduler-del",
        ]
    assert pipe_writer.messages == [{"ready": True}]
