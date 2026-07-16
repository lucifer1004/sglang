from types import MethodType, SimpleNamespace

import torch

from sglang.srt.model_executor.model_runner import ModelRunner


SHARED_KEY = "model.embed_tokens.weight"
ERROR = (
    "shared WeLM host embedding weights are immutable for the process lifetime "
    "and require a service restart"
)


class _FakeModel:
    def __init__(self):
        self.load_calls = []

    def load_weights(self, weights):
        self.load_calls.append(list(weights))


def _runner():
    runner = object.__new__(ModelRunner)
    runner.server_args = SimpleNamespace(
        welm_shared_embedding_manifest_path="/tmp/manifest.json",
        custom_weight_loader=[],
    )
    runner.model = _FakeModel()
    return runner


def test_disk_whole_model_update_is_rejected_before_state_changes():
    runner = _runner()

    success, message = runner.update_weights_from_disk("new-model", "safetensors")

    assert not success
    assert ERROR in message
    assert runner.model.load_calls == []


def test_named_tensor_update_rejects_batch_before_first_mutation():
    runner = _runner()
    named_tensors = [
        ("model.layers.0.weight", torch.ones(1)),
        (SHARED_KEY, torch.ones(1)),
    ]

    success, message = runner.update_weights_from_tensor(named_tensors)

    assert not success
    assert ERROR in message
    assert runner.model.load_calls == []


def test_distributed_update_rejects_before_group_or_broadcast_access():
    runner = _runner()

    success, message = runner.update_weights_from_distributed(
        names=[SHARED_KEY],
        dtypes=["bfloat16"],
        shapes=[(1,)],
        group_name="missing-group",
    )

    assert not success
    assert ERROR in message


def test_split_distributed_receive_rejects_before_group_or_broadcast_access():
    runner = _runner()

    success, message, weights = runner.receive_weights_from_distributed(
        names=[SHARED_KEY],
        dtypes=["bfloat16"],
        shapes=[(1,)],
        group_name="missing-group",
    )

    assert not success
    assert ERROR in message
    assert weights == []


def test_flattened_bucket_rejects_metadata_before_reconstruction():
    runner = _runner()
    bucket = {
        "flattened_tensor": torch.ones(1),
        "metadata": [SimpleNamespace(name=SHARED_KEY)],
    }

    success, message = runner._update_weights_from_flattened_bucket(bucket)

    assert not success
    assert ERROR in message
    assert runner.model.load_calls == []


def test_opaque_ipc_update_is_rejected_while_shared_arena_is_active():
    runner = _runner()

    success, message = runner.update_weights_from_ipc(
        SimpleNamespace(zmq_handles=[object()])
    )

    assert not success
    assert ERROR in message


def test_non_shared_named_update_has_no_preflight_rejection():
    runner = _runner()

    assert (
        runner._get_shared_welm_embedding_update_rejection(
            ["model.layers.0.weight"]
        )
        is None
    )


def test_weight_checker_reset_is_rejected_before_checker_mutation():
    runner = _runner()
    calls = []
    runner._weight_checker = SimpleNamespace(
        handle=lambda action: calls.append(action)
    )

    try:
        runner.check_weights("reset_tensors")
    except RuntimeError as exc:
        assert ERROR in str(exc)
    else:
        raise AssertionError("reset_tensors must be rejected")
    assert calls == []


def test_filtered_disk_update_passes_filter_before_materialization(monkeypatch):
    from sglang.srt.model_executor import model_runner as model_runner_module
    from sglang.srt.model_loader.loader import DefaultModelLoader

    runner = _runner()
    runner.device = "cpu"
    runner.gpu_id = 0
    runner.model_config = SimpleNamespace(
        model_path="old-model",
        revision=None,
        dtype=torch.float32,
    )
    runner.server_args.model_path = "old-model"
    runner.server_args.load_format = "safetensors"
    loader = object.__new__(DefaultModelLoader)
    captured = []

    def fake_iterator(self, source, weight_name_filter=None):
        captured.append(weight_name_filter)
        assert weight_name_filter("model.layers.0.weight")
        assert not weight_name_filter(SHARED_KEY)
        return iter([("model.layers.0.weight", torch.ones(1))])

    def fake_load(self, model, weights, target_device):
        model.load_calls.append(list(weights))

    loader._get_weights_iterator = MethodType(fake_iterator, loader)
    loader.load_weights_and_postprocess = MethodType(fake_load, loader)
    monkeypatch.setattr(
        model_runner_module, "get_model_loader", lambda *_args, **_kwargs: loader
    )
    monkeypatch.setattr(
        model_runner_module, "get_available_gpu_memory", lambda *_args, **_kwargs: 0
    )

    success, _message = runner.update_weights_from_disk(
        "new-model",
        "safetensors",
        weight_name_filter=lambda name: name.startswith("model.layers"),
    )

    assert success
    assert len(captured) == 1
    assert len(runner.model.load_calls) == 1
    assert runner.model.load_calls[0][0][0] == "model.layers.0.weight"
    torch.testing.assert_close(runner.model.load_calls[0][0][1], torch.ones(1))
