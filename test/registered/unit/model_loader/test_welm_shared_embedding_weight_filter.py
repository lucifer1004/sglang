from types import MethodType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.srt.model_loader import weight_utils
from sglang.srt.model_loader.loader import DefaultModelLoader


SHARED_KEYS = (
    "model.embed_tokens.weight",
    "model.oe_embed.0.weight",
    "model.oe_embed.1.weight",
    "model.oe_embed.2.weight",
    "model.oe_embed.3.weight",
)


@pytest.mark.parametrize(
    "iterator_name",
    [
        "safetensors_weights_iterator",
        "multi_thread_safetensors_weights_iterator",
        "buffered_multi_thread_safetensors_weights_iterator",
    ],
)
def test_safetensors_iterators_filter_before_get_tensor(
    tmp_path, monkeypatch, iterator_name
):
    ordinary_key = "model.layers.0.input_layernorm.weight"
    path = tmp_path / "model.safetensors"
    save_file(
        {
            **{key: torch.ones((2, 2)) for key in SHARED_KEYS},
            ordinary_key: torch.arange(4, dtype=torch.float32),
        },
        path,
    )
    calls = []
    real_safe_open = weight_utils.safetensors.safe_open

    class SpySafeOpen:
        def __init__(self, *args, **kwargs):
            self._context = real_safe_open(*args, **kwargs)

        def __enter__(self):
            self._handle = self._context.__enter__()
            return self

        def __exit__(self, *args):
            return self._context.__exit__(*args)

        def keys(self):
            return self._handle.keys()

        def get_tensor(self, name):
            calls.append(name)
            return self._handle.get_tensor(name)

    monkeypatch.setattr(weight_utils.safetensors, "safe_open", SpySafeOpen)
    iterator = getattr(weight_utils, iterator_name)
    kwargs = {
        "hf_weights_files": [str(path)],
        "weight_name_filter": lambda name: name not in SHARED_KEYS,
    }
    if iterator_name != "safetensors_weights_iterator":
        kwargs["max_workers"] = 1

    yielded = dict(iterator(**kwargs))

    assert yielded.keys() == {ordinary_key}
    assert calls == [ordinary_key]


def test_default_loader_passes_external_name_filter_to_all_sources():
    loader = object.__new__(DefaultModelLoader)
    model_config = SimpleNamespace(model_path="checkpoint", revision=None)
    secondary = DefaultModelLoader.Source(
        model_or_path="secondary-checkpoint",
        revision=None,
        prefix="secondary.",
    )
    model = SimpleNamespace(
        secondary_weights=(secondary,),
        get_externally_owned_weight_names=lambda: frozenset(SHARED_KEYS),
    )
    calls = []

    def fake_get_weights_iterator(self, source, weight_name_filter=None):
        calls.append((source, weight_name_filter))
        return iter(())

    loader._get_weights_iterator = MethodType(fake_get_weights_iterator, loader)

    assert list(loader._get_all_weights(model_config, model)) == []
    assert len(calls) == 2
    for source, predicate in calls:
        assert predicate is not None
        assert predicate("model.layers.0.weight")
        assert not predicate(SHARED_KEYS[0])


def test_default_loader_preserves_behavior_without_external_name_hook():
    loader = object.__new__(DefaultModelLoader)
    model_config = SimpleNamespace(model_path="checkpoint", revision=None)
    model = SimpleNamespace(secondary_weights=())
    predicates = []

    def fake_get_weights_iterator(self, source, weight_name_filter=None):
        predicates.append(weight_name_filter)
        return iter(())

    loader._get_weights_iterator = MethodType(fake_get_weights_iterator, loader)

    assert list(loader._get_all_weights(model_config, model)) == []
    assert predicates == [None]


def test_welm_model_exposes_shared_names_only_when_policy_is_enabled(monkeypatch):
    from sglang.srt.models import welmv4

    model = object.__new__(welmv4.WeLMV4MoeForCausalLM)
    monkeypatch.setattr(
        welmv4,
        "get_global_server_args",
        lambda: SimpleNamespace(welm_shared_embedding_policy="interleave"),
    )
    assert model.get_externally_owned_weight_names() == frozenset(SHARED_KEYS)

    monkeypatch.setattr(
        welmv4,
        "get_global_server_args",
        lambda: SimpleNamespace(welm_shared_embedding_policy="disabled"),
    )
    assert model.get_externally_owned_weight_names() == frozenset()
