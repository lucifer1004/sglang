from types import SimpleNamespace

import pytest

from sglang.srt.models.welmv4 import (
    _welm_nextn_hf_to_local_name,
    welm_nextn_local_to_hf_name,
)


@pytest.mark.parametrize(
    ("hf_name", "local_name", "is_projector"),
    [
        (
            "model.layers.48.self_attn.q_proj.weight",
            "model.decoder_layers.0.self_attn.q_proj.weight",
            False,
        ),
        (
            "model.layers.50.shared_head.norm.weight",
            "model.projectors.2.ln_f.weight",
            True,
        ),
        (
            "model.layers.49.eh_proj.weight",
            "model.projectors.1.eh_proj.weight",
            True,
        ),
    ],
)
def test_welm_nextn_weight_names_round_trip(hf_name, local_name, is_projector):
    config = SimpleNamespace(num_hidden_layers=48, num_target_hidden_layers=48)

    mapped_name, mapped_is_projector = _welm_nextn_hf_to_local_name(hf_name, 48)

    assert (mapped_name, mapped_is_projector) == (local_name, is_projector)
    assert welm_nextn_local_to_hf_name(config, local_name) == hf_name


def test_welm_nextn_local_name_rejects_non_local_surface():
    config = SimpleNamespace(num_hidden_layers=48, num_target_hidden_layers=48)

    assert welm_nextn_local_to_hf_name(config, "model.layers.7.self_attn.k_proj.weight") is None
    assert welm_nextn_local_to_hf_name(config, "model.embed_tokens.weight") is None
