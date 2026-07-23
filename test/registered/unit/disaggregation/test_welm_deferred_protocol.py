from dataclasses import replace

import numpy as np
import pytest

from sglang.srt.disaggregation.common.welm_deferred_protocol import (
    WELM_DEFERRED_COMMITTED_LENGTH_SEMANTICS,
    WELM_DEFERRED_COMPLETION_MAGIC,
    WELM_DEFERRED_PROTOCOL_VERSION,
    WelmDeferredCompletion,
    WelmDeferredMirrorCapability,
    build_welm_deferred_mirror_capability,
    decode_welm_deferred_completion,
    encode_welm_deferred_completion,
    resolve_runtime_welm_deferred_mirror_capability,
    validate_welm_deferred_mirror_capabilities,
)
from sglang.srt.models.welm_deferred_mirror import (
    DeferredDecodeInputKind,
    WelmDeferredMirrorPair,
    WelmDeferredMirrorPlan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _plan() -> WelmDeferredMirrorPlan:
    return WelmDeferredMirrorPlan(
        num_hidden_layers=48,
        execution_end_layer=33,
        pairs=(WelmDeferredMirrorPair(source_layer=15, target_layer=33),),
        fingerprint="mirror-fingerprint",
    )


def _capability() -> WelmDeferredMirrorCapability:
    return build_welm_deferred_mirror_capability(
        model_identity="/models/welm/checkpoint-3972",
        plan=_plan(),
    )


def test_completion_record_has_fixed_int32_wire_layout():
    completion = WelmDeferredCompletion(
        committed_kv_len=31,
        seed_position=31,
        seed_token_id=104857,
    )

    encoded = encode_welm_deferred_completion(completion)

    assert encoded.dtype == np.int32
    assert encoded.shape == (16,)
    assert encoded.tolist() == [
        WELM_DEFERRED_COMPLETION_MAGIC,
        WELM_DEFERRED_PROTOCOL_VERSION,
        1,
        31,
        31,
        104857,
        int(DeferredDecodeInputKind.TOKEN_ID),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert decode_welm_deferred_completion(encoded) == completion


@pytest.mark.parametrize(
    ("index", "value", "message"),
    [
        (0, 0, "magic"),
        (1, 2, "version"),
        (2, 2, "completion kind"),
        (3, -1, "committed_kv_len"),
        (4, 30, "seed_position.*committed_kv_len"),
        (5, -1, "seed_token_id"),
        (6, int(DeferredDecodeInputKind.EMBEDDING), "TOKEN_ID"),
        (7, 1, "flags"),
        (8, 1, "reserved"),
    ],
)
def test_completion_record_rejects_malformed_fields(index, value, message):
    encoded = encode_welm_deferred_completion(
        WelmDeferredCompletion(
            committed_kv_len=31,
            seed_position=31,
            seed_token_id=7,
        )
    )
    encoded[index] = value

    with pytest.raises(ValueError, match=message):
        decode_welm_deferred_completion(encoded)


def test_completion_record_rejects_wrong_shape_and_dtype():
    with pytest.raises(ValueError, match="16"):
        decode_welm_deferred_completion(np.zeros(15, dtype=np.int32))
    with pytest.raises(ValueError, match="int32"):
        decode_welm_deferred_completion(np.zeros(16, dtype=np.int64))


def test_capability_roundtrip_is_nested_and_strict():
    capability = _capability()

    wire = capability.to_wire()

    assert wire == {
        "mode": "deferred-last-prompt",
        "protocol_version": 1,
        "model_identity": "/models/welm/checkpoint-3972",
        "mirror_fingerprint": "mirror-fingerprint",
        "execution_end_layer": 33,
        "committed_length_semantics": WELM_DEFERRED_COMMITTED_LENGTH_SEMANTICS,
    }
    assert WelmDeferredMirrorCapability.from_wire(wire) == capability

    malformed = dict(wire)
    malformed["unknown"] = 1
    with pytest.raises(ValueError, match="fields"):
        WelmDeferredMirrorCapability.from_wire(malformed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "legacy"),
        ("protocol_version", 2),
        ("model_identity", "/models/other"),
        ("mirror_fingerprint", "other-fingerprint"),
        ("execution_end_layer", 34),
        ("committed_length_semantics", "full-prompt"),
    ],
)
def test_peer_capability_mismatch_prints_both_sides(field, value):
    local = _capability()
    remote = replace(local, **{field: value})

    with pytest.raises(RuntimeError, match="local=.*remote="):
        validate_welm_deferred_mirror_capabilities(local, remote)


def test_peer_capability_requires_both_legacy_or_both_deferred():
    validate_welm_deferred_mirror_capabilities(None, None)
    validate_welm_deferred_mirror_capabilities(_capability(), _capability())

    with pytest.raises(RuntimeError, match="local=.*remote="):
        validate_welm_deferred_mirror_capabilities(_capability(), None)
    with pytest.raises(RuntimeError, match="local=.*remote="):
        validate_welm_deferred_mirror_capabilities(None, _capability())


def test_runtime_capability_requires_mode_and_plan_to_agree():
    deferred_args = type(
        "Args",
        (),
        {"welm_kv_mirror_pd_mode": "deferred-last-prompt"},
    )()
    legacy_args = type("Args", (), {"welm_kv_mirror_pd_mode": "legacy"})()
    model_config = type("ModelConfig", (), {"model_path": "/models/welm"})()

    with pytest.raises(RuntimeError, match="missing execution plan"):
        resolve_runtime_welm_deferred_mirror_capability(
            deferred_args,
            model_config,
            None,
        )
    with pytest.raises(RuntimeError, match="legacy.*execution plan"):
        resolve_runtime_welm_deferred_mirror_capability(
            legacy_args,
            model_config,
            _plan(),
        )

    assert (
        resolve_runtime_welm_deferred_mirror_capability(
            legacy_args,
            model_config,
            None,
        )
        is None
    )
    assert resolve_runtime_welm_deferred_mirror_capability(
        deferred_args,
        model_config,
        _plan(),
    ) == build_welm_deferred_mirror_capability(
        model_identity="/models/welm",
        plan=_plan(),
    )
