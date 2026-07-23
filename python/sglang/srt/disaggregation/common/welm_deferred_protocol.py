# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.models.welm_deferred_mirror import (
    DeferredDecodeInputKind,
    WelmDeferredMirrorPlan,
    WelmPDExecutionMode,
)

WELM_DEFERRED_COMPLETION_MAGIC = 0x57454C4D
WELM_DEFERRED_PROTOCOL_VERSION = 1
WELM_DEFERRED_COMPLETION_WORDS = 16
WELM_DEFERRED_COMMITTED_LENGTH_SEMANTICS = "prompt_without_last_token_v1"


class WelmDeferredCompletionKind(IntEnum):
    LAST_PROMPT_SEED = 1


@dataclass(frozen=True)
class WelmDeferredCompletion:
    committed_kv_len: int
    seed_position: int
    seed_token_id: int
    input_kind: DeferredDecodeInputKind = DeferredDecodeInputKind.TOKEN_ID


def encode_welm_deferred_completion(
    completion: WelmDeferredCompletion,
) -> npt.NDArray[np.int32]:
    values = np.zeros(WELM_DEFERRED_COMPLETION_WORDS, dtype=np.int32)
    values[:8] = (
        WELM_DEFERRED_COMPLETION_MAGIC,
        WELM_DEFERRED_PROTOCOL_VERSION,
        int(WelmDeferredCompletionKind.LAST_PROMPT_SEED),
        completion.committed_kv_len,
        completion.seed_position,
        completion.seed_token_id,
        int(completion.input_kind),
        0,
    )
    decode_welm_deferred_completion(values)
    return values


def decode_welm_deferred_completion(
    values: npt.NDArray[np.int32],
) -> WelmDeferredCompletion:
    if not isinstance(values, np.ndarray):
        raise ValueError("WeLM deferred completion record must be a numpy array")
    if values.dtype != np.dtype(np.int32):
        raise ValueError(
            "WeLM deferred completion record must use int32, "
            f"got {values.dtype}"
        )
    if values.shape != (WELM_DEFERRED_COMPLETION_WORDS,):
        raise ValueError(
            "WeLM deferred completion record must contain exactly "
            f"{WELM_DEFERRED_COMPLETION_WORDS} int32 values, got {values.shape}"
        )

    words = [int(value) for value in values]
    if words[0] != WELM_DEFERRED_COMPLETION_MAGIC:
        raise ValueError(
            "invalid WeLM deferred completion magic: "
            f"expected {WELM_DEFERRED_COMPLETION_MAGIC:#x}, got {words[0]:#x}"
        )
    if words[1] != WELM_DEFERRED_PROTOCOL_VERSION:
        raise ValueError(
            "unsupported WeLM deferred completion version: "
            f"expected {WELM_DEFERRED_PROTOCOL_VERSION}, got {words[1]}"
        )
    if words[2] != int(WelmDeferredCompletionKind.LAST_PROMPT_SEED):
        raise ValueError(f"unsupported WeLM deferred completion kind: {words[2]}")

    committed_kv_len = words[3]
    seed_position = words[4]
    seed_token_id = words[5]
    if committed_kv_len < 0:
        raise ValueError("committed_kv_len must be non-negative")
    if seed_position < 0:
        raise ValueError("seed_position must be non-negative")
    if seed_position != committed_kv_len:
        raise ValueError(
            "seed_position must equal committed_kv_len; "
            f"got seed_position={seed_position}, committed_kv_len={committed_kv_len}"
        )
    if seed_token_id < 0:
        raise ValueError("seed_token_id must be non-negative")

    try:
        input_kind = DeferredDecodeInputKind(words[6])
    except ValueError as exc:
        raise ValueError(f"unsupported deferred input kind: {words[6]}") from exc
    if input_kind is not DeferredDecodeInputKind.TOKEN_ID:
        raise ValueError(
            "this deferred protocol version only supports TOKEN_ID input"
        )
    if words[7] != 0:
        raise ValueError(f"deferred completion flags must be zero, got {words[7]}")
    if any(words[8:]):
        raise ValueError("deferred completion reserved fields must be zero")

    return WelmDeferredCompletion(
        committed_kv_len=committed_kv_len,
        seed_position=seed_position,
        seed_token_id=seed_token_id,
        input_kind=input_kind,
    )


@dataclass(frozen=True)
class WelmDeferredMirrorCapability:
    mode: str
    protocol_version: int
    model_identity: str
    mirror_fingerprint: str
    execution_end_layer: int
    committed_length_semantics: str

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "WelmDeferredMirrorCapability":
        if not isinstance(payload, Mapping):
            raise ValueError("WeLM deferred capability must be a JSON object")
        expected_fields = {
            "mode",
            "protocol_version",
            "model_identity",
            "mirror_fingerprint",
            "execution_end_layer",
            "committed_length_semantics",
        }
        actual_fields = set(payload)
        if actual_fields != expected_fields:
            raise ValueError(
                "invalid WeLM deferred capability fields: "
                f"expected={sorted(expected_fields)}, got={sorted(actual_fields)}"
            )

        mode = _require_str(payload, "mode")
        protocol_version = _require_int(payload, "protocol_version")
        model_identity = _require_str(payload, "model_identity")
        mirror_fingerprint = _require_str(payload, "mirror_fingerprint")
        execution_end_layer = _require_int(payload, "execution_end_layer")
        committed_length_semantics = _require_str(
            payload, "committed_length_semantics"
        )
        if mode != WelmPDExecutionMode.DEFERRED_LAST_PROMPT.value:
            raise ValueError(f"unsupported WeLM deferred capability mode {mode!r}")
        if protocol_version != WELM_DEFERRED_PROTOCOL_VERSION:
            raise ValueError(
                "unsupported WeLM deferred capability version "
                f"{protocol_version}; expected {WELM_DEFERRED_PROTOCOL_VERSION}"
            )
        if not model_identity:
            raise ValueError("WeLM deferred model_identity must not be empty")
        if not mirror_fingerprint:
            raise ValueError("WeLM deferred mirror_fingerprint must not be empty")
        if execution_end_layer <= 0:
            raise ValueError("WeLM deferred execution_end_layer must be positive")
        if (
            committed_length_semantics
            != WELM_DEFERRED_COMMITTED_LENGTH_SEMANTICS
        ):
            raise ValueError(
                "unsupported WeLM deferred committed-length semantics "
                f"{committed_length_semantics!r}"
            )
        return cls(
            mode=mode,
            protocol_version=protocol_version,
            model_identity=model_identity,
            mirror_fingerprint=mirror_fingerprint,
            execution_end_layer=execution_end_layer,
            committed_length_semantics=committed_length_semantics,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "protocol_version": self.protocol_version,
            "model_identity": self.model_identity,
            "mirror_fingerprint": self.mirror_fingerprint,
            "execution_end_layer": self.execution_end_layer,
            "committed_length_semantics": self.committed_length_semantics,
        }


def _require_str(payload: Mapping[str, Any], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str):
        raise ValueError(f"WeLM deferred capability field {name!r} must be a string")
    return value


def _require_int(payload: Mapping[str, Any], name: str) -> int:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"WeLM deferred capability field {name!r} must be an integer")
    return value


def build_welm_deferred_mirror_capability(
    *,
    model_identity: str,
    plan: WelmDeferredMirrorPlan,
) -> WelmDeferredMirrorCapability:
    return WelmDeferredMirrorCapability.from_wire(
        {
            "mode": WelmPDExecutionMode.DEFERRED_LAST_PROMPT.value,
            "protocol_version": WELM_DEFERRED_PROTOCOL_VERSION,
            "model_identity": str(model_identity),
            "mirror_fingerprint": plan.fingerprint,
            "execution_end_layer": plan.execution_end_layer,
            "committed_length_semantics": WELM_DEFERRED_COMMITTED_LENGTH_SEMANTICS,
        }
    )


def resolve_runtime_welm_deferred_mirror_capability(
    server_args: Any,
    model_config: Any,
    plan: Optional[WelmDeferredMirrorPlan],
) -> Optional[WelmDeferredMirrorCapability]:
    raw_mode = getattr(server_args, "welm_kv_mirror_pd_mode", "legacy")
    try:
        mode = WelmPDExecutionMode(raw_mode)
    except ValueError as exc:
        raise RuntimeError(
            f"unknown runtime WeLM mirror P/D mode {raw_mode!r}"
        ) from exc

    if mode is WelmPDExecutionMode.LEGACY:
        if plan is not None:
            raise RuntimeError(
                "legacy WeLM mirror P/D mode unexpectedly received an execution plan"
            )
        return None

    if plan is None:
        raise RuntimeError(
            "deferred WeLM mirror P/D mode is missing execution plan propagation"
        )
    model_identity = getattr(model_config, "model_path", None)
    if not model_identity:
        raise RuntimeError("deferred WeLM mirror P/D mode is missing model identity")
    return build_welm_deferred_mirror_capability(
        model_identity=str(model_identity),
        plan=plan,
    )


def validate_welm_deferred_mirror_capabilities(
    local: Optional[WelmDeferredMirrorCapability],
    remote: Optional[WelmDeferredMirrorCapability],
) -> None:
    if local == remote:
        return

    local_wire = local.to_wire() if local is not None else None
    remote_wire = remote.to_wire() if remote is not None else None
    raise RuntimeError(
        "WeLM deferred P/D capability mismatch: "
        f"local={json.dumps(local_wire, sort_keys=True)} "
        f"remote={json.dumps(remote_wire, sort_keys=True)}"
    )
