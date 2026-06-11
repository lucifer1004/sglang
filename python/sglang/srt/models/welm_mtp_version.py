from enum import Enum
from typing import Optional

from transformers import PretrainedConfig


class WelmMTPVersion(Enum):
    V1 = "v1"
    V2 = "v2"


_NUM_NEXTN_CACHE_ATTR = "_sglang_welm_mtp_num_nextn_predict_layers"
_VERSION_CACHE_ATTR = "_sglang_welm_mtp_version"


def get_welm_mtp_num_nextn_predict_layers(config: PretrainedConfig) -> int:
    cached = getattr(config, _NUM_NEXTN_CACHE_ATTR, None)
    if cached is not None:
        return cached
    if not hasattr(config, "num_nextn_predict_layers"):
        raise ValueError("WeLM MTP config is missing num_nextn_predict_layers.")
    num_layers = int(config.num_nextn_predict_layers)
    if num_layers <= 0:
        raise ValueError(
            "WeLM MTP config requires a positive num_nextn_predict_layers, "
            f"got {num_layers}."
        )
    setattr(config, _NUM_NEXTN_CACHE_ATTR, num_layers)
    return num_layers


def get_welm_mtp_version(
    config: PretrainedConfig, draft_steps: Optional[int] = None
) -> WelmMTPVersion:
    cached = getattr(config, _VERSION_CACHE_ATTR, None)
    if cached is not None:
        if draft_steps is not None:
            num_layers = get_welm_mtp_num_nextn_predict_layers(config)
            if num_layers != 1 and num_layers != int(draft_steps):
                raise ValueError(
                    "WeLM MTP config num_nextn_predict_layers must be 1 or match "
                    "speculative_num_steps, got "
                    f"num_nextn_predict_layers={num_layers}, draft_steps={draft_steps}."
                )
        return cached
    num_layers = get_welm_mtp_num_nextn_predict_layers(config)
    if num_layers == 1:
        version = WelmMTPVersion.V1
    else:
        if draft_steps is not None and num_layers != int(draft_steps):
            raise ValueError(
                "WeLM MTP config num_nextn_predict_layers must be 1 or match "
                "speculative_num_steps, got "
                f"num_nextn_predict_layers={num_layers}, draft_steps={draft_steps}."
            )
        version = WelmMTPVersion.V2
    setattr(config, _VERSION_CACHE_ATTR, version)
    return version
