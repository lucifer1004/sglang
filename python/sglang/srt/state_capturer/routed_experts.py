from typing import Any, Optional

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_attention_dp_rank,
    get_attention_tp_size,
    get_dp_local_slice_cpu,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.state_capturer.base import BaseTopkCapturer, TopkCaptureOutput


class RoutedExpertsCapturer(BaseTopkCapturer):
    """Capturer for router replay expert ids.

    The host cache mirrors KV slots: every forward writes a dense
    [tokens, layers, topk] slice indexed by forward_batch.out_cache_loc. There is
    no per-forward clear and no validity mask; stale rows are unreachable unless
    a live request still references the same KV slots.
    """

    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ) -> Optional["RoutedExpertsCapturer"]:
        if not enable:
            return None
        return RoutedExpertsCapturer(
            model_config,
            num_tokens=num_tokens,
            max_running_requests=max_running_requests,
            num_fused_shared_experts=num_fused_shared_experts,
            device=device,
        )

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        topk_size = model_config.hf_text_config.num_experts_per_tok
        num_layers = model_config.hf_text_config.num_hidden_layers

        server_args = get_global_server_args()
        max_batch_size = max(
            server_args.chunked_prefill_size * server_args.dp_size,
            max_running_requests * server_args.dp_size,
        )

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            topk_size=topk_size,
            device=device,
            name="routed_experts",
            device_topk_size=topk_size + num_fused_shared_experts,
        )
        self.host_cache.buffer.fill_(-1)

        if get_moe_a2a_backend().is_deepep():
            attn_tp_size = get_attention_tp_size() if is_dp_attention_enabled() else 1
            self.gather_buffer = torch.empty(
                (
                    self.device_cache.buffer.shape[0] * attn_tp_size,
                    self.device_cache.buffer.shape[2],
                ),
                dtype=torch.int32,
                device=device,
            )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        if get_moe_a2a_backend().is_deepep():
            local_topk = topk_indices
            topk_indices = self.gather_buffer[
                : local_topk.size(0) * get_attention_tp_size()
            ]
            attn_tp_all_gather_into_tensor(topk_indices, local_topk)
        super().capture(layer_id, topk_indices)

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        if is_dp_attention_enabled() and not get_moe_a2a_backend().is_deepep():
            local_start_pos, local_num_tokens = get_dp_local_slice_cpu(
                forward_batch, can_run_graph, cuda_graph_batch
            )
            local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos, local_end_pos = 0, forward_batch.out_cache_loc.shape[0]
        return self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.topk_size
        ]

    def _get_local_out_cache_loc_and_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ):
        if not is_dp_attention_enabled():
            out_cache_loc = forward_batch.out_cache_loc
            return out_cache_loc, 0, out_cache_loc.shape[0]

        if get_moe_a2a_backend().is_deepep():
            out_cache_loc = forward_batch.out_cache_loc
            return out_cache_loc, 0, out_cache_loc.shape[0]

        local_start_pos, local_num_tokens = get_dp_local_slice_cpu(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        dp_rank = get_attention_dp_rank()
        out_cache_local_start_pos = sum(forward_batch.global_num_tokens_cpu[:dp_rank])
        out_cache_local_end_pos = out_cache_local_start_pos + local_num_tokens
        out_cache_loc = forward_batch.out_cache_loc[
            out_cache_local_start_pos:out_cache_local_end_pos
        ]
        local_end_pos = local_start_pos + local_num_tokens
        return out_cache_loc, local_start_pos, local_end_pos

    def _dense_forward_capture_state(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ):
        out_cache_loc, local_start_pos, local_end_pos = (
            self._get_local_out_cache_loc_and_slice(
                forward_batch, can_run_graph, cuda_graph_batch
            )
        )
        routed_experts = self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.topk_size
        ]
        if routed_experts.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                "Unable to align routed_experts rows to KV cache locations: "
                f"routed_shape={tuple(routed_experts.shape)} "
                f"out_cache_loc={tuple(out_cache_loc.shape)}."
            )
        return out_cache_loc, routed_experts

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[TopkCaptureOutput]:
        out_cache_loc, routed_experts = self._dense_forward_capture_state(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        if no_copy_to_cpu:
            return TopkCaptureOutput(
                out_cache_loc=out_cache_loc,
                topk=routed_experts,
                host_cache=self.host_cache,
            )
        self.host_cache.buffer[out_cache_loc.cpu()] = routed_experts.cpu()
        return None

    def get_topk(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        start_len: int = 0,
    ):
        if start_len < 0:
            raise ValueError(f"{start_len=} must be non-negative")
        end_len = max(0, seqlen - 1)
        start_len = min(start_len, end_len)
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][start_len:end_len]
            .cpu()
            .clone()
        )
        routed_experts = torch.full(
            (*cache_pool_idx.shape, self.host_cache.buffer.shape[1], self.topk_size),
            -1,
            dtype=self.host_cache.buffer.dtype,
            device=self.host_cache.buffer.device,
        )
        valid_idx = (cache_pool_idx >= 0) & (
            cache_pool_idx < self.host_cache.buffer.shape[0]
        )
        if bool(valid_idx.any()):
            routed_experts[valid_idx] = self.host_cache.buffer[cache_pool_idx[valid_idx]]
        return {
            "schema_version": 3,
            "format": "dense",
            "values": routed_experts,
            "shape": list(routed_experts.shape),
            "missing_value": -1,
            "invalid_cache_locs": int((~valid_idx).sum().item()),
        }


_global_expert_capturer: Optional[RoutedExpertsCapturer] = None


def get_global_experts_capturer() -> Optional[RoutedExpertsCapturer]:
    return _global_expert_capturer


def set_global_experts_capturer(capturer: Optional[RoutedExpertsCapturer]):
    global _global_expert_capturer
    _global_expert_capturer = capturer


def _b64_encode_tensor(tensor: torch.Tensor) -> str:
    return pybase64.b64encode(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).decode("utf-8")


def encode_routed_experts_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _b64_encode_tensor(value)
    if isinstance(value, (list, tuple)):
        return [encode_routed_experts_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            key: encode_routed_experts_payload(item)
            if isinstance(item, (dict, list, tuple, torch.Tensor))
            else item
            for key, item in value.items()
        }
    return value


def _decode_int32_payload(value: Any):
    if isinstance(value, str):
        return np.frombuffer(pybase64.b64decode(value.encode("utf-8")), dtype=np.int32)
    return value


def extract_routed_experts_from_meta_info(data):
    routed_experts_payload = data["meta_info"].get("routed_experts", None)
    if isinstance(routed_experts_payload, dict):
        payload = dict(routed_experts_payload)
        if "values" in payload:
            payload["values"] = _decode_int32_payload(payload["values"])
        if payload.get("format") == "dense":
            values = payload["values"]
            if isinstance(values, np.ndarray) and payload.get("shape") is not None:
                return values.reshape(payload["shape"])
            return values
        return payload

    return np.frombuffer(
        pybase64.b64decode(routed_experts_payload.encode("utf-8")), dtype=np.int32
    )


def disable_routed_experts_capture_for_draft(model: Any) -> None:
    """Opt every draft MoE ``TopK`` out of routed-experts capture."""
    from sglang.srt.layers.moe.topk import TopK

    for module in model.modules():
        if isinstance(module, TopK):
            module.topk_config.allow_routed_experts_capture = False
