import dataclasses
from typing import Any, Optional, Tuple

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_attention_dp_rank,
    get_attention_tp_size,
    get_dp_local_info,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_context import get_current_forward_batch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.state_capturer.base import get_tensor_size_bytes


@dataclasses.dataclass
class RoutedExpertsCaptureOutput:
    """Pending async D2H state for routed experts capture."""

    clear_cache_locs: torch.Tensor
    cache_locs: torch.Tensor
    routed_experts: torch.Tensor
    valid_mask: torch.Tensor
    host_cache: "_RoutedExpertsHostCache"

    def copy_to_cpu(self):
        self.clear_cache_locs = self.clear_cache_locs.to("cpu", non_blocking=True)
        self.cache_locs = self.cache_locs.to("cpu", non_blocking=True)
        self.routed_experts = self.routed_experts.to("cpu", non_blocking=True)
        self.valid_mask = self.valid_mask.to("cpu", non_blocking=True)

    def finalize(self):
        self.host_cache.clear(self.clear_cache_locs)
        self.host_cache.scatter(self.cache_locs, self.routed_experts, self.valid_mask)


class _RoutedExpertsDeviceCache:
    def __init__(
        self,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        device: str,
    ):
        self.buffer = torch.empty(
            (max_batch_size, num_layers, topk_size),
            dtype=torch.int32,
            device=device,
        )
        self.cache_locs = torch.empty(
            (max_batch_size, num_layers), dtype=torch.int64, device=device
        )
        self.valid_mask = torch.empty(
            (max_batch_size, num_layers), dtype=torch.bool, device=device
        )
        self.reset()

    def reset(self):
        self.buffer.fill_(-1)
        self.cache_locs.fill_(-1)
        self.valid_mask.zero_()

    def capture(self, layer_id: int, topk_indices: torch.Tensor, cache_locs: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but got layer_id None"
        batch = topk_indices.shape[0]
        if batch == 0:
            return
        self.buffer[:batch, layer_id, :] = topk_indices
        self.cache_locs[:batch, layer_id] = cache_locs.to(
            device=self.cache_locs.device, dtype=self.cache_locs.dtype
        )
        self.valid_mask[:batch, layer_id] = topk_indices.ge(0).any(dim=1)

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)


class _RoutedExpertsHostCache:
    def __init__(self, num_tokens: int, num_layers: int, topk_size: int):
        self.buffer = torch.empty(
            (num_tokens, num_layers, topk_size),
            dtype=torch.int32,
            device="cpu",
            pin_memory=torch.cuda.is_available(),
        )
        self.valid_mask = torch.empty(
            (num_tokens, num_layers),
            dtype=torch.bool,
            device="cpu",
            pin_memory=torch.cuda.is_available(),
        )
        self.buffer.fill_(-1)
        self.valid_mask.zero_()

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def clear(self, loc: torch.Tensor):
        loc = loc.to(device="cpu", dtype=torch.long, non_blocking=True)
        if loc.shape[0] == 0:
            return
        self.buffer[loc].fill_(-1)
        self.valid_mask[loc].zero_()

    def scatter(
        self,
        cache_locs: torch.Tensor,
        routed_experts: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        cache_locs = cache_locs.to(device="cpu", dtype=torch.long, non_blocking=True)
        routed_experts = routed_experts.to(device="cpu", non_blocking=True)
        valid_mask = valid_mask.to(device="cpu", non_blocking=True)
        for layer_id in range(valid_mask.shape[1]):
            layer_mask = valid_mask[:, layer_id]
            if not bool(layer_mask.any()):
                continue
            layer_locs = cache_locs[layer_mask, layer_id]
            self.buffer[layer_locs, layer_id, :] = routed_experts[
                layer_mask, layer_id, :
            ]
            self.valid_mask[layer_locs, layer_id] = True

    def scatter_layer(
        self, layer_id: int, cache_locs: torch.Tensor, routed_experts: torch.Tensor
    ):
        cache_locs = cache_locs.to(device="cpu", dtype=torch.long, non_blocking=True)
        routed_experts = routed_experts.to(device="cpu", dtype=self.buffer.dtype)
        if cache_locs.shape[0] == 0:
            return
        self.buffer[cache_locs, layer_id, :] = routed_experts
        self.valid_mask[cache_locs, layer_id] = True

    def get(self, cache_pool_idx: torch.Tensor):
        routed_experts = self.buffer[cache_pool_idx]
        valid_mask = self.valid_mask[cache_pool_idx]
        if bool(valid_mask.all()):
            return routed_experts
        return {
            "schema_version": 2,
            "format": "masked_dense",
            "values": routed_experts,
            "shape": list(routed_experts.shape),
            "valid_mask": valid_mask.to(dtype=torch.uint8),
            "valid_mask_shape": list(valid_mask.shape),
            "missing_value": -1,
        }


class RoutedExpertsCapturer:
    """Capturer for routed experts with mask-aware host cache.

    Unlike generic top-k capture, routed experts can be intentionally absent
    for some token/layer pairs. The validity mask keeps those holes stable so
    response bytes do not depend on stale device-buffer contents.
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
        self.num_layers = model_config.hf_text_config.num_hidden_layers
        self.topk_size = model_config.hf_text_config.num_experts_per_tok

        server_args = get_global_server_args()
        max_batch_size = max(
            server_args.chunked_prefill_size * server_args.dp_size,
            max_running_requests,
        )

        self.host_cache = _RoutedExpertsHostCache(
            num_tokens=num_tokens,
            num_layers=self.num_layers,
            topk_size=self.topk_size,
        )
        self.device_cache = _RoutedExpertsDeviceCache(
            max_batch_size=max_batch_size,
            num_layers=self.num_layers,
            topk_size=self.topk_size + self.num_fused_shared_experts,
            device=device,
        )
        self._active_forward_batch_id = None

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

    def _maybe_begin_forward(self, forward_batch: Optional[ForwardBatch]):
        if forward_batch is None:
            return
        forward_batch_id = id(forward_batch)
        if self._active_forward_batch_id is None:
            self.device_cache.reset()
            self._active_forward_batch_id = forward_batch_id

    def _get_local_out_cache_loc_and_device_range(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> Tuple[torch.Tensor, int, int]:
        if is_dp_attention_enabled():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            out_cache_local_start_pos = int(local_start_pos.item())
            local_num_tokens = int(local_num_tokens.item())
            out_cache_loc_len = forward_batch.out_cache_loc.shape[0]

            if can_run_graph:
                device_local_start_pos = get_attention_dp_rank() * cuda_graph_batch
            else:
                device_local_start_pos = out_cache_local_start_pos

            out_cache_local_end_pos = out_cache_local_start_pos + local_num_tokens
            if out_cache_local_end_pos <= out_cache_loc_len:
                out_cache_loc = forward_batch.out_cache_loc[
                    out_cache_local_start_pos:out_cache_local_end_pos
                ]
            else:
                out_cache_loc = forward_batch.out_cache_loc[:local_num_tokens]
        else:
            device_local_start_pos = 0
            out_cache_loc = forward_batch.out_cache_loc

        device_local_end_pos = device_local_start_pos + out_cache_loc.shape[0]
        return out_cache_loc, device_local_start_pos, device_local_end_pos

    def _get_capture_rows_and_cache_locs(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        forward_batch: Optional[ForwardBatch],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch = topk_ids.shape[0]
        if forward_batch is None or forward_batch.out_cache_loc is None:
            return topk_ids, torch.arange(
                batch, device=topk_ids.device, dtype=torch.int64
            )

        out_cache_loc = forward_batch.out_cache_loc
        if is_dp_attention_enabled():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            local_start_pos = int(local_start_pos.item())
            local_num_tokens = int(local_num_tokens.item())
            if local_num_tokens == 0:
                return None, None
            local_out_cache_loc = out_cache_loc[
                local_start_pos : local_start_pos + local_num_tokens
            ]
        else:
            local_start_pos = 0
            local_num_tokens = out_cache_loc.shape[0]
            local_out_cache_loc = out_cache_loc

        req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        seq_lens = getattr(forward_batch, "seq_lens", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        custom_last_index = getattr(forward_batch, "custom_last_index", None)
        custom_last_cache_loc = getattr(forward_batch, "custom_last_cache_loc", None)
        is_extend = (
            forward_batch.forward_mode.is_extend()
            if getattr(forward_batch, "forward_mode", None) is not None
            else False
        )

        if (
            is_extend
            and custom_last_index is None
            and req_to_token_pool is not None
            and req_pool_indices is not None
            and seq_lens is not None
            and extend_seq_lens is not None
        ):
            total_extend_tokens = int(extend_seq_lens.sum().item())
            if total_extend_tokens > 0:
                max_extend_len = int(extend_seq_lens.max().item())
                offsets = torch.arange(
                    max_extend_len, device=seq_lens.device, dtype=torch.int64
                )
                extend_seq_lens_i64 = extend_seq_lens.to(dtype=torch.int64)
                token_positions = (
                    seq_lens.to(dtype=torch.int64) - extend_seq_lens_i64
                ).clamp(min=0)[:, None] + offsets[None, :]
                token_mask = offsets[None, :] < extend_seq_lens_i64[:, None]
                extend_cache_locs = req_to_token_pool.req_to_token[
                    req_pool_indices[:, None], token_positions
                ][token_mask]
                if batch == extend_cache_locs.shape[0]:
                    return topk_ids, extend_cache_locs
                if (
                    is_dp_attention_enabled()
                    and batch == local_num_tokens
                    and local_start_pos + local_num_tokens
                    <= extend_cache_locs.shape[0]
                ):
                    return (
                        topk_ids,
                        extend_cache_locs[
                            local_start_pos : local_start_pos + local_num_tokens
                        ],
                    )

        if custom_last_index is not None and custom_last_cache_loc is not None:
            if custom_last_index.shape[0] == 0:
                return topk_ids[:0], local_out_cache_loc[:0]
            if batch == custom_last_cache_loc.shape[0]:
                return topk_ids, custom_last_cache_loc
            if batch > int(custom_last_index.max().item()):
                if is_dp_attention_enabled():
                    local_end_pos = local_start_pos + local_num_tokens
                    local_mask = (custom_last_index >= local_start_pos) & (
                        custom_last_index < local_end_pos
                    )
                    if not bool(local_mask.any()):
                        return None, None
                    return (
                        topk_ids[custom_last_index[local_mask]],
                        custom_last_cache_loc[local_mask],
                    )
                return topk_ids[custom_last_index], custom_last_cache_loc

        if (
            layer_id == self.num_layers - 1
            and custom_last_cache_loc is None
            and req_to_token_pool is not None
            and req_pool_indices is not None
            and seq_lens is not None
            and batch == req_pool_indices.shape[0]
            and batch == local_out_cache_loc.shape[0]
        ):
            last_token_positions = seq_lens.to(dtype=torch.int64) - 2
            last_token_positions = torch.clamp(last_token_positions, min=0)
            return (
                topk_ids,
                req_to_token_pool.req_to_token[req_pool_indices, last_token_positions],
            )

        in_cuda_graph_capture = (
            topk_ids.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        )
        if custom_last_index is None and not in_cuda_graph_capture:
            existing_valid_rows = self.device_cache.valid_mask.any(dim=1)
            existing_valid_count = int(existing_valid_rows.sum().item())
        else:
            existing_valid_rows = None
            existing_valid_count = 0

        if custom_last_index is None and existing_valid_count > batch:
            if (
                req_to_token_pool is not None
                and req_pool_indices is not None
                and seq_lens is not None
                and batch == req_pool_indices.shape[0]
            ):
                last_token_positions = seq_lens.to(dtype=torch.int64) - 2
                last_token_positions = torch.clamp(last_token_positions, min=0)
                return (
                    topk_ids,
                    req_to_token_pool.req_to_token[
                        req_pool_indices, last_token_positions
                    ],
                )
            existing_rows = torch.nonzero(existing_valid_rows, as_tuple=False).flatten()
            target_rows = existing_rows[-batch:]
            row_cache_locs = self.device_cache.cache_locs[target_rows]
            row_valid_mask = self.device_cache.valid_mask[target_rows]
            first_valid_layer = row_valid_mask.to(torch.int64).argmax(dim=1)
            return (
                topk_ids,
                row_cache_locs[
                    torch.arange(batch, device=topk_ids.device), first_valid_layer
                ],
            )

        if extend_seq_lens is not None and int(extend_seq_lens.sum().item()) > batch:
            custom_last_index = torch.cumsum(extend_seq_lens, dim=0) - 1
            if (
                req_to_token_pool is not None
                and req_pool_indices is not None
                and seq_lens is not None
            ):
                last_token_positions = seq_lens.to(dtype=torch.int64) - 2
                last_token_positions = torch.clamp(last_token_positions, min=0)
                custom_last_cache_loc = req_to_token_pool.req_to_token[
                    req_pool_indices, last_token_positions
                ]
            else:
                valid_rows = self.device_cache.valid_mask.any(dim=1)
                custom_last_cache_loc = None
                if bool(valid_rows.any()) and batch == custom_last_index.shape[0]:
                    row_cache_locs = self.device_cache.cache_locs[custom_last_index]
                    row_valid_mask = self.device_cache.valid_mask[custom_last_index]
                    first_valid_layer = row_valid_mask.to(torch.int64).argmax(dim=1)
                    custom_last_cache_loc = row_cache_locs[
                        torch.arange(batch, device=topk_ids.device), first_valid_layer
                    ]
            if custom_last_cache_loc is not None:
                if custom_last_index.shape[0] == 0:
                    return topk_ids[:0], local_out_cache_loc[:0]
                if batch == custom_last_cache_loc.shape[0]:
                    return topk_ids, custom_last_cache_loc
                if batch > int(custom_last_index.max().item()):
                    if is_dp_attention_enabled():
                        local_end_pos = local_start_pos + local_num_tokens
                        local_mask = (custom_last_index >= local_start_pos) & (
                            custom_last_index < local_end_pos
                        )
                        if not bool(local_mask.any()):
                            return None, None
                        return (
                            topk_ids[custom_last_index[local_mask]],
                            custom_last_cache_loc[local_mask],
                        )
                    return topk_ids[custom_last_index], custom_last_cache_loc

        if (
            req_to_token_pool is not None
            and req_pool_indices is not None
            and seq_lens is not None
            and batch == req_pool_indices.shape[0]
            and is_extend
            and (
                int(extend_seq_lens.sum().item()) > batch
                if extend_seq_lens is not None
                else True
            )
        ):
            last_token_positions = seq_lens.to(dtype=torch.int64) - 2
            last_token_positions = torch.clamp(last_token_positions, min=0)
            return (
                topk_ids,
                req_to_token_pool.req_to_token[req_pool_indices, last_token_positions],
            )

        if custom_last_index is not None and custom_last_cache_loc is None:
            if (
                req_to_token_pool is not None
                and req_pool_indices is not None
                and seq_lens is not None
            ):
                last_token_positions = seq_lens.to(dtype=torch.int64) - 2
                last_token_positions = torch.clamp(last_token_positions, min=0)
                custom_last_cache_loc = req_to_token_pool.req_to_token[
                    req_pool_indices, last_token_positions
                ]
        if custom_last_index is not None and custom_last_cache_loc is not None:
            if custom_last_index.shape[0] == 0:
                return topk_ids[:0], local_out_cache_loc[:0]
            if batch == custom_last_cache_loc.shape[0]:
                return topk_ids, custom_last_cache_loc
            if batch > int(custom_last_index.max().item()):
                if is_dp_attention_enabled():
                    local_end_pos = local_start_pos + local_num_tokens
                    local_mask = (custom_last_index >= local_start_pos) & (
                        custom_last_index < local_end_pos
                    )
                    if not bool(local_mask.any()):
                        return None, None
                    return (
                        topk_ids[custom_last_index[local_mask]],
                        custom_last_cache_loc[local_mask],
                    )
                return topk_ids[custom_last_index], custom_last_cache_loc

        if custom_last_index is not None and batch == custom_last_index.shape[0]:
            if custom_last_index.shape[0] == 0:
                return topk_ids[:0], local_out_cache_loc[:0]
            if is_dp_attention_enabled():
                local_end_pos = local_start_pos + local_num_tokens
                local_mask = (custom_last_index >= local_start_pos) & (
                    custom_last_index < local_end_pos
                )
                if not bool(local_mask.any()):
                    return None, None
                return topk_ids[local_mask], out_cache_loc[custom_last_index[local_mask]]
            return topk_ids, out_cache_loc[custom_last_index]

        if batch == local_out_cache_loc.shape[0]:
            return topk_ids, local_out_cache_loc

        if is_dp_attention_enabled() and batch >= local_start_pos + local_num_tokens:
            return (
                topk_ids[local_start_pos : local_start_pos + local_num_tokens],
                local_out_cache_loc,
            )

        if batch <= local_out_cache_loc.shape[0]:
            return topk_ids, local_out_cache_loc[:batch]

        raise RuntimeError(
            "Unable to align routed_experts rows to KV cache locations: "
            f"{batch=} out_cache_loc={tuple(out_cache_loc.shape)}."
        )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        if get_moe_a2a_backend().is_deepep():
            local_topk = topk_indices
            topk_indices = self.gather_buffer[
                : local_topk.size(0) * get_attention_tp_size()
            ]
            attn_tp_all_gather_into_tensor(topk_indices, local_topk)

        forward_batch = get_current_forward_batch()
        self._maybe_begin_forward(forward_batch)
        capture_topk, cache_locs = self._get_capture_rows_and_cache_locs(
            layer_id, topk_indices, forward_batch
        )
        if capture_topk is None or cache_locs is None:
            return
        if cache_locs.shape[0] != capture_topk.shape[0]:
            return

        if layer_id == self.num_layers - 1 and forward_batch is not None:
            req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
            req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
            seq_lens = getattr(forward_batch, "seq_lens", None)
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            is_extend = (
                forward_batch.forward_mode.is_extend()
                if getattr(forward_batch, "forward_mode", None) is not None
                else False
            )
            if (
                is_extend
                and getattr(forward_batch, "custom_last_cache_loc", None) is None
                and req_to_token_pool is not None
                and req_pool_indices is not None
                and seq_lens is not None
                and extend_seq_lens is not None
                and int(extend_seq_lens.sum().item()) > capture_topk.shape[0]
            ):
                for position_offset in (1, 2):
                    token_positions = seq_lens.to(dtype=torch.int64) - position_offset
                    token_positions = torch.clamp(token_positions, min=0)
                    visible_cache_locs = req_to_token_pool.req_to_token[
                        req_pool_indices, token_positions
                    ]
                    if visible_cache_locs.shape[0] == capture_topk.shape[0]:
                        self.host_cache.scatter_layer(
                            layer_id, visible_cache_locs, capture_topk
                        )

        in_cuda_graph_capture = (
            capture_topk.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        )
        if not in_cuda_graph_capture:
            self.host_cache.scatter_layer(layer_id, cache_locs, capture_topk)
        self.device_cache.capture(layer_id, capture_topk, cache_locs)

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
        return self.host_cache.get(cache_pool_idx)

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[RoutedExpertsCaptureOutput]:
        self._maybe_begin_forward(forward_batch)
        if no_copy_to_cpu:
            output = self._prepare_routed_experts_output(
                forward_batch=forward_batch,
                can_run_graph=can_run_graph,
                cuda_graph_batch=cuda_graph_batch,
            )
            self._active_forward_batch_id = None
            return output

        self._sync_fwd_experts_buffer_DtoH(
            forward_batch=forward_batch,
            can_run_graph=can_run_graph,
            cuda_graph_batch=cuda_graph_batch,
        )
        self.device_cache.reset()
        self._active_forward_batch_id = None
        return None

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ):
        out_cache_loc, _, _ = self._get_local_out_cache_loc_and_device_range(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        if out_cache_loc.shape[0] == 0:
            return
        self.host_cache.clear(out_cache_loc)
        row_valid_mask = self.device_cache.valid_mask.any(dim=1)
        if not bool(row_valid_mask.any()):
            return
        self.host_cache.scatter(
            self.device_cache.cache_locs[row_valid_mask],
            self.device_cache.buffer[row_valid_mask, :, : self.topk_size],
            self.device_cache.valid_mask[row_valid_mask],
        )

    def _prepare_routed_experts_output(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> RoutedExpertsCaptureOutput:
        out_cache_loc, _, _ = self._get_local_out_cache_loc_and_device_range(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        row_valid_mask = self.device_cache.valid_mask.any(dim=1)
        return RoutedExpertsCaptureOutput(
            clear_cache_locs=out_cache_loc,
            cache_locs=self.device_cache.cache_locs[row_valid_mask],
            routed_experts=self.device_cache.buffer[row_valid_mask, :, : self.topk_size],
            valid_mask=self.device_cache.valid_mask[row_valid_mask],
            host_cache=self.host_cache,
        )


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


def extract_routed_experts_from_meta_info(data):
    routed_experts_payload = data["meta_info"].get("routed_experts", None)
    if isinstance(routed_experts_payload, dict):
        payload = dict(routed_experts_payload)
        if isinstance(payload.get("values"), str):
            payload["values"] = np.frombuffer(
                pybase64.b64decode(payload["values"].encode("utf-8")), dtype=np.int32
            )
        if isinstance(payload.get("valid_mask"), str):
            payload["valid_mask"] = np.frombuffer(
                pybase64.b64decode(payload["valid_mask"].encode("utf-8")),
                dtype=np.uint8,
            )
        return payload
    routed_experts = np.frombuffer(
        pybase64.b64decode(routed_experts_payload.encode("utf-8")), dtype=np.int32
    )
    return routed_experts
