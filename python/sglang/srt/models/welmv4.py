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

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import (
    RotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
    yarn_get_mscale,
)
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.welmv4_op import (
    WelmV4FusedRMSNorm,
    WelmV4InplaceRotaryEmbedding,
    inplace_sigmoid_mul,
    mmq_style_add_residual,
    mmq_style_expert_bias_topk,
    mmq_style_k_rms_norm,
    mmq_style_norm_after_attn,
    mmq_style_router_linear,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.welm_perf_opt import (
    compute_welm_oe_embedding,
)
from sglang.srt.server_args import get_global_server_args

# from sglang.srt.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.utils import (
    add_prefix,
    get_bool_env_var,
    is_cuda,
    make_layers,
    set_weight_attrs,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
WELM_KV_MIRROR_CTX_KEY = "welm_kv_mirror_ctx"
_WELM_DUMP_PROCESS_DIR = None


def _welm_dump_enabled() -> bool:
    return os.getenv("SGLANG_DUMP_ACTIVATIONS", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _welm_should_dump_layer(layer_idx: int) -> bool:
    if not _welm_dump_enabled():
        return False
    layer_idxs = os.getenv("SGLANG_DUMP_ACTIVATIONS_LAYER_IDXS")
    if not layer_idxs:
        return True
    return str(layer_idx) in {x.strip() for x in layer_idxs.split(",") if x.strip()}


def _welm_dump_tensor(name: str, tensor: torch.Tensor) -> None:
    global _WELM_DUMP_PROCESS_DIR
    if not isinstance(tensor, torch.Tensor):
        return
    if _WELM_DUMP_PROCESS_DIR is None:
        process_dir = os.getenv("SGLANG_DUMP_ACTIVATIONS_PROCESS_DIR")
        if process_dir:
            _WELM_DUMP_PROCESS_DIR = Path(process_dir)
        else:
            base_dir = Path(
                os.getenv("SGLANG_DUMP_ACTIVATIONS_DIR", "/tmp/sglang_welm_dump")
            )
            _WELM_DUMP_PROCESS_DIR = (
                base_dir / f"TP0_PP0_Rank0_pid{os.getpid()}" / "Pass00000"
            )
            os.environ["SGLANG_DUMP_ACTIVATIONS_PROCESS_DIR"] = str(
                _WELM_DUMP_PROCESS_DIR
            )
        _WELM_DUMP_PROCESS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), _WELM_DUMP_PROCESS_DIR / f"{name}.pt")


def _get_welm_kv_mirror_ctx(forward_batch: ForwardBatch) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    if forward_batch.model_specific_states is None:
        forward_batch.model_specific_states = {}
    ctx = forward_batch.model_specific_states.get(WELM_KV_MIRROR_CTX_KEY)
    if ctx is None:
        ctx = {}
        forward_batch.model_specific_states[WELM_KV_MIRROR_CTX_KEY] = ctx
    return ctx


def _welm_should_contract_kv_mirror(forward_batch: ForwardBatch) -> bool:
    return (
        forward_batch.enable_welm_kv_mirror_opt
        and forward_batch.forward_mode.is_extend_without_speculative()
    )


def _get_kv_mirror_pair_maps(
    kv_mirror_layers: List[int],
    kv_mirror_imitated_layers: List[int],
    *,
    num_hidden_layers: int,
    num_nextn_predict_layers: int,
    is_nextn: bool,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    valid_pairs = []
    for mirror, imitated in zip(kv_mirror_layers, kv_mirror_imitated_layers):
        if is_nextn:
            mirror_valid = num_hidden_layers <= mirror < (
                num_hidden_layers + num_nextn_predict_layers
            )
            imitated_valid = 0 <= imitated < num_hidden_layers
        else:
            mirror_valid = 0 <= mirror < num_hidden_layers
            imitated_valid = 0 <= imitated < num_hidden_layers
        if mirror_valid and imitated_valid:
            valid_pairs.append((mirror, imitated))
    mirror_to_imitated = dict(valid_pairs)
    imitated_to_mirror = {imitated: mirror for mirror, imitated in mirror_to_imitated.items()}
    return mirror_to_imitated, imitated_to_mirror


class BaseWelmQkvProjection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        qkv_bias: bool,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        tp_rank: int,
        tp_size: int,
        shard_layout: Tuple[str, ...] = ("q", "k", "v"),
    ) -> None:
        super().__init__()
        if quant_config is not None:
            logger.warning(
                "WeLM qkv projection uses an unquantized custom module; ignoring "
                "quant_config=%s for %s",
                type(quant_config).__name__,
                prefix,
            )
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.shard_layout = shard_layout
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_proj_shard_size = self.num_heads * self.head_dim
        self.kv_proj_shard_size = self.num_kv_heads * self.head_dim
        self.output_size_per_partition = sum(
            self._get_local_shard_size(shard_id) for shard_id in self.shard_layout
        )
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, self.hidden_size),
            requires_grad=False,
        )
        set_weight_attrs(
            self.weight,
            {"input_dim": 1, "output_dim": 0, "weight_loader": self.weight_loader},
        )
        if qkv_bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition),
                requires_grad=False,
            )
            set_weight_attrs(
                self.bias,
                {"output_dim": 0, "weight_loader": self.weight_loader},
            )
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        attn: "Qwen2MoeAttention",
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _apply_qkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)

    def _get_primary_kv_shard_names(self) -> Tuple[str, str]:
        if "k" in self.shard_layout and "v" in self.shard_layout:
            return "k", "v"
        return "k1", "v1"

    def _get_local_shard_size(self, shard_id: str) -> int:
        return self.q_proj_shard_size if shard_id == "q" else self.kv_proj_shard_size

    def _get_total_shard_size(self, shard_id: str) -> int:
        return (
            self.total_num_heads * self.head_dim
            if shard_id == "q"
            else self.total_num_kv_heads * self.head_dim
        )

    def _get_tp_shard_index(self, shard_id: str) -> int:
        if shard_id == "q":
            return self.tp_rank
        return self.tp_rank // self.num_kv_head_replicas

    def _get_shard_slice(
        self, shard_id: str
    ) -> Tuple[slice, Optional[slice]]:
        start = 0
        for layout_shard_id in self.shard_layout:
            size = self._get_local_shard_size(layout_shard_id)
            if layout_shard_id == shard_id:
                bias_slice = slice(start, start + size) if self.bias is not None else None
                return slice(start, start + size), bias_slice
            start += size
        raise KeyError(f"Unknown shard_id={shard_id!r} for layout={self.shard_layout}")

    def _split_qkv(
        self, attn: "Qwen2MoeAttention", qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    def get_kv_weight_bias(
        self, attn: "Qwen2MoeAttention"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        k_shard_id, v_shard_id = self._get_primary_kv_shard_names()
        k_weight_slice, k_bias_slice = self._get_shard_slice(k_shard_id)
        v_weight_slice, v_bias_slice = self._get_shard_slice(v_shard_id)
        kv_weight = torch.cat(
            (self.weight[k_weight_slice, :], self.weight[v_weight_slice, :]), dim=0
        )
        if self.bias is None:
            kv_bias = None
        else:
            kv_bias = torch.cat(
                (self.bias[k_bias_slice], self.bias[v_bias_slice]), dim=0
            )
        return kv_weight, kv_bias

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):
        output_dim = getattr(param, "output_dim", None)
        param_data = param.data
        if loaded_shard_id is None:
            if output_dim is None:
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offset = 0
            for shard_id in self.shard_layout:
                shard_size = self._get_total_shard_size(shard_id)
                loaded_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_shard, shard_id)
                shard_offset += shard_size
            return

        assert loaded_shard_id in self.shard_layout
        if output_dim is not None:
            shard_offset = self._get_shard_slice(loaded_shard_id)[0].start
            shard_size = self._get_local_shard_size(loaded_shard_id)
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            shard_id = self._get_tp_shard_index(loaded_shard_id)
            start_idx = shard_id * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class StandardQkvProjection(BaseWelmQkvProjection):
    def forward(
        self,
        attn: "Qwen2MoeAttention",
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self._apply_qkv(hidden_states)
        q, k, v = self._split_qkv(attn, qkv)
        return q, k, v, hidden_states


class ImitateQkvKvProjection(BaseWelmQkvProjection):
    def __init__(self, mirror_layer_idx: int, **kwargs) -> None:
        super().__init__(shard_layout=("q", "k1", "v1", "k2", "v2"), **kwargs)
        self.mirror_layer_idx = mirror_layer_idx

    def forward(
        self,
        attn: "Qwen2MoeAttention",
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self._apply_qkv(hidden_states)
        q, k, v, mirror_k, mirror_v = qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size, attn.kv_size, attn.kv_size],
            dim=-1,
        )
        _get_welm_kv_mirror_ctx(forward_batch)[attn.kv_mirror_layer_idx] = (
            mirror_k,
            mirror_v,
        )
        return q, k, v, hidden_states


class MirrorQProjection(BaseWelmQkvProjection):
    def __init__(self, imitated_layer_idx: int, **kwargs) -> None:
        super().__init__(shard_layout=("q",), **kwargs)
        self.imitated_layer_idx = imitated_layer_idx

    def forward(
        self,
        attn: "Qwen2MoeAttention",
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        project_hidden_states = hidden_states
        if (
            _welm_should_contract_kv_mirror(forward_batch)
            and not hasattr(forward_batch, "custom_last_index")
        ):
            forward_batch.custom_last_index = (
                torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            )
            project_hidden_states = hidden_states[forward_batch.custom_last_index]

        mirror_ctx = _get_welm_kv_mirror_ctx(forward_batch)
        kv_activation = mirror_ctx.pop(self.imitated_layer_idx, None)
        if kv_activation is None:
            raise RuntimeError(
                f"Missing mirrored KV activation for imitated_layer_idx={self.imitated_layer_idx}"
            )

        k, v = kv_activation
        if attn.need_clear_kv_cache:
            mirror_ctx.clear()
        if not mirror_ctx:
            forward_batch.model_specific_states.pop(WELM_KV_MIRROR_CTX_KEY, None)
        if not forward_batch.model_specific_states:
            forward_batch.model_specific_states = None
        q = F.linear(project_hidden_states, self.weight, self.bias)
        return q, k, v, project_hidden_states


class NextnMirrorQProjection(BaseWelmQkvProjection):
    def __init__(self, imitated_layer_idx: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.imitated_layer_idx = imitated_layer_idx

    def forward(
        self,
        attn: "Qwen2MoeAttention",
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not _welm_should_contract_kv_mirror(forward_batch):
            qkv = self._apply_qkv(hidden_states)
            q, k, v = self._split_qkv(attn, qkv)
            return q, k, v, hidden_states

        project_hidden_states = hidden_states
        if not hasattr(forward_batch, "custom_last_index"):
            forward_batch.custom_last_index = (
                torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            )
        project_hidden_states = hidden_states[forward_batch.custom_last_index]

        mirror_ctx = _get_welm_kv_mirror_ctx(forward_batch)
        kv_activation = mirror_ctx.pop(self.imitated_layer_idx, None)
        assert (
            kv_activation is not None
        ), f"Missing mirrored KV activation for nextn imitated_layer_idx={self.imitated_layer_idx}"

        k, v = kv_activation
        if attn.need_clear_kv_cache:
            mirror_ctx.clear()
        if not mirror_ctx:
            forward_batch.model_specific_states.pop(WELM_KV_MIRROR_CTX_KEY, None)
        if not forward_batch.model_specific_states:
            forward_batch.model_specific_states = None
        q_weight = self.weight[: attn.q_size, :]
        q_bias = None if self.bias is None else self.bias[: attn.q_size]
        q = F.linear(project_hidden_states, q_weight, q_bias)
        return q, k, v, project_hidden_states

class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        swiglu_clamp_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.swiglu_clamp_limit = swiglu_clamp_limit

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        if self.swiglu_clamp_limit is not None and self.swiglu_clamp_limit > 0:
            d = gate_up.shape[-1] // 2
            gate = F.silu(gate_up[..., :d]).clamp_(max=self.swiglu_clamp_limit)
            up = gate_up[..., d:].clamp(min=-self.swiglu_clamp_limit, max=self.swiglu_clamp_limit)
            x = gate * up
        else:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


def expert_bias_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    expert_bias: torch.Tensor,
    renormalize: bool = False,
    score_func: str = "sigmoid",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    if score_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1).type_as(gating_output)
    else:
        scores = torch.sigmoid(gating_output).type_as(gating_output)

    if (
        scores.is_cuda
        and scores.dtype == torch.float32
        and expert_bias.dtype == torch.float32
    ):
        topk_scores, indices = mmq_style_expert_bias_topk(scores, expert_bias, topk)
    else:
        scores_for_routing = scores + expert_bias
        _, indices = torch.topk(scores_for_routing, topk, dim=-1)
        topk_scores = torch.gather(scores, dim=1, index=indices).type_as(scores)

    return topk_scores, indices


def sigmoid_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # if softmax, then use qwen3 moe's routing function
    scores = torch.sigmoid(gating_output).type_as(gating_output)
    scores_for_routing = scores
    if correction_bias is not None:
        scores += correction_bias
    _, indices = torch.topk(scores, topk, dim=-1)
    topk_scores = torch.gather(scores, dim=1, index=indices).type_as(scores)
    return topk_scores, indices


class Qwen2MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.expert_bias = torch.nn.Parameter(
            torch.zeros((config.num_experts), dtype=torch.float32)
        )
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        moe_clamp_limits = getattr(config, "moe_expert_swiglu_clamp_limit_layerwise", [])
        moe_clamp_limit = (
            moe_clamp_limits[layer_id]
            if layer_id < len(moe_clamp_limits) and moe_clamp_limits[layer_id] > 0
            else None
        )
        shared_clamp_limits = getattr(config, "shared_expert_swiglu_clamp_limit_layerwise", [])
        shared_clamp_limit = (
            shared_clamp_limits[layer_id]
            if layer_id < len(shared_clamp_limits) and shared_clamp_limits[layer_id] > 0
            else None
        )

        self.router_score_func = (
            config.router_score_func
            if hasattr(config, "router_score_func")
            else "softmax"
        )
        if config.moe_routing_type == "expert_bias":
            from functools import partial

            custom_routing_function = partial(
                expert_bias_routing,
                expert_bias=self.expert_bias,
                score_func=self.router_score_func,
            )
            self.custom_routing_function = custom_routing_function
        else:
            if self.router_score_func == "softmax":
                self.custom_routing_function = None
            elif self.router_score_func == "sigmoid":
                self.custom_routing_function = sigmoid_routing_function
            else:
                raise ValueError(f"Unknown router_score_func: {self.router_score_func}")

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            custom_routing_function=self.custom_routing_function,
        )

        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            swiglu_clamp_limit=moe_clamp_limit,
            apply_router_weight_on_swiglu=get_bool_env_var(
                "SGLANG_WELMV4_MMQ_SCORE_ON_SWIGLU", "false"
            ),
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.weight.data = self.gate.weight.to(torch.float32)
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
                swiglu_clamp_limit=shared_clamp_limit,
            )
        else:
            self.shared_expert = None

        self.shared_expert_gate = None
        has_shared_expert_gate = getattr(
            config, "has_shared_expert_gate", True
        )  # default to true since qwen2_moe always has it
        if has_shared_expert_gate:
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
        return_components: bool = False,
        skip_component_output: bool = False,
    ) -> torch.Tensor:
        dump_this_layer = _welm_should_dump_layer(self.layer_id)
        dump_prefix = f"model.layers.{self.layer_id}.mlp"
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.router.input", hidden_states)
            _welm_dump_tensor(f"{dump_prefix}.router.input_fp32", hidden_states_fp32)
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
                )
        router_logits = mmq_style_router_linear(hidden_states, self.gate.weight)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.router.logits", router_logits)
            if self.router_score_func == "softmax":
                router_scores = torch.softmax(router_logits, dim=-1).type_as(
                    router_logits
                )
            else:
                router_scores = torch.sigmoid(router_logits).type_as(router_logits)
            _welm_dump_tensor(f"{dump_prefix}.router.scores", router_scores)
        topk_output = self.topk(hidden_states, router_logits)
        if dump_this_layer and hasattr(topk_output, "topk_weights"):
            _welm_dump_tensor(f"{dump_prefix}.router.topk_scores", topk_output.topk_weights)
            _welm_dump_tensor(f"{dump_prefix}.router.topk_ids", topk_output.topk_ids)
        final_hidden_states = self.experts(hidden_states, topk_output)
        experts_output = final_hidden_states
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.experts_output", final_hidden_states)
        if return_components and skip_component_output:
            return (
                experts_output.view(num_tokens, hidden_dim),
                experts_output.view(num_tokens, hidden_dim),
                None
                if shared_output is None
                else shared_output.view(num_tokens, hidden_dim),
            )
        if shared_output is not None:
            if dump_this_layer:
                _welm_dump_tensor(f"{dump_prefix}.shared_output", shared_output)
            final_hidden_states = final_hidden_states + shared_output
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.output", final_hidden_states)
        if self.tp_size > 1 and not use_reduce_scatter:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        if return_components:
            return (
                final_hidden_states,
                experts_output.view(num_tokens, hidden_dim),
                None
                if shared_output is None
                else shared_output.view(num_tokens, hidden_dim),
            )
        return final_hidden_states


class LinearScalingRotaryEmbedding(WelmV4InplaceRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: Union[List[float], float],
        dtype: torch.dtype,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors  # noqa
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        # Lazy initialized.
        self._scaling_factor_to_offset: Dict[float, int]

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: List[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        offsets: List[int] = []
        for scaling_factor in self.scaling_factors:
            # NOTE(woosuk): self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        return torch.cat(cache_list, dim=0)

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self._scaling_factor_to_offset


# WelmV4InplaceRotaryEmbedding
class Qwen2MoeYarnScalingRotaryEmbedding(WelmV4InplaceRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        compress: float = 0,
        max_position: int = 40 * 4096,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.compress = compress
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.max_position = max_position
        inv_freq_extra = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        inv_freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq_extra", inv_freq_extra, persistent=False)
        self.register_buffer("inv_freq_inter", inv_freq_inter, persistent=False)

        self.cos_sin_cache = self._update_cos_sin_cache(self.max_position)

    def _update_cos_sin_cache(self, seqlen: int):
        """Update cos/sin cache with YaRN scaling"""
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2, dtype=torch.float32
        ).to(device=self.inv_freq_inter.device)

        inv_freq = (
            self.inv_freq_inter * (1 - inv_freq_mask)
            + self.inv_freq_extra * inv_freq_mask
        )

        seq = (
            torch.arange(seqlen, device=self.inv_freq_extra.device, dtype=torch.float32)
            * self.compress
        )

        freqs = torch.outer(seq, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        _cos_cached = (torch.cos(freqs) * _mscale).to(torch.float32)
        _sin_cached = (torch.sin(freqs) * _mscale).to(torch.float32)
        cache = torch.cat((_cos_cached, _sin_cached), dim=-1)
        return cache


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    compress: float = 1.0,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        raise ValueError(f"Please set RoPE scaling")
    else:
        scaling_type = rope_scaling["type"]

        if scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = LinearScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )

        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            base_max_position = int(original_max_position * scaling_factor)
            if max_position < base_max_position:
                raise ValueError(
                    f"max_position ({max_position}) < original_max_position "
                    f"({original_max_position}) * scaling_factor ({scaling_factor})"
                )
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in (
                    "extrapolation_factor",
                    "attn_factor",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                )
            }
            rotary_emb = Qwen2MoeYarnScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
                compress=compress,
                max_position=max_position,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


class Qwen2MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        compress: float = 1.0,
        max_position_embeddings: int = 8192,
        qkv_bias: int = True,
        out_bias: int = False,
        qk_norm: bool = False,
        k_norm: bool = False,
        qk_rope_head_dim: int = 0,
        qk_norm_eps: float = 1e-5,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
        kv_mirror_layers=[],
        kv_mirror_imitated_layers=[],
        sliding_window_size_layerwise=[],
        enable_attn_sink_layerwise=[],
        layer_idx: Optional[int] = None,
        o_norm=False,
        total_layer_num: int = 1,
        num_nextn_predict_layers: int = 0,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.compress = compress
        self.max_position_embeddings = max_position_embeddings
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_norm = qk_norm
        self.only_k_norm = k_norm
        self.use_o_norm = o_norm
        self.total_layer_num = total_layer_num
        self.o_norm = RMSNorm(self.hidden_size) if self.use_o_norm else nn.Identity()

        self.q_norm = (
            WelmV4FusedRMSNorm(self.head_dim, eps=qk_norm_eps)
            if self.qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            WelmV4FusedRMSNorm(self.head_dim, eps=qk_norm_eps)
            if self.qk_norm or self.only_k_norm
            else nn.Identity()
        )

        self.kv_mirror_layers = kv_mirror_layers
        self.kv_mirror_imitated_layers = kv_mirror_imitated_layers
        self.layer_idx = layer_idx
        print(
            "self.layer_idx:{}".format(layer_idx),
            "self.kv_mirror_layers:",
            self.kv_mirror_layers,
            "self.kv_mirror_imitated_layers:",
            self.kv_mirror_imitated_layers,
            flush=True,
        )
        if len(sliding_window_size_layerwise) > layer_idx:
            self.sliding_window_size = sliding_window_size_layerwise[layer_idx]
        else:
            self.sliding_window_size = -1
        print(
            "self.layer_idx:{}".format(layer_idx),
            "self.sliding_window_size:",
            self.sliding_window_size,
            flush=True,
        )
        if len(enable_attn_sink_layerwise) > layer_idx:
            self.enable_attention_sink = enable_attn_sink_layerwise[layer_idx]
        else:
            self.enable_attention_sink = False
        print(
            "self.layer_idx:{}".format(layer_idx),
            "self.enable_attention_sink:",
            self.enable_attention_sink,
            flush=True,
        )
        if self.enable_attention_sink == True:
            self.attn_sink = nn.Parameter(
                torch.empty(self.num_heads), requires_grad=False
            )
        else:
            self.attn_sink = None

        qkv_proj_kwargs = dict(
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            qkv_bias=qkv_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=out_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=True,
            prefix=add_prefix("o_proj", prefix),
        )
        if rope_scaling is None:
            rope_scaling = {"type": "linear", "factor": 1 / self.compress}
        else:
            assert self.compress == 1.0, "Compress must be 1.0 for custom rope scaling."
            if rope_scaling["type"] == "yarn":
                mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
                apply_softmax_scale = rope_scaling.get("apply_softmax_scale", False)
                scaling_factor = rope_scaling["factor"]
                if apply_softmax_scale and mscale_all_dim:
                    mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                    self.scaling = self.scaling * mscale * mscale

        self.rotary_emb = get_rope(
            # self.qk_rope_head_dim,
            self.head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            compress=self.compress,
            rope_scaling=rope_scaling,
        )

        self.rotary_emb_orig = get_rope(
            self.qk_rope_head_dim,
            # self.head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            compress=self.compress,
            rope_scaling=rope_scaling,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=self.sliding_window_size,
        )
        self.gated_self_attention_headwise = True
        if self.gated_self_attention_headwise:
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
            )
        self.attn.is_kv_mirror = (
            get_global_server_args().enable_welm_kv_mirror_opt
            and self.layer_idx in self.kv_mirror_layers
        )
        self.kv_mirror_layer_idx = (
            layer_idx if not is_nextn else layer_idx + total_layer_num
        )
        mirror_to_imitated, imitated_to_mirror = _get_kv_mirror_pair_maps(
            self.kv_mirror_layers,
            self.kv_mirror_imitated_layers,
            num_hidden_layers=total_layer_num,
            num_nextn_predict_layers=num_nextn_predict_layers,
            is_nextn=is_nextn,
        )
        if self.kv_mirror_layer_idx in mirror_to_imitated:
            projection_cls = NextnMirrorQProjection if is_nextn else MirrorQProjection
            self.qkv_proj = projection_cls(
                imitated_layer_idx=mirror_to_imitated[self.kv_mirror_layer_idx],
                **qkv_proj_kwargs,
            )
        elif self.kv_mirror_layer_idx in imitated_to_mirror:
            self.qkv_proj = ImitateQkvKvProjection(
                mirror_layer_idx=imitated_to_mirror[self.kv_mirror_layer_idx],
                **qkv_proj_kwargs,
            )
        else:
            self.qkv_proj = StandardQkvProjection(**qkv_proj_kwargs)
        if get_global_server_args().speculative_algorithm is not None:
            self.need_clear_kv_cache = self.layer_idx == num_nextn_predict_layers - 1
        else:
            self.need_clear_kv_cache = self.layer_idx == total_layer_num - 1
        self.is_nextn = is_nextn

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        skip_o_norm: bool = False,
    ) -> torch.Tensor:
        dump_this_layer = _welm_should_dump_layer(self.layer_idx)
        dump_prefix = f"model.layers.{self.layer_idx}.self_attn"
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.positions", positions)
            if forward_batch.extend_seq_lens is not None:
                _welm_dump_tensor(
                    f"{dump_prefix}.extend_seq_lens",
                    forward_batch.extend_seq_lens,
                )
            if forward_batch.extend_prefix_lens is not None:
                _welm_dump_tensor(
                    f"{dump_prefix}.extend_prefix_lens",
                    forward_batch.extend_prefix_lens,
                )
            if forward_batch.out_cache_loc is not None:
                _welm_dump_tensor(
                    f"{dump_prefix}.out_cache_loc", forward_batch.out_cache_loc
                )
            if forward_batch.req_pool_indices is not None:
                _welm_dump_tensor(
                    f"{dump_prefix}.req_pool_indices",
                    forward_batch.req_pool_indices,
                )
        q, k, v, hidden_states = self.qkv_proj.forward(
            self, hidden_states, forward_batch
        )
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.positions", positions)
            _welm_dump_tensor(f"{dump_prefix}.q_pre_rope", q)
            _welm_dump_tensor(f"{dump_prefix}.k_pre_rope", k)
            _welm_dump_tensor(f"{dump_prefix}.v", v)

        q_shape = q.shape
        k_shape = k.shape

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        if self.qk_norm:
            q_by_head, _ = self.q_norm(q_by_head)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.q_after_norm", q_by_head.view(q.shape))
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        # for welmv4, qk_norm is false and only_k_norm is true
        # fuse: implement a high precision and fused k_norm
        if self.qk_norm or self.only_k_norm:
            k_by_head = mmq_style_k_rms_norm(
                k_by_head.contiguous(), self.k_norm.weight, self.k_norm.eps
            )
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.k_after_norm", k_by_head.view(k.shape))
        k = k_by_head.view(k.shape)

        qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        if qk_nope_head_dim > 0:
            if (
                _welm_should_contract_kv_mirror(forward_batch)
                and self.kv_mirror_layer_idx in self.kv_mirror_layers
            ):
                self.rotary_emb.forward_cuda(
                    positions, q, k, last_index=forward_batch.custom_last_index
                )
            else:
                self.rotary_emb.forward_cuda(positions, q, k)
            q = q.view(q_shape)
            k = k.view(k_shape)
        else:
            q, k = self.rotary_emb(positions, q, k)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.q_post_rope", q)
            _welm_dump_tensor(f"{dump_prefix}.k_post_rope", k)

        attn_kwargs = {}
        if self.attn_sink is not None:
            attn_kwargs["sinks"] = self.attn_sink
        attn_output = self.attn(q, k, v, forward_batch, **attn_kwargs)
        if dump_this_layer:
            _welm_dump_tensor(f"{dump_prefix}.attn_output", attn_output)
        if self.gated_self_attention_headwise:
            attn_shape = attn_output.shape
            gate = self.gate_proj(hidden_states)[0].unsqueeze(
                -1
            )  # (bs * seq_len, num_heads, 1)
            if dump_this_layer:
                _welm_dump_tensor(
                    f"model.layers.{self.layer_idx}.attn.router.0", gate.squeeze(-1)
                )
            attn_output = attn_output.view(attn_shape[0], self.num_heads, -1)
            inplace_sigmoid_mul(gate, attn_output)
            attn_output = attn_output.view(attn_shape)
            if dump_this_layer:
                _welm_dump_tensor(f"{dump_prefix}.gated_attn_output", attn_output)

        output, _ = self.o_proj(attn_output)
        if dump_this_layer:
            _welm_dump_tensor(
                f"model.layers.{self.layer_idx}.attn.mixer.o_proj_out", output
            )
        if self.use_o_norm and not skip_o_norm:
            output = self.o_norm(output)
            if dump_this_layer:
                _welm_dump_tensor(
                    f"model.layers.{self.layer_idx}.attn.mixer.o_norm_out", output
                )
        if dump_this_layer:
            _welm_dump_tensor(f"model.layers.{self.layer_idx}.attn.mixer.0", output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        scale_seq_times = getattr(config, "scale_seq_times", 0)
        if scale_seq_times > 0:
            max_position_embeddings = max_position_embeddings * (scale_seq_times + 1)
        if getattr(config, "qkv_bias", None) is not None:
            qkv_bias = getattr(config, "qkv_bias")
        elif getattr(config, "qkv_proj_bias", None) is not None:
            qkv_bias = getattr(config, "qkv_proj_bias")
        else:
            qkv_bias = True
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        qk_norm = getattr(config, "qk_norm", False)
        k_norm = getattr(config, "k_norm", False)
        out_bias = getattr(config, "out_proj_bias", False)
        head_dim = getattr(
            config, "head_dim", self.hidden_size // config.num_attention_heads
        )
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", head_dim)

        self.kv_mirror_layers = getattr(config, "kv_mirror_layers", [])
        self.kv_mirror_imitated_layers = getattr(
            config, "kv_mirror_imitated_layers", []
        )
        self.sliding_window_size_layerwise = getattr(
            config, "sliding_window_size_layerwise", []
        )
        self.enable_attn_sink_layerwise = getattr(
            config, "enable_attn_sink_layerwise", []
        )
        self.ppln = getattr(config, "ppln", False)
        o_norm = getattr(config, "o_norm", False)
        self.prenorm_layer_idx = getattr(config, "prenorm_layer_idx", [])
        print(
            "self.ppln:",
            self.ppln,
            "o_norm:",
            o_norm,
            "self.prenorm_layer_idx:",
            self.prenorm_layer_idx,
        )
        total_layer_num = config.num_hidden_layers

        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            qk_norm=qk_norm,
            k_norm=k_norm,
            qk_rope_head_dim=qk_rope_head_dim,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            prefix=add_prefix("self_attn", prefix),
            kv_mirror_layers=self.kv_mirror_layers,
            kv_mirror_imitated_layers=self.kv_mirror_imitated_layers,
            sliding_window_size_layerwise=self.sliding_window_size_layerwise,
            enable_attn_sink_layerwise=self.enable_attn_sink_layerwise,
            layer_idx=layer_id,
            o_norm=o_norm and layer_id not in self.prenorm_layer_idx,
            total_layer_num=total_layer_num,
            num_nextn_predict_layers=getattr(config, "num_nextn_predict_layers", 0),
            is_nextn=is_nextn,
        )
        self.layer_id = layer_id
        self.is_final_layer = layer_id == total_layer_num - 1

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Qwen2MoE all layers are sparse (include nextn layers)
        self.is_layer_sparse = True
        is_previous_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=True,
        )

        if self.is_layer_sparse:
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = WelmV4FusedRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = WelmV4FusedRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dump_this_layer = _welm_should_dump_layer(self.layer_id)
        if dump_this_layer:
            _welm_dump_tensor(f"model.layers.{self.layer_id}.__input__.0", hidden_states)
        residual_after_layernorm = (
            self.ppln and self.layer_id not in self.prenorm_layer_idx
        )
        use_fp32_ppln_residual = residual_after_layernorm
        if use_fp32_ppln_residual:
            hidden_states, _, residual = self.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=residual_after_layernorm,
                clone_fp32_out=True,
                output_dtype=self.input_layernorm.weight.dtype
                if hidden_states.dtype == torch.float32
                else hidden_states.dtype,
            )
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=residual_after_layernorm,
                output_dtype=self.input_layernorm.weight.dtype
                if hidden_states.dtype == torch.float32
                else hidden_states.dtype,
            )
        if dump_this_layer:
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.input_layernorm.0", hidden_states
            )
            if residual is not None:
                _welm_dump_tensor(f"model.layers.{self.layer_id}.attn.mixer.1", residual)

        if residual_after_layernorm and not use_fp32_ppln_residual:
            residual = hidden_states.clone().to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        use_mmq_norm_after_attn = use_fp32_ppln_residual and self.self_attn.use_o_norm
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                skip_o_norm=use_mmq_norm_after_attn,
            )
        if (
            _welm_should_contract_kv_mirror(forward_batch)
            and self.layer_id == self.kv_mirror_layers[-1]
        ):
            residual = residual[forward_batch.custom_last_index]
        if use_mmq_norm_after_attn:
            hidden_states, residual, hidden_states_fp32 = mmq_style_norm_after_attn(
                hidden_states,
                residual,
                self.self_attn.o_norm.weight,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )
        else:
            hidden_states, residual, hidden_states_fp32 = self.post_attention_layernorm(
                hidden_states, residual, clone_fp32_out=True
            )
        if dump_this_layer:
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.norm_after_attn.output", hidden_states
            )
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.norm_after_attn.output_fp32",
                hidden_states_fp32,
            )
            if residual is not None:
                _welm_dump_tensor(
                    f"model.layers.{self.layer_id}.norm_after_attn.residual", residual
                )
        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        mlp_output = self.mlp(
            hidden_states,
            hidden_states_fp32,
            forward_batch,
            use_reduce_scatter,
            return_components=dump_this_layer or self.is_final_layer,
            skip_component_output=(
                self.is_final_layer
                and residual is not None
                and not dump_this_layer
                and getattr(self.mlp, "tp_size", 1) == 1
            ),
        )
        experts_output = None
        shared_output = None
        if isinstance(mlp_output, tuple):
            hidden_states, experts_output, shared_output = mlp_output
        else:
            hidden_states = mlp_output
        if dump_this_layer:
            output_with_residual = hidden_states
            if residual is not None and experts_output is not None:
                output_with_residual = experts_output.float() + residual.float()
                if shared_output is not None:
                    output_with_residual = output_with_residual + shared_output.float()
            _welm_dump_tensor(
                f"model.layers.{self.layer_id}.mlp.output_with_residual",
                output_with_residual,
            )
        can_add_final_components = (
            residual is not None
            and self.is_final_layer
            and experts_output is not None
            and getattr(self.mlp, "tp_size", 1) == 1
        )
        can_add_first_residual = getattr(self.mlp, "tp_size", 1) == 1
        if can_add_final_components:
            hidden_states = experts_output.float() + residual.float()
            if shared_output is not None:
                hidden_states = hidden_states + shared_output.float()
            residual = None
        elif residual is not None and can_add_first_residual:
            hidden_states = mmq_style_add_residual(
                hidden_states.contiguous(), residual.contiguous()
            )
            residual = None
        return hidden_states, residual


class Qwen2MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2MoeDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        self.oe_dim = config.oe_dim
        self.oe_grams = config.oe_grams
        self.oe_vocab_sizes = config.oe_vocab_sizes
        self.scale_seq_times = getattr(config, "scale_seq_times", 0)

        if len(self.oe_vocab_sizes) > 0:
            self.oe_embed = nn.ModuleList(
                [
                    VocabParallelEmbedding(
                        self.oe_vocab_sizes[i],
                        self.oe_dim,
                    )
                    for i in range(len(self.oe_vocab_sizes))
                ]
            )
            self.oe_gate_up_proj = ReplicatedLinear(
                self.oe_dim * len(self.oe_vocab_sizes),
                config.hidden_size,
                bias=False,
                quant_config=None,
            )

        # Scale sequence length embeddings: N additional embedding groups
        if self.scale_seq_times > 0:
            self.scale_seq_embed_tokens_list = nn.ModuleList(
                [
                    VocabParallelEmbedding(
                        config.vocab_size,
                        config.hidden_size,
                        enable_tp=not is_dp_attention_enabled(),
                    )
                    for _ in range(self.scale_seq_times)
                ]
            )
            if len(self.oe_vocab_sizes) > 0:
                self.scale_seq_oe_embed_list = nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                VocabParallelEmbedding(
                                    self.oe_vocab_sizes[j], self.oe_dim
                                )
                                for j in range(len(self.oe_vocab_sizes))
                            ]
                        )
                        for _ in range(self.scale_seq_times)
                    ]
                )
                self.scale_seq_oe_up_proj_list = nn.ModuleList(
                    [
                        ReplicatedLinear(
                            self.oe_dim * len(self.oe_vocab_sizes),
                            config.hidden_size,
                            bias=False,
                            quant_config=None,
                        )
                        for _ in range(self.scale_seq_times)
                    ]
                )

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2MoeDecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2MoeDecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    def _compute_oe_embedding(
        self,
        input_ids,
        forward_batch,
        base_hidden_states,
        oe_embed_modules=None,
        oe_up_proj_module=None,
    ):
        """Compute over-encoding embedding and combine with base hidden states.
        If oe_embed_modules/oe_up_proj_module are None, use the main OE modules."""
        if oe_embed_modules is None:
            oe_embed_modules = self.oe_embed
        if oe_up_proj_module is None:
            oe_up_proj_module = self.oe_gate_up_proj
        return compute_welm_oe_embedding(
            input_ids=input_ids,
            forward_batch=forward_batch,
            base_hidden_states=base_hidden_states,
            oe_grams=self.oe_grams,
            oe_vocab_sizes=self.oe_vocab_sizes,
            vocab_size=self.vocab_size,
            oe_embed_modules=oe_embed_modules,
            oe_proj_module=oe_up_proj_module,
        )

    def _expand_scale_seq(self, input_ids, forward_batch, hidden_states):
        """Expand hidden_states from (T, D) to (T * scale, D) by interleaving
        main embedding with scale_seq embeddings.

        Layout per original token i:
          [main_emb_i, scale_seq_1_emb_i, ..., scale_seq_N_emb_i]
        """
        scale = self.scale_seq_times + 1
        T = hidden_states.shape[0]
        D = hidden_states.shape[1]

        # (T, D) -> (T, 1, D)
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states_list = [hidden_states]

        for s in range(self.scale_seq_times):
            hs_s = self.scale_seq_embed_tokens_list[s](input_ids)  # (T, D)
            if len(self.oe_grams) > 0:
                hs_s = self._compute_oe_embedding(
                    input_ids,
                    forward_batch,
                    hs_s,
                    oe_embed_modules=self.scale_seq_oe_embed_list[s],
                    oe_up_proj_module=self.scale_seq_oe_up_proj_list[s],
                )
            hs_s = hs_s.unsqueeze(1)  # (T, 1, D)
            hidden_states_list.append(hs_s)

        # (T, scale, D) -> (T * scale, D)
        hidden_states = torch.cat(hidden_states_list, dim=1)
        hidden_states = hidden_states.reshape(T * scale, D).contiguous()
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds

            if len(self.oe_grams) > 0:
                hidden_states = self._compute_oe_embedding(
                    input_ids, forward_batch, hidden_states
                )

            if self.scale_seq_times > 0:
                hidden_states = self._expand_scale_seq(
                    input_ids, forward_batch, hidden_states
                )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors.tensors.get("residual", None)

        aux_hidden_states = []
        if forward_batch.can_run_tbo:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(
                        hidden_states + residual
                        if residual is not None
                        else hidden_states
                    )
                with get_global_expert_distribution_recorder().with_current_layer(i):
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions, hidden_states, forward_batch, residual
                    )
        if not self.pp_group.is_last_rank:
            proxy_tensors = {"hidden_states": hidden_states}
            if residual is not None:
                proxy_tensors["residual"] = residual
            return PPProxyTensors(proxy_tensors)
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    if _welm_dump_enabled():
                        _welm_dump_tensor("model.ln_f.__input__.0", hidden_states)
                    if hidden_states.dtype == torch.float32:
                        hidden_states = F.rms_norm(
                            hidden_states.to(self.norm.weight.dtype),
                            self.norm.weight.shape,
                            self.norm.weight,
                            eps=self.norm.variance_epsilon,
                        )
                    else:
                        hidden_states = self.norm(hidden_states)
                else:
                    last_layer = self.layers[self.end_layer - 1]
                    final_experts_output = getattr(
                        last_layer, "final_mlp_experts_output", None
                    )
                    final_shared_output = getattr(
                        last_layer, "final_mlp_shared_output", None
                    )
                    # In TP>1, component tensors are still pre-all-reduce.
                    can_rebuild_final_mlp = (
                        final_experts_output is not None
                        and getattr(last_layer.mlp, "tp_size", 1) == 1
                    )
                    if can_rebuild_final_mlp:
                        hidden_states = final_experts_output.float() + residual.float()
                        if final_shared_output is not None:
                            hidden_states = hidden_states + final_shared_output.float()
                    else:
                        hidden_states = hidden_states.float() + residual.float()
                    if _welm_dump_enabled():
                        _welm_dump_tensor("model.ln_f.__input__.0", hidden_states)
                    hidden_states = F.rms_norm(
                        hidden_states.to(self.norm.weight.dtype),
                        self.norm.weight.shape,
                        self.norm.weight,
                        eps=self.norm.variance_epsilon,
                    )
                if _welm_dump_enabled():
                    _welm_dump_tensor("model.ln_f", hidden_states)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class WeLMV4MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self.model = Qwen2MoeModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            alt_stream=alt_stream,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if self.pp_group.is_last_rank:
            # Contract expanded hidden_states back to logical size for logits.
            # Transformer layers have already processed all T*scale states and
            # written KV cache.  For logits we only need the last state in each
            # scale group (matches MMQ's [:, -1, :] semantic).
            if self.model.scale_seq_times > 0:
                scale = self.model.scale_seq_times + 1
                # Select every scale-th element (last of each group)
                kv_mirror_contracted = (
                    forward_batch.enable_welm_kv_mirror_opt
                    and forward_batch.forward_mode.is_extend_without_speculative()
                )
                if not kv_mirror_contracted:
                    indices = torch.arange(
                        scale - 1,
                        hidden_states.shape[0],
                        scale,
                        device=hidden_states.device,
                    )
                    hidden_states = hidden_states[indices]

                # Restore forward_batch metadata to logical space so that
                # LogitsProcessor sees the un-expanded lengths.
                if forward_batch.extend_seq_lens is not None:
                    forward_batch.extend_seq_lens = (
                        forward_batch.extend_seq_lens // scale
                    )
                    if forward_batch.extend_seq_lens_cpu is not None:
                        forward_batch.extend_seq_lens_cpu = [
                            x // scale for x in forward_batch.extend_seq_lens_cpu
                        ]
                    forward_batch.extend_num_tokens = (
                        forward_batch.extend_num_tokens // scale
                    )

            logits_output = self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
            logits_output.model_specific_states = forward_batch.model_specific_states
            return logits_output
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

            if len(self.model.oe_grams) > 0:
                forward_batch.hidden_states = self.model._compute_oe_embedding(
                    input_ids, forward_batch, forward_batch.hidden_states
                )

            if self.model.scale_seq_times > 0:
                forward_batch.hidden_states = self.model._expand_scale_seq(
                    input_ids, forward_batch, forward_batch.hidden_states
                )

        # decoder layer
        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states

            # Contract expanded hidden_states back to logical size
            if self.model.scale_seq_times > 0:
                scale = self.model.scale_seq_times + 1
                kv_mirror_contracted = (
                    forward_batch.enable_welm_kv_mirror_opt
                    and forward_batch.forward_mode.is_extend_without_speculative()
                )
                if not kv_mirror_contracted:
                    indices = torch.arange(
                        scale - 1,
                        hidden_states.shape[0],
                        scale,
                        device=hidden_states.device,
                    )
                    forward_batch.hidden_states = hidden_states[indices]
                else:
                    forward_batch.hidden_states = hidden_states
                if forward_batch.extend_seq_lens is not None:
                    forward_batch.extend_seq_lens = (
                        forward_batch.extend_seq_lens // scale
                    )
                    if forward_batch.extend_seq_lens_cpu is not None:
                        forward_batch.extend_seq_lens_cpu = [
                            x // scale for x in forward_batch.extend_seq_lens_cpu
                        ]
                    forward_batch.extend_num_tokens = (
                        forward_batch.extend_num_tokens // scale
                    )

            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
            result.model_specific_states = forward_batch.model_specific_states
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                num_target_layers = self.config.num_hidden_layers
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())
        attn_by_kv_mirror_layer_idx = {
            module.kv_mirror_layer_idx: module
            for module in modules_dict.values()
            if isinstance(module, Qwen2MoeAttention)
        }
        if is_nextn:
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        for name, loaded_weight in weights:
            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                name_list = name.split(".")
                if len(name_list) < 3 or name_list[0] != "model" or name_list[1] != "layers":
                    continue
                layer_idx = int(name_list[2])
                if not (num_target_layers <= layer_idx < num_target_layers + num_nextn_layers):
                    continue
                # if not name.startswith(nextn_layer_prefix):
                #     continue
                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = ".".join(["model", *name_list[3:]])
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    name = ".".join(
                        [
                            "model",
                            "decoder_layers",
                            str(layer_idx - num_target_layers),
                            *name_list[3:],
                        ]
                    )
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "self_attn.gate_proj" in name:
                    continue
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                if weight_name == "up_proj" and "scale_seq_oe_up_proj" in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                if name == "model.oe_gate_up_proj.weight":
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                attn_prefix = name.split(".qkv_proj.")[0] if ".qkv_proj." in name else None
                current_attn = (
                    modules_dict.get(attn_prefix)
                    if attn_prefix is not None
                    else None
                )
                effective_shard_id = shard_id
                if (
                    current_attn is not None
                    and isinstance(current_attn, Qwen2MoeAttention)
                    and isinstance(current_attn.qkv_proj, ImitateQkvKvProjection)
                    and shard_id in ("k", "v")
                ):
                    effective_shard_id = "k1" if shard_id == "k" else "v1"
                if not (
                    current_attn is not None
                    and isinstance(current_attn, Qwen2MoeAttention)
                    and isinstance(current_attn.qkv_proj, MirrorQProjection)
                    and shard_id in ("k", "v")
                ):
                    weight_loader(param, loaded_weight, effective_shard_id)
                if (
                    current_attn is not None
                    and isinstance(current_attn, Qwen2MoeAttention)
                    and isinstance(current_attn.qkv_proj, MirrorQProjection)
                    and shard_id in ("k", "v")
                ):
                    pair_attn = attn_by_kv_mirror_layer_idx.get(
                        current_attn.qkv_proj.imitated_layer_idx
                    )
                    if (
                        pair_attn is not None
                        and isinstance(pair_attn.qkv_proj, ImitateQkvKvProjection)
                    ):
                        pair_param = (
                            pair_attn.qkv_proj.bias
                            if name.endswith(".bias")
                            else pair_attn.qkv_proj.weight
                        )
                        if pair_param is not None:
                            pair_param.weight_loader(
                                pair_param,
                                loaded_weight,
                                "k2" if shard_id == "k" else "v2",
                            )
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        if "attn_sink" in name:
                            start = get_attention_tp_rank() * param.numel()
                            param.data.copy_(
                                loaded_weight[start : start + param.numel()]
                            )
                        else:
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return [
            self.model.embed_tokens,
            self.model.oe_embed,
            self.model.oe_gate_up_proj,
        ], self.lm_head

    def set_embed_and_head(self, embed, head):
        self.model.embed_tokens = embed[0]
        self.model.oe_embed = embed[1]
        self.model.oe_gate_up_proj = embed[2]
        self.lm_head = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.set_eagle3_layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = WeLMV4MoeForCausalLM
