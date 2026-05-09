"""Inference-only WeLMV4 NextN Speculative Decoding."""

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.welmv4 import (
    Qwen2MoeDecoderLayer,
    WelmV4FusedRMSNorm,
    WeLMV4MoeForCausalLM,
    hash_input_ids_vectorized,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_npu

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_npu = is_npu()
_MTP_DUMP_PASS = 0
_MTP_DUMP_WRITTEN = set()


def _mtp_dump_enabled() -> bool:
    return os.environ.get("SGLANG_DUMP_MTP_ACTIVATIONS", "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _mtp_dump_dir() -> Path:
    root = os.environ.get("SGLANG_DUMP_MTP_ACTIVATIONS_DIR", "./sglang_mtp_dump")
    rank = os.environ.get("RANK", "0")
    path = Path(root) / f"Rank{rank}_pid{os.getpid()}" / f"Pass{_MTP_DUMP_PASS:05d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _dump_tensor(name: str, value) -> None:
    if not _mtp_dump_enabled() or name in _MTP_DUMP_WRITTEN:
        return
    if isinstance(value, torch.Tensor):
        torch.save(value.detach().cpu(), _mtp_dump_dir() / f"{_safe_name(name)}.pt")
        _MTP_DUMP_WRITTEN.add(name)


class WeLMV4ModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = None
        self.oe_embed = None
        self.oe_gate_up_proj = None
        self.oe_dim = config.oe_dim
        self.oe_grams = config.oe_grams
        self.oe_vocab_sizes = config.oe_vocab_sizes

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

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True)

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None

        layer_name = "decoder"
        if _is_npu and (
            get_global_server_args().speculative_draft_model_path
            == get_global_server_args().model_path
        ):
            layer_name = "layers." + str(config.num_hidden_layers)

        self.decoder_layers = nn.ModuleList(
            [
                Qwen2MoeDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    is_nextn=True,
                    prefix=add_prefix(layer_name, prefix),
                    alt_stream=self.alt_stream,
                )
                for i in range(config.num_nextn_predict_layers)
            ]
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = WelmV4FusedRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:

        _dump_tensor("model.mtp.0.input_ids", input_ids)
        _dump_tensor("model.mtp.0.positions", positions)
        _dump_tensor("model.mtp.0.main_hidden_in", forward_batch.spec_info.hidden_states)

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if len(self.oe_grams) > 0:
            input_ids_ngram = []
            input_ids_ngram_tmp = input_ids
            max_n = max(self.oe_grams)
            if getattr(forward_batch, "n_gram_input_ids", None) is not None:
                input_ids_gram_n = []
                for n in range(2, max_n + 1):
                    gram = forward_batch.n_gram_input_ids.get_gram(n)
                    input_ids_gram_n.append(
                        gram if gram is not None else torch.zeros_like(input_ids)
                    )
            else:
                zero_ids = torch.zeros_like(input_ids)
                input_ids_gram_n = [zero_ids for _ in range(max_n - 1)]
            for g in range(1, max_n):
                input_ids_ngram_tmp = input_ids_ngram_tmp + input_ids_gram_n[g - 1] * (
                    self.vocab_size**g
                )
                input_ids_ngram.append(hash_input_ids_vectorized(input_ids_ngram_tmp))

            emb_ngram = []
            for i, vs in enumerate(self.oe_vocab_sizes):
                input_ids_ngram_hashed_tmp = input_ids_ngram[self.oe_grams[i] - 2] % vs
                emb_ngram_tmp = self.oe_embed[i](input_ids_ngram_hashed_tmp)
                emb_ngram.append(emb_ngram_tmp)
            emb_new, _ = self.oe_gate_up_proj(torch.cat(emb_ngram, dim=-1))
            hidden_states = (hidden_states + emb_new) / 2.0

        _dump_tensor("model.mtp.0.embedding", hidden_states)

        if (
            forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            main_hidden_states = forward_batch.spec_info.hidden_states
            if (
                main_hidden_states is not None
                and hidden_states.shape[0] != main_hidden_states.shape[0]
            ):
                if not hasattr(forward_batch, "custom_last_index"):
                    forward_batch.custom_last_index = (
                        torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                    )
                hidden_states = hidden_states[forward_batch.custom_last_index]

        if hidden_states.shape[0] > 0:
            enorm_output = self.enorm(hidden_states)
            hnorm_output = self.hnorm(forward_batch.spec_info.hidden_states)
            _dump_tensor("model.mtp.0.enorm", enorm_output)
            _dump_tensor("model.mtp.0.hnorm", hnorm_output)
            hidden_states = self.eh_proj(
                torch.cat((enorm_output, hnorm_output), dim=-1)
            )
            _dump_tensor("model.mtp.0.projector_out", hidden_states)

        residual = None
        final_experts_output = None
        final_shared_output = None
        with get_global_expert_distribution_recorder().disable_this_region():
            for layer_idx, layer in enumerate(self.decoder_layers):
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )
                final_experts_output = getattr(layer, "final_mlp_experts_output", None)
                final_shared_output = getattr(layer, "final_mlp_shared_output", None)
                _dump_tensor(f"model.mtp.0.decoder.{layer_idx}.hidden", hidden_states)
                _dump_tensor(f"model.mtp.0.decoder.{layer_idx}.residual", residual)

        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                if final_experts_output is not None:
                    hidden_states = final_experts_output.float() + residual.float()
                    if final_shared_output is not None:
                        hidden_states = hidden_states + final_shared_output.float()
                else:
                    hidden_states = hidden_states.float() + residual.float()
                hidden_states = hidden_states.to(self.shared_head.norm.weight.dtype)
                _dump_tensor("model.mtp.0.decoder.0.output", hidden_states)
                hidden_states, _ = self.shared_head.norm(hidden_states)
            else:
                hidden_states, _ = self.shared_head.norm(hidden_states)
        _dump_tensor("model.mtp.0.ln_f", hidden_states)
        return hidden_states


class WeLMV4MoeForCausalLMNextN(WeLMV4MoeForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        # if not set, model load will be broken in DeepseekV3ForCausalLM load_weights()
        self.pp_group = get_pp_group()

        self.model = WeLMV4ModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = None
        self.logits_processor = LogitsProcessor(config)
        super().post_init_after_load_weights(is_nextn=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        logits_output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        _dump_tensor("model.mtp.0.logits", logits_output.next_token_logits)
        return logits_output

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)


EntryClass = WeLMV4MoeForCausalLMNextN
