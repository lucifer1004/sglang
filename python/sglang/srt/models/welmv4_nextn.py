"""Inference-only WeLMV4 NextN Speculative Decoding."""

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.welm_perf_opt import (
    compute_welm_oe_embedding,
    hash_input_ids_vectorized,
    should_use_welm_oe_hash_kernel,
)
import sglang.srt.models.welmv4 as welmv4_module
from sglang.srt.models.welmv4 import (
    Qwen2MoeDecoderLayer,
    WelmV4FusedRMSNorm,
    WeLMV4MoeForCausalLM,
    _get_welm_kv_mirror_states,
    _welm_init_kv_mirror_last_q_indices,
    _welm_prepare_kv_mirror_logits_states,
    _welm_select_kv_mirror_rows,
    _welm_should_contract_kv_mirror,
    welm_use_previous_precision,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_npu

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_npu = is_npu()
_MTP_DUMP_PASS = 0
_MTP_DUMP_PASS_ACTIVE = False
_MTP_GRAPH_DUMP_PASS_ACTIVE = False
_MTP_DUMP_WRITTEN = set()
_MTP_GRAPH_DUMP_CALL_INDEX = 0
_MTP_GRAPH_DUMP_CURRENT_PREFIX = None
_MTP_GRAPH_DUMP_PREV_NAME_PREFIX = ""
_MTP_DUMP_CONTEXT_ENV = "SGLANG_DUMP_MTP_ACTIVATIONS_CONTEXT"
_MTP_TRUE_ENV_VALUES = {"1", "true", "on", "yes"}
_MTP_DUMP_ENABLED = (
    os.environ.get("SGLANG_DUMP_MTP_ACTIVATIONS", "0").strip().lower()
    in _MTP_TRUE_ENV_VALUES
)


def _reset_mtp_graph_dump_call_index() -> None:
    if not _MTP_DUMP_ENABLED:
        return
    global _MTP_GRAPH_DUMP_CALL_INDEX, _MTP_GRAPH_DUMP_CURRENT_PREFIX
    global _MTP_GRAPH_DUMP_PREV_NAME_PREFIX
    _MTP_GRAPH_DUMP_CALL_INDEX = 0
    _MTP_GRAPH_DUMP_CURRENT_PREFIX = None
    _MTP_GRAPH_DUMP_PREV_NAME_PREFIX = ""
    welmv4_module._welm_set_graph_dump_name_prefix(None)


def _cuda_graph_capture_active() -> bool:
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if get_is_capture_mode():
            return True
    except Exception:
        pass
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def _mtp_dump_enabled() -> bool:
    return _MTP_DUMP_ENABLED


def _update_mtp_kv_mirror_dp_metadata(
    forward_batch: ForwardBatch, new_local_num_tokens: int
) -> None:
    if (
        not is_dp_attention_enabled()
        or welm_use_previous_precision()
        or getattr(forward_batch, "global_num_tokens_gpu", None) is None
    ):
        return

    from sglang.srt.layers.dp_attention import (
        get_attention_dp_rank,
        set_dp_buffer_len,
    )

    dp_rank = get_attention_dp_rank()
    scale = max(getattr(forward_batch, "scale_seq_factor", 1), 1)
    if scale > 1:
        new_global_num_tokens_gpu = forward_batch.global_num_tokens_gpu // scale
        forward_batch.global_num_tokens_gpu.copy_(new_global_num_tokens_gpu)
        new_global_num_tokens = [int(x) for x in new_global_num_tokens_gpu.tolist()]
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.global_num_tokens_cpu = new_global_num_tokens
    else:
        forward_batch.global_num_tokens_gpu[dp_rank] = new_local_num_tokens
        new_global_num_tokens = None

    forward_batch.dp_local_start_pos = None
    forward_batch.dp_local_num_tokens = None
    if new_global_num_tokens is not None:
        if forward_batch.dp_padding_mode.is_max_len():
            global_dp_buffer_len = max(new_global_num_tokens) * len(
                new_global_num_tokens
            )
        else:
            global_dp_buffer_len = sum(new_global_num_tokens)
        forward_batch.global_dp_buffer_len = global_dp_buffer_len
    else:
        global_dp_buffer_len = forward_batch.global_dp_buffer_len
    set_dp_buffer_len(
        global_dp_buffer_len,
        new_local_num_tokens,
        forward_batch.dp_padding_mode.is_max_len(),
        new_global_num_tokens,
    )


def _mtp_dump_dir() -> Path:
    root = os.environ.get("SGLANG_DUMP_MTP_ACTIVATIONS_DIR", "./sglang_mtp_dump")
    rank = os.environ.get("RANK", "0")
    path = Path(root) / f"Rank{rank}_pid{os.getpid()}" / f"Pass{_MTP_DUMP_PASS:05d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _current_mtp_dump_context() -> str:
    return os.environ.get(_MTP_DUMP_CONTEXT_ENV, "").strip()


def _write_mtp_pass_metadata(
    dump_dir: Path,
    *,
    pass_id: int,
    context: str,
    graph_step: Optional[str] = None,
    first_dim_limit: Optional[int] = None,
) -> None:
    if not _MTP_DUMP_ENABLED:
        return
    metadata = {
        "pass_id": pass_id,
        "context": context,
        "graph_step": graph_step,
        "first_dim_limit": first_dim_limit,
        "pid": os.getpid(),
        "rank": os.environ.get("RANK", "0"),
    }
    torch.save(metadata, dump_dir / "mtp_pass_metadata.pt")


def _start_mtp_dump_pass() -> None:
    if not _MTP_DUMP_ENABLED:
        return
    global _MTP_DUMP_PASS_ACTIVE, _MTP_GRAPH_DUMP_PASS_ACTIVE
    global _MTP_GRAPH_DUMP_CALL_INDEX, _MTP_GRAPH_DUMP_CURRENT_PREFIX
    global _MTP_GRAPH_DUMP_PREV_NAME_PREFIX
    capture_active = _cuda_graph_capture_active()
    _MTP_DUMP_PASS_ACTIVE = not capture_active
    _MTP_GRAPH_DUMP_PASS_ACTIVE = capture_active
    if _MTP_GRAPH_DUMP_PASS_ACTIVE:
        _MTP_GRAPH_DUMP_CURRENT_PREFIX = (
            f"graph_step_{_MTP_GRAPH_DUMP_CALL_INDEX:05d}."
        )
        _MTP_GRAPH_DUMP_CALL_INDEX += 1
        _MTP_GRAPH_DUMP_PREV_NAME_PREFIX = (
            welmv4_module._welm_set_graph_dump_name_prefix(
                _MTP_GRAPH_DUMP_CURRENT_PREFIX
            )
        )
    else:
        _MTP_GRAPH_DUMP_CURRENT_PREFIX = None
        _MTP_GRAPH_DUMP_PREV_NAME_PREFIX = ""
    if _MTP_DUMP_PASS_ACTIVE or _MTP_GRAPH_DUMP_PASS_ACTIVE:
        _MTP_DUMP_WRITTEN.clear()
        dump_dir = _mtp_dump_dir() if _MTP_DUMP_PASS_ACTIVE else None
        if dump_dir is not None:
            _write_mtp_pass_metadata(
                dump_dir,
                pass_id=_MTP_DUMP_PASS,
                context=_current_mtp_dump_context(),
            )
            os.environ["SGLANG_DUMP_MTP_ACTIVATIONS_PROCESS_DIR"] = str(dump_dir)


def _finish_mtp_dump_pass() -> None:
    if not _MTP_DUMP_ENABLED:
        return
    global _MTP_DUMP_PASS, _MTP_DUMP_PASS_ACTIVE, _MTP_GRAPH_DUMP_PASS_ACTIVE
    global _MTP_GRAPH_DUMP_CURRENT_PREFIX
    global _MTP_GRAPH_DUMP_PREV_NAME_PREFIX
    if _MTP_DUMP_PASS_ACTIVE or _MTP_GRAPH_DUMP_PASS_ACTIVE:
        if _MTP_GRAPH_DUMP_PASS_ACTIVE:
            welmv4_module._welm_set_graph_dump_name_prefix(
                _MTP_GRAPH_DUMP_PREV_NAME_PREFIX
            )
        if _MTP_DUMP_PASS_ACTIVE:
            _MTP_DUMP_PASS += 1
        _MTP_DUMP_WRITTEN.clear()
        os.environ.pop("SGLANG_DUMP_MTP_ACTIVATIONS_PROCESS_DIR", None)
    _MTP_DUMP_PASS_ACTIVE = False
    _MTP_GRAPH_DUMP_PASS_ACTIVE = False
    _MTP_GRAPH_DUMP_CURRENT_PREFIX = None
    _MTP_GRAPH_DUMP_PREV_NAME_PREFIX = ""


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _dump_tensor(name: str, value) -> None:
    if not _MTP_DUMP_ENABLED:
        return
    if isinstance(value, torch.Tensor):
        safe_name = _safe_name(name)
        capture_active = _cuda_graph_capture_active()
        if not capture_active and name in _MTP_DUMP_WRITTEN:
            return
        if capture_active:
            welmv4_module._welm_graph_dump_tensor(safe_name, value)
            return
        torch.save(value.detach().cpu(), _mtp_dump_dir() / f"{safe_name}.pt")
        _MTP_DUMP_WRITTEN.add(name)


def _flush_mtp_graph_dump_pass(
    context: str,
    first_dim_limit: Optional[int] = None,
) -> list[Path]:
    global _MTP_DUMP_PASS
    if not _MTP_DUMP_ENABLED:
        return []
    dump_dir = _mtp_dump_dir()
    saved = welmv4_module._welm_flush_graph_dump_buffers(
        context, dump_dir, first_dim_limit=first_dim_limit
    )
    if not saved:
        return []

    graph_step_files = []
    for path in dump_dir.glob("graph_step_*.pt"):
        stem = path.name[: -len(".pt")]
        prefix, sep, rest = stem.partition(".")
        if not sep or not prefix.startswith("graph_step_") or not rest:
            continue
        graph_step_files.append((prefix, rest, path))

    if not graph_step_files:
        _write_mtp_pass_metadata(
            dump_dir,
            pass_id=_MTP_DUMP_PASS,
            context=context,
            first_dim_limit=first_dim_limit,
        )
        _MTP_DUMP_PASS += 1
        return [dump_dir]

    written_dirs = []
    step_order = {
        step: idx for idx, step in enumerate(sorted({x[0] for x in graph_step_files}))
    }
    for step, rest, path in graph_step_files:
        pass_id = _MTP_DUMP_PASS + step_order[step]
        step_dir = dump_dir.parent / f"Pass{pass_id:05d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        if step_dir not in written_dirs:
            written_dirs.append(step_dir)
        path.replace(step_dir / f"{rest}.pt")
        _write_mtp_pass_metadata(
            step_dir,
            pass_id=pass_id,
            context=context,
            graph_step=step,
            first_dim_limit=first_dim_limit,
        )
    _MTP_DUMP_PASS += len(step_order)
    return written_dirs


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
                        use_attn_tp_group=is_dp_attention_enabled(),
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
        self.shared_head.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if welm_use_previous_precision()
            else WelmV4FusedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        dump_enabled = _MTP_DUMP_ENABLED
        main_hidden_states = forward_batch.spec_info.hidden_states
        if dump_enabled:
            _dump_tensor("model.mtp.0.input_ids", input_ids)
            _dump_tensor("model.mtp.0.positions", positions)
            _dump_tensor("model.mtp.0.main_hidden_in", main_hidden_states)

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        use_fused_hash = (
            len(self.oe_grams) > 0
            and not dump_enabled
            and should_use_welm_oe_hash_kernel()
        )
        if use_fused_hash and input_ids.numel() > 0:
            if getattr(forward_batch, "welm_oe_decode_hashed_inputs", None) is None:
                raise RuntimeError(
                    "WeLMV4 MTP fused OE hash path is enabled but cached hash "
                    "inputs are missing."
                )
            hidden_states = compute_welm_oe_embedding(
                input_ids=input_ids,
                forward_batch=forward_batch,
                base_hidden_states=hidden_states,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=self.oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=self.oe_embed,
                oe_proj_module=self.oe_gate_up_proj,
            )
        elif len(self.oe_grams) > 0:
            input_ids_ngram = []
            input_ids_ngram_tmp = input_ids
            max_n = max(self.oe_grams)
            if getattr(forward_batch, "oe_context", None) is not None:
                input_ids_gram_n = []
                for n in range(2, max_n + 1):
                    gram = forward_batch.oe_context.get_gram(n)
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

        if (
            _welm_should_contract_kv_mirror(forward_batch)
            and main_hidden_states is not None
        ):
            _welm_init_kv_mirror_last_q_indices(forward_batch)
            first_contract = (
                hidden_states.shape[0] != forward_batch.kv_mirror_output_size
            )
            hidden_states = _welm_select_kv_mirror_rows(
                hidden_states, forward_batch, first_contract=first_contract
            )
            main_first_contract = (
                main_hidden_states.shape[0] != forward_batch.kv_mirror_output_size
            )
            main_hidden_states = _welm_select_kv_mirror_rows(
                main_hidden_states,
                forward_batch,
                first_contract=main_first_contract,
            )
            if first_contract:
                _update_mtp_kv_mirror_dp_metadata(
                    forward_batch, hidden_states.shape[0]
                )

        needs_empty_dp_collectives = (
            is_dp_attention_enabled()
            and getattr(forward_batch, "global_num_tokens_gpu", None) is not None
        )
        if hidden_states.shape[0] == 0 and not needs_empty_dp_collectives:
            # KV-mirror contraction can select no active rows for this draft
            # extend chunk. Avoid running zero-token decoder kernels; the caller
            # scatters this empty result back to the logical batch shape.
            return hidden_states, hidden_states

        if hidden_states.shape[0] > 0:
            enorm_output = self.enorm(hidden_states)
            hnorm_output = self.hnorm(main_hidden_states)
            hidden_states = self.eh_proj(
                torch.cat((enorm_output, hnorm_output), dim=-1)
            )
            if dump_enabled:
                _dump_tensor("model.mtp.0.projector_out", hidden_states)

        residual = None
        kv_mirror_states = _get_welm_kv_mirror_states(forward_batch)
        final_experts_output = None
        final_shared_output = None
        with get_global_expert_distribution_recorder().disable_this_region():
            for layer_idx, layer in enumerate(self.decoder_layers):
                hidden_states, residual, kv_mirror_states = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    kv_mirror_states,
                )
                final_experts_output = getattr(layer, "final_mlp_experts_output", None)
                final_shared_output = getattr(layer, "final_mlp_shared_output", None)

        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states

        hidden_states_for_next_mtp = None
        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                if welm_use_previous_precision():
                    hidden_states_for_next_mtp = (
                        hidden_states.float() + residual.float()
                    ).to(self.shared_head.norm.weight.dtype)
                    hidden_states, _ = self.shared_head.norm(hidden_states, residual)
                else:
                    final_layer = self.decoder_layers[-1]
                    can_rebuild_final_mlp = (
                        final_experts_output is not None
                        and getattr(final_layer.mlp, "tp_size", 1) == 1
                        and not is_dp_attention_enabled()
                    )
                    if can_rebuild_final_mlp:
                        hidden_states = final_experts_output.float() + residual.float()
                        if final_shared_output is not None:
                            hidden_states = hidden_states + final_shared_output.float()
                    else:
                        hidden_states = hidden_states.float() + residual.float()
                if hidden_states_for_next_mtp is None:
                    # MMQ feeds the MTP layer output before apply_ln_f into the next
                    # recursive MTP step; shared_head.norm is only for logits.
                    hidden_states_for_next_mtp = hidden_states.to(
                        self.shared_head.norm.weight.dtype
                    )
                    hidden_states, _ = self.shared_head.norm(hidden_states_for_next_mtp)
            else:
                hidden_states_for_next_mtp = hidden_states.to(
                    self.shared_head.norm.weight.dtype
                )
                norm_output = self.shared_head.norm(hidden_states_for_next_mtp)
                hidden_states = (
                    norm_output[0] if isinstance(norm_output, tuple) else norm_output
                )
            if dump_enabled:
                _dump_tensor(
                    "model.mtp.0.decoder.0.output", hidden_states_for_next_mtp
                )
        if dump_enabled:
            _dump_tensor("model.mtp.0.ln_f", hidden_states)
        return hidden_states, hidden_states_for_next_mtp


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

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if _MTP_DUMP_ENABLED:
            _start_mtp_dump_pass()
        try:
            hidden_states, hidden_states_for_next_mtp = self.model(
                input_ids, positions, forward_batch
            )
            aux_hidden_states = (
                [hidden_states_for_next_mtp]
                if hidden_states_for_next_mtp is not None
                else None
            )
            if _welm_should_contract_kv_mirror(forward_batch):
                hidden_states, aux_hidden_states = _welm_prepare_kv_mirror_logits_states(
                    hidden_states, aux_hidden_states, forward_batch
                )
            logits_output = self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
            if _MTP_DUMP_ENABLED:
                _dump_tensor("model.mtp.0.logits", logits_output.next_token_logits)
            return logits_output
        finally:
            if _MTP_DUMP_ENABLED:
                _finish_mtp_dump_pass()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)


EntryClass = WeLMV4MoeForCausalLMNextN
