from __future__ import annotations

import bisect
import contextlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.models.welm_perf_opt import get_welm_oe_hash_config
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_info_v2 import assign_draft_cache_locs_page_size_1
from sglang.srt.speculative.welmv4_mtp_sampling import welm_mtp_deterministic_uniforms
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)
from sglang.srt.utils.common import is_cuda, is_musa

_is_cuda = is_cuda()
_is_musa = is_musa()

if _is_cuda or _is_musa:
    from sgl_kernel import top_p_renorm_prob

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

_WELM_MTP_DRAFT_GRAPH_IO_DUMP = os.environ.get(
    "SGLANG_DUMP_WELM_MTP_DRAFT_GRAPH_IO", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_WELM_MTP_DRAFT_GRAPH_IO_COUNTERS = {}


def _to_cpu_payload(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu_payload(v) for v in value)
    return value


def _dump_welm_mtp_draft_graph_io(name: str, payload: dict) -> None:
    if not _WELM_MTP_DRAFT_GRAPH_IO_DUMP:
        return
    root = os.environ.get(
        "SGLANG_DUMP_WELM_MTP_DRAFT_GRAPH_IO_DIR",
        "/tmp/sglang_welm_mtp_draft_graph_io",
    )
    rank = os.environ.get("SGLANG_TP_RANK") or os.environ.get("RANK") or "0"
    dump_dir = os.path.join(root, f"pid{os.getpid()}_rank{rank}")
    os.makedirs(dump_dir, exist_ok=True)
    index = _WELM_MTP_DRAFT_GRAPH_IO_COUNTERS.get(name, 0)
    _WELM_MTP_DRAFT_GRAPH_IO_COUNTERS[name] = index + 1
    torch.save(
        {"name": name, "index": index, **_to_cpu_payload(payload)},
        os.path.join(dump_dir, f"{name}_{index:05d}.pt"),
    )


@dataclass
class WelmMTPDraftProposalInputBuffers(ForwardInputBuffers):
    input_ids: torch.Tensor
    first_input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor
    draft_out_cache_loc: torch.Tensor
    positions: torch.Tensor
    draft_positions: torch.Tensor
    mrope_positions: torch.Tensor
    hidden_states: torch.Tensor
    mirrored_kv_indices: Optional[torch.Tensor]
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    draft_seq_lens: torch.Tensor
    draft_seq_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    num_accepted_drafts: torch.Tensor
    num_accepted_tokens: torch.Tensor
    custom_last_index: torch.Tensor
    custom_last_cache_loc: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    temperature: torch.Tensor
    top_p: torch.Tensor
    uniform_samples: torch.Tensor
    welm_mtp_oe_hash_out: Optional[torch.Tensor]
    welm_mtp_query_hash_inputs: Optional[torch.Tensor]
    welm_mtp_oe_entry_history: Optional[torch.Tensor]
    welm_mtp_oe_work_history_a: Optional[torch.Tensor]
    welm_mtp_oe_work_history_b: Optional[torch.Tensor]
    welm_mtp_oe_parent_indices: Optional[torch.Tensor]
    welm_mtp_oe_prev_input_ids: Optional[torch.Tensor]
    welm_mtp_oe_prev_prev_input_ids: Optional[torch.Tensor]
    welm_mtp_oe_output_prev_input_ids: Optional[torch.Tensor]
    welm_mtp_oe_hash_out_batch_major: Optional[torch.Tensor]
    welm_mtp_draft_input_ids: Optional[torch.Tensor]
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]


class WelmMTPDraftProposalCudaGraphRunner:
    """CUDA graph runner for WeLM MTP merged extend-draft proposal generation.

    The graph layer captures a callable supplied by EagleDraftWorker. V1/V2
    semantic differences should stay inside that callable/program; this runner
    only owns static buffers, graph capture, replay, and output copying.
    """

    def __init__(
        self,
        eagle_worker: "EagleDraftWorker",
        *,
        draft_extend_attn_backend=None,
    ):
        self.eagle_worker = eagle_worker
        self.model_runner = model_runner = eagle_worker.draft_runner
        self.graphs = {}
        self.output_buffers = {}
        self.forward_batches = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = self.model_runner.dp_size
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.draft_extend_attn_backend = (
            draft_extend_attn_backend or eagle_worker.draft_extend_attn_backend
        )
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()
        self.welm_mtp_mirror_kv_states = None
        self.welm_mtp_mirror_padding_index = 0
        self.welm_mtp_mirror_kv_len = 0
        self.sample_draft = (
            eagle_worker.welmv4_mtp_sample_draft
            and eagle_worker._has_welmv4_mtp_fixed_draft_sampling_params()
            and eagle_worker._get_welmv4_mtp_draft_sampling_topk() > 0
        )
        self.sampling_topk = (
            eagle_worker._get_welmv4_mtp_draft_sampling_topk()
            if self.sample_draft
            else 0
        )
        self.use_top_p = (
            self.sample_draft
            and eagle_worker.welmv4_mtp_draft_fixed_top_p is not None
            and eagle_worker.welmv4_mtp_draft_fixed_top_p < 1.0
            and (_is_cuda or _is_musa)
        )

        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.num_tokens_per_bs = int(self.speculative_num_draft_tokens)
        self.topk = int(model_runner.server_args.speculative_eagle_topk)
        self.v1_draft_num_cache_locs = self.topk * self.speculative_num_steps
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.max_v1_draft_cache_locs = self.max_bs * self.v1_draft_num_cache_locs
        self._welm_mtp_mirror_cu_seqlens_q = {
            bs: torch.arange(
                0,
                bs + 1,
                dtype=torch.int32,
                device=model_runner.device,
            )
            for bs in self.capture_bs
        }

        self.draft_extend_attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
        v1_base_kv_attn_backend = self._get_welmv4_mtp_base_kv_attn_backend()
        if v1_base_kv_attn_backend is not None:
            v1_base_kv_attn_backend.init_cuda_graph_state(
                self.max_bs, self.max_bs * self.topk
            )
        self.seq_len_fill_value = (
            self.draft_extend_attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        with torch.device(model_runner.device):
            input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            first_input_ids = torch.zeros((self.max_bs,), dtype=torch.int64)
            req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
            draft_out_cache_loc = torch.zeros(
                (self.max_v1_draft_cache_locs,), dtype=torch.int64
            )
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            draft_positions = torch.zeros(
                (self.max_bs * self.topk,), dtype=torch.int64
            )
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)
            hidden_states = torch.zeros(
                (self.max_num_token, self.model_runner.model_config.spec_hidden_size),
                dtype=self.model_runner.dtype,
            )
            mirrored_kv_indices = torch.arange(self.max_num_token, dtype=torch.int64)
            seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            draft_seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            extend_seq_lens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            num_accepted_drafts = torch.full(
                (self.max_bs,), self.num_tokens_per_bs - 1, dtype=torch.int32
            )
            num_accepted_tokens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            custom_last_index = (
                torch.arange(self.max_bs, dtype=torch.long) * self.num_tokens_per_bs
                + self.num_tokens_per_bs
                - 1
            )
            custom_last_cache_loc = torch.zeros((self.max_bs,), dtype=torch.int64)
            if hasattr(
                self.model_runner.model_config.hf_config, "draft_vocab_size"
            ):
                vocab_size = self.model_runner.model_config.hf_config.draft_vocab_size
            elif hasattr(self.model_runner.model_config.hf_config, "hot_vocab_size"):
                vocab_size = self.model_runner.model_config.hf_config.hot_vocab_size
            else:
                vocab_size = self.model_runner.model_config.vocab_size
            next_token_logits_buffer = torch.zeros(
                (self.max_num_token, vocab_size),
                dtype=torch.float,
            )
            temperature = torch.ones((self.max_bs, 1), dtype=torch.float32)
            top_p = torch.ones((self.max_bs,), dtype=torch.float32)
            uniform_samples = torch.full(
                (self.speculative_num_steps, self.max_bs, 1),
                0.5,
                dtype=torch.float32,
            )

            if eagle_worker._should_use_welmv4_mtp_oe_hash_kernel():
                oe_grams, oe_vocab_sizes = get_welm_oe_hash_config(
                    self.model_runner.model_config
                )
                history_width = max(int(g) for g in oe_grams)
                welm_mtp_oe_hash_out = torch.zeros(
                    (len(oe_vocab_sizes), self.max_num_token), dtype=torch.int64
                )
                welm_mtp_query_hash_inputs = torch.zeros(
                    (len(oe_vocab_sizes), self.max_bs), dtype=torch.int64
                )
                welm_mtp_oe_entry_history = torch.zeros(
                    (self.max_bs, history_width), dtype=torch.int64
                )
                welm_mtp_oe_work_history_a = torch.zeros(
                    (self.max_num_token, history_width), dtype=torch.int64
                )
                welm_mtp_oe_work_history_b = torch.zeros(
                    (self.max_num_token, history_width), dtype=torch.int64
                )
                welm_mtp_oe_parent_indices = torch.zeros(
                    (self.max_num_token,), dtype=torch.int64
                )
                welm_mtp_oe_prev_input_ids = torch.zeros(
                    (self.max_num_token,), dtype=torch.int64
                )
                welm_mtp_oe_prev_prev_input_ids = torch.zeros(
                    (self.max_num_token,), dtype=torch.int64
                )
                welm_mtp_oe_output_prev_input_ids = torch.zeros(
                    (self.max_num_token,), dtype=torch.int64
                )
                welm_mtp_oe_hash_out_batch_major = torch.zeros(
                    (self.max_num_token, len(oe_vocab_sizes)), dtype=torch.int64
                )
                welm_mtp_draft_input_ids = torch.zeros(
                    (self.speculative_num_steps, self.max_num_token),
                    dtype=torch.int64,
                )
            else:
                welm_mtp_oe_hash_out = None
                welm_mtp_query_hash_inputs = None
                welm_mtp_oe_entry_history = None
                welm_mtp_oe_work_history_a = None
                welm_mtp_oe_work_history_b = None
                welm_mtp_oe_parent_indices = None
                welm_mtp_oe_prev_input_ids = None
                welm_mtp_oe_prev_prev_input_ids = None
                welm_mtp_oe_output_prev_input_ids = None
                welm_mtp_oe_hash_out_batch_major = None
                welm_mtp_draft_input_ids = None

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

            self.welm_mtp_mirror_padding_index = self.max_num_token
            self.welm_mtp_mirror_kv_len = self.max_num_token + 1
            self.welm_mtp_mirror_kv_states = {}
            for layer_idx, kv_size in self._welmv4_mtp_mirror_kv_specs():
                self.welm_mtp_mirror_kv_states[layer_idx] = (
                    torch.zeros(
                        (self.welm_mtp_mirror_kv_len, kv_size),
                        dtype=self.model_runner.dtype,
                    ),
                    torch.zeros(
                        (self.welm_mtp_mirror_kv_len, kv_size),
                        dtype=self.model_runner.dtype,
                    ),
                )

        seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        draft_seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        self.buffers = WelmMTPDraftProposalInputBuffers(
            input_ids=input_ids,
            first_input_ids=first_input_ids,
            req_pool_indices=req_pool_indices,
            out_cache_loc=out_cache_loc,
            draft_out_cache_loc=draft_out_cache_loc,
            positions=positions,
            draft_positions=draft_positions,
            mrope_positions=mrope_positions,
            hidden_states=hidden_states,
            mirrored_kv_indices=mirrored_kv_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            draft_seq_lens=draft_seq_lens,
            draft_seq_lens_cpu=draft_seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            num_accepted_drafts=num_accepted_drafts,
            num_accepted_tokens=num_accepted_tokens,
            custom_last_index=custom_last_index,
            custom_last_cache_loc=custom_last_cache_loc,
            next_token_logits_buffer=next_token_logits_buffer,
            temperature=temperature,
            top_p=top_p,
            uniform_samples=uniform_samples,
            welm_mtp_oe_hash_out=welm_mtp_oe_hash_out,
            welm_mtp_query_hash_inputs=welm_mtp_query_hash_inputs,
            welm_mtp_oe_entry_history=welm_mtp_oe_entry_history,
            welm_mtp_oe_work_history_a=welm_mtp_oe_work_history_a,
            welm_mtp_oe_work_history_b=welm_mtp_oe_work_history_b,
            welm_mtp_oe_parent_indices=welm_mtp_oe_parent_indices,
            welm_mtp_oe_prev_input_ids=welm_mtp_oe_prev_input_ids,
            welm_mtp_oe_prev_prev_input_ids=welm_mtp_oe_prev_prev_input_ids,
            welm_mtp_oe_output_prev_input_ids=welm_mtp_oe_output_prev_input_ids,
            welm_mtp_oe_hash_out_batch_major=welm_mtp_oe_hash_out_batch_major,
            welm_mtp_draft_input_ids=welm_mtp_draft_input_ids,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )
        self.buffers.share_buffers()

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture WeLM MTP draft proposal cuda graph failed: {e}\n"
                f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _welmv4_mtp_mirror_kv_specs(self) -> list[tuple[int, int]]:
        specs = {}
        model = getattr(self.model_runner.model, "model", None)
        for layer in getattr(model, "decoder_layers", []):
            attn = getattr(layer, "self_attn", None)
            qkv_proj = getattr(attn, "qkv_proj", None)
            imitated_layer_idx = getattr(qkv_proj, "imitated_layer_idx", None)
            mirror_layer_idx = getattr(qkv_proj, "mirror_layer_idx", None)
            kv_size = getattr(attn, "kv_size", None)
            if imitated_layer_idx is not None and kv_size is not None:
                layer_idx = (
                    int(mirror_layer_idx)
                    if mirror_layer_idx is not None
                    else int(imitated_layer_idx)
                )
                specs[layer_idx] = int(kv_size)
        return sorted(specs.items())

    def _welmv4_mtp_graph_model_specific_states(self):
        if not self.welm_mtp_mirror_kv_states:
            return None
        return {
            "welm_kv_mirror_states": {
                layer_idx: (k, v)
                for layer_idx, (k, v) in self.welm_mtp_mirror_kv_states.items()
            }
        }

    def _is_welmv4_mtp_v1_draft_model(self) -> bool:
        return bool(self.eagle_worker._is_welmv4_mtp_v1_draft_model())

    def _is_welmv4_mtp_v2_draft_model(self) -> bool:
        return bool(self.eagle_worker._is_welmv4_mtp_v2_draft_model())

    def _copy_welmv4_mtp_mirror_kv_states(
        self, forward_batch: ForwardBatch, graph_num_tokens: int
    ) -> None:
        if not self.welm_mtp_mirror_kv_states:
            return
        model_specific_states = forward_batch.model_specific_states or {}
        kv_states = model_specific_states.get("welm_kv_mirror_states")
        if not isinstance(kv_states, dict):
            raise RuntimeError(
                "Missing WeLM MTP mirrored KV states for draft proposal graph replay."
            )
        mirrored_kv_indices = getattr(
            forward_batch.spec_info, "mirrored_kv_indices", None
        )
        required_kv_len = (
            int(forward_batch.input_ids.numel())
            if mirrored_kv_indices is None
            else 0
        )
        if mirrored_kv_indices is not None and mirrored_kv_indices.numel() > 0:
            required_kv_len = int(mirrored_kv_indices.max().item()) + 1
        if required_kv_len > self.welm_mtp_mirror_padding_index:
            raise RuntimeError(
                "WeLM MTP mirrored_kv_indices exceed draft proposal graph "
                "mirror-KV source capacity: "
                f"{required_kv_len=} capacity={self.welm_mtp_mirror_padding_index}"
            )
        for layer_idx, (dst_k, dst_v) in self.welm_mtp_mirror_kv_states.items():
            src_pair = kv_states.get(layer_idx)
            if src_pair is None:
                raise RuntimeError(
                    "Missing WeLM MTP mirrored KV state for draft proposal graph "
                    f"replay: {layer_idx=}"
                )
            src_k, src_v = src_pair
            if src_k.shape[0] < required_kv_len or src_v.shape[0] < required_kv_len:
                raise RuntimeError(
                    "WeLM MTP mirrored KV state is shorter than mirrored_kv_indices "
                    f"require: {layer_idx=} {required_kv_len=} "
                    f"k_len={src_k.shape[0]} v_len={src_v.shape[0]}"
                )
            if required_kv_len > 0:
                dst_k[:required_kv_len].copy_(src_k[:required_kv_len])
                dst_v[:required_kv_len].copy_(src_v[:required_kv_len])
                dst_k[self.welm_mtp_mirror_padding_index].copy_(
                    src_k[required_kv_len - 1]
                )
                dst_v[self.welm_mtp_mirror_padding_index].copy_(
                    src_v[required_kv_len - 1]
                )
            else:
                dst_k[self.welm_mtp_mirror_padding_index].zero_()
                dst_v[self.welm_mtp_mirror_padding_index].zero_()

    def _copy_welmv4_mtp_mirrored_kv_indices(
        self, forward_batch: ForwardBatch, graph_num_tokens: int
    ) -> None:
        buffers = self.buffers
        if buffers.mirrored_kv_indices is None:
            return
        buffers.mirrored_kv_indices[:graph_num_tokens].fill_(
            self.welm_mtp_mirror_padding_index
        )
        mirrored_kv_indices = getattr(
            forward_batch.spec_info, "mirrored_kv_indices", None
        )
        if mirrored_kv_indices is None:
            raw_num_tokens = int(forward_batch.input_ids.numel())
            buffers.mirrored_kv_indices[:raw_num_tokens].copy_(
                torch.arange(
                    raw_num_tokens,
                    dtype=buffers.mirrored_kv_indices.dtype,
                    device=buffers.mirrored_kv_indices.device,
                )
            )
            return
        if mirrored_kv_indices.numel() > graph_num_tokens:
            raise RuntimeError(
                "WeLM MTP mirrored_kv_indices is longer than draft proposal graph "
                f"tokens: {mirrored_kv_indices.numel()} > {graph_num_tokens}"
            )
        buffers.mirrored_kv_indices[: mirrored_kv_indices.numel()].copy_(
            mirrored_kv_indices
        )

    def _make_padded_accept_lens(
        self, accepted_lens_cpu: list[int], graph_bs: int
    ) -> Optional[list[int]]:
        if not 1 <= len(accepted_lens_cpu) <= graph_bs:
            return None
        if any(
            accepted_len <= 0 or accepted_len > self.num_tokens_per_bs
            for accepted_len in accepted_lens_cpu
        ):
            return None

        padded_lens_cpu = list(accepted_lens_cpu)
        padded_lens_cpu.extend([1] * (graph_bs - len(padded_lens_cpu)))

        graph_num_tokens = graph_bs * self.num_tokens_per_bs
        real_total = sum(padded_lens_cpu)
        if real_total > graph_num_tokens:
            return None

        pad_tokens = graph_num_tokens - real_total
        for i in range(graph_bs - 1, -1, -1):
            if pad_tokens <= 0:
                break
            room = self.num_tokens_per_bs - padded_lens_cpu[i]
            if room <= 0:
                continue
            add = min(room, pad_tokens)
            padded_lens_cpu[i] += add
            pad_tokens -= add

        if pad_tokens != 0 or sum(padded_lens_cpu) != graph_num_tokens:
            return None
        return padded_lens_cpu

    def _copy_runtime_flat_layout(
        self,
        *,
        input_ids: torch.Tensor,
        out_cache_loc: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mirrored_kv_indices: Optional[torch.Tensor],
        cached_hash: Optional[torch.Tensor],
        accepted_lens_cpu: list[int],
        padded_lens_cpu: list[int],
        raw_bs: int,
        graph_num_tokens: int,
    ) -> None:
        buffers = self.buffers
        buffers.input_ids[:graph_num_tokens].zero_()
        buffers.out_cache_loc[:graph_num_tokens].zero_()
        buffers.positions[:graph_num_tokens].zero_()
        buffers.hidden_states[:graph_num_tokens].zero_()
        if buffers.mirrored_kv_indices is not None:
            buffers.mirrored_kv_indices[:graph_num_tokens].fill_(
                self.welm_mtp_mirror_padding_index
            )
        if buffers.welm_mtp_oe_hash_out is not None:
            buffers.welm_mtp_oe_hash_out[:, :graph_num_tokens].zero_()

        real_last_indices = []
        src_start = 0
        dst_start = 0
        for row, padded_len in enumerate(padded_lens_cpu):
            if row < raw_bs:
                real_len = accepted_lens_cpu[row]
                src_end = src_start + real_len
                dst_end = dst_start + real_len
                pad_len = padded_len - real_len

                buffers.input_ids[dst_start:dst_end].copy_(
                    input_ids[src_start:src_end]
                )
                buffers.out_cache_loc[dst_start:dst_end].copy_(
                    out_cache_loc[src_start:src_end]
                )
                buffers.positions[dst_start:dst_end].copy_(
                    positions[src_start:src_end]
                )
                buffers.hidden_states[dst_start:dst_end].copy_(
                    hidden_states[src_start:src_end]
                )
                if buffers.mirrored_kv_indices is not None:
                    if mirrored_kv_indices is None:
                        buffers.mirrored_kv_indices[dst_start:dst_end].copy_(
                            torch.arange(
                                src_start,
                                src_end,
                                dtype=buffers.mirrored_kv_indices.dtype,
                                device=buffers.mirrored_kv_indices.device,
                            )
                        )
                    else:
                        buffers.mirrored_kv_indices[dst_start:dst_end].copy_(
                            mirrored_kv_indices[src_start:src_end]
                        )
                if cached_hash is not None and buffers.welm_mtp_oe_hash_out is not None:
                    buffers.welm_mtp_oe_hash_out[:, dst_start:dst_end].copy_(
                        cached_hash[:, src_start:src_end]
                    )

                real_last = dst_end - 1
                real_last_indices.append(real_last)
                if pad_len > 0:
                    pad_start = dst_end
                    pad_end = dst_start + padded_len
                    buffers.input_ids[pad_start:pad_end].copy_(
                        input_ids[src_end - 1].expand(pad_len)
                    )
                    buffers.positions[pad_start:pad_end].copy_(
                        positions[src_end - 1]
                        + torch.arange(
                            1,
                            pad_len + 1,
                            dtype=buffers.positions.dtype,
                            device=buffers.positions.device,
                        )
                    )
                src_start = src_end
            else:
                real_last_indices.append(dst_start + padded_len - 1)

            dst_start += padded_len

        if src_start != input_ids.numel() or dst_start != graph_num_tokens:
            raise RuntimeError(
                "WeLM MTP draft proposal graph padding layout mismatch: "
                f"src={src_start}/{input_ids.numel()} "
                f"dst={dst_start}/{graph_num_tokens}."
            )
        buffers.custom_last_index[: len(padded_lens_cpu)].copy_(
            torch.tensor(
                real_last_indices,
                dtype=buffers.custom_last_index.dtype,
                device=buffers.custom_last_index.device,
            )
        )
        real_rows = min(raw_bs, len(real_last_indices))
        if real_rows > 0:
            buffers.custom_last_cache_loc[:real_rows].copy_(
                buffers.out_cache_loc[buffers.custom_last_index[:real_rows]]
            )
        if len(real_last_indices) > real_rows:
            buffers.custom_last_cache_loc[real_rows : len(real_last_indices)].zero_()

    def _copy_v1_runtime_flat_layout(
        self,
        *,
        input_ids: torch.Tensor,
        out_cache_loc: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mirrored_kv_indices: Optional[torch.Tensor],
        cached_hash: Optional[torch.Tensor],
        accepted_lens_cpu: list[int],
        raw_bs: int,
        graph_bs: int,
        graph_num_tokens: int,
    ) -> None:
        buffers = self.buffers
        raw_num_tokens = int(input_ids.numel())
        if raw_num_tokens != raw_bs * self.num_tokens_per_bs:
            raise RuntimeError(
                "WeLM MTP V1 draft proposal graph expects a dense per-request "
                f"draft-token layout: {raw_num_tokens=} raw_bs={raw_bs} "
                f"num_tokens_per_bs={self.num_tokens_per_bs}."
            )

        buffers.input_ids[:graph_num_tokens].zero_()
        buffers.out_cache_loc[:graph_num_tokens].zero_()
        buffers.positions[:graph_num_tokens].zero_()
        buffers.hidden_states[:graph_num_tokens].zero_()
        buffers.draft_out_cache_loc[: self.max_v1_draft_cache_locs].zero_()
        buffers.draft_positions[: graph_bs * self.topk].zero_()
        if buffers.mirrored_kv_indices is not None:
            buffers.mirrored_kv_indices[:graph_num_tokens].fill_(
                self.welm_mtp_mirror_padding_index
            )
        if buffers.welm_mtp_oe_hash_out is not None:
            buffers.welm_mtp_oe_hash_out[:, :graph_num_tokens].zero_()

        buffers.input_ids[:raw_num_tokens].copy_(input_ids)
        buffers.out_cache_loc[:raw_num_tokens].copy_(out_cache_loc)
        buffers.positions[:raw_num_tokens].copy_(positions)
        buffers.hidden_states[:raw_num_tokens].copy_(hidden_states)
        if buffers.mirrored_kv_indices is not None:
            if mirrored_kv_indices is None:
                buffers.mirrored_kv_indices[:raw_num_tokens].copy_(
                    torch.arange(
                        raw_num_tokens,
                        dtype=buffers.mirrored_kv_indices.dtype,
                        device=buffers.mirrored_kv_indices.device,
                    )
                )
            else:
                buffers.mirrored_kv_indices[:raw_num_tokens].copy_(mirrored_kv_indices)
        if cached_hash is not None and buffers.welm_mtp_oe_hash_out is not None:
            buffers.welm_mtp_oe_hash_out[:, :raw_num_tokens].copy_(cached_hash)

        real_last_indices = [
            row * self.num_tokens_per_bs + int(accepted_lens_cpu[row]) - 1
            for row in range(raw_bs)
        ]
        if graph_bs > raw_bs:
            real_last_indices.extend(
                [
                    row * self.num_tokens_per_bs + self.num_tokens_per_bs - 1
                    for row in range(raw_bs, graph_bs)
                ]
            )
        buffers.custom_last_index[:graph_bs].copy_(
            torch.tensor(
                real_last_indices,
                dtype=buffers.custom_last_index.dtype,
                device=buffers.custom_last_index.device,
            )
        )
        if raw_bs > 0:
            buffers.custom_last_cache_loc[:raw_bs].copy_(
                buffers.out_cache_loc[buffers.custom_last_index[:raw_bs]]
            )
        if graph_bs > raw_bs:
            buffers.custom_last_cache_loc[raw_bs:graph_bs].zero_()

    def _set_welmv4_mtp_mirror_metadata(self, bs: int) -> None:
        if not self.model_runner.server_args.enable_welm_kv_mirror_opt:
            return
        metadata = self.draft_extend_attn_backend.draft_extend_metadata.get(bs)
        if metadata is None:
            return
        mirror_cu_seqlens_q = self._welm_mtp_mirror_cu_seqlens_q.get(bs)
        if mirror_cu_seqlens_q is None:
            mirror_cu_seqlens_q = torch.arange(
                0,
                bs + 1,
                dtype=torch.int32,
                device=self.model_runner.device,
            )
            self._welm_mtp_mirror_cu_seqlens_q[bs] = mirror_cu_seqlens_q
        metadata.mirror_cu_seqlens_q = mirror_cu_seqlens_q
        metadata.mirror_max_seq_len_q = 1

    def _copy_sampling_params(self, forward_batch: ForwardBatch, raw_bs: int, bs: int):
        buffers = self.buffers
        fixed_temperature = self.eagle_worker.welmv4_mtp_draft_fixed_temperature
        fixed_top_p = self.eagle_worker.welmv4_mtp_draft_fixed_top_p

        buffers.temperature[:bs].fill_(
            1.0 if fixed_temperature is None else fixed_temperature
        )
        buffers.top_p[:bs].fill_(1.0 if fixed_top_p is None else fixed_top_p)

        sampling_info = forward_batch.sampling_info
        if sampling_info is None:
            return
        if fixed_temperature is None:
            temperatures = getattr(sampling_info, "temperatures", None)
            if temperatures is not None:
                buffers.temperature[:raw_bs].copy_(
                    temperatures[:raw_bs].to(dtype=torch.float32)
                )
        if fixed_top_p is None:
            top_ps = getattr(sampling_info, "top_ps", None)
            if top_ps is not None:
                buffers.top_p[:raw_bs].copy_(top_ps[:raw_bs].to(dtype=torch.float32))

    def _copy_sampling_randomness(
        self, forward_batch: ForwardBatch, raw_bs: int, bs: int
    ):
        if not self.sample_draft:
            return
        sampling_info = forward_batch.sampling_info
        if (
            sampling_info is not None
            and getattr(sampling_info, "sampling_seed", None) is not None
        ):
            base_positions = self.buffers.positions[
                self.buffers.custom_last_index[:raw_bs]
            ]
            deterministic_uniforms = welm_mtp_deterministic_uniforms(
                sampling_info=sampling_info,
                positions=base_positions,
                batch_size=raw_bs,
                width=self.speculative_num_steps,
                salt=3000,
            )
            if deterministic_uniforms is not None:
                self.buffers.uniform_samples[:, :bs].fill_(0.5)
                self.buffers.uniform_samples[:, :raw_bs, 0].copy_(
                    deterministic_uniforms.transpose(0, 1).contiguous()
                )
                if self.model_runner.tp_group.world_size > 1:
                    uniform_samples = self.buffers.uniform_samples[:, :bs]
                    broadcast_samples = uniform_samples.contiguous()
                    self.model_runner.tp_group.broadcast(broadcast_samples, src=0)
                    uniform_samples.copy_(broadcast_samples)
                return
        uniform_samples = self.buffers.uniform_samples[:, :bs]
        uniform_samples.uniform_()
        if self.model_runner.tp_group.world_size > 1:
            broadcast_samples = uniform_samples.contiguous()
            self.model_runner.tp_group.broadcast(broadcast_samples, src=0)
            uniform_samples.copy_(broadcast_samples)

    def _copy_v1_recursive_draft_state(
        self,
        forward_batch: ForwardBatch,
        raw_bs: int,
        bs: int,
    ) -> None:
        buffers = self.buffers
        draft_seq_lens = getattr(
            forward_batch, "welm_mtp_recursive_draft_seq_lens", None
        )
        if draft_seq_lens is None:
            draft_seq_lens = forward_batch.seq_lens
        buffers.draft_seq_lens[:bs].fill_(self.seq_len_fill_value)
        buffers.draft_seq_lens[:raw_bs].copy_(draft_seq_lens[:raw_bs])

        draft_seq_lens_cpu = getattr(
            forward_batch, "welm_mtp_recursive_draft_seq_lens_cpu", None
        )
        buffers.draft_seq_lens_cpu[:bs].fill_(self.seq_len_fill_value)
        if draft_seq_lens_cpu is not None:
            buffers.draft_seq_lens_cpu[:raw_bs].copy_(draft_seq_lens_cpu[:raw_bs])
        elif forward_batch.seq_lens_cpu is not None:
            buffers.draft_seq_lens_cpu[:raw_bs].copy_(
                forward_batch.seq_lens_cpu[:raw_bs]
            )
        else:
            buffers.draft_seq_lens_cpu[:raw_bs].copy_(draft_seq_lens[:raw_bs].cpu())

        draft_cache_locs = buffers.draft_out_cache_loc[
            : raw_bs * self.v1_draft_num_cache_locs
        ]
        assign_draft_cache_locs_page_size_1[(raw_bs,)](
            buffers.req_pool_indices,
            self.model_runner.req_to_token_pool.req_to_token,
            buffers.draft_seq_lens,
            draft_cache_locs,
            self.model_runner.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
        )
        if bs > raw_bs:
            buffers.draft_out_cache_loc[
                raw_bs
                * self.v1_draft_num_cache_locs : bs
                * self.v1_draft_num_cache_locs
            ].zero_()

    def _get_welmv4_mtp_base_kv_attn_backend(self):
        if not self._is_welmv4_mtp_v1_draft_model():
            return None
        return (
            self.model_runner.decode_attn_backend
            if self.model_runner.server_args.enable_pdmux
            else self.model_runner.attn_backend
        )

    def _prepare_v1_base_kv_capture_metadata(
        self,
        forward_batch: ForwardBatch,
        bs: int,
    ) -> None:
        attn_backend = self._get_welmv4_mtp_base_kv_attn_backend()
        if attn_backend is None:
            return
        num_tokens = bs * self.topk
        spec_info_backup = forward_batch.spec_info
        forward_batch.spec_info = None
        try:
            if not (
                self.topk > 1
                and hasattr(
                    attn_backend,
                    "init_welm_mtp_base_kv_metadata_capture_cuda_graph",
                )
                and attn_backend.init_welm_mtp_base_kv_metadata_capture_cuda_graph(
                    bs,
                    num_tokens,
                    forward_batch.req_pool_indices,
                    forward_batch.welm_mtp_recursive_draft_seq_lens,
                )
            ):
                attn_backend.init_forward_metadata_capture_cuda_graph(
                    bs,
                    num_tokens,
                    forward_batch.req_pool_indices,
                    forward_batch.welm_mtp_recursive_draft_seq_lens,
                    None,
                    ForwardMode.DECODE,
                    None,
                )
        finally:
            forward_batch.spec_info = spec_info_backup
        forward_batch.attn_backend = attn_backend
        forward_batch.welm_mtp_base_kv_metadata_prepared = True

    def _prepare_v1_base_kv_replay_metadata(
        self,
        replay_forward_batch: ForwardBatch,
        bs: int,
        raw_bs: int,
    ) -> None:
        attn_backend = self._get_welmv4_mtp_base_kv_attn_backend()
        if attn_backend is None:
            return
        buffers = self.buffers
        num_tokens = bs * self.topk
        seq_lens_sum = int(buffers.draft_seq_lens_cpu[:raw_bs].sum().item()) + (
            bs - raw_bs
        ) * int(self.seq_len_fill_value)
        spec_info_backup = replay_forward_batch.spec_info
        replay_forward_batch.spec_info = None
        try:
            if not (
                self.topk > 1
                and hasattr(
                    attn_backend,
                    "init_welm_mtp_base_kv_metadata_replay_cuda_graph",
                )
                and attn_backend.init_welm_mtp_base_kv_metadata_replay_cuda_graph(
                    bs,
                    self.topk,
                    buffers.req_pool_indices[:bs],
                    buffers.draft_seq_lens[:bs],
                    buffers.draft_seq_lens_cpu[:bs],
                )
            ):
                attn_backend.init_forward_metadata_replay_cuda_graph(
                    bs,
                    buffers.req_pool_indices[:bs],
                    buffers.draft_seq_lens[:bs],
                    seq_lens_sum,
                    None,
                    ForwardMode.DECODE,
                    None,
                    seq_lens_cpu=buffers.draft_seq_lens_cpu[:bs],
                )
        finally:
            replay_forward_batch.spec_info = spec_info_backup
        replay_forward_batch.attn_backend = attn_backend
        replay_forward_batch.welm_mtp_base_kv_metadata_prepared = True

    def _select_topk(
        self,
        logits: torch.Tensor,
        step: int,
        bs: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if not self.sample_draft:
            topk_index = torch.argmax(logits, dim=-1, keepdim=True)
            topk_p = torch.ones(
                topk_index.shape,
                dtype=logits.dtype,
                device=logits.device,
            )
            return topk_p, topk_index, None, None

        top_logits, top_indices = torch.topk(
            logits.float(),
            k=self.sampling_topk,
            dim=-1,
            sorted=self.use_top_p,
        )
        temperature = self.buffers.temperature[:bs]
        probs = F.softmax(top_logits / temperature, dim=-1)
        if self.use_top_p:
            probs = top_p_renorm_prob(probs, self.buffers.top_p[:bs])
        cdf = torch.cumsum(probs, dim=-1)
        uniform = self.buffers.uniform_samples[step, :bs]
        sample_pos = torch.sum(cdf < uniform, dim=-1, keepdim=True).to(torch.int64)
        sample_pos = torch.clamp(sample_pos, max=self.sampling_topk - 1)
        topk_p = torch.gather(probs, dim=-1, index=sample_pos)
        topk_index = torch.gather(top_indices, dim=-1, index=sample_pos)
        return topk_p, topk_index, top_indices, probs

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        if not forward_batch.forward_mode.is_draft_extend(include_v2=True):
            return False
        if self.eagle_worker._should_sample_welmv4_mtp_draft(forward_batch):
            if not self.sample_draft:
                return False
            if self.use_top_p != self.eagle_worker._should_use_welmv4_mtp_draft_top_p(
                forward_batch
            ):
                return False
        elif self.sample_draft:
            return False
        if forward_batch.input_ids is None or forward_batch.out_cache_loc is None:
            return False
        raw_bs = int(forward_batch.batch_size)
        raw_num_tokens = int(forward_batch.input_ids.numel())
        if not (raw_bs <= raw_num_tokens <= raw_bs * self.num_tokens_per_bs):
            return False
        if self.require_mlp_tp_gather:
            cuda_graph_bs = max(forward_batch.global_num_tokens_cpu) // (
                self.num_tokens_per_bs
            )
        else:
            cuda_graph_bs = raw_bs
        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )
        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph
        if not is_bs_supported:
            return False
        spec_info = forward_batch.spec_info
        if not isinstance(spec_info, EagleDraftInput):
            return False
        if (
            spec_info.num_accepted_tokens_cpu is None
            or spec_info.num_accepted_tokens is None
            or spec_info.num_accepted_drafts is None
        ):
            return False
        accepted_lens_cpu = [int(x) for x in spec_info.num_accepted_tokens_cpu]
        if len(accepted_lens_cpu) != raw_bs:
            return False
        index = bisect.bisect_left(self.capture_bs, cuda_graph_bs)
        if index >= len(self.capture_bs):
            return False
        graph_bs = self.capture_bs[index]
        if self._is_welmv4_mtp_v1_draft_model():
            return (
                raw_num_tokens == raw_bs * self.num_tokens_per_bs
                and all(0 < x <= self.num_tokens_per_bs for x in accepted_lens_cpu)
            )
        if sum(accepted_lens_cpu) != raw_num_tokens:
            return False
        return self._make_padded_accept_lens(accepted_lens_cpu, graph_bs) is not None

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _replay(self):
        ctx = (
            self.model_runner.device_timer.wrap(
                metadata={"category": "welm_mtp_draft_proposal"}
            )
            if self.model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            self.graphs[self.bs].replay()

    def capture(self):
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(
        self, bs: int, forward: Callable, stream_idx: int = 0
    ):
        buffers = self.buffers
        graph = self._create_graph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs
        forward_mode = (
            ForwardMode.DRAFT_EXTEND_V2
            if self._is_welmv4_mtp_v1_draft_model()
            else ForwardMode.DRAFT_EXTEND
        )

        input_ids = buffers.input_ids[:num_tokens]
        first_input_ids = buffers.first_input_ids[:bs]
        req_pool_indices = buffers.req_pool_indices[:bs]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        positions = buffers.positions[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        hidden_states = buffers.hidden_states[:num_tokens]
        mirrored_kv_indices = buffers.mirrored_kv_indices[:num_tokens]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        extend_seq_lens = buffers.extend_seq_lens[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        num_accepted_drafts = buffers.num_accepted_drafts[:bs]
        num_accepted_tokens = buffers.num_accepted_tokens[:bs]
        custom_last_index = buffers.custom_last_index[:bs]
        custom_last_cache_loc = buffers.custom_last_cache_loc[:bs]
        next_token_logits_buffer_rows = num_tokens
        if self._is_welmv4_mtp_v2_draft_model() or (
            self._is_welmv4_mtp_v1_draft_model()
            and self.model_runner.server_args.enable_welm_kv_mirror_opt
        ):
            next_token_logits_buffer_rows = bs
        next_token_logits_buffer = buffers.next_token_logits_buffer[
            :next_token_logits_buffer_rows
        ]
        draft_out_cache_loc = buffers.draft_out_cache_loc[
            : bs * self.v1_draft_num_cache_locs
        ]
        draft_positions = buffers.draft_positions[: bs * self.topk]
        draft_seq_lens = buffers.draft_seq_lens[:bs]
        draft_seq_lens_cpu = buffers.draft_seq_lens_cpu[:bs]

        if self._is_welmv4_mtp_v1_draft_model():
            out_cache_loc.fill_(1)
            custom_last_cache_loc.fill_(1)

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor([num_tokens], dtype=torch.int32, device=input_ids.device)
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor([num_tokens], dtype=torch.int32, device=input_ids.device)
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
            global_dp_buffer_len = num_tokens
        else:
            global_num_tokens = None
            global_num_tokens_for_logprob = None
            global_dp_buffer_len = None

        capture_hidden_mode = (
            CaptureHiddenMode.FULL
            if self._is_welmv4_mtp_v1_draft_model()
            else CaptureHiddenMode.LAST
        )

        spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            mirrored_kv_indices=mirrored_kv_indices,
            num_accepted_drafts=num_accepted_drafts,
            num_accepted_tokens=num_accepted_tokens,
            num_accepted_drafts_cpu=[self.num_tokens_per_bs - 1] * bs,
            num_accepted_tokens_cpu=[self.num_tokens_per_bs] * bs,
            capture_hidden_mode=capture_hidden_mode,
            num_tokens_per_req=self.num_tokens_per_bs,
            num_tokens_for_logprob_per_req=self.num_tokens_per_bs,
            model_specific_states=self._welmv4_mtp_graph_model_specific_states(),
        )
        spec_info.extend_seq_lens_cpu = list(extend_seq_lens_cpu)
        spec_info.extend_seq_lens_tensor = extend_seq_lens
        forward_batch = ForwardBatch(
            forward_mode=forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(self.seq_len_fill_value) * bs,
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            model_specific_states=spec_info.model_specific_states,
            capture_hidden_mode=capture_hidden_mode,
            attn_backend=self.draft_extend_attn_backend,
            oe_context=None,
        )
        forward_batch.custom_last_index = custom_last_index
        forward_batch.custom_last_cache_loc = custom_last_cache_loc
        forward_batch.welm_mtp_recursive_draft_out_cache_loc = draft_out_cache_loc
        forward_batch.welm_mtp_recursive_draft_positions = draft_positions
        forward_batch.welm_mtp_recursive_draft_seq_lens = draft_seq_lens
        forward_batch.welm_mtp_recursive_draft_seq_lens_cpu = draft_seq_lens_cpu
        forward_batch.welm_mtp_draft_extend_metadata_bs = bs
        forward_batch.welm_mtp_use_legacy_recursive_base_positions = (
            self._is_welmv4_mtp_v1_draft_model() and self.topk == 1
        )
        forward_batch.enable_welm_kv_mirror_opt = (
            self.model_runner.server_args.enable_welm_kv_mirror_opt
        )
        if buffers.welm_mtp_oe_hash_out is not None:
            forward_batch.welm_oe_decode_hashed_inputs = (
                buffers.welm_mtp_oe_hash_out[:, :num_tokens]
            )
            if self._is_welmv4_mtp_v1_draft_model():
                work_history_size = num_tokens
            else:
                work_history_size = bs
            forward_batch.welm_mtp_oe_work_history = [
                buffers.welm_mtp_oe_work_history_a[:work_history_size],
                buffers.welm_mtp_oe_work_history_b[:work_history_size],
            ]
            forward_batch.welm_mtp_oe_entry_history_state = (
                buffers.welm_mtp_oe_entry_history[:bs]
            )
            if self._is_welmv4_mtp_v1_draft_model():
                forward_batch.welm_mtp_graph_oe_context_prepared = True
                forward_batch.welm_mtp_oe_hash_out = (
                    buffers.welm_mtp_oe_hash_out[:, :num_tokens]
                )
                forward_batch.welm_mtp_oe_parent_scratch = (
                    buffers.welm_mtp_oe_parent_indices[:num_tokens]
                )
            forward_batch.welm_mtp_oe_prev_input_ids = (
                buffers.welm_mtp_oe_prev_input_ids[:num_tokens]
            )
            forward_batch.welm_mtp_oe_prev_prev_input_ids = (
                buffers.welm_mtp_oe_prev_prev_input_ids[:num_tokens]
            )
            forward_batch.welm_mtp_oe_output_prev_input_ids = (
                buffers.welm_mtp_oe_output_prev_input_ids[:num_tokens]
            )
            forward_batch.welm_mtp_oe_hash_out_batch_major = (
                buffers.welm_mtp_oe_hash_out_batch_major[:num_tokens]
            )
            forward_batch.welm_mtp_draft_input_ids = (
                buffers.welm_mtp_draft_input_ids[:, :num_tokens]
            )

        self.draft_extend_attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )
        self._set_welmv4_mtp_mirror_metadata(bs)
        self._prepare_v1_base_kv_capture_metadata(forward_batch, bs)
        forward_batch.welm_mtp_base_kv_metadata_bs = bs

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            if global_num_tokens is not None:
                global_num_tokens.fill_(num_tokens)
            if global_num_tokens_for_logprob is not None:
                global_num_tokens_for_logprob.fill_(num_tokens)
            forward_batch.global_dp_buffer_len = global_dp_buffer_len
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            hidden_states_backup = forward_batch.spec_info.hidden_states
            forward_batch.welm_mtp_draft_graph_select_fn = (
                lambda logits, step: self._select_topk(logits, step, bs)
            )
            forward_batch.welm_mtp_skip_draft_proposal_build = True
            try:
                if self._is_welmv4_mtp_v1_draft_model():
                    self.eagle_worker._run_welmv4_mtp_recursive_draft_proposal(
                        forward_batch,
                        skip_attn_backend_init=True,
                        draft_path="decode_graph",
                    )
                else:
                    self.eagle_worker._run_welmv4_mtp_merged_extend_draft(
                        forward_batch,
                        first_input_ids,
                        skip_attn_backend_init=True,
                        first_query_hashed_inputs=(
                            None
                            if buffers.welm_mtp_query_hash_inputs is None
                            else buffers.welm_mtp_query_hash_inputs[:, :bs]
                        ),
                        first_query_history_state=(
                            None
                            if buffers.welm_mtp_oe_entry_history is None
                            else buffers.welm_mtp_oe_entry_history[:bs]
                        ),
                        draft_path="decode_graph",
                    )
            finally:
                forward_batch.welm_mtp_draft_graph_select_fn = None
                forward_batch.welm_mtp_skip_draft_proposal_build = False
            out = (
                forward_batch.spec_info.draft_proposal_parent_list,
                forward_batch.spec_info.draft_proposal_top_scores_index,
                forward_batch.spec_info.draft_proposal_tokens,
                forward_batch.spec_info.topk_p,
                forward_batch.spec_info.topk_index,
                forward_batch.spec_info.hidden_states,
                forward_batch.spec_info.draft_probs,
                forward_batch.spec_info.welm_mtp_draft_topk_indices,
                forward_batch.spec_info.welm_mtp_draft_topk_values,
            )
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return out

        self.deepep_adapter.capture(is_extend_in_batch=True)
        self._capture_init(run_once)
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )
        set_global_graph_memory_pool(graph.pool())
        self.forward_batches[bs] = forward_batch
        return graph, out

    def replay(
        self,
        forward_batch: ForwardBatch,
        first_input_ids: torch.Tensor,
        *,
        first_query_hashed_inputs: Optional[torch.Tensor],
        first_query_history_state: Optional[torch.Tensor],
    ) -> None:
        self.deepep_adapter.replay()
        buffers = self.buffers
        raw_bs = int(forward_batch.batch_size)
        raw_num_tokens = int(forward_batch.input_ids.numel())

        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            index = bisect.bisect_left(
                self.capture_bs, max_num_tokens // self.num_tokens_per_bs
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        graph_num_tokens = bs * self.num_tokens_per_bs
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        accepted_lens_cpu = [int(x) for x in spec_info.num_accepted_tokens_cpu]
        if self._is_welmv4_mtp_v1_draft_model():
            padded_lens_cpu = [self.num_tokens_per_bs] * bs
        else:
            padded_lens_cpu = self._make_padded_accept_lens(accepted_lens_cpu, bs)
        if spec_info.hidden_states is None:
            raise RuntimeError(
                "WeLM MTP draft proposal graph replay requires hidden states."
            )
        cached_hash = getattr(forward_batch, "welm_oe_decode_hashed_inputs", None)
        if buffers.welm_mtp_oe_hash_out is not None and cached_hash is None:
            raise RuntimeError(
                "WeLM MTP draft proposal graph replay is missing dense OE hashes."
            )

        buffers.seq_lens[:bs].fill_(self.seq_len_fill_value)
        buffers.seq_lens_cpu[:bs].fill_(self.seq_len_fill_value)
        buffers.req_pool_indices[:bs].zero_()
        buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        buffers.first_input_ids[:bs].zero_()
        if buffers.welm_mtp_oe_hash_out is not None:
            buffers.welm_mtp_query_hash_inputs[:, :bs].zero_()
            buffers.welm_mtp_oe_entry_history[:bs].zero_()
            buffers.welm_mtp_oe_work_history_a[:graph_num_tokens].zero_()
            buffers.welm_mtp_oe_work_history_b[:graph_num_tokens].zero_()
            buffers.welm_mtp_oe_parent_indices[:graph_num_tokens].zero_()
            buffers.welm_mtp_oe_prev_input_ids[:graph_num_tokens].zero_()
            buffers.welm_mtp_oe_prev_prev_input_ids[:graph_num_tokens].zero_()
            buffers.welm_mtp_oe_output_prev_input_ids[:graph_num_tokens].zero_()
            buffers.welm_mtp_oe_hash_out_batch_major[:graph_num_tokens].zero_()

        if self._is_welmv4_mtp_v1_draft_model():
            self._copy_v1_runtime_flat_layout(
                input_ids=forward_batch.input_ids,
                out_cache_loc=forward_batch.out_cache_loc,
                positions=forward_batch.positions,
                hidden_states=spec_info.hidden_states,
                mirrored_kv_indices=getattr(spec_info, "mirrored_kv_indices", None),
                cached_hash=cached_hash,
                accepted_lens_cpu=accepted_lens_cpu,
                raw_bs=raw_bs,
                graph_bs=bs,
                graph_num_tokens=graph_num_tokens,
            )
        else:
            if padded_lens_cpu is None:
                raise RuntimeError(
                    "Cannot fit WeLM MTP draft proposal accepted lengths into graph "
                    f"shape: bs={bs}, accepted_lens={accepted_lens_cpu}."
                )
            self._copy_runtime_flat_layout(
                input_ids=forward_batch.input_ids,
                out_cache_loc=forward_batch.out_cache_loc,
                positions=forward_batch.positions,
                hidden_states=spec_info.hidden_states,
                mirrored_kv_indices=getattr(spec_info, "mirrored_kv_indices", None),
                cached_hash=cached_hash,
                accepted_lens_cpu=accepted_lens_cpu,
                padded_lens_cpu=padded_lens_cpu,
                raw_bs=raw_bs,
                graph_num_tokens=graph_num_tokens,
            )
        self._copy_welmv4_mtp_mirror_kv_states(forward_batch, graph_num_tokens)
        if self._is_welmv4_mtp_v1_draft_model():
            self._copy_v1_recursive_draft_state(forward_batch, raw_bs, bs)
        self._copy_sampling_params(forward_batch, raw_bs, bs)
        self._copy_sampling_randomness(forward_batch, raw_bs, bs)
        buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        if forward_batch.seq_lens_cpu is not None:
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
        buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        buffers.first_input_ids[:raw_bs].copy_(first_input_ids)
        padded_lens = torch.tensor(
            padded_lens_cpu,
            dtype=torch.int32,
            device=buffers.extend_seq_lens.device,
        )
        buffers.extend_seq_lens[:bs].copy_(padded_lens)
        buffers.num_accepted_tokens[:bs].copy_(padded_lens)
        buffers.num_accepted_drafts[:bs].copy_(padded_lens - 1)
        self.extend_seq_lens_cpu[:bs] = padded_lens_cpu

        if buffers.welm_mtp_oe_hash_out is not None:
            if self._is_welmv4_mtp_v1_draft_model():
                entry_history_state = getattr(
                    spec_info, "welm_mtp_oe_history_state", None
                )
                if entry_history_state is None:
                    raise RuntimeError(
                        "WeLM MTP V1 draft proposal graph replay is missing OE "
                        "entry history state."
                    )
                buffers.welm_mtp_oe_entry_history[:raw_bs].copy_(
                    entry_history_state[:raw_bs]
                )
            else:
                if first_query_hashed_inputs is None or first_query_history_state is None:
                    raise RuntimeError(
                        "WeLM MTP draft proposal graph replay is missing query OE state."
                    )
                buffers.welm_mtp_query_hash_inputs[:, :raw_bs].copy_(
                    first_query_hashed_inputs
                )
                buffers.welm_mtp_oe_entry_history[:raw_bs].copy_(
                    first_query_history_state
                )

        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(graph_num_tokens)
            buffers.global_num_tokens_for_logprob_gpu.fill_(graph_num_tokens)

        replay_forward_batch = self.forward_batches[bs]
        replay_forward_batch.spec_info.extend_seq_lens_cpu = list(
            self.extend_seq_lens_cpu[:bs]
        )
        replay_forward_batch.spec_info.extend_seq_lens_tensor = (
            buffers.extend_seq_lens[:bs]
        )
        replay_forward_batch.seq_lens_sum = (
            forward_batch.seq_lens_sum
            + (bs - raw_bs) * int(self.seq_len_fill_value)
        )
        self.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_sum=replay_forward_batch.seq_lens_sum,
            encoder_lens=None,
            forward_mode=replay_forward_batch.forward_mode,
            spec_info=replay_forward_batch.spec_info,
            seq_lens_cpu=buffers.seq_lens_cpu,
        )
        self._set_welmv4_mtp_mirror_metadata(bs)
        self._prepare_v1_base_kv_replay_metadata(replay_forward_batch, bs, raw_bs)

        if self._is_welmv4_mtp_v1_draft_model():
            _dump_welm_mtp_draft_graph_io(
                "unified_v1_replay_input",
                {
                    "raw_bs": raw_bs,
                    "bs": bs,
                    "raw_num_tokens": raw_num_tokens,
                    "graph_num_tokens": graph_num_tokens,
                    "input_ids": buffers.input_ids[:graph_num_tokens],
                    "out_cache_loc": buffers.out_cache_loc[:graph_num_tokens],
                    "positions": buffers.positions[:graph_num_tokens],
                    "custom_last_index": buffers.custom_last_index[:bs],
                    "custom_last_cache_loc": buffers.custom_last_cache_loc[:bs],
                    "seq_lens": buffers.seq_lens[:bs],
                    "seq_lens_cpu": buffers.seq_lens_cpu[:bs],
                    "extend_seq_lens": buffers.extend_seq_lens[:bs],
                    "draft_seq_lens": buffers.draft_seq_lens[:bs],
                    "draft_seq_lens_cpu": buffers.draft_seq_lens_cpu[:bs],
                    "draft_out_cache_loc": buffers.draft_out_cache_loc[
                        : bs * self.v1_draft_num_cache_locs
                    ],
                    "draft_positions": buffers.draft_positions[: bs * self.topk],
                    "topk_p": spec_info.topk_p,
                    "topk_index": spec_info.topk_index,
                    "hidden_states": spec_info.hidden_states,
                    "base_positions": getattr(
                        spec_info, "welm_mtp_base_positions", None
                    ),
                    "oe_entry_history": (
                        None
                        if buffers.welm_mtp_oe_entry_history is None
                        else buffers.welm_mtp_oe_entry_history[:bs]
                    ),
                },
            )

        self.raw_bs = raw_bs
        self.bs = bs
        self._replay()
        (
            draft_proposal_parent_list,
            draft_proposal_top_scores_index,
            draft_proposal_tokens,
            topk_p,
            topk_index,
            hidden_states,
            draft_probs,
            draft_topk_indices,
            draft_topk_values,
        ) = self.output_buffers[bs]

        if self._is_welmv4_mtp_v1_draft_model():
            _dump_welm_mtp_draft_graph_io(
                "unified_v1_replay_output",
                {
                    "raw_bs": raw_bs,
                    "bs": bs,
                    "draft_proposal_parent_list": draft_proposal_parent_list,
                    "draft_proposal_top_scores_index": draft_proposal_top_scores_index,
                    "draft_proposal_tokens": draft_proposal_tokens,
                    "topk_p": topk_p,
                    "topk_index": topk_index,
                    "hidden_states": hidden_states,
                },
            )

        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        if self._is_welmv4_mtp_v1_draft_model():
            spec_info.topk_p = topk_p[:raw_bs]
            spec_info.topk_index = topk_index[:raw_bs]
            spec_info.hidden_states = hidden_states[:raw_bs]
            spec_info.draft_probs = None if draft_probs is None else draft_probs[:raw_bs]
            spec_info.welm_mtp_draft_topk_indices = (
                None if draft_topk_indices is None else draft_topk_indices[:raw_bs]
            )
            spec_info.welm_mtp_draft_topk_values = (
                None if draft_topk_values is None else draft_topk_values[:raw_bs]
            )
            spec_info.draft_proposal_parent_list = (
                None
                if draft_proposal_parent_list is None
                else draft_proposal_parent_list[:raw_bs]
            )
            spec_info.draft_proposal_top_scores_index = (
                None
                if draft_proposal_top_scores_index is None
                else draft_proposal_top_scores_index[:raw_bs]
            )
            spec_info.draft_proposal_tokens = (
                None if draft_proposal_tokens is None else draft_proposal_tokens[:raw_bs]
            )
        else:
            spec_info.topk_p = topk_p[:raw_bs].clone()
            spec_info.topk_index = topk_index[:raw_bs].clone()
            spec_info.hidden_states = hidden_states[:raw_bs].clone()
            spec_info.draft_probs = (
                None if draft_probs is None else draft_probs[:raw_bs].clone()
            )
            spec_info.welm_mtp_draft_topk_indices = (
                None
                if draft_topk_indices is None
                else draft_topk_indices[:raw_bs].clone()
            )
            spec_info.welm_mtp_draft_topk_values = (
                None if draft_topk_values is None else draft_topk_values[:raw_bs].clone()
            )
            spec_info.draft_proposal_parent_list = (
                None
                if draft_proposal_parent_list is None
                else draft_proposal_parent_list[:raw_bs].clone()
            )
            spec_info.draft_proposal_top_scores_index = (
                None
                if draft_proposal_top_scores_index is None
                else draft_proposal_top_scores_index[:raw_bs].clone()
            )
            spec_info.draft_proposal_tokens = (
                None
                if draft_proposal_tokens is None
                else draft_proposal_tokens[:raw_bs].clone()
            )
        spec_info.welm_mtp_has_draft_proposal = True
        if self._is_welmv4_mtp_v1_draft_model() and self.topk == 1:
            spec_info.welm_mtp_base_positions = buffers.positions[
                self.num_tokens_per_bs
                - 1 : raw_bs * self.num_tokens_per_bs : self.num_tokens_per_bs
            ]
        else:
            base_position_index = buffers.custom_last_index[:raw_bs]
            spec_info.welm_mtp_base_positions = buffers.positions[
                base_position_index
            ].clone()
        forward_batch.mtp_step_idx = 0
