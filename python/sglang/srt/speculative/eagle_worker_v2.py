import contextlib
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.models.welm_perf_opt import (
    get_welm_oe_hash_config,
    should_use_welm_oe_hash_kernel,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,
    fill_accepted_out_cache_loc,
    fill_new_verified_id,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    generate_token_bitmask,
    load_token_map,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()

logger = logging.getLogger(__name__)
_WELM_MTP_DUMP_ENABLED = os.environ.get(
    "SGLANG_DUMP_MTP_ACTIVATIONS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_WELM_VERIFY_AFTER_DUMP_ENABLED = os.environ.get(
    "SGLANG_DUMP_VERIFY_AFTER_MTP_METADATA", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_WELM_TRUE_VALUES = {"1", "true", "yes", "on"}
_WELM_DISABLE_TARGET_VERIFY_GRAPH_FOR_DUMP = (
    os.environ.get("SGLANG_WELMV4_DISABLE_TARGET_VERIFY_GRAPH_FOR_DUMP", "0")
    .strip()
    .lower()
    in _WELM_TRUE_VALUES
)
_WELM_VERIFY_AFTER_EVENT_COUNTERS = {}


def _to_cpu_payload(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu_payload(v) for v in value)
    return value


def _dump_verify_after_event(name: str, payload: dict) -> None:
    if not _WELM_VERIFY_AFTER_DUMP_ENABLED:
        return
    root = os.environ.get(
        "SGLANG_DUMP_VERIFY_AFTER_MTP_DIR",
        os.environ.get("SGLANG_DUMP_MTP_ACTIVATIONS_DIR", "./sglang_mtp_dump"),
    )
    rank = os.environ.get("RANK", "0")
    dump_dir = Path(root) / f"Rank{rank}_pid{os.getpid()}" / "verify_after_events"
    dump_dir.mkdir(parents=True, exist_ok=True)
    idx = _WELM_VERIFY_AFTER_EVENT_COUNTERS.get(name, 0)
    _WELM_VERIFY_AFTER_EVENT_COUNTERS[name] = idx + 1
    event = {"name": name, "index": idx, **payload}
    torch.save(_to_cpu_payload(event), dump_dir / f"{name}_{idx:05d}.pt")


@contextlib.contextmanager
def _welmv4_mtp_dump_context(context: str):
    if not _WELM_MTP_DUMP_ENABLED:
        yield
        return
    env_key = "SGLANG_DUMP_MTP_ACTIVATIONS_CONTEXT"
    previous = os.environ.get(env_key)
    os.environ[env_key] = context
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


def _flush_welmv4_mtp_graph_dump(
    context: str, first_dim_limit: Optional[int] = None
) -> list[Path]:
    if not _WELM_MTP_DUMP_ENABLED:
        return []
    from sglang.srt.models import welmv4_nextn as welmv4_nextn_module

    return welmv4_nextn_module._flush_mtp_graph_dump_pass(
        context, first_dim_limit=first_dim_limit
    )


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class EagleDraftWorker(BaseDraftWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Init draft worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx,
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            # Init draft worker
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            eagle_config = getattr(
                self.draft_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        self.init_token_map()
        self.init_lm_head()
        self._welmv4_mtp_base_kv_attn_backend = None

        # Init attention backend and cuda graphs
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.init_attention_backend()
            self.init_cuda_graphs()

        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)

    def _is_welmv4_mtp_draft_model(self) -> bool:
        architectures = getattr(
            self.draft_runner.model_config.hf_config, "architectures", []
        )
        return bool(architectures and architectures[0] == "WeLMV4MoeForCausalLMNextN")

    def _get_welmv4_mtp_base_positions(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        custom_last_index = getattr(forward_batch, "custom_last_index", None)
        if custom_last_index is not None:
            return forward_batch.positions[custom_last_index].clone()

        if forward_batch.extend_seq_lens is not None:
            last_query_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            return forward_batch.positions[last_query_indices].clone()

        return (forward_batch.seq_lens - 1).to(forward_batch.positions.dtype)

    def _init_welmv4_mtp_base_kv_decode_metadata(
        self, forward_batch: ForwardBatch, base_query_count: Optional[int] = None
    ) -> Optional[Tuple[int, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]]:
        attn_backend = self._get_welmv4_mtp_base_kv_decode_attn_backend()
        if getattr(forward_batch, "welm_mtp_base_kv_metadata_prepared", False):
            forward_batch.attn_backend = attn_backend
            return None
        batch_size = forward_batch.batch_size
        num_queries = forward_batch.input_ids.numel()
        if base_query_count is None:
            base_query_count = batch_size
        if base_query_count <= 0:
            raise RuntimeError(
                "WeLMV4 MTP base query count is not aligned with the draft "
                f"batch: {base_query_count=} {batch_size=}"
            )
        restore_state = None
        if num_queries != batch_size or base_query_count != batch_size:
            if num_queries % base_query_count != 0:
                raise RuntimeError(
                    "WeLMV4 MTP draft decode query count must be a multiple of "
                    f"base query count: {num_queries=} {base_query_count=}"
                )
            if base_query_count <= batch_size:
                base_seq_lens = forward_batch.seq_lens[:base_query_count]
                base_seq_lens_cpu = (
                    forward_batch.seq_lens_cpu[:base_query_count]
                    if forward_batch.seq_lens_cpu is not None
                    else None
                )
                base_req_pool_indices = forward_batch.req_pool_indices[
                    :base_query_count
                ]
            elif base_query_count % batch_size == 0:
                base_repeat = base_query_count // batch_size
                base_seq_lens = forward_batch.seq_lens.repeat_interleave(base_repeat)
                base_seq_lens_cpu = (
                    forward_batch.seq_lens_cpu.repeat_interleave(base_repeat)
                    if forward_batch.seq_lens_cpu is not None
                    else None
                )
                base_req_pool_indices = (
                    forward_batch.req_pool_indices.repeat_interleave(base_repeat)
                )
            else:
                raise RuntimeError(
                    "WeLMV4 MTP base query count cannot be derived from the draft "
                    f"batch: {base_query_count=} {batch_size=}"
                )
            repeat = num_queries // base_query_count
            restore_state = (
                forward_batch.batch_size,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens_sum,
            )
            forward_batch.batch_size = num_queries
            forward_batch.seq_lens = base_seq_lens.repeat_interleave(repeat)
            if base_seq_lens_cpu is not None:
                forward_batch.seq_lens_cpu = base_seq_lens_cpu.repeat_interleave(repeat)
            forward_batch.req_pool_indices = base_req_pool_indices.repeat_interleave(
                repeat
            )
            forward_batch.seq_lens_sum = int(forward_batch.seq_lens.sum().item())

        had_base_kv_attr = hasattr(forward_batch, "welm_mtp_use_base_kv_cache")
        base_kv_attr_backup = getattr(
            forward_batch, "welm_mtp_use_base_kv_cache", None
        )
        forward_batch.welm_mtp_use_base_kv_cache = True
        spec_info_backup = forward_batch.spec_info
        clear_spec_info_for_metadata = not hasattr(attn_backend, "fa_impl_ver")
        if clear_spec_info_for_metadata:
            forward_batch.spec_info = None
        try:
            attn_backend.init_forward_metadata(forward_batch)
        finally:
            forward_batch.spec_info = spec_info_backup
            if had_base_kv_attr:
                forward_batch.welm_mtp_use_base_kv_cache = base_kv_attr_backup
            else:
                delattr(forward_batch, "welm_mtp_use_base_kv_cache")
        forward_batch.attn_backend = attn_backend
        return restore_state

    def _get_welmv4_mtp_base_kv_decode_attn_backend(self):
        return (
            self.draft_runner.decode_attn_backend
            if self.draft_runner.server_args.enable_pdmux
            else self.draft_runner.attn_backend
        )

    def _restore_welmv4_mtp_base_kv_decode_metadata(
        self,
        forward_batch: ForwardBatch,
        restore_state: Optional[
            Tuple[int, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]
        ],
    ) -> None:
        if restore_state is None:
            return
        (
            forward_batch.batch_size,
            forward_batch.seq_lens,
            forward_batch.seq_lens_cpu,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens_sum,
        ) = restore_state

    def _expand_welmv4_mtp_base_positions(
        self,
        base_positions: torch.Tensor,
        num_queries: int,
    ) -> torch.Tensor:
        if base_positions.numel() == num_queries:
            return base_positions
        if num_queries % base_positions.numel() != 0:
            raise RuntimeError(
                "WeLMV4 MTP base positions cannot be expanded to draft queries: "
                f"{base_positions.numel()=} {num_queries=}"
            )
        return base_positions.repeat_interleave(num_queries // base_positions.numel())

    def _get_welmv4_mtp_selected_parent_indices(
        self,
        i: int,
        tree_info: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if i == 0:
            return None
        topk_cs_index = tree_info[2] - (self.topk**2 * (i - 1) + self.topk)
        parent_offsets = torch.arange(
            0,
            topk_cs_index.shape[0] * self.topk,
            step=self.topk,
            dtype=topk_cs_index.dtype,
            device=device,
        ).repeat_interleave(self.topk)
        return topk_cs_index.flatten() // self.topk + parent_offsets

    def _should_use_welmv4_mtp_oe_hash_kernel(self) -> bool:
        return self._is_welmv4_mtp_draft_model() and should_use_welm_oe_hash_kernel(
            self.draft_runner.model_config
        )

    def _welmv4_mtp_oe_hash_config(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        oe_grams, oe_vocab_sizes = get_welm_oe_hash_config(
            self.draft_runner.model_config
        )
        if not oe_grams:
            return oe_grams, oe_vocab_sizes, 0
        # Target-verify sees the current verified token both in draft_token_ids
        # and in the rolling history state. Keep one extra column so the target
        # verify hash kernel can skip that current-token column and still read
        # max_gram - 1 tokens before it.
        return oe_grams, oe_vocab_sizes, max(int(g) for g in oe_grams)

    def _welmv4_mtp_oe_prefix_width(self) -> int:
        oe_grams, _, _ = self._welmv4_mtp_oe_hash_config()
        return max((int(g) for g in oe_grams), default=1) - 1

    @staticmethod
    def _flatten_welmv4_mtp_hash_prefixes(prefix_rows) -> list[int]:
        return [int(token) for row in prefix_rows for token in row]

    @staticmethod
    def _welmv4_mtp_hash_prefix_rows_from_context(
        oe_context,
        history_width: int,
    ) -> list[list[int]]:
        prefix_rows = getattr(oe_context, "hash_prefixes", None)
        if prefix_rows is None:
            raise RuntimeError(
                "WeLMV4 MTP fused OE hash path requires CPU prefix rows at "
                "stage entry."
            )
        rows = [list(row) for row in prefix_rows[:history_width]]
        if not rows:
            return rows
        while len(rows) < history_width:
            rows.append([0] * len(rows[0]))
        return rows

    @staticmethod
    def _welmv4_mtp_prefix_rows_from_reqs(
        reqs,
        prefix_width: int,
        skip_latest_output: bool,
    ) -> list[list[int]]:
        if prefix_width <= 0:
            return []
        rows = []
        for lag in range(1, prefix_width + 1):
            row = []
            for req in reqs:
                ids = req.origin_input_ids + req.output_ids
                pos = len(ids) - lag - (1 if skip_latest_output else 0)
                row.append(ids[pos] if pos >= 0 else 0)
            rows.append(row)
        return rows

    def _init_welmv4_mtp_oe_history_from_context(
        self,
        oe_context,
        *,
        device: torch.device,
        first_token_ids: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        prefix_rows: Optional[list[list[int]]] = None,
    ) -> Optional[torch.Tensor]:
        if not self._should_use_welmv4_mtp_oe_hash_kernel() or oe_context is None:
            return None
        _, _, history_width = self._welmv4_mtp_oe_hash_config()
        if history_width <= 0:
            return None
        if prefix_rows is None:
            prefix_rows = self._welmv4_mtp_hash_prefix_rows_from_context(
                oe_context,
                history_width,
            )
        else:
            prefix_rows = [list(row) for row in prefix_rows[:history_width]]
            if prefix_rows:
                while len(prefix_rows) < history_width:
                    prefix_rows.append([0] * len(prefix_rows[0]))
        batch_size = len(prefix_rows[0]) if prefix_rows else 0
        if batch_size == 0:
            return None
        if out is None:
            out = torch.empty(
                (batch_size, history_width), device=device, dtype=torch.int64
            )
        elif out.shape[0] < batch_size or out.shape[1] != history_width:
            raise RuntimeError(
                "WeLMV4 MTP OE history output has incompatible shape: "
                f"{tuple(out.shape)} vs ({batch_size}, {history_width})."
            )
        history_out = out[:batch_size]
        flat_prefixes = self._flatten_welmv4_mtp_hash_prefixes(prefix_rows)
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_mtp_init_history_from_prefixes_cuda,
        )

        if first_token_ids is not None:
            if first_token_ids.numel() != batch_size:
                raise RuntimeError(
                    "WeLMV4 MTP OE first-token count mismatch: "
                    f"{first_token_ids.numel()} vs {batch_size}."
                )
        welm_oe_hash_mtp_init_history_from_prefixes_cuda(
            flat_prefixes,
            history_out,
            first_token_ids=first_token_ids,
        )
        return history_out

    def _get_welmv4_mtp_hash_out(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        num_branches: int,
    ) -> torch.Tensor:
        num_tokens = int(input_ids.numel())
        buffer = getattr(forward_batch, "welm_mtp_oe_hash_out", None)
        if (
            buffer is not None
            and buffer.shape[0] == num_branches
            and buffer.shape[1] >= num_tokens
        ):
            return buffer[:, :num_tokens]
        return torch.empty(
            (num_branches, num_tokens), device=input_ids.device, dtype=torch.int64
        )

    def _get_welmv4_mtp_next_history_out(
        self,
        forward_batch: ForwardBatch,
        rows: int,
        history_width: int,
        *,
        step_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if step_idx is not None:
            work_buffers = getattr(forward_batch, "welm_mtp_oe_work_history", None)
            if work_buffers:
                buffer = work_buffers[step_idx % len(work_buffers)]
                if buffer.shape[0] >= rows and buffer.shape[1] == history_width:
                    return buffer[:rows]
        buffer = getattr(forward_batch, "welm_mtp_oe_next_history", None)
        if (
            buffer is not None
            and buffer.shape[0] >= rows
            and buffer.shape[1] == history_width
        ):
            return buffer[:rows]
        return torch.empty((rows, history_width), device=self.device, dtype=torch.int64)

    def _prepare_welmv4_mtp_draft_decode_entry_history(
        self,
        forward_batch: ForwardBatch,
        draft_input: EagleDraftInput,
    ) -> Optional[torch.Tensor]:
        if (
            not self._should_use_welmv4_mtp_oe_hash_kernel()
            or forward_batch.forward_mode.is_idle()
        ):
            return None
        history_state = getattr(draft_input, "welm_mtp_oe_history_state", None)
        if history_state is not None:
            forward_batch.spec_info.welm_mtp_oe_history_state = history_state
            return history_state
        first_token_ids = getattr(draft_input, "verified_id", None)
        if first_token_ids is None:
            raise RuntimeError(
                "WeLMV4 MTP fused OE hash draft-decode path requires verified_id."
            )
        prefix_rows = getattr(draft_input, "welm_mtp_oe_prefix_rows", None)
        history_state = self._init_welmv4_mtp_oe_history_from_context(
            forward_batch.oe_context,
            device=forward_batch.input_ids.device,
            first_token_ids=first_token_ids,
            prefix_rows=prefix_rows,
        )
        if history_state is None:
            raise RuntimeError(
                "WeLMV4 MTP fused OE hash draft-decode path is missing CPU "
                "prefix state."
            )
        draft_input.welm_mtp_oe_history_state = history_state
        forward_batch.spec_info.welm_mtp_oe_history_state = history_state
        return history_state

    def _prepare_welmv4_mtp_segment_hash_inputs_from_prefixes(
        self,
        forward_batch: ForwardBatch,
    ) -> None:
        from sglang.jit_kernel.welm_oe import welm_oe_hash_segments_from_prefixes_cuda

        input_ids = forward_batch.input_ids
        oe_grams, oe_vocab_sizes, _ = self._welmv4_mtp_oe_hash_config()
        hashed_out = self._get_welmv4_mtp_hash_out(
            forward_batch, input_ids, len(oe_vocab_sizes)
        )
        num_segments = int(forward_batch.extend_seq_lens.numel())
        prefix_width = self._welmv4_mtp_oe_prefix_width()
        prefixes = [0] * (prefix_width * num_segments)
        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu is not None:
            real_num_tokens = sum(int(x) for x in extend_seq_lens_cpu)
            if real_num_tokens < input_ids.numel():
                hashed_out[:, real_num_tokens:].zero_()
        welm_oe_hash_segments_from_prefixes_cuda(
            input_ids,
            forward_batch.extend_start_loc,
            forward_batch.extend_seq_lens,
            prefixes,
            oe_grams,
            oe_vocab_sizes,
            hashed_out,
            self.draft_runner.model_config.vocab_size,
        )
        forward_batch.welm_oe_decode_hashed_inputs = hashed_out

    def _compute_welmv4_mtp_target_verify_hash_inputs(
        self,
        input_ids: torch.Tensor,
        tree_mask: torch.Tensor,
        seq_lens: torch.Tensor,
        history_state: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_mtp_target_verify_from_history_cuda,
        )

        oe_grams, oe_vocab_sizes, _ = self._welmv4_mtp_oe_hash_config()
        batch_size = int(seq_lens.numel())
        real_num_tokens = batch_size * int(draft_token_num)
        hashed_out = torch.empty(
            (len(oe_vocab_sizes), int(input_ids.numel())),
            device=input_ids.device,
            dtype=torch.int64,
        )
        if real_num_tokens < input_ids.numel():
            hashed_out[:, real_num_tokens:].zero_()
        welm_oe_hash_mtp_target_verify_from_history_cuda(
            input_ids[:real_num_tokens],
            tree_mask,
            seq_lens,
            history_state[:batch_size],
            oe_grams,
            oe_vocab_sizes,
            hashed_out[:, :real_num_tokens],
            self.target_worker.model_config.vocab_size,
            int(draft_token_num),
        )
        return hashed_out

    def _prepare_welmv4_mtp_target_verify_hash_inputs(
        self,
        forward_batch: ForwardBatch,
        history_state: Optional[torch.Tensor],
    ) -> None:
        if (
            not self._should_use_welmv4_mtp_oe_hash_kernel()
            or history_state is None
            or forward_batch.forward_mode.is_idle()
        ):
            return
        spec_info = forward_batch.spec_info
        forward_batch.welm_oe_decode_hashed_inputs = (
            self._compute_welmv4_mtp_target_verify_hash_inputs(
                forward_batch.input_ids,
                spec_info.custom_mask,
                forward_batch.seq_lens,
                history_state,
                int(spec_info.draft_token_num),
            )
        )

    def _precompute_welmv4_mtp_target_verify_hash_inputs(
        self,
        verify_input: EagleVerifyInput,
        batch: ModelWorkerBatch,
    ) -> None:
        history_state = getattr(verify_input, "welm_mtp_oe_history_state", None)
        if (
            not self._should_use_welmv4_mtp_oe_hash_kernel()
            or history_state is None
            or batch.forward_mode.is_idle()
        ):
            return
        verify_input.welm_mtp_oe_hashed_inputs = (
            self._compute_welmv4_mtp_target_verify_hash_inputs(
                verify_input.draft_token,
                verify_input.custom_mask,
                batch.seq_lens,
                history_state,
                int(verify_input.draft_token_num),
            )
        )

    def _prepare_welmv4_mtp_draft_extend_hash_inputs(
        self,
        forward_batch: ForwardBatch,
        accept_lens: torch.Tensor,
        accepted_draft_token_ids: Optional[torch.Tensor],
        entry_history_state: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            not self._should_use_welmv4_mtp_oe_hash_kernel()
            or forward_batch.forward_mode.is_idle()
        ):
            return None
        if accepted_draft_token_ids is None:
            raise RuntimeError(
                "WeLMV4 MTP fused OE hash draft-extend path requires accepted "
                "draft token ids from target verify."
            )
        if entry_history_state is None:
            raise RuntimeError(
                "WeLMV4 MTP fused OE hash draft-extend path is missing entry "
                "history state."
            )
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda,
        )

        input_ids = forward_batch.input_ids
        oe_grams, oe_vocab_sizes, history_width = self._welmv4_mtp_oe_hash_config()
        batch_size = int(accept_lens.numel())
        draft_token_num = self.speculative_num_draft_tokens
        hashed_out = self._get_welmv4_mtp_hash_out(
            forward_batch, input_ids, len(oe_vocab_sizes)
        )
        next_history = self._get_welmv4_mtp_next_history_out(
            forward_batch,
            batch_size,
            history_width,
        )
        welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda(
            input_ids,
            accepted_draft_token_ids.reshape(batch_size, -1),
            accept_lens,
            entry_history_state[:batch_size],
            oe_grams,
            oe_vocab_sizes,
            hashed_out,
            next_history,
            self.draft_runner.model_config.vocab_size,
            draft_token_num,
        )
        forward_batch.welm_oe_decode_hashed_inputs = hashed_out
        return next_history

    def _prepare_welmv4_mtp_draft_decode_hash_inputs(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        entry_history_state: torch.Tensor,
        draft_history_state: Optional[torch.Tensor],
        selected_parent_indices: Optional[torch.Tensor],
        base_query_count: int,
        step_idx: int,
    ) -> torch.Tensor:
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_mtp_draft_decode_from_history_cuda,
        )

        oe_grams, oe_vocab_sizes, history_width = self._welmv4_mtp_oe_hash_config()
        source_history = (
            entry_history_state if draft_history_state is None else draft_history_state
        )
        use_parent = selected_parent_indices is not None
        if selected_parent_indices is None:
            parent_indices = getattr(forward_batch, "welm_mtp_oe_parent_scratch", None)
            if parent_indices is None or parent_indices.numel() < input_ids.numel():
                parent_indices = torch.empty(
                    (input_ids.numel(),), device=input_ids.device, dtype=torch.int64
                )
        else:
            parent_indices = selected_parent_indices.to(dtype=torch.int64)
        hashed_out = self._get_welmv4_mtp_hash_out(
            forward_batch, input_ids, len(oe_vocab_sizes)
        )
        next_history = self._get_welmv4_mtp_next_history_out(
            forward_batch,
            int(input_ids.numel()),
            history_width,
            step_idx=step_idx,
        )
        welm_oe_hash_mtp_draft_decode_from_history_cuda(
            input_ids,
            source_history,
            parent_indices,
            oe_grams,
            oe_vocab_sizes,
            hashed_out,
            next_history,
            self.draft_runner.model_config.vocab_size,
            base_query_count,
            use_parent,
        )
        forward_batch.welm_oe_decode_hashed_inputs = hashed_out
        return next_history

    def _forward_welmv4_mtp_base_kv_decode(self, forward_batch: ForwardBatch):
        graph_runner_backup = self.draft_runner.graph_runner
        had_base_kv_attr = hasattr(forward_batch, "welm_mtp_use_base_kv_cache")
        base_kv_attr_backup = getattr(forward_batch, "welm_mtp_use_base_kv_cache", None)
        # Recursive WeLMV4 MTP draft decode reads the committed/base KV cache
        # populated from mirrored main-model KV, so the normal draft-decode
        # graph runner cannot be used here.
        forward_batch.welm_mtp_use_base_kv_cache = True
        self.draft_runner.graph_runner = None
        try:
            return self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
        finally:
            self.draft_runner.graph_runner = graph_runner_backup
            if had_base_kv_attr:
                forward_batch.welm_mtp_use_base_kv_cache = base_kv_attr_backup
            else:
                delattr(forward_batch, "welm_mtp_use_base_kv_cache")

    def _capture_for_decode(
        self,
        logits_output,
        draft_input: EagleDraftInput,
        forward_batch: Optional[ForwardBatch] = None,
        welmv4_mtp_base_positions: Optional[torch.Tensor] = None,
    ) -> None:
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states
        if self._is_welmv4_mtp_draft_model() and forward_batch is not None:
            if welmv4_mtp_base_positions is None:
                welmv4_mtp_base_positions = self._get_welmv4_mtp_base_positions(
                    forward_batch
                )
            draft_input.welm_mtp_base_positions = welmv4_mtp_base_positions

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.has_prefill_wrapper_verify = False
        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        if self.server_args.model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_aiter_draft_extend_graph = False
        if _is_hip:
            # Keep import local so non-HIP environments do not require aiter.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        supports_cuda_draft_extend_graph = (
            _is_cuda
            and isinstance(self.draft_extend_attn_backend, FlashAttentionBackend)
        ) or (
            (_is_cuda or _is_musa)
            and (
                isinstance(self.draft_extend_attn_backend, TritonAttnBackend)
                or isinstance(self.draft_extend_attn_backend, TRTLLMMLABackend)
            )
        )
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if (
            self.draft_extend_attn_backend
            and (
                _is_npu
                or supports_cuda_draft_extend_graph
                or supports_hip_aiter_draft_extend_graph
            )
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    def draft(self, model_worker_batch: ModelWorkerBatch):
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            model_worker_batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        cached_mtp_oe_history = getattr(
            draft_input, "welm_mtp_oe_history_state", None
        )
        needs_mtp_oe_history_reinit = (
            cached_mtp_oe_history is None
            or cached_mtp_oe_history.shape[0] != forward_batch.batch_size
        )
        if (
            self._should_use_welmv4_mtp_oe_hash_kernel()
            and not model_worker_batch.forward_mode.is_idle()
            and needs_mtp_oe_history_reinit
        ):
            draft_input.welm_mtp_oe_prefix_rows = (
                self._welmv4_mtp_prefix_rows_from_reqs(
                    model_worker_batch.reqs,
                    self._welmv4_mtp_oe_prefix_width(),
                    skip_latest_output=not hasattr(draft_input, "future_indices"),
                )
            )
            if cached_mtp_oe_history is not None:
                delattr(draft_input, "welm_mtp_oe_history_state")
        self._prepare_welmv4_mtp_draft_decode_entry_history(
            forward_batch,
            draft_input,
        )

        # Run draft
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch,
            )
            if _WELM_MTP_DUMP_ENABLED and self._is_welmv4_mtp_draft_model():
                _flush_welmv4_mtp_graph_dump(
                    "draft",
                    first_dim_limit=getattr(
                        self.cuda_graph_runner,
                        "raw_bs",
                        forward_batch.batch_size,
                    ),
                )
        else:
            if _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context("draft"):
                    if (
                        not forward_batch.forward_mode.is_idle()
                        and self.speculative_num_steps > 1
                        and not self._is_welmv4_mtp_draft_model()
                    ):
                        # Skip attention backend init for 1-step draft,
                        # `draft_forward` only does sample in this case.
                        self.draft_attn_backend.init_forward_metadata(forward_batch)
                    parent_list, top_scores_index, draft_tokens = self.draft_forward(
                        forward_batch
                    )
            else:
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                    and not self._is_welmv4_mtp_draft_model()
                ):
                    # Skip attention backend init for 1-step draft,
                    # `draft_forward` only does sample in this case.
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                parent_list, top_scores_index, draft_tokens = self.draft_forward(
                    forward_batch
                )

        if model_worker_batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        verify_input = EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )
        if (
            self._should_use_welmv4_mtp_oe_hash_kernel()
            and hasattr(draft_input, "welm_mtp_oe_history_state")
        ):
            verify_input.welm_mtp_oe_history_state = (
                draft_input.welm_mtp_oe_history_state
            )
        return verify_input

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        is_welmv4_mtp = self._is_welmv4_mtp_draft_model()
        use_welmv4_mtp_oe_hash = (
            is_welmv4_mtp
            and self._should_use_welmv4_mtp_oe_hash_kernel()
            and not forward_batch.forward_mode.is_idle()
        )
        welmv4_mtp_base_positions = None
        welmv4_mtp_entry_oe_history = None
        welmv4_mtp_draft_oe_history = None
        if is_welmv4_mtp and not forward_batch.forward_mode.is_idle():
            if use_welmv4_mtp_oe_hash:
                welmv4_mtp_entry_oe_history = getattr(
                    spec_info, "welm_mtp_oe_history_state", None
                )
                if welmv4_mtp_entry_oe_history is None:
                    welmv4_mtp_entry_oe_history = getattr(
                        forward_batch, "welm_mtp_oe_entry_history_state", None
                    )
                if welmv4_mtp_entry_oe_history is None:
                    raise RuntimeError(
                        "WeLMV4 MTP fused OE hash draft-decode path is missing "
                        "entry history state."
                    )
            welmv4_mtp_base_positions = spec_info.welm_mtp_base_positions
            if welmv4_mtp_base_positions is None:
                welmv4_mtp_base_positions = (forward_batch.seq_lens - 1).to(
                    forward_batch.positions.dtype
                )
            if self.topk > 1 and topk_p.shape[0] != welmv4_mtp_base_positions.numel():
                if welmv4_mtp_base_positions.numel() > topk_p.shape[0]:
                    welmv4_mtp_base_positions = welmv4_mtp_base_positions[
                        : topk_p.shape[0]
                    ]
                else:
                    padded_base_positions = (forward_batch.seq_lens - 1).to(
                        device=welmv4_mtp_base_positions.device,
                        dtype=welmv4_mtp_base_positions.dtype,
                    )[: topk_p.shape[0]]
                    padded_base_positions[: welmv4_mtp_base_positions.numel()] = (
                        welmv4_mtp_base_positions
                    )
                    welmv4_mtp_base_positions = padded_base_positions
                spec_info.welm_mtp_base_positions = welmv4_mtp_base_positions

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            step_cache_loc = out_cache_loc[i]
            step_positions = forward_batch.positions[: input_ids.numel()].clone()
            if step_cache_loc.numel() != input_ids.numel():
                step_cache_loc = step_cache_loc[: input_ids.numel()]
            forward_batch.out_cache_loc = step_cache_loc
            if is_welmv4_mtp:
                base_query_count = welmv4_mtp_base_positions.numel()
                if (
                    input_ids.numel() % base_query_count != 0
                    and input_ids.numel() % self.topk == 0
                    and base_query_count > input_ids.numel() // self.topk
                ):
                    base_query_count = input_ids.numel() // self.topk
                step_base_positions = welmv4_mtp_base_positions[:base_query_count]
                forward_batch.positions = self._expand_welmv4_mtp_base_positions(
                    step_base_positions, input_ids.numel()
                ).to(dtype=step_positions.dtype)
                selected_parent_indices = self._get_welmv4_mtp_selected_parent_indices(
                    i, tree_info, input_ids.device
                )
                if use_welmv4_mtp_oe_hash:
                    welmv4_mtp_draft_oe_history = (
                        self._prepare_welmv4_mtp_draft_decode_hash_inputs(
                            forward_batch,
                            input_ids,
                            welmv4_mtp_entry_oe_history,
                            welmv4_mtp_draft_oe_history,
                            selected_parent_indices,
                            base_query_count,
                            i,
                        )
                    )
                restore_mtp_metadata = self._init_welmv4_mtp_base_kv_decode_metadata(
                    forward_batch, base_query_count
                )
                if (
                    self.topk > 1
                    and hidden_states is not None
                    and hasattr(forward_batch, "hidden_states_backup")
                ):
                    forward_batch.hidden_states_backup = hidden_states
            else:
                forward_batch.positions = step_positions.add(1)
                forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            if is_welmv4_mtp:
                try:
                    logits_output = self._forward_welmv4_mtp_base_kv_decode(
                        forward_batch
                    )
                finally:
                    self._restore_welmv4_mtp_base_kv_decode_metadata(
                        forward_batch, restore_mtp_metadata
                    )
            else:
                logits_output = self.draft_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
        model_specific_states: Optional[dict] = None,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Construct input_ids
        if not batch.forward_mode.is_idle():
            use_welmv4_mtp_oe_hash = (
                self._should_use_welmv4_mtp_oe_hash_kernel()
                and batch.oe_context is not None
            )
            pt = 0
            for i, extend_len in enumerate(batch.extend_seq_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], next_token_ids[i].reshape(1))
                )
                pt += extend_len

        # Construct spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids,
            new_seq_lens=batch.seq_lens,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            model_specific_states=model_specific_states,
        )

        batch.spec_info = next_draft_input

        # Run forward
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        forward_batch.return_logprob = False
        if (
            self._should_use_welmv4_mtp_oe_hash_kernel()
            and batch.oe_context is not None
        ):
            self._prepare_welmv4_mtp_segment_hash_inputs_from_prefixes(forward_batch)
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds
        if _WELM_MTP_DUMP_ENABLED:
            with _welmv4_mtp_dump_context("draft_extend_for_prefill"):
                logits_output = self.draft_runner.forward(forward_batch).logits_output
        else:
            logits_output = self.draft_runner.forward(forward_batch).logits_output
        if _WELM_VERIFY_AFTER_DUMP_ENABLED:
            _dump_verify_after_event(
                "draft_extend_for_prefill",
                {
                    "seq_lens": batch.seq_lens,
                    "extend_seq_lens": getattr(batch, "extend_seq_lens", None),
                    "input_ids": batch.input_ids,
                    "next_token_ids": next_token_ids,
                },
            )
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")

        # Update spec_info for the next draft step
        self._capture_for_decode(logits_output, next_draft_input, forward_batch)
        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ModelWorkerBatch, batch_result: GenerationBatchResult
    ):
        next_draft_input = batch_result.next_draft_input
        welmv4_mtp_oe_entry_history = getattr(
            getattr(batch, "spec_info", None), "welm_mtp_oe_history_state", None
        )
        welmv4_mtp_oe_next_history = None
        # Batch 2: Draft extend
        draft_hidden_states = next_draft_input.hidden_states
        if draft_hidden_states is None:
            draft_hidden_states = batch_result.logits_output.hidden_states
            if self.topk > 1 and not batch.forward_mode.is_idle():
                raise RuntimeError(
                    "Spec v2 topk>1 draft extend requires packed hidden states "
                    "from the verify step."
                )
        draft_input = EagleDraftInput(
            hidden_states=draft_hidden_states,
            num_tokens_per_req=self.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
            model_specific_states=batch_result.logits_output.model_specific_states,
            mirrored_kv_indices=next_draft_input.mirrored_kv_indices,
        )
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )
        if _WELM_VERIFY_AFTER_DUMP_ENABLED:
            _dump_verify_after_event(
                "draft_extend_for_decode",
                {
                    "seq_lens": batch.seq_lens,
                    "accept_lens": batch_result.accept_lens,
                    "select_index": select_index,
                    "predict": batch_result.next_token_ids,
                },
            )

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
            )
            oe_context = getattr(forward_batch, "oe_context", None)
            if (
                self._is_welmv4_mtp_draft_model()
                and self._should_use_welmv4_mtp_oe_hash_kernel()
            ):
                welmv4_mtp_oe_next_history = (
                    self._prepare_welmv4_mtp_draft_extend_hash_inputs(
                        forward_batch,
                        batch_result.accept_lens,
                        getattr(
                            batch_result, "welm_mtp_accepted_draft_token_ids", None
                        ),
                        welmv4_mtp_oe_entry_history,
                    )
                )
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
        if welmv4_mtp_oe_next_history is not None:
            next_draft_input.welm_mtp_oe_history_state = welmv4_mtp_oe_next_history

        if forward_batch.spec_info.num_accepted_drafts is None:
            # `batch_result.accept_lens` already includes the bonus token, so use it
            # directly for `num_accepted_tokens` and subtract 1 for `num_accepted_drafts`.
            forward_batch.spec_info.num_accepted_drafts = batch_result.accept_lens - 1
            forward_batch.spec_info.num_accepted_tokens = batch_result.accept_lens

        # Run draft extend batch in the main compute stream
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
            if _WELM_MTP_DUMP_ENABLED and self._is_welmv4_mtp_draft_model():
                _flush_welmv4_mtp_graph_dump(
                    "draft_extend",
                    first_dim_limit=int(forward_batch.input_ids.shape[0]),
                )
        else:
            if _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context("draft_extend"):
                    draft_logits_output = self.draft_runner.forward(
                        forward_batch, skip_attn_backend_init=True
                    ).logits_output
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[
            select_index
        ]
        welmv4_mtp_base_positions = None
        if self._is_welmv4_mtp_draft_model() and self.topk > 1:
            welmv4_mtp_base_positions = forward_batch.positions[select_index].clone()

        # Construct the return values
        self._capture_for_decode(
            draft_logits_output,
            next_draft_input,
            forward_batch,
            welmv4_mtp_base_positions,
        )


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self._welmv4_mtp_target_verify_attn_backend = None

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def _get_welmv4_mtp_target_verify_attn_backend(self):
        if self._welmv4_mtp_target_verify_attn_backend is None:
            target_model_runner = self.target_worker.model_runner
            backend = target_model_runner._get_attention_backend_from_str("triton")
            init_forward_metadata = backend.init_forward_metadata

            def init_forward_metadata_with_swa_fallback(forward_batch):
                init_forward_metadata(forward_batch)
                if not forward_batch.forward_mode.is_target_verify():
                    return
                metadata = backend.forward_metadata
                if metadata.window_kv_indptr is None:
                    metadata.window_kv_indptr = metadata.kv_indptr
                    metadata.window_kv_indices = metadata.kv_indices
                    metadata.window_kv_offsets = torch.zeros(
                        metadata.kv_indptr.numel() - 1,
                        dtype=torch.int32,
                        device=metadata.kv_indptr.device,
                    )

            backend.init_forward_metadata = init_forward_metadata_with_swa_fallback
            self._welmv4_mtp_target_verify_attn_backend = backend
        return self._welmv4_mtp_target_verify_attn_backend

    def _should_use_welmv4_mtp_target_verify_attn_backend(self) -> bool:
        return False

    def _should_disable_welmv4_mtp_target_verify_graph(self) -> bool:
        return False

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Draft prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        model_worker_batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                        batch_output.logits_output.model_specific_states,
                    )
                )
                return batch_output
        else:
            if model_worker_batch.spec_info is None:
                model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self.target_worker.model_config.spec_hidden_size,
                    dtype=self.target_worker.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                verify_input: EagleVerifyInput = self.draft_worker.draft(
                    model_worker_batch
                )
            assert verify_input.is_verify_input()
            # Record a CUDA event after draft() GPU work is dispatched.
            # This event will be waited on by plan_stream in verify()
            # to ensure draft CUDA graph kernels finish before plan_stream
            # begins metadata preparation.
            if self.plan_stream:
                self._draft_done_event = torch.get_device_module(self.device).Event()
                self._draft_done_event.record()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch)
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.draft_worker._draft_extend_for_decode(
                    model_worker_batch, batch_output
                )
            return batch_output

    def _pack_accept_path_indices(
        self,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        if accept_index.numel() == 0:
            return accept_index.to(dtype=torch.long).flatten()

        bs, max_accept_len = accept_index.shape
        if max_accept_len > draft_token_num:
            raise RuntimeError(
                "Spec v2 accept path is wider than the dense draft row: "
                f"{max_accept_len=} {draft_token_num=}"
            )

        device = accept_index.device
        row_offsets = (
            torch.arange(bs, device=device, dtype=torch.long) * draft_token_num
        )
        draft_offsets = torch.arange(draft_token_num, device=device, dtype=torch.long)
        rank = draft_offsets.unsqueeze(0).expand(bs, draft_token_num).clone()
        rank += max_accept_len

        path_order = torch.arange(max_accept_len, device=device, dtype=torch.long)
        valid_accept = path_order.unsqueeze(0) < accept_lens.to(torch.long).unsqueeze(1)
        accepted_offsets = accept_index.to(torch.long) - row_offsets.unsqueeze(1)
        row_ids = torch.arange(bs, device=device, dtype=torch.long).unsqueeze(1)
        row_ids = row_ids.expand(bs, max_accept_len)
        rank[row_ids[valid_accept], accepted_offsets[valid_accept]] = path_order[
            None, :
        ].expand(bs, max_accept_len)[valid_accept]

        packed_offsets = torch.argsort(rank, dim=1)
        return (row_offsets.unsqueeze(1) + packed_offsets).flatten()

    def _compact_topk_accept_path_in_req_pool(
        self,
        batch: ModelWorkerBatch,
        packed_indices: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        bs = len(batch.seq_lens)
        if bs == 0:
            return batch.out_cache_loc

        packed_cache_locs = batch.out_cache_loc[packed_indices]
        logical_offsets = torch.arange(
            draft_token_num, device=batch.seq_lens.device, dtype=torch.long
        )
        logical_positions = batch.seq_lens.to(torch.long).unsqueeze(1) + (
            logical_offsets.unsqueeze(0)
        )
        req_pool_indices = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        req_pool_indices = req_pool_indices.expand(bs, draft_token_num)
        self.req_to_token_pool.req_to_token[req_pool_indices, logical_positions] = (
            packed_cache_locs.view(bs, draft_token_num).to(
                dtype=self.req_to_token_pool.req_to_token.dtype
            )
        )
        batch.out_cache_loc = packed_cache_locs
        return packed_cache_locs

    def verify(self, batch: ModelWorkerBatch):
        # Since batch.seq_lens is allocated in another stream, we need
        # record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        verify_input.num_tokens_per_req = verify_input.draft_token_num
        bs = len(batch.seq_lens)
        use_welmv4_mtp_verify_backend = (
            self._should_use_welmv4_mtp_target_verify_attn_backend()
        )
        disable_target_graph_for_welmv4_mtp = (
            _WELM_DISABLE_TARGET_VERIFY_GRAPH_FOR_DUMP
            or self._should_disable_welmv4_mtp_target_verify_graph()
        )
        target_model_runner = self.target_worker.model_runner
        target_graph_runner = None
        target_attn_backend = None
        if disable_target_graph_for_welmv4_mtp:
            target_graph_runner = target_model_runner.graph_runner
            target_model_runner.graph_runner = None
        if use_welmv4_mtp_verify_backend:
            target_attn_backend = target_model_runner.attn_backend
            target_model_runner.attn_backend = (
                self._get_welmv4_mtp_target_verify_attn_backend()
            )

        try:
            # Batch 1: Target verify
            # Prepare for target verify in a separate stream
            with self.plan_stream_ctx:
                # Wait for the draft CUDA graph to finish before plan_stream
                # begins its work. Using an event is more targeted than
                # wait_stream(main_stream) — it only waits for draft GPU
                # work, not all queued main_stream operations.
                if self.plan_stream and hasattr(self, "_draft_done_event"):
                    self.plan_stream.wait_event(self._draft_done_event)
                self.draft_worker._precompute_welmv4_mtp_target_verify_hash_inputs(
                    verify_input,
                    batch,
                )
                verify_forward_batch, can_run_cuda_graph = (
                    verify_input.prepare_for_v2_verify(
                        self.req_to_token_pool,
                        batch,
                        self.target_worker,
                    )
                )

            # Correct some buffers due to the overlap plan
            if self.plan_stream:
                torch.get_device_module(self.device).current_stream().wait_stream(
                    self.plan_stream
                )

                # Some values such as custom_mask and position depend on the output of draft,
                # so the previous plan step used the wrong values. Here, we need to run the related
                # computation again to update them to the correct values.
                self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                    verify_input,
                    (
                        self.target_worker.model_runner.graph_runner.bs
                        if can_run_cuda_graph
                        else None
                    ),
                )

            if getattr(verify_input, "welm_mtp_oe_hashed_inputs", None) is None:
                self.draft_worker._prepare_welmv4_mtp_target_verify_hash_inputs(
                    verify_forward_batch,
                    getattr(verify_input, "welm_mtp_oe_history_state", None),
                )

            # Prepare grammar data on CPU if needed
            if batch.has_grammar:
                retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
                retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
                draft_tokens_cpu = verify_input.draft_token.view(
                    verify_input.retrieve_next_token.shape
                ).cpu()

            # Run target verify batch in the main compute stream (GPU compute)
            forward_batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
                skip_attn_backend_init=True,
            )
        finally:
            if target_attn_backend is not None:
                target_model_runner.attn_backend = target_attn_backend
            if disable_target_graph_for_welmv4_mtp:
                target_model_runner.graph_runner = target_graph_runner
        logits_output = forward_batch_output.logits_output

        # Generate vocab mask for constrained decoding
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        # Sample
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,
            accept_lens,
            accept_index,
        ) = verify_input.sample(batch, logits_output, vocab_mask)
        new_seq_lens = batch.seq_lens + accept_lens
        welm_mtp_accepted_draft_token_ids = None
        has_welmv4_mtp_oe_context = (
            self.draft_worker._is_welmv4_mtp_draft_model()
            and not batch.forward_mode.is_idle()
            and (
                (
                    getattr(batch, "oe_context", None) is not None
                    and batch.oe_context.input_ids_buffer is not None
                )
                or (
                    self.draft_worker._should_use_welmv4_mtp_oe_hash_kernel()
                    and getattr(verify_input, "welm_mtp_oe_history_state", None)
                    is not None
                )
            )
        )
        if has_welmv4_mtp_oe_context:
            accepted_mask = accept_index >= 0
            safe_accept_index = torch.where(
                accepted_mask, accept_index, torch.zeros_like(accept_index)
            ).to(torch.long)
            accepted_draft_token_ids = (
                verify_input.draft_token.reshape(-1)
                .index_select(0, safe_accept_index.reshape(-1))
                .reshape_as(accept_index)
                .to(accept_index.dtype)
            )
            welm_mtp_accepted_draft_token_ids = torch.where(
                accepted_mask,
                accepted_draft_token_ids,
                torch.full_like(accept_index, -1),
            )
        if _WELM_VERIFY_AFTER_DUMP_ENABLED:
            verify_dump_extra = {}
            if not batch.forward_mode.is_idle():
                accepted_mask = accept_index >= 0
                accepted_predict = torch.full_like(accept_index, -1)
                accepted_draft = (
                    welm_mtp_accepted_draft_token_ids
                    if welm_mtp_accepted_draft_token_ids is not None
                    else torch.full_like(accept_index, -1)
                )
                if bool(accepted_mask.any().item()):
                    accepted_indices = accept_index[accepted_mask].to(torch.long)
                    accepted_predict[accepted_mask] = predict.index_select(
                        0, accepted_indices
                    ).to(accepted_predict.dtype)
                verify_dump_extra = {
                    "accepted_predict_token_ids": accepted_predict,
                    "accepted_draft_token_ids": accepted_draft,
                }
            _dump_verify_after_event(
                "verify",
                {
                    "seq_lens_before_verify": batch.seq_lens,
                    "new_seq_lens": new_seq_lens,
                    "predict": predict,
                    "accept_lens": accept_lens,
                    "accept_index": accept_index,
                    "draft_token": verify_input.draft_token,
                    "can_run_cuda_graph": can_run_cuda_graph,
                    "speculative_num_draft_tokens": self.speculative_num_draft_tokens,
                    "speculative_num_steps": self.speculative_num_steps,
                    **verify_dump_extra,
                },
            )
        # Update mamba state for hybrid GDN models after verification
        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
        ):
            self._mamba_verify_update(
                batch, verify_input, accept_lens, accept_index, bs
            )

        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        next_token_ids = predict
        packed_hidden_states = None
        mirrored_kv_indices = None
        if self.topk > 1 and not batch.forward_mode.is_idle():
            packed_indices = self._pack_accept_path_indices(
                accept_index, accept_lens, self.speculative_num_draft_tokens
            )
            next_token_ids = predict[packed_indices]
            mirrored_kv_indices = packed_indices
            if logits_output.hidden_states is not None:
                packed_hidden_states = logits_output.hidden_states[packed_indices]
            self._compact_topk_accept_path_in_req_pool(
                batch, packed_indices, self.speculative_num_draft_tokens
            )

        if not batch.forward_mode.is_idle():
            verified_id = torch.empty_like(accept_lens, dtype=torch.int32)
            if self.topk > 1:
                fill_new_verified_id[(bs,)](
                    next_token_ids,
                    accept_lens,
                    verified_id,
                    self.speculative_num_draft_tokens,
                )
            else:
                all_verified_id = predict[accept_index]
                fill_new_verified_id[(bs,)](
                    all_verified_id,
                    accept_lens,
                    verified_id,
                    self.speculative_num_draft_tokens,
                )
        else:
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        # Construct the next draft input
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
            hidden_states=packed_hidden_states,
            mirrored_kv_indices=mirrored_kv_indices,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            welm_mtp_accepted_draft_token_ids=welm_mtp_accepted_draft_token_ids,
        )

    def _mamba_verify_update(
        self,
        batch: ModelWorkerBatch,
        verify_input: EagleVerifyInput,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
        bs: int,
    ):
        """Update mamba state for hybrid GDN models after verification."""
        # `accept_lens` already includes the bonus token (drafts + 1 per req).
        accepted_length_with_bonus = accept_lens
        if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
            if verify_input.topk != 1:
                raise ValueError("Spec v2 currently only supports topk = 1.")

            accepted_indices_offset = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=accepted_length_with_bonus.dtype,
                device=accepted_length_with_bonus.device,
            )
            accepted_steps = accepted_length_with_bonus - 1

            if batch.mamba_track_indices is not None:
                # If after verify, the request's seq_lens has crossed a mamba track interval,
                # we need to update the mamba state for the request at the crossing point.
                seq_lens_pre_verify = batch.seq_lens
                seq_lens_post_verify = batch.seq_lens + accepted_length_with_bonus
                mamba_track_interval = self.server_args.mamba_track_interval
                to_track_mask = (
                    seq_lens_pre_verify // mamba_track_interval
                    != seq_lens_post_verify // mamba_track_interval
                )
                tracking_point = (
                    seq_lens_post_verify // mamba_track_interval * mamba_track_interval
                )
                to_track_ith = torch.clamp(
                    tracking_point - seq_lens_pre_verify - 1, min=0
                ).to(torch.int64)
                req_idx = torch.arange(
                    bs,
                    dtype=torch.int64,
                    device=accepted_length_with_bonus.device,
                )
                candidate_track_steps = (
                    accept_index[req_idx, to_track_ith] - accepted_indices_offset
                )
                mamba_steps_to_track = torch.where(
                    to_track_mask,
                    candidate_track_steps,
                    torch.full_like(candidate_track_steps, -1),
                )
            else:
                mamba_steps_to_track = None

            self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
                accepted_steps=accepted_steps,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=self.target_worker.model_runner.model,
            )

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        num_accepted_drafts: torch.Tensor,
    ):
        """
        Move accepted tokens to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens.
            num_accepted_drafts: The length of the accepted tokens.
        """
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_accepted_drafts,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.draft_worker.draft_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message
