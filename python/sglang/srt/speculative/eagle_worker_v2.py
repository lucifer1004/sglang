import contextlib
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed import get_tp_group
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
from sglang.srt.layers.dp_attention import get_attention_tp_group, set_dp_buffer_len
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
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.models.welm_perf_opt import (
    get_welm_oe_hash_config,
    should_use_welm_oe_hash_kernel,
)
from sglang.srt.models.welm_mtp_version import WelmMTPVersion, get_welm_mtp_version
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
    assign_draft_cache_locs_page_size_1,
    assign_extend_cache_locs,
    fill_accepted_out_cache_loc,
    fill_new_verified_id,
)
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    organize_draft_results,
)
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
from sglang.srt.speculative.welmv4_mtp_draft_proposal_cuda_graph_runner import (
    WelmMTPDraftProposalCudaGraphRunner,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

if TYPE_CHECKING:
    from sglang.srt.speculative.welm_mtp_draft_ngram_hash import (
        WelmMTPDraftNGramHistoryState,
    )

_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()

if _is_cuda or _is_musa:
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

logger = logging.getLogger(__name__)
_WELM_MTP_DUMP_ENABLED = os.environ.get(
    "SGLANG_DUMP_MTP_ACTIVATIONS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_WELM_VERIFY_AFTER_DUMP_ENABLED = os.environ.get(
    "SGLANG_DUMP_VERIFY_AFTER_MTP_METADATA", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_WELM_TRUE_VALUES = {"1", "true", "yes", "on"}
_WELM_FALSE_VALUES = {"0", "false", "no", "off"}
_WELM_MTP_DRAFT_CUDA_GRAPH_ENABLED = (
    os.environ.get("SGLANG_WELM_MTP_DRAFT_CUDA_GRAPH", "1").strip().lower()
    not in _WELM_FALSE_VALUES
)
_WELM_MTP_DRAFT_FIXED_TEMPERATURE_ENV = "SGLANG_WELM_MTP_DRAFT_FIXED_TEMPERATURE"
_WELM_MTP_DRAFT_FIXED_TOP_P_ENV = "SGLANG_WELM_MTP_DRAFT_FIXED_TOP_P"
_WELM_MTP_DRAFT_SAMPLING_TOPK_ENV = "SGLANG_WELM_MTP_DRAFT_SAMPLING_TOPK"
_WELM_DISABLE_TARGET_VERIFY_GRAPH_FOR_DUMP = (
    os.environ.get("SGLANG_WELMV4_DISABLE_TARGET_VERIFY_GRAPH_FOR_DUMP", "0")
    .strip()
    .lower()
    in _WELM_TRUE_VALUES
)
_WELM_VERIFY_AFTER_EVENT_COUNTERS = {}


def _welm_mtp_env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _WELM_TRUE_VALUES


def _parse_optional_float_env(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return float(raw)


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
        self.welm_mtp_version: Optional[WelmMTPVersion] = None
        self.welmv4_mtp_sample_draft = _welm_mtp_env_flag(
            "SGLANG_WELM_MTP_SAMPLE_DRAFT"
        )
        self.welmv4_mtp_draft_fixed_temperature = _parse_optional_float_env(
            _WELM_MTP_DRAFT_FIXED_TEMPERATURE_ENV
        )
        self.welmv4_mtp_draft_fixed_top_p = _parse_optional_float_env(
            _WELM_MTP_DRAFT_FIXED_TOP_P_ENV
        )
        if (
            self.welmv4_mtp_draft_fixed_temperature is not None
            and self.welmv4_mtp_draft_fixed_temperature <= 0
        ):
            raise ValueError(
                f"{_WELM_MTP_DRAFT_FIXED_TEMPERATURE_ENV} must be positive, "
                f"got {self.welmv4_mtp_draft_fixed_temperature}."
            )
        if self.welmv4_mtp_draft_fixed_top_p is not None and not (
            0 < self.welmv4_mtp_draft_fixed_top_p <= 1
        ):
            raise ValueError(
                f"{_WELM_MTP_DRAFT_FIXED_TOP_P_ENV} must be in (0, 1], "
                f"got {self.welmv4_mtp_draft_fixed_top_p}."
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

        if self._is_welmv4_mtp_draft_model():
            self.welm_mtp_version = get_welm_mtp_version(
                self.draft_runner.model_config.hf_config,
                self.speculative_num_steps,
            )
            if self._is_welmv4_mtp_v2_draft_model():
                if self.topk != 1:
                    raise ValueError(
                        "WeLM MTP V2 requires speculative_eagle_topk=1, "
                        f"got {self.topk}."
                    )
                if self.speculative_num_draft_tokens != self.speculative_num_steps + 1:
                    raise ValueError(
                        "WeLM MTP V2 requires speculative_num_draft_tokens to "
                        "equal speculative_num_steps + 1, got "
                        f"steps={self.speculative_num_steps}, "
                        f"draft_tokens={self.speculative_num_draft_tokens}."
                    )

    def _is_welmv4_mtp_draft_model(self) -> bool:
        if hasattr(self.draft_runner.model_config.hf_config, "num_nextn_predict_layers"):
            return True
        architectures = getattr(
            self.draft_runner.model_config.hf_config, "architectures", []
        )
        return bool(architectures and architectures[0] == "WeLMV4MoeForCausalLMNextN")

    def _is_welmv4_mtp_v1_draft_model(self) -> bool:
        return (
            self._is_welmv4_mtp_draft_model()
            and self.welm_mtp_version == WelmMTPVersion.V1
        )

    def _is_welmv4_mtp_v2_draft_model(self) -> bool:
        return (
            self._is_welmv4_mtp_draft_model()
            and self.welm_mtp_version == WelmMTPVersion.V2
        )

    def _maybe_map_hot_token_id(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.hot_token_id is None:
            return token_ids
        return self.hot_token_id[token_ids]

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

    def _init_welmv4_mtp_oe_history_from_extend(
        self,
        forward_batch: ForwardBatch,
        *,
        first_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self._should_use_welmv4_mtp_oe_hash_kernel():
            return None
        _, _, history_width = self._welmv4_mtp_oe_hash_config()
        if history_width <= 0:
            return None
        input_ids = forward_batch.input_ids
        extend_seq_lens = forward_batch.extend_seq_lens
        extend_start_loc = forward_batch.extend_start_loc
        device = (
            first_token_ids.device
            if first_token_ids is not None
            else input_ids.device
            if input_ids is not None
            else self.device
        )
        if input_ids is None or extend_seq_lens is None or extend_start_loc is None:
            return self._init_welmv4_mtp_oe_history_from_context(
                forward_batch.oe_context,
                device=device,
                first_token_ids=first_token_ids,
            )

        batch_size = int(
            first_token_ids.numel()
            if first_token_ids is not None
            else extend_seq_lens.numel()
        )
        if int(extend_seq_lens.numel()) != batch_size:
            raise RuntimeError(
                "WeLMV4 MTP prefill OE history batch mismatch: "
                f"extend_seq_lens={extend_seq_lens.numel()} vs rows={batch_size}."
            )
        history = torch.empty(
            (batch_size, history_width),
            device=device,
            dtype=torch.int64,
        )

        has_first_token = first_token_ids is not None
        if has_first_token:
            history[:, -1] = first_token_ids.to(
                device=history.device, dtype=torch.int64
            )

        prefix_width = max(history_width - (1 if has_first_token else 0), 0)
        if prefix_width == 0:
            return history
        prefix_rows = self._welmv4_mtp_hash_prefix_rows_from_context(
            forward_batch.oe_context,
            prefix_width,
        )
        prefix_tensor = torch.tensor(
            prefix_rows,
            device=history.device,
            dtype=torch.int64,
        )
        extend_seq_lens = extend_seq_lens.to(device=history.device, dtype=torch.long)
        extend_start_loc = extend_start_loc.to(device=history.device, dtype=torch.long)
        input_ids = input_ids.to(device=history.device, dtype=torch.int64)

        for col in range(prefix_width):
            lag = prefix_width - col
            in_segment = extend_seq_lens >= lag
            segment_pos = extend_start_loc + (extend_seq_lens - lag).clamp_min(0)
            segment_values = input_ids[segment_pos]

            prefix_lag = (lag - extend_seq_lens).clamp_min(1)
            prefix_idx = (prefix_lag - 1).clamp_max(prefix_width - 1)
            prefix_values = prefix_tensor.gather(0, prefix_idx.view(1, -1)).squeeze(0)
            history[:, col] = torch.where(in_segment, segment_values, prefix_values)

        return history

    def _compute_welmv4_mtp_first_query_hash_from_entry_history(
        self,
        forward_batch: ForwardBatch,
        first_input_ids: torch.Tensor,
        entry_history_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, "WelmMTPDraftNGramHistoryState", torch.Tensor]:
        from sglang.srt.speculative.welm_mtp_draft_ngram_hash import (
            WelmMTPDraftNGramEntryHistory,
            materialize_welm_mtp_draft_ngram_history,
            should_use_mk_welm_mtp_draft_ngram_hash,
        )

        entry_hash_history: "WelmMTPDraftNGramHistoryState" = entry_history_state
        if (
            should_use_mk_welm_mtp_draft_ngram_hash()
            and entry_history_state.shape[1] >= 2
        ):
            entry_hash_history = WelmMTPDraftNGramEntryHistory(
                prev_input_ids=entry_history_state[:, -1],
                prev_prev_input_ids=[
                    int(x) for x in entry_history_state[:, -2].detach().cpu().tolist()
                ],
            )
        first_query_hashed_inputs, first_query_draft_history_state = (
            self._compute_welmv4_mtp_draft_decode_hash_inputs(
                forward_batch,
                first_input_ids.to(dtype=torch.int64),
                entry_hash_history,
                draft_history_state=None,
                selected_parent_indices=None,
                base_query_count=int(entry_hash_history.shape[0]),
                step_idx=0,
                use_forward_hash_buffer=False,
            )
        )
        _, _, history_width = self._welmv4_mtp_oe_hash_config()
        first_query_verify_history_state = materialize_welm_mtp_draft_ngram_history(
            first_query_draft_history_state,
            history_width=history_width,
        )
        return (
            first_query_hashed_inputs,
            first_query_draft_history_state,
            first_query_verify_history_state,
        )

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
        prefix_rows = self._welmv4_mtp_hash_prefix_rows_from_context(
            forward_batch.oe_context,
            prefix_width,
        )
        prefixes = self._flatten_welmv4_mtp_hash_prefixes(prefix_rows)
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
        kernel_input_ids = input_ids
        kernel_hashed_out = hashed_out
        accepted_lens_cpu = None
        dense_token_num = batch_size * draft_token_num
        if int(input_ids.numel()) != dense_token_num:
            accepted_lens_cpu = [int(x) for x in accept_lens.detach().cpu().tolist()]
            if sum(accepted_lens_cpu) != int(input_ids.numel()):
                raise RuntimeError(
                    "WeLMV4 MTP accepted-only draft extend input length mismatch: "
                    f"sum(accept_lens)={sum(accepted_lens_cpu)} "
                    f"input_tokens={input_ids.numel()}."
                )
            kernel_input_ids = torch.empty(
                (dense_token_num,), device=input_ids.device, dtype=input_ids.dtype
            )
            src_start = 0
            for row, accepted_len in enumerate(accepted_lens_cpu):
                src_end = src_start + accepted_len
                dst_start = row * draft_token_num
                dst_end = dst_start + accepted_len
                kernel_input_ids[dst_start:dst_end].copy_(input_ids[src_start:src_end])
                if accepted_len < draft_token_num:
                    kernel_input_ids[dst_end : dst_start + draft_token_num].copy_(
                        input_ids[src_end - 1].expand(draft_token_num - accepted_len)
                    )
                src_start = src_end
            kernel_hashed_out = torch.empty(
                (len(oe_vocab_sizes), dense_token_num),
                device=input_ids.device,
                dtype=torch.int64,
            )
        welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda(
            kernel_input_ids,
            accepted_draft_token_ids.reshape(batch_size, -1),
            accept_lens,
            entry_history_state[:batch_size],
            oe_grams,
            oe_vocab_sizes,
            kernel_hashed_out,
            next_history,
            self.draft_runner.model_config.vocab_size,
            draft_token_num,
            use_entry_history_for_extend_hash_prefix=(
                self._is_welmv4_mtp_v2_draft_model()
            ),
        )
        if kernel_hashed_out is not hashed_out:
            assert accepted_lens_cpu is not None
            src_start = 0
            for row, accepted_len in enumerate(accepted_lens_cpu):
                dense_start = row * draft_token_num
                dense_end = dense_start + accepted_len
                src_end = src_start + accepted_len
                hashed_out[:, src_start:src_end].copy_(
                    kernel_hashed_out[:, dense_start:dense_end]
                )
                src_start = src_end
        forward_batch.welm_oe_decode_hashed_inputs = hashed_out
        return next_history

    def _prepare_welmv4_mtp_draft_decode_hash_inputs(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        entry_history_state: "WelmMTPDraftNGramHistoryState",
        draft_history_state: Optional["WelmMTPDraftNGramHistoryState"],
        selected_parent_indices: Optional[torch.Tensor],
        base_query_count: int,
        step_idx: int,
    ) -> "WelmMTPDraftNGramHistoryState":
        hashed_out, next_history = self._compute_welmv4_mtp_draft_decode_hash_inputs(
            forward_batch,
            input_ids,
            entry_history_state,
            draft_history_state,
            selected_parent_indices,
            base_query_count,
            step_idx,
            use_forward_hash_buffer=True,
        )
        forward_batch.welm_oe_decode_hashed_inputs = hashed_out
        return next_history

    def _compute_welmv4_mtp_draft_decode_hash_inputs(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        entry_history_state: "WelmMTPDraftNGramHistoryState",
        draft_history_state: Optional["WelmMTPDraftNGramHistoryState"],
        selected_parent_indices: Optional[torch.Tensor],
        base_query_count: int,
        step_idx: int,
        *,
        use_forward_hash_buffer: bool = False,
    ) -> Tuple[torch.Tensor, "WelmMTPDraftNGramHistoryState"]:
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_mtp_draft_decode_from_history_cuda,
        )
        from sglang.srt.speculative.welm_mtp_draft_ngram_hash import (
            should_use_mk_welm_mtp_draft_ngram_hash,
            welm_mtp_draft_ngram_hash_from_history,
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
        if use_forward_hash_buffer:
            hashed_out = self._get_welmv4_mtp_hash_out(
                forward_batch, input_ids, len(oe_vocab_sizes)
            )
        else:
            batch_major_hash_out = getattr(
                forward_batch, "welm_mtp_oe_hash_out_batch_major", None
            )
            if (
                batch_major_hash_out is not None
                and getattr(forward_batch, "welm_mtp_skip_draft_proposal_build", False)
                and batch_major_hash_out.shape[0] >= int(input_ids.numel())
                and batch_major_hash_out.shape[1] == len(oe_vocab_sizes)
            ):
                hashed_out = batch_major_hash_out[: int(input_ids.numel())].t()
            else:
                hashed_out = torch.empty(
                    (len(oe_vocab_sizes), int(input_ids.numel())),
                    device=input_ids.device,
                    dtype=torch.int64,
                )
        mk_history_state = welm_mtp_draft_ngram_hash_from_history(
            forward_batch=forward_batch,
            input_ids=input_ids,
            history_state=source_history,
            parent_indices=parent_indices,
            oe_grams=oe_grams,
            oe_vocab_sizes=oe_vocab_sizes,
            hashed_out=hashed_out,
            next_history_state=None,
            vocab_size=self.draft_runner.model_config.vocab_size,
            base_query_count=base_query_count,
            use_parent=use_parent,
            prev_input_ids_scratch=getattr(
                forward_batch, "welm_mtp_oe_prev_input_ids", None
            ),
            prev_prev_input_ids_scratch=getattr(
                forward_batch, "welm_mtp_oe_prev_prev_input_ids", None
            ),
            output_ids_scratch=getattr(
                forward_batch, "welm_mtp_oe_hash_out_batch_major", None
            ),
            output_prev_input_ids_scratch=getattr(
                forward_batch, "welm_mtp_oe_output_prev_input_ids", None
            ),
            source_indices_scratch=getattr(
                forward_batch, "welm_mtp_oe_parent_scratch", None
            ),
        )
        if mk_history_state is not None:
            return hashed_out, mk_history_state
        if should_use_mk_welm_mtp_draft_ngram_hash():
            raise RuntimeError(
                "SGLANG_WELM_MTP_DRAFT_NGRAM_HASH is enabled, but draft decode "
                "ngram hash did not run through mk."
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
        return hashed_out, next_history

    def _forward_welmv4_mtp_base_kv_decode(self, forward_batch: ForwardBatch):
        graph_runner_backup = self.draft_runner.graph_runner
        had_base_kv_attr = hasattr(forward_batch, "welm_mtp_use_base_kv_cache")
        base_kv_attr_backup = getattr(forward_batch, "welm_mtp_use_base_kv_cache", None)
        logits_buffer_backup = forward_batch.next_token_logits_buffer
        # Recursive WeLMV4 MTP draft decode reads the committed/base KV cache
        # populated from mirrored main-model KV, so the normal draft-decode
        # graph runner cannot be used here.
        forward_batch.welm_mtp_use_base_kv_cache = True
        forward_batch.next_token_logits_buffer = None
        self.draft_runner.graph_runner = None
        try:
            return self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
        finally:
            self.draft_runner.graph_runner = graph_runner_backup
            forward_batch.next_token_logits_buffer = logits_buffer_backup
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

    def _select_welmv4_mtp_prefill_last_hidden_states(
        self,
        forward_batch: ForwardBatch,
        hidden_states: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if hidden_states is None or hidden_states.shape[0] == batch_size:
            return hidden_states

        active_indices = getattr(forward_batch, "kv_mirror_active_batch_indices", None)
        if (
            active_indices is not None
            and hidden_states.shape[0] == int(active_indices.numel())
        ):
            output = hidden_states.new_zeros((batch_size, *hidden_states.shape[1:]))
            if active_indices.numel() > 0:
                output[active_indices.to(device=hidden_states.device, dtype=torch.long)] = (
                    hidden_states
                )
            return output

        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu is not None:
            extend_lens = [int(x) for x in extend_seq_lens_cpu[:batch_size]]
            if sum(extend_lens) != hidden_states.shape[0]:
                raise RuntimeError(
                    "WeLMV4 MTP prefill hidden-state layout mismatch: "
                    f"sum(extend_lens)={sum(extend_lens)}, "
                    f"hidden_rows={hidden_states.shape[0]}, batch_size={batch_size}."
                )
            last_indices_cpu = []
            offset = 0
            for extend_len in extend_lens:
                if extend_len <= 0:
                    raise RuntimeError(
                        "WeLMV4 MTP prefill hidden-state layout mismatch: "
                        f"non-positive extend_len={extend_len}."
                    )
                offset += extend_len
                last_indices_cpu.append(offset - 1)
            last_indices = torch.tensor(
                last_indices_cpu,
                dtype=torch.long,
                device=hidden_states.device,
            )
            return hidden_states[last_indices]

        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            raise RuntimeError(
                "WeLMV4 MTP prefill hidden-state layout mismatch: "
                f"hidden_rows={hidden_states.shape[0]}, batch_size={batch_size}."
            )
        extend_seq_lens = extend_seq_lens[:batch_size].to(
            device=hidden_states.device, dtype=torch.long
        )
        total_extend_len = int(extend_seq_lens.sum().item())
        if total_extend_len != hidden_states.shape[0]:
            raise RuntimeError(
                "WeLMV4 MTP prefill hidden-state layout mismatch: "
                f"sum(extend_lens)={total_extend_len}, "
                f"hidden_rows={hidden_states.shape[0]}, batch_size={batch_size}."
            )
        return hidden_states[torch.cumsum(extend_seq_lens, dim=0) - 1]

    def _select_welmv4_mtp_prefill_last_positions(
        self,
        forward_batch: ForwardBatch,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        positions = getattr(forward_batch, "positions", None)
        if positions is None:
            return None
        if positions.shape[0] == batch_size:
            return positions.clone()

        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu is not None:
            extend_lens = [int(x) for x in extend_seq_lens_cpu[:batch_size]]
            if sum(extend_lens) == positions.shape[0]:
                last_indices_cpu = []
                offset = 0
                for extend_len in extend_lens:
                    if extend_len <= 0:
                        return positions[:batch_size].clone()
                    offset += extend_len
                    last_indices_cpu.append(offset - 1)
                last_indices = torch.tensor(
                    last_indices_cpu,
                    dtype=torch.long,
                    device=positions.device,
                )
                return positions[last_indices].clone()
            return positions[:batch_size].clone()

        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            return positions[:batch_size].clone()
        extend_seq_lens = extend_seq_lens[:batch_size].to(
            device=positions.device, dtype=torch.long
        )
        if int(extend_seq_lens.sum().item()) != positions.shape[0]:
            return positions[:batch_size].clone()
        return positions[torch.cumsum(extend_seq_lens, dim=0) - 1].clone()

    def _set_welmv4_mtp_synthetic_draft_proposal(
        self,
        spec_info: EagleDraftInput,
        forward_batch: ForwardBatch,
        *,
        batch_size: int,
        hidden_states: Optional[torch.Tensor],
        score_dtype: torch.dtype,
    ) -> None:
        verified_id = spec_info.verified_id
        if verified_id is None or int(verified_id.numel()) != batch_size:
            raise RuntimeError(
                "WeLMV4 MTP synthetic draft proposal batch mismatch: "
                f"verified_rows={None if verified_id is None else verified_id.numel()}, "
                f"batch_size={batch_size}."
            )
        verified_id = verified_id.to(dtype=torch.long).view(batch_size)
        device = verified_id.device

        topk_p = torch.zeros(
            (batch_size, self.topk),
            dtype=score_dtype,
            device=device,
        )
        topk_p[:, 0] = 1.0
        topk_index = verified_id.view(batch_size, 1).expand(batch_size, self.topk)

        first_parent = (
            torch.arange(-1, self.topk, dtype=torch.long, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        score_list: List[torch.Tensor] = [topk_p.unsqueeze(1)]
        token_list: List[torch.Tensor] = [topk_index.contiguous()]
        parents_list: List[torch.Tensor] = [first_parent]
        for step in range(1, self.speculative_num_steps):
            step_scores = torch.zeros(
                (batch_size, self.topk, self.topk),
                dtype=score_dtype,
                device=device,
            )
            step_scores[:, :, 0] = 1.0
            step_tokens = verified_id.view(batch_size, 1).expand(
                batch_size, self.topk * self.topk
            )
            parent_offset = self.topk**2 * (step - 1) + self.topk
            step_parents = torch.full(
                (batch_size, self.topk),
                parent_offset,
                dtype=torch.long,
                device=device,
            )
            score_list.append(step_scores)
            token_list.append(step_tokens.contiguous())
            parents_list.append(step_parents)

        (
            spec_info.draft_proposal_parent_list,
            spec_info.draft_proposal_top_scores_index,
            spec_info.draft_proposal_tokens,
        ) = organize_draft_results(
            score_list,
            token_list,
            parents_list,
            self.speculative_num_draft_tokens,
        )
        spec_info.topk_p = topk_p
        spec_info.topk_index = topk_index
        spec_info.hidden_states = self._select_welmv4_mtp_prefill_last_hidden_states(
            forward_batch,
            hidden_states,
            batch_size,
        )
        spec_info.welm_mtp_base_positions = (
            self._select_welmv4_mtp_prefill_last_positions(forward_batch, batch_size)
        )
        spec_info.welm_mtp_has_draft_proposal = True
        spec_info.draft_probs = None
        spec_info.welm_mtp_draft_topk_indices = None
        spec_info.welm_mtp_draft_topk_values = None
        spec_info.welm_mtp_deferred_prefill_draft = False

    def _check_welmv4_mtp_token_range(
        self,
        token_ids: Optional[torch.Tensor],
        high: int,
        label: str,
    ) -> None:
        if token_ids is None or token_ids.numel() == 0:
            return
        maybe_detect_oob(
            token_ids,
            0,
            high,
            f"WeLMV4 MTP {label} token id OOB vs vocab_size={high}",
        )

    def _should_use_welmv4_mtp_greedy_draft(
        self, forward_batch: ForwardBatch
    ) -> bool:
        if (
            self.welmv4_mtp_sample_draft
            and self._has_welmv4_mtp_fixed_draft_sampling_params()
        ):
            return False
        sampling_info = forward_batch.sampling_info
        return (
            self.topk == 1
            and (sampling_info is None or sampling_info.is_all_greedy)
            and not forward_batch.forward_mode.is_idle()
        )

    def _has_welmv4_mtp_fixed_draft_sampling_params(self) -> bool:
        return (
            self.welmv4_mtp_draft_fixed_temperature is not None
            or self.welmv4_mtp_draft_fixed_top_p is not None
        )

    def _should_sample_welmv4_mtp_draft(self, forward_batch: ForwardBatch) -> bool:
        should_sample = (
            self.welmv4_mtp_sample_draft
            and self._is_welmv4_mtp_v2_draft_model()
            and self.topk == 1
            and not forward_batch.forward_mode.is_idle()
        )
        if not should_sample:
            return False
        if self._has_welmv4_mtp_fixed_draft_sampling_params():
            return True
        sampling_info = forward_batch.sampling_info
        return sampling_info is not None and not sampling_info.is_all_greedy

    def _get_welmv4_mtp_draft_sampling_topk(self) -> int:
        value = int(os.environ.get(_WELM_MTP_DRAFT_SAMPLING_TOPK_ENV, "0"))
        if value <= 0:
            return 0
        vocab_size = int(self.draft_runner.model_config.vocab_size)
        if value >= vocab_size:
            return 0
        return value

    @staticmethod
    def _expand_sampling_tensor_for_logits(
        values: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape[0] == logits.shape[0]:
            return values
        if logits.shape[0] % values.shape[0] == 0:
            return torch.repeat_interleave(
                values, logits.shape[0] // values.shape[0], dim=0
            )
        return values[: logits.shape[0]]

    def _get_welmv4_mtp_draft_temperature(
        self, forward_batch: ForwardBatch, logits: torch.Tensor
    ) -> torch.Tensor:
        if self.welmv4_mtp_draft_fixed_temperature is not None:
            return torch.full(
                (logits.shape[0], 1),
                self.welmv4_mtp_draft_fixed_temperature,
                dtype=torch.float32,
                device=logits.device,
            )

        sampling_info = forward_batch.sampling_info
        temperatures = (
            getattr(sampling_info, "temperatures", None)
            if sampling_info is not None
            else None
        )
        if temperatures is None:
            return torch.ones(
                (logits.shape[0], 1), dtype=torch.float32, device=logits.device
            )
        return self._expand_sampling_tensor_for_logits(temperatures, logits).to(
            device=logits.device, dtype=torch.float32
        )

    def _should_use_welmv4_mtp_draft_top_p(self, forward_batch: ForwardBatch) -> bool:
        if self.welmv4_mtp_draft_fixed_top_p is not None:
            return self.welmv4_mtp_draft_fixed_top_p < 1.0 and (_is_cuda or _is_musa)
        sampling_info = forward_batch.sampling_info
        return (
            sampling_info is not None
            and bool(getattr(sampling_info, "need_top_p_sampling", False))
            and (_is_cuda or _is_musa)
        )

    def _get_welmv4_mtp_draft_top_p(
        self, forward_batch: ForwardBatch, logits: torch.Tensor
    ) -> torch.Tensor:
        if self.welmv4_mtp_draft_fixed_top_p is not None:
            return torch.full(
                (logits.shape[0],),
                self.welmv4_mtp_draft_fixed_top_p,
                dtype=torch.float32,
                device=logits.device,
            )

        sampling_info = forward_batch.sampling_info
        top_ps = (
            getattr(sampling_info, "top_ps", None) if sampling_info is not None else None
        )
        if top_ps is None:
            return torch.ones(
                (logits.shape[0],), dtype=torch.float32, device=logits.device
            )
        return self._expand_sampling_tensor_for_logits(top_ps, logits).to(
            device=logits.device, dtype=torch.float32
        )

    def _sample_welmv4_mtp_probs_top1(
        self,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_index = torch.multinomial(probs, num_samples=1)
        topk_p = torch.gather(probs, dim=1, index=topk_index)

        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            tp_group.broadcast(topk_index, src=0)
            topk_p = torch.gather(probs, dim=1, index=topk_index)
            tp_group.broadcast(topk_p, src=0)

        return topk_p, topk_index

    def _sample_welmv4_mtp_draft_top1(
        self,
        logits: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        sampling_info = forward_batch.sampling_info
        if (
            sampling_info is None
            and not self._has_welmv4_mtp_fixed_draft_sampling_params()
        ):
            raise RuntimeError(
                "WeLMV4 MTP stochastic draft sampling requires sampling_info."
            )

        temperature = self._get_welmv4_mtp_draft_temperature(forward_batch, logits)
        scaled_logits = logits.float() / temperature
        use_top_p = self._should_use_welmv4_mtp_draft_top_p(forward_batch)
        sampling_topk = self._get_welmv4_mtp_draft_sampling_topk()
        vocab_size = int(scaled_logits.shape[-1])

        if 0 < sampling_topk < vocab_size:
            top_logits, top_indices = torch.topk(
                scaled_logits,
                k=sampling_topk,
                dim=-1,
                sorted=use_top_p,
            )
            probs = F.softmax(top_logits, dim=-1)
            if use_top_p:
                top_ps = self._get_welmv4_mtp_draft_top_p(forward_batch, top_logits)
                probs = top_p_renorm_prob(probs, top_ps)

            topk_p, topk_pos = self._sample_welmv4_mtp_probs_top1(probs)
            topk_index = torch.gather(top_indices, dim=-1, index=topk_pos)
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                tp_group.broadcast(topk_index, src=0)
                topk_p = torch.gather(probs, dim=1, index=topk_pos)
                tp_group.broadcast(topk_p, src=0)
            return topk_p, topk_index, None, top_indices, probs

        probs = F.softmax(scaled_logits, dim=-1)
        if (
            not self._has_welmv4_mtp_fixed_draft_sampling_params()
            and getattr(sampling_info, "need_top_k_sampling", False)
            and (_is_cuda or _is_musa)
        ):
            top_ks = self._expand_sampling_tensor_for_logits(
                sampling_info.top_ks, logits
            )
            probs = top_k_renorm_prob(probs, top_ks)
        if use_top_p:
            top_ps = self._get_welmv4_mtp_draft_top_p(forward_batch, logits)
            probs = top_p_renorm_prob(probs, top_ps)

        topk_p, topk_index = self._sample_welmv4_mtp_probs_top1(probs)
        return topk_p, topk_index, probs.contiguous(), None, None

    def _select_or_sample_welmv4_mtp_draft_topk(
        self, logits: torch.Tensor, forward_batch: ForwardBatch
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if self._should_sample_welmv4_mtp_draft(forward_batch):
            return self._sample_welmv4_mtp_draft_top1(logits, forward_batch)

        topk_p, topk_index = self._select_welmv4_mtp_draft_topk(logits, forward_batch)
        return topk_p, topk_index, None, None, None

    def _select_welmv4_mtp_draft_topk(
        self,
        logits: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (self._is_welmv4_mtp_v1_draft_model() and self.topk == 1)
            or self._should_use_welmv4_mtp_greedy_draft(forward_batch)
        ):
            topk_index = torch.argmax(logits, dim=-1, keepdim=True)
            topk_p = torch.ones(
                topk_index.shape,
                dtype=logits.dtype,
                device=logits.device,
            )
            return topk_p, topk_index

        probs = torch.softmax(logits, dim=-1)
        return fast_topk(probs, self.topk, dim=-1)

    def _pad_welmv4_mtp_draft_probs_for_verify(
        self, draft_probs_list: List[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if not draft_probs_list:
            return None
        draft_probs = torch.cat(draft_probs_list, dim=1).contiguous()
        pad_len = self.speculative_num_draft_tokens - draft_probs.shape[1]
        if pad_len <= 0:
            return draft_probs
        return torch.cat(
            [
                draft_probs,
                torch.zeros(
                    (draft_probs.shape[0], pad_len, draft_probs.shape[2]),
                    dtype=draft_probs.dtype,
                    device=draft_probs.device,
                ),
            ],
            dim=1,
        )

    def _pad_welmv4_mtp_draft_topk_for_verify(
        self,
        values: List[torch.Tensor],
        *,
        pad_value: int | float,
    ) -> Optional[torch.Tensor]:
        if not values:
            return None
        out = torch.cat(values, dim=1).contiguous()
        pad_len = self.speculative_num_draft_tokens - out.shape[1]
        if pad_len <= 0:
            return out
        return torch.cat(
            [
                out,
                torch.full(
                    (out.shape[0], pad_len, out.shape[2]),
                    pad_value,
                    dtype=out.dtype,
                    device=out.device,
                ),
            ],
            dim=1,
        )

    def _select_welmv4_mtp_last_extend_hash_inputs(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        if not self._should_use_welmv4_mtp_oe_hash_kernel():
            return None
        hashed_inputs = getattr(forward_batch, "welm_oe_decode_hashed_inputs", None)
        if hashed_inputs is None:
            raise RuntimeError(
                "WeLMV4 MTP merged extend-draft requires OE hashes for the "
                "extend/fill input."
            )
        custom_last_index = getattr(forward_batch, "custom_last_index", None)
        if custom_last_index is None:
            if forward_batch.extend_seq_lens is None:
                return hashed_inputs
            custom_last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        return hashed_inputs[:, custom_last_index.to(torch.long)]

    def _set_welmv4_mtp_merged_query(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        query_hashed_inputs: Optional[torch.Tensor],
        query_positions: Optional[torch.Tensor],
    ) -> None:
        forward_batch.welm_mtp_merge_kv_fill_draft = True
        forward_batch.welm_mtp_query_input_ids = input_ids.to(torch.int64)
        forward_batch.welm_mtp_query_oe_hashed_inputs = query_hashed_inputs
        forward_batch.welm_mtp_query_positions = query_positions

    def _clear_welmv4_mtp_merged_query(self, forward_batch: ForwardBatch) -> None:
        forward_batch.welm_mtp_merge_kv_fill_draft = False
        forward_batch.welm_mtp_query_input_ids = None
        forward_batch.welm_mtp_query_oe_hashed_inputs = None
        forward_batch.welm_mtp_query_positions = None

    def _forward_welmv4_mtp_merged_extend_draft_step(
        self,
        forward_batch: ForwardBatch,
        step: int,
        input_ids: torch.Tensor,
        main_hidden_states: Optional[torch.Tensor],
        query_hashed_inputs: Optional[torch.Tensor],
        query_positions: Optional[torch.Tensor],
        *,
        skip_attn_backend_init: bool,
    ):
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.hidden_states = main_hidden_states
        forward_batch.mtp_step_idx = step
        self._set_welmv4_mtp_merged_query(
            forward_batch,
            input_ids,
            query_hashed_inputs,
            query_positions,
        )
        try:
            logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=skip_attn_backend_init
            ).logits_output
        finally:
            self._clear_welmv4_mtp_merged_query(forward_batch)
            forward_batch.mtp_step_idx = 0
        maybe_detect_nan(
            logits_output.next_token_logits,
            f"welmv4_mtp_merged_extend_draft_step{step}",
        )
        return logits_output

    def _should_defer_welmv4_mtp_prefill_draft(self, batch: ModelWorkerBatch) -> bool:
        """Return True for intermediate chunked prefill chunks."""

        reqs = getattr(batch, "reqs", None)
        if not reqs:
            return False
        return all(getattr(req, "is_chunked", 0) > 0 for req in reqs)

    def _get_welmv4_mtp_last_prefill_hidden_states(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(batch.seq_lens)
        if target_hidden_states.shape[0] == batch_size:
            return target_hidden_states

        if batch.extend_seq_lens is None:
            raise RuntimeError(
                "WeLMV4 MTP deferred prefill requires extend_seq_lens when "
                "target hidden states contain more than one row per request."
            )

        last_indices_cpu = []
        offset = 0
        for extend_len in batch.extend_seq_lens:
            extend_len = int(extend_len)
            if extend_len <= 0:
                raise RuntimeError(
                    "WeLMV4 MTP deferred prefill saw a non-positive extend "
                    f"length: {extend_len}."
                )
            last_indices_cpu.append(offset + extend_len - 1)
            offset += extend_len
        if offset != target_hidden_states.shape[0]:
            raise RuntimeError(
                "WeLMV4 MTP deferred prefill hidden-state layout mismatch: "
                f"sum(extend_lens)={offset}, hidden_rows={target_hidden_states.shape[0]}."
            )

        last_indices = torch.tensor(
            last_indices_cpu,
            dtype=torch.long,
            device=target_hidden_states.device,
        )
        return target_hidden_states[last_indices]

    def _prepare_welmv4_mtp_deferred_prefill_draft_input(
        self,
        draft_input: EagleDraftInput,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> None:
        batch_size = len(batch.seq_lens)
        draft_input.hidden_states = self._get_welmv4_mtp_last_prefill_hidden_states(
            batch, target_hidden_states
        )

        dummy_ids = next_token_ids.to(dtype=torch.int64).view(-1, 1)
        if dummy_ids.shape[0] != batch_size:
            raise RuntimeError(
                "WeLMV4 MTP deferred prefill next-token layout mismatch: "
                f"next_token_rows={dummy_ids.shape[0]}, batch_size={batch_size}."
            )
        draft_input.topk_index = dummy_ids.expand(-1, self.topk).contiguous()
        draft_input.topk_p = torch.ones(
            (batch_size, self.topk),
            dtype=torch.float32,
            device=next_token_ids.device,
        )
        draft_input.draft_probs = None
        draft_input.welm_mtp_draft_topk_indices = None
        draft_input.welm_mtp_draft_topk_values = None
        draft_input.welm_mtp_base_positions = None
        draft_input.draft_proposal_parent_list = None
        draft_input.draft_proposal_top_scores_index = None
        draft_input.draft_proposal_tokens = None
        draft_input.welm_mtp_has_draft_proposal = False
        draft_input.welm_mtp_deferred_prefill_draft = True

    def _run_welmv4_mtp_merged_extend_draft(
        self,
        forward_batch: ForwardBatch,
        first_input_ids: torch.Tensor,
        *,
        skip_attn_backend_init: bool,
        first_query_hashed_inputs: Optional[torch.Tensor] = None,
        first_query_history_state: Optional["WelmMTPDraftNGramHistoryState"] = None,
        draft_path: str = "merged_extend_draft",
    ) -> None:
        if self.topk != 1:
            raise RuntimeError("WeLM MTP V2 merged_extend_draft requires topk=1.")

        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        input_ids = first_input_ids.to(dtype=torch.int64)
        main_hidden_states = spec_info.hidden_states
        query_positions = self._get_welmv4_mtp_base_positions(forward_batch)
        draft_history_state = None
        query_hashed_inputs = first_query_hashed_inputs
        topk_p_list: List[torch.Tensor] = []
        topk_index_list: List[torch.Tensor] = []
        draft_probs_list: List[torch.Tensor] = []
        draft_topk_indices_list: List[torch.Tensor] = []
        draft_topk_values_list: List[torch.Tensor] = []
        last_logits_output = None

        for step in range(self.speculative_num_steps):
            self._check_welmv4_mtp_token_range(
                input_ids,
                self.draft_runner.model_config.vocab_size,
                f"{draft_path} step{step} input",
            )
            if step > 0 and self._should_use_welmv4_mtp_oe_hash_kernel():
                if first_query_history_state is None:
                    raise RuntimeError(
                        "WeLMV4 MTP merged_extend_draft is missing OE history "
                        f"for {draft_path} step{step}."
                    )
                query_hashed_inputs, draft_history_state = (
                    self._compute_welmv4_mtp_draft_decode_hash_inputs(
                        forward_batch,
                        input_ids,
                        first_query_history_state,
                        draft_history_state,
                        selected_parent_indices=None,
                        base_query_count=int(first_query_history_state.shape[0]),
                        step_idx=step,
                        use_forward_hash_buffer=False,
                    )
                )
            elif step == 0 and self._should_use_welmv4_mtp_oe_hash_kernel():
                if query_hashed_inputs is None:
                    raise RuntimeError(
                        "WeLMV4 MTP merged_extend_draft step0 is missing query "
                        f"OE hashes for {draft_path}."
                    )

            logits_output = self._forward_welmv4_mtp_merged_extend_draft_step(
                forward_batch,
                step,
                input_ids,
                main_hidden_states,
                query_hashed_inputs,
                query_positions,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            graph_select_fn = getattr(
                forward_batch, "welm_mtp_draft_graph_select_fn", None
            )
            if graph_select_fn is None:
                (
                    topk_p,
                    topk_index,
                    draft_probs,
                    draft_topk_indices,
                    draft_topk_values,
                ) = self._select_or_sample_welmv4_mtp_draft_topk(
                    logits_output.next_token_logits, forward_batch
                )
            else:
                (
                    topk_p,
                    topk_index,
                    draft_topk_indices,
                    draft_topk_values,
                ) = graph_select_fn(logits_output.next_token_logits, step)
                draft_probs = None
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"{draft_path} step{step}: topk_index OOB vs "
                f"vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            mapped_topk_index = self._maybe_map_hot_token_id(topk_index)
            self._check_welmv4_mtp_token_range(
                mapped_topk_index,
                self.target_worker.model_config.vocab_size,
                f"{draft_path} step{step} topk",
            )
            topk_p_list.append(topk_p)
            topk_index_list.append(topk_index)
            if draft_probs is not None:
                draft_probs_list.append(draft_probs.unsqueeze(1))
            if draft_topk_indices is not None:
                draft_topk_indices_list.append(draft_topk_indices.unsqueeze(1))
                draft_topk_values_list.append(draft_topk_values.unsqueeze(1))

            main_hidden_states = logits_output.hidden_states
            next_input_ids = mapped_topk_index.flatten().to(dtype=torch.int64)
            draft_input_id_buffers = getattr(
                forward_batch, "welm_mtp_draft_input_ids", None
            )
            if draft_input_id_buffers is not None:
                input_ids = draft_input_id_buffers[step, : next_input_ids.numel()]
                input_ids.copy_(next_input_ids)
            else:
                input_ids = next_input_ids
            last_logits_output = logits_output

        spec_info.topk_p = torch.cat(topk_p_list, dim=1)
        spec_info.topk_index = torch.cat(topk_index_list, dim=1)
        spec_info.draft_probs = self._pad_welmv4_mtp_draft_probs_for_verify(
            draft_probs_list
        )
        spec_info.welm_mtp_draft_topk_indices = (
            self._pad_welmv4_mtp_draft_topk_for_verify(
                draft_topk_indices_list,
                pad_value=0,
            )
        )
        spec_info.welm_mtp_draft_topk_values = (
            self._pad_welmv4_mtp_draft_topk_for_verify(
                draft_topk_values_list,
                pad_value=0.0,
            )
        )
        spec_info.hidden_states = (
            None if last_logits_output is None else last_logits_output.hidden_states
        )
        if getattr(forward_batch, "welm_mtp_skip_draft_proposal_build", False):
            spec_info.draft_proposal_parent_list = None
            spec_info.draft_proposal_top_scores_index = None
            spec_info.draft_proposal_tokens = None
            spec_info.welm_mtp_has_draft_proposal = True
        else:
            parent_list, top_scores_index, draft_tokens = (
                self._build_welmv4_mtp_linear_draft_proposal(spec_info)
            )
            self._store_welmv4_mtp_draft_proposal(
                spec_info,
                parent_list,
                top_scores_index,
                draft_tokens,
            )
        spec_info.welm_mtp_deferred_prefill_draft = False
        spec_info.welm_mtp_base_positions = query_positions
        forward_batch.mtp_step_idx = 0

    @staticmethod
    def _copy_welmv4_mtp_draft_proposal_state(
        src: EagleDraftInput,
        dst: EagleDraftInput,
    ) -> None:
        dst.topk_p = src.topk_p
        dst.topk_index = src.topk_index
        dst.hidden_states = src.hidden_states
        dst.draft_probs = src.draft_probs
        dst.welm_mtp_draft_topk_indices = src.welm_mtp_draft_topk_indices
        dst.welm_mtp_draft_topk_values = src.welm_mtp_draft_topk_values
        dst.welm_mtp_base_positions = src.welm_mtp_base_positions
        dst.draft_proposal_parent_list = src.draft_proposal_parent_list
        dst.draft_proposal_top_scores_index = src.draft_proposal_top_scores_index
        dst.draft_proposal_tokens = src.draft_proposal_tokens
        dst.welm_mtp_has_draft_proposal = src.welm_mtp_has_draft_proposal
        dst.welm_mtp_deferred_prefill_draft = False

    def _store_welmv4_mtp_draft_proposal(
        self,
        draft_input: EagleDraftInput,
        parent_list: torch.Tensor,
        top_scores_index: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> None:
        draft_input.draft_proposal_parent_list = parent_list
        draft_input.draft_proposal_top_scores_index = top_scores_index
        draft_input.draft_proposal_tokens = draft_tokens
        draft_input.welm_mtp_has_draft_proposal = True
        draft_input.welm_mtp_deferred_prefill_draft = False

    def _run_welmv4_mtp_v1_recursive_draft_proposal(
        self,
        batch: ModelWorkerBatch,
        draft_input: EagleDraftInput,
        *,
        draft_path: str,
    ) -> None:
        if (
            not self._is_welmv4_mtp_v1_draft_model()
            or batch.forward_mode.is_idle()
            or draft_input.verified_id is None
            or draft_input.verified_id.numel() == 0
        ):
            return

        batch_state = (
            batch.spec_info,
            batch.forward_mode,
            batch.out_cache_loc,
            batch.capture_hidden_mode,
            batch.seq_lens,
            batch.seq_lens_cpu,
            batch.seq_lens_sum,
            batch.req_pool_indices,
        )
        base_positions_backup = draft_input.welm_mtp_base_positions
        base_positions_to_restore = base_positions_backup
        hidden_states_backup = draft_input.hidden_states

        try:
            batch.spec_info = draft_input
            batch.forward_mode = ForwardMode.DECODE
            if self.topk == 1:
                # Legacy V1 topk=1 draft graphs computed base positions from
                # the padded draft-extend block length, not from the accepted
                # token length carried by seq_lens_for_draft_extend.
                proposal_seq_lens = batch.seq_lens
                proposal_seq_lens_cpu = batch.seq_lens_cpu
            else:
                proposal_seq_lens = (
                    draft_input.seq_lens_for_draft_extend
                    if draft_input.seq_lens_for_draft_extend is not None
                    else draft_input.new_seq_lens
                )
                proposal_seq_lens_cpu = draft_input.seq_lens_for_draft_extend_cpu
            if proposal_seq_lens is not None:
                batch.seq_lens = proposal_seq_lens
                if proposal_seq_lens_cpu is None and batch.seq_lens_cpu is not None:
                    proposal_seq_lens_cpu = proposal_seq_lens.detach().cpu()
            if proposal_seq_lens_cpu is not None:
                batch.seq_lens_cpu = proposal_seq_lens_cpu
            if draft_input.req_pool_indices_for_draft_extend is not None:
                batch.req_pool_indices = draft_input.req_pool_indices_for_draft_extend
            if batch.seq_lens_cpu is not None:
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
            elif batch.seq_lens is not None:
                batch.seq_lens_sum = int(batch.seq_lens.sum().item())
            forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
                self.req_to_token_pool,
                batch,
                self.cuda_graph_runner,
                self.draft_runner,
                self.topk,
                self.speculative_num_steps,
            )
            created_oe_prefix_rows = False
            if (
                self._should_use_welmv4_mtp_oe_hash_kernel()
                and getattr(draft_input, "welm_mtp_oe_history_state", None) is None
                and getattr(draft_input, "welm_mtp_oe_prefix_rows", None) is None
            ):
                draft_input.welm_mtp_oe_prefix_rows = (
                    self._welmv4_mtp_prefix_rows_from_reqs(
                        batch.reqs,
                        self._welmv4_mtp_oe_prefix_width(),
                        skip_latest_output=False,
                    )
                )
                created_oe_prefix_rows = True
            self._prepare_welmv4_mtp_draft_decode_entry_history(
                forward_batch,
                draft_input,
            )
            if created_oe_prefix_rows:
                delattr(draft_input, "welm_mtp_oe_prefix_rows")
            if base_positions_to_restore is None:
                base_positions_to_restore = self._get_welmv4_mtp_base_positions(
                    forward_batch
                )

            if can_cuda_graph:
                parent_list, top_scores_index, draft_tokens = (
                    self.cuda_graph_runner.replay(forward_batch)
                )
                if _WELM_MTP_DUMP_ENABLED:
                    _flush_welmv4_mtp_graph_dump(
                        f"{draft_path}_proposal",
                        first_dim_limit=getattr(
                            self.cuda_graph_runner,
                            "raw_bs",
                            forward_batch.batch_size,
                        ),
                    )
            elif _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context(f"{draft_path}_proposal"):
                    parent_list, top_scores_index, draft_tokens = self.draft_forward(
                        forward_batch
                    )
            else:
                parent_list, top_scores_index, draft_tokens = self.draft_forward(
                    forward_batch
                )

            self._store_welmv4_mtp_draft_proposal(
                draft_input,
                parent_list,
                top_scores_index,
                draft_tokens,
            )
        finally:
            draft_input.hidden_states = hidden_states_backup
            draft_input.welm_mtp_base_positions = base_positions_to_restore
            (
                batch.spec_info,
                batch.forward_mode,
                batch.out_cache_loc,
                batch.capture_hidden_mode,
                batch.seq_lens,
                batch.seq_lens_cpu,
                batch.seq_lens_sum,
                batch.req_pool_indices,
            ) = batch_state

    def _prepare_welmv4_mtp_recursive_draft_cache_locs(
        self,
        forward_batch: ForwardBatch,
        draft_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(draft_seq_lens.numel())
        out_cache_loc = torch.empty(
            (batch_size * self.topk * self.speculative_num_steps,),
            dtype=torch.int64,
            device=draft_seq_lens.device,
        )
        assign_draft_cache_locs_page_size_1[(batch_size,)](
            forward_batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            draft_seq_lens,
            out_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
        )
        return out_cache_loc

    def _make_welmv4_mtp_recursive_decode_batch(
        self,
        forward_batch: ForwardBatch,
        spec_info: EagleDraftInput,
        *,
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
        selected_hidden_states: torch.Tensor,
        base_positions: Optional[torch.Tensor],
        recursive_out_cache_loc: torch.Tensor,
        draft_positions: torch.Tensor,
        draft_seq_lens: torch.Tensor,
        draft_seq_lens_cpu: Optional[torch.Tensor],
        batch_size: int,
    ) -> ForwardBatch:
        seq_lens = draft_seq_lens[:batch_size]
        seq_lens_cpu = (
            None if draft_seq_lens_cpu is None else draft_seq_lens_cpu[:batch_size]
        )
        seq_lens_sum = (
            int(seq_lens_cpu.sum().item())
            if seq_lens_cpu is not None
            else int(seq_lens.sum().item())
        )

        spec_info.topk_p = topk_p
        spec_info.topk_index = topk_index
        spec_info.hidden_states = selected_hidden_states
        spec_info.welm_mtp_base_positions = base_positions

        recursive_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=batch_size,
            input_ids=None,
            req_pool_indices=forward_batch.req_pool_indices[:batch_size],
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=None,
            req_to_token_pool=forward_batch.req_to_token_pool,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            out_cache_loc=recursive_out_cache_loc,
            seq_lens_sum=seq_lens_sum,
            return_logprob=False,
            positions=draft_positions,
            mrope_positions=(
                None
                if forward_batch.mrope_positions is None
                else forward_batch.mrope_positions[:, : batch_size * self.topk]
            ),
            original_global_num_tokens_cpu=forward_batch.original_global_num_tokens_cpu,
            global_num_tokens_cpu=forward_batch.global_num_tokens_cpu,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            global_num_tokens_for_logprob_cpu=(
                forward_batch.global_num_tokens_for_logprob_cpu
            ),
            global_num_tokens_for_logprob_gpu=(
                forward_batch.global_num_tokens_for_logprob_gpu
            ),
            dp_padding_mode=forward_batch.dp_padding_mode,
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            spec_algorithm=forward_batch.spec_algorithm,
            spec_info=spec_info,
            model_specific_states=forward_batch.model_specific_states,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            attn_backend=self._get_welmv4_mtp_base_kv_decode_attn_backend(),
            oe_context=None,
        )
        recursive_batch.global_forward_mode = forward_batch.global_forward_mode
        recursive_batch.can_run_dp_cuda_graph = forward_batch.can_run_dp_cuda_graph
        for attr in (
            "enable_welm_kv_mirror_opt",
            "welm_mtp_base_kv_metadata_prepared",
            "welm_mtp_base_kv_metadata_bs",
            "welm_mtp_graph_oe_context_prepared",
            "welm_mtp_oe_hash_out",
            "welm_mtp_oe_entry_history_state",
            "welm_mtp_oe_work_history",
            "welm_mtp_oe_parent_scratch",
            "welm_mtp_oe_prev_input_ids",
            "welm_mtp_oe_prev_prev_input_ids",
            "welm_mtp_oe_output_prev_input_ids",
            "welm_mtp_oe_hash_out_batch_major",
            "welm_mtp_oe_draft_ngram_prepared_cache",
        ):
            if hasattr(forward_batch, attr):
                setattr(recursive_batch, attr, getattr(forward_batch, attr))
        return recursive_batch

    def _run_welmv4_mtp_recursive_draft_proposal(
        self,
        forward_batch: ForwardBatch,
        *,
        skip_attn_backend_init: bool,
        draft_path: str,
    ) -> None:
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        if not self._is_welmv4_mtp_v1_draft_model():
            raise RuntimeError("WeLM MTP recursive draft proposal is only for V1.")

        draft_extend_attn_backend = getattr(
            forward_batch,
            "welm_mtp_draft_extend_attn_backend",
            self.draft_extend_attn_backend,
        )
        if draft_extend_attn_backend is not None:
            metadata_bs = getattr(
                forward_batch, "welm_mtp_draft_extend_metadata_bs", None
            )
            if metadata_bs is not None and hasattr(
                draft_extend_attn_backend, "draft_extend_metadata"
            ):
                metadata = draft_extend_attn_backend.draft_extend_metadata.get(
                    metadata_bs
                )
                if metadata is not None:
                    draft_extend_attn_backend.forward_metadata = metadata
                    draft_extend_attn_backend.forward_metadata_spec_decode_expand = None
            elif not skip_attn_backend_init:
                draft_extend_attn_backend.init_forward_metadata(forward_batch)
                skip_attn_backend_init = True
            forward_batch.attn_backend = draft_extend_attn_backend

        hidden_custom_last = {}
        if not getattr(forward_batch, "enable_welm_kv_mirror_opt", False):
            for attr in (
                "custom_last_index",
                "custom_last_cache_loc",
                "kv_mirror_active_batch_indices",
                "kv_mirror_output_size",
            ):
                if hasattr(forward_batch, attr):
                    hidden_custom_last[attr] = getattr(forward_batch, attr)
                    delattr(forward_batch, attr)
        try:
            logits_output = self.draft_runner.forward(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
            ).logits_output
        finally:
            for attr, value in hidden_custom_last.items():
                setattr(forward_batch, attr, value)
        maybe_detect_nan(
            logits_output.next_token_logits,
            f"welmv4_mtp_{draft_path}: draft_extend logits",
        )

        prefill_batch_size = (
            int(spec_info.verified_id.numel())
            if draft_path == "prefill" and spec_info.verified_id is not None
            else None
        )
        custom_last_index = getattr(forward_batch, "custom_last_index", None)
        if custom_last_index is None:
            if forward_batch.extend_seq_lens is not None:
                custom_last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            else:
                custom_last_index = torch.arange(
                    int(forward_batch.batch_size),
                    dtype=torch.long,
                    device=logits_output.next_token_logits.device,
                )
        custom_last_index = custom_last_index.to(
            device=logits_output.next_token_logits.device, dtype=torch.long
        )
        logits_rows = int(logits_output.next_token_logits.shape[0])
        custom_last_count = int(custom_last_index.numel())
        prefill_base_positions = None
        if prefill_batch_size is not None and logits_rows == prefill_batch_size:
            selected_logits = logits_output.next_token_logits
            selected_hidden_states = (
                self._select_welmv4_mtp_prefill_last_hidden_states(
                    forward_batch,
                    logits_output.hidden_states,
                    prefill_batch_size,
                )
            )
            prefill_base_positions = self._select_welmv4_mtp_prefill_last_positions(
                forward_batch,
                prefill_batch_size,
            )
        elif prefill_batch_size is not None and logits_rows != custom_last_count:
            self._set_welmv4_mtp_synthetic_draft_proposal(
                spec_info,
                forward_batch,
                batch_size=prefill_batch_size,
                hidden_states=spec_info.hidden_states,
                score_dtype=logits_output.next_token_logits.dtype,
            )
            return
        elif logits_rows == custom_last_count:
            selected_logits = logits_output.next_token_logits
            if (
                logits_output.hidden_states is None
                or logits_output.hidden_states.shape[0] == custom_last_count
            ):
                selected_hidden_states = logits_output.hidden_states
            else:
                maybe_detect_oob(
                    custom_last_index,
                    0,
                    logits_output.hidden_states.shape[0],
                    f"welmv4_mtp_{draft_path}: hidden custom_last_index OOB",
                )
                selected_hidden_states = logits_output.hidden_states[custom_last_index]
        else:
            maybe_detect_oob(
                custom_last_index,
                0,
                logits_rows,
                f"welmv4_mtp_{draft_path}: custom_last_index OOB",
            )
            selected_logits = logits_output.next_token_logits[custom_last_index]
            if (
                logits_output.hidden_states is None
                or logits_output.hidden_states.shape[0] == custom_last_count
            ):
                selected_hidden_states = logits_output.hidden_states
            else:
                maybe_detect_oob(
                    custom_last_index,
                    0,
                    logits_output.hidden_states.shape[0],
                    f"welmv4_mtp_{draft_path}: hidden custom_last_index OOB",
                )
                selected_hidden_states = logits_output.hidden_states[custom_last_index]
        topk_p, topk_index, _, _, _ = self._select_or_sample_welmv4_mtp_draft_topk(
            selected_logits,
            forward_batch,
        )
        maybe_detect_oob(
            topk_index,
            0,
            selected_logits.shape[-1],
            f"welmv4_mtp_{draft_path}: initial topk OOB vs "
            f"vocab_size={selected_logits.shape[-1]}",
        )

        recursive_out_cache_loc = getattr(
            forward_batch, "welm_mtp_recursive_draft_out_cache_loc", None
        )
        draft_seq_lens = getattr(
            forward_batch, "welm_mtp_recursive_draft_seq_lens", None
        )
        draft_seq_lens_cpu = getattr(
            forward_batch, "welm_mtp_recursive_draft_seq_lens_cpu", None
        )
        if draft_seq_lens is None:
            draft_seq_lens = forward_batch.seq_lens[: selected_logits.shape[0]]
        if recursive_out_cache_loc is None:
            recursive_out_cache_loc = (
                self._prepare_welmv4_mtp_recursive_draft_cache_locs(
                    forward_batch, draft_seq_lens
                )
            )
        recursive_out_cache_loc = recursive_out_cache_loc[
            : selected_logits.shape[0] * self.topk * self.speculative_num_steps
        ]

        base_positions = prefill_base_positions
        if base_positions is None:
            base_position_index = custom_last_index
            if self.topk == 1 and forward_batch.extend_seq_lens is not None:
                base_position_index = (
                    torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                )
                base_position_index = base_position_index.to(
                    device=forward_batch.positions.device, dtype=torch.long
                )
            maybe_detect_oob(
                base_position_index,
                0,
                forward_batch.positions.shape[0],
                f"welmv4_mtp_{draft_path}: base_position_index OOB",
            )
            base_positions = forward_batch.positions[base_position_index].clone()
        draft_positions = getattr(
            forward_batch, "welm_mtp_recursive_draft_positions", None
        )
        initial_draft_positions = draft_seq_lens[: selected_logits.shape[0]].to(
            device=forward_batch.positions.device, dtype=forward_batch.positions.dtype
        )
        initial_draft_positions = initial_draft_positions.repeat_interleave(self.topk)
        if draft_positions is None:
            draft_positions = initial_draft_positions
        else:
            draft_positions = draft_positions[: selected_logits.shape[0] * self.topk]
            draft_positions.copy_(initial_draft_positions)

        recursive_base_positions = (
            None
            if getattr(
                forward_batch,
                "welm_mtp_use_legacy_recursive_base_positions",
                False,
            )
            else base_positions
        )
        recursive_batch = self._make_welmv4_mtp_recursive_decode_batch(
            forward_batch,
            spec_info,
            topk_p=topk_p,
            topk_index=topk_index,
            selected_hidden_states=selected_hidden_states,
            base_positions=recursive_base_positions,
            recursive_out_cache_loc=recursive_out_cache_loc,
            draft_positions=draft_positions,
            draft_seq_lens=draft_seq_lens,
            draft_seq_lens_cpu=draft_seq_lens_cpu,
            batch_size=int(selected_logits.shape[0]),
        )

        saved_spec_contract = (
            spec_info.num_tokens_per_req,
            spec_info.num_tokens_for_logprob_per_req,
            spec_info.capture_hidden_mode,
        )
        recursive_num_tokens = int(selected_logits.shape[0]) * self.topk
        capture_mode = get_is_capture_mode()
        global_num_tokens_gpu = getattr(recursive_batch, "global_num_tokens_gpu", None)
        global_num_tokens_for_logprob_gpu = getattr(
            recursive_batch, "global_num_tokens_for_logprob_gpu", None
        )
        saved_global_num_tokens_gpu = (
            None
            if capture_mode or global_num_tokens_gpu is None
            else global_num_tokens_gpu.clone()
        )
        saved_global_num_tokens_for_logprob_gpu = (
            None
            if capture_mode or global_num_tokens_for_logprob_gpu is None
            else global_num_tokens_for_logprob_gpu.clone()
        )
        saved_global_dp_buffer_len = getattr(
            recursive_batch, "global_dp_buffer_len", None
        )
        saved_local_dp_buffer_len = (
            int(forward_batch.input_ids.numel())
            if forward_batch.input_ids is not None
            else recursive_num_tokens
        )
        if global_num_tokens_gpu is not None:
            global_num_tokens_gpu.fill_(recursive_num_tokens)
        if global_num_tokens_for_logprob_gpu is not None:
            global_num_tokens_for_logprob_gpu.fill_(recursive_num_tokens)
        if global_num_tokens_gpu is not None:
            dp_size = int(global_num_tokens_gpu.numel())
            recursive_batch.global_dp_buffer_len = recursive_num_tokens * dp_size
            set_dp_buffer_len(
                recursive_batch.global_dp_buffer_len,
                recursive_num_tokens,
                recursive_batch.dp_padding_mode.is_max_len(),
            )

        try:
            spec_info.num_tokens_per_req = self.topk
            spec_info.num_tokens_for_logprob_per_req = self.topk
            spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
            (
                spec_info.draft_proposal_parent_list,
                spec_info.draft_proposal_top_scores_index,
                spec_info.draft_proposal_tokens,
            ) = self.draft_forward(recursive_batch)
        finally:
            (
                spec_info.num_tokens_per_req,
                spec_info.num_tokens_for_logprob_per_req,
                spec_info.capture_hidden_mode,
            ) = saved_spec_contract
            if saved_global_num_tokens_gpu is not None:
                global_num_tokens_gpu.copy_(saved_global_num_tokens_gpu)
            if saved_global_num_tokens_for_logprob_gpu is not None:
                global_num_tokens_for_logprob_gpu.copy_(
                    saved_global_num_tokens_for_logprob_gpu
                )
            if saved_global_dp_buffer_len is not None:
                recursive_batch.global_dp_buffer_len = saved_global_dp_buffer_len
            if (
                global_num_tokens_gpu is not None
                and saved_global_dp_buffer_len is not None
            ):
                set_dp_buffer_len(
                    saved_global_dp_buffer_len,
                    saved_local_dp_buffer_len,
                    recursive_batch.dp_padding_mode.is_max_len(),
                )

        spec_info.topk_p = topk_p
        spec_info.topk_index = topk_index
        spec_info.hidden_states = selected_hidden_states
        spec_info.welm_mtp_base_positions = base_positions
        spec_info.welm_mtp_has_draft_proposal = True
        spec_info.draft_probs = None
        spec_info.welm_mtp_draft_topk_indices = None
        spec_info.welm_mtp_draft_topk_values = None
        spec_info.welm_mtp_deferred_prefill_draft = False

    def _build_welmv4_mtp_linear_draft_proposal(
        self,
        draft_input: EagleDraftInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_p = draft_input.topk_p
        topk_index = self._maybe_map_hot_token_id(draft_input.topk_index)
        if topk_p is None or topk_index is None:
            raise RuntimeError("WeLM MTP draft proposal is missing topk tensors.")
        if (
            topk_p.ndim != 2
            or topk_index.ndim != 2
            or topk_p.shape != topk_index.shape
            or topk_p.shape[1] != self.speculative_num_steps
        ):
            raise RuntimeError(
                "Linear WeLM MTP draft proposal expects one topk entry per "
                f"step: topk_p={None if topk_p is None else tuple(topk_p.shape)}, "
                f"topk_index={None if topk_index is None else tuple(topk_index.shape)}, "
                f"steps={self.speculative_num_steps}."
            )

        batch_size = int(topk_p.shape[0])
        first_parent = (
            torch.arange(-1, self.topk, dtype=torch.long, device=topk_p.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        cumulative_score = None
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        for step in range(self.speculative_num_steps):
            step_score = topk_p[:, step].view(batch_size, 1)
            cumulative_score = (
                step_score
                if cumulative_score is None
                else cumulative_score * step_score
            )
            score_list.append(cumulative_score.view(batch_size, 1, 1))
            token_list.append(topk_index[:, step].view(batch_size, 1))
            if step == 0:
                parents_list.append(first_parent)
            else:
                parents_list.append(
                    torch.full(
                        (batch_size, 1),
                        step,
                        dtype=torch.long,
                        device=topk_p.device,
                    )
                )

        return organize_draft_results(
            score_list,
            token_list,
            parents_list,
            self.speculative_num_draft_tokens,
        )

    def _build_welmv4_mtp_draft_proposal_results(
        self,
        draft_input: EagleDraftInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._is_welmv4_mtp_linear_draft_proposal_input(draft_input):
            return self._build_welmv4_mtp_linear_draft_proposal(draft_input)
        if (
            draft_input.draft_proposal_parent_list is not None
            and draft_input.draft_proposal_top_scores_index is not None
            and draft_input.draft_proposal_tokens is not None
        ):
            return (
                draft_input.draft_proposal_parent_list,
                draft_input.draft_proposal_top_scores_index,
                draft_input.draft_proposal_tokens,
            )
        return self._build_welmv4_mtp_linear_draft_proposal(draft_input)

    def _is_welmv4_mtp_linear_draft_proposal_input(
        self,
        draft_input: EagleDraftInput,
    ) -> bool:
        topk_p = draft_input.topk_p
        topk_index = draft_input.topk_index
        return (
            self.topk == 1
            and topk_p is not None
            and topk_index is not None
            and topk_p.ndim == 2
            and topk_index.ndim == 2
            and topk_p.shape == topk_index.shape
            and topk_p.shape[1] == self.speculative_num_steps
        )

    def _has_welmv4_mtp_draft_proposal(
        self,
        draft_input: EagleDraftInput,
    ) -> bool:
        return bool(
            getattr(draft_input, "welm_mtp_has_draft_proposal", False)
        ) or self._is_welmv4_mtp_linear_draft_proposal_input(draft_input)

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
        self.cuda_graph_runner_for_draft_proposal = None

        if self.server_args.disable_cuda_graph:
            return

        if self.server_args.model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        use_welmv4_mtp_draft_proposal_graph = self._is_welmv4_mtp_draft_model()
        skip_legacy_welmv4_mtp_graph = use_welmv4_mtp_draft_proposal_graph
        # Capture draft
        if self.speculative_num_steps > 1 and not skip_legacy_welmv4_mtp_graph:
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
        elif self.speculative_num_steps > 1 and skip_legacy_welmv4_mtp_graph:
            logger.info(
                "Skip legacy WeLM MTP draft cuda graph; WeLM MTP uses "
                "unified draft proposal generation."
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
        supports_draft_extend_graph_backend = (
            _is_npu
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        )
        if (
            self.draft_extend_attn_backend
            and not skip_legacy_welmv4_mtp_graph
            and supports_draft_extend_graph_backend
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
        elif self.draft_extend_attn_backend and skip_legacy_welmv4_mtp_graph:
            logger.info(
                "Skip legacy WeLM MTP draft-extend cuda graph; WeLM MTP uses "
                "unified draft proposal generation."
            )

        if (
            self.draft_extend_attn_backend
            and use_welmv4_mtp_draft_proposal_graph
            and supports_draft_extend_graph_backend
            and _WELM_MTP_DRAFT_CUDA_GRAPH_ENABLED
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                "Capture WeLM MTP draft proposal cuda graph begin. "
                f"avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_proposal = (
                WelmMTPDraftProposalCudaGraphRunner(self)
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                "Capture WeLM MTP draft proposal cuda graph end. "
                f"Time elapsed: {time.perf_counter() - tic:.2f} s. "
                f"mem usage={(before_mem - after_mem):.2f} GB. "
                f"avail mem={after_mem:.2f} GB."
            )
            logger.info(
                "Capture draft cuda graph end. WeLM MTP uses unified draft "
                "proposal cuda graph."
            )
        elif (
            self.draft_extend_attn_backend
            and use_welmv4_mtp_draft_proposal_graph
            and not _WELM_MTP_DRAFT_CUDA_GRAPH_ENABLED
        ):
            logger.info(
                "Skip WeLM MTP unified draft proposal cuda graph because "
                "SGLANG_WELM_MTP_DRAFT_CUDA_GRAPH is disabled."
            )

    def draft(self, model_worker_batch: ModelWorkerBatch):
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        is_welmv4_mtp = self._is_welmv4_mtp_draft_model()
        use_welmv4_mtp_draft_proposal = (
            is_welmv4_mtp and self._has_welmv4_mtp_draft_proposal(draft_input)
        )
        if (
            is_welmv4_mtp
            and not model_worker_batch.forward_mode.is_idle()
            and getattr(draft_input, "welm_mtp_deferred_prefill_draft", False)
        ):
            if self._is_welmv4_mtp_v1_draft_model():
                self._run_welmv4_mtp_v1_recursive_draft_proposal(
                    model_worker_batch,
                    draft_input,
                    draft_path="prefill_deferred",
                )
                use_welmv4_mtp_draft_proposal = (
                    self._has_welmv4_mtp_draft_proposal(draft_input)
                )
            else:
                raise RuntimeError(
                    "Deferred WeLM MTP prefill draft input reached draft(); "
                    "the final prefill chunk should replace it with a merged_extend_draft "
                    "proposal before decode."
                )
        if (
            is_welmv4_mtp
            and not model_worker_batch.forward_mode.is_idle()
            and not use_welmv4_mtp_draft_proposal
        ):
            raise RuntimeError(
                "WeLM MTP requires a draft proposal from merged_extend_draft."
            )

        # Run draft
        if use_welmv4_mtp_draft_proposal:
            parent_list, top_scores_index, draft_tokens = (
                self._build_welmv4_mtp_draft_proposal_results(draft_input)
            )
        else:
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

        if not use_welmv4_mtp_draft_proposal:
            if can_cuda_graph:
                parent_list, top_scores_index, draft_tokens = (
                    self.cuda_graph_runner.replay(forward_batch)
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
            elif _WELM_MTP_DUMP_ENABLED:
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
            draft_probs=draft_input.draft_probs,
            draft_topk_indices=draft_input.welm_mtp_draft_topk_indices,
            draft_topk_values=draft_input.welm_mtp_draft_topk_values,
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
            if is_welmv4_mtp:
                (
                    topk_p,
                    topk_index,
                    _,
                    _,
                    _,
                ) = self._select_or_sample_welmv4_mtp_draft_topk(
                    logits_output.next_token_logits,
                    forward_batch,
                )
            else:
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
        use_welmv4_mtp_prefill_proposal = (
            self._is_welmv4_mtp_draft_model()
            and self.speculative_num_steps > 1
            and not batch.forward_mode.is_idle()
        )
        use_welmv4_mtp_v2_merged_extend_draft = (
            use_welmv4_mtp_prefill_proposal and self._is_welmv4_mtp_v2_draft_model()
        )
        if use_welmv4_mtp_prefill_proposal:
            batch.capture_hidden_mode = CaptureHiddenMode.LAST
            if self._should_defer_welmv4_mtp_prefill_draft(batch):
                next_draft_input = EagleDraftInput(
                    hidden_states=target_hidden_states,
                    verified_id=next_token_ids,
                    new_seq_lens=batch.seq_lens,
                    num_tokens_per_req=1,
                    num_tokens_for_logprob_per_req=1,
                    model_specific_states=model_specific_states,
                )
                self._prepare_welmv4_mtp_deferred_prefill_draft_input(
                    next_draft_input,
                    batch,
                    target_hidden_states,
                    next_token_ids,
                )
                batch.spec_info = next_draft_input
                return next_draft_input

        # Construct input_ids.  The merged V2 path fills the draft KV cache from
        # the original extend tokens and injects the target next token as the
        # merged query, so it must not use the legacy shifted input layout.
        if not batch.forward_mode.is_idle() and not use_welmv4_mtp_v2_merged_extend_draft:
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
        if use_welmv4_mtp_v2_merged_extend_draft:
            first_query_hashed_inputs = None
            first_query_history_state = None
            if self._should_use_welmv4_mtp_oe_hash_kernel():
                entry_history_state = self._init_welmv4_mtp_oe_history_from_extend(
                    forward_batch,
                    first_token_ids=None,
                )
                if entry_history_state is None:
                    raise RuntimeError(
                        "WeLMV4 MTP merged prefill draft is missing entry OE "
                        "history for the first query."
                    )
                (
                    first_query_hashed_inputs,
                    first_query_history_state,
                    first_query_verify_history_state,
                ) = self._compute_welmv4_mtp_first_query_hash_from_entry_history(
                    forward_batch,
                    next_token_ids,
                    entry_history_state,
                )
                next_draft_input.welm_mtp_oe_history_state = (
                    first_query_verify_history_state
                )
            if mm_input_embeds is not None:
                forward_batch.mm_input_embeds = mm_input_embeds
            if _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context("merged_extend_draft_prefill"):
                    self._run_welmv4_mtp_merged_extend_draft(
                        forward_batch,
                        next_token_ids,
                        skip_attn_backend_init=False,
                        first_query_hashed_inputs=first_query_hashed_inputs,
                        first_query_history_state=first_query_history_state,
                        draft_path="prefill",
                    )
            else:
                self._run_welmv4_mtp_merged_extend_draft(
                    forward_batch,
                    next_token_ids,
                    skip_attn_backend_init=False,
                    first_query_hashed_inputs=first_query_hashed_inputs,
                    first_query_history_state=first_query_history_state,
                    draft_path="prefill",
                )
            return next_draft_input
        if use_welmv4_mtp_prefill_proposal and self._is_welmv4_mtp_v1_draft_model():
            if self._should_use_welmv4_mtp_oe_hash_kernel():
                entry_history_state = self._init_welmv4_mtp_oe_history_from_extend(
                    forward_batch,
                    first_token_ids=None,
                )
                if entry_history_state is None:
                    raise RuntimeError(
                        "WeLMV4 MTP V1 prefill draft is missing entry OE history "
                        "for the first query."
                    )
                next_draft_input.welm_mtp_oe_history_state = entry_history_state
            if mm_input_embeds is not None:
                forward_batch.mm_input_embeds = mm_input_embeds
            forward_batch.welm_mtp_recursive_draft_seq_lens = (
                next_draft_input.new_seq_lens
            )
            forward_batch.welm_mtp_recursive_draft_seq_lens_cpu = (
                next_draft_input.new_seq_lens.detach().cpu()
            )
            forward_batch.welm_mtp_use_legacy_recursive_base_positions = (
                self.topk == 1
            )
            if _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context("recursive_draft_prefill"):
                    self._run_welmv4_mtp_recursive_draft_proposal(
                        forward_batch,
                        skip_attn_backend_init=False,
                        draft_path="prefill",
                    )
            else:
                self._run_welmv4_mtp_recursive_draft_proposal(
                    forward_batch,
                    skip_attn_backend_init=False,
                    draft_path="prefill",
                )
            return next_draft_input
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
        if use_welmv4_mtp_prefill_proposal and self._is_welmv4_mtp_v1_draft_model():
            next_draft_input.welm_mtp_deferred_prefill_draft = True
            return next_draft_input
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

        if (
            self._is_welmv4_mtp_v2_draft_model()
            and self.speculative_num_steps > 1
            and not batch.forward_mode.is_idle()
        ):
            with self.plan_stream_ctx:
                accept_lens_cpu = batch_result.accept_lens.detach().cpu().tolist()
                accept_index = getattr(batch_result, "spec_accept_index", None)
                if accept_index is None:
                    raise RuntimeError(
                        "WeLMV4 MTP V2 merged_extend_draft requires the verify "
                        "accept_index."
                    )
                accepted_indices = accept_index[accept_index != -1].to(torch.long)
                if envs.SGLANG_SPEC_OOB_DETECTION.get():
                    torch._assert_async(
                        batch_result.accept_lens.sum() == accepted_indices.numel(),
                        "WeLMV4 MTP V2 accept_index/accept_lens mismatch.",
                    )
                max_rows = min(
                    int(batch_result.next_token_ids.numel()),
                    int(batch_result.logits_output.hidden_states.shape[0]),
                    int(batch.out_cache_loc.numel()),
                )
                maybe_detect_oob(
                    accepted_indices,
                    0,
                    max_rows,
                    f"WeLMV4 MTP V2 accepted index OOB vs rows={max_rows}",
                )

                draft_input.hidden_states = batch_result.logits_output.hidden_states[
                    accepted_indices
                ]
                draft_input.verified_id = next_draft_input.verified_id
                draft_input.new_seq_lens = next_draft_input.new_seq_lens
                draft_input.verify_done = next_draft_input.verify_done
                draft_input.num_tokens_for_logprob_per_req = 1
                draft_input.num_accepted_drafts = batch_result.accept_lens - 1
                draft_input.num_accepted_tokens = batch_result.accept_lens
                draft_input.num_accepted_drafts_cpu = [
                    int(x) - 1 for x in accept_lens_cpu
                ]
                draft_input.num_accepted_tokens_cpu = [int(x) for x in accept_lens_cpu]
                draft_input.mirrored_kv_indices = accepted_indices

                seq_lens_before = batch.seq_lens
                seq_lens_cpu_before = batch.seq_lens_cpu
                if seq_lens_cpu_before is not None:
                    extend_prefix_lens = seq_lens_cpu_before.tolist()
                    accept_lens_cpu_tensor = batch_result.accept_lens.detach().to(
                        device=seq_lens_cpu_before.device,
                        dtype=seq_lens_cpu_before.dtype,
                    )
                    seq_lens_cpu_after = seq_lens_cpu_before + accept_lens_cpu_tensor
                else:
                    extend_prefix_lens = seq_lens_before.detach().cpu().tolist()
                    seq_lens_cpu_after = next_draft_input.new_seq_lens.detach().cpu()

                batch.spec_info = draft_input
                batch.input_ids = batch_result.next_token_ids[accepted_indices].to(
                    torch.int64
                )
                batch.out_cache_loc = batch.out_cache_loc[accepted_indices]
                batch.seq_lens = next_draft_input.new_seq_lens
                batch.seq_lens_cpu = seq_lens_cpu_after
                batch.seq_lens_sum = int(seq_lens_cpu_after.sum().item())
                batch.extend_seq_lens = [int(x) for x in accept_lens_cpu]
                batch.extend_prefix_lens = extend_prefix_lens
                batch.extend_num_tokens = int(accepted_indices.numel())
                batch.capture_hidden_mode = CaptureHiddenMode.LAST
                batch.forward_mode = ForwardMode.DRAFT_EXTEND
                batch.can_run_dp_cuda_graph = False

                forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
                forward_batch.return_logprob = False
                real_last_index = (
                    torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                )
                forward_batch.custom_last_index = real_last_index
                if forward_batch.out_cache_loc is not None:
                    forward_batch.custom_last_cache_loc = forward_batch.out_cache_loc[
                        real_last_index
                    ]
                first_query_history_state = None
                first_query_hashed_inputs = None
                if self._should_use_welmv4_mtp_oe_hash_kernel():
                    first_query_history_state = (
                        self._prepare_welmv4_mtp_draft_extend_hash_inputs(
                            forward_batch,
                            batch_result.accept_lens,
                            getattr(
                                batch_result,
                                "welm_mtp_accepted_draft_token_ids",
                                None,
                            ),
                            welmv4_mtp_oe_entry_history,
                        )
                    )
                    if first_query_history_state is not None:
                        next_draft_input.welm_mtp_oe_history_state = (
                            first_query_history_state
                        )
                    first_query_hashed_inputs = (
                        self._select_welmv4_mtp_last_extend_hash_inputs(
                            forward_batch
                        )
                    )
                attn_backend = (
                    self.draft_extend_attn_backend or self.draft_runner.attn_backend
                )
                forward_batch.attn_backend = attn_backend
                can_cuda_graph = (
                    self.cuda_graph_runner_for_draft_proposal is not None
                    and self.cuda_graph_runner_for_draft_proposal.can_run(
                        forward_batch
                    )
                )
                if not can_cuda_graph:
                    attn_backend.init_forward_metadata(forward_batch)
            if self.plan_stream:
                torch.get_device_module(self.device).current_stream().wait_stream(
                    self.plan_stream
                )

            if can_cuda_graph:
                self.cuda_graph_runner_for_draft_proposal.replay(
                    forward_batch,
                    next_draft_input.verified_id,
                    first_query_hashed_inputs=first_query_hashed_inputs,
                    first_query_history_state=first_query_history_state,
                )
            elif _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context("merged_extend_draft_decode"):
                    self._run_welmv4_mtp_merged_extend_draft(
                        forward_batch,
                        next_draft_input.verified_id,
                        skip_attn_backend_init=True,
                        first_query_hashed_inputs=first_query_hashed_inputs,
                        first_query_history_state=first_query_history_state,
                        draft_path="decode",
                    )
            else:
                self._run_welmv4_mtp_merged_extend_draft(
                    forward_batch,
                    next_draft_input.verified_id,
                    skip_attn_backend_init=True,
                    first_query_hashed_inputs=first_query_hashed_inputs,
                    first_query_history_state=first_query_history_state,
                    draft_path="decode",
                )
            self._copy_welmv4_mtp_draft_proposal_state(
                forward_batch.spec_info,
                next_draft_input,
            )
            return

        if (
            self._is_welmv4_mtp_v1_draft_model()
            and self.speculative_num_steps > 1
            and not batch.forward_mode.is_idle()
        ):
            with self.plan_stream_ctx:
                seq_lens_cpu_before = batch.seq_lens_cpu
                extend_num_tokens = (
                    len(batch.seq_lens) * self.speculative_num_draft_tokens
                )
                batch.spec_info = draft_input
                batch.input_ids = batch_result.next_token_ids
                batch.seq_lens = batch.seq_lens + self.speculative_num_draft_tokens
                batch.seq_lens_cpu = (
                    batch.seq_lens_cpu + self.speculative_num_draft_tokens
                )
                batch.seq_lens_sum += extend_num_tokens
                batch.extend_seq_lens = [
                    self.speculative_num_draft_tokens
                    for _ in range(len(batch.seq_lens))
                ]
                batch.extend_prefix_lens = seq_lens_cpu_before.tolist()
                batch.extend_num_tokens = extend_num_tokens
                batch.capture_hidden_mode = CaptureHiddenMode.FULL
                batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2

                forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
                forward_batch.return_logprob = False
                if self._should_use_welmv4_mtp_oe_hash_kernel():
                    welmv4_mtp_oe_next_history = (
                        self._prepare_welmv4_mtp_draft_extend_hash_inputs(
                            forward_batch,
                            batch_result.accept_lens,
                            getattr(
                                batch_result,
                                "welm_mtp_accepted_draft_token_ids",
                                None,
                            ),
                            welmv4_mtp_oe_entry_history,
                        )
                    )
                    if welmv4_mtp_oe_next_history is not None:
                        draft_input.welm_mtp_oe_history_state = (
                            welmv4_mtp_oe_next_history
                        )
                        next_draft_input.welm_mtp_oe_history_state = (
                            welmv4_mtp_oe_next_history
                        )

                draft_input.num_accepted_drafts = batch_result.accept_lens - 1
                draft_input.num_accepted_tokens = batch_result.accept_lens
                accept_lens_cpu = batch_result.accept_lens.detach().cpu().tolist()
                draft_input.num_accepted_drafts_cpu = [
                    int(x) - 1 for x in accept_lens_cpu
                ]
                draft_input.num_accepted_tokens_cpu = [
                    int(x) for x in accept_lens_cpu
                ]
                forward_batch.spec_info = draft_input
                forward_batch.welm_mtp_recursive_draft_seq_lens = (
                    next_draft_input.new_seq_lens
                )
                forward_batch.welm_mtp_recursive_draft_seq_lens_cpu = (
                    next_draft_input.new_seq_lens.detach().cpu()
                )
                forward_batch.welm_mtp_use_legacy_recursive_base_positions = (
                    self.topk == 1
                )

                attn_backend = (
                    self.draft_extend_attn_backend or self.draft_runner.attn_backend
                )
                forward_batch.attn_backend = attn_backend
                can_cuda_graph = (
                    self.cuda_graph_runner_for_draft_proposal is not None
                    and self.cuda_graph_runner_for_draft_proposal.can_run(
                        forward_batch
                    )
                )
                if not can_cuda_graph:
                    attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.welm_mtp_recursive_draft_out_cache_loc = (
                        self._prepare_welmv4_mtp_recursive_draft_cache_locs(
                            forward_batch,
                            next_draft_input.new_seq_lens,
                        )
                    )
                forward_batch.custom_last_index = select_index
                if forward_batch.out_cache_loc is not None:
                    forward_batch.custom_last_cache_loc = forward_batch.out_cache_loc[
                        select_index
                    ]
            if self.plan_stream:
                torch.get_device_module(self.device).current_stream().wait_stream(
                    self.plan_stream
                )

            if can_cuda_graph:
                self.cuda_graph_runner_for_draft_proposal.replay(
                    forward_batch,
                    next_draft_input.verified_id,
                    first_query_hashed_inputs=None,
                    first_query_history_state=None,
                )
            elif _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context("recursive_draft_decode"):
                    self._run_welmv4_mtp_recursive_draft_proposal(
                        forward_batch,
                        skip_attn_backend_init=True,
                        draft_path="decode",
                    )
            else:
                self._run_welmv4_mtp_recursive_draft_proposal(
                    forward_batch,
                    skip_attn_backend_init=True,
                    draft_path="decode",
                )
            self._copy_welmv4_mtp_draft_proposal_state(
                forward_batch.spec_info,
                next_draft_input,
            )
            return

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
        self._run_welmv4_mtp_v1_recursive_draft_proposal(
            batch,
            next_draft_input,
            draft_path="decode",
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
                verify_forward_batch, can_run_cuda_graph = (
                    verify_input.prepare_for_v2_verify(
                        self.req_to_token_pool,
                        batch,
                        self.target_worker,
                        prepare_attn_backend=False,
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

            if getattr(verify_input, "welm_mtp_oe_hashed_inputs", None) is not None:
                verify_forward_batch.welm_oe_decode_hashed_inputs = (
                    verify_input.welm_mtp_oe_hashed_inputs
                )
            else:
                self.draft_worker._prepare_welmv4_mtp_target_verify_hash_inputs(
                    verify_forward_batch,
                    getattr(verify_input, "welm_mtp_oe_history_state", None),
                )
                hashed_inputs = getattr(
                    verify_forward_batch, "welm_oe_decode_hashed_inputs", None
                )
                if hashed_inputs is not None:
                    verify_input.welm_mtp_oe_hashed_inputs = hashed_inputs

            if can_run_cuda_graph:
                self.target_worker.model_runner.graph_runner.replay_prepare(
                    verify_forward_batch
                )
            elif not batch.forward_mode.is_idle():
                self.target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
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
            spec_accept_index=accept_index,
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
