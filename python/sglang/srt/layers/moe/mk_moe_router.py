from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from functools import partial
from inspect import signature
from typing import Any, Callable, Optional

import torch

from sglang.srt.environ import Envs
from sglang.srt.layers.moe.topk import StandardTopKOutput

logger = logging.getLogger(__name__)

_HIDDEN_SIZE = 2048
_NUM_EXPERTS = 512
_TOP_K = 10
_MAX_M = 512
_MODE_ENV = "SGLANG_WELM_V45_80A3_MK_MOE_ROUTER_MODE"


class MkMoeRouterMode(Enum):
    OFF = "off"
    TF32 = "tf32"
    BF16 = "bf16"
    FP32_EXACT = "fp32_exact"

    @property
    def gate_weight_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self is MkMoeRouterMode.BF16 else torch.float32


def get_mk_moe_router_mode() -> MkMoeRouterMode:
    value = Envs.SGLANG_WELM_V45_80A3_MK_MOE_ROUTER_MODE.get().strip().lower()
    try:
        return MkMoeRouterMode(value)
    except ValueError:
        choices = ", ".join(mode.value for mode in MkMoeRouterMode)
        raise RuntimeError(
            f"{_MODE_ENV} must be one of {{{choices}}}, got {value!r}"
        ) from None


@dataclass
class _MkRouterPlanState:
    plan: Any
    workspace: Any
    topk_weights: tuple[torch.Tensor, ...]
    topk_ids: tuple[torch.Tensor, ...]
    empty_router_logits: torch.Tensor


@dataclass(frozen=True)
class _MkRouterOps:
    plan: Callable[..., Any]
    init_workspace: Callable[..., Any]
    run: Callable[..., Any]


def _load_mk_router_ops(mode: MkMoeRouterMode) -> _MkRouterOps:
    if mode in (MkMoeRouterMode.TF32, MkMoeRouterMode.FP32_EXACT):
        from mk.kernels.welm_v45_80a3_moe_router import (
            welm_v45_80a3_moe_router_init_workspace,
            welm_v45_80a3_moe_router_plan,
            welm_v45_80a3_moe_router_run,
        )

        if (
            mode is MkMoeRouterMode.FP32_EXACT
            and "bitwise_sglang"
            not in signature(welm_v45_80a3_moe_router_plan).parameters
        ):
            raise ImportError("MK does not provide the SGLang-bitwise router API")

        return _MkRouterOps(
            plan=(
                partial(
                    welm_v45_80a3_moe_router_plan,
                    bitwise_sglang=True,
                )
                if mode is MkMoeRouterMode.FP32_EXACT
                else welm_v45_80a3_moe_router_plan
            ),
            init_workspace=welm_v45_80a3_moe_router_init_workspace,
            run=welm_v45_80a3_moe_router_run,
        )
    if mode is MkMoeRouterMode.BF16:
        from mk.kernels.welm_v45_80a3_moe_router_bf16 import (
            welm_v45_80a3_moe_router_bf16_init_workspace,
            welm_v45_80a3_moe_router_bf16_plan,
            welm_v45_80a3_moe_router_bf16_run,
        )

        return _MkRouterOps(
            plan=welm_v45_80a3_moe_router_bf16_plan,
            init_workspace=welm_v45_80a3_moe_router_bf16_init_workspace,
            run=welm_v45_80a3_moe_router_bf16_run,
        )
    raise RuntimeError(f"Cannot initialize MK MoE router in {mode.value!r} mode")


class WelmV45_80A3MkMoeRouterAdapter:
    """Adapter for the fused WeLM decode and MTP Router/TopK kernel."""

    def __init__(
        self,
        *,
        mode: MkMoeRouterMode,
        config: Any,
        server_args: Any,
        slot_count: int,
        use_previous_precision: bool,
        use_mxfp8: bool,
    ) -> None:
        if mode is MkMoeRouterMode.OFF:
            raise RuntimeError("MK MoE router adapter cannot use off mode")
        self._validate_config(
            mode=mode,
            config=config,
            server_args=server_args,
            slot_count=slot_count,
            use_previous_precision=use_previous_precision,
            use_mxfp8=use_mxfp8,
        )

        setting = f"{_MODE_ENV}={mode.value}"
        if not torch.cuda.is_available():
            raise RuntimeError(f"{setting} requires CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(self.device)
        if "H20" not in properties.name or properties.multi_processor_count != 78:
            raise RuntimeError(
                f"{setting} currently requires an "
                "NVIDIA H20 with "
                f"78 SMs, got {properties.name!r} with "
                f"{properties.multi_processor_count} SMs"
            )

        try:
            import mk
        except ImportError as exc:
            raise RuntimeError(
                f"{setting} requires the vendored "
                "MK package. "
                "Initialize 3rdparty/mk and install it with "
                "`pip install -e 3rdparty/mk --no-deps`."
            ) from exc
        try:
            ops = _load_mk_router_ops(mode)
        except ImportError as exc:
            raise RuntimeError(
                "The installed MK package does not provide the WeLM "
                f"router API required by {setting}."
            ) from exc

        self.mode = mode
        self.slot_count = slot_count
        self._plan_fn = ops.plan
        self._init_workspace_fn = ops.init_workspace
        self._run_fn = ops.run
        self._states: dict[int, _MkRouterPlanState] = {}
        self._active_state: Optional[_MkRouterPlanState] = None
        self._active_m = 0
        self._active_forward_mode = ""
        self._logged_dispatches: set[tuple[str, int]] = set()
        self._logged_fallbacks: set[str] = set()

        logger.info(
            "Enabled MK WeLM MoE router: mode=%s gate_weight_dtype=%s "
            "package=%s device=%s local_slots=%d",
            mode.value,
            mode.gate_weight_dtype,
            getattr(mk, "__file__", "unknown"),
            properties.name,
            slot_count,
        )

    @staticmethod
    def _validate_config(
        *,
        mode: MkMoeRouterMode,
        config: Any,
        server_args: Any,
        slot_count: int,
        use_previous_precision: bool,
        use_mxfp8: bool,
    ) -> None:
        setting = f"{_MODE_ENV}={mode.value}"
        if slot_count <= 0:
            raise RuntimeError("MK MoE router requires at least one local MoE layer")

        expected_model_values = {
            "hidden_size": _HIDDEN_SIZE,
            "num_experts": _NUM_EXPERTS,
            "num_experts_per_tok": _TOP_K,
            "router_score_func": "sigmoid",
            "moe_routing_type": "expert_bias",
            "norm_topk_prob": False,
            "scale_seq_times": 0,
        }
        mismatches: list[str] = []
        for name, expected in expected_model_values.items():
            actual = getattr(config, name, 0 if name == "scale_seq_times" else None)
            if actual != expected:
                mismatches.append(f"{name}={actual!r} (expected {expected!r})")
        if mismatches:
            raise RuntimeError(
                f"{setting} does not support this "
                "model config: " + ", ".join(mismatches)
            )

        required_server_values = {
            "ep_size": 1,
            "moe_a2a_backend": "none",
            "enable_eplb": False,
            "expert_distribution_recorder_mode": None,
            "enable_expert_distribution_metrics": False,
            "enable_moe_router_replay": False,
            "enable_return_routed_experts": False,
            "enable_dp_attention": False,
            "enable_pdmux": False,
            "enable_torch_compile": False,
        }
        unsupported: list[str] = []
        for name, expected in required_server_values.items():
            actual = getattr(server_args, name, expected)
            if actual != expected:
                unsupported.append(f"{name}={actual!r} (expected {expected!r})")
        if unsupported:
            raise RuntimeError(
                f"{setting} P0 has unsupported "
                "server features: " + ", ".join(unsupported)
            )
        if getattr(server_args, "enable_lora", False):
            raise RuntimeError(f"{setting} P0 does not support LoRA")
        moe_runner_backend = getattr(server_args, "moe_runner_backend", "auto")
        if moe_runner_backend not in ("auto", "triton"):
            raise RuntimeError(
                f"{setting} P0 supports only the "
                "auto or triton "
                f"MoE runner, got {moe_runner_backend!r}"
            )
        if use_previous_precision:
            raise RuntimeError(
                f"{setting} does not support WELM_USE_PREVIOUS_PRECISION"
            )
        if use_mxfp8:
            raise RuntimeError(f"{setting} P0 does not support MXFP8 experts")

    def begin_forward(
        self,
        *,
        forward_batch: Any,
        num_tokens: int,
        allow_fused_router: bool,
    ) -> None:
        self._active_state = None
        self._active_m = 0
        self._active_forward_mode = ""

        if not allow_fused_router:
            self._log_fallback_once("debug", "WeLM tensor dumping is enabled")
            return
        forward_mode = forward_batch.forward_mode
        if not (
            forward_mode.is_decode()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend(include_v2=True)
        ):
            self._log_fallback_once(
                "forward_mode",
                f"forward_mode={getattr(forward_mode, 'name', forward_mode)!s} "
                "is not a supported decode/MTP mode",
            )
            return
        if getattr(forward_batch, "can_run_tbo", False):
            self._log_fallback_once("tbo", "the current batch uses TBO")
            return
        if num_tokens <= 0:
            return
        if num_tokens > _MAX_M:
            self._log_fallback_once(
                "max_m",
                f"M={num_tokens} exceeds the MK kernel limit {_MAX_M}",
            )
            return

        state = self._states.get(num_tokens)
        if state is None:
            if self._is_capturing():
                raise RuntimeError(
                    "MK MoE router plan was not created during CUDA graph warmup "
                    f"for M={num_tokens}"
                )
            state = self._create_state(num_tokens)
            self._states[num_tokens] = state
        else:
            self._init_workspace_fn(state.plan, workspace=state.workspace)

        self._active_state = state
        self._active_m = num_tokens
        self._active_forward_mode = getattr(forward_mode, "name", str(forward_mode))

    def route(
        self,
        *,
        slot_id: int,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        expert_bias: torch.Tensor,
    ) -> Optional[StandardTopKOutput]:
        state = self._active_state
        if state is None:
            return None
        if hidden_states.shape != (self._active_m, _HIDDEN_SIZE):
            self._log_fallback_once(
                "hidden_shape",
                f"router hidden shape is {tuple(hidden_states.shape)}",
            )
            return None
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
            self._log_fallback_once(
                "hidden_layout",
                f"router hidden dtype/layout is {hidden_states.dtype}/"
                f"contiguous={hidden_states.is_contiguous()}",
            )
            return None
        required_dtype = self.mode.gate_weight_dtype
        if gate_weight.dtype != required_dtype:
            raise RuntimeError(
                f"MK MoE router mode {self.mode.value!r} requires gate weight "
                f"dtype {required_dtype}, got {gate_weight.dtype}"
            )

        topk_weights = state.topk_weights[slot_id]
        topk_ids = state.topk_ids[slot_id]
        self._run_fn(
            state.workspace,
            slot_id=slot_id,
            hidden_states=hidden_states,
            gate_weight=gate_weight,
            expert_bias=expert_bias,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        self._log_dispatch_once(self._active_forward_mode, self._active_m)
        return StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=state.empty_router_logits,
        )

    def _create_state(self, num_tokens: int) -> _MkRouterPlanState:
        plan = self._plan_fn(
            m=num_tokens,
            slot_count=self.slot_count,
            device=self.device,
        )
        workspace = self._init_workspace_fn(plan)
        # Separate allocations preserve MK's 16-byte alignment for odd M values.
        topk_weights = tuple(
            torch.empty(
                (num_tokens, _TOP_K),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(self.slot_count)
        )
        topk_ids = tuple(
            torch.empty(
                (num_tokens, _TOP_K),
                dtype=torch.int32,
                device=self.device,
            )
            for _ in range(self.slot_count)
        )
        empty_router_logits = torch.empty(
            (num_tokens, 0), dtype=torch.float32, device=self.device
        )
        logger.info(
            "Prepared MK WeLM MoE router plan: M=%d slots=%d cache_key=%s",
            num_tokens,
            self.slot_count,
            getattr(plan, "cache_key", "unknown"),
        )
        return _MkRouterPlanState(
            plan=plan,
            workspace=workspace,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            empty_router_logits=empty_router_logits,
        )

    @staticmethod
    def _is_capturing() -> bool:
        try:
            return torch.cuda.is_current_stream_capturing()
        except RuntimeError:
            return False

    def _log_fallback_once(self, key: str, reason: str) -> None:
        if key in self._logged_fallbacks:
            return
        self._logged_fallbacks.add(key)
        logger.warning(
            "MK WeLM MoE router FALLBACK -> native SGLang Router: %s", reason
        )

    def _log_dispatch_once(self, forward_mode: str, num_tokens: int) -> None:
        key = (forward_mode, num_tokens)
        if key in self._logged_dispatches:
            return
        self._logged_dispatches.add(key)
        logger.info(
            "MK WeLM MoE router ACTIVE: mode=%s fused kernel dispatched for M=%d; "
            "native Router bypassed",
            forward_mode,
            num_tokens,
        )
