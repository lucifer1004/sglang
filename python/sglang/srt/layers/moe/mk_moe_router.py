from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput

logger = logging.getLogger(__name__)

_HIDDEN_SIZE = 2048
_NUM_EXPERTS = 512
_TOP_K = 10


@dataclass
class _MkRouterPlanState:
    plan: Any
    workspace: Any
    topk_weights: tuple[torch.Tensor, ...]
    topk_ids: tuple[torch.Tensor, ...]
    empty_router_logits: torch.Tensor


class WelmV45_80A3MkMoeRouterAdapter:
    """Decode-only adapter for the fused WeLM MK router and TopK kernel."""

    def __init__(
        self,
        *,
        config: Any,
        server_args: Any,
        slot_count: int,
        use_previous_precision: bool,
        use_mxfp8: bool,
    ) -> None:
        self._validate_config(
            config=config,
            server_args=server_args,
            slot_count=slot_count,
            use_previous_precision=use_previous_precision,
            use_mxfp8=use_mxfp8,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "--enable-welm-v45-80a3-mk-moe-router requires CUDA"
            )
        self.device = torch.device("cuda", torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(self.device)
        if "H20" not in properties.name or properties.multi_processor_count != 78:
            raise RuntimeError(
                "--enable-welm-v45-80a3-mk-moe-router currently requires an "
                "NVIDIA H20 with "
                f"78 SMs, got {properties.name!r} with "
                f"{properties.multi_processor_count} SMs"
            )

        try:
            import mk
        except ImportError as exc:
            raise RuntimeError(
                "--enable-welm-v45-80a3-mk-moe-router requires the vendored "
                "MK package. "
                "Initialize 3rdparty/mk and install it with "
                "`pip install -e 3rdparty/mk --no-deps`."
            ) from exc
        try:
            from mk.kernels.welm_v45_80a3_moe_router import (
                welm_v45_80a3_moe_router_init_workspace,
                welm_v45_80a3_moe_router_plan,
                welm_v45_80a3_moe_router_run,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The installed MK package does not provide the WeLM "
                "router API required by "
                "--enable-welm-v45-80a3-mk-moe-router."
            ) from exc

        self.slot_count = slot_count
        self._plan_fn = welm_v45_80a3_moe_router_plan
        self._init_workspace_fn = welm_v45_80a3_moe_router_init_workspace
        self._run_fn = welm_v45_80a3_moe_router_run
        self._states: dict[int, _MkRouterPlanState] = {}
        self._active_state: Optional[_MkRouterPlanState] = None
        self._active_m = 0
        self._logged_dispatch_ms: set[int] = set()
        self._logged_fallbacks: set[str] = set()

        logger.info(
            "Enabled MK WeLM MoE router: package=%s device=%s local_slots=%d",
            getattr(mk, "__file__", "unknown"),
            properties.name,
            slot_count,
        )

    @staticmethod
    def _validate_config(
        *,
        config: Any,
        server_args: Any,
        slot_count: int,
        use_previous_precision: bool,
        use_mxfp8: bool,
    ) -> None:
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
                "--enable-welm-v45-80a3-mk-moe-router does not support this "
                "model config: "
                + ", ".join(mismatches)
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
                "--enable-welm-v45-80a3-mk-moe-router P0 has unsupported "
                "server features: "
                + ", ".join(unsupported)
            )
        if getattr(server_args, "enable_lora", False):
            raise RuntimeError(
                "--enable-welm-v45-80a3-mk-moe-router P0 does not support LoRA"
            )
        moe_runner_backend = getattr(server_args, "moe_runner_backend", "auto")
        if moe_runner_backend not in ("auto", "triton"):
            raise RuntimeError(
                "--enable-welm-v45-80a3-mk-moe-router P0 supports only the "
                "auto or triton "
                f"MoE runner, got {moe_runner_backend!r}"
            )
        if use_previous_precision:
            raise RuntimeError(
                "--enable-welm-v45-80a3-mk-moe-router does not support "
                "WELM_USE_PREVIOUS_PRECISION"
            )
        if use_mxfp8:
            raise RuntimeError(
                "--enable-welm-v45-80a3-mk-moe-router P0 does not support "
                "MXFP8 experts"
            )

    def begin_forward(
        self,
        *,
        forward_batch: Any,
        num_tokens: int,
        allow_fused_router: bool,
    ) -> None:
        self._active_state = None
        self._active_m = 0

        if not allow_fused_router:
            self._log_fallback_once("debug", "WeLM tensor dumping is enabled")
            return
        if not forward_batch.forward_mode.is_decode():
            forward_mode = forward_batch.forward_mode
            self._log_fallback_once(
                "forward_mode",
                f"forward_mode={getattr(forward_mode, 'name', forward_mode)!s} "
                "is not decode",
            )
            return
        if getattr(forward_batch, "can_run_tbo", False):
            self._log_fallback_once("tbo", "the current batch uses TBO")
            return
        if num_tokens <= 0:
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
        self._log_dispatch_once(self._active_m)
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

    def _log_dispatch_once(self, num_tokens: int) -> None:
        if num_tokens in self._logged_dispatch_ms:
            return
        self._logged_dispatch_ms.add(num_tokens)
        logger.info(
            "MK WeLM MoE router ACTIVE: fused kernel dispatched for M=%d; "
            "native Router bypassed",
            num_tokens,
        )
