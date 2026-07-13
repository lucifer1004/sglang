"""Shard transfer backend registry.

The control-plane shells for shard transfer live in SGLang mainline (io_struct,
http_server, tokenizer_control_mixin, scheduler dispatcher). The receiver-side
implementation is provided out-of-tree by a slime plugin that registers a
backend here at plugin-load time.

The backend protocol takes ``model_runner`` (not ``scheduler``): the scheduler
handler keeps the orchestration (enumerating main/draft model runners, flush,
TP result aggregation) and hands each model runner to the backend, which does
pure pull work.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol


class ShardTransferBackend(Protocol):
    def report_targets(self, targets: dict[str, Any]) -> str:
        """Return an opaque, backend-encoded report for the named model runners."""
        ...

    def install_plan(self, payload: str, tp_rank: int) -> str:
        """Decode and store this TP rank's active plan, returning a summary."""
        ...

    def update_weights(self, target_name: str, model_runner: Any) -> dict:
        """Pull + replay + load this target's local slice of the active plan.
        Returns
        per-target stats: {"entries": int, "bytes": int, "seconds": float}."""
        ...


_BACKEND: Optional[ShardTransferBackend] = None


def register_shard_transfer_backend(backend: ShardTransferBackend) -> None:
    global _BACKEND
    _BACKEND = backend


def get_shard_transfer_backend() -> ShardTransferBackend:
    if _BACKEND is None:
        raise RuntimeError(
            "no shard transfer backend registered "
            "(expected a slime plugin to call register_shard_transfer_backend)"
        )
    return _BACKEND
