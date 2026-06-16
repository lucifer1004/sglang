from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_WELM_MTP_DRAFT_NGRAM_HASH"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_mk_draft_ngram_handle = None
_mk_draft_ngram_spec_cache = None
_mk_draft_ngram_logged: set[str] = set()
_DEFERRED_PREPARED_LAUNCH_ATTR = "welm_mtp_draft_ngram_prepared_launch"


@dataclass(frozen=True)
class WelmMTPDraftNGramHistory:
    prev_input_ids: torch.Tensor
    prev_prev_input_ids: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.prev_input_ids.numel()), 2)


@dataclass(frozen=True)
class WelmMTPDraftNGramEntryHistory:
    prev_input_ids: torch.Tensor
    prev_prev_input_ids: Sequence[int]

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.prev_input_ids.numel()), 2)


WelmMTPDraftNGramHistoryState: TypeAlias = (
    torch.Tensor | WelmMTPDraftNGramHistory | WelmMTPDraftNGramEntryHistory
)


def should_use_mk_welm_mtp_draft_ngram_hash() -> bool:
    return os.environ.get(_ENV_NAME, "0").strip().lower() in _TRUE_VALUES


def _log_once(key: str, level: int, message: str, *args) -> None:
    if key in _mk_draft_ngram_logged:
        return
    _mk_draft_ngram_logged.add(key)
    logger.log(level, message, *args)


def _load_mk_draft_ngram():
    global _mk_draft_ngram_handle
    if _mk_draft_ngram_handle is False:
        return None
    if _mk_draft_ngram_handle is not None:
        return _mk_draft_ngram_handle

    try:
        from mk.kernels import (  # type: ignore[import-not-found]
            DraftNGramHashParams,
            NGramSpec,
            draft_ngram_hash,
            prepare_draft_ngram_hash,
        )
    except Exception as exc:  # pragma: no cover - import failure path
        _log_once(
            "import_failed",
            logging.WARNING,
            "mk draft ngram hash is unavailable: %r",
            exc,
        )
        _mk_draft_ngram_handle = False
        return None

    _mk_draft_ngram_handle = (
        DraftNGramHashParams,
        NGramSpec,
        draft_ngram_hash,
        prepare_draft_ngram_hash,
    )
    return _mk_draft_ngram_handle


def _build_mk_draft_ngram_spec(
    ngram_spec_cls,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
):
    global _mk_draft_ngram_spec_cache
    cache_key = (tuple(int(g) for g in oe_grams), tuple(int(v) for v in oe_vocab_sizes))
    cached = _mk_draft_ngram_spec_cache
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    specs = tuple(
        ngram_spec_cls(int(n), int(v)) for n, v in zip(oe_grams, oe_vocab_sizes)
    )
    _mk_draft_ngram_spec_cache = (cache_key, specs)
    return specs


def _valid_int64_scratch(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
    numel: int,
    name: str,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if (
        tensor.device != device
        or tensor.dtype is not torch.int64
        or tensor.numel() < numel
    ):
        _log_once(
            f"bad_scratch:{name}",
            logging.WARNING,
            "mk draft ngram hash scratch %s is incompatible; allocating a "
            "temporary fallback buffer.",
            name,
        )
        return None
    return tensor.reshape(-1)[:numel]


def _valid_output_scratch(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
    shape: tuple[int, int],
    name: str,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if (
        tensor.device != device
        or tensor.dtype is not torch.int64
        or tensor.ndim != 2
        or tensor.shape[0] < shape[0]
        or tensor.shape[1] != shape[1]
        or not tensor.is_contiguous()
    ):
        _log_once(
            f"bad_scratch:{name}",
            logging.WARNING,
            "mk draft ngram hash output scratch %s is incompatible; allocating "
            "a temporary fallback buffer.",
            name,
        )
        return None
    return tensor[: shape[0], :]


def _is_same_tensor_view(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return (
        tuple(lhs.shape) == tuple(rhs.shape)
        and tuple(lhs.stride()) == tuple(rhs.stride())
        and int(lhs.data_ptr()) == int(rhs.data_ptr())
    )


def _draft_ngram_source_indices(
    *,
    input_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    base_query_count: int,
    use_parent: bool,
    source_indices_scratch: torch.Tensor | None,
) -> torch.Tensor | None:
    num_tokens = int(input_ids.numel())
    if use_parent:
        if parent_indices.dtype is not torch.int64:
            return parent_indices.to(device=input_ids.device, dtype=torch.int64)
        return parent_indices.reshape(-1)[:num_tokens]

    if base_query_count <= 0 or num_tokens % int(base_query_count) != 0:
        return None
    repeat = num_tokens // int(base_query_count)
    if repeat == 1:
        return None

    source_indices = _valid_int64_scratch(
        source_indices_scratch,
        device=input_ids.device,
        numel=num_tokens,
        name="source_indices",
    )
    if source_indices is None:
        source_indices = torch.empty(
            (num_tokens,), device=input_ids.device, dtype=torch.int64
        )
    torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64, out=source_indices)
    torch.floor_divide(source_indices, repeat, out=source_indices)
    return source_indices


def _copy_draft_ngram_history_column(
    history_column: torch.Tensor,
    source_indices: torch.Tensor | None,
    out: torch.Tensor,
) -> None:
    if source_indices is None:
        out.copy_(history_column[: out.numel()])
        return
    if out.untyped_storage().data_ptr() == history_column.untyped_storage().data_ptr():
        out.copy_(torch.index_select(history_column, 0, source_indices))
        return
    torch.index_select(history_column, 0, source_indices, out=out)


def materialize_welm_mtp_draft_ngram_history(
    history_state: WelmMTPDraftNGramHistoryState,
    *,
    history_width: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(history_state, torch.Tensor):
        return history_state

    rows = int(history_state.prev_input_ids.numel())
    if out is None:
        out = torch.empty(
            (rows, int(history_width)),
            device=history_state.prev_input_ids.device,
            dtype=torch.int64,
        )
    if int(out.shape[0]) < rows or int(out.shape[1]) != int(history_width):
        raise RuntimeError(
            "WeLM MTP draft ngram history materialization buffer has incompatible "
            f"shape: {tuple(out.shape)} vs ({rows}, {history_width})."
        )
    history_out = out[:rows]
    if history_width > 2:
        history_out[:, : history_width - 2].zero_()
    if history_width > 1:
        if isinstance(history_state.prev_prev_input_ids, torch.Tensor):
            history_out[:, -2].copy_(history_state.prev_prev_input_ids)
        else:
            history_out[:, -2].copy_(
                torch.tensor(
                    [int(x) for x in history_state.prev_prev_input_ids],
                    device=history_out.device,
                    dtype=torch.int64,
                )
            )
    history_out[:, -1].copy_(history_state.prev_input_ids)
    return history_out


def _draft_ngram_history_vectors(
    history_state: WelmMTPDraftNGramHistoryState,
    *,
    requires_prev_prev: bool,
) -> tuple[torch.Tensor, torch.Tensor | Sequence[int] | None, int]:
    if isinstance(
        history_state, (WelmMTPDraftNGramHistory, WelmMTPDraftNGramEntryHistory)
    ):
        return (
            history_state.prev_input_ids,
            history_state.prev_prev_input_ids if requires_prev_prev else None,
            int(history_state.prev_input_ids.numel()),
        )

    if history_state.ndim != 2:
        raise RuntimeError(
            "mk draft ngram hash requires 2D tensor history or MK draft history "
            f"state, got ndim={history_state.ndim}."
        )
    if int(history_state.shape[1]) < (3 if requires_prev_prev else 2):
        raise RuntimeError(
            "mk draft ngram hash history width is too small: "
            f"shape={tuple(history_state.shape)}."
        )
    return (
        history_state[:, -1],
        history_state[:, -2] if requires_prev_prev else None,
        int(history_state.shape[0]),
    )


def _expanded_prev_prev_input_ids_list(
    prev_prev_input_ids: Sequence[int],
    *,
    num_tokens: int,
    base_query_count: int,
    use_parent: bool,
) -> list[int]:
    _require_mk_draft_ngram(
        not use_parent,
        "mk draft ngram hash first-step list prev_prev_input_ids does not "
        "support parent-selected history.",
    )
    values = [int(x) for x in prev_prev_input_ids]
    _require_mk_draft_ngram(
        len(values) == int(base_query_count),
        "mk draft ngram hash first-step prev_prev_input_ids length does not "
        f"match base_query_count: {len(values)} vs {base_query_count}.",
    )
    _require_mk_draft_ngram(
        base_query_count > 0 and num_tokens % int(base_query_count) == 0,
        "mk draft ngram hash cannot expand first-step prev_prev_input_ids: "
        f"num_tokens={num_tokens}, base_query_count={base_query_count}.",
    )
    repeat = num_tokens // int(base_query_count)
    if repeat == 1:
        return values
    expanded: list[int] = []
    for token_id in values:
        expanded.extend([token_id] * repeat)
    return expanded


def _require_mk_draft_ngram(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def launch_deferred_welm_mtp_draft_ngram_hash(forward_batch: Any) -> bool:
    prepared = getattr(forward_batch, _DEFERRED_PREPARED_LAUNCH_ATTR, None)
    if prepared is None:
        return False
    try:
        prepared.launch()
    except Exception as exc:
        _log_once(
            "deferred_run_failed",
            logging.WARNING,
            "mk draft ngram hash deferred launch failed: %r",
            exc,
        )
        raise RuntimeError("mk draft ngram hash deferred launch failed") from exc
    setattr(forward_batch, _DEFERRED_PREPARED_LAUNCH_ATTR, None)
    return True


def _is_current_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _prepared_cache_key(
    *,
    input_ids: torch.Tensor,
    prev_input_ids: torch.Tensor,
    prev_prev_input_ids: torch.Tensor,
    topk_cs_idx: torch.Tensor | None,
    output_ids: torch.Tensor,
    ngram_spec: Sequence[Any],
    topk: int,
    vocab_size: int,
) -> tuple[Any, ...]:
    topk_cs_key: tuple[str, int | None] = (
        ("tensor", int(topk_cs_idx.data_ptr()))
        if topk_cs_idx is not None
        else ("none", None)
    )
    return (
        int(input_ids.data_ptr()),
        int(prev_input_ids.data_ptr()),
        int(prev_prev_input_ids.data_ptr()),
        topk_cs_key,
        int(output_ids.data_ptr()),
        int(input_ids.numel()),
        int(vocab_size),
        int(topk),
        tuple(ngram_spec),
    )


def _get_or_prepare_draft_ngram_hash(
    *,
    forward_batch: Any,
    prepare_draft_ngram_hash,
    input_ids: torch.Tensor,
    prev_input_ids: torch.Tensor,
    prev_prev_input_ids: torch.Tensor,
    topk_cs_idx: torch.Tensor | None,
    output_ids: torch.Tensor,
    ngram_spec: Sequence[Any],
    topk: int,
    vocab_size: int,
):
    cache = getattr(forward_batch, "welm_mtp_oe_draft_ngram_prepared_cache", None)
    if cache is None:
        if _is_current_stream_capturing():
            return None
        cache = {}
        forward_batch.welm_mtp_oe_draft_ngram_prepared_cache = cache

    key = _prepared_cache_key(
        input_ids=input_ids,
        prev_input_ids=prev_input_ids,
        prev_prev_input_ids=prev_prev_input_ids,
        topk_cs_idx=topk_cs_idx,
        output_ids=output_ids,
        ngram_spec=ngram_spec,
        topk=topk,
        vocab_size=vocab_size,
    )
    prepared = cache.get(key)
    if prepared is not None:
        return prepared
    if _is_current_stream_capturing():
        return None

    prepared = prepare_draft_ngram_hash(
        input_ids=input_ids,
        prev_input_ids=prev_input_ids,
        prev_prev_input_ids=prev_prev_input_ids,
        topk_cs_idx=topk_cs_idx,
        output_ids=output_ids,
        ngram_spec=ngram_spec,
        topk=int(topk),
        vocab_size=int(vocab_size),
    )
    cache[key] = prepared
    return prepared


def welm_mtp_draft_ngram_hash_from_history(
    *,
    forward_batch: Any,
    input_ids: torch.Tensor,
    history_state: WelmMTPDraftNGramHistoryState,
    parent_indices: torch.Tensor,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    hashed_out: torch.Tensor,
    next_history_state: torch.Tensor | None,
    vocab_size: int,
    base_query_count: int,
    use_parent: bool,
    topk: int = 1,
    topk_cs_idx: torch.Tensor | None = None,
    prev_input_ids_scratch: torch.Tensor | None = None,
    prev_prev_input_ids_scratch: torch.Tensor | None = None,
    output_ids_scratch: torch.Tensor | None = None,
    output_prev_input_ids_scratch: torch.Tensor | None = None,
    source_indices_scratch: torch.Tensor | None = None,
) -> WelmMTPDraftNGramHistory | None:
    if not should_use_mk_welm_mtp_draft_ngram_hash():
        return None
    _require_mk_draft_ngram(
        getattr(forward_batch, _DEFERRED_PREPARED_LAUNCH_ATTR, None) is None,
        "previous mk draft ngram hash prepared launch was not consumed by forward.",
    )
    assert input_ids.device.type == "cuda", "MTP draft input_ids must be on CUDA"
    assert input_ids.dtype == torch.int64, "MTP draft input_ids must be int64"
    assert input_ids.is_contiguous(), "MTP draft input_ids must be contiguous"
    if input_ids.numel() == 0:
        return WelmMTPDraftNGramHistory(input_ids, input_ids)
    _require_mk_draft_ngram(
        bool(oe_grams) and len(oe_grams) == len(oe_vocab_sizes),
        "mk draft ngram hash requires matching non-empty ngram specs.",
    )
    max_gram = max(int(g) for g in oe_grams)
    _require_mk_draft_ngram(
        max_gram <= 3,
        f"mk draft ngram hash supports ngram orders up to 3, got {max_gram}.",
    )

    num_tokens = int(input_ids.numel())
    _require_mk_draft_ngram(
        num_tokens <= 1024,
        f"mk draft ngram hash supports up to 1024 tokens, got {num_tokens}.",
    )
    requires_prev_prev = max_gram >= 3
    base_prev_input_ids, base_prev_prev_input_ids, history_batch_size = (
        _draft_ngram_history_vectors(
            history_state,
            requires_prev_prev=requires_prev_prev,
        )
    )
    _require_mk_draft_ngram(
        base_prev_input_ids.device == input_ids.device,
        "mk draft ngram history must be on the same CUDA device as input_ids.",
    )
    _require_mk_draft_ngram(
        base_prev_input_ids.dtype == torch.int64,
        "mk draft ngram history prev_input_ids must be int64.",
    )
    if isinstance(base_prev_prev_input_ids, torch.Tensor):
        _require_mk_draft_ngram(
            base_prev_prev_input_ids.device == input_ids.device,
            "mk draft ngram history prev_prev_input_ids must be on the same "
            "CUDA device as input_ids.",
        )
        _require_mk_draft_ngram(
            base_prev_prev_input_ids.dtype == torch.int64,
            "mk draft ngram history prev_prev_input_ids must be int64.",
        )

    handle = _load_mk_draft_ngram()
    _require_mk_draft_ngram(
        handle is not None,
        "SGLANG_WELM_MTP_DRAFT_NGRAM_HASH is enabled, but mk draft ngram hash "
        "could not be imported.",
    )
    assert handle is not None
    params_cls, ngram_spec_cls, draft_ngram_hash, prepare_draft_ngram_hash = handle

    device = input_ids.device
    input_ids_i64 = input_ids

    _require_mk_draft_ngram(
        use_parent or base_query_count > 0,
        "mk draft ngram hash requires positive base_query_count.",
    )
    _require_mk_draft_ngram(
        use_parent or int(base_query_count) == history_batch_size,
        "mk draft ngram hash base_query_count does not match history batch "
        f"size: {base_query_count} vs {history_batch_size}.",
    )

    requested_topk = int(topk)
    _require_mk_draft_ngram(
        requested_topk > 0,
        f"mk draft ngram hash requires positive topk, got {requested_topk}.",
    )
    if topk_cs_idx is not None:
        _require_mk_draft_ngram(
            use_parent,
            "mk draft ngram hash topk_cs_idx is only valid for parent-selected draft.",
        )
        if topk_cs_idx.device != device or topk_cs_idx.dtype != torch.int64:
            topk_cs_idx = topk_cs_idx.to(device=device, dtype=torch.int64)
        if not topk_cs_idx.is_contiguous():
            topk_cs_idx = topk_cs_idx.contiguous()
        _require_mk_draft_ngram(
            topk_cs_idx.ndim == 2
            and topk_cs_idx.shape[1] == requested_topk
            and topk_cs_idx.numel() == num_tokens,
            "mk draft ngram hash topk_cs_idx shape must be "
            f"({num_tokens // requested_topk}, {requested_topk}), got "
            f"{tuple(topk_cs_idx.shape)}.",
        )
        _require_mk_draft_ngram(
            history_batch_size >= num_tokens,
            "mk draft ngram hash topk_cs_idx requires draft history rows for "
            f"all parent candidates: {history_batch_size} vs {num_tokens}.",
        )
        mk_topk = requested_topk
        source_indices = None
    elif use_parent:
        mk_topk = 1
        source_indices = _draft_ngram_source_indices(
            input_ids=input_ids_i64,
            parent_indices=parent_indices,
            base_query_count=base_query_count,
            use_parent=True,
            source_indices_scratch=source_indices_scratch,
        )
        _require_mk_draft_ngram(
            source_indices is not None,
            "mk draft ngram hash requires parent indices for parent-selected draft.",
        )
    else:
        _require_mk_draft_ngram(
            num_tokens % int(base_query_count) == 0,
            "mk draft ngram hash cannot infer first-step topk: "
            f"num_tokens={num_tokens}, base_query_count={base_query_count}.",
        )
        mk_topk = num_tokens // int(base_query_count)
        source_indices = None

    if source_indices is None:
        prev_input_ids = base_prev_input_ids
        if not prev_input_ids.is_contiguous():
            prev_input_ids = _valid_int64_scratch(
                prev_input_ids_scratch,
                device=device,
                numel=max(num_tokens, history_batch_size),
                name="prev_input_ids",
            )
            if prev_input_ids is None:
                prev_input_ids = torch.empty(
                    (max(num_tokens, history_batch_size),),
                    device=device,
                    dtype=torch.int64,
                )
            prev_input_ids[:history_batch_size].copy_(base_prev_input_ids)
    else:
        prev_input_ids = _valid_int64_scratch(
            prev_input_ids_scratch,
            device=device,
            numel=num_tokens,
            name="prev_input_ids",
        )
        if prev_input_ids is None:
            prev_input_ids = torch.empty((num_tokens,), device=device, dtype=torch.int64)
        _copy_draft_ngram_history_column(
            base_prev_input_ids, source_indices, prev_input_ids
        )

    prev_prev_input_ids = _valid_int64_scratch(
        output_prev_input_ids_scratch,
        device=device,
        numel=num_tokens,
        name="prev_prev_input_ids",
    )
    if prev_prev_input_ids is None:
        prev_prev_input_ids = _valid_int64_scratch(
            prev_prev_input_ids_scratch,
            device=device,
            numel=num_tokens,
            name="prev_prev_input_ids",
        )
    if prev_prev_input_ids is None:
        if (
            isinstance(base_prev_prev_input_ids, torch.Tensor)
            and base_prev_prev_input_ids.is_contiguous()
            and base_prev_prev_input_ids.numel() >= num_tokens
        ):
            prev_prev_input_ids = base_prev_prev_input_ids.reshape(-1)[:num_tokens]
        else:
            prev_prev_input_ids = torch.empty(
                (num_tokens,), device=device, dtype=torch.int64
            )
    if requires_prev_prev:
        assert base_prev_prev_input_ids is not None
        if isinstance(base_prev_prev_input_ids, torch.Tensor):
            if source_indices is None:
                rows_to_copy = min(int(base_prev_prev_input_ids.numel()), num_tokens)
                if not _is_same_tensor_view(
                    prev_prev_input_ids[:rows_to_copy],
                    base_prev_prev_input_ids.reshape(-1)[:rows_to_copy],
                ):
                    prev_prev_input_ids[:rows_to_copy].copy_(
                        base_prev_prev_input_ids.reshape(-1)[:rows_to_copy]
                    )
            else:
                _copy_draft_ngram_history_column(
                    base_prev_prev_input_ids, source_indices, prev_prev_input_ids
                )
        else:
            values = [int(x) for x in base_prev_prev_input_ids]
            _require_mk_draft_ngram(
                len(values) == int(base_query_count),
                "mk draft ngram hash first-step prev_prev_input_ids length does not "
                f"match base_query_count: {len(values)} vs {base_query_count}.",
            )
            prev_prev_input_ids[: len(values)].copy_(
                torch.tensor(values, device=device, dtype=torch.int64)
            )
    elif source_indices is not None:
        prev_prev_input_ids.zero_()

    ngram_spec = _build_mk_draft_ngram_spec(
        ngram_spec_cls, oe_grams, oe_vocab_sizes
    )
    output_ids = _valid_output_scratch(
        output_ids_scratch,
        device=device,
        shape=(num_tokens, len(ngram_spec)),
        name="output_ids",
    )
    if output_ids is None:
        output_ids = torch.empty(
            (num_tokens, len(ngram_spec)), device=device, dtype=torch.int64
        )
    output_ids_branch_major = output_ids.t()
    can_defer_prepared_launch = _is_same_tensor_view(
        hashed_out, output_ids_branch_major
    )

    use_prepared = output_ids_scratch is not None and getattr(
        forward_batch, "welm_mtp_skip_draft_proposal_build", False
    )
    try:
        prepared = None
        if use_prepared:
            prepared = _get_or_prepare_draft_ngram_hash(
                forward_batch=forward_batch,
                prepare_draft_ngram_hash=prepare_draft_ngram_hash,
                input_ids=input_ids_i64,
                prev_input_ids=prev_input_ids,
                prev_prev_input_ids=prev_prev_input_ids,
                topk_cs_idx=topk_cs_idx,
                output_ids=output_ids,
                ngram_spec=ngram_spec,
                topk=mk_topk,
                vocab_size=int(vocab_size),
            )
        if prepared is not None:
            if can_defer_prepared_launch:
                setattr(forward_batch, _DEFERRED_PREPARED_LAUNCH_ATTR, prepared)
            else:
                prepared.launch()
        else:
            if _is_current_stream_capturing():
                raise RuntimeError(
                    "mk draft ngram hash was not prepared before CUDA graph capture."
                )
            draft_ngram_hash(
                params_cls(
                    input_ids=input_ids_i64,
                    prev_input_ids=prev_input_ids,
                    prev_prev_input_ids=prev_prev_input_ids,
                    ngram_spec=ngram_spec,
                    topk=mk_topk,
                    topk_cs_idx=topk_cs_idx,
                    vocab_size=int(vocab_size),
                    output_ids=output_ids,
                )
            )
    except Exception as exc:
        _log_once(
            "run_failed",
            logging.WARNING,
            "mk draft ngram hash failed: %r",
            exc,
        )
        raise RuntimeError("mk draft ngram hash failed") from exc

    if (
        getattr(forward_batch, _DEFERRED_PREPARED_LAUNCH_ATTR, None) is None
        and not can_defer_prepared_launch
    ):
        hashed_out.copy_(output_ids_branch_major)
    _log_once(
        "use_mk_draft_decode_from_history",
        logging.INFO,
        "Using mk WeLM MTP draft ngram hash path: draft_decode_from_history",
    )
    return WelmMTPDraftNGramHistory(
        prev_input_ids=input_ids_i64,
        prev_prev_input_ids=prev_prev_input_ids,
    )
