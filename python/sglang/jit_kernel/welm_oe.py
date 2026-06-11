from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import torch
import tvm_ffi

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)
_logged_hash_kernel_calls: set[str] = set()


def _log_hash_kernel_call_once(name: str) -> None:
    if name in _logged_hash_kernel_calls:
        return
    logger.info("Using WeLM OE hash JIT kernel path: %s", name)
    _logged_hash_kernel_calls.add(name)


@cache_once
def _jit_welm_oe_hash_module() -> Module:
    logger.info("Loading WeLM OE hash JIT module.")
    return load_jit(
        "welm_oe_hash",
        cuda_files=["welm/oe_decode_hash.cuh"],
        cuda_wrappers=[
            (
                "welm_oe_hash_mtp_init_history_from_prefixes",
                "WelmOeHashMtpInitHistoryFromPrefixes::run",
            ),
            (
                "welm_oe_hash_decode_from_prefixes",
                "WelmOeHashDecodeFromPrefixes::run",
            ),
            (
                "welm_oe_hash_segments_from_prefixes",
                "WelmOeHashSegmentsFromPrefixes::run",
            ),
            (
                "welm_oe_hash_mtp_target_verify_from_history",
                "WelmOeHashMtpTargetVerifyFromHistory::run",
            ),
            (
                "welm_oe_hash_mtp_draft_extend_after_verify_from_history",
                "WelmOeHashMtpDraftExtendAfterVerifyFromHistory::run",
            ),
            (
                "welm_oe_hash_mtp_draft_decode_from_history",
                "WelmOeHashMtpDraftDecodeFromHistory::run",
            ),
        ],
    )


def _shape(values: Sequence[int]) -> tvm_ffi.Shape:
    return tvm_ffi.Shape(tuple(int(v) for v in values))


def welm_oe_hash_decode_from_prefixes_cuda(
    input_ids: torch.Tensor,
    prefixes: Sequence[int],
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    hashed_out: torch.Tensor,
    vocab_size: int,
) -> None:
    """Compute WeLM OE hashes for decode segments.

    Args:
        input_ids: ``int32`` or ``int64`` CUDA tensor with shape
            ``[num_tokens]``. Each token is the current decode token for one
            request.
        prefixes: CPU sequence with shape ``[history_width, num_tokens]`` in
            row-major flattened order. ``prefixes[(lag - 1) * num_tokens + i]``
            is the token ``lag`` positions before ``input_ids[i]`` under the
            decode/overlap scheduler semantics.
        oe_grams: CPU sequence with shape ``[num_branches]``. Each value is the
            n-gram length for the corresponding OE branch.
        oe_vocab_sizes: CPU sequence with shape ``[num_branches]``. Each value
            is the modulo vocabulary size for that OE branch.
        hashed_out: ``int64`` CUDA tensor with shape
            ``[num_branches, num_tokens]``. The kernel writes
            ``hashed_out[branch, token]``.
        vocab_size: Base model vocabulary size used to compose n-gram ids.
    """
    module = _jit_welm_oe_hash_module()
    _log_hash_kernel_call_once("decode_from_prefixes")
    module.welm_oe_hash_decode_from_prefixes(
        input_ids,
        _shape(prefixes),
        _shape(oe_grams),
        _shape(oe_vocab_sizes),
        hashed_out,
        int(vocab_size),
    )


def welm_oe_hash_mtp_init_history_from_prefixes_cuda(
    prefixes: Sequence[int],
    history_out: torch.Tensor,
    first_token_ids: torch.Tensor | None = None,
) -> None:
    """Initialize compact MTP OE history from CPU prefix rows.

    Args:
        prefixes: CPU sequence with flattened shape ``[history_width, bs]``.
            ``prefixes[(lag - 1) * bs + i]`` is the token ``lag`` positions
            before request ``i`` enters the MTP stage.
        history_out: ``int64`` CUDA tensor with shape ``[bs, history_width]``.
            The last column is the immediate previous token.
        first_token_ids: Optional CUDA tensor with shape ``[bs]``. When set,
            the kernel appends this already-verified token as the last history
            column and shifts prefix rows one column earlier.
    """
    module = _jit_welm_oe_hash_module()
    _log_hash_kernel_call_once("mtp_init_history_from_prefixes")
    has_first_token = first_token_ids is not None
    if first_token_ids is None:
        first_token_ids = history_out
    module.welm_oe_hash_mtp_init_history_from_prefixes(
        first_token_ids,
        _shape(prefixes),
        history_out,
        int(has_first_token),
    )


def welm_oe_hash_segments_from_prefixes_cuda(
    input_ids: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    prefixes: Sequence[int],
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    hashed_out: torch.Tensor,
    vocab_size: int,
) -> None:
    """Compute WeLM OE hashes for prefill/mixed token segments.

    Args:
        input_ids: ``int32`` or ``int64`` CUDA tensor with shape
            ``[num_tokens]`` containing the normal forward input ids for all
            current segments.
        extend_start_loc: ``int32`` CUDA tensor with shape ``[num_segments]``.
            ``extend_start_loc[s]`` is the start offset of segment ``s`` in
            ``input_ids``.
        extend_seq_lens: ``int32`` CUDA tensor with shape ``[num_segments]``.
            ``extend_seq_lens[s]`` is the current forward length of segment
            ``s``.
        prefixes: CPU sequence with shape ``[history_width, num_segments]`` in
            row-major flattened order. ``prefixes[(lag - 1) * num_segments + s]``
            is the token ``lag`` positions before segment ``s`` starts.
        oe_grams: CPU sequence with shape ``[num_branches]``.
        oe_vocab_sizes: CPU sequence with shape ``[num_branches]``.
        hashed_out: ``int64`` CUDA tensor with shape
            ``[num_branches, num_tokens]``.
        vocab_size: Base model vocabulary size used to compose n-gram ids.
    """
    module = _jit_welm_oe_hash_module()
    _log_hash_kernel_call_once("segments_from_prefixes")
    module.welm_oe_hash_segments_from_prefixes(
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        _shape(prefixes),
        _shape(oe_grams),
        _shape(oe_vocab_sizes),
        hashed_out,
        int(vocab_size),
    )


def welm_oe_hash_mtp_target_verify_from_history_cuda(
    draft_token_ids: torch.Tensor,
    tree_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    history_state: torch.Tensor,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    hashed_out: torch.Tensor,
    vocab_size: int,
    draft_token_num: int,
) -> None:
    """Compute target-verify OE hashes from compact GPU history state."""
    module = _jit_welm_oe_hash_module()
    _log_hash_kernel_call_once("mtp_target_verify_from_history")
    module.welm_oe_hash_mtp_target_verify_from_history(
        draft_token_ids,
        tree_mask,
        seq_lens,
        history_state,
        _shape(oe_grams),
        _shape(oe_vocab_sizes),
        hashed_out,
        int(vocab_size),
        int(draft_token_num),
    )


def welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda(
    input_ids: torch.Tensor,
    accepted_draft_token_ids: torch.Tensor,
    accept_lens: torch.Tensor,
    entry_history: torch.Tensor,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    hashed_out: torch.Tensor,
    next_history_state: torch.Tensor,
    vocab_size: int,
    draft_token_num: int,
    use_entry_history_for_extend_hash_prefix: bool = False,
) -> None:
    """Compute MTP draft-extend hashes and next compact history.

    ``entry_history`` has shape ``[bs, history_width]`` and already includes
    the current verified token in its last column. ``accept_lens`` includes the
    bonus token. ``next_history_state`` receives the compact history for the
    next draft entry after applying the accepted draft tail and selected bonus
    token.
    """
    module = _jit_welm_oe_hash_module()
    _log_hash_kernel_call_once("mtp_draft_extend_after_verify_from_history")
    module.welm_oe_hash_mtp_draft_extend_after_verify_from_history(
        input_ids,
        accepted_draft_token_ids,
        accept_lens,
        entry_history,
        _shape(oe_grams),
        _shape(oe_vocab_sizes),
        hashed_out,
        next_history_state,
        int(vocab_size),
        int(draft_token_num),
        int(use_entry_history_for_extend_hash_prefix),
    )


def welm_oe_hash_mtp_draft_decode_from_history_cuda(
    input_ids: torch.Tensor,
    history_state: torch.Tensor,
    parent_indices: torch.Tensor,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    hashed_out: torch.Tensor,
    next_history_state: torch.Tensor,
    vocab_size: int,
    base_query_count: int,
    use_parent: bool,
) -> None:
    """Compute MTP draft-decode hashes and per-candidate next history state."""
    module = _jit_welm_oe_hash_module()
    _log_hash_kernel_call_once("mtp_draft_decode_from_history")
    module.welm_oe_hash_mtp_draft_decode_from_history(
        input_ids,
        history_state,
        parent_indices,
        _shape(oe_grams),
        _shape(oe_vocab_sizes),
        hashed_out,
        next_history_state,
        int(vocab_size),
        int(base_query_count),
        int(use_parent),
    )


def warmup_welm_oe_hash_kernel(
    device: torch.device | str = "cuda",
    *,
    history_width: int,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
) -> None:
    if not oe_grams:
        return

    max_gram = max(int(g) for g in oe_grams)
    prefix_width = max(int(history_width), max_gram - 1)
    mtp_history_width = max(prefix_width + 1, max_gram)
    prefixes = [0] * prefix_width
    history_prefixes = [0] * mtp_history_width
    input_ids = torch.zeros((1,), dtype=torch.int64, device=device)
    history_state = torch.zeros(
        (1, mtp_history_width), dtype=torch.int64, device=device
    )
    hashed_out = torch.empty((len(oe_grams), 1), dtype=torch.int64, device=device)

    welm_oe_hash_mtp_init_history_from_prefixes_cuda(history_prefixes, history_state)
    welm_oe_hash_mtp_init_history_from_prefixes_cuda(
        history_prefixes, history_state, first_token_ids=input_ids
    )

    welm_oe_hash_decode_from_prefixes_cuda(
        input_ids,
        prefixes,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        vocab_size=1,
    )

    extend_start_loc = torch.zeros((1,), dtype=torch.int32, device=device)
    extend_seq_lens = torch.ones((1,), dtype=torch.int32, device=device)
    welm_oe_hash_segments_from_prefixes_cuda(
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        prefixes,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        vocab_size=1,
    )
    tree_mask = torch.ones((2,), dtype=torch.bool, device=device)
    seq_lens = torch.ones((1,), dtype=torch.int64, device=device)
    welm_oe_hash_mtp_target_verify_from_history_cuda(
        input_ids,
        tree_mask,
        seq_lens,
        history_state,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        vocab_size=1,
        draft_token_num=1,
    )
    accepted = torch.zeros((1, 1), dtype=torch.int64, device=device)
    accept_lens = torch.ones((1,), dtype=torch.int64, device=device)
    next_history_state = torch.empty_like(history_state)
    welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda(
        input_ids,
        accepted,
        accept_lens,
        history_state,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        next_history_state,
        vocab_size=1,
        draft_token_num=1,
    )
    parent_indices = torch.zeros((1,), dtype=torch.int64, device=device)
    welm_oe_hash_mtp_draft_decode_from_history_cuda(
        input_ids,
        history_state,
        parent_indices,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        next_history_state,
        vocab_size=1,
        base_query_count=1,
        use_parent=False,
    )
