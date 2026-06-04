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
                "welm_oe_hash_decode_from_prefixes",
                "WelmOeHashDecodeFromPrefixes::run",
            ),
            (
                "welm_oe_hash_segments_from_prefixes",
                "WelmOeHashSegmentsFromPrefixes::run",
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
    history_width = max(int(history_width), max_gram - 1)
    prefixes = [0] * history_width
    input_ids = torch.zeros((1,), dtype=torch.int64, device=device)
    hashed_out = torch.empty((len(oe_grams), 1), dtype=torch.int64, device=device)

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
