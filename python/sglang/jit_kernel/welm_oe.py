from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import tvm_ffi

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_welm_oe_decode_hash_module() -> Module:
    return load_jit(
        "welm_oe_decode_hash",
        cuda_files=["welm/oe_decode_hash.cuh"],
        cuda_wrappers=[
            (
                "welm_oe_decode_hash_from_prefixes",
                "WelmOeDecodeHashFromPrefixes::run",
            )
        ],
    )


def welm_oe_decode_hash_from_prefixes_cuda(
    input_ids: torch.Tensor,
    prefixes: Sequence[int],
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    hashed_out: torch.Tensor,
    vocab_size: int,
) -> None:
    """Compute WeLM OE decode hashes from CPU-packed per-sample history.

    Args:
        input_ids: ``int64`` CUDA tensor with shape ``[num_tokens]``. One
            current decode token per request in the batch.
        prefixes: CPU sequence with shape ``[history_width, num_tokens]`` in
            row-major flattened order. ``prefixes[(lag - 1) * num_tokens + i]``
            is the token ``lag`` positions before ``input_ids[i]`` under the
            decode/overlap scheduler semantics.
        oe_grams: CPU sequence with shape ``[num_branches]``. Each element is
            the n-gram length for the corresponding OE branch, e.g.
            ``[2, 2, 3, 3]``.
        oe_vocab_sizes: CPU sequence with shape ``[num_branches]``.
            ``oe_vocab_sizes[j]`` is the modulo vocabulary size for branch
            ``j``.
        hashed_out: ``int64`` CUDA tensor with shape
            ``[num_branches, num_tokens]``. The kernel writes hashed OE ids as
            ``hashed_out[j, i]`` for branch ``j`` and token ``i``.
        vocab_size: Base model vocabulary size used when composing n-gram ids.
    """
    oe_grams = tuple(int(g) for g in oe_grams)
    oe_vocab_sizes = tuple(int(v) for v in oe_vocab_sizes)

    module = _jit_welm_oe_decode_hash_module()
    module.welm_oe_decode_hash_from_prefixes(
        input_ids,
        tvm_ffi.Shape(prefixes),
        tvm_ffi.Shape(oe_grams),
        tvm_ffi.Shape(oe_vocab_sizes),
        hashed_out,
        int(vocab_size),
    )


def warmup_welm_oe_decode_hash_kernel(
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
    input_ids = torch.zeros((1,), dtype=torch.int64, device=device)
    prefixes = [0] * history_width
    hashed_out = torch.empty((len(oe_grams), 1), dtype=torch.int64, device=device)
    welm_oe_decode_hash_from_prefixes_cuda(
        input_ids,
        prefixes,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        vocab_size=1,
    )
