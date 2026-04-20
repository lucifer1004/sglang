from __future__ import annotations

import logging
import os
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.vocab_parallel_embedding import get_masked_input_and_mask

logger = logging.getLogger(__name__)

WELM_OE_IMPL_ENV = "SGLANG_WELM_OE_IMPL"
WELM_OE_TRITON_PREPROCESS_ENV = "SGLANG_WELM_OE_TRITON_PREPROCESS"
WELM_OE_IMPL_LEGACY = "legacy"
WELM_OE_IMPL_TP_FUSED = "tp_fused"


@triton.jit
def _hash_mod_localize_kernel(
    input_ptr,
    hashed_ptr,
    local_idx_ptr,
    valid_mask_ptr,
    numel,
    vocab_size,
    shard_start,
    shard_end,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    input_ids = tl.load(input_ptr + offs, mask=mask, other=0).to(tl.uint64)
    vocab_size = vocab_size.to(tl.uint64)
    shard_start = shard_start.to(tl.uint64)
    shard_end = shard_end.to(tl.uint64)
    hashed = (input_ids * 2654435761) & 0xFFFFFFFF
    hashed = hashed % vocab_size
    valid = (hashed >= shard_start) & (hashed < shard_end)
    local_idx = tl.where(valid, hashed - shard_start, 0)

    tl.store(hashed_ptr + offs, hashed.to(tl.int64), mask=mask)
    tl.store(local_idx_ptr + offs, local_idx.to(tl.int64), mask=mask)
    tl.store(valid_mask_ptr + offs, valid.to(tl.int8), mask=mask)


def hash_input_ids_vectorized(input_ids: torch.Tensor) -> torch.Tensor:
    ids = input_ids.to(torch.int64)
    result = ids * 2654435761
    result = result & 0xFFFFFFFF
    return result.to(input_ids.dtype)


def _compute_welm_oe_hashed_inputs_fused(
    *,
    input_ids: torch.Tensor,
    oe_context,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    vocab_size: int,
    oe_embed_modules: Sequence,
    use_triton_preprocess: bool,
) -> list[torch.Tensor]:
    """Build n-grams and hash/localize them in a single pass over gram depth."""
    if not oe_grams:
        return []

    gram_to_branch_indices: dict[int, list[int]] = {}
    for branch_idx, gram in enumerate(oe_grams):
        gram_to_branch_indices.setdefault(gram, []).append(branch_idx)

    hashed_inputs: list[torch.Tensor | None] = [None] * len(oe_grams)
    running_ids = input_ids
    for g in range(1, max(oe_grams)):
        gram_tensor = oe_context.get_gram(g + 1)
        if gram_tensor is not None:
            running_ids = running_ids + gram_tensor * (vocab_size**g)

        for branch_idx in gram_to_branch_indices.get(g + 1, []):
            module = oe_embed_modules[branch_idx]
            hashed_ids, _, _ = hash_and_localize_welm_oe_input_ids(
                running_ids,
                oe_vocab_sizes[branch_idx],
                module.shard_indices.org_vocab_start_index,
                module.shard_indices.org_vocab_end_index,
                use_triton=use_triton_preprocess,
            )
            hashed_inputs[branch_idx] = hashed_ids

    assert all(hashed_input is not None for hashed_input in hashed_inputs)
    return [hashed_input for hashed_input in hashed_inputs]


def hash_and_localize_welm_oe_input_ids(
    input_ids: torch.Tensor,
    vocab_size: int,
    shard_start: int,
    shard_end: int,
    *,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hash OE token ids, apply vocab modulo, and map valid ids into local TP indices."""
    if input_ids.numel() == 0:
        empty = torch.empty_like(input_ids, dtype=torch.int64)
        return empty, empty, torch.empty_like(input_ids, dtype=torch.bool)

    if use_triton and input_ids.is_cuda:
        hashed = torch.empty_like(input_ids, dtype=torch.int64)
        local_idx = torch.empty_like(input_ids, dtype=torch.int64)
        valid_mask = torch.empty_like(input_ids, dtype=torch.int8)
        grid = (triton.cdiv(input_ids.numel(), 256),)
        _hash_mod_localize_kernel[grid](
            input_ids,
            hashed,
            local_idx,
            valid_mask,
            input_ids.numel(),
            vocab_size,
            shard_start,
            shard_end,
            BLOCK_SIZE=256,
        )
        return hashed, local_idx, valid_mask.to(torch.bool)

    hashed = hash_input_ids_vectorized(input_ids.to(torch.int64)) % vocab_size
    valid_mask = (hashed >= shard_start) & (hashed < shard_end)
    local_idx = torch.where(valid_mask, hashed - shard_start, torch.zeros_like(hashed))
    return hashed.to(torch.int64), local_idx.to(torch.int64), valid_mask


def _apply_oe_proj(oe_proj_module, hidden_states: torch.Tensor) -> torch.Tensor:
    if hasattr(oe_proj_module, "weight"):
        bias = getattr(oe_proj_module, "bias", None)
        return F.linear(hidden_states, oe_proj_module.weight, bias)
    output = oe_proj_module(hidden_states)
    if isinstance(output, tuple):
        return output[0]
    return output


def _supports_tp_fused_lookup(module) -> bool:
    required_attrs = ("weight", "tp_size", "shard_indices")
    return all(hasattr(module, attr) for attr in required_attrs)


def _get_oe_proj_bias(oe_proj_module) -> torch.Tensor | None:
    return getattr(oe_proj_module, "bias", None) if hasattr(oe_proj_module, "weight") else None


def _lookup_local_embedding(module, token_ids: torch.Tensor) -> torch.Tensor:
    shard_indices = module.shard_indices
    masked_input, input_mask = get_masked_input_and_mask(
        token_ids.long(),
        shard_indices.org_vocab_start_index,
        shard_indices.org_vocab_end_index,
        shard_indices.num_org_vocab_padding,
        shard_indices.added_vocab_start_index,
        shard_indices.added_vocab_end_index,
    )
    if hasattr(module, "quant_method") and hasattr(module.quant_method, "embedding"):
        emb_local = module.quant_method.embedding(module, masked_input.long())
    else:
        emb_local = F.embedding(masked_input.long(), module.weight)
    emb_local.masked_fill_(input_mask.unsqueeze(-1), 0)
    return emb_local


def compute_welm_oe_concat_local_partials(
    *,
    input_ids: torch.Tensor,
    forward_batch,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    vocab_size: int,
    oe_embed_modules: Sequence,
    use_triton_preprocess: bool = True,
) -> torch.Tensor:
    """Compute local OE embedding contributions before a single concatenated all-reduce."""
    if not oe_grams:
        return input_ids.new_zeros((input_ids.shape[0], 0), dtype=torch.float32)

    hashed_inputs = _compute_welm_oe_hashed_inputs_fused(
        input_ids=input_ids,
        oe_context=forward_batch.oe_context,
        oe_grams=oe_grams,
        oe_vocab_sizes=oe_vocab_sizes,
        vocab_size=vocab_size,
        oe_embed_modules=oe_embed_modules,
        use_triton_preprocess=use_triton_preprocess,
    )

    local_embeddings = []
    for i, _ in enumerate(oe_vocab_sizes):
        module = oe_embed_modules[i]
        if not _supports_tp_fused_lookup(module):
            raise TypeError(
                "OE TP fused lookup requires embedding modules with weight/tp_size/shard_indices"
            )

        local_embeddings.append(_lookup_local_embedding(module, hashed_inputs[i]))

    return torch.cat(local_embeddings, dim=-1)


def _compute_welm_oe_proj_reference(
    *,
    input_ids: torch.Tensor,
    forward_batch,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    vocab_size: int,
    oe_embed_modules: Sequence,
    oe_proj_module,
) -> torch.Tensor:
    input_ids_ngram = []
    input_ids_ngram_tmp = input_ids
    for g in range(1, max(oe_grams)):
        gram_tensor = forward_batch.oe_context.get_gram(g + 1)
        if gram_tensor is not None:
            input_ids_ngram_tmp = input_ids_ngram_tmp + gram_tensor * (vocab_size**g)
        input_ids_ngram.append(hash_input_ids_vectorized(input_ids_ngram_tmp))

    emb_ngram = []
    for i, vs in enumerate(oe_vocab_sizes):
        input_ids_ngram_hashed_tmp = input_ids_ngram[oe_grams[i] - 2] % vs
        emb_ngram_tmp = oe_embed_modules[i](input_ids_ngram_hashed_tmp)
        emb_ngram.append(emb_ngram_tmp)
    return _apply_oe_proj(oe_proj_module, torch.cat(emb_ngram, dim=-1))


def _should_use_tp_fused_path(
    oe_embed_modules: Sequence,
    *,
    implementation: str,
) -> bool:
    if implementation != WELM_OE_IMPL_TP_FUSED:
        return False
    if not oe_embed_modules:
        return False
    if get_attn_tp_context().input_scattered:
        return False
    return all(_supports_tp_fused_lookup(module) for module in oe_embed_modules)


def get_welm_oe_implementation(implementation: str | None = None) -> str:
    if implementation is None:
        implementation = os.getenv(WELM_OE_IMPL_ENV, WELM_OE_IMPL_LEGACY)

    normalized = implementation.strip().lower()
    if normalized in {"legacy", "reference", "old"}:
        return WELM_OE_IMPL_LEGACY
    if normalized in {"tp_fused", "fused", "new", "optimized"}:
        return WELM_OE_IMPL_TP_FUSED

    logger.warning(
        "%s=%r is invalid; falling back to %s",
        WELM_OE_IMPL_ENV,
        implementation,
        WELM_OE_IMPL_LEGACY,
    )
    return WELM_OE_IMPL_LEGACY


def should_use_welm_oe_triton_preprocess(
    use_triton_preprocess: bool | None = None,
) -> bool:
    if use_triton_preprocess is not None:
        return use_triton_preprocess

    value = os.getenv(WELM_OE_TRITON_PREPROCESS_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def compute_welm_oe_embedding(
    *,
    input_ids: torch.Tensor,
    forward_batch,
    base_hidden_states: torch.Tensor,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    vocab_size: int,
    oe_embed_modules: Sequence,
    oe_proj_module,
    implementation: str | None = None,
    use_triton_preprocess: bool | None = None,
    all_reduce_fn=tensor_model_parallel_all_reduce,
) -> torch.Tensor:
    """Compute WelmV4 OE embeddings with an OE-specific delayed-all-reduce fast path."""
    if not oe_grams:
        return base_hidden_states

    implementation = get_welm_oe_implementation(implementation)
    use_triton_preprocess = should_use_welm_oe_triton_preprocess(
        use_triton_preprocess
    )
    if _should_use_tp_fused_path(
        oe_embed_modules,
        implementation=implementation,
    ):
        concat_hidden = compute_welm_oe_concat_local_partials(
            input_ids=input_ids,
            forward_batch=forward_batch,
            oe_grams=oe_grams,
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=vocab_size,
            oe_embed_modules=oe_embed_modules,
            use_triton_preprocess=use_triton_preprocess,
        )
        if any(getattr(module, "tp_size", 1) > 1 for module in oe_embed_modules):
            concat_hidden = all_reduce_fn(concat_hidden)
        emb_new = _apply_oe_proj(oe_proj_module, concat_hidden)
    else:
        emb_new = _compute_welm_oe_proj_reference(
            input_ids=input_ids,
            forward_batch=forward_batch,
            oe_grams=oe_grams,
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=vocab_size,
            oe_embed_modules=oe_embed_modules,
            oe_proj_module=oe_proj_module,
        )

    return (base_hidden_states + emb_new) / 2.0
