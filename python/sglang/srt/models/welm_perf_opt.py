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
SPECIALIZED_WELM_OE_GRAMS = (2, 2, 3, 3)
SPECIALIZED_WELM_OE_BRANCHES = 4
SPECIALIZED_WELM_OE_DIM = 512
DEFAULT_SPECIALIZED_WELM_OE_EMBED_BLOCK_D = 512
DEFAULT_SPECIALIZED_WELM_OE_EMBED_NUM_WARPS = 1


@triton.jit
def _welm_oe_lookup_concat_2233_kernel(
    input_ptr,
    gram2_ptr,
    gram3_ptr,
    weight0_ptr,
    weight1_ptr,
    weight2_ptr,
    weight3_ptr,
    out_ptr,
    num_tokens,
    vocab_size,
    vocab_size_sq,
    oe_vocab_size_0,
    oe_vocab_size_1,
    oe_vocab_size_2,
    oe_vocab_size_3,
    shard_start_0,
    shard_start_1,
    shard_start_2,
    shard_start_3,
    shard_end_0,
    shard_end_1,
    shard_end_2,
    shard_end_3,
    weight0_row_stride,
    weight1_row_stride,
    weight2_row_stride,
    weight3_row_stride,
    out_row_stride,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dim_block_idx = tl.program_id(1)
    offs_d = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = token_idx < num_tokens
    dim_mask = offs_d < EMBED_DIM

    input_ids = tl.load(input_ptr + token_idx, mask=token_mask, other=0).to(tl.uint32)
    gram2 = tl.load(gram2_ptr + token_idx, mask=token_mask, other=0).to(tl.uint32)
    gram3 = tl.load(gram3_ptr + token_idx, mask=token_mask, other=0).to(tl.uint32)

    vocab_size = vocab_size.to(tl.uint32)
    vocab_size_sq = vocab_size_sq.to(tl.uint32)
    running_ids_2 = input_ids + gram2 * vocab_size
    running_ids_3 = running_ids_2 + gram3 * vocab_size_sq
    hashed_2 = running_ids_2 * 2654435761
    hashed_3 = running_ids_3 * 2654435761

    bucket0 = hashed_2 % oe_vocab_size_0.to(tl.uint32)
    bucket1 = hashed_2 % oe_vocab_size_1.to(tl.uint32)
    bucket2 = hashed_3 % oe_vocab_size_2.to(tl.uint32)
    bucket3 = hashed_3 % oe_vocab_size_3.to(tl.uint32)
    valid0 = token_mask & (bucket0 >= shard_start_0.to(tl.uint32)) & (
        bucket0 < shard_end_0.to(tl.uint32)
    )
    valid1 = token_mask & (bucket1 >= shard_start_1.to(tl.uint32)) & (
        bucket1 < shard_end_1.to(tl.uint32)
    )
    valid2 = token_mask & (bucket2 >= shard_start_2.to(tl.uint32)) & (
        bucket2 < shard_end_2.to(tl.uint32)
    )
    valid3 = token_mask & (bucket3 >= shard_start_3.to(tl.uint32)) & (
        bucket3 < shard_end_3.to(tl.uint32)
    )

    row0 = (bucket0 - shard_start_0.to(tl.uint32)).to(tl.int64)
    row1 = (bucket1 - shard_start_1.to(tl.uint32)).to(tl.int64)
    row2 = (bucket2 - shard_start_2.to(tl.uint32)).to(tl.int64)
    row3 = (bucket3 - shard_start_3.to(tl.uint32)).to(tl.int64)

    mask0 = valid0 & dim_mask
    mask1 = valid1 & dim_mask
    mask2 = valid2 & dim_mask
    mask3 = valid3 & dim_mask

    emb0 = tl.load(weight0_ptr + row0 * weight0_row_stride + offs_d, mask=mask0, other=0.0)
    emb1 = tl.load(weight1_ptr + row1 * weight1_row_stride + offs_d, mask=mask1, other=0.0)
    emb2 = tl.load(weight2_ptr + row2 * weight2_row_stride + offs_d, mask=mask2, other=0.0)
    emb3 = tl.load(weight3_ptr + row3 * weight3_row_stride + offs_d, mask=mask3, other=0.0)

    out_token_base = token_idx * out_row_stride
    tl.store(out_ptr + out_token_base + offs_d, emb0, mask=token_mask & dim_mask)
    tl.store(
        out_ptr + out_token_base + EMBED_DIM + offs_d,
        emb1,
        mask=token_mask & dim_mask,
    )
    tl.store(
        out_ptr + out_token_base + 2 * EMBED_DIM + offs_d,
        emb2,
        mask=token_mask & dim_mask,
    )
    tl.store(
        out_ptr + out_token_base + 3 * EMBED_DIM + offs_d,
        emb3,
        mask=token_mask & dim_mask,
    )


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
    """Build n-grams first, then hash/localize the selected gram per branch."""
    if not oe_grams:
        return []

    ngram_inputs: list[torch.Tensor] = []
    running_ids = input_ids
    for g in range(1, max(oe_grams)):
        gram_tensor = oe_context.get_gram(g + 1)
        if gram_tensor is not None:
            running_ids = running_ids + gram_tensor * (vocab_size**g)
        ngram_inputs.append(running_ids)

    hashed_inputs = []
    for branch_idx, vocab_size_branch in enumerate(oe_vocab_sizes):
        module = oe_embed_modules[branch_idx]
        hashed_ids, _, _ = hash_and_localize_welm_oe_input_ids(
            ngram_inputs[oe_grams[branch_idx] - 2],
            vocab_size_branch,
            module.shard_indices.org_vocab_start_index,
            module.shard_indices.org_vocab_end_index,
        )
        hashed_inputs.append(hashed_ids)
    return hashed_inputs

def _can_use_specialized_welm_oe_lookup_concat(
    input_ids: torch.Tensor,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    oe_embed_modules: Sequence,
    use_triton_preprocess: bool,
) -> bool:
    if not (
        use_triton_preprocess
        and input_ids.is_cuda
        and tuple(oe_grams) == SPECIALIZED_WELM_OE_GRAMS
        and len(oe_vocab_sizes) == SPECIALIZED_WELM_OE_BRANCHES
        and len(oe_embed_modules) == SPECIALIZED_WELM_OE_BRANCHES
    ):
        return False
    return all(
        hasattr(module, "weight")
        and (
            not hasattr(module, "quant_method")
            or module.quant_method.__class__.__name__ == "UnquantizedEmbeddingMethod"
        )
        and module.weight.is_cuda
        and module.weight.dim() == 2
        and module.weight.shape[1] == SPECIALIZED_WELM_OE_DIM
        and module.weight.stride(1) == 1
        and module.weight.dtype == oe_embed_modules[0].weight.dtype
        and module.shard_indices.num_org_vocab_padding == 0
        and module.shard_indices.num_added_elements_padded == 0
        for module in oe_embed_modules
    )


def _compute_welm_oe_concat_local_partials_specialized_2233(
    *,
    input_ids: torch.Tensor,
    oe_context,
    oe_vocab_sizes: Sequence[int],
    vocab_size: int,
    oe_embed_modules: Sequence,
) -> torch.Tensor:
    if input_ids.numel() == 0:
        return torch.empty(
            (0, SPECIALIZED_WELM_OE_BRANCHES * SPECIALIZED_WELM_OE_DIM),
            device=input_ids.device,
            dtype=oe_embed_modules[0].weight.dtype,
        )

    gram2 = oe_context.get_gram(2)
    gram3 = oe_context.get_gram(3)
    assert gram2 is not None, "2233 specialized OE path requires oe_context.get_gram(2)"
    assert gram3 is not None, "2233 specialized OE path requires oe_context.get_gram(3)"

    output = torch.empty(
        (input_ids.numel(), SPECIALIZED_WELM_OE_BRANCHES * SPECIALIZED_WELM_OE_DIM),
        device=input_ids.device,
        dtype=oe_embed_modules[0].weight.dtype,
    )
    grid = (
        input_ids.numel(),
        triton.cdiv(SPECIALIZED_WELM_OE_DIM, DEFAULT_SPECIALIZED_WELM_OE_EMBED_BLOCK_D),
    )
    _welm_oe_lookup_concat_2233_kernel[grid](
        input_ids,
        gram2,
        gram3,
        oe_embed_modules[0].weight,
        oe_embed_modules[1].weight,
        oe_embed_modules[2].weight,
        oe_embed_modules[3].weight,
        output,
        input_ids.numel(),
        vocab_size,
        vocab_size * vocab_size,
        oe_vocab_sizes[0],
        oe_vocab_sizes[1],
        oe_vocab_sizes[2],
        oe_vocab_sizes[3],
        oe_embed_modules[0].shard_indices.org_vocab_start_index,
        oe_embed_modules[1].shard_indices.org_vocab_start_index,
        oe_embed_modules[2].shard_indices.org_vocab_start_index,
        oe_embed_modules[3].shard_indices.org_vocab_start_index,
        oe_embed_modules[0].shard_indices.org_vocab_end_index,
        oe_embed_modules[1].shard_indices.org_vocab_end_index,
        oe_embed_modules[2].shard_indices.org_vocab_end_index,
        oe_embed_modules[3].shard_indices.org_vocab_end_index,
        oe_embed_modules[0].weight.stride(0),
        oe_embed_modules[1].weight.stride(0),
        oe_embed_modules[2].weight.stride(0),
        oe_embed_modules[3].weight.stride(0),
        output.stride(0),
        BLOCK_D=DEFAULT_SPECIALIZED_WELM_OE_EMBED_BLOCK_D,
        EMBED_DIM=SPECIALIZED_WELM_OE_DIM,
        num_warps=DEFAULT_SPECIALIZED_WELM_OE_EMBED_NUM_WARPS,
    )
    return output


def hash_and_localize_welm_oe_input_ids(
    input_ids: torch.Tensor,
    vocab_size: int,
    shard_start: int,
    shard_end: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hash OE token ids, apply vocab modulo, and map valid ids into local TP indices."""
    if input_ids.numel() == 0:
        empty = torch.empty_like(input_ids, dtype=torch.int64)
        return empty, empty, torch.empty_like(input_ids, dtype=torch.bool)

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

    if _can_use_specialized_welm_oe_lookup_concat(
        input_ids,
        oe_grams,
        oe_vocab_sizes,
        oe_embed_modules,
        use_triton_preprocess,
    ):
        return _compute_welm_oe_concat_local_partials_specialized_2233(
            input_ids=input_ids,
            oe_context=forward_batch.oe_context,
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=vocab_size,
            oe_embed_modules=oe_embed_modules,
        )

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
