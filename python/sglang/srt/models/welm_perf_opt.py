from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import attn_tp_all_reduce
from sglang.srt.layers.vocab_parallel_embedding import get_masked_input_and_mask

logger = logging.getLogger(__name__)

WELM_OE_IMPL_ENV = "SGLANG_WELM_OE_IMPL"
WELM_OE_TRITON_PREPROCESS_ENV = "SGLANG_WELM_OE_TRITON_PREPROCESS"
WELM_OE_POST_PROJ_ALL_REDUCE_ENV = "SGLANG_WELM_OE_POST_PROJ_ALL_REDUCE"
# Fuses token embedding + concat OE embedding + oe_gate_up_proj GEMM + all-reduce
# into a single mk CUDA kernel. Only kicks in for low-batch decode and when
# all shape/world-size constraints match mk's supported instantiations. mk is
# imported lazily so that environments without mk installed still work as long
# as this env is left disabled.
WELM_OE_FUSED_DECODE_GEMM_ENV = "SGLANG_WELM_OE_FUSED_DECODE_GEMM"
WELM_OE_IMPL_FUSED_NGRAM_HASH = "fused_ngram_hash"
WELM_OE_HASH_INCOMPATIBLE_ENVS = (
    "SGLANG_DUMP_ACTIVATIONS",
    "WELM_USE_PREVIOUS_PRECISION",
)
SPECIALIZED_WELM_OE_GRAMS = (2, 2, 3, 3)
SPECIALIZED_WELM_OE_BRANCHES = 4
SPECIALIZED_WELM_OE_DIM = 512
DEFAULT_SPECIALIZED_WELM_OE_EMBED_BLOCK_D = 512
DEFAULT_SPECIALIZED_WELM_OE_EMBED_NUM_WARPS = 1

# mk fused decode embedding-GEMM-all-reduce limits (mirroring mk's static
# asserts; kept here so we can fast-fail without paying the import cost).
_MK_FUSED_DECODE_GEMM_MAX_BATCH = 32
_MK_FUSED_DECODE_GEMM_SUPPORTED_HIDDEN = frozenset(((1024, 256), (2048, 512)))
_MK_FUSED_DECODE_GEMM_SUPPORTED_WORLD_SIZES = frozenset((2, 4, 8))
_MK_FUSED_DECODE_GEMM_NGRAMS = (2, 2, 3, 3)
# Cached lazy import handle. Tuple of (kernels_module, params_cls,
# all_reduce_fn, ngram_spec_cls, supported_fn) once successfully imported.
# Set to ``False`` after a failed import so we only warn once.
_MK_FUSED_DECODE_GEMM_HANDLE = None
_MK_FUSED_DECODE_GEMM_NGRAM_SPEC_CACHE = None


@triton.jit
def _welm_oe_lookup_concat_prehashed_4x512_kernel(
    hash0_ptr,
    hash1_ptr,
    hash2_ptr,
    hash3_ptr,
    weight0_ptr,
    weight1_ptr,
    weight2_ptr,
    weight3_ptr,
    out_ptr,
    num_tokens,
    shard_start_0,
    shard_start_1,
    shard_start_2,
    shard_start_3,
    shard_end_0,
    shard_end_1,
    shard_end_2,
    shard_end_3,
    hash0_stride,
    hash1_stride,
    hash2_stride,
    hash3_stride,
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

    bucket0 = tl.load(hash0_ptr + token_idx * hash0_stride, mask=token_mask, other=0).to(
        tl.uint32
    )
    bucket1 = tl.load(hash1_ptr + token_idx * hash1_stride, mask=token_mask, other=0).to(
        tl.uint32
    )
    bucket2 = tl.load(hash2_ptr + token_idx * hash2_stride, mask=token_mask, other=0).to(
        tl.uint32
    )
    bucket3 = tl.load(hash3_ptr + token_idx * hash3_stride, mask=token_mask, other=0).to(
        tl.uint32
    )

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

    emb0 = tl.load(
        weight0_ptr + row0 * weight0_row_stride + offs_d, mask=mask0, other=0.0
    )
    emb1 = tl.load(
        weight1_ptr + row1 * weight1_row_stride + offs_d, mask=mask1, other=0.0
    )
    emb2 = tl.load(
        weight2_ptr + row2 * weight2_row_stride + offs_d, mask=mask2, other=0.0
    )
    emb3 = tl.load(
        weight3_ptr + row3 * weight3_row_stride + offs_d, mask=mask3, other=0.0
    )

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


def _get_cached_welm_oe_hashed_inputs(forward_batch):
    return getattr(forward_batch, "welm_oe_decode_hashed_inputs", None)


def _compute_welm_oe_hash_inputs(
    *,
    input_ids: torch.Tensor,
    forward_batch,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    vocab_size: int,
) -> list[torch.Tensor]:
    num_branches = len(oe_vocab_sizes)
    num_tokens = input_ids.numel()
    cached_hashed_inputs = _get_cached_welm_oe_hashed_inputs(forward_batch)
    if (
        cached_hashed_inputs is not None
        and cached_hashed_inputs.shape == (num_branches, num_tokens)
    ):
        return [cached_hashed_inputs[i] for i in range(num_branches)]

    hashed_inputs = torch.empty(
        (num_branches, num_tokens),
        device=input_ids.device,
        dtype=torch.int64,
    )

    if num_tokens == 0:
        return [hashed_inputs[i] for i in range(num_branches)]

    fill_welm_oe_hash_inputs(
        input_ids,
        hashed_inputs,
        forward_batch,
        oe_grams,
        oe_vocab_sizes,
        vocab_size,
    )
    return [hashed_inputs[i] for i in range(num_branches)]


def _flatten_hash_prefixes(prefix_rows) -> list[int]:
    return [token for row in prefix_rows for token in row]


def _extend_seq_lens_cpu_list(forward_batch) -> list[int]:
    extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_seq_lens_cpu is None:
        raise RuntimeError(
            "WeLM OE hash segment path requires extend_seq_lens_cpu."
        )
    if isinstance(extend_seq_lens_cpu, torch.Tensor):
        return [int(x) for x in extend_seq_lens_cpu.tolist()]
    return [int(x) for x in extend_seq_lens_cpu]


def fill_welm_oe_hash_inputs(
    input_ids: torch.Tensor,
    hashed_out: torch.Tensor,
    forward_batch,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    vocab_size: int,
) -> None:
    from sglang.jit_kernel.welm_oe import (
        welm_oe_hash_decode_from_prefixes_cuda,
        welm_oe_hash_segments_from_prefixes_cuda,
    )

    if input_ids.numel() == 0:
        return
    oe_context = getattr(forward_batch, "oe_context", None)
    prefix_rows = getattr(oe_context, "hash_prefixes", None)

    forward_mode = getattr(forward_batch, "forward_mode", None)
    is_target_verify = forward_mode is not None and forward_mode.is_target_verify()
    is_scale_seq_pseudo_target_verify = (
        is_target_verify
        and getattr(forward_batch, "spec_info", None) is None
        and max(getattr(forward_batch, "scale_seq_factor", 1), 1) > 1
    )
    if is_target_verify and not is_scale_seq_pseudo_target_verify:
        raise RuntimeError(
            "WeLM OE fused target-verify requires precomputed hashed inputs "
            "from MTP history state; CPU prefix fallback is not supported."
        )

    if prefix_rows is None:
        raise RuntimeError(
            "WeLM OE hash kernel path is missing CPU prefix state."
        )

    if forward_mode is not None and forward_mode.is_decode():
        num_segments = input_ids.numel()
        for row in prefix_rows:
            if len(row) != num_segments:
                raise RuntimeError(
                    "WeLM OE hash prefix rows must match decode tokens: "
                    f"{len(row)} vs {num_segments}."
                )
        welm_oe_hash_decode_from_prefixes_cuda(
            input_ids,
            _flatten_hash_prefixes(prefix_rows),
            oe_grams,
            oe_vocab_sizes,
            hashed_out,
            vocab_size,
        )
        return

    extend_start_loc = getattr(forward_batch, "extend_start_loc", None)
    extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
    if extend_start_loc is None or extend_seq_lens is None:
        raise RuntimeError(
            "WeLM OE hash segment path requires extend_start_loc and "
            "extend_seq_lens."
        )
    extend_seq_lens_cpu = _extend_seq_lens_cpu_list(forward_batch)
    # In scale_seq mode the OE hash kernel is invoked with bs-sized
    # input_ids (one input token per request) before the model expands
    # hidden_states by scale_seq_factor. forward_batch.extend_seq_lens
    # / extend_start_loc still reflect the post-expansion num_tokens
    # view (= sum * scale), so rescale the segment metadata so it
    # matches input_ids.numel().
    scale_seq_factor = max(getattr(forward_batch, "scale_seq_factor", 1), 1)
    if scale_seq_factor > 1:
        extend_seq_lens_cpu = [x // scale_seq_factor for x in extend_seq_lens_cpu]
        extend_start_loc = extend_start_loc // scale_seq_factor
        extend_seq_lens = extend_seq_lens // scale_seq_factor
    num_segments = len(extend_seq_lens_cpu)
    real_num_tokens = sum(extend_seq_lens_cpu)
    if real_num_tokens > input_ids.numel():
        raise RuntimeError(
            "WeLM OE hash segment lengths must sum to input tokens: "
            f"{real_num_tokens} vs {input_ids.numel()}."
        )
    if real_num_tokens < input_ids.numel():
        # AttnDP/MLP sync may pad input_ids for communication alignment. Synthetic
        # padding tokens should contribute zeroed hash buckets.
        hashed_out[:, real_num_tokens:].zero_()
    for row in prefix_rows:
        if len(row) != num_segments:
            raise RuntimeError(
                "WeLM OE hash prefix rows must match segments: "
                f"{len(row)} vs {num_segments}."
            )
    welm_oe_hash_segments_from_prefixes_cuda(
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        _flatten_hash_prefixes(prefix_rows),
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        vocab_size,
    )


def _has_welm_oe_hash_inputs(
    input_ids: torch.Tensor,
    forward_batch,
    oe_vocab_sizes: Sequence[int],
) -> bool:
    oe_context = getattr(forward_batch, "oe_context", None)
    cached_hashed_inputs = _get_cached_welm_oe_hashed_inputs(forward_batch)
    prefix_rows = getattr(oe_context, "hash_prefixes", None)
    has_cached_hash = cached_hashed_inputs is not None
    has_prefixes = prefix_rows is not None
    if not has_cached_hash and not has_prefixes:
        return False

    if not should_use_welm_oe_hash_kernel() or not input_ids.is_cuda:
        raise RuntimeError(
            "WeLM OE hash state is present, but the hash kernel path is not "
            "enabled for this forward."
        )
    expected_shape = (len(oe_vocab_sizes), input_ids.numel())
    if has_cached_hash:
        if cached_hashed_inputs.shape != expected_shape:
            raise RuntimeError(
                "WeLM OE cached hash shape mismatch: "
                f"{tuple(cached_hashed_inputs.shape)} vs {expected_shape}."
            )
        return True

    if len(prefix_rows) == 0:
        raise RuntimeError(
            "WeLM OE hash graph marker is missing cached hash inputs."
        )
    return True


def _can_use_specialized_welm_oe_prehashed_lookup_concat(
    hashed_inputs: Sequence[torch.Tensor],
    oe_embed_modules: Sequence,
    use_triton_preprocess: bool,
) -> bool:
    if not (
        use_triton_preprocess
        and len(hashed_inputs) == SPECIALIZED_WELM_OE_BRANCHES
        and len(oe_embed_modules) == SPECIALIZED_WELM_OE_BRANCHES
        and all(tensor.is_cuda and tensor.dim() == 1 for tensor in hashed_inputs)
        and all(tensor.numel() == hashed_inputs[0].numel() for tensor in hashed_inputs)
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


def _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512(
    *,
    hashed_inputs: Sequence[torch.Tensor],
    oe_embed_modules: Sequence,
) -> torch.Tensor:
    num_tokens = hashed_inputs[0].numel()
    if num_tokens == 0:
        return torch.empty(
            (0, SPECIALIZED_WELM_OE_BRANCHES * SPECIALIZED_WELM_OE_DIM),
            device=hashed_inputs[0].device,
            dtype=oe_embed_modules[0].weight.dtype,
        )

    output = torch.empty(
        (num_tokens, SPECIALIZED_WELM_OE_BRANCHES * SPECIALIZED_WELM_OE_DIM),
        device=hashed_inputs[0].device,
        dtype=oe_embed_modules[0].weight.dtype,
    )
    grid = (
        num_tokens,
        triton.cdiv(SPECIALIZED_WELM_OE_DIM, DEFAULT_SPECIALIZED_WELM_OE_EMBED_BLOCK_D),
    )
    _welm_oe_lookup_concat_prehashed_4x512_kernel[grid](
        hashed_inputs[0],
        hashed_inputs[1],
        hashed_inputs[2],
        hashed_inputs[3],
        oe_embed_modules[0].weight,
        oe_embed_modules[1].weight,
        oe_embed_modules[2].weight,
        oe_embed_modules[3].weight,
        output,
        num_tokens,
        oe_embed_modules[0].shard_indices.org_vocab_start_index,
        oe_embed_modules[1].shard_indices.org_vocab_start_index,
        oe_embed_modules[2].shard_indices.org_vocab_start_index,
        oe_embed_modules[3].shard_indices.org_vocab_start_index,
        oe_embed_modules[0].shard_indices.org_vocab_end_index,
        oe_embed_modules[1].shard_indices.org_vocab_end_index,
        oe_embed_modules[2].shard_indices.org_vocab_end_index,
        oe_embed_modules[3].shard_indices.org_vocab_end_index,
        hashed_inputs[0].stride(0),
        hashed_inputs[1].stride(0),
        hashed_inputs[2].stride(0),
        hashed_inputs[3].stride(0),
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


def _apply_oe_proj_no_bias(oe_proj_module, hidden_states: torch.Tensor) -> torch.Tensor:
    if hasattr(oe_proj_module, "weight"):
        return F.linear(hidden_states, oe_proj_module.weight, bias=None)
    return _apply_oe_proj(oe_proj_module, hidden_states)


def _add_oe_proj_bias(oe_proj_module, hidden_states: torch.Tensor) -> torch.Tensor:
    bias = getattr(oe_proj_module, "bias", None)
    if bias is None:
        return hidden_states
    return hidden_states + bias


def _lookup_local_embedding(module, token_ids: torch.Tensor) -> torch.Tensor:
    if not hasattr(module, "shard_indices"):
        if hasattr(module, "quant_method") and hasattr(module.quant_method, "embedding"):
            return module.quant_method.embedding(module, token_ids.long())
        return F.embedding(token_ids.long(), module.weight)

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

    if not _has_welm_oe_hash_inputs(
        input_ids,
        forward_batch,
        oe_vocab_sizes,
    ):
        raise RuntimeError(
            "WeLM OE requires CUDA hash-kernel inputs. Materialized n-gram "
            "fallback is no longer supported."
        )

    hashed_inputs = _compute_welm_oe_hash_inputs(
        input_ids=input_ids,
        forward_batch=forward_batch,
        oe_grams=oe_grams,
        oe_vocab_sizes=oe_vocab_sizes,
        vocab_size=vocab_size,
    )
    if _can_use_specialized_welm_oe_prehashed_lookup_concat(
        hashed_inputs,
        oe_embed_modules,
        use_triton_preprocess,
    ):
        return _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512(
            hashed_inputs=hashed_inputs,
            oe_embed_modules=oe_embed_modules,
        )
    local_embeddings = []
    for i, _ in enumerate(oe_vocab_sizes):
        module = oe_embed_modules[i]
        local_embeddings.append(_lookup_local_embedding(module, hashed_inputs[i]))

    return torch.cat(local_embeddings, dim=-1)


def get_welm_oe_implementation(implementation: str | None = None) -> str:
    if implementation is None:
        implementation = os.getenv(WELM_OE_IMPL_ENV, WELM_OE_IMPL_FUSED_NGRAM_HASH)

    normalized = implementation.strip().lower()
    if normalized in {"legacy", "reference", "old", "tp_fused", "fused", "new", "optimized"}:
        raise ValueError(
            f"{WELM_OE_IMPL_ENV}={implementation!r} is no longer supported. "
            f"Use {WELM_OE_IMPL_FUSED_NGRAM_HASH!r}."
        )
    if normalized == "fused_ngram_hash":
        return WELM_OE_IMPL_FUSED_NGRAM_HASH

    raise ValueError(
        f"{WELM_OE_IMPL_ENV}={implementation!r} is invalid; expected "
        f"{WELM_OE_IMPL_FUSED_NGRAM_HASH!r}."
    )


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_model_config_sequence(model_config, name: str):
    if model_config is None:
        return None
    for config in (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    ):
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def get_welm_oe_hash_config(model_config) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    oe_grams = tuple(
        int(g) for g in (_get_model_config_sequence(model_config, "oe_grams") or ())
    )
    oe_vocab_sizes = tuple(
        int(v)
        for v in (_get_model_config_sequence(model_config, "oe_vocab_sizes") or ())
    )
    return oe_grams, oe_vocab_sizes


def _model_config_supports_welm_oe_hash_kernel(model_config) -> bool:
    oe_grams = _get_model_config_sequence(model_config, "oe_grams")
    oe_vocab_sizes = _get_model_config_sequence(model_config, "oe_vocab_sizes")
    if oe_grams is None and oe_vocab_sizes is None:
        return True
    if not oe_grams or not oe_vocab_sizes:
        return False
    if len(oe_grams) != len(oe_vocab_sizes):
        return False
    return all(int(g) >= 2 for g in oe_grams) and all(
        int(v) > 0 for v in oe_vocab_sizes
    )


def validate_welm_oe_hash_kernel_compatibility(
    *,
    implementation: str | None = None,
) -> bool:
    get_welm_oe_implementation(implementation)
    conflicts = [
        name for name in WELM_OE_HASH_INCOMPATIBLE_ENVS if _env_flag(name)
    ]
    if conflicts:
        raise ValueError(
            f"{WELM_OE_IMPL_ENV}={WELM_OE_IMPL_FUSED_NGRAM_HASH} is "
            f"incompatible with {', '.join(conflicts)}."
        )
    return True


def should_use_welm_oe_triton_preprocess(
    use_triton_preprocess: bool | None = None,
) -> bool:
    if use_triton_preprocess is not None:
        return use_triton_preprocess

    value = os.getenv(WELM_OE_TRITON_PREPROCESS_ENV, "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def should_use_welm_oe_hash_kernel(model_config=None) -> bool:
    return (
        validate_welm_oe_hash_kernel_compatibility()
        and _model_config_supports_welm_oe_hash_kernel(model_config)
    )


def should_use_welm_oe_post_proj_all_reduce() -> bool:
    value = os.getenv(WELM_OE_POST_PROJ_ALL_REDUCE_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def should_use_welm_oe_fused_decode_gemm() -> bool:
    """Whether ``SGLANG_WELM_OE_FUSED_DECODE_GEMM`` opts in to the mk fused path.

    The actual decision still depends on runtime shape/state checks performed
    by :func:`_try_apply_welm_oe_fused_decode_gemm` — this only reflects the
    user intent expressed via the environment variable.
    """
    return _env_flag(WELM_OE_FUSED_DECODE_GEMM_ENV, default="0")


# Warnings about the fused decode path are deliberately rate-limited: once the
# precondition is missed it stays missed for the lifetime of the process, so
# logging on every forward would spam the server log.
_WELM_OE_FUSED_DECODE_GEMM_WARNED: set[str] = set()
_WELM_OE_FUSED_DECODE_GEMM_PROBE_STATE: dict[str, float] = {}


def _warn_welm_oe_fused_disabled_once(message: str, *args) -> None:
    """Emit a single warning about the fused decode path being disabled.

    Subsequent calls with the same ``message`` template are dropped.
    """
    if message in _WELM_OE_FUSED_DECODE_GEMM_WARNED:
        return
    _WELM_OE_FUSED_DECODE_GEMM_WARNED.add(message)
    logger.warning(
        f"{WELM_OE_FUSED_DECODE_GEMM_ENV}=1 disabled: {message}", *args
    )


def _load_mk_fused_decode_gemm():
    """Lazy-import the mk fused decode embedding-GEMM-all-reduce entry point.

    Returns ``None`` if mk is not installed (or fails to import). The first
    failure is logged at WARNING level; subsequent calls return ``None`` silently
    so a misconfigured environment does not flood the logs.

    Tuple layout: ``(params_cls, ngram_spec_cls, run_fused, is_supported,
    prepare_fn, prepared_cls)``. The last two are the CUDA-graph-friendly
    handle helpers; ``prepare_fn`` is ``None`` on older mk builds that
    pre-date the graph support, in which case the prepared path stays
    disabled even with the env on.
    """
    global _MK_FUSED_DECODE_GEMM_HANDLE
    if _MK_FUSED_DECODE_GEMM_HANDLE is False:
        return None
    if _MK_FUSED_DECODE_GEMM_HANDLE is not None:
        return _MK_FUSED_DECODE_GEMM_HANDLE
    try:
        from mk.kernels import (  # type: ignore[import-not-found]
            FusedDecodeNGramHashEmbeddingGemmAllReduceParams,
            NGramSpec,
            fused_decode_ngram_hash_embedding_gemm_all_reduce,
            is_fused_decode_ngram_hash_embedding_gemm_supported,
        )
    except Exception as exc:  # pragma: no cover - import failure path
        logger.warning(
            "%s=1 requested but mk fused decode GEMM is unavailable: %s. "
            "Falling back to the unfused embedding path.",
            WELM_OE_FUSED_DECODE_GEMM_ENV,
            exc,
        )
        _MK_FUSED_DECODE_GEMM_HANDLE = False
        return None
    # Optional graph-friendly helpers — present only on mk >= the commit that
    # introduced ``prepare_fused_decode_ngram_hash_embedding_gemm_all_reduce``.
    # We tolerate older builds by leaving these as None and surfacing the
    # missing symbol in :func:`prepare_welm_oe_fused_decode_handle`.
    try:
        from mk.kernels import (  # type: ignore[import-not-found]
            PreparedFusedDecodeNGramHashEmbeddingGemmAllReduce,
            prepare_fused_decode_ngram_hash_embedding_gemm_all_reduce,
        )
    except (ImportError, AttributeError):
        prepare_fused_decode_ngram_hash_embedding_gemm_all_reduce = None
        PreparedFusedDecodeNGramHashEmbeddingGemmAllReduce = None
    _MK_FUSED_DECODE_GEMM_HANDLE = (
        FusedDecodeNGramHashEmbeddingGemmAllReduceParams,
        NGramSpec,
        fused_decode_ngram_hash_embedding_gemm_all_reduce,
        is_fused_decode_ngram_hash_embedding_gemm_supported,
        prepare_fused_decode_ngram_hash_embedding_gemm_all_reduce,
        PreparedFusedDecodeNGramHashEmbeddingGemmAllReduce,
    )
    return _MK_FUSED_DECODE_GEMM_HANDLE


def _build_mk_ngram_spec(
    ngram_spec_cls,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
):
    """Build (and cache) the mk ``NGramSpec`` tuple for the OE branches.

    The OE n-gram layout is fixed by the model config and does not change
    per-forward, so we memoize the tuple to avoid re-allocating namedtuples on
    every decode step.
    """
    global _MK_FUSED_DECODE_GEMM_NGRAM_SPEC_CACHE
    cache_key = (tuple(oe_grams), tuple(oe_vocab_sizes))
    cached = _MK_FUSED_DECODE_GEMM_NGRAM_SPEC_CACHE
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    specs = tuple(
        ngram_spec_cls(int(n), int(v)) for n, v in zip(oe_grams, oe_vocab_sizes)
    )
    _MK_FUSED_DECODE_GEMM_NGRAM_SPEC_CACHE = (cache_key, specs)
    return specs


def _try_apply_welm_oe_fused_decode_gemm(
    *,
    input_ids: torch.Tensor,
    forward_batch,
    embed_tokens,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    oe_embed_modules,
    oe_proj_module,
) -> torch.Tensor | None:
    """Run the mk fused decode embedding-GEMM-all-reduce kernel if applicable.

    Returns the post-reduce hidden states tensor on success, or ``None`` when
    any precondition (decode mode, low batch, supported shape, ...) is not met
    — in which case the caller falls back to the unfused embedding path.
    """
    # Fast path: cuda_graph_runner pre-built a CUDA-graph-friendly Prepared
    # handle for this batch bucket. The handle owns persistent GPU/host
    # buffers (prefixes, output) whose pointers were baked into the captured
    # graph; refreshing the prefixes is done in cuda_graph_runner.replay_prepare
    # OUTSIDE the graph, and ``handle.launch()`` is the only thing captured
    # inside the graph. We do not re-validate shapes / re-resolve modules here
    # — the runner already validated everything at prepare time.
    prepared = getattr(forward_batch, "welm_oe_fused_prepared", None)
    fused_output = getattr(forward_batch, "welm_oe_fused_output", None)
    if prepared is not None and fused_output is not None:
        prepared.launch()
        return fused_output

    def _bail(reason: str) -> None:
        # Each distinct reason logs at most once per process. We dedupe on the
        # full reason text rather than the format template so the user can see
        # every precondition that ever blocked the fused path.
        key = f"precondition:{reason}"
        if key in _WELM_OE_FUSED_DECODE_GEMM_WARNED:
            return
        _WELM_OE_FUSED_DECODE_GEMM_WARNED.add(key)
        logger.warning(
            "%s=1 disabled: precondition not met: %s.",
            WELM_OE_FUSED_DECODE_GEMM_ENV,
            reason,
        )

    if oe_embed_modules is None or oe_proj_module is None:
        _bail("oe modules not present")
        return None
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_decode():
        _bail(f"not a decode forward (mode={forward_mode!r})")
        return None

    # mk hard-asserts batch ∈ [1, 32]; sglang's decode tokens-per-step covers
    # this for typical low-batch deployments.
    batch_size = int(input_ids.numel())
    if batch_size <= 0 or batch_size > _MK_FUSED_DECODE_GEMM_MAX_BATCH:
        _bail(f"batch_size={batch_size} out of [1, {_MK_FUSED_DECODE_GEMM_MAX_BATCH}]")
        return None

    if tuple(oe_grams) != _MK_FUSED_DECODE_GEMM_NGRAMS:
        _bail(f"oe_grams={tuple(oe_grams)} != {_MK_FUSED_DECODE_GEMM_NGRAMS}")
        return None
    if len(oe_vocab_sizes) != SPECIALIZED_WELM_OE_BRANCHES:
        _bail(f"len(oe_vocab_sizes)={len(oe_vocab_sizes)} != {SPECIALIZED_WELM_OE_BRANCHES}")
        return None
    if len(oe_embed_modules) != SPECIALIZED_WELM_OE_BRANCHES:
        _bail(f"len(oe_embed_modules)={len(oe_embed_modules)}")
        return None

    # OE projection must be a bias-less ReplicatedLinear sized
    # (hidden_size, 4 * oe_dim) — anything else (e.g. quantized, biased) means
    # the fused kernel's math contract no longer holds.
    proj_weight = getattr(oe_proj_module, "weight", None)
    if proj_weight is None or proj_weight.dim() != 2:
        _bail("oe_gate_up_proj.weight missing or not 2D")
        return None
    if getattr(oe_proj_module, "bias", None) is not None:
        _warn_welm_oe_fused_disabled_once(
            "oe_gate_up_proj has a bias term, which the fused decode kernel "
            "does not support."
        )
        return None

    input_hidden_size = int(proj_weight.shape[0])
    hash_hidden_size = int(proj_weight.shape[1] // SPECIALIZED_WELM_OE_BRANCHES)
    if (
        input_hidden_size,
        hash_hidden_size,
    ) not in _MK_FUSED_DECODE_GEMM_SUPPORTED_HIDDEN:
        _bail(
            f"unsupported (input_hidden_size, hash_hidden_size)="
            f"({input_hidden_size}, {hash_hidden_size})"
        )
        return None

    # OE module sharding determines which process group's all-reduce the
    # kernel must run; mismatch with embed_tokens' sharding would mix partials.
    first_oe = oe_embed_modules[0]
    use_attn_tp_group = bool(getattr(first_oe, "use_attn_tp_group", False))
    if bool(getattr(embed_tokens, "use_attn_tp_group", False)) != use_attn_tp_group:
        _bail("embed_tokens / oe_embed sharded on different TP groups")
        return None
    world_size = int(getattr(first_oe, "tp_size", 1) or 1)
    if world_size not in _MK_FUSED_DECODE_GEMM_SUPPORTED_WORLD_SIZES:
        _bail(f"world_size={world_size} not in {{2, 4, 8}}")
        return None

    # Scattered attention-TP inputs would feed only the local-rank slice into
    # the kernel, but mk expects the unscattered token ids. Bail out and let
    # the unfused path (which handles scatter explicitly) take over.
    if get_attn_tp_context().input_scattered:
        _bail("attention-TP input is scattered")
        return None

    oe_context = getattr(forward_batch, "oe_context", None)
    prefix_rows = getattr(oe_context, "hash_prefixes", None)
    if not prefix_rows:
        _bail("forward_batch.oe_context.hash_prefixes is empty/None")
        return None
    if len(prefix_rows) < (max(oe_grams) - 1):
        _bail(
            f"len(prefix_rows)={len(prefix_rows)} < max(oe_grams)-1={max(oe_grams) - 1}"
        )
        return None
    for row in prefix_rows:
        if len(row) != batch_size:
            _bail(
                f"prefix row width {len(row)} != batch_size {batch_size}"
            )
            return None

    # Embedding tables must be plain bf16 vocab-parallel shards aligned with
    # mk's ``(partition_size, hidden)`` contract. ``VocabParallelEmbedding``
    # rounds the underlying allocation up to the next ``padding_size *
    # world_size`` and stores the full padded shard locally; we pass that
    # whole tensor through and tell mk the partition size separately so it
    # can use the same boundaries when masking out-of-shard hashes. Quantized
    # tables are not supported.
    embed_weight = getattr(embed_tokens, "weight", None)
    if embed_weight is None or embed_weight.dtype is not torch.bfloat16:
        _bail("embed_tokens.weight missing or not bf16")
        return None
    if embed_weight.dim() != 2 or embed_weight.shape[1] != input_hidden_size:
        _bail(
            f"embed_tokens.weight has unexpected shape "
            f"{tuple(embed_weight.shape)}"
        )
        return None
    embed_shard_indices = getattr(embed_tokens, "shard_indices", None)
    if embed_shard_indices is not None and (
        embed_shard_indices.num_added_elements_padded != 0
    ):
        # Added vocab (e.g. extra trained tokens after the base vocab) breaks
        # the simple ``vocab_size / world_size`` contract mk validates against.
        _bail("embed_tokens has nonzero added-vocab padding")
        return None
    if not embed_weight.is_contiguous():
        _bail("embed_tokens.weight not contiguous")
        return None
    embed_table_vocab = int(
        getattr(embed_tokens, "org_vocab_size", embed_weight.shape[0] * world_size)
    )
    if embed_weight.shape[0] * world_size != embed_table_vocab:
        # mk's ``input_embedding_table`` validation expects an exact
        # ``vocab_size / world_size`` shard. For the WeLMV4 base vocab
        # (155648) with TP=4 there is no padding, so this is normally a
        # no-op; bail out instead of silently mis-sharding when it isn't.
        _bail(
            f"embed_tokens.weight rows {embed_weight.shape[0]} * world {world_size} "
            f"!= org_vocab {embed_table_vocab} (padded base vocab not supported)"
        )
        return None
    embed_table = embed_weight

    hash_tables: list[torch.Tensor] = []
    # mk supports a separate ``embedding_partition_size`` parameter so that
    # sglang can keep its native VocabParallelEmbedding sharding (which pads
    # vocab to a multiple of ``DEFAULT_VOCAB_PADDING_SIZE * world_size``)
    # while mk still applies the correct hash modulus. We resolve it from the
    # actual local OE shard rows; all four branches must agree (they do under
    # WeLMV4: the four ``oe_vocab_sizes`` differ by 8 each but pad to the same
    # 64-multiple shard width).
    embedding_partition_size: int | None = None
    for i, module in enumerate(oe_embed_modules):
        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype is not torch.bfloat16:
            _bail("oe_embed[i].weight missing or not bf16")
            return None
        if weight.dim() != 2 or weight.shape[1] != hash_hidden_size:
            _bail(
                f"oe_embed[i].weight unexpected shape {tuple(weight.shape)}"
            )
            return None
        shard_indices = getattr(module, "shard_indices", None)
        if shard_indices is not None and (
            shard_indices.num_added_elements_padded != 0
        ):
            _bail("oe_embed[i] has nonzero added-vocab padding")
            return None
        local_partition = int(weight.shape[0])
        if embedding_partition_size is None:
            embedding_partition_size = local_partition
        elif embedding_partition_size != local_partition:
            _bail(
                f"oe_embed[i].weight partition rows {local_partition} != "
                f"first branch's {embedding_partition_size}"
            )
            return None
        if not weight.is_contiguous():
            _bail("oe_embed[i].weight not contiguous")
            return None
        hash_tables.append(weight)

    handle = _load_mk_fused_decode_gemm()
    if handle is None:
        return None
    (
        params_cls,
        ngram_spec_cls,
        run_fused,
        is_supported,
        _prepare_fn,
        _prepared_cls,
    ) = handle
    ngram_spec = _build_mk_ngram_spec(ngram_spec_cls, oe_grams, oe_vocab_sizes)
    if not is_supported(
        input_hidden_size,
        hash_hidden_size,
        world_size,
        batch_size,
        ngram_spec,
        embedding_partition_size,
    ):
        _bail(
            f"mk reports unsupported "
            f"(in={input_hidden_size}, hash={hash_hidden_size}, "
            f"world={world_size}, b={batch_size}, "
            f"partition={embedding_partition_size})"
        )
        return None

    # Resolve the right process group: OE / embed sharded along the attention
    # TP group means the all-reduce must run on attn-TP, not the global TP.
    if use_attn_tp_group:
        from sglang.srt.layers.dp_attention import get_attention_tp_group

        process_group = get_attention_tp_group().device_group
    else:
        from sglang.srt.distributed import get_tp_group

        process_group = get_tp_group().device_group

    if not proj_weight.is_contiguous() or proj_weight.dtype is not torch.bfloat16:
        _bail("oe_gate_up_proj.weight not contiguous bf16")
        return None
    # mk requires 16-byte alignment on the GEMM weight; ReplicatedLinear gives
    # us a fresh contiguous bf16 tensor so this is normally fine, but verify.
    if proj_weight.data_ptr() % 16 != 0:
        _bail("oe_gate_up_proj.weight not 16B-aligned")
        return None

    input_ids_int64 = (
        input_ids if input_ids.dtype is torch.int64 else input_ids.to(torch.int64)
    )
    if not input_ids_int64.is_contiguous():
        input_ids_int64 = input_ids_int64.contiguous()

    # mk takes prefixes as a per-token list of lag-major ints; reuse the
    # cached oe_context rows (lag-major already; outer index = lag, inner =
    # token). Convert to mk's ``Sequence[Sequence[int]]`` layout (per-token
    # rows of length max_history).
    max_history = max(int(spec.n) for spec in ngram_spec) - 1
    per_token_prefixes = [
        [int(prefix_rows[lag][token]) for lag in range(max_history)]
        for token in range(batch_size)
    ]

    params = params_cls(
        input_ids=input_ids_int64,
        prefixes=per_token_prefixes,
        input_embedding_table=embed_table,
        hash_embedding_tables=hash_tables,
        weight=proj_weight,
        process_group=process_group,
        vocab_size=embed_table_vocab,
        ngram_spec=ngram_spec,
        input_hidden_size=input_hidden_size,
        hash_hidden_size=hash_hidden_size,
        world_size=world_size,
        embedding_partition_size=embedding_partition_size,
    )
    try:
        result = run_fused(params)
    except Exception as exc:  # pragma: no cover - mk runtime failure path
        logger.warning(
            "mk fused decode embedding-GEMM-all-reduce raised %r; falling back "
            "to the unfused path for this step.",
            exc,
        )
        return None
    if "fused_kernel_fired" not in _WELM_OE_FUSED_DECODE_GEMM_WARNED:
        # One-time positive signal so we can confirm in the server log that the
        # opt-in actually took effect (very useful for catching silent fallback).
        _WELM_OE_FUSED_DECODE_GEMM_WARNED.add("fused_kernel_fired")
        logger.info(
            "%s=1 active: mk fused decode embedding-GEMM-all-reduce engaged "
            "(input_hidden_size=%d, hash_hidden_size=%d, world_size=%d, "
            "batch_size=%d).",
            WELM_OE_FUSED_DECODE_GEMM_ENV,
            input_hidden_size,
            hash_hidden_size,
            world_size,
            batch_size,
        )
    return result


# ----------------------------------------------------------------------
# CUDA-graph-friendly entry points
# ----------------------------------------------------------------------
#
# The functions below feed mk's ``prepare_fused_decode_ngram_hash_embedding_gemm
# _all_reduce`` API. cuda_graph_runner builds one Prepared handle per captured
# decode batch size at runner-init time; the captured graph then only invokes
# ``prepared.launch()`` and the per-step prefix update is done OUTSIDE the
# graph in ``replay_prepare`` via :func:`build_welm_oe_fused_prefix_list` +
# ``handle.set_prefixes(...)``. Eligibility / shape gates duplicate the eager
# logic in :func:`_try_apply_welm_oe_fused_decode_gemm` because the runner
# cannot construct a real ``ForwardBatch`` at init time, only a model handle.
# Keeping the duplication explicit is intentional — the eager path remains
# the primary fallback for everything we don't capture (bs > 32, MTP,
# DP-attn, scale_seq, --disable-cuda-graph, env off, mk missing).


@dataclass
class WelmOEFusedDecodeConfig:
    """Stable per-runner shape/topology bundle for the mk fused decode kernel.

    All fields are derived from model construction state and the resolved
    process group; none of them depend on per-step input. Discovery is
    therefore safe to run once at ``CudaGraphRunner.__init__`` time and the
    result reused for every captured batch bucket.
    """

    embed_tokens: object  # the VocabParallelEmbedding for the base vocab
    oe_embed_modules: tuple
    oe_proj_module: object
    process_group: object
    world_size: int
    rank: int
    vocab_size: int  # base ``org_vocab_size`` of embed_tokens
    input_hidden_size: int
    hash_hidden_size: int
    embedding_partition_size: int
    oe_grams: tuple
    oe_vocab_sizes: tuple


def discover_welm_oe_fused_decode_modules(
    model,
) -> "WelmOEFusedDecodeConfig | None":
    """Resolve the fused-decode topology for ``model`` or return ``None``.

    Mirrors the structural checks in :func:`_try_apply_welm_oe_fused_decode_gemm`
    but is purely structural (no ``ForwardBatch``, no ``input_ids``). Returns
    a :class:`WelmOEFusedDecodeConfig` on success.
    """
    def _trace(reason: str) -> None:
        # Each distinct reason logs at most once per process so a misconfigured
        # WeLM deployment surfaces every gate that ever blocked the prepared
        # path, while a non-WeLM model — which fails at the first gate — only
        # emits a single line.
        key = f"discover_trace:{reason}"
        if key in _WELM_OE_FUSED_DECODE_GEMM_WARNED:
            return
        _WELM_OE_FUSED_DECODE_GEMM_WARNED.add(key)
        logger.info(
            "WeLM OE fused-decode discover rejected: %s", reason
        )

    embed_tokens = getattr(model, "embed_tokens", None) or getattr(
        getattr(model, "model", None), "embed_tokens", None
    )
    oe_embed_modules = getattr(model, "oe_embed", None) or getattr(
        getattr(model, "model", None), "oe_embed", None
    )
    oe_proj_module = getattr(model, "oe_gate_up_proj", None) or getattr(
        getattr(model, "model", None), "oe_gate_up_proj", None
    )
    if (
        embed_tokens is None
        or oe_embed_modules is None
        or oe_proj_module is None
    ):
        _trace(
            f"missing modules (embed_tokens={embed_tokens is not None}, "
            f"oe_embed={oe_embed_modules is not None}, "
            f"oe_gate_up_proj={oe_proj_module is not None})"
        )
        return None

    # Match the eager path's shape/topology contract.
    proj_weight = getattr(oe_proj_module, "weight", None)
    if proj_weight is None or proj_weight.dim() != 2:
        _trace("oe_gate_up_proj.weight missing or not 2D")
        return None
    if getattr(oe_proj_module, "bias", None) is not None:
        _trace("oe_gate_up_proj has bias")
        return None
    if not proj_weight.is_contiguous() or proj_weight.dtype is not torch.bfloat16:
        _trace(
            f"oe_gate_up_proj.weight contiguous={proj_weight.is_contiguous()} "
            f"dtype={proj_weight.dtype}"
        )
        return None
    if proj_weight.data_ptr() % 16 != 0:
        _trace("oe_gate_up_proj.weight not 16B-aligned")
        return None

    input_hidden_size = int(proj_weight.shape[0])
    hash_hidden_size = int(proj_weight.shape[1] // SPECIALIZED_WELM_OE_BRANCHES)
    if (
        input_hidden_size,
        hash_hidden_size,
    ) not in _MK_FUSED_DECODE_GEMM_SUPPORTED_HIDDEN:
        _trace(
            f"unsupported (input_hidden_size, hash_hidden_size)=("
            f"{input_hidden_size}, {hash_hidden_size})"
        )
        return None

    if oe_embed_modules is None:
        _trace("oe_embed is None")
        return None
    try:
        oe_embed_modules_count = len(oe_embed_modules)
    except TypeError:
        _trace("oe_embed is not iterable")
        return None
    if oe_embed_modules_count == 0:
        _trace("oe_embed is empty")
        return None
    if oe_embed_modules_count != SPECIALIZED_WELM_OE_BRANCHES:
        _trace(
            f"oe_embed has {oe_embed_modules_count} branches != "
            f"{SPECIALIZED_WELM_OE_BRANCHES}"
        )
        return None

    first_oe = oe_embed_modules[0]
    use_attn_tp_group = bool(getattr(first_oe, "use_attn_tp_group", False))
    if bool(getattr(embed_tokens, "use_attn_tp_group", False)) != use_attn_tp_group:
        _trace("embed_tokens / oe_embed sharded on different TP groups")
        return None
    world_size = int(getattr(first_oe, "tp_size", 1) or 1)
    if world_size not in _MK_FUSED_DECODE_GEMM_SUPPORTED_WORLD_SIZES:
        _trace(f"world_size={world_size} not in {{2,4,8}}")
        return None

    embed_weight = getattr(embed_tokens, "weight", None)
    if embed_weight is None or embed_weight.dtype is not torch.bfloat16:
        _trace("embed_tokens.weight missing or not bf16")
        return None
    if embed_weight.dim() != 2 or embed_weight.shape[1] != input_hidden_size:
        _trace(
            f"embed_tokens.weight shape {tuple(embed_weight.shape)} doesn't "
            f"match input_hidden_size={input_hidden_size}"
        )
        return None
    if not embed_weight.is_contiguous():
        _trace("embed_tokens.weight not contiguous")
        return None
    embed_shard_indices = getattr(embed_tokens, "shard_indices", None)
    if embed_shard_indices is not None and (
        embed_shard_indices.num_added_elements_padded != 0
    ):
        _trace("embed_tokens has nonzero added-vocab padding")
        return None
    embed_table_vocab = int(
        getattr(embed_tokens, "org_vocab_size", embed_weight.shape[0] * world_size)
    )
    if embed_weight.shape[0] * world_size != embed_table_vocab:
        _trace(
            f"embed_tokens shard rows {embed_weight.shape[0]} * world {world_size} "
            f"!= org_vocab {embed_table_vocab}"
        )
        return None

    embedding_partition_size: int | None = None
    for i, module in enumerate(oe_embed_modules):
        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype is not torch.bfloat16:
            _trace(f"oe_embed[{i}].weight missing or not bf16")
            return None
        if weight.dim() != 2 or weight.shape[1] != hash_hidden_size:
            _trace(
                f"oe_embed[{i}].weight unexpected shape {tuple(weight.shape)}"
            )
            return None
        if not weight.is_contiguous():
            _trace(f"oe_embed[{i}].weight not contiguous")
            return None
        shard_indices = getattr(module, "shard_indices", None)
        if shard_indices is not None and (
            shard_indices.num_added_elements_padded != 0
        ):
            _trace(f"oe_embed[{i}] has nonzero added-vocab padding")
            return None
        local_partition = int(weight.shape[0])
        if embedding_partition_size is None:
            embedding_partition_size = local_partition
        elif embedding_partition_size != local_partition:
            _trace(
                f"oe_embed partitions disagree: "
                f"{embedding_partition_size} vs {local_partition}"
            )
            return None

    if embedding_partition_size is None:
        _trace("embedding_partition_size is None after loop")
        return None

    # Resolve OE config (oe_grams / oe_vocab_sizes) from the model_config.
    model_config = getattr(model, "config", None) or getattr(
        getattr(model, "model", None), "config", None
    )
    oe_grams = tuple(int(g) for g in getattr(model_config, "oe_grams", ()) or ())
    oe_vocab_sizes = tuple(
        int(v) for v in getattr(model_config, "oe_vocab_sizes", ()) or ()
    )
    if (
        tuple(oe_grams) != _MK_FUSED_DECODE_GEMM_NGRAMS
        or len(oe_vocab_sizes) != SPECIALIZED_WELM_OE_BRANCHES
    ):
        _trace(
            f"unsupported config: oe_grams={oe_grams} "
            f"oe_vocab_sizes={oe_vocab_sizes}"
        )
        return None

    # Resolve the right process_group: OE / embed sharded along the attention
    # TP group means the all-reduce must run on attn-TP, not the global TP.
    try:
        if use_attn_tp_group:
            from sglang.srt.layers.dp_attention import get_attention_tp_group

            coord = get_attention_tp_group()
        else:
            from sglang.srt.distributed import get_tp_group

            coord = get_tp_group()
    except Exception as exc:
        _trace(f"failed to resolve process group coordinator: {exc!r}")
        return None
    process_group = getattr(coord, "device_group", None)
    if process_group is None:
        _trace("coordinator has no device_group")
        return None

    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            _trace("torch.distributed is not initialized")
            return None
        rank = int(dist.get_rank(process_group))
        pg_world = int(dist.get_world_size(process_group))
    except Exception as exc:
        _trace(f"failed to read rank/world_size from PG: {exc!r}")
        return None
    if pg_world != world_size:
        _trace(
            f"process_group world {pg_world} != module tp_size {world_size}"
        )
        return None

    return WelmOEFusedDecodeConfig(
        embed_tokens=embed_tokens,
        oe_embed_modules=tuple(oe_embed_modules),
        oe_proj_module=oe_proj_module,
        process_group=process_group,
        world_size=world_size,
        rank=rank,
        vocab_size=embed_table_vocab,
        input_hidden_size=input_hidden_size,
        hash_hidden_size=hash_hidden_size,
        embedding_partition_size=embedding_partition_size,
        oe_grams=tuple(oe_grams),
        oe_vocab_sizes=tuple(oe_vocab_sizes),
    )


def prepare_welm_oe_fused_decode_handle(
    *,
    config: WelmOEFusedDecodeConfig,
    static_input_ids: torch.Tensor,
    static_output: torch.Tensor,
    runtime_batch_size: int,
):
    """Build one mk Prepared handle for ``runtime_batch_size`` decode tokens.

    ``static_input_ids`` and ``static_output`` MUST be persistent GPU tensors
    whose ``data_ptr()`` is stable for the lifetime of every cuda graph that
    will capture ``handle.launch()``. cuda_graph_runner satisfies this by
    slicing the per-runner ``DecodeInputBuffers`` buffers (which themselves
    live for the runner's lifetime).

    Returns ``None`` if mk is missing, the prepared API isn't exported yet,
    or mk rejects the (shape, world, batch) tuple. The runner then falls
    back to the eager path inside the captured graph for that bucket.
    """
    if runtime_batch_size <= 0 or runtime_batch_size > _MK_FUSED_DECODE_GEMM_MAX_BATCH:
        return None
    if static_input_ids.numel() < runtime_batch_size:
        return None
    if static_output.shape != (runtime_batch_size, config.input_hidden_size):
        return None
    if static_output.dtype is not torch.bfloat16 or not static_output.is_contiguous():
        return None

    handle = _load_mk_fused_decode_gemm()
    if handle is None:
        return None
    (
        _params_cls,
        ngram_spec_cls,
        _run_fused,
        is_supported,
        prepare_fn,
        _prepared_cls,
    ) = handle
    if prepare_fn is None:
        # Older mk build without the graph-friendly API; the runner caller
        # treats this as ineligibility and stays on the eager captured path.
        if "prepare_api_missing" not in _WELM_OE_FUSED_DECODE_GEMM_WARNED:
            _WELM_OE_FUSED_DECODE_GEMM_WARNED.add("prepare_api_missing")
            logger.warning(
                "%s=1 cannot use cuda graph: installed mk lacks "
                "prepare_fused_decode_ngram_hash_embedding_gemm_all_reduce; "
                "upgrade mk to enable graph capture of the fused embedding.",
                WELM_OE_FUSED_DECODE_GEMM_ENV,
            )
        return None

    ngram_spec = _build_mk_ngram_spec(
        ngram_spec_cls, config.oe_grams, config.oe_vocab_sizes
    )
    if not is_supported(
        config.input_hidden_size,
        config.hash_hidden_size,
        config.world_size,
        runtime_batch_size,
        ngram_spec,
        config.embedding_partition_size,
    ):
        return None

    try:
        prepared = prepare_fn(
            input_ids=static_input_ids,
            input_embedding_table=config.embed_tokens.weight,
            hash_embedding_tables=tuple(
                m.weight for m in config.oe_embed_modules
            ),
            weight=config.oe_proj_module.weight,
            output=static_output,
            process_group=config.process_group,
            runtime_batch_size=runtime_batch_size,
            vocab_size=config.vocab_size,
            ngram_spec=ngram_spec,
            input_hidden_size=config.input_hidden_size,
            hash_hidden_size=config.hash_hidden_size,
            world_size=config.world_size,
            embedding_partition_size=config.embedding_partition_size,
        )
    except Exception as exc:
        # Broad except per audit: includes MKConfigError, RuntimeError from
        # NCCL/symm-mem rendezvous, CUDA OOM. Any failure here just means
        # this bucket falls through to the eager captured path.
        logger.warning(
            "%s=1: failed to prepare mk fused decode for bs=%d: %r; "
            "this batch size will use the eager captured fused path.",
            WELM_OE_FUSED_DECODE_GEMM_ENV,
            runtime_batch_size,
            exc,
        )
        return None

    if (
        f"prepared_engaged_bs{runtime_batch_size}"
        not in _WELM_OE_FUSED_DECODE_GEMM_WARNED
    ):
        _WELM_OE_FUSED_DECODE_GEMM_WARNED.add(
            f"prepared_engaged_bs{runtime_batch_size}"
        )
        logger.info(
            "%s=1 active: mk fused decode embedding-GEMM-all-reduce prepared "
            "for cuda graph (bs=%d, in=%d, hash=%d, world=%d, partition=%d).",
            WELM_OE_FUSED_DECODE_GEMM_ENV,
            runtime_batch_size,
            config.input_hidden_size,
            config.hash_hidden_size,
            config.world_size,
            config.embedding_partition_size,
        )
    return prepared


def build_welm_oe_fused_prefix_list(
    *,
    forward_batch,
    runtime_batch_size: int,
    max_history: int,
) -> list[list[int]]:
    """Build a ``list[list[int]]`` of prefix tokens for ``handle.set_prefixes``.

    Inner length is exactly ``max_history`` (mk pads missing slots with 0
    automatically, but we standardize for predictable host writes). Outer
    length is exactly ``runtime_batch_size``. When ``oe_context`` or
    ``hash_prefixes`` is missing (e.g. very first decode step before any
    history accumulates), every inner list is empty; mk zero-fills the
    pinned host buffer in that case.
    """
    oe_context = getattr(forward_batch, "oe_context", None)
    prefix_rows = getattr(oe_context, "hash_prefixes", None) if oe_context is not None else None
    if not prefix_rows:
        return [[] for _ in range(runtime_batch_size)]
    avail_history = min(max_history, len(prefix_rows))
    rows = []
    for token_idx in range(runtime_batch_size):
        row = []
        for lag in range(avail_history):
            lag_row = prefix_rows[lag]
            if lag_row is None or token_idx >= len(lag_row):
                # Missing prefix at this lag — mk fills with 0.
                continue
            row.append(int(lag_row[token_idx]))
        rows.append(row)
    return rows


def welm_embeddings(
    *,
    input_ids: torch.Tensor,
    forward_batch,
    embed_tokens,
    oe_grams: Sequence[int],
    oe_vocab_sizes: Sequence[int],
    vocab_size: int,
    oe_embed_modules,
    oe_proj_module,
    scale_seq_times: int,
    scale_seq_embed_tokens_list=None,
    scale_seq_oe_embed_list=None,
    scale_seq_oe_up_proj_list=None,
    input_embeds: torch.Tensor | None = None,
    skip_oe_fusion: bool = False,
) -> torch.Tensor:
    """Compute the WelmV4 input embeddings consumed by the decoder stack.

    This mirrors the embedding block at the head of ``Qwen2MoeModel.forward``
    (the logic that runs before ``model_forward_maybe_tbo`` on the first PP
    rank): token embedding lookup, optional over-encoding (OE) fusion, and
    optional scale-seq expansion. It is a free function with no ``model``
    parameter so the forward, split-forward, and VLM call sites can share the
    same embedding pipeline by passing in the relevant submodules and
    config values directly.

    Args:
        input_ids: Token ids for the current batch.
        forward_batch: The active ``ForwardBatch`` (carries ``oe_context``).
        embed_tokens: ``VocabParallelEmbedding`` for the base token table.
        oe_grams: OE n-gram sizes; when empty OE fusion is skipped entirely.
        oe_vocab_sizes: Per-branch OE vocab sizes.
        vocab_size: Base token vocab size.
        oe_embed_modules: Main OE embedding module list (matches ``oe_grams``).
            May be ``None`` when ``oe_grams`` is empty.
        oe_proj_module: Main OE projection ``ReplicatedLinear``.
            May be ``None`` when ``oe_grams`` is empty.
        scale_seq_times: Number of additional scale-seq embedding groups.
        scale_seq_embed_tokens_list: ``nn.ModuleList`` of base embeddings, one
            per scale-seq group. Required when ``scale_seq_times > 0``.
        scale_seq_oe_embed_list: ``nn.ModuleList[nn.ModuleList]`` of OE
            embeddings, one outer entry per scale-seq group. Required when
            ``scale_seq_times > 0`` and ``oe_grams`` is non-empty.
        scale_seq_oe_up_proj_list: ``nn.ModuleList`` of OE projection
            ``ReplicatedLinear`` modules, one per scale-seq group. Required
            when ``scale_seq_times > 0`` and ``oe_grams`` is non-empty.
        input_embeds: Optional precomputed token embeddings (e.g. for VLM
            paths). When provided, ``embed_tokens`` is skipped.
        skip_oe_fusion: If True, skip OE fusion on the main embedding path
            (scale-seq groups always get OE when configured, matching the
            historical behavior).

    Returns:
        ``hidden_states`` ready to feed into the decoder layers.
    """
    # Lazy import: ``welmv4`` imports this module at top level, so we defer
    # access to its module-level dump helpers to avoid a circular import.
    from sglang.srt.models.welmv4 import _WELM_DUMP_ENABLED, _welm_dump_tensor

    has_oe = (
        len(oe_grams) > 0 and getattr(forward_batch, "oe_context", None) is not None
    )

    # Fused fast path: a single mk kernel covers token embedding lookup +
    # OE-branch hash lookups + concat + oe_gate_up_proj GEMM + all-reduce, and
    # is enabled only for low-batch decode where every shape/world-size check
    # passes. ``scale_seq_times`` interleaves multiple embeddings per token,
    # which the fused kernel does not model — keep the unfused path for that.
    if (
        has_oe
        and not skip_oe_fusion
        and input_embeds is None
        and should_use_welm_oe_fused_decode_gemm()
    ):
        if scale_seq_times != 0:
            _warn_welm_oe_fused_disabled_once(
                "scale_seq_times=%d is not supported; "
                "fused decode embedding GEMM stays disabled.",
                scale_seq_times,
            )
        else:
            fused_hidden = _try_apply_welm_oe_fused_decode_gemm(
                input_ids=input_ids,
                forward_batch=forward_batch,
                embed_tokens=embed_tokens,
                oe_grams=oe_grams,
                oe_vocab_sizes=oe_vocab_sizes,
                oe_embed_modules=oe_embed_modules,
                oe_proj_module=oe_proj_module,
            )
            if fused_hidden is not None:
                # Optional numerical probe: when
                # ``SGLANG_WELM_OE_FUSED_DECODE_GEMM_PROBE=1`` is set, compute
                # the unfused reference on every fused call and log the
                # worst-ever max-abs / rel diff. Useful for catching a kernel
                # regression in production traffic without paying the cost on
                # the hot path by default. Skipped under cuda graph capture
                # (would bake unfused reference kernels into the graph and
                # blow up the capture region) and under cuda graph replay
                # (the host-side recompute would race against the captured
                # graph's reads of the static buffers).
                if (
                    _env_flag("SGLANG_WELM_OE_FUSED_DECODE_GEMM_PROBE", "0")
                    and not torch.cuda.is_current_stream_capturing()
                    and getattr(forward_batch, "welm_oe_fused_prepared", None) is None
                ):
                    try:
                        ref_base = embed_tokens(input_ids)
                        ref = compute_welm_oe_embedding(
                            input_ids=input_ids,
                            forward_batch=forward_batch,
                            base_hidden_states=ref_base,
                            oe_grams=oe_grams,
                            oe_vocab_sizes=oe_vocab_sizes,
                            vocab_size=vocab_size,
                            oe_embed_modules=oe_embed_modules,
                            oe_proj_module=oe_proj_module,
                        )
                        max_abs = float(
                            (ref.float() - fused_hidden.float()).abs().max().item()
                        )
                        max_ref = float(ref.float().abs().max().item())
                        prev = _WELM_OE_FUSED_DECODE_GEMM_PROBE_STATE.get(
                            "max_abs", 0.0
                        )
                        if max_abs > prev:
                            _WELM_OE_FUSED_DECODE_GEMM_PROBE_STATE["max_abs"] = max_abs
                            logger.info(
                                "fused vs unfused embedding probe NEW WORST: "
                                "max_abs_diff=%.6g max_ref=%.6g rel=%.6g "
                                "shape=%s input_ids[0]=%s",
                                max_abs,
                                max_ref,
                                max_abs / max_ref if max_ref > 0 else 0.0,
                                tuple(fused_hidden.shape),
                                int(input_ids[0].item())
                                if input_ids.numel() > 0
                                else None,
                            )
                    except Exception as exc:  # pragma: no cover - probe is best-effort
                        if "probe_failed" not in _WELM_OE_FUSED_DECODE_GEMM_WARNED:
                            _WELM_OE_FUSED_DECODE_GEMM_WARNED.add("probe_failed")
                            logger.warning("fused embedding probe failed: %r", exc)
                if _WELM_DUMP_ENABLED:
                    _welm_dump_tensor("model.embed_tokens.output", fused_hidden)
                return fused_hidden

    if input_embeds is None:
        hidden_states = embed_tokens(input_ids)
    else:
        hidden_states = input_embeds
    if _WELM_DUMP_ENABLED:
        _welm_dump_tensor("model.embed_tokens.output", hidden_states)

    if has_oe and not skip_oe_fusion:
        hidden_states = compute_welm_oe_embedding(
            input_ids=input_ids,
            forward_batch=forward_batch,
            base_hidden_states=hidden_states,
            oe_grams=oe_grams,
            oe_vocab_sizes=oe_vocab_sizes,
            vocab_size=vocab_size,
            oe_embed_modules=oe_embed_modules,
            oe_proj_module=oe_proj_module,
        )

    if scale_seq_times > 0:
        # Expand hidden_states from (T, D) to (T * scale, D) by interleaving
        # main embedding with scale_seq embeddings.
        # Layout per original token i:
        #   [main_emb_i, scale_seq_1_emb_i, ..., scale_seq_N_emb_i]
        scale = scale_seq_times + 1
        T = hidden_states.shape[0]
        D = hidden_states.shape[1]
        hidden_states = hidden_states.unsqueeze(1)  # (T, 1, D)
        hidden_states_list = [hidden_states]
        for s in range(scale_seq_times):
            hs_s = scale_seq_embed_tokens_list[s](input_ids)  # (T, D)
            if has_oe:
                hs_s = compute_welm_oe_embedding(
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    base_hidden_states=hs_s,
                    oe_grams=oe_grams,
                    oe_vocab_sizes=oe_vocab_sizes,
                    vocab_size=vocab_size,
                    oe_embed_modules=scale_seq_oe_embed_list[s],
                    oe_proj_module=scale_seq_oe_up_proj_list[s],
                )
            hs_s = hs_s.unsqueeze(1)  # (T, 1, D)
            hidden_states_list.append(hs_s)
        # (T, scale, D) -> (T * scale, D)
        hidden_states = torch.cat(hidden_states_list, dim=1)
        hidden_states = hidden_states.reshape(T * scale, D).contiguous()

    return hidden_states


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
    all_reduce_fn=None,
) -> torch.Tensor:
    """Compute WelmV4 OE embeddings with an OE-specific delayed-all-reduce fast path.

    The default ``all_reduce_fn`` is selected at call time based on whether the
    OE embedding modules are sharded along the attention TP group (DP attention
    deployments) or the global TP group (pure TP deployments). This keeps the
    collective in lock-step with how ``VocabParallelEmbedding`` partitioned the
    vocab — using the global all-reduce on attn-TP-sharded modules under DP
    attention would mix partials across DP groups (different input_ids per
    group) and produce wrong results.
    """
    if not oe_grams:
        return base_hidden_states

    if all_reduce_fn is None:
        # Pick the all-reduce primitive that matches how the OE modules were
        # sharded. ``use_attn_tp_group`` is set when DP attention is enabled.
        first_module = oe_embed_modules[0] if oe_embed_modules else None
        if getattr(first_module, "use_attn_tp_group", False):
            all_reduce_fn = attn_tp_all_reduce
        else:
            all_reduce_fn = tensor_model_parallel_all_reduce

    get_welm_oe_implementation(implementation)
    use_triton_preprocess = should_use_welm_oe_triton_preprocess(
        use_triton_preprocess
    )
    if not oe_embed_modules:
        raise RuntimeError(
            "WeLM OE requires embedding modules for the CUDA hash-kernel path."
        )
    if get_attn_tp_context().input_scattered:
        raise RuntimeError(
            f"{WELM_OE_IMPL_ENV}={WELM_OE_IMPL_FUSED_NGRAM_HASH} does not "
            "support scattered attention-TP inputs."
        )
    concat_hidden = compute_welm_oe_concat_local_partials(
        input_ids=input_ids,
        forward_batch=forward_batch,
        oe_grams=oe_grams,
        oe_vocab_sizes=oe_vocab_sizes,
        vocab_size=vocab_size,
        oe_embed_modules=oe_embed_modules,
        use_triton_preprocess=use_triton_preprocess,
    )
    if should_use_welm_oe_post_proj_all_reduce():
        emb_new_local = _apply_oe_proj_no_bias(oe_proj_module, concat_hidden)
        if any(getattr(module, "tp_size", 1) > 1 for module in oe_embed_modules):
            emb_new_local = all_reduce_fn(emb_new_local)
        emb_new = _add_oe_proj_bias(oe_proj_module, emb_new_local)
    else:
        if any(getattr(module, "tp_size", 1) > 1 for module in oe_embed_modules):
            concat_hidden = all_reduce_fn(concat_hidden)
        emb_new = _apply_oe_proj(oe_proj_module, concat_hidden)

    return (base_hidden_states + emb_new) / 2.0
