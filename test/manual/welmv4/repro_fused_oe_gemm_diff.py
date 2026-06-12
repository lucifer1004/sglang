"""Single-process repro of mk fused decode embedding-GEMM-all-reduce.

Loads a TP-1 ``embed_tokens`` + ``oe_embed`` from ``~/models`` weights, hands
synthetic ``input_ids`` / prefixes through both the unfused reference and
mk's fused kernel, and reports the per-step max-abs diff. Designed to be run
under torchrun --nproc_per_node=4.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def _load_oe_shapes(model_path: str) -> dict:
    cfg = json.loads(Path(model_path, "config.json").read_text())
    return {
        "vocab_size": int(cfg["vocab_size"]),
        "hidden_size": int(cfg["hidden_size"]),
        "oe_dim": int(cfg["oe_dim"]),
        "oe_grams": list(cfg["oe_grams"]),
        "oe_vocab_sizes": list(cfg["oe_vocab_sizes"]),
    }


def _hash_branch_ref(input_ids, prefixes, gram, oe_vocab_size, vocab_size):
    """Match sglang's CUDA hash kernel byte-for-byte."""
    mask = 0xFFFFFFFF
    out = []
    for token_idx in range(input_ids.numel()):
        running = int(input_ids[token_idx].item()) & mask
        vocab_power = vocab_size
        for lag in range(1, gram):
            prev = int(prefixes[lag - 1][token_idx]) & mask
            running = (running + prev * vocab_power) & mask
            vocab_power = (vocab_power * vocab_size) & mask
        hashed = (running * 2654435761) & mask
        out.append(hashed % oe_vocab_size)
    return torch.tensor(out, dtype=torch.int64, device=input_ids.device)


def _unfused_reference(
    input_ids,
    prefixes,
    embed_table,
    hash_tables,
    oe_proj_weight,
    *,
    rank,
    world_size,
    vocab_size,
    oe_grams,
    oe_vocab_sizes,
    hidden_size,
    oe_dim,
    process_group,
):
    """Reproduce sglang's compute_welm_oe_embedding + (base+emb)/2 path."""
    # 1. Token embedding lookup with shard mask + all-reduce.
    shard_size = vocab_size // world_size
    shard_begin = rank * shard_size
    shard_end = shard_begin + shard_size
    in_shard = (input_ids >= shard_begin) & (input_ids < shard_end)
    rows = torch.where(in_shard, input_ids - shard_begin, torch.zeros_like(input_ids))
    base_local = embed_table[rows.long()]  # (B, H)
    base_local.masked_fill_(~in_shard.unsqueeze(-1), 0)
    base = base_local.clone()
    dist.all_reduce(base, op=dist.ReduceOp.SUM, group=process_group)

    # 2. Per-branch hash lookup with shard mask, then concat.
    pieces = []
    for branch_idx, table in enumerate(hash_tables):
        gram = oe_grams[branch_idx]
        oe_vocab_size = oe_vocab_sizes[branch_idx]
        hashed = _hash_branch_ref(input_ids, prefixes, gram, oe_vocab_size, vocab_size)
        branch_shard = oe_vocab_size // world_size
        branch_begin = rank * branch_shard
        branch_end = branch_begin + branch_shard
        branch_in = (hashed >= branch_begin) & (hashed < branch_end)
        branch_rows = torch.where(
            branch_in, hashed - branch_begin, torch.zeros_like(hashed)
        )
        branch_local = table[branch_rows.long()]
        branch_local.masked_fill_(~branch_in.unsqueeze(-1), 0)
        pieces.append(branch_local)
    concat_local = torch.cat(pieces, dim=-1)  # (B, 4*oe_dim)
    concat = concat_local.clone()
    dist.all_reduce(concat, op=dist.ReduceOp.SUM, group=process_group)

    # 3. oe_proj GEMM (replicated).
    emb_new = torch.nn.functional.linear(concat, oe_proj_weight, bias=None)
    return ((base + emb_new) / 2.0).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/josephyu/models")
    parser.add_argument("--input-id", type=int, default=13)
    parser.add_argument("--prefix1", type=int, default=19260)
    parser.add_argument("--prefix2", type=int, default=356)
    parser.add_argument(
        "--use-real-weights",
        action="store_true",
        help="load actual model.embed_tokens / oe_embed / oe_up_proj from "
        "the safetensors checkpoint instead of synthetic random tables",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    shapes = _load_oe_shapes(args.model)
    vocab_size = shapes["vocab_size"]
    hidden_size = shapes["hidden_size"]
    oe_dim = shapes["oe_dim"]
    oe_grams = shapes["oe_grams"]
    oe_vocab_sizes = shapes["oe_vocab_sizes"]

    if rank == 0:
        print(f"# shapes: {shapes}", flush=True)

    # Synthetic embedding tables. We don't need real weights to test that
    # mk's fused kernel agrees with the same-input reference — both sides
    # see the same random tables.
    torch.manual_seed(20260609 + rank * 0)  # same seed across ranks for replicated proj
    embed_table = torch.randn(
        (vocab_size // world_size, hidden_size), dtype=torch.bfloat16, device=device
    )
    hash_tables = [
        torch.randn(
            (oev // world_size, oe_dim), dtype=torch.bfloat16, device=device
        )
        for oev in oe_vocab_sizes
    ]
    # Replicated proj — must be identical across ranks.
    proj_gen = torch.Generator(device=device).manual_seed(7)
    proj_weight = torch.randn(
        (hidden_size, len(oe_vocab_sizes) * oe_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=proj_gen,
    ).contiguous()

    # Same shard table needs to be different per rank (vocab partition).
    # Above we used per-rank seed=20260609 — that gives different content
    # per rank. We need the same content per row. Re-seed the embed table
    # so that rank r gets rows [r*shard:(r+1)*shard] of a globally consistent
    # random matrix.
    if args.use_real_weights:
        from safetensors import safe_open

        ckpt_idx = json.loads(Path(args.model, "model.safetensors.index.json").read_text())
        weight_map = ckpt_idx["weight_map"]

        def _load(name: str) -> torch.Tensor:
            shard_file = Path(args.model, weight_map[name])
            with safe_open(str(shard_file), framework="pt") as f:
                return f.get_tensor(name).to(device)

        if rank == 0:
            print("# loading real weights from checkpoint", flush=True)
        full_embed = _load("model.embed_tokens.weight")
        s = vocab_size // world_size
        embed_table = full_embed[rank * s : (rank + 1) * s].contiguous()
        del full_embed

        hash_tables = []
        for branch_idx, oev in enumerate(oe_vocab_sizes):
            full_h = _load(f"model.oe_embed.{branch_idx}.weight")
            s = oev // world_size
            hash_tables.append(full_h[rank * s : (rank + 1) * s].contiguous())
            del full_h

        proj_weight = _load("model.oe_up_proj.weight").contiguous()
    else:
        full_gen = torch.Generator(device=device).manual_seed(20260609)
        full_embed = torch.randn(
            (vocab_size, hidden_size), dtype=torch.bfloat16, device=device,
            generator=full_gen,
        )
        shard = vocab_size // world_size
        embed_table = full_embed[rank * shard : (rank + 1) * shard].contiguous()
        del full_embed

        full_hash_tables = []
        for branch_idx, oev in enumerate(oe_vocab_sizes):
            g = torch.Generator(device=device).manual_seed(20260609 + 100 + branch_idx)
            full_h = torch.randn(
                (oev, oe_dim), dtype=torch.bfloat16, device=device, generator=g,
            )
            s = oev // world_size
            full_hash_tables.append(full_h[rank * s : (rank + 1) * s].contiguous())
            del full_h
        hash_tables = full_hash_tables

    input_ids = torch.tensor([args.input_id], dtype=torch.int64, device=device)
    prefixes = [
        torch.tensor([args.prefix1], dtype=torch.int64, device=device),
        torch.tensor([args.prefix2], dtype=torch.int64, device=device),
    ]
    per_token_prefixes = [[args.prefix1, args.prefix2]]

    # Reference.
    ref = _unfused_reference(
        input_ids,
        prefixes,
        embed_table,
        hash_tables,
        proj_weight,
        rank=rank,
        world_size=world_size,
        vocab_size=vocab_size,
        oe_grams=oe_grams,
        oe_vocab_sizes=oe_vocab_sizes,
        hidden_size=hidden_size,
        oe_dim=oe_dim,
        process_group=dist.group.WORLD,
    )

    # mk fused.
    from mk.kernels import (
        FusedDecodeNGramHashEmbeddingGemmAllReduceParams,
        NGramSpec,
        fused_decode_ngram_hash_embedding_gemm_all_reduce,
    )

    ngram_spec = tuple(
        NGramSpec(int(g), int(v)) for g, v in zip(oe_grams, oe_vocab_sizes)
    )

    # Run mk twice in a row to expose any stale-state bugs (e.g. workspace
    # barrier counters not being reset).
    fused_a = fused_decode_ngram_hash_embedding_gemm_all_reduce(
        FusedDecodeNGramHashEmbeddingGemmAllReduceParams(
            input_ids=input_ids,
            prefixes=per_token_prefixes,
            input_embedding_table=embed_table,
            hash_embedding_tables=hash_tables,
            weight=proj_weight,
            process_group=dist.group.WORLD,
            vocab_size=vocab_size,
            ngram_spec=ngram_spec,
            input_hidden_size=hidden_size,
            hash_hidden_size=oe_dim,
            world_size=world_size,
        )
    )
    fused = fused_decode_ngram_hash_embedding_gemm_all_reduce(
        FusedDecodeNGramHashEmbeddingGemmAllReduceParams(
            input_ids=input_ids,
            prefixes=per_token_prefixes,
            input_embedding_table=embed_table,
            hash_embedding_tables=hash_tables,
            weight=proj_weight,
            process_group=dist.group.WORLD,
            vocab_size=vocab_size,
            ngram_spec=ngram_spec,
            input_hidden_size=hidden_size,
            hash_hidden_size=oe_dim,
            world_size=world_size,
        )
    )

    consec_diff = float((fused_a.float() - fused.float()).abs().max().item())

    diff = (ref.float() - fused.float()).abs()
    max_diff = float(diff.max().item())
    max_ref = float(ref.float().abs().max().item())
    if rank == 0:
        print(
            f"input_id={args.input_id} prefixes=({args.prefix1},{args.prefix2}) "
            f"max_abs_diff={max_diff:.6g} max_ref={max_ref:.6g} "
            f"rel={max_diff / max_ref if max_ref > 0 else 0.0:.6g} "
            f"consecutive_fused_diff={consec_diff:.6g}",
            flush=True,
        )
        print(f"ref[0,:8]={ref[0, :8].float().tolist()}", flush=True)
        print(f"fused[0,:8]={fused[0, :8].float().tolist()}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
