"""Replay an in-server fused decode embedding dump against mk standalone.

Loads the per-rank ``.pt`` dumps written by sglang's fused-vs-unfused probe
(``SGLANG_WELM_OE_FUSED_DECODE_GEMM_DUMP_DIR``) and re-runs mk's fused kernel
on the same input/weights/process_group, comparing against:

  1. The ``ref`` tensor saved alongside the dump (sglang unfused output).
  2. A live unfused reference recomputed from the dumped weights here.

Run with ``torchrun --nproc_per_node=4`` so the all-reduce world matches.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist


def _hash_branch_ref(input_ids, prefixes, gram, oe_vocab_size, vocab_size):
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
    process_group,
):
    shard_size = vocab_size // world_size
    shard_begin = rank * shard_size
    shard_end = shard_begin + shard_size
    in_shard = (input_ids >= shard_begin) & (input_ids < shard_end)
    rows = torch.where(in_shard, input_ids - shard_begin, torch.zeros_like(input_ids))
    base_local = embed_table[rows.long()]
    base_local.masked_fill_(~in_shard.unsqueeze(-1), 0)
    base = base_local.clone()
    dist.all_reduce(base, op=dist.ReduceOp.SUM, group=process_group)

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
    concat_local = torch.cat(pieces, dim=-1)
    concat = concat_local.clone()
    dist.all_reduce(concat, op=dist.ReduceOp.SUM, group=process_group)
    emb_new = torch.nn.functional.linear(concat, oe_proj_weight, bias=None)
    return ((base + emb_new) / 2.0).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", default="/tmp/welmv4_fused_oe_dump")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    dump = torch.load(
        Path(args.dump_dir, f"rank{rank}.pt"), map_location="cpu", weights_only=False
    )

    input_ids = dump["input_ids"].to(device)
    prefix_rows = dump["prefix_rows"]
    prefixes = [
        torch.tensor(row, dtype=torch.int64, device=device) for row in prefix_rows
    ]

    embed_full = dump["embed_weight_full"].to(device)
    org_vocab_size = int(dump["embed_org_vocab_size"])
    s = org_vocab_size // world_size
    embed_table = embed_full[:s].contiguous()  # rank already loaded its own shard
    del embed_full

    oe_vocab_sizes = list(dump["oe_vocab_sizes"])
    oe_grams = list(dump["oe_grams"])
    vocab_size = int(dump["vocab_size"])

    hash_tables = []
    for branch_idx, oev in enumerate(oe_vocab_sizes):
        full = dump["oe_embed_weights"][branch_idx].to(device)
        s_h = oev // world_size
        hash_tables.append(full[:s_h].contiguous())
        del full

    proj_weight = dump["proj_weight"].to(device).contiguous()
    saved_ref = dump["ref"].to(device)
    saved_fused = dump["fused"].to(device)

    if rank == 0:
        print(
            f"# replay rank{rank}: input_ids={input_ids.tolist()} "
            f"prefix_rows={prefix_rows} world_size={world_size}",
            flush=True,
        )

    # 1. Replay live unfused.
    live_ref = _unfused_reference(
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
        process_group=dist.group.WORLD,
    )

    # 2. Replay live mk fused.
    from mk.kernels import (
        FusedDecodeNGramHashEmbeddingGemmAllReduceParams,
        NGramSpec,
        fused_decode_ngram_hash_embedding_gemm_all_reduce,
    )

    ngram_spec = tuple(
        NGramSpec(int(g), int(v)) for g, v in zip(oe_grams, oe_vocab_sizes)
    )
    per_token_prefixes = [
        [int(prefix_rows[lag][token]) for lag in range(len(prefix_rows))]
        for token in range(input_ids.numel())
    ]
    params = FusedDecodeNGramHashEmbeddingGemmAllReduceParams(
        input_ids=input_ids,
        prefixes=per_token_prefixes,
        input_embedding_table=embed_table,
        hash_embedding_tables=hash_tables,
        weight=proj_weight,
        process_group=dist.group.WORLD,
        vocab_size=vocab_size,
        ngram_spec=ngram_spec,
        input_hidden_size=embed_table.shape[1],
        hash_hidden_size=hash_tables[0].shape[1],
        world_size=world_size,
    )
    live_fused = fused_decode_ngram_hash_embedding_gemm_all_reduce(params)

    def _diff(a, b):
        d = (a.float() - b.float()).abs()
        return float(d.max().item()), float(b.float().abs().max().item())

    sref_lref, _ = _diff(saved_ref, live_ref)
    sfused_lfused, _ = _diff(saved_fused, live_fused)
    sref_sfused, max_ref = _diff(saved_ref, saved_fused)
    lref_lfused, _ = _diff(live_ref, live_fused)
    sref_lfused, _ = _diff(saved_ref, live_fused)

    if rank == 0:
        print(
            f"# saved_ref vs saved_fused (sglang in-server): "
            f"max_abs_diff={sref_sfused:.6g} max_ref={max_ref:.6g} "
            f"rel={sref_sfused / max_ref if max_ref > 0 else 0.0:.6g}",
            flush=True,
        )
        print(
            f"# saved_ref vs live_ref (replay determinism of unfused): "
            f"max_abs_diff={sref_lref:.6g}",
            flush=True,
        )
        print(
            f"# saved_fused vs live_fused (replay determinism of mk): "
            f"max_abs_diff={sfused_lfused:.6g}",
            flush=True,
        )
        print(
            f"# live_ref vs live_fused (replay equivalence): "
            f"max_abs_diff={lref_lfused:.6g}",
            flush=True,
        )
        print(
            f"# saved_ref vs live_fused (apples-to-apples): "
            f"max_abs_diff={sref_lfused:.6g}",
            flush=True,
        )
        print(f"saved_ref[0,:8]={saved_ref[0, :8].float().tolist()}", flush=True)
        print(f"saved_fused[0,:8]={saved_fused[0, :8].float().tolist()}", flush=True)
        print(f"live_fused[0,:8]={live_fused[0, :8].float().tolist()}", flush=True)
        print(f"live_ref[0,:8]={live_ref[0, :8].float().tolist()}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
