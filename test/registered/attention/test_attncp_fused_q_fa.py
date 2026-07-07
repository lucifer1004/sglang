import math
import unittest

import torch

from sglang.srt.layers.attention.attncp_fused_ops import (
    attncp_cp2_fused_q_fa_decode,
    attncp_cp2_fused_q_fa_max_splits,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=20, stage="stage-b", runner_config="1-gpu-large")


def _make_inputs(
    *,
    batch_size: int,
    max_seq_len: int,
    page_size: int,
    local_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    cp_rank: int = 0,
    dtype: torch.dtype = torch.bfloat16,
):
    torch.manual_seed(1234 + page_size * 17 + max_seq_len + cp_rank)
    device = "cuda"
    max_pages = (max_seq_len + page_size - 1) // page_size
    total_pages = batch_size * max_pages + 1

    q_local = torch.randn(
        batch_size, local_q_heads, head_dim, device=device, dtype=dtype
    )
    q_peer = torch.randn(
        batch_size, local_q_heads, head_dim, device=device, dtype=dtype
    )
    key_cache = torch.randn(
        total_pages, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    # Page 0 is the dummy slot used by paged attention; keep real pages non-zero.
    key_cache[0].zero_()
    value_cache[0].zero_()
    page_table = (
        torch.arange(1, total_pages, device=device, dtype=torch.int32)
        .view(batch_size, max_pages)
        .contiguous()
    )
    # Include partial last pages for page_size > 1.
    low = max(1, max_seq_len - max(64, page_size * 3))
    cache_seqlens = torch.randint(
        low, max_seq_len + 1, (batch_size,), device=device, dtype=torch.int32
    )

    full_q_heads = local_q_heads * 2
    out_o = torch.empty(
        batch_size, full_q_heads, head_dim, device=device, dtype=dtype
    )
    out_lse = torch.empty(batch_size, full_q_heads, device=device, dtype=torch.float32)
    return q_local, q_peer, key_cache, value_cache, page_table, cache_seqlens, out_o, out_lse


def _reference_attncp_cp2_fused_q_fa(
    q_local,
    q_peer,
    key_cache,
    value_cache,
    page_table,
    cache_seqlens,
    *,
    cp_rank: int,
    softmax_scale: float,
    softcap: float,
    window_left: int,
    sinks: torch.Tensor | None,
    page_size: int,
):
    batch_size, local_q_heads, head_dim = q_local.shape
    full_q_heads = local_q_heads * 2
    num_kv_heads = key_cache.shape[2]
    qh_per_kv = full_q_heads // num_kv_heads
    out = torch.empty(
        batch_size, full_q_heads, head_dim, device=q_local.device, dtype=torch.float32
    )
    lse = torch.empty(
        batch_size, full_q_heads, device=q_local.device, dtype=torch.float32
    )

    if cp_rank == 0:
        q_full = torch.cat([q_local, q_peer], dim=1)
    else:
        q_full = torch.cat([q_peer, q_local], dim=1)
    q_full = q_full.to(torch.float32)

    for b in range(batch_size):
        seq_len = int(cache_seqlens[b].item())
        start = max(seq_len - int(window_left) - 1, 0) if window_left >= 0 else 0
        positions = torch.arange(start, seq_len, device=q_local.device)
        pages = page_table[b, positions // page_size].long()
        offsets = positions % page_size
        k = key_cache[pages, offsets].to(torch.float32)
        v = value_cache[pages, offsets].to(torch.float32)
        k = k.repeat_interleave(qh_per_kv, dim=1)
        v = v.repeat_interleave(qh_per_kv, dim=1)

        scores = torch.einsum("hd,lhd->hl", q_full[b], k) * float(softmax_scale)
        if softcap and softcap > 0:
            scores = (2.0 * torch.sigmoid(2.0 * scores / softcap) - 1.0) * softcap

        if sinks is None:
            probs = torch.softmax(scores, dim=-1)
            out[b] = torch.einsum("hl,lhd->hd", probs, v)
            lse[b] = torch.logsumexp(scores, dim=-1)
            continue

        sink = sinks.to(torch.float32)
        max_scores = scores.max(dim=-1).values
        m = torch.maximum(max_scores, sink)
        exp_scores = torch.exp(scores - m[:, None])
        exp_sink = torch.where(
            torch.isfinite(sink),
            torch.exp(sink - m),
            torch.zeros_like(sink),
        )
        denom = exp_scores.sum(dim=-1) + exp_sink
        out[b] = torch.einsum("hl,lhd->hd", exp_scores / denom[:, None], v)
        lse[b] = torch.log(denom) + m

    return out, lse


def _reference_attncp_cp2_masked_pages(
    q_local,
    q_peer,
    key_cache,
    value_cache,
    page_table,
    page_start_offsets,
    page_token_counts,
    *,
    cp_rank: int,
    softmax_scale: float,
    sinks: torch.Tensor,
):
    batch_size, local_q_heads, head_dim = q_local.shape
    full_q_heads = local_q_heads * 2
    num_kv_heads = key_cache.shape[2]
    qh_per_kv = full_q_heads // num_kv_heads
    if cp_rank == 0:
        q_full = torch.cat([q_local, q_peer], dim=1)
    else:
        q_full = torch.cat([q_peer, q_local], dim=1)
    q_full = q_full.float()

    out = torch.empty(
        batch_size, full_q_heads, head_dim, device=q_local.device, dtype=torch.float32
    )
    lse = torch.empty(
        batch_size, full_q_heads, device=q_local.device, dtype=torch.float32
    )
    for b in range(batch_size):
        page_ids = []
        offsets = []
        for col in range(page_table.shape[1]):
            count = int(page_token_counts[b, col].item())
            if count <= 0:
                continue
            page = int(page_table[b, col].item())
            start = int(page_start_offsets[b, col].item())
            for token_offset in range(count):
                page_ids.append(page)
                offsets.append(start + token_offset)
        pages = torch.tensor(page_ids, device=q_local.device, dtype=torch.long)
        in_page_offsets = torch.tensor(
            offsets, device=q_local.device, dtype=torch.long
        )
        k = key_cache[pages, in_page_offsets].float()
        v = value_cache[pages, in_page_offsets].float()
        k = k.repeat_interleave(qh_per_kv, dim=1)
        v = v.repeat_interleave(qh_per_kv, dim=1)
        scores = torch.einsum("hd,lhd->hl", q_full[b], k) * float(softmax_scale)

        sink = sinks.float()
        max_score = torch.maximum(scores.max(dim=-1).values, sink)
        exp_scores = torch.exp(scores - max_score[:, None])
        exp_sink = torch.where(
            torch.isfinite(sink),
            torch.exp(sink - max_score),
            torch.zeros_like(sink),
        )
        denom = exp_scores.sum(dim=-1) + exp_sink
        out[b] = torch.einsum("hl,lhd->hd", exp_scores / denom[:, None], v)
        lse[b] = torch.log(denom) + max_score
    return out, lse


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestAttnCPFusedQFA(CustomTestCase):
    def _run_correctness_case(
        self,
        *,
        page_size: int,
        max_seq_len: int,
        cp_rank: int,
        window_left: int = -1,
        use_sinks: bool = False,
        use_splits: bool = False,
    ):
        dtype = torch.bfloat16
        (
            q_local,
            q_peer,
            key_cache,
            value_cache,
            page_table,
            cache_seqlens,
            out_o,
            out_lse,
        ) = _make_inputs(
            batch_size=3,
            max_seq_len=max_seq_len,
            page_size=page_size,
            cp_rank=cp_rank,
            dtype=dtype,
        )
        full_q_heads = q_local.shape[1] * 2
        head_dim = q_local.shape[2]
        softmax_scale = 1.0 / math.sqrt(head_dim)
        softcap = 30.0
        sinks = None
        if use_sinks:
            sinks = torch.randn(full_q_heads, device="cuda", dtype=dtype) * 0.2
            # Include a disabled sink to match the production empty-sink behavior.
            sinks[-1] = -float("inf")

        split_o = split_lse = None
        max_splits = 1
        if use_splits:
            max_splits = attncp_cp2_fused_q_fa_max_splits(max_seq_len)
            split_o = torch.empty(
                max_splits,
                out_o.shape[0],
                full_q_heads,
                head_dim,
                device="cuda",
                dtype=dtype,
            )
            split_lse = torch.empty(
                max_splits,
                out_o.shape[0],
                full_q_heads,
                device="cuda",
                dtype=torch.float32,
            )

        attncp_cp2_fused_q_fa_decode(
            q_local,
            q_peer,
            key_cache,
            value_cache,
            page_table,
            cache_seqlens,
            out_o,
            out_lse,
            cp_rank=cp_rank,
            softmax_scale=softmax_scale,
            softcap=softcap,
            window_left=window_left,
            sinks=sinks,
            page_size=page_size,
            split_o=split_o,
            split_lse=split_lse,
            max_splits=max_splits,
        )
        ref_o, ref_lse = _reference_attncp_cp2_fused_q_fa(
            q_local,
            q_peer,
            key_cache,
            value_cache,
            page_table,
            cache_seqlens,
            cp_rank=cp_rank,
            softmax_scale=softmax_scale,
            softcap=softcap,
            window_left=window_left,
            sinks=sinks,
            page_size=page_size,
        )

        torch.testing.assert_close(
            out_o.float(), ref_o, atol=2.5e-2, rtol=2.5e-2
        )
        torch.testing.assert_close(out_lse, ref_lse, atol=2.5e-2, rtol=2.5e-2)

    def test_page_size_1_and_16_correctness(self):
        for cp_rank in (0, 1):
            self._run_correctness_case(
                page_size=1,
                max_seq_len=513,
                cp_rank=cp_rank,
                window_left=-1,
                use_sinks=False,
            )
            self._run_correctness_case(
                page_size=16,
                max_seq_len=521,
                cp_rank=cp_rank,
                window_left=-1,
                use_sinks=False,
            )

    def test_page_size_16_window_sink_and_split_correctness(self):
        self._run_correctness_case(
            page_size=16,
            max_seq_len=4609,
            cp_rank=1,
            window_left=511,
            use_sinks=True,
            use_splits=True,
        )

    def test_page_size_16_masked_pages_correctness(self):
        torch.manual_seed(777)
        dtype = torch.bfloat16
        batch_size = 2
        local_q_heads = 6
        full_q_heads = local_q_heads * 2
        num_kv_heads = 1
        head_dim = 128
        page_size = 16
        page_cap = 515
        total_pages = batch_size * page_cap + 17
        q_local = torch.randn(
            batch_size, local_q_heads, head_dim, device="cuda", dtype=dtype
        )
        q_peer = torch.randn_like(q_local)
        key_cache = torch.randn(
            total_pages,
            page_size,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        value_cache = torch.randn_like(key_cache)
        key_cache[0].zero_()
        value_cache[0].zero_()
        page_table = (
            torch.arange(
                1,
                1 + batch_size * page_cap,
                device="cuda",
                dtype=torch.int32,
            )
            .view(batch_size, page_cap)
            .contiguous()
        )
        page_start_offsets = torch.zeros_like(page_table)
        page_token_counts = torch.full_like(page_table, page_size)
        page_start_offsets[0, 0] = 5
        page_token_counts[0, 0] = 11
        page_token_counts[0, -1] = 7
        page_start_offsets[1, 0] = 13
        page_token_counts[1, 0] = 3
        page_token_counts[1, -1] = 1
        page_token_counts[:, 123] = 0
        page_table[:, 123] = 0
        cache_seqlens = page_token_counts.sum(dim=1).to(torch.int32)
        sinks = torch.randn(full_q_heads, device="cuda", dtype=dtype)
        out_o = torch.empty(
            batch_size, full_q_heads, head_dim, device="cuda", dtype=dtype
        )
        out_lse = torch.empty(
            batch_size, full_q_heads, device="cuda", dtype=torch.float32
        )
        # Force a constrained split workspace. Without page-aligned split sizes,
        # this masked SWA case would enter the unmasked split indexing branch.
        max_splits = min(2, attncp_cp2_fused_q_fa_max_splits(page_cap * page_size))
        split_o = torch.empty(
            max_splits,
            batch_size,
            full_q_heads,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        split_lse = torch.empty(
            max_splits,
            batch_size,
            full_q_heads,
            device="cuda",
            dtype=torch.float32,
        )
        softmax_scale = 1.0 / math.sqrt(head_dim)

        attncp_cp2_fused_q_fa_decode(
            q_local,
            q_peer,
            key_cache,
            value_cache,
            page_table,
            cache_seqlens,
            out_o,
            out_lse,
            cp_rank=0,
            softmax_scale=softmax_scale,
            sinks=sinks,
            page_size=page_size,
            page_start_offsets=page_start_offsets,
            page_token_counts=page_token_counts,
            split_o=split_o,
            split_lse=split_lse,
            max_splits=max_splits,
        )
        ref_o, ref_lse = _reference_attncp_cp2_masked_pages(
            q_local,
            q_peer,
            key_cache,
            value_cache,
            page_table,
            page_start_offsets,
            page_token_counts,
            cp_rank=0,
            softmax_scale=softmax_scale,
            sinks=sinks,
        )
        torch.testing.assert_close(
            out_o.float(), ref_o, atol=2.5e-2, rtol=2.5e-2
        )
        torch.testing.assert_close(out_lse, ref_lse, atol=2.5e-2, rtol=2.5e-2)


if __name__ == "__main__":
    unittest.main()
