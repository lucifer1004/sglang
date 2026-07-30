"""Focused SM120 FlashAttention-4 regression tests."""

import math

import pytest
import torch

from sglang.kernels.ops.attention.fa4_sm120.runtime import (
    Sm120ForwardHost,
)
from sglang.kernels.ops.attention.flash_attention_v4 import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=50,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-small",
)

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)):
    pytest.skip(
        "SM120 FlashAttention-4 test requires CUDA SM 12.0.",
        allow_module_level=True,
    )


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    window_left: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 causal attention-with-sink output and LSE."""
    seq = q.shape[0]
    scale = q.shape[-1] ** -0.5
    scores = (
        torch.einsum(
            "qhd,kd->hqk",
            q.float(),
            k[:, 0].float(),
        )
        * scale
    )
    q_idx = torch.arange(seq, device=q.device)[:, None]
    kv_idx = torch.arange(seq, device=q.device)[None, :]
    mask = kv_idx <= q_idx
    if window_left is not None:
        mask &= kv_idx >= q_idx - window_left
    scores.masked_fill_(~mask[None], -torch.inf)

    row_max = torch.maximum(scores.amax(dim=-1), sinks.float()[:, None])
    weights = torch.exp(scores - row_max[:, :, None])
    denominator = weights.sum(dim=-1) + torch.exp(sinks.float()[:, None] - row_max)
    output = (
        torch.einsum(
            "hqk,kd->qhd",
            weights,
            v[:, 0].float(),
        )
        / denominator.T[:, :, None]
    )
    lse = torch.log(denominator) + row_max
    return output, lse


@pytest.mark.parametrize("window_left", [None, 250])
def test_sm120_varlen_mqa_hd256_learnable_sink(window_left):
    """Cover the WeLMv4 Q6/KV1/hd256 global and local prefill shapes."""
    torch.manual_seed(1234)
    seq, num_q_heads, head_dim = 512, 6, 256
    q = torch.randn(seq, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seq, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
    window_size = (None, None) if window_left is None else (window_left, 0)
    out_ref, lse_ref = _reference(q, k, v, sinks, window_left)

    outputs = []
    for pack_gqa in (False, True, None):
        out, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq,
            max_seqlen_k=seq,
            softmax_scale=head_dim**-0.5,
            causal=True,
            window_size=window_size,
            sinks=sinks,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )
        output_error = (out.float() - out_ref).abs()
        lse_error = (lse - lse_ref).abs()
        assert output_error.max().item() < 1e-2
        assert output_error.mean().item() < 5e-4
        assert lse_error.max().item() < 5e-5
        outputs.append(out)

    torch.testing.assert_close(outputs[0], outputs[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(outputs[1], outputs[2], atol=0.0, rtol=0.0)


def test_sm120_varlen_padding_ctas_are_inert_across_tile_specializations():
    """Padding CTAs must not consume stale SMEM or write another batch's output."""
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_q_heads, head_dim = 6, 256

    def make_inputs(lengths, seed):
        torch.manual_seed(seed)
        total = sum(lengths)
        q = torch.randn(
            total,
            num_q_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn(total, 1, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
        cuts = [0]
        for length in lengths:
            cuts.append(cuts[-1] + length)
        cu_seqlens = torch.tensor(cuts, device="cuda", dtype=torch.int32)
        return q, k, v, sinks, cu_seqlens

    def run(inputs, lengths, pack_gqa):
        q, k, v, sinks, cu_seqlens = inputs
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(lengths),
            softmax_scale=head_dim**-0.5,
            causal=True,
            sinks=sinks,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )

    # Exercise the three SM-normalized selector regions before the padding
    # case. This is the order that exposed stale shared-memory state.
    rows_per_sm = (58, 84, 100)
    for seed, ratio in enumerate(rows_per_sm):
        seq = math.ceil(ratio * num_sms / num_q_heads)
        inputs = make_inputs([seq], seed)
        for pack_gqa in (False, True, False):
            run(inputs, [seq], pack_gqa)
        torch.cuda.synchronize()

    # At 64 query rows/SM, two batches select M64 while the conservative
    # varlen grid contains padding CTAs.
    seq = math.ceil(32 * num_sms / num_q_heads)
    lengths = [seq, seq]
    inputs = make_inputs(lengths, 10)
    q, k, v, sinks, _ = inputs
    references = [
        _reference(
            q[start : start + seq],
            k[start : start + seq],
            v[start : start + seq],
            sinks,
            None,
        )
        for start in (0, seq)
    ]
    out_ref = torch.cat([reference[0] for reference in references], dim=0)
    lse_ref = torch.cat([reference[1] for reference in references], dim=1)

    for pack_gqa in (False, True, None):
        out, lse = run(inputs, lengths, pack_gqa)
        output_error = (out.float() - out_ref).abs()
        lse_error = (lse - lse_ref).abs()
        assert output_error.max().item() < 1e-2
        assert output_error.mean().item() < 5e-4
        assert lse_error.max().item() < 5e-5


@pytest.mark.parametrize("window_left", [None, 192])
def test_sm120_paged_decode_ragged_splits_are_cache_order_independent(window_left):
    """Ragged SplitKV must remain correct across uniform/ragged cache reuse."""
    torch.manual_seed(1234)
    batch_size, num_q_heads, head_dim = 4, 6, 256
    max_seqlen, page_size = 1024, 64
    pages_per_request = max_seqlen // page_size
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        batch_size * pages_per_request,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(
        batch_size * pages_per_request,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, pages_per_request)
    cu_seqlens_q = torch.arange(
        batch_size + 1,
        device="cuda",
        dtype=torch.int32,
    )
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)

    def run(lengths):
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=torch.tensor(
                lengths,
                device="cuda",
                dtype=torch.int32,
            ),
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            window_size=((None, None) if window_left is None else (window_left, 0)),
            num_splits=4,
            pack_gqa=True,
            sinks=sinks,
        )

    uniform_lengths = [max_seqlen] * batch_size
    # Exercise both sides of every 64-token tile boundary. N-distributed QK
    # has a different accumulator-column layout, so its mask must not use the
    # ordinary QK path's R2P column mapping.
    ragged_lengths = [1000, 513, 257, 63]
    uniform_first = run(uniform_lengths)
    ragged_first = run(ragged_lengths)
    ragged_second = run(ragged_lengths)
    uniform_second = run(uniform_lengths)

    torch.testing.assert_close(ragged_first, ragged_second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(uniform_first, uniform_second, atol=0.0, rtol=0.0)

    reference = torch.empty_like(ragged_first, dtype=torch.float32)
    scale = head_dim**-0.5
    for batch_idx, length in enumerate(ragged_lengths):
        pages = page_table[batch_idx]
        start = 0 if window_left is None else max(0, length - 1 - window_left)
        k = k_cache.index_select(0, pages).flatten(0, 1)[start:length, 0].float()
        v = v_cache.index_select(0, pages).flatten(0, 1)[start:length, 0].float()
        scores = q[batch_idx].float() @ k.T * scale
        row_max = torch.maximum(scores.amax(dim=-1), sinks.float())
        weights = torch.exp(scores - row_max[:, None])
        denominator = weights.sum(dim=-1) + torch.exp(sinks.float() - row_max)
        reference[batch_idx] = weights @ v / denominator[:, None]

    error = (ragged_first.float() - reference).abs()
    assert error.max().item() < 1e-2
    assert error.mean().item() < 5e-4


def test_sm120_paged_decode_transpose_is_cache_order_independent():
    """Gather and page-TMA transpose must not share a compiled specialization."""
    torch.manual_seed(20260729)
    batch_size, num_q_heads, head_dim = 1, 6, 256
    max_seqlen, page_size = 2048, 64
    num_pages = max_seqlen // page_size
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        num_pages,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.randperm(
        num_pages,
        device="cuda",
        dtype=torch.int64,
    ).to(
        torch.int32
    )[None]
    cache_seqlens = torch.tensor(
        [max_seqlen],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)

    def run(num_splits):
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            num_splits=num_splits,
            pack_gqa=True,
            sinks=sinks,
        )

    # S32 owns one KV tile per CTA and selects gather. S16 owns two and
    # selects the full-transpose page-TMA class. Alternate both compile/cache
    # entries in the order that previously exposed an illegal access.
    normal_first = run(32)
    transpose_after = run(16)
    transpose_repeat = run(16)
    normal_after = run(32)
    torch.cuda.synchronize()

    torch.testing.assert_close(normal_first, normal_after, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        transpose_after,
        transpose_repeat,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        normal_first,
        transpose_after,
        atol=2e-3,
        rtol=0.0,
    )

    pages = page_table[0]
    k = k_cache.index_select(0, pages).flatten(0, 1)[:, 0].float()
    v = v_cache.index_select(0, pages).flatten(0, 1)[:, 0].float()
    scores = q[0].float() @ k.T * head_dim**-0.5
    row_max = torch.maximum(scores.amax(dim=-1), sinks.float())
    weights = torch.exp(scores - row_max[:, None])
    denominator = weights.sum(dim=-1) + torch.exp(sinks.float() - row_max)
    reference = weights @ v / denominator[:, None]
    error = (transpose_after[0].float() - reference).abs()
    assert error.max().item() < 1e-2
    assert error.mean().item() < 5e-4


@pytest.mark.parametrize(
    ("num_q_heads", "pack_gqa", "expected_transpose", "expected_split_qk"),
    [
        pytest.param(6, True, True, False, id="transpose"),
        pytest.param(16, True, False, True, id="split-qk"),
        pytest.param(6, False, False, False, id="single-qk"),
    ],
)
def test_sm120_paged_decode_graph_pdl_is_correct_and_eager_reusable(
    monkeypatch,
    num_q_heads,
    pack_gqa,
    expected_transpose,
    expected_split_qk,
):
    """Every SplitKV dataflow must safely launch its captured combine early."""
    torch.manual_seed(20260730)
    batch_size, head_dim = 1, 256
    max_seqlen, page_size = 2048, 64
    num_pages = max_seqlen // page_size
    captured_plans = []
    original_resolve_plan = Sm120ForwardHost.resolve_plan

    def recording_resolve_plan(**kwargs):
        plan = original_resolve_plan(**kwargs)
        if kwargs["is_stream_capturing"]:
            captured_plans.append(plan)
        return plan

    monkeypatch.setattr(
        Sm120ForwardHost,
        "resolve_plan",
        staticmethod(recording_resolve_plan),
    )
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        num_pages,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.randperm(
        num_pages,
        device="cuda",
        dtype=torch.int64,
    ).to(
        torch.int32
    )[None]
    cache_seqlens = torch.tensor(
        [max_seqlen],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)

    def run():
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            num_splits=16,
            pack_gqa=pack_gqa,
            sinks=sinks,
        )

    eager_before = run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    eager_after = run()
    torch.cuda.synchronize()

    torch.testing.assert_close(eager_before, eager_after, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        graph_output,
        eager_before,
        atol=2e-3,
        rtol=0.0,
    )
    assert captured_plans
    assert all(plan.num_splits > 1 for plan in captured_plans)
    assert all(plan.launch_split_combine_early for plan in captured_plans)
    assert all(plan.transpose_qk_pv is expected_transpose for plan in captured_plans)
    assert all(plan.split_qk_n is expected_split_qk for plan in captured_plans)
