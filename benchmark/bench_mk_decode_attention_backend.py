import argparse
import statistics
import time
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.mk_decode_attention_backend import (
    MkDecodeAttentionBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class FakeKVPool:
    def __init__(self, layer_num: int, size: int, page_size: int, device: str):
        self.page_size = page_size
        self.dtype = torch.bfloat16
        self.k = [
            torch.randn(
                size + page_size, 1, 256, device=device, dtype=torch.bfloat16
            )
            for _ in range(layer_num)
        ]
        self.v = [
            torch.randn(
                size + page_size, 1, 256, device=device, dtype=torch.bfloat16
            )
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id: int):
        return self.k[layer_id]

    def get_value_buffer(self, layer_id: int):
        return self.v[layer_id]

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args):
        self.k[layer.layer_id][loc] = cache_k.view(-1, 1, 256)
        self.v[layer.layer_id][loc] = cache_v.view(-1, 1, 256)

    def translate_loc_from_full_to_swa(self, loc):
        return loc


def percentile(values, q):
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def build_case(batch_size: int, seq_len: int, window_left: int, layer_num: int):
    page_size = 16
    device = "cuda"
    max_pages = (seq_len + page_size - 1) // page_size
    size = batch_size * max_pages * page_size
    req_to_token = torch.zeros(
        (batch_size + 1, max_pages * page_size), dtype=torch.int32, device=device
    )
    for row in range(1, batch_size + 1):
        base = (row - 1) * max_pages * page_size
        req_to_token[row, :seq_len] = torch.arange(
            base, base + seq_len, dtype=torch.int32, device=device
        )
    kv_pool = FakeKVPool(layer_num, size, page_size, device)
    layerwise = [window_left if window_left > 0 else -1] * layer_num
    runner = SimpleNamespace(
        device=device,
        server_args=SimpleNamespace(page_size=page_size),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=kv_pool,
        is_hybrid_swa=window_left > 0,
        sliding_window_size=window_left if window_left > 0 else None,
        tp_size=4,
        model_config=SimpleNamespace(
            num_attention_heads=24,
            head_dim=256,
            context_len=seq_len,
            num_hidden_layers=layer_num,
            hf_config=SimpleNamespace(sliding_window_size_layerwise=layerwise),
            get_num_kv_heads=lambda tp: max(1, 4 // tp),
        ),
    )
    req_pool_indices = torch.arange(
        1, batch_size + 1, dtype=torch.int64, device=device
    )
    seq_lens_cpu = torch.full((batch_size,), seq_len, dtype=torch.int32)
    out_cache_loc = torch.tensor(
        [row * max_pages * page_size + seq_len - 1 for row in range(batch_size)],
        dtype=torch.int64,
        device=device,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens_cpu=seq_lens_cpu,
        req_pool_indices=req_pool_indices,
        forward_mode=ForwardMode.DECODE,
        token_to_kv_pool=kv_pool,
        out_cache_loc=out_cache_loc,
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=window_left,
        tp_q_head_num=6,
        tp_k_head_num=1,
        qk_head_dim=256,
        v_head_dim=256,
        logit_cap=0.0,
        k_scale=None,
        v_scale=None,
    )
    q = torch.randn(batch_size, 6 * 256, device=device, dtype=torch.bfloat16)
    k = kv_pool.k[0][out_cache_loc].reshape(batch_size, 256).clone()
    v = kv_pool.v[0][out_cache_loc].reshape(batch_size, 256).clone()
    sinks = torch.zeros((6,), device=device, dtype=torch.bfloat16)
    return runner, forward_batch, layer, q, k, v, sinks


def warmup_gpu_with_gemm(device: str, iters: int):
    if iters <= 0:
        return
    a = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    b = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--window-left", type=int, default=-1)
    parser.add_argument("--layer-num", type=int, default=32)
    parser.add_argument("--gemm-warmup", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    warmup_gpu_with_gemm("cuda", args.gemm_warmup)

    runner, forward_batch, layer, q, k, v, sinks = build_case(
        args.batch_size,
        args.seq_len,
        args.window_left,
        args.layer_num,
    )
    backend = MkDecodeAttentionBackend(runner)

    for _ in range(args.warmup):
        backend.init_forward_metadata(forward_batch)
        backend.forward_decode(q, k, v, layer, forward_batch, sinks=sinks)
    torch.cuda.synchronize()

    init_us = []
    run_us = []
    e2e_us = []
    for _ in range(args.iters):
        start = time.perf_counter()
        backend.init_forward_metadata(forward_batch)
        torch.cuda.synchronize()
        mid = time.perf_counter()
        backend.forward_decode(q, k, v, layer, forward_batch, sinks=sinks)
        torch.cuda.synchronize()
        end = time.perf_counter()
        init_us.append((mid - start) * 1e6)
        run_us.append((end - mid) * 1e6)
        e2e_us.append((end - start) * 1e6)

    print(
        "shape "
        f"bs={args.batch_size} seq_len={args.seq_len} "
        f"window_left={args.window_left} layers={args.layer_num}"
    )
    for name, values in (("init", init_us), ("run", run_us), ("e2e", e2e_us)):
        print(
            f"{name}_us p50={statistics.median(values):.1f} "
            f"p90={percentile(values, 0.90):.1f} "
            f"p99={percentile(values, 0.99):.1f}"
        )


if __name__ == "__main__":
    main()
