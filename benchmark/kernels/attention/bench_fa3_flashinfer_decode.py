import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata

from sglang.srt.layers.attention.flashinfer_backend import should_use_tensor_core


DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def load_model_shape(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    dtype_name = str(config.get("torch_dtype", "bfloat16")).lower()
    dtype = DTYPE_MAP.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported torch_dtype in {config_path}: {dtype_name}")

    num_q_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config.get("num_key_value_heads", num_q_heads))
    head_dim = int(config.get("head_dim", config["hidden_size"] // num_q_heads))

    return {
        "config_path": str(config_path),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "torch_dtype": dtype_name,
        "dtype": dtype,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": config.get("max_position_embeddings"),
        "sliding_window": config.get("sliding_window"),
    }


def local_heads(total_q_heads: int, total_kv_heads: int, tp_size: int) -> tuple[int, int]:
    if total_q_heads % tp_size != 0:
        raise ValueError(f"num_attention_heads={total_q_heads} is not divisible by TP={tp_size}")
    return total_q_heads // tp_size, max(1, total_kv_heads // tp_size)


def elapsed_us(fn, iters: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def quantiles(values: list[float]) -> dict[str, float]:
    if len(values) == 1:
        return {"min_us": values[0], "median_us": values[0], "max_us": values[0]}
    return {
        "min_us": min(values),
        "median_us": statistics.median(values),
        "max_us": max(values),
    }


def make_tensors(
    *,
    batch_size: int,
    kv_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    page_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    num_pages = (kv_len + page_size - 1) // page_size
    if num_pages * page_size != kv_len:
        raise ValueError("This benchmark expects kv_len to be divisible by page_size.")
    q = torch.randn(batch_size, q_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, dtype=dtype, device=device)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(1, -1)
    if batch_size != 1:
        page_table = page_table.repeat(batch_size, 1)
    cache_seqlens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_table": page_table,
        "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q,
    }


def bench_fa3(
    tensors: dict[str, torch.Tensor],
    *,
    batch_size: int,
    kv_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    page_size: int,
    num_splits: int,
    warmup: int,
    iters: int,
    trials: int,
) -> dict[str, Any]:
    softmax_scale = 1.0 / math.sqrt(head_dim)
    scheduler_metadata = get_scheduler_metadata(
        batch_size,
        1,
        kv_len,
        q_heads,
        kv_heads,
        head_dim,
        tensors["cache_seqlens"],
        qkv_dtype=dtype,
        page_size=page_size,
        causal=True,
        num_splits=num_splits,
    )

    def run():
        return flash_attn_with_kvcache(
            q=tensors["q"],
            k_cache=tensors["k_cache"],
            v_cache=tensors["v_cache"],
            page_table=tensors["page_table"],
            cache_seqlens=tensors["cache_seqlens"],
            cu_seqlens_q=tensors["cu_seqlens_q"],
            max_seqlen_q=1,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(-1, -1),
            num_splits=num_splits,
            ver=3,
            scheduler_metadata=scheduler_metadata,
        )

    for _ in range(warmup):
        run()
    times = [elapsed_us(run, iters) for _ in range(trials)]
    return {
        "backend": "fa3",
        "num_splits": num_splits,
        **quantiles(times),
        "trials_us": times,
    }


def bench_flashinfer(
    tensors: dict[str, torch.Tensor],
    *,
    batch_size: int,
    kv_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    page_size: int,
    warmup: int,
    iters: int,
    trials: int,
) -> dict[str, Any]:
    softmax_scale = 1.0 / math.sqrt(head_dim)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=tensors["q"].device)
    use_tensor_cores = should_use_tensor_core(dtype, q_heads, kv_heads)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=use_tensor_cores,
    )
    indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=tensors["q"].device) * kv_len
    indices = torch.arange(batch_size * kv_len, dtype=torch.int32, device=tensors["q"].device)
    last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device=tensors["q"].device)
    wrapper.begin_forward(
        indptr,
        indices,
        last_page_len,
        q_heads,
        kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=dtype,
        q_data_type=dtype,
        sm_scale=softmax_scale,
    )

    def run():
        return wrapper.forward(
            tensors["q"],
            (tensors["k_cache"], tensors["v_cache"]),
            sm_scale=softmax_scale,
        )

    for _ in range(warmup):
        run()
    times = [elapsed_us(run, iters) for _ in range(trials)]
    return {
        "backend": "flashinfer",
        "use_tensor_cores": use_tensor_cores,
        **quantiles(times),
        "trials_us": times,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = load_model_shape(Path(args.model_config).expanduser())
    dtype = model["dtype"]
    kv_lens = [2**i for i in range(args.kv_start_power, args.kv_end_power + 1)]
    tp_sizes = list(range(args.tp_start, args.tp_end + 1))

    meta = {
        "created_unix": time.time(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "measurement": {
            "batch_size": args.batch_size,
            "decode_tokens_per_request": 1,
            "page_size": args.page_size,
            "warmup": args.warmup,
            "iters": args.iters,
            "trials": args.trials,
            "timing": "CUDA events around backend op loop; metadata/plan allocation excluded",
            "fa3_op": "sgl_kernel.flash_attn.flash_attn_with_kvcache",
            "flashinfer_op": "flashinfer.BatchDecodeWithPagedKVCacheWrapper.forward",
        },
        "model": {k: v for k, v in model.items() if k != "dtype"},
    }
    rows: list[dict[str, Any]] = []

    for tp_size in tp_sizes:
        q_heads, kv_heads = local_heads(
            model["num_attention_heads"],
            model["num_key_value_heads"],
            tp_size,
        )
        for kv_len in kv_lens:
            tensors = make_tensors(
                batch_size=args.batch_size,
                kv_len=kv_len,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=model["head_dim"],
                dtype=dtype,
                page_size=args.page_size,
                device=device,
            )
            common = {
                "tp_size": tp_size,
                "kv_len": kv_len,
                "q_heads_per_rank": q_heads,
                "kv_heads_per_rank": kv_heads,
                "head_dim": model["head_dim"],
                "dtype": model["torch_dtype"],
            }
            for bench in (bench_fa3, bench_flashinfer):
                result = bench(
                    tensors,
                    batch_size=args.batch_size,
                    kv_len=kv_len,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=model["head_dim"],
                    dtype=dtype,
                    page_size=args.page_size,
                    num_splits=args.fa3_num_splits,
                    warmup=args.warmup,
                    iters=args.iters,
                    trials=args.trials,
                ) if bench is bench_fa3 else bench(
                    tensors,
                    batch_size=args.batch_size,
                    kv_len=kv_len,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=model["head_dim"],
                    dtype=dtype,
                    page_size=args.page_size,
                    warmup=args.warmup,
                    iters=args.iters,
                    trials=args.trials,
                )
                row = {**common, **result}
                rows.append(row)
                print(
                    f"tp={tp_size} kv={kv_len:6d} backend={row['backend']:<10} "
                    f"median={row['median_us']:.3f} us min={row['min_us']:.3f} us max={row['max_us']:.3f} us",
                    flush=True,
                )
            del tensors
            torch.cuda.empty_cache()

    return {"meta": meta, "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="~/models/config.json")
    parser.add_argument("--output", default="docs/microbenchs/attn/fa3_flashinfer_decode_results.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--tp-start", type=int, default=1)
    parser.add_argument("--tp-end", type=int, default=4)
    parser.add_argument("--kv-start-power", type=int, default=10)
    parser.add_argument("--kv-end-power", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--fa3-num-splits", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
