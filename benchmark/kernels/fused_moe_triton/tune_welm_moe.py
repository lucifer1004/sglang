"""Tune the fused-MoE Triton kernel for WeLM-family models.

Single-node multi-GPU, no Ray. For the requested model, shards BATCH_SIZES
across N workers (LPT scheduling) and merges per-GPU partials into the
canonical E=..,N=..,device_name=..json that SGLang looks up at startup.

Examples:
    python tune_welm_moe.py --model welm_600b                  # 8 GPUs, tp=32, ep=16
    python tune_welm_moe.py --model welm_600b --tp-size 16 --ep-size 8
    python tune_welm_moe.py --model welm_600b --num-gpus 4
    python tune_welm_moe.py --model welm_600b --max-configs 3  # smoke (~30s)

Add a model: append one entry to MODELS with its intrinsic fields only
(parallelism is a deployment concern, not a model property). The tuner derives
    E = total_experts / ep_size
    N = 2 * moe_intermediate_size / (tp_size / ep_size)
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """Intrinsic MoE architecture fields from the HF config."""

    total_experts: int
    topk: int
    hidden_size: int
    moe_intermediate_size: int
    torch_dtype: str  # "bfloat16" / "float16" / "float32"
    block_shape: Optional[list[int]] = None  # e.g. [128, 128] for block-wise quant


MODELS: dict[str, ModelSpec] = {
    "welm_600b": ModelSpec(
        total_experts=512,
        topk=10,
        hidden_size=4096,
        moe_intermediate_size=1024,
        torch_dtype="bfloat16",
    ),
}


# Batch sizes that cover both decode (tiny) and prefill (thousands).
BATCH_SIZES: list[int] = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
]


# ---------------------------------------------------------------------------
# Kernel-level parameters (what benchmark_config actually needs)
# ---------------------------------------------------------------------------
@dataclass
class KernelParams:
    num_experts: int  # E per rank = total_experts / ep_size
    topk: int
    hidden_size: int
    shard_intermediate_size: int  # 2 * moe_intermediate_size / (tp / ep)
    dtype: Any  # torch.dtype (Any to avoid a top-level torch import)
    block_shape: Optional[list[int]]


def derive_kernel_params(spec: ModelSpec, tp_size: int, ep_size: int) -> KernelParams:
    import torch

    if tp_size % ep_size != 0:
        raise ValueError(f"tp_size={tp_size} must be divisible by ep_size={ep_size}")
    moe_tp = tp_size // ep_size
    if spec.moe_intermediate_size % moe_tp != 0:
        raise ValueError(
            f"moe_intermediate_size={spec.moe_intermediate_size} not divisible by "
            f"tp/ep={moe_tp}"
        )
    return KernelParams(
        num_experts=spec.total_experts // ep_size,
        topk=spec.topk,
        hidden_size=spec.hidden_size,
        shard_intermediate_size=2 * spec.moe_intermediate_size // moe_tp,
        dtype=getattr(torch, spec.torch_dtype),
        block_shape=spec.block_shape,
    )


# ---------------------------------------------------------------------------
# Per-GPU worker: runs in a spawned child pinned to one device
# ---------------------------------------------------------------------------
def tune_on_gpu(
    gpu_index: int,
    model_name: str,
    tp_size: int,
    ep_size: int,
    batch_sizes: list[int],
    partial_output_path: str,
    log_path: str,
    max_configs: Optional[int],
) -> None:
    # Must happen BEFORE importing torch. Safe because this module doesn't
    # import torch at top-level, so the spawned child's re-import is clean.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.stdout = sys.stderr = open(log_path, "w", buffering=1)

    import torch
    from common_utils import get_config_filename, get_configs_compute_bound, sort_config
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from tuning_fused_moe_triton import benchmark_config

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)

    kernel_params = derive_kernel_params(MODELS[model_name], tp_size, ep_size)

    candidate_configs = get_configs_compute_bound()
    if kernel_params.block_shape is not None:
        block_k = kernel_params.block_shape[1]
        candidate_configs = [
            c for c in candidate_configs if block_k % c["BLOCK_SIZE_K"] == 0
        ]
    if max_configs:
        candidate_configs = candidate_configs[:max_configs]

    def log(message: str) -> None:
        print(f"[{model_name} gpu#{gpu_index}] {message}", flush=True)

    log(f"start; batches={batch_sizes}, {len(candidate_configs)} configs each")

    best_per_batch: dict[int, dict] = {}
    for batch_size in batch_sizes:
        best_config = None
        best_latency_us = float("inf")
        start_time = time.time()
        for candidate in candidate_configs:
            try:
                latency_us = benchmark_config(
                    candidate,
                    batch_size,
                    kernel_params.num_experts,
                    kernel_params.shard_intermediate_size,
                    kernel_params.hidden_size,
                    kernel_params.topk,
                    kernel_params.dtype,
                    False,
                    False,
                    False,
                    False,
                    kernel_params.block_shape,
                    num_iters=10,
                )
            except Exception:
                continue
            if latency_us < best_latency_us:
                best_config, best_latency_us = candidate, latency_us

        if best_config is None:
            log(f"batch={batch_size}: no working config")
            continue
        best_per_batch[batch_size] = sort_config(best_config)
        log(
            f"batch={batch_size} done in {time.time() - start_time:.1f}s "
            f"best={best_latency_us:.1f}us"
        )

    output_filename = get_config_filename(
        kernel_params.num_experts,
        kernel_params.shard_intermediate_size,
        kernel_params.hidden_size,
        kernel_params.topk,
        kernel_params.dtype,
        False,
        False,
        False,
        False,
        kernel_params.block_shape,
    )
    Path(partial_output_path).write_text(
        json.dumps({"filename": output_filename, "configs": best_per_batch})
    )


# ---------------------------------------------------------------------------
# Load balancing: Longest-Processing-Time greedy
# ---------------------------------------------------------------------------
def _estimate_cost_seconds(batch_size: int) -> float:
    """Rough wall-time per tuned config; only used to balance shards."""
    if batch_size <= 128:
        return 0.8
    if batch_size <= 1024:
        return 1.2
    return 1.8


def _lpt_schedule(batch_sizes: list[int], num_shards: int) -> list[list[int]]:
    """Assign heaviest batch to currently least-loaded shard; repeat."""
    shards: list[list[int]] = [[] for _ in range(num_shards)]
    shard_loads: list[float] = [0.0] * num_shards
    for batch_size in sorted(batch_sizes, reverse=True):
        target = min(range(num_shards), key=lambda i: shard_loads[i])
        shards[target].append(batch_size)
        shard_loads[target] += _estimate_cost_seconds(batch_size)
    return shards


# ---------------------------------------------------------------------------
# Driver: fork N per-GPU workers, wait, merge partials
# ---------------------------------------------------------------------------
def tune_model(
    model_name: str,
    tp_size: int,
    ep_size: int,
    num_gpus: int,
    max_configs: Optional[int],
    workdir: Path,
) -> None:
    num_workers = max(1, min(num_gpus, len(BATCH_SIZES)))
    batches_per_gpu = _lpt_schedule(BATCH_SIZES, num_workers)
    output_dir = workdir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== Tuning {model_name} (tp={tp_size}, ep={ep_size}) on {num_workers} GPUs ==="
    )
    for gpu_index, batches in enumerate(batches_per_gpu):
        print(f"  gpu#{gpu_index}: {batches}")

    spawn_ctx = mp.get_context("spawn")
    processes: list[tuple[int, mp.Process, Path]] = []
    start_time = time.time()
    for gpu_index, batches in enumerate(batches_per_gpu):
        partial_path = output_dir / f"partial_gpu{gpu_index}.json"
        log_path = output_dir / f"gpu{gpu_index}.log"
        proc = spawn_ctx.Process(
            target=tune_on_gpu,
            args=(
                gpu_index,
                model_name,
                tp_size,
                ep_size,
                batches,
                str(partial_path),
                str(log_path),
                max_configs,
            ),
        )
        proc.start()
        processes.append((gpu_index, proc, partial_path))

    for gpu_index, proc, _ in processes:
        proc.join()
        if proc.exitcode:
            print(f"  gpu#{gpu_index} failed (rc={proc.exitcode})", file=sys.stderr)

    merged_configs: dict[int, dict] = {}
    output_filename: Optional[str] = None
    for _, _, partial_path in processes:
        if not partial_path.exists():
            continue
        partial_data = json.loads(partial_path.read_text())
        output_filename = output_filename or partial_data["filename"]
        merged_configs.update(
            {int(batch): cfg for batch, cfg in partial_data["configs"].items()}
        )

    if not merged_configs:
        print(f"ERROR: no results for {model_name}", file=sys.stderr)
        return

    Path(output_filename).write_text(
        json.dumps(
            {str(batch): merged_configs[batch] for batch in sorted(merged_configs)},
            indent=4,
        )
        + "\n"
    )
    elapsed = time.time() - start_time
    print(
        f"=== {model_name} done in {elapsed:.1f}s -> "
        f"{Path(output_filename).resolve()} ==="
    )
    print(f"    Covered batches: {sorted(merged_configs)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--tp-size", type=int, default=32)
    parser.add_argument("--ep-size", type=int, default=16)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Cap search space (for smoke tests).",
    )
    parser.add_argument(
        "--workdir",
        default="./.tune_welm_moe_out",
        help="Per-GPU partials + logs (default: ./.tune_welm_moe_out).",
    )
    args = parser.parse_args()

    try:
        visible_gpus = len(subprocess.check_output(["nvidia-smi", "-L"]).splitlines())
    except Exception:
        visible_gpus = args.num_gpus
    num_gpus = min(args.num_gpus, visible_gpus)
    if num_gpus < args.num_gpus:
        print(
            f"WARN: clamping --num-gpus {args.num_gpus} -> {num_gpus}",
            file=sys.stderr,
        )

    workdir = Path(args.workdir).absolute()
    workdir.mkdir(parents=True, exist_ok=True)
    tune_model(
        model_name=args.model,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        num_gpus=num_gpus,
        max_configs=args.max_configs,
        workdir=workdir,
    )


if __name__ == "__main__":
    main()
