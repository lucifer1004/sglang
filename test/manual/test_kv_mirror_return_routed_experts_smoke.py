"""
Manual smoke test for return_routed_experts with WeLM KV mirror optimization.

Example:
PYTHONPATH=python .venv/bin/python test/manual/test_kv_mirror_return_routed_experts_smoke.py \
  --model-path /path/to/welm/moe/model
"""

import argparse
import concurrent.futures
import os
import random
import sys
import urllib.error
from typing import List, Optional, Tuple

import numpy as np
import pandas
import requests

from sglang.srt.layers.moe.routed_experts_capturer import (
    extract_routed_experts_from_meta_info,
)
from sglang.test.simple_eval_common import format_multichoice_question
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)


def _infer_routed_experts_shape(
    model_path: str,
    num_layers: Optional[int],
    top_k: Optional[int],
) -> Tuple[int, int]:
    if num_layers is not None and top_k is not None:
        return num_layers, top_k

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        inferred_num_layers = getattr(text_config, "num_hidden_layers", None)
        inferred_top_k = getattr(text_config, "num_experts_per_tok", None)
    except Exception as exc:
        raise RuntimeError(
            "Unable to infer routed experts shape from model config. "
            "Pass --num-layers and --top-k explicitly."
        ) from exc

    num_layers = num_layers or inferred_num_layers
    top_k = top_k or inferred_top_k
    if num_layers is None or top_k is None:
        raise RuntimeError(
            "Model config is missing num_hidden_layers or num_experts_per_tok. "
            "Pass --num-layers and --top-k explicitly."
        )
    return int(num_layers), int(top_k)


def _decode_routed_experts(response_json, model_path, num_layers, top_k):
    routed_experts = extract_routed_experts_from_meta_info(response_json)
    if isinstance(routed_experts, dict):
        values = routed_experts["values"].reshape(routed_experts["shape"])
        valid_mask = routed_experts["valid_mask"].reshape(
            routed_experts["valid_mask_shape"]
        )
        return values, valid_mask.astype(bool)

    num_layers, top_k = _infer_routed_experts_shape(model_path, num_layers, top_k)
    if routed_experts.size % (num_layers * top_k) != 0:
        raise ValueError(
            f"Cannot reshape routed experts of size {routed_experts.size} into "
            f"[-1, {num_layers}, {top_k}]"
        )
    values = routed_experts.reshape(-1, num_layers, top_k)
    valid_mask = np.ones(values.shape[:2], dtype=bool)
    return values, valid_mask


def _print_layer_router_ids(values: np.ndarray, valid_mask: np.ndarray, layer_id: int):
    print(f"Layer {layer_id} router ids:")
    for token_idx, router_ids in enumerate(values[:, layer_id, :].tolist()):
        valid = bool(valid_mask[token_idx, layer_id])
        print(f"  token[{token_idx}] valid={valid}: {router_ids}")


def _post_generate(
    base_url: str,
    prompt: str,
    max_new_tokens: int,
    request_timeout: float,
):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "return_routed_experts": True,
    }
    response = requests.post(
        f"{base_url}/generate", json=payload, timeout=request_timeout
    )
    response.raise_for_status()
    response_json = response.json()
    if "error" in response_json:
        raise RuntimeError(response_json["error"])
    return response_json


def _load_mmlu_prompts(num_examples: int) -> List[str]:
    filename = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
    if os.getenv("SGLANG_MMLU_OFFLINE", "0") == "1":
        examples = None
    else:
        try:
            df = pandas.read_csv(filename, storage_options={"timeout": 30})
            examples = [row.to_dict() for _, row in df.iterrows()]
        except (OSError, TimeoutError, urllib.error.URLError):
            examples = None
    if examples is None:
        from datasets import load_dataset

        dataset = load_dataset(
            "cais/mmlu",
            "all",
            split="test",
            download_mode="reuse_dataset_if_exists",
        )
        limit = num_examples if num_examples else len(dataset)
        examples = [dict(dataset[i]) for i in range(min(limit, len(dataset)))]
    if num_examples:
        examples = random.Random(0).sample(examples, num_examples)
    prompts = []
    for row in examples:
        if "Question" not in row and "question" in row:
            choices = row.get("choices") or []
            row = {
                "Question": row["question"],
                "A": choices[0] if len(choices) > 0 else "",
                "B": choices[1] if len(choices) > 1 else "",
                "C": choices[2] if len(choices) > 2 else "",
                "D": choices[3] if len(choices) > 3 else "",
            }
        prompts.append(format_multichoice_question(row))
    return prompts


def _run_mmlu_concurrency_smoke(
    base_url: str,
    model_path: str,
    num_layers: Optional[int],
    top_k: Optional[int],
    num_examples: int,
    concurrency: int,
    max_new_tokens: int,
    request_timeout: float,
):
    prompts = _load_mmlu_prompts(num_examples)

    def run_one(index_and_prompt):
        index, prompt = index_and_prompt
        response_json = _post_generate(
            base_url, prompt, max_new_tokens, request_timeout
        )
        values, valid_mask = _decode_routed_experts(
            response_json, model_path, num_layers, top_k
        )
        if values.ndim != 3 or valid_mask.ndim != 2:
            raise RuntimeError(
                f"Unexpected routed experts rank: {values.shape=} {valid_mask.shape=}"
            )
        if values.shape[:2] != valid_mask.shape:
            raise RuntimeError(
                f"Routed experts values and valid_mask mismatch: "
                f"{values.shape=} {valid_mask.shape=}"
            )
        layer0_valid = int(valid_mask[:, 0].sum())
        last_layer_valid = int(valid_mask[:, -1].sum())
        return {
            "index": index,
            "shape": tuple(values.shape),
            "layer0_valid": layer0_valid,
            "last_layer_valid": last_layer_valid,
            "generated_text": response_json.get("generated_text", ""),
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_one, item) for item in enumerate(prompts)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["index"])
    missing = [
        item
        for item in results
        if item["layer0_valid"] == 0 or item["last_layer_valid"] == 0
    ]
    unique_shapes = sorted({item["shape"] for item in results})
    layer0_counts = [item["layer0_valid"] for item in results]
    last_counts = [item["last_layer_valid"] for item in results]
    print(
        "mmlu_concurrency_summary: "
        f"ok={len(results)}/{len(prompts)} concurrency={concurrency} "
        f"max_new_tokens={max_new_tokens}"
    )
    print(f"mmlu_routed_experts_shapes: {unique_shapes}")
    print(
        "mmlu_layer_valid_counts: "
        f"layer0_min={min(layer0_counts)} layer0_max={max(layer0_counts)} "
        f"last_min={min(last_counts)} last_max={max(last_counts)}"
    )
    for item in results[:3]:
        print(
            f"mmlu_sample[{item['index']}]: "
            f"shape={list(item['shape'])} "
            f"layer0_valid={item['layer0_valid']} "
            f"last_layer_valid={item['last_layer_valid']} "
            f"generated_text={item['generated_text']!r}"
        )
    if missing:
        preview = ", ".join(
            f"idx={item['index']} shape={list(item['shape'])} "
            f"layer0={item['layer0_valid']} last={item['last_layer_valid']}"
            for item in missing[:20]
        )
        raise RuntimeError(
            f"MMLU concurrency routed experts missing valid layers: "
            f"{len(missing)}/{len(results)} failures; {preview}"
        )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("SGLANG_KV_MIRROR_SMOKE_MODEL"),
        help="Model path or HF id. Can also be set with SGLANG_KV_MIRROR_SMOKE_MODEL.",
    )
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    )
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--mmlu-num-examples", type=int, default=0)
    parser.add_argument("--mmlu-concurrency", type=int, default=32)
    parser.add_argument("--mmlu-max-new-tokens", type=int, default=5)
    parser.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Extra server args after '--', for example: -- --tp 2",
    )
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or SGLANG_KV_MIRROR_SMOKE_MODEL is required")
    return args


def main():
    args = _parse_args()
    port = args.port or find_available_port(30000)
    base_url = f"http://{args.host}:{port}"

    server_args = [
        "--enable-return-routed-experts",
        "--enable-welm-kv-mirror-opt",
        *args.server_args,
    ]
    if server_args[2:3] == ["--"]:
        server_args.pop(2)

    env = os.environ.copy()
    venv_bin = os.path.join(sys.prefix, "bin")
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    process = popen_launch_server(
        args.model_path,
        base_url,
        timeout=args.timeout,
        other_args=server_args,
        env=env,
    )
    try:
        response_json = _post_generate(
            base_url, args.prompt, args.max_new_tokens, args.request_timeout
        )
        values, valid_mask = _decode_routed_experts(
            response_json, args.model_path, args.num_layers, args.top_k
        )
        print(f"generated_text: {response_json.get('generated_text')}")
        print(f"routed_experts_shape: {list(values.shape)}")
        _print_layer_router_ids(values, valid_mask, 0)
        _print_layer_router_ids(values, valid_mask, values.shape[1] - 1)

        if args.mmlu_num_examples > 0:
            _run_mmlu_concurrency_smoke(
                base_url=base_url,
                model_path=args.model_path,
                num_layers=args.num_layers,
                top_k=args.top_k,
                num_examples=args.mmlu_num_examples,
                concurrency=args.mmlu_concurrency,
                max_new_tokens=args.mmlu_max_new_tokens,
                request_timeout=args.request_timeout,
            )
    finally:
        kill_process_tree(process.pid)


if __name__ == "__main__":
    main()
