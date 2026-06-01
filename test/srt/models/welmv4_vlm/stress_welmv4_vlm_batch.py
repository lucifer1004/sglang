#!/usr/bin/env python3
"""Concurrent VLM request stress test for a running WeLMV4 SGLang server.

This script sends many concurrent OpenAI-compatible chat completion requests,
cycling through local .jpg/.jpeg/.png files. Concurrent single requests are the
normal way to test whether the server batches VLM requests correctly.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import mimetypes
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
MODEL = os.environ.get("MODEL", "welmv4")
IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "."))
TOTAL_REQUESTS = int(os.environ.get("TOTAL_REQUESTS", "24"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TIMEOUT = float(os.environ.get("TIMEOUT", "300"))
OUTPUT_JSONL = Path(os.environ.get("OUTPUT_JSONL", "welmv4_vlm_stress_outputs.jsonl"))
PROMPT = os.environ.get(
    "PROMPT",
    "请识别图片中的主要内容。如果图片中包含文字，请按原始顺序输出文字。",
)


def discover_images(image_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    if not images:
        raise FileNotFoundError(
            f"No .jpg/.jpeg/.png files found in {image_dir.resolve()}"
        )
    return images


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(image_path: Path) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }


def send_one(request_id: int, image_path: Path) -> dict[str, Any]:
    endpoint = f"{BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = build_payload(image_path)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read().decode("utf-8")
        elapsed = time.perf_counter() - start
        result = json.loads(body)
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        finish_reason = result.get("choices", [{}])[0].get("finish_reason")
        return {
            "request_id": request_id,
            "image": str(image_path),
            "ok": bool(content.strip()),
            "latency_s": elapsed,
            "finish_reason": finish_reason,
            "content": content,
            "usage": result.get("usage"),
            "raw": result,
        }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        return {
            "request_id": request_id,
            "image": str(image_path),
            "ok": False,
            "latency_s": elapsed,
            "error": f"HTTP {exc.code}",
            "body": exc.read().decode("utf-8", errors="replace"),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "request_id": request_id,
            "image": str(image_path),
            "ok": False,
            "latency_s": elapsed,
            "error": repr(exc),
        }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


def main() -> int:
    images = discover_images(IMAGE_DIR)
    assignments = [images[i % len(images)] for i in range(TOTAL_REQUESTS)]
    print("============================================================")
    print(" WeLMV4 VLM concurrent request stress test")
    print("============================================================")
    print(f"  endpoint       : {BASE_URL.rstrip('/')}/v1/chat/completions")
    print(f"  model          : {MODEL}")
    print(f"  image_dir      : {IMAGE_DIR.resolve()}")
    print(f"  images         : {', '.join(path.name for path in images)}")
    print(f"  total_requests : {TOTAL_REQUESTS}")
    print(f"  concurrency    : {CONCURRENCY}")
    print(f"  max_tokens     : {MAX_TOKENS}")
    print(f"  output_jsonl   : {OUTPUT_JSONL}")
    print("============================================================")

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_to_id = {
            executor.submit(send_one, request_id, image_path): request_id
            for request_id, image_path in enumerate(assignments)
        }
        for future in concurrent.futures.as_completed(future_to_id):
            result = future.result()
            results.append(result)
            status = "OK" if result["ok"] else "FAIL"
            content = (result.get("content") or "").replace("\n", " ")
            print(
                f"[{status}] id={result['request_id']:03d} "
                f"image={Path(result['image']).name} "
                f"latency={result['latency_s']:.2f}s "
                f"chars={len(result.get('content') or '')} "
                f"{content[:80]}"
            )

    total_elapsed = time.perf_counter() - started
    results.sort(key=lambda item: item["request_id"])
    with OUTPUT_JSONL.open("w", encoding="utf-8") as fout:
        for result in results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    ok_results = [result for result in results if result["ok"]]
    failed_results = [result for result in results if not result["ok"]]
    latencies = [result["latency_s"] for result in ok_results]

    print("\n========================== Summary =========================")
    print(f"success              : {len(ok_results)}/{len(results)}")
    print(f"failed               : {len(failed_results)}")
    print(f"total_elapsed_s      : {total_elapsed:.2f}")
    print(f"throughput_req_per_s : {len(results) / total_elapsed:.2f}")
    if latencies:
        print(f"latency_avg_s        : {statistics.mean(latencies):.2f}")
        print(f"latency_p50_s        : {percentile(latencies, 0.50):.2f}")
        print(f"latency_p90_s        : {percentile(latencies, 0.90):.2f}")
        print(f"latency_p99_s        : {percentile(latencies, 0.99):.2f}")

    if failed_results:
        print("\nFailures:")
        for result in failed_results[:10]:
            print(
                f"  id={result['request_id']} image={result['image']} "
                f"error={result.get('error')} body={result.get('body', '')[:200]}"
            )
        return 1

    print(f"\nDetailed outputs written to {OUTPUT_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
