"""End-to-end smoke test for the WeLMV4 VLM two-mode interface defined by
``refine_vlm_interface.md``.

Run AFTER the SGLang server is up (e.g. via ``./server.sh``). By default the
script targets ``http://127.0.0.1:30000`` and uses ``cat.jpg`` and
``char.png`` next to this file as the test images.

Cases:
- Mode 1 (POST /generate): raw prompt + image_data as a SINGLE data-URL string
- Mode 1 (POST /generate): raw prompt + image_data as a LIST of data-URLs (multi-image)
- Mode 2 (POST /v1/chat/completions): OpenAI format with ONE image
- Mode 2 (POST /v1/chat/completions): OpenAI format with TWO images
- Mode 2 (POST /v1/chat/completions): pure-text dialogue (no image)

Environment overrides:
- ``SGLANG_URL``: server base URL (default ``http://127.0.0.1:30000``)
- ``MODEL``: served model name (default ``welmv4``)
- ``IMG_DIR``: directory holding test images (default: this script's directory)
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import requests

BASE = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000")
MODEL = os.environ.get("MODEL", "welmv4")
IMG_DIR = Path(os.environ.get("IMG_DIR", Path(__file__).resolve().parent))
IMG_CAT = IMG_DIR / "cat.jpg"
IMG_CHAR = IMG_DIR / "char.png"

VISION = "<|vision_start|><|image_pad|><|vision_end|>"


def _data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _hr(title: str) -> None:
    print("\n" + "=" * 8 + " " + title + " " + "=" * 8)


def _post(path: str, payload: dict, timeout: float = 120) -> dict:
    url = BASE + path
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        print(f"!! HTTP {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
    return r.json()


def test_generate_single_image() -> str:
    _hr("Mode 1 / generate single-image (image_data as string)")
    # Mode 1 contract: caller supplies the fully-formed prompt with ALL
    # special tokens (server applies NO chat template). Use Qwen3-VL ChatML
    # formatting to match what the model was trained on.
    prompt = (
        "<|im_start|>system\n你是一个有用的助手。<|im_end|>\n"
        f"<|im_start|>user\n{VISION}\n请用一句话描述这张图片。<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    payload = {
        "text": prompt,
        "image_data": _data_url(IMG_CAT),
        "sampling_params": {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": ["<|im_end|>"],
        },
        "stream": False,
    }
    out = _post("/generate", payload)
    text = out.get("text", "")
    print("Reply:", text)
    assert text.strip(), "expected non-empty text"
    return text


def test_generate_multi_image() -> str:
    _hr("Mode 1 / generate multi-image (image_data as list)")
    prompt = (
        "<|im_start|>system\n你是一个有用的助手。<|im_end|>\n"
        f"<|im_start|>user\n{VISION}{VISION}\n请用一句话比较这两张图片的内容。<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    payload = {
        "text": prompt,
        "image_data": [_data_url(IMG_CAT), _data_url(IMG_CHAR)],
        "sampling_params": {
            "max_new_tokens": 192,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": ["<|im_end|>"],
        },
        "stream": False,
    }
    out = _post("/generate", payload)
    text = out.get("text", "")
    print("Reply:", text)
    assert text.strip(), "expected non-empty text"
    return text


def test_chat_single_image() -> str:
    _hr("Mode 2 / v1/chat/completions single-image")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个有用的助手。"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(IMG_CAT)}},
                    {"type": "text", "text": "请用一句话描述这张图片。"},
                ],
            },
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    out = _post("/v1/chat/completions", payload)
    msg = out["choices"][0]["message"]["content"]
    print("Reply:", msg)
    assert msg.strip(), "expected non-empty assistant text"
    return msg


def test_chat_multi_image() -> str:
    _hr("Mode 2 / v1/chat/completions multi-image")
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(IMG_CAT)}},
                    {"type": "image_url", "image_url": {"url": _data_url(IMG_CHAR)}},
                    {"type": "text", "text": "请用一句话比较这两张图片的区别。"},
                ],
            }
        ],
        "max_tokens": 192,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    out = _post("/v1/chat/completions", payload)
    msg = out["choices"][0]["message"]["content"]
    print("Reply:", msg)
    assert msg.strip(), "expected non-empty assistant text"
    return msg


def test_chat_text_only() -> str:
    _hr("Mode 2 / v1/chat/completions text-only")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": "用一句话介绍一下你自己。"},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    out = _post("/v1/chat/completions", payload)
    msg = out["choices"][0]["message"]["content"]
    print("Reply:", msg)
    assert msg.strip(), "expected non-empty assistant text"
    return msg


def main() -> None:
    for image in (IMG_CAT, IMG_CHAR):
        if not image.exists():
            print(f"!! missing test image: {image}", file=sys.stderr)
            sys.exit(2)

    cases = [
        ("generate-single", test_generate_single_image),
        ("generate-multi", test_generate_multi_image),
        ("chat-single", test_chat_single_image),
        ("chat-multi", test_chat_multi_image),
        ("chat-text-only", test_chat_text_only),
    ]
    failures: list[tuple[str, str]] = []
    for name, fn in cases:
        try:
            fn()
        except Exception as exc:
            failures.append((name, repr(exc)))
            print(f"!! {name} FAILED: {exc}")
    print("\n========= SUMMARY =========")
    if failures:
        for n, err in failures:
            print(f"FAIL {n}: {err}")
        sys.exit(1)
    print(f"ALL {len(cases)} END-TO-END CASES PASSED")


if __name__ == "__main__":
    main()
