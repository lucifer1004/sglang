#!/usr/bin/env python3
"""Encoder disaggregation test for WeLMV4 VLM.

Validates that vision encoder and LLM can be deployed on separate GPUs:
- Encoder server (TP=1, GPU 4): loads only vision encoder (~4GB), runs encoding
- Language server (TP=4, GPUs 0-3): loads LLM only (~210GB), receives embeddings

Uses zmq_to_scheduler transfer backend (simplest, no RDMA needed).

Requirements:
- 5+ GPUs available
- Model checkpoint at /root/welmv4_vlm2 (override via WELMV4_MODEL env)

Usage:
    python test_epd_welmv4_vlm.py
"""

from __future__ import annotations

import base64
import os
import sys
import time
import unittest
from pathlib import Path

import openai

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

# ---- Config ----
MODEL_PATH = os.environ.get("WELMV4_MODEL", "/root/welmv4_vlm2")
BASE_HOST = "127.0.0.1"
BASE_PORT = int(os.environ.get("WELMV4_EPD_BASE_PORT", "31000"))
ENCODE_PORT = str(BASE_PORT + 300)
LANGUAGE_PORT = str(BASE_PORT)

ENCODE_URL = f"http://{BASE_HOST}:{ENCODE_PORT}"
LANGUAGE_URL = f"http://{BASE_HOST}:{LANGUAGE_PORT}"

ENCODER_TRANSFER_BACKEND = os.environ.get(
    "WELMV4_EPD_TRANSFER_BACKEND", "zmq_to_scheduler"
)

# GPU assignment: encoder TP=1 on GPU 4 (vision encoder is small ~4GB),
# language TP=4 on GPUs 0-3 (full LLM ~210GB)
ENCODE_TP = os.environ.get("WELMV4_EPD_ENCODE_TP", "1")
ENCODE_BASE_GPU = os.environ.get("WELMV4_EPD_ENCODE_BASE_GPU", "4")
LANGUAGE_TP = os.environ.get("WELMV4_EPD_LANGUAGE_TP", "4")
LANGUAGE_BASE_GPU = os.environ.get("WELMV4_EPD_LANGUAGE_BASE_GPU", "0")

# Mooncake-specific config (only used when ENCODER_TRANSFER_BACKEND=mooncake).
# Mooncake transfers embeddings via RDMA, so we must specify the IB device.
# RDMA also requires `memlock` ulimit raised — launch with
# `sudo prlimit --memlock=unlimited <cmd>` if your container caps it.
MOONCAKE_IB_DEVICE = os.environ.get("WELMV4_EPD_MOONCAKE_IB_DEVICE", "mlx5_bond_1")

# Test images
IMG_DIR = Path(__file__).resolve().parent
IMG_CAT = IMG_DIR / "cat_small.jpg"
# Optional second image for the multi-image test. Falls back to the same
# small cat (a deliberate "same image twice" multi-image case) when
# char.png is not present in the repo. Avoid falling back to the full-size
# cat.jpg because two large images would push the prompt past the model's
# context window.
IMG_CHAR = IMG_DIR / "char.png"
if not IMG_CHAR.exists():
    IMG_CHAR = IMG_CAT

SERVER_TIMEOUT = int(os.environ.get("WELMV4_EPD_TIMEOUT", "1800"))


def _data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class TestEncoderDisaggWeLMV4VLM(unittest.TestCase):
    """Encoder disaggregation test: encoder-only + language-only servers."""

    process_encode = None
    process_language = None

    @classmethod
    def setUpClass(cls):
        # Verify test images exist
        for img in (IMG_CAT, IMG_CHAR):
            if not img.exists():
                raise unittest.SkipTest(f"Test image not found: {img}")

        # Verify model exists
        if not Path(MODEL_PATH).exists():
            raise unittest.SkipTest(f"Model not found: {MODEL_PATH}")

        print(f"\n{'='*60}")
        print("WeLMV4 VLM Encoder Disaggregation Test")
        print(f"  Model: {MODEL_PATH}")
        print(f"  Encoder: {ENCODE_URL} (TP={ENCODE_TP}, base_gpu={ENCODE_BASE_GPU})")
        print(
            f"  Language: {LANGUAGE_URL} (TP={LANGUAGE_TP}, base_gpu={LANGUAGE_BASE_GPU})"
        )
        print(f"  Backend: {ENCODER_TRANSFER_BACKEND}")
        print(f"{'='*60}\n")

        # Start encoder first (fast: only loads vision encoder ~4GB)
        print("Starting encoder server...")
        cls.start_encode()
        wait_for_http_ready(
            ENCODE_URL + "/health",
            timeout=SERVER_TIMEOUT,
            process=cls.process_encode,
        )
        print("Encoder server ready!")

        # Start language server (slower: loads full LLM ~210GB)
        print("Starting language server...")
        cls.start_language()
        wait_for_http_ready(
            LANGUAGE_URL + "/health",
            timeout=SERVER_TIMEOUT,
            process=cls.process_language,
        )
        print("Language server ready!")

    @classmethod
    def _mooncake_args(cls):
        """Extra CLI args required when using the mooncake transfer backend.

        Mooncake uses RDMA for embedding transfer, so we must tell it which
        InfiniBand device to bind to. Returns an empty list for non-mooncake
        backends so the args are unchanged. Skips the test when mooncake is
        requested but ``WELMV4_EPD_MOONCAKE_IB_DEVICE`` was not explicitly
        set, since the default ``mlx5_bond_1`` is host-specific and would
        produce a confusing RDMA-init failure on machines without it.
        """
        if ENCODER_TRANSFER_BACKEND != "mooncake":
            return []
        if not os.environ.get("WELMV4_EPD_MOONCAKE_IB_DEVICE"):
            raise unittest.SkipTest(
                "WELMV4_EPD_MOONCAKE_IB_DEVICE is not set; skipping mooncake "
                "EPD test (the default 'mlx5_bond_1' is host-specific). Set "
                "the env var to your IB device name to enable this run."
            )
        return ["--mooncake-ib-device", MOONCAKE_IB_DEVICE]

    @classmethod
    def start_encode(cls):
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            ENCODER_TRANSFER_BACKEND,
            "--mm-enable-dp-encoder",
            "--tp",
            ENCODE_TP,
            "--base-gpu-id",
            ENCODE_BASE_GPU,
            "--port",
            ENCODE_PORT,
        ]
        encode_args += cls._mooncake_args()
        cls.process_encode = popen_launch_server(
            MODEL_PATH,
            base_url=ENCODE_URL,
            timeout=SERVER_TIMEOUT,
            other_args=encode_args,
        )

    @classmethod
    def start_language(cls):
        language_args = [
            "--trust-remote-code",
            "--language-only",
            "--encoder-urls",
            ENCODE_URL,
            "--encoder-transfer-backend",
            ENCODER_TRANSFER_BACKEND,
            "--tp",
            LANGUAGE_TP,
            "--base-gpu-id",
            LANGUAGE_BASE_GPU,
            "--port",
            LANGUAGE_PORT,
        ]
        language_args += cls._mooncake_args()
        cls.process_language = popen_launch_server(
            MODEL_PATH,
            base_url=LANGUAGE_URL,
            timeout=SERVER_TIMEOUT,
            other_args=language_args,
        )

    @classmethod
    def tearDownClass(cls):
        for name, process in [
            ("Language", cls.process_language),
            ("Encode", cls.process_encode),
        ]:
            if process:
                try:
                    print(f"Killing {name} server (pid={process.pid})...")
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing {name}: {e}")
        time.sleep(3)

    def _client(self):
        return openai.Client(api_key="sk-test", base_url=f"{LANGUAGE_URL}/v1")

    # ---- Test Cases ----

    def test_01_single_image_chat(self):
        """Single image chat completion via encoder disaggregation."""
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(IMG_CAT)},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in one sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=128,
        )
        text = response.choices[0].message.content
        print(f"\n[EPD] Single image response: {text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text.strip()), 0, "Expected non-empty response")

    def test_02_multi_image_chat(self):
        """Multi-image chat completion via encoder disaggregation."""
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(IMG_CAT)},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(IMG_CHAR)},
                        },
                        {
                            "type": "text",
                            "text": "Compare these two images briefly.",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=192,
        )
        text = response.choices[0].message.content
        print(f"\n[EPD] Multi-image response: {text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text.strip()), 0, "Expected non-empty response")

    def test_03_text_only_chat(self):
        """Pure text chat should also work with encoder disaggregation."""
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": "What is 2 + 3? Answer with just the number.",
                },
            ],
            temperature=0,
            max_tokens=32,
        )
        text = response.choices[0].message.content
        print(f"\n[EPD] Text-only response: {text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text.strip()), 0, "Expected non-empty response")


if __name__ == "__main__":
    unittest.main(verbosity=2)
