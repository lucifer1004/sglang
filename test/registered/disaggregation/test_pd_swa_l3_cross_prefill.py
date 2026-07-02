"""End-to-end test: PD-disaggregation + SWA L3 + cross-prefill-node simulation.

This test is the single most important runtime check for the SWA HiCache L3
feature. Unlike the single-server byte-equal test in
``test/registered/hicache/test_hicache_storage_swa_l3.py``, this one drives a
real PD topology with KV transfer between prefill and decode, and exercises
the SWA-side BACKUP_STORAGE / PREFETCH branches that Commits 1 and 2 added.

Why GPT-OSS 20B specifically:
    sglang's hybrid SWA memory path (UnifiedRadixCache + SWAComponent +
    SWAKVPool) is gated by ``is_hybrid_swa_model()``. The current allowlist
    is {Llama4, DeepseekV4, GptOss, MiMoV2, Step3p5, Gemma4, Laguna}. Of
    these, ``openai/gpt-oss-20b`` is the only one that fits in a single
    H20-class GPU (~40GB bf16) AND is publicly downloadable without HF
    gating. Gemma-2 is auto-disabled by ``server_args.py``'s PR #7367 fixme,
    so it does NOT exercise this path even with our flags on.

Cross-prefill-node simulation:
    To validate that L3 lets a *different* prefill node hit the cached
    prefix, we use the kill-restart pattern: serve turn 1 through prefill_A,
    kill prefill_A (which wipes its L1 + L2), launch prefill_A again with
    the same config (effectively a "fresh" node from L1/L2's perspective —
    L3 file backend persists across the kill), then serve turn 2 with the
    same prompt and assert the new prefill instance hits L3.

Usage:
    python -m pytest test/registered/disaggregation/test_pd_swa_l3_cross_prefill.py -v -s
"""

import json
import os
import random
import tempfile
import time
import unittest
from typing import Dict

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
)

# GPT-OSS 20B is in sglang's `is_hybrid_swa_model` allowlist (see
# python/sglang/srt/configs/model_config.py:1683) — meaning it actually
# routes through the SWAKVPool + SWAComponent path that Commits 1/2/3
# add L3 backup/prefetch hooks to. Public, non-gated, and small enough to
# fit one instance per GPU on H20.
SWA_MODEL_FOR_TEST = "openai/gpt-oss-20b"

register_cuda_ci(est_time=900, stage="extra-a", runner_config="2-gpu-large")


class TestPDSWAL3CrossPrefill(PDDisaggregationServerBase):
    """PD with SWA HiCache L3 — verifies a fresh prefill instance hits L3
    for the SWA window-tail of a previous turn."""

    capture_per_side_logs = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = SWA_MODEL_FOR_TEST
        cls.temp_dir = tempfile.mkdtemp()

        # Both sides share the same file-backed L3 store (content-addressed).
        common_env = {
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "SGLANG_HICACHE_SWA_STORAGE_ENABLE": "1",
            # CRITICAL for SWA L3: routes the tree cache through
            # UnifiedRadixCache+SWAComponent (where Commits 1/2/3 live)
            # instead of legacy HiRadixCache (which doesn't support SWAKVPool).
            "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
        }
        cls.extra_prefill_env = dict(common_env)
        # Decode side opts in to the dual-pool offload path added by Commit 2.
        cls.extra_decode_env = {
            **common_env,
            "SGLANG_HICACHE_SWA_DECODE_STORAGE_ENABLE": "1",
        }

        common_hicache_args = [
            "--mem-fraction-static",
            "0.5",
            # Absolute size in GB — overrides hicache_ratio. Two servers
            # share one container's host RAM, so a 30GB cap per side keeps
            # the combined host allocation under typical cgroup limits.
            "--hicache-size",
            "30",
            "--page-size",
            "64",
            "--enable-cache-report",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-backend-extra-config",
            json.dumps({"hicache_storage_pass_prefix_keys": True}),
        ]
        # Prefill side uses the standard tree-cache hicache path (this is
        # where SWAComponent's BACKUP_STORAGE / PREFETCH from Commit 1 fire).
        cls.extra_prefill_args = ["--enable-hierarchical-cache"] + common_hicache_args
        # Decode side uses the disagg-specific offload path (this is where
        # DecodeKVCacheOffloadManager from Commit 2 fires). PD-decode has
        # disable_radix_cache=True by default, so --enable-hierarchical-cache
        # would conflict; the offload-kvcache flag is the right knob.
        cls.extra_decode_args = [
            "--disaggregation-decode-enable-offload-kvcache",
        ] + common_hicache_args

        # Boot full PD stack: prefill_A + decode + LB.
        cls.launch_all()
        print(
            f"Initial PD stack ready. prefill_url={cls.prefill_url} "
            f"decode_url={cls.decode_url} lb_url={cls.lb_url} "
            f"L3_dir={cls.temp_dir}"
        )

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            import shutil

            if hasattr(cls, "temp_dir") and cls.temp_dir:
                shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _send(
        self, prompt: str, max_tokens: int = 64, temperature: float = 0.0
    ) -> Dict:
        """Hit the LB; it routes to the live prefill+decode pair."""
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Generate failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def _gen_prompt(self, token_count: int) -> str:
        # Simple deterministic prompt — same length each call would also
        # work, but using a long Python integer-derived prompt avoids the
        # per-call randomness that would defeat byte-equal comparison.
        random.seed(0xCAFE_BEEF)
        # Use words from a stable source so encoding is reproducible.
        words = [
            "The",
            "sliding",
            "window",
            "attention",
            "mechanism",
            "limits",
            "context",
            "to",
            "a",
            "fixed",
            "size",
            "for",
            "memory",
            "efficiency",
            "during",
            "long-context",
            "inference",
            "tasks",
            "across",
            "many",
            "model",
            "architectures",
            "including",
            "Mistral",
            "Gemma",
            "and",
            "GPT-OSS",
            "with",
            "trade-offs",
            "in",
            "modeling",
            "quality",
        ]
        # Repeat enough words to roughly hit the token count target.
        return " ".join(random.choices(words, k=token_count))

    def _restart_prefill(self) -> None:
        """Kill prefill_A, then start a fresh prefill_A' (same port, same
        config). The new instance has empty L1+L2 but the L3 file store
        survives the kill."""
        print("Killing prefill (PID=%d)..." % self.process_prefill.pid)
        kill_process_tree(self.process_prefill.pid)
        # Give the OS time to release the bound port and the LB time to
        # observe the disconnect. 2s is enough for sr/router on localhost.
        time.sleep(3)
        print("Starting fresh prefill instance with the same config...")
        self.start_prefill()
        self.wait_server_ready(
            self.prefill_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=self.process_prefill,
        )
        print("Fresh prefill is ready (simulating cross-node prefill).")

    def test_swa_l3_survives_prefill_restart(self):
        """Greedy output of the same prompt before vs. after a prefill
        restart must be identical, with the post-restart request showing a
        large cached_tokens (proving L3 was used)."""
        # Pick a prompt long enough to span multiple page_size=64 pages.
        prompt = self._gen_prompt(640)

        # ===== Turn 1: cold start. prefill_A populates L1 + L2 + L3. =====
        print("\n=== Turn 1: cold start populates L3 ===")
        r1 = self._send(prompt, max_tokens=64, temperature=0.0)
        text1 = r1.get("text", "")
        cached1 = int(r1.get("meta_info", {}).get("cached_tokens", 0))
        print(f"  cold response: text[:80]={text1[:80]!r} cached_tokens={cached1}")

        # Sanity: the first call's prompt is brand new — almost zero cached.
        self.assertLess(
            cached1,
            64,
            f"Cold call should have minimal cache; got cached_tokens={cached1}",
        )

        # ===== Restart prefill — kills L1 + L2, keeps L3. =====
        self._restart_prefill()

        # ===== Turn 2: replay same prompt. Must hit L3 from new prefill. =====
        print("\n=== Turn 2: same prompt on fresh prefill, must hit L3 ===")
        r2 = self._send(prompt, max_tokens=64, temperature=0.0)
        text2 = r2.get("text", "")
        cached2 = int(r2.get("meta_info", {}).get("cached_tokens", 0))
        print(f"  l3-hit response: text[:80]={text2[:80]!r} cached_tokens={cached2}")

        # Strong invariant: bytewise-identical greedy output is the only
        # way the SWA window-tail K/V loaded from L3 can differ from what
        # was backed up.
        self.assertEqual(
            text1,
            text2,
            (
                "Output diverged across the prefill restart — SWA-layer K/V "
                "loaded from L3 file store does not match what was backed up.\n"
                f"  cold (turn 1) : cached={cached1} text={text1!r}\n"
                f"  l3-hit (turn 2): cached={cached2} text={text2!r}"
            ),
        )

        # The second request must show a real L3 hit. With page_size=64
        # and a 640-token prompt, expect ≥ ~512 cached tokens (8 pages).
        self.assertGreater(
            cached2,
            500,
            (
                "Fresh prefill should hit L3 for most of the prefix; "
                f"got cached_tokens={cached2}. Without L3 working this would "
                "stay close to zero."
            ),
        )

        # Sanity-print whether decode-side dual offload was actually wired —
        # if hybrid_swa wasn't engaged, this test still passes (Full L3
        # only) but the SWA-side path isn't exercised.
        if hasattr(self, "_prefill_stdout_buf") and self._prefill_stdout_buf:
            log = self._prefill_stdout_buf.getvalue()
            assert "hybrid_swa=True" in log, (
                "Tree cache did not initialize with hybrid_swa=True for "
                f"{SWA_MODEL_FOR_TEST}. SWA L3 path is not being exercised. "
                "Check is_hybrid_swa_model allowlist and disable_hybrid_swa_memory."
            )
            print("Confirmed: hybrid_swa=True in prefill log.")


if __name__ == "__main__":
    unittest.main()
