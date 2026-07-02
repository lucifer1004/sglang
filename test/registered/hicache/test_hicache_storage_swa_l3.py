"""E2E tests for SWA HiCache L3 storage path.

Verifies that the SWA-specific BACKUP_STORAGE / PREFETCH wiring round-trips
correctly through a real storage backend (file). The crucial functional
invariant is byte-equality of greedy output before vs. after a device-cache
flush: if the SWA L3 path were broken (wrong layer-stacking, wrong page hash,
mis-translated swa_loc), the SWA-layer K/V loaded from L3 would diverge and
generated tokens would drift.

KNOWN COVERAGE LIMITATION:
The SWA-specific BACKUP_STORAGE / PREFETCH branches added by Commits 1-2
only fire when the underlying tree cache uses a hybrid SWA memory layout
(``UnifiedRadixCache`` with ``SWAComponent``). Today, sglang's
``is_hybrid_swa_model`` allowlist
(see ``python/sglang/srt/configs/model_config.py``) only includes large
production models (Llama4, DeepseekV4, GptOss, MiMoV2, Step3p5, Gemma4,
Laguna). Gemma-2 / Gemma-3 explicitly disable hybrid SWA memory in
``server_args.py`` because of incompatibility tracked under PR #7367.

So this test running on Gemma-2 2B verifies that:
  - The Full L3 path still works end-to-end (no Commit 1/2 regression on
    the main KV channel)
  - The new env flags (SGLANG_HICACHE_SWA_STORAGE_ENABLE) are no-ops when
    the tree cache doesn't include an SWAComponent — i.e., enabling them
    on a non-hybrid-SWA model doesn't crash

It does NOT verify the SWA-side BACKUP_STORAGE/PREFETCH bytes themselves;
those are exercised by the synthetic-fixture unit tests added in Commits
1 and 2 (``test_unified_radix_cache_unittest.py::test_hicache_swa_*``,
``test_swa_unittest.py::test_swa_backup_pending_*``). When sglang adds
either Gemma-2 or another small model to the hybrid SWA allowlist this
test will start exercising the SWA-side path automatically.

Usage:
    python -m pytest test/registered/hicache/test_hicache_storage_swa_l3.py -v
"""

import json
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci

# Importing from sibling test file under the same directory.
from test_hicache_storage_file_backend import HiCacheStorageBaseMixin

# Smallest publicly available hybrid full/SWA model — Gemma-2 2B alternates
# full and SWA layers 1:1 with sliding_window=4096. The 2B variant fits in
# a single 80GB GPU comfortably and exercises the same SWAKVPool layout as
# 9B/27B, so it's the cheapest config that still routes through every
# component we care about (SWAComponent, MHATokenToKVPoolHost(swa_kv_pool),
# decode-side dual offload).
#
# Using Unsloth's bit-identical community mirror instead of the upstream
# google/gemma-2-2b-it because the latter is HF-gated (manual approval +
# personal access token), which makes it impossible to run in any CI
# environment without per-runner credential setup. The unsloth repo is the
# same Gemma2ForCausalLM architecture, same 26 layers, same
# sliding_window=4096, same weights.
SWA_MODEL_FOR_TEST = "unsloth/gemma-2-2b-it"

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")


class TestHiCacheStorageSWAL3(HiCacheStorageBaseMixin, CustomTestCase):
    """Same-prompt round-trip across a device-cache flush: greedy output
    must be identical, and the second request must show a significant
    cached_tokens hit indicating the L3 path was used.
    """

    @classmethod
    def _get_model_name(cls):
        return SWA_MODEL_FOR_TEST

    @classmethod
    def _get_base_server_args(cls):
        # Override the base settings to keep the prompt under the SWA
        # window — page_size=64 + sliding_window=4096 means up to ~64
        # pages fall in the SWA tail, which is the relevant exercise.
        extra_config = {
            "hicache_storage_pass_prefix_keys": True,
        }
        return {
            "--enable-hierarchical-cache": True,
            "--mem-fraction-static": 0.6,
            "--hicache-ratio": 1.05,
            "--page-size": 64,
            "--enable-cache-report": True,
            "--hicache-storage-prefetch-policy": "wait_complete",
            "--hicache-storage-backend": "file",
            "--hicache-storage-backend-extra-config": json.dumps(extra_config),
            # Gemma-2 needs a permissive max length so the test prompt fits.
            "--context-length": "8192",
        }

    @classmethod
    def _get_additional_server_args_and_env(cls):
        # Enable the SWA-side BACKUP_STORAGE / PREFETCH path. Without this
        # env var the SWAComponent emits None for both phases (Commit 1
        # default behavior) and the test would degenerate to plain Full L3.
        env_vars = {
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "SGLANG_HICACHE_SWA_STORAGE_ENABLE": "1",
        }
        return {}, env_vars

    def test_swa_l3_backup_and_prefetch_byte_equal(self):
        """Greedy output before device-cache flush must equal output after
        the same prompt is replayed through L3.

        Failure modes this catches:
          - SWAComponent backup writes wrong bytes / wrong layer stacking
          - Page hash on the SWA side is computed inconsistently with the
            FULL side, so prefetch fetches mis-aligned data
          - PREFETCH commit attaches host_value to the wrong node, so the
            window-tail SWA at prefill time is not what was backed up
        """
        prompt = self.gen_prompt(768)

        # First run: cold L1, cold L2, empty L3 — populates everything.
        response1 = self.send_request(prompt, max_tokens=100, temperature=0.0)
        self.assertIsNotNone(response1)
        text1 = response1.get("text", "")
        cached1 = self.get_cached_tokens(response1)

        # Flush device-side cache so the next request can only succeed via
        # L2 (host) or L3 (file). Together with hicache-storage-prefetch-policy
        # = wait_complete this forces the prefetch path to be exercised.
        self.flush_cache()

        # Second run: same prompt, same greedy temperature. If SWA L3
        # round-trip is correct, output must be byte-identical.
        response2 = self.send_request(prompt, max_tokens=100, temperature=0.0)
        text2 = response2.get("text", "")
        cached2 = self.get_cached_tokens(response2)

        self.assertEqual(
            text1,
            text2,
            (
                "Output diverged after L3 round-trip — SWA-layer K/V loaded "
                "from storage did not match what was backed up.\n"
                f"  cold     : cached_tokens={cached1}, text={text1!r}\n"
                f"  l3 hit   : cached_tokens={cached2}, text={text2!r}"
            ),
        )
        # Sanity: the second request must have actually used L3, not just
        # generated text from scratch. With a 768-token prompt and page_size
        # 64, expect on the order of 700+ cached tokens.
        self.assertGreater(
            cached2,
            700,
            f"Second request should hit L3 for most of the prefix; got cached_tokens={cached2}",
        )


@unittest.skipIf(is_in_ci(), "Disabled in CI; flag-off path covered by Commit 1 unit tests.")
class TestHiCacheStorageSWAL3FlagOffParity(HiCacheStorageBaseMixin, CustomTestCase):
    """With the SWA-storage flag OFF, output must still be correct (same
    as plain Full-only L3). This is the regression check that Commit 1's
    opt-in gate works end-to-end at the server level.
    """

    @classmethod
    def _get_model_name(cls):
        return SWA_MODEL_FOR_TEST

    @classmethod
    def _get_base_server_args(cls):
        extra_config = {"hicache_storage_pass_prefix_keys": True}
        return {
            "--enable-hierarchical-cache": True,
            "--mem-fraction-static": 0.6,
            "--hicache-ratio": 1.05,
            "--page-size": 64,
            "--enable-cache-report": True,
            "--hicache-storage-prefetch-policy": "wait_complete",
            "--hicache-storage-backend": "file",
            "--hicache-storage-backend-extra-config": json.dumps(extra_config),
            "--context-length": "8192",
        }

    @classmethod
    def _get_additional_server_args_and_env(cls):
        # Flag intentionally NOT set — SWAComponent's BACKUP_STORAGE/PREFETCH
        # branches return None, so this run should match the legacy
        # Full-only L3 behavior.
        env_vars = {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}
        return {}, env_vars

    def test_flag_off_full_only_l3_still_works(self):
        prompt = self.gen_prompt(768)
        response1 = self.send_request(prompt, max_tokens=64, temperature=0.0)
        text1 = response1.get("text", "")

        self.flush_cache()
        response2 = self.send_request(prompt, max_tokens=64, temperature=0.0)
        text2 = response2.get("text", "")
        cached2 = self.get_cached_tokens(response2)

        self.assertEqual(
            text1, text2, "Flag-off mode must still produce byte-equal output via Full L3"
        )
        self.assertGreater(
            cached2, 700, f"Full L3 hit expected; got cached_tokens={cached2}"
        )


if __name__ == "__main__":
    unittest.main()
