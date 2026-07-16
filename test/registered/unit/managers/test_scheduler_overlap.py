import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import (
    Scheduler,
    _should_process_cache_hit_extend_before_schedule,
)
from sglang.srt.managers.scheduler_dp_attn_mixin import (
    MLPSyncBatchInfo,
    _has_cache_hit_extend,
    _will_contract_welm_kv_mirror,
)


class _ForwardMode:
    def __init__(self, *, is_extend: bool):
        self._is_extend = is_extend

    def is_extend(self) -> bool:
        return self._is_extend

    def is_extend_without_speculative(self) -> bool:
        return self._is_extend

    def is_decode(self) -> bool:
        return not self._is_extend


def _batch(
    *,
    is_extend: bool,
    cached_tokens: int = 0,
    has_cache_hit_extend_in_batch: bool = False,
    return_logprob: bool = False,
    extend_len: int = 1,
    logprob_start_len: int = 1,
    decoding_cached_tokens=None,
):
    reqs = [SimpleNamespace(cached_tokens=cached_tokens)]
    decoding_reqs = None
    if decoding_cached_tokens is not None:
        decoding_reqs = [SimpleNamespace(cached_tokens=decoding_cached_tokens)]
        reqs.extend(decoding_reqs)

    return SimpleNamespace(
        forward_mode=_ForwardMode(is_extend=is_extend),
        is_extend_in_batch=is_extend,
        has_cache_hit_extend_in_batch=has_cache_hit_extend_in_batch,
        reqs=reqs,
        decoding_reqs=decoding_reqs,
        return_logprob=return_logprob,
        extend_lens=[extend_len],
        extend_logprob_start_lens=[logprob_start_len],
        is_spec_v2=False,
        has_grammar=False,
    )


def _scheduler(
    *,
    attn_cp_size: int = 2,
    attn_cp_mode: str = "sharded-kv",
    last_batch=None,
    has_result: bool = True,
):
    return SimpleNamespace(
        require_mlp_sync=False,
        attn_cp_size=attn_cp_size,
        server_args=SimpleNamespace(attn_cp_mode=attn_cp_mode),
        result_queue=[object()] if has_result else [],
        last_batch=last_batch,
    )


class TestSchedulerOverlap(unittest.TestCase):
    def test_only_extend_batches_publish_cache_hit_flag(self):
        self.assertFalse(
            _has_cache_hit_extend(_batch(is_extend=False, cached_tokens=8192))
        )
        self.assertTrue(
            _has_cache_hit_extend(_batch(is_extend=True, cached_tokens=8192))
        )

    def test_mixed_batch_only_checks_extend_requests_for_cache_hits(self):
        self.assertFalse(
            _has_cache_hit_extend(
                _batch(
                    is_extend=True,
                    cached_tokens=0,
                    decoding_cached_tokens=8192,
                )
            )
        )
        self.assertTrue(
            _has_cache_hit_extend(
                _batch(
                    is_extend=True,
                    cached_tokens=8192,
                    decoding_cached_tokens=0,
                )
            )
        )

    def test_dp_cache_hit_extend_decision_uses_global_flag(self):
        # A decode request keeps its prefix-cache accounting after prefill.  It
        # must not make only that DP rank process the pending result early.
        decode_batch = _batch(is_extend=False, cached_tokens=8192)
        self.assertFalse(
            _should_process_cache_hit_extend_before_schedule(
                decode_batch,
                has_pending_result=True,
                require_mlp_sync=True,
            )
        )

        # If any peer actually has a cache-hit extend, all DP ranks receive the
        # synchronized flag and take the early-processing path together.
        decode_batch.has_cache_hit_extend_in_batch = True
        self.assertTrue(
            _should_process_cache_hit_extend_before_schedule(
                decode_batch,
                has_pending_result=True,
                require_mlp_sync=True,
            )
        )

    def test_mlp_sync_packs_cache_hit_extend_in_existing_flags_word(self):
        info = MLPSyncBatchInfo(
            dp_size=2,
            tp_size=1,
            cp_size=1,
            num_tokens=1,
            num_tokens_for_logprob=1,
            num_reqs=1,
            can_cuda_graph=False,
            is_extend_in_batch=True,
            local_can_run_tbo=True,
            local_forward_mode=2,
            has_router_replay=True,
            has_cache_hit_extend=True,
            will_contract_welm_kv_mirror=True,
        )

        # All booleans share the already-gathered flags element, so the fix
        # does not add a collective or increase its payload.
        self.assertEqual(info._get_local_tensor(device="cpu")[6].item(), 7)

    def test_input_logprob_extend_does_not_publish_contract_flag(self):
        self.assertFalse(
            _will_contract_welm_kv_mirror(
                _batch(
                    is_extend=True,
                    return_logprob=True,
                    extend_len=1,
                    logprob_start_len=0,
                )
            )
        )
        self.assertTrue(
            _will_contract_welm_kv_mirror(
                _batch(
                    is_extend=True,
                    return_logprob=True,
                    extend_len=1,
                    logprob_start_len=1,
                )
            )
        )

    def test_attncp_sharded_kv_serializes_prefill_after_decode(self):
        scheduler = _scheduler(last_batch=_batch(is_extend=False))

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertTrue(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=True)
                )
            )

    def test_attncp_sharded_kv_serializes_decode_after_prefill(self):
        scheduler = _scheduler(last_batch=_batch(is_extend=True))

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertTrue(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=False)
                )
            )

    def test_attncp_sharded_kv_serializes_consecutive_prefills(self):
        scheduler = _scheduler(last_batch=_batch(is_extend=True))

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertTrue(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=True)
                )
            )

    def test_attncp_sharded_kv_keeps_decode_decode_overlap(self):
        scheduler = _scheduler(last_batch=_batch(is_extend=False))

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertFalse(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=False)
                )
            )

    def test_non_attncp_does_not_disable_decode_prefill_boundary(self):
        scheduler = _scheduler(
            attn_cp_size=1, attn_cp_mode="none", last_batch=_batch(is_extend=False)
        )

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertFalse(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=True)
                )
            )

    def test_attncp_sharded_kv_requires_pending_result_to_disable_overlap(self):
        scheduler = _scheduler(last_batch=_batch(is_extend=False), has_result=False)

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertFalse(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=True)
                )
            )


if __name__ == "__main__":
    unittest.main()
