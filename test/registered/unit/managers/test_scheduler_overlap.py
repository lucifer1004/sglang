import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler


class _ForwardMode:
    def __init__(self, *, is_extend: bool):
        self._is_extend = is_extend

    def is_extend(self) -> bool:
        return self._is_extend

    def is_decode(self) -> bool:
        return not self._is_extend


def _batch(*, is_extend: bool):
    return SimpleNamespace(
        forward_mode=_ForwardMode(is_extend=is_extend),
        is_extend_in_batch=is_extend,
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
