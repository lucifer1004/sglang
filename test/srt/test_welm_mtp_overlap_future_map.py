import unittest
from types import SimpleNamespace

import torch

import sglang.srt.managers.overlap_utils as overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _make_draft_input(batch_size=2, has_sampling_state=True):
    topk = 1
    draft_input = EagleDraftInput(
        hidden_states=torch.arange(batch_size * 4, dtype=torch.float32).reshape(
            batch_size, 4
        ),
        verified_id=torch.arange(batch_size, dtype=torch.int64),
        topk_p=torch.ones((batch_size, topk), dtype=torch.float32),
        topk_index=torch.arange(batch_size, dtype=torch.int64).reshape(batch_size, 1),
        new_seq_lens=torch.arange(batch_size, dtype=torch.int32) + 8,
        num_tokens_per_req=1,
        num_tokens_for_logprob_per_req=1,
    )
    if has_sampling_state:
        draft_input.draft_probs = (
            torch.arange(batch_size * 2 * 4, dtype=torch.float32).reshape(
                batch_size, 2, 4
            )
            / 100
        )
        draft_input.welm_mtp_draft_topk_indices = torch.arange(
            batch_size * 2 * 3, dtype=torch.int64
        ).reshape(batch_size, 2, 3)
        draft_input.welm_mtp_draft_topk_values = (
            torch.arange(batch_size * 2 * 3, dtype=torch.float32).reshape(
                batch_size, 2, 3
            )
            / 10
        )
    return draft_input


def _attach_oe_history(draft_input):
    batch_size = int(draft_input.verified_id.numel())
    draft_input.welm_mtp_oe_history_state = (
        torch.arange(batch_size * 3, dtype=torch.int64).reshape(batch_size, 3) + 1000
    )
    return draft_input


class TestWelmMTPOverlapFutureMap(unittest.TestCase):
    def setUp(self):
        self._record_stream = torch.Tensor.record_stream
        torch.Tensor.record_stream = lambda _tensor, _stream: None
        self._spec_need_hidden_states = overlap_utils.spec_need_hidden_states
        overlap_utils.spec_need_hidden_states = lambda: False

    def tearDown(self):
        torch.Tensor.record_stream = self._record_stream
        overlap_utils.spec_need_hidden_states = self._spec_need_hidden_states

    def test_resolve_preserves_welm_mtp_sampling_state(self):
        future_map = FutureMap(
            max_running_requests=4,
            chunked_prefill_size=0,
            context_len=128,
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.EAGLE,
        )
        stored = _make_draft_input()
        future_indices = future_map.alloc_future_indices(2)
        future_map.store_to_map_for_new_batch(future_indices, stored)

        resolved = _make_draft_input(has_sampling_state=False)
        resolved.future_indices = future_indices
        future_map.resolve_future(SimpleNamespace(spec_info=resolved))

        torch.testing.assert_close(
            resolved.draft_probs,
            stored.draft_probs,
        )
        torch.testing.assert_close(
            resolved.welm_mtp_draft_topk_indices,
            stored.welm_mtp_draft_topk_indices,
        )
        torch.testing.assert_close(
            resolved.welm_mtp_draft_topk_values,
            stored.welm_mtp_draft_topk_values,
        )

    def test_resolve_drops_sampling_state_when_mixed_with_deferred_prefill(self):
        future_map = FutureMap(
            max_running_requests=4,
            chunked_prefill_size=0,
            context_len=128,
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.EAGLE,
        )
        stored = _make_draft_input()
        future_indices = future_map.alloc_future_indices(2)
        future_map.store_to_map_for_new_batch(future_indices, stored)
        missing_index = future_indices.indices[1]
        future_map.welm_mtp_has_draft_probs_buf[missing_index] = False
        future_map.welm_mtp_has_draft_topk_buf[missing_index] = False
        future_map.welm_mtp_deferred_prefill_draft_buf[missing_index] = True
        future_map.welm_mtp_draft_probs_buf[missing_index].zero_()
        future_map.welm_mtp_draft_topk_indices_buf[missing_index].zero_()
        future_map.welm_mtp_draft_topk_values_buf[missing_index].zero_()

        resolved = _make_draft_input(has_sampling_state=False)
        resolved.future_indices = future_indices
        future_map.resolve_future(SimpleNamespace(spec_info=resolved))

        self.assertTrue(resolved.welm_mtp_deferred_prefill_draft)
        torch.testing.assert_close(
            resolved.welm_mtp_deferred_prefill_draft_mask,
            torch.tensor([False, True]),
        )
        self.assertIsNone(resolved.draft_probs)
        self.assertIsNone(resolved.welm_mtp_draft_topk_indices)
        self.assertIsNone(resolved.welm_mtp_draft_topk_values)

    def test_resolve_preserves_welm_mtp_oe_history_state(self):
        future_map = FutureMap(
            max_running_requests=4,
            chunked_prefill_size=0,
            context_len=128,
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.EAGLE,
        )
        stored = _attach_oe_history(_make_draft_input())
        future_indices = future_map.alloc_future_indices(2)
        future_map.store_to_map_for_new_batch(future_indices, stored)

        resolved = _make_draft_input(has_sampling_state=False)
        resolved.future_indices = future_indices
        future_map.resolve_future(SimpleNamespace(spec_info=resolved))

        torch.testing.assert_close(
            resolved.welm_mtp_oe_history_state,
            stored.welm_mtp_oe_history_state,
        )

    def test_resolve_preserves_welm_mtp_oe_history_state_for_mixed_rows(self):
        future_map = FutureMap(
            max_running_requests=4,
            chunked_prefill_size=0,
            context_len=128,
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.EAGLE,
        )
        stored = _attach_oe_history(_make_draft_input())
        future_indices = future_map.alloc_future_indices(2)
        future_map.store_to_map_for_new_batch(future_indices, stored)
        missing_index = future_indices.indices[1]
        future_map.welm_mtp_has_oe_history_buf[missing_index] = False
        future_map.welm_mtp_oe_history_buf[missing_index].zero_()

        resolved = _make_draft_input(has_sampling_state=False)
        resolved.welm_mtp_oe_history_state = torch.ones((2, 3), dtype=torch.int64)
        resolved.future_indices = future_indices
        future_map.resolve_future(SimpleNamespace(spec_info=resolved))

        self.assertTrue(hasattr(resolved, "welm_mtp_oe_history_state"))
        torch.testing.assert_close(
            resolved.welm_mtp_oe_history_state[0],
            stored.welm_mtp_oe_history_state[0],
        )
        torch.testing.assert_close(
            resolved.welm_mtp_oe_history_state[1],
            torch.zeros_like(resolved.welm_mtp_oe_history_state[1]),
        )


class TestWelmMTPPrefillDefer(unittest.TestCase):
    @staticmethod
    def _worker():
        return EagleDraftWorker.__new__(EagleDraftWorker)

    @staticmethod
    def _req(fill_len, origin_len, *, is_chunked=1):
        return SimpleNamespace(
            rid=f"req-{fill_len}-{origin_len}",
            is_chunked=is_chunked,
            fill_ids=list(range(fill_len)),
            origin_input_ids=list(range(origin_len)),
            output_ids=[],
        )

    def test_defer_intermediate_chunk(self):
        worker = self._worker()
        batch = SimpleNamespace(reqs=[self._req(fill_len=8192, origin_len=11333)])

        self.assertTrue(worker._should_defer_welmv4_mtp_prefill_draft(batch))

    def test_do_not_defer_final_chunk(self):
        worker = self._worker()
        batch = SimpleNamespace(reqs=[self._req(fill_len=11333, origin_len=11333)])

        self.assertFalse(worker._should_defer_welmv4_mtp_prefill_draft(batch))

    def test_mixed_intermediate_and_completed_prefill_rows_use_row_mask(self):
        worker = self._worker()
        batch = SimpleNamespace(
            reqs=[
                self._req(fill_len=8192, origin_len=11333),
                self._req(fill_len=10, origin_len=10, is_chunked=0),
            ]
        )

        self.assertFalse(worker._should_defer_welmv4_mtp_prefill_draft(batch))
        torch.testing.assert_close(
            worker._get_welmv4_mtp_deferred_prefill_mask(batch),
            torch.tensor([True, False]),
        )


if __name__ == "__main__":
    unittest.main()
