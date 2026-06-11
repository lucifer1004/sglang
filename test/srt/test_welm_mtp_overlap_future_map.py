import unittest
from types import SimpleNamespace

import torch

import sglang.srt.managers.overlap_utils as overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _make_draft_input(batch_size=2, has_proposal=True):
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
    if has_proposal:
        draft_input.draft_proposal_parent_list = torch.arange(
            batch_size * 2, dtype=torch.int64
        ).reshape(batch_size, 2)
        draft_input.draft_proposal_top_scores_index = (
            torch.arange(batch_size * 3, dtype=torch.int64).reshape(batch_size, 3)
            + 10
        )
        draft_input.draft_proposal_tokens = (
            torch.arange(batch_size * 3, dtype=torch.int64).reshape(batch_size, 3)
            + 100
        )
        draft_input.welm_mtp_has_draft_proposal = True
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

    def test_resolve_preserves_v1_draft_proposal_tensors(self):
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

        resolved = _make_draft_input(has_proposal=False)
        resolved.future_indices = future_indices
        future_map.resolve_future(SimpleNamespace(spec_info=resolved))

        self.assertTrue(resolved.welm_mtp_has_draft_proposal)
        torch.testing.assert_close(
            resolved.draft_proposal_parent_list,
            stored.draft_proposal_parent_list,
        )
        torch.testing.assert_close(
            resolved.draft_proposal_top_scores_index,
            stored.draft_proposal_top_scores_index,
        )
        torch.testing.assert_close(
            resolved.draft_proposal_tokens,
            stored.draft_proposal_tokens,
        )

    def test_resolve_preserves_draft_proposal_when_mixed_with_deferred_prefill(self):
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
        future_map.welm_mtp_has_draft_proposal_buf[missing_index] = False
        future_map.welm_mtp_deferred_prefill_draft_buf[missing_index] = True
        future_map.welm_mtp_draft_proposal_parent_list_buf[missing_index].zero_()
        future_map.welm_mtp_draft_proposal_top_scores_index_buf[missing_index].zero_()
        future_map.welm_mtp_draft_proposal_tokens_buf[missing_index].zero_()

        resolved = _make_draft_input(has_proposal=False)
        resolved.future_indices = future_indices
        future_map.resolve_future(SimpleNamespace(spec_info=resolved))

        self.assertTrue(resolved.welm_mtp_has_draft_proposal)
        self.assertFalse(resolved.welm_mtp_deferred_prefill_draft)
        torch.testing.assert_close(
            resolved.draft_proposal_parent_list[0],
            stored.draft_proposal_parent_list[0],
        )
        torch.testing.assert_close(
            resolved.draft_proposal_top_scores_index[0],
            stored.draft_proposal_top_scores_index[0],
        )
        torch.testing.assert_close(
            resolved.draft_proposal_tokens[0],
            stored.draft_proposal_tokens[0],
        )
        torch.testing.assert_close(
            resolved.draft_proposal_parent_list[1],
            torch.zeros_like(resolved.draft_proposal_parent_list[1]),
        )
        torch.testing.assert_close(
            resolved.draft_proposal_top_scores_index[1],
            torch.zeros_like(resolved.draft_proposal_top_scores_index[1]),
        )
        torch.testing.assert_close(
            resolved.draft_proposal_tokens[1],
            torch.zeros_like(resolved.draft_proposal_tokens[1]),
        )

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

        resolved = _make_draft_input(has_proposal=False)
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

        resolved = _make_draft_input(has_proposal=False)
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


if __name__ == "__main__":
    unittest.main()
