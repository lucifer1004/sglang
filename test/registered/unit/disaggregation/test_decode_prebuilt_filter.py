from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import (
    HashInputIdsBuffer,
    OverEncodingContext,
    ScheduleBatch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class _SamplingInfo:
    def filter_batch(self, keep_indices, keep_indices_device):
        self.keep_indices = list(keep_indices)


def _req(rid: str):
    return SimpleNamespace(
        rid=rid,
        attn_cp_prefill_split_spec=None,
        return_logprob=False,
        stream=False,
        grammar=None,
    )


def _make_prebuilt_batch() -> ScheduleBatch:
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.forward_mode = ForwardMode.PREBUILT
    batch.enable_overlap = False
    batch.spec_algorithm = SpeculativeAlgorithm.NONE
    batch.device = "cpu"
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.reqs = [_req("finished"), _req("survivor")]
    batch.req_pool_indices = torch.tensor([7, 8], dtype=torch.int64)
    batch.seq_lens = torch.tensor([2, 5], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([2, 5], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([2, 5], dtype=torch.int32)
    batch.seq_lens_sum = 7
    batch.input_ids = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32)
    batch.out_cache_loc = torch.tensor(
        [100, 101, 200, 201, 202], dtype=torch.int64
    )
    batch.prefix_lens = [0, 2]
    batch.extend_lens = [2, 3]
    batch.extend_num_tokens = 5
    batch.extend_logprob_start_lens = [0, 0]
    batch.extend_input_logprob_token_ids = None
    batch.output_ids = torch.tensor([30, 31], dtype=torch.int64)
    batch.multimodal_inputs = None
    batch.mamba_track_indices = None
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None
    batch.router_replay_topk_ids = None
    batch.router_replay_mask = None
    batch.return_logprob = False
    batch.top_logprobs_nums = None
    batch.token_ids_logprobs = None
    batch.has_stream = False
    batch.has_grammar = False
    batch.sampling_info = _SamplingInfo()
    batch.spec_info = None
    batch.attn_cp_prefill_split_specs = None
    batch.oe_context = OverEncodingContext(
        input_ids_buffer=HashInputIdsBuffer([[1, 2], [3, 4]]),
        legacy_prefixes=[[5, 6], [7, 8], [9, 10]],
    )
    return batch


def test_prebuilt_filter_preserves_surviving_packed_prompt_state():
    batch = _make_prebuilt_batch()

    batch.filter_batch(keep_indices=[1])

    assert [req.rid for req in batch.reqs] == ["survivor"]
    torch.testing.assert_close(
        batch.input_ids, torch.tensor([20, 21, 22], dtype=torch.int32)
    )
    torch.testing.assert_close(batch.out_cache_loc, torch.tensor([200, 201, 202]))
    assert batch.prefix_lens == [2]
    assert batch.extend_lens == [3]
    assert batch.extend_num_tokens == 3
    assert batch.extend_logprob_start_lens == [0]
    assert batch.oe_context.hash_prefixes == [[2], [4]]
    assert batch.oe_context.legacy_prefixes == [[6], [8], [10]]


def test_filter_empty_uninitialized_batch_does_not_require_forward_mode():
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.forward_mode = None
    batch.enable_overlap = False
    batch.spec_algorithm = SpeculativeAlgorithm.NONE
    batch.reqs = []

    batch.filter_batch()

    assert batch.reqs == []
    assert batch.spec_info is None
