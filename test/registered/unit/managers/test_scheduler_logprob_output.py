from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)


def test_add_logprob_return_values_normalizes_top_logprob_tensors():
    scheduler = SchedulerOutputProcessorMixin()
    req = SimpleNamespace(
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        input_top_logprobs_val=None,
        input_top_logprobs_idx=None,
        input_token_ids_logprobs_val=None,
        input_token_ids_logprobs_idx=None,
        output_token_logprobs_val=[],
        output_token_logprobs_idx=[],
        output_top_logprobs_val=[],
        output_top_logprobs_idx=[],
        output_token_ids_logprobs_val=[],
        output_token_ids_logprobs_idx=[],
        top_logprobs_num=2,
        token_ids_logprob=None,
    )
    output = SimpleNamespace(
        next_token_logprobs=None,
        next_token_top_logprobs_val=[torch.tensor([-0.25, -1.5])],
        next_token_top_logprobs_idx=[torch.tensor([17, 29])],
    )

    scheduler.add_logprob_return_values(
        i=0,
        req=req,
        pt=0,
        next_token_ids=[17],
        num_input_logprobs=0,
        output=output,
    )

    assert req.output_top_logprobs_val == [[-0.25, -1.5]]
    assert req.output_top_logprobs_idx == [[17, 29]]
