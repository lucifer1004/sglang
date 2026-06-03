import sys

import pytest
import torch

from sglang.jit_kernel.welm_oe import (
    warmup_welm_oe_decode_hash_kernel,
    welm_oe_decode_hash_from_prefixes_cuda,
)
from sglang.srt.managers.schedule_batch import OverEncodingContext
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


class _Req:
    def __init__(self, origin_input_ids, output_ids, overlap_decode_count=0):
        self.origin_input_ids = origin_input_ids
        self.output_ids = output_ids
        self._overlap_decode_count = overlap_decode_count


def test_welm_oe_decode_hash_context_builds_lag_major_prefixes():
    reqs = [
        _Req(origin_input_ids=[1, 2], output_ids=[3, 4]),
        _Req(origin_input_ids=[10], output_ids=[20]),
    ]

    ctx = OverEncodingContext.from_decode_hash_kernel(
        reqs, enable_overlap=False, history_width=3
    )

    assert len(ctx.decode_hash_prefixes) == 3
    assert ctx.input_ids_buffer is None
    assert ctx.decode_hash_prefixes == [[3, 10], [2, 0], [1, 0]]


def test_welm_oe_decode_hash_context_rejects_two_history_buffers():
    with pytest.raises(ValueError, match="cannot hold both"):
        OverEncodingContext(
            input_ids_buffer=torch.empty(4, dtype=torch.int64),
            decode_hash_prefixes=[[1]],
        )


def _reference_decode_hash(
    input_ids,
    prefixes,
    oe_grams,
    oe_vocab_sizes,
    vocab_size,
):
    mask = 0xFFFFFFFF
    hashed = torch.empty((len(oe_grams), input_ids.numel()), dtype=torch.int64)
    history_width = len(prefixes) // input_ids.numel()

    for token_idx in range(input_ids.numel()):
        input_id = int(input_ids[token_idx].item()) & mask
        for branch_idx, (gram, oe_vocab_size) in enumerate(
            zip(oe_grams, oe_vocab_sizes)
        ):
            running_ids = input_id
            vocab_power = vocab_size & mask
            for lag in range(1, gram):
                prev = int(prefixes[(lag - 1) * input_ids.numel() + token_idx]) & mask
                running_ids = (running_ids + prev * vocab_power) & mask
                vocab_power = (vocab_power * vocab_size) & mask
            hashed[branch_idx, token_idx] = (
                (running_ids * 2654435761) & mask
            ) % oe_vocab_size

    assert history_width >= max(oe_grams) - 1
    return hashed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_welm_oe_decode_hash_dynamic_prefixes_and_config():
    device = "cuda"
    vocab_size = 32017
    oe_grams = (2, 3, 4)
    oe_vocab_sizes = (257, 263, 269)

    input_ids_cpu = torch.tensor([11, 42, 70001], dtype=torch.int64)
    prefixes = [
        19,
        7,
        43,
        17,
        5,
        41,
        13,
        3,
        37,
    ]
    expected_hash = _reference_decode_hash(
        input_ids_cpu,
        prefixes,
        oe_grams,
        oe_vocab_sizes,
        vocab_size,
    )

    input_ids = input_ids_cpu.to(device)
    hashed_out = torch.empty(
        (len(oe_grams), input_ids.numel()), dtype=torch.int64, device=device
    )

    welm_oe_decode_hash_from_prefixes_cuda(
        input_ids,
        prefixes,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        vocab_size,
    )
    torch.cuda.synchronize()

    assert torch.equal(hashed_out.cpu(), expected_hash)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_welm_oe_decode_hash_accepts_graph_buffer_view():
    device = "cuda"
    vocab_size = 32017
    oe_grams = (2, 3, 4)
    oe_vocab_sizes = (257, 263, 269)

    input_ids_cpu = torch.tensor([11, 42, 70001], dtype=torch.int64)
    prefixes = [19, 7, 43, 17, 5, 41, 13, 3, 37]
    expected_hash = _reference_decode_hash(
        input_ids_cpu,
        prefixes,
        oe_grams,
        oe_vocab_sizes,
        vocab_size,
    )

    input_ids = input_ids_cpu.to(device)
    graph_buffer = torch.empty(
        (len(oe_grams), 16), dtype=torch.int64, device=device
    )
    hashed_out = graph_buffer[:, : input_ids.numel()]

    assert not hashed_out.is_contiguous()
    welm_oe_decode_hash_from_prefixes_cuda(
        input_ids,
        prefixes,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        vocab_size,
    )
    torch.cuda.synchronize()

    assert torch.equal(hashed_out.cpu(), expected_hash)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_welm_oe_decode_hash_warmup_uses_model_width():
    warmup_welm_oe_decode_hash_kernel(
        "cuda",
        history_width=3,
        oe_grams=(2, 3, 4),
        oe_vocab_sizes=(1, 1, 1),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
