import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.jit_kernel.welm_oe import (
    warmup_welm_oe_hash_kernel,
    welm_oe_hash_decode_from_prefixes_cuda,
    welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda,
    welm_oe_hash_segments_from_prefixes_cuda,
)
from sglang.srt.managers.schedule_batch import HashInputIdsBuffer, OverEncodingContext
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.welm_perf_opt import fill_welm_oe_hash_inputs
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

    assert len(ctx.hash_prefixes) == 3
    assert isinstance(ctx.input_ids_buffer, HashInputIdsBuffer)
    assert ctx.hash_prefixes == [[3, 10], [2, 0], [1, 0]]


def test_welm_oe_extend_hash_context_builds_boundary_prefixes():
    reqs = [
        type("Req", (), {"fill_ids": [1, 2, 3, 4]})(),
        type("Req", (), {"fill_ids": [10, 20]})(),
    ]

    ctx = OverEncodingContext.from_extend_hash_kernel(
        reqs, logical_prefix_lens=[2, 0], history_width=3
    )

    assert isinstance(ctx.input_ids_buffer, HashInputIdsBuffer)
    assert ctx.hash_prefixes == [[2, 0], [1, 0], [0, 0]]


def test_welm_oe_hash_context_merges_prefix_rows_for_mixed_batch():
    lhs = OverEncodingContext(input_ids_buffer=HashInputIdsBuffer([[1, 2], [3, 4]]))
    rhs = OverEncodingContext(input_ids_buffer=HashInputIdsBuffer([[5], [6]]))

    lhs.merge_buffer(rhs)

    assert lhs.hash_prefixes == [[1, 2, 5], [3, 4, 6]]


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
@pytest.mark.parametrize("input_dtype", [torch.int64, torch.int32])
def test_welm_oe_decode_hash_dynamic_prefixes_and_config(input_dtype):
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

    input_ids = input_ids_cpu.to(device=device, dtype=input_dtype)
    hashed_out = torch.empty(
        (len(oe_grams), input_ids.numel()), dtype=torch.int64, device=device
    )

    welm_oe_hash_decode_from_prefixes_cuda(
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
    welm_oe_hash_decode_from_prefixes_cuda(
        input_ids,
        prefixes,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        vocab_size,
    )
    torch.cuda.synchronize()

    assert torch.equal(hashed_out.cpu(), expected_hash)


def _reference_segment_hash(
    input_ids,
    extend_start_loc,
    extend_seq_lens,
    prefixes,
    oe_grams,
    oe_vocab_sizes,
    vocab_size,
):
    mask = 0xFFFFFFFF
    hashed = torch.empty((len(oe_grams), input_ids.numel()), dtype=torch.int64)
    num_segments = len(extend_seq_lens)
    history_width = len(prefixes) // num_segments

    for segment_idx, (start, seg_len) in enumerate(
        zip(extend_start_loc, extend_seq_lens)
    ):
        for local_pos in range(seg_len):
            token_idx = start + local_pos
            input_id = int(input_ids[token_idx].item()) & mask
            for branch_idx, (gram, oe_vocab_size) in enumerate(
                zip(oe_grams, oe_vocab_sizes)
            ):
                running_ids = input_id
                vocab_power = vocab_size & mask
                for lag in range(1, gram):
                    if local_pos >= lag:
                        prev = int(input_ids[token_idx - lag].item()) & mask
                    else:
                        prefix_lag = lag - local_pos - 1
                        prev = (
                            int(prefixes[prefix_lag * num_segments + segment_idx])
                            & mask
                        )
                    running_ids = (running_ids + prev * vocab_power) & mask
                    vocab_power = (vocab_power * vocab_size) & mask
                hashed[branch_idx, token_idx] = (
                    (running_ids * 2654435761) & mask
                ) % oe_vocab_size

    assert history_width >= max(oe_grams) - 1
    return hashed


def _accepted_history_token(entry_history, accepted_row, accept_len, seq_id, lag):
    extra_lens = max(0, min(int(accept_len) - 1, len(accepted_row) - 1))
    valid_tail = [int(token) for token in accepted_row if int(token) >= 0][1:]
    if lag <= extra_lens:
        tail_idx = extra_lens - lag
        if tail_idx < len(valid_tail):
            return valid_tail[tail_idx]

    history_lag = lag - extra_lens
    col = entry_history.shape[1] - history_lag
    if col < 0 or col >= entry_history.shape[1]:
        return 0
    return int(entry_history[seq_id, col].item())


def _reference_draft_extend_after_verify_hash_from_history(
    input_ids,
    accepted_draft_token_ids,
    accept_lens,
    entry_history,
    oe_grams,
    oe_vocab_sizes,
    vocab_size,
    draft_token_num,
):
    mask = 0xFFFFFFFF
    num_segments = len(accept_lens)
    history_width = entry_history.shape[1]
    hashed = torch.empty((len(oe_grams), input_ids.numel()), dtype=torch.int64)
    next_history = torch.empty((num_segments, history_width), dtype=torch.int64)

    for seq_id in range(num_segments):
        accepted_row = accepted_draft_token_ids[seq_id]
        selected_pos = max(0, min(int(accept_lens[seq_id]) - 1, draft_token_num - 1))
        for col in range(history_width):
            if col + 1 == history_width:
                next_history[seq_id, col] = int(
                    input_ids[seq_id * draft_token_num + selected_pos].item()
                )
            else:
                next_history[seq_id, col] = _accepted_history_token(
                    entry_history,
                    accepted_row,
                    accept_lens[seq_id],
                    seq_id,
                    history_width - col - 1,
                )

        for local_pos in range(draft_token_num):
            token_idx = seq_id * draft_token_num + local_pos
            input_id = int(input_ids[token_idx].item()) & mask
            for branch_idx, (gram, oe_vocab_size) in enumerate(
                zip(oe_grams, oe_vocab_sizes)
            ):
                running_ids = input_id
                vocab_power = vocab_size & mask
                for lag in range(1, gram):
                    if local_pos >= lag:
                        prev = int(input_ids[token_idx - lag].item()) & mask
                    else:
                        prev = (
                            _accepted_history_token(
                                entry_history,
                                accepted_row,
                                accept_lens[seq_id],
                                seq_id,
                                lag - local_pos,
                            )
                            & mask
                        )
                    running_ids = (running_ids + prev * vocab_power) & mask
                    vocab_power = (vocab_power * vocab_size) & mask
                hashed[branch_idx, token_idx] = (
                    (running_ids * 2654435761) & mask
                ) % oe_vocab_size

    return hashed, next_history


def test_welm_oe_target_verify_fill_hash_inputs_rejects_prefix_fallback():
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
    input_ids = torch.tensor([100, 101], dtype=torch.int64)
    hashed_out = torch.empty((3, input_ids.numel()), dtype=torch.int64)

    with pytest.raises(RuntimeError, match="precomputed hashed inputs"):
        fill_welm_oe_hash_inputs(
            input_ids,
            hashed_out,
            forward_batch,
            oe_grams=(2, 3, 4),
            oe_vocab_sizes=(257, 263, 269),
            vocab_size=32017,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", [torch.int64, torch.int32])
def test_welm_oe_segment_hash_dynamic_prefixes_and_config(input_dtype):
    device = "cuda"
    vocab_size = 32017
    oe_grams = (2, 3, 4)
    oe_vocab_sizes = (257, 263, 269)

    input_ids_cpu = torch.tensor([3, 4, 5, 20, 30, 40], dtype=torch.int64)
    extend_start_loc_cpu = [0, 3]
    extend_seq_lens_cpu = [3, 3]
    prefixes = [
        2,
        10,
        1,
        0,
        0,
        0,
    ]
    expected_hash = _reference_segment_hash(
        input_ids_cpu,
        extend_start_loc_cpu,
        extend_seq_lens_cpu,
        prefixes,
        oe_grams,
        oe_vocab_sizes,
        vocab_size,
    )

    input_ids = input_ids_cpu.to(device=device, dtype=input_dtype)
    extend_start_loc = torch.tensor(
        extend_start_loc_cpu, dtype=torch.int32, device=device
    )
    extend_seq_lens = torch.tensor(
        extend_seq_lens_cpu, dtype=torch.int32, device=device
    )
    hashed_out = torch.empty(
        (len(oe_grams), input_ids.numel()), dtype=torch.int64, device=device
    )

    welm_oe_hash_segments_from_prefixes_cuda(
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        prefixes,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        vocab_size,
    )
    torch.cuda.synchronize()

    assert torch.equal(hashed_out.cpu(), expected_hash)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("input_dtype", [torch.int64, torch.int32])
def test_welm_oe_draft_extend_after_verify_hash_from_history(input_dtype):
    device = "cuda"
    vocab_size = 32017
    oe_grams = (2, 3, 4)
    oe_vocab_sizes = (257, 263, 269)
    draft_token_num = 4

    input_ids_cpu = torch.tensor([31, 32, 33, 34, 41, 42, 43, 44], dtype=torch.int64)
    accepted_cpu = torch.tensor(
        [
            [20, 21, -1, -1],
            [-1, 30, -1, -1],
        ],
        dtype=torch.int64,
    )
    accept_lens_cpu = [2, 1]
    entry_history_cpu = torch.tensor(
        [
            [10, 11, 12, 20],
            [20, 21, 22, 30],
        ],
        dtype=torch.int64,
    )
    expected_hash, expected_history = (
        _reference_draft_extend_after_verify_hash_from_history(
            input_ids_cpu,
            accepted_cpu.tolist(),
            accept_lens_cpu,
            entry_history_cpu,
            oe_grams,
            oe_vocab_sizes,
            vocab_size,
            draft_token_num,
        )
    )

    input_ids = input_ids_cpu.to(device=device, dtype=input_dtype)
    accepted = accepted_cpu.to(device=device, dtype=input_dtype)
    accept_lens = torch.tensor(accept_lens_cpu, dtype=torch.int64, device=device)
    entry_history = entry_history_cpu.to(device=device)
    hashed_out = torch.empty(
        (len(oe_grams), input_ids.numel()), dtype=torch.int64, device=device
    )
    next_history = torch.empty_like(entry_history)

    welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda(
        input_ids,
        accepted,
        accept_lens,
        entry_history,
        oe_grams,
        oe_vocab_sizes,
        hashed_out,
        next_history,
        vocab_size,
        draft_token_num,
    )
    torch.cuda.synchronize()

    assert torch.equal(hashed_out.cpu(), expected_hash)
    assert torch.equal(next_history.cpu(), expected_history)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_welm_oe_hash_warmup_uses_model_width():
    warmup_welm_oe_hash_kernel(
        "cuda",
        history_width=3,
        oe_grams=(2, 3, 4),
        oe_vocab_sizes=(1, 1, 1),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
