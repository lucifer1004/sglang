from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.welm_mtp_draft_ngram_hash import (
    WelmMTPDraftNGramEntryHistory,
    WelmMTPDraftNGramHistory,
    launch_deferred_welm_mtp_draft_ngram_hash,
    welm_mtp_draft_ngram_hash_from_history,
)


def _hash_ngram(cur_token, history_token_fn, oe_grams, oe_vocab_sizes, vocab_size):
    mask = 0xFFFFFFFF
    values = []
    for gram, oe_vocab_size in zip(oe_grams, oe_vocab_sizes):
        value = int(cur_token) & mask
        vocab_power = int(vocab_size) & mask
        for lag in range(1, gram):
            prev = int(history_token_fn(lag)) & mask
            value = (value + prev * vocab_power) & mask
            vocab_power = (vocab_power * int(vocab_size)) & mask
        values.append(((value * 2654435761) & mask) % int(oe_vocab_size))
    return values


def _reference_draft_decode_hash_from_history(
    input_ids,
    history_state,
    parent_indices,
    oe_grams,
    oe_vocab_sizes,
    vocab_size,
    base_query_count,
    use_parent,
):
    hashed = torch.empty((len(oe_grams), input_ids.numel()), dtype=torch.int64)
    history_width = history_state.shape[1]
    next_history = torch.empty((input_ids.numel(), history_width), dtype=torch.int64)
    repeat = max(1, input_ids.numel() // max(1, base_query_count))

    for token_idx in range(input_ids.numel()):
        source_row = (
            int(parent_indices[token_idx].item())
            if use_parent
            else int(token_idx) // repeat
        )

        for col in range(history_width):
            if col + 1 == history_width:
                next_history[token_idx, col] = int(input_ids[token_idx].item())
            else:
                next_history[token_idx, col] = history_state[source_row, col + 1]

        def history_token(lag):
            col = history_width - lag
            if col < 0 or col >= history_width:
                return 0
            return int(history_state[source_row, col].item())

        hashed[:, token_idx] = torch.tensor(
            _hash_ngram(
                int(input_ids[token_idx].item()),
                history_token,
                oe_grams,
                oe_vocab_sizes,
                vocab_size,
            ),
            dtype=torch.int64,
        )

    return hashed, next_history


def _has_mk_draft_ngram_hash():
    try:
        from mk.kernels import draft_ngram_hash  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_welm_mtp_draft_ngram_hash_requires_env(monkeypatch):
    monkeypatch.delenv("SGLANG_WELM_MTP_DRAFT_NGRAM_HASH", raising=False)
    input_ids = torch.tensor([11], dtype=torch.int64, device="cuda")
    history = torch.tensor([[101, 102, 103]], dtype=torch.int64, device="cuda")
    parent_indices = torch.empty((1,), dtype=torch.int64, device="cuda")
    hashed_out = torch.empty((1, 1), dtype=torch.int64, device="cuda")
    next_history = torch.empty_like(history)

    mk_state = welm_mtp_draft_ngram_hash_from_history(
        forward_batch=SimpleNamespace(),
        input_ids=input_ids,
        history_state=history,
        parent_indices=parent_indices,
        oe_grams=(2,),
        oe_vocab_sizes=(257,),
        hashed_out=hashed_out,
        next_history_state=next_history,
        vocab_size=32017,
        base_query_count=1,
        use_parent=False,
    )

    assert mk_state is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    not _has_mk_draft_ngram_hash(), reason="mk draft ngram hash is unavailable"
)
def test_welm_mtp_draft_ngram_hash_v2_chained_history_matches_reference(monkeypatch):
    monkeypatch.setenv("SGLANG_WELM_MTP_DRAFT_NGRAM_HASH", "true")
    device = "cuda"
    vocab_size = 32017
    oe_grams = (2, 2, 3, 3)
    oe_vocab_sizes = (257, 263, 269, 271)

    entry_history_cpu = torch.tensor(
        [
            [101, 102, 103],
            [201, 202, 203],
            [301, 302, 303],
        ],
        dtype=torch.int64,
    )
    step0_input_cpu = torch.tensor([11, 12, 21, 22, 31, 32], dtype=torch.int64)
    base_query_count = entry_history_cpu.shape[0]
    parent_step0_cpu = torch.empty((step0_input_cpu.numel(),), dtype=torch.int64)
    expected_hash0, expected_history0 = _reference_draft_decode_hash_from_history(
        step0_input_cpu,
        entry_history_cpu,
        parent_step0_cpu,
        oe_grams,
        oe_vocab_sizes,
        vocab_size,
        base_query_count,
        use_parent=False,
    )

    entry_history = entry_history_cpu.to(device=device)
    entry_ngram_history = WelmMTPDraftNGramEntryHistory(
        prev_input_ids=entry_history[:, -1],
        prev_prev_input_ids=[int(x) for x in entry_history_cpu[:, -2].tolist()],
    )
    step0_input = step0_input_cpu.to(device=device)
    parent_step0 = torch.empty_like(step0_input)
    scratch_len = step0_input.numel()
    forward_batch = SimpleNamespace(
        welm_mtp_skip_draft_proposal_build=True,
        welm_mtp_oe_prev_input_ids=torch.empty(
            (scratch_len,), dtype=torch.int64, device=device
        ),
        welm_mtp_oe_prev_prev_input_ids=torch.empty(
            (scratch_len,), dtype=torch.int64, device=device
        ),
        welm_mtp_oe_output_prev_input_ids=torch.empty(
            (scratch_len,), dtype=torch.int64, device=device
        ),
        welm_mtp_oe_parent_scratch=torch.empty(
            (scratch_len,), dtype=torch.int64, device=device
        ),
        welm_mtp_oe_hash_out_batch_major=torch.empty(
            (scratch_len, len(oe_grams)), dtype=torch.int64, device=device
        ),
    )
    hash0 = forward_batch.welm_mtp_oe_hash_out_batch_major[:scratch_len].t()
    assert not hash0.is_contiguous()

    state0 = welm_mtp_draft_ngram_hash_from_history(
        forward_batch=forward_batch,
        input_ids=step0_input,
        history_state=entry_ngram_history,
        parent_indices=parent_step0,
        oe_grams=oe_grams,
        oe_vocab_sizes=oe_vocab_sizes,
        hashed_out=hash0,
        next_history_state=None,
        vocab_size=vocab_size,
        base_query_count=base_query_count,
        use_parent=False,
        prev_input_ids_scratch=forward_batch.welm_mtp_oe_prev_input_ids,
        prev_prev_input_ids_scratch=forward_batch.welm_mtp_oe_prev_prev_input_ids,
        output_ids_scratch=forward_batch.welm_mtp_oe_hash_out_batch_major,
        output_prev_input_ids_scratch=(
            forward_batch.welm_mtp_oe_output_prev_input_ids
        ),
        source_indices_scratch=forward_batch.welm_mtp_oe_parent_scratch,
    )
    assert forward_batch.welm_mtp_draft_ngram_prepared_launch is not None
    assert launch_deferred_welm_mtp_draft_ngram_hash(forward_batch)
    assert forward_batch.welm_mtp_draft_ngram_prepared_launch is None
    torch.cuda.synchronize()

    assert isinstance(state0, WelmMTPDraftNGramHistory)
    assert torch.equal(hash0.cpu(), expected_hash0)
    assert torch.equal(state0.prev_input_ids.cpu(), step0_input_cpu)
    assert torch.equal(state0.prev_prev_input_ids.cpu(), expected_history0[:, -2])
    assert torch.equal(
        forward_batch.welm_mtp_oe_output_prev_input_ids.cpu(),
        expected_history0[:, -2],
    )

    step1_input_cpu = torch.tensor([13, 24, 35], dtype=torch.int64)
    parent_step1_cpu = torch.tensor([1, 2, 5], dtype=torch.int64)
    expected_hash1, expected_history1 = _reference_draft_decode_hash_from_history(
        step1_input_cpu,
        expected_history0,
        parent_step1_cpu,
        oe_grams,
        oe_vocab_sizes,
        vocab_size,
        base_query_count,
        use_parent=True,
    )

    step1_input = step1_input_cpu.to(device=device)
    parent_step1 = parent_step1_cpu.to(device=device)
    hash1 = forward_batch.welm_mtp_oe_hash_out_batch_major[
        : step1_input.numel()
    ].t()
    assert not hash1.is_contiguous()
    state1 = welm_mtp_draft_ngram_hash_from_history(
        forward_batch=forward_batch,
        input_ids=step1_input,
        history_state=state0,
        parent_indices=parent_step1,
        oe_grams=oe_grams,
        oe_vocab_sizes=oe_vocab_sizes,
        hashed_out=hash1,
        next_history_state=None,
        vocab_size=vocab_size,
        base_query_count=base_query_count,
        use_parent=True,
        prev_input_ids_scratch=forward_batch.welm_mtp_oe_prev_input_ids[
            : step1_input.numel()
        ],
        prev_prev_input_ids_scratch=forward_batch.welm_mtp_oe_prev_prev_input_ids[
            : step1_input.numel()
        ],
        output_ids_scratch=forward_batch.welm_mtp_oe_hash_out_batch_major[
            : step1_input.numel()
        ],
        output_prev_input_ids_scratch=(
            forward_batch.welm_mtp_oe_output_prev_input_ids[: step1_input.numel()]
        ),
        source_indices_scratch=forward_batch.welm_mtp_oe_parent_scratch[
            : step1_input.numel()
        ],
    )
    assert forward_batch.welm_mtp_draft_ngram_prepared_launch is not None
    assert launch_deferred_welm_mtp_draft_ngram_hash(forward_batch)
    assert forward_batch.welm_mtp_draft_ngram_prepared_launch is None
    torch.cuda.synchronize()

    assert isinstance(state1, WelmMTPDraftNGramHistory)
    assert torch.equal(hash1.cpu(), expected_hash1)
    assert torch.equal(state1.prev_input_ids.cpu(), step1_input_cpu)
    assert torch.equal(state1.prev_prev_input_ids.cpu(), expected_history1[:, -2])
    assert torch.equal(
        forward_batch.welm_mtp_oe_output_prev_input_ids[: step1_input.numel()].cpu(),
        expected_history1[:, -2],
    )
