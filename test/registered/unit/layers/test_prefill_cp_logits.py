"""Contracts for routing persistent-token prefill CP hidden rows to the LM head."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.context_parallel import (
    build_cp_prefill_split_spec,
    contract_cp_prefill_runtime_to_last_q,
    materialize_cp_prefill_runtime_layout,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.prefill_cp_logits import route_cp_prefill_hidden_states
from sglang.srt.models import welmv4 as welmv4_model
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _Group:
    def __init__(self, *, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank


class _GlobalTPGroup(_Group):
    def __init__(self, output: torch.Tensor, *, rank: int, world_size: int = 8):
        super().__init__(world_size=world_size, rank=rank)
        self.output = output
        self.calls = []

    def all_gatherv(self, tensors, sizes):
        self.calls.append((tensors, tuple(sizes)))
        return [self.output]


def _runtime(*, cp_rank: int, owner_rotation: int, contracted: bool):
    spec = build_cp_prefill_split_spec(
        extend_start=16,
        extend_len=32,
        cp_size=4,
        page_size=4,
        owner_rotation=owner_rotation,
    )
    out_cache_loc = torch.zeros(spec.extend_len, dtype=torch.int64)
    for block in spec.local_blocks(cp_rank):
        relative_start = block.logical_start - spec.extend_start
        out_cache_loc[relative_start : relative_start + block.token_count] = (
            torch.arange(block.token_count) + 64 + block.logical_start
        )
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=cp_rank,
        input_ids=torch.arange(spec.extend_len),
        positions=torch.arange(16, 48, dtype=torch.int32),
        out_cache_loc=out_cache_loc,
    )
    return contract_cp_prefill_runtime_to_last_q(runtime) if contracted else runtime


def _runtime_from_spec(spec, cp_rank: int, *, contracted: bool = False):
    out_cache_loc = torch.zeros(spec.extend_len, dtype=torch.int64)
    for block in spec.local_blocks(cp_rank):
        relative_start = block.logical_start - spec.extend_start
        out_cache_loc[relative_start : relative_start + block.token_count] = (
            torch.arange(block.token_count) + 64 + block.logical_start
        )
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=cp_rank,
        input_ids=torch.arange(spec.extend_len),
        positions=torch.arange(
            spec.extend_start,
            spec.extend_start + spec.extend_len,
            dtype=torch.int32,
        ),
        out_cache_loc=out_cache_loc,
    )
    return contract_cp_prefill_runtime_to_last_q(runtime) if contracted else runtime


def _rank_packed_logical_rows(spec, logical_start: int, token_count: int):
    logical_end = logical_start + token_count
    parts = []
    for cp_rank in range(len(spec.per_rank_tokens)):
        for block in spec.local_blocks(cp_rank):
            start = max(block.logical_start, logical_start)
            end = min(block.logical_start + block.token_count, logical_end)
            if start < end:
                parts.append(torch.arange(start, end, dtype=torch.float32).view(-1, 1))
    return torch.cat(parts) if parts else torch.empty((0, 1))


@pytest.mark.parametrize("owner_rotation", range(4))
@pytest.mark.parametrize("contracted", [False, True])
def test_routes_one_final_owner_row_to_every_global_tp_rank(
    owner_rotation: int,
    contracted: bool,
):
    for global_rank in range(8):
        cp_rank, lane = divmod(global_rank, 2)
        runtime = _runtime(
            cp_rank=cp_rank,
            owner_rotation=owner_rotation,
            contracted=contracted,
        )
        hidden = runtime.local_logical_indices.to(torch.float32).view(-1, 1)
        final_logical = runtime.spec.extend_start + runtime.spec.extend_len - 1
        final_owner = runtime.spec.blocks[-1].owner_rank
        global_group = _GlobalTPGroup(
            torch.tensor([[float(final_logical)]]), rank=global_rank
        )

        routed = route_cp_prefill_hidden_states(
            hidden,
            runtime,
            logical_start=final_logical,
            token_count=1,
            global_tp_group=global_group,
            attn_tp_group=_Group(world_size=2, rank=lane),
        )

        assert routed.tolist() == [[float(final_logical)]]
        sent, sizes = global_group.calls[0]
        expected_sizes = tuple(
            1 if rank == 2 * final_owner else 0 for rank in range(8)
        )
        assert sizes == expected_sizes
        if cp_rank == final_owner and lane == 0:
            assert sent[0].tolist() == [[float(final_logical)]]
        else:
            assert sent[0].shape == (0, 1)


def test_restores_rotated_zigzag_chunk_to_logical_order():
    for global_rank in range(8):
        cp_rank, lane = divmod(global_rank, 2)
        runtime = _runtime(cp_rank=cp_rank, owner_rotation=2, contracted=False)
        hidden = runtime.local_logical_indices.to(torch.float32).view(-1, 1)
        packed = _rank_packed_logical_rows(
            runtime.spec,
            runtime.spec.extend_start,
            runtime.spec.extend_len,
        )
        global_group = _GlobalTPGroup(packed, rank=global_rank)

        routed = route_cp_prefill_hidden_states(
            hidden,
            runtime,
            logical_start=runtime.spec.extend_start,
            token_count=runtime.spec.extend_len,
            global_tp_group=global_group,
            attn_tp_group=_Group(world_size=2, rank=lane),
        )

        assert torch.equal(
            routed[:, 0],
            torch.arange(16, 48, dtype=torch.float32),
        )
        sent, sizes = global_group.calls[0]
        assert sizes == tuple(
            count
            for cp_count in runtime.spec.per_rank_tokens
            for count in (cp_count, 0)
        )
        if lane == 0:
            assert torch.equal(sent[0], hidden)
        else:
            assert sent[0].shape == (0, 1)


def test_rejects_global_tp_rank_order_mismatch_before_communication():
    runtime = _runtime(cp_rank=1, owner_rotation=0, contracted=False)
    hidden = runtime.local_logical_indices.to(torch.float32).view(-1, 1)
    global_group = _GlobalTPGroup(torch.empty((1, 1)), rank=0)

    with pytest.raises(RuntimeError, match="global-TP rank"):
        route_cp_prefill_hidden_states(
            hidden,
            runtime,
            logical_start=47,
            token_count=1,
            global_tp_group=global_group,
            attn_tp_group=_Group(world_size=2, rank=0),
        )

    assert global_group.calls == []


def test_routes_cp2_attntp2_hidden_to_tp4():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=8,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    final_logical = spec.extend_start + spec.extend_len - 1
    final_owner = spec.blocks[-1].owner_rank
    for global_rank in range(4):
        cp_rank, lane = divmod(global_rank, 2)
        runtime = _runtime_from_spec(spec, cp_rank=cp_rank)
        hidden = runtime.local_logical_indices.to(torch.float32).view(-1, 1)
        global_group = _GlobalTPGroup(
            torch.tensor([[float(final_logical)]]),
            rank=global_rank,
            world_size=4,
        )

        routed = route_cp_prefill_hidden_states(
            hidden,
            runtime,
            logical_start=final_logical,
            token_count=1,
            global_tp_group=global_group,
            attn_tp_group=_Group(world_size=2, rank=lane),
        )

        assert routed.tolist() == [[float(final_logical)]]
        sent, sizes = global_group.calls[0]
        assert sizes == tuple(
            1 if rank == 2 * final_owner else 0 for rank in range(4)
        )
        if cp_rank == final_owner and lane == 0:
            assert sent[0].tolist() == [[float(final_logical)]]
        else:
            assert sent[0].shape == (0, 1)


def test_rejects_global_tp_size_not_twice_cp_size_before_communication():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=8,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    runtime = _runtime_from_spec(spec, cp_rank=0)
    hidden = runtime.local_logical_indices.to(torch.float32).view(-1, 1)
    global_group = _GlobalTPGroup(
        torch.empty((1, 1)), rank=0, world_size=8
    )

    with pytest.raises(RuntimeError, match="twice the CP size"):
        route_cp_prefill_hidden_states(
            hidden,
            runtime,
            logical_start=7,
            token_count=1,
            global_tp_group=global_group,
            attn_tp_group=_Group(world_size=2, rank=0),
        )

    assert global_group.calls == []


def test_routes_unaligned_prefix_and_uneven_owner_rows():
    spec = build_cp_prefill_split_spec(
        extend_start=5,
        extend_len=23,
        cp_size=4,
        page_size=4,
        owner_rotation=3,
        leading_page_owner=2,
    )
    assert len(set(spec.per_rank_tokens)) > 1

    for global_rank in range(8):
        cp_rank, lane = divmod(global_rank, 2)
        runtime = _runtime_from_spec(spec, cp_rank)
        hidden = runtime.local_logical_indices.to(torch.float32).view(-1, 1)
        packed = _rank_packed_logical_rows(
            spec,
            spec.extend_start,
            spec.extend_len,
        )
        global_group = _GlobalTPGroup(packed, rank=global_rank)

        routed = route_cp_prefill_hidden_states(
            hidden,
            runtime,
            logical_start=spec.extend_start,
            token_count=spec.extend_len,
            global_tp_group=global_group,
            attn_tp_group=_Group(world_size=2, rank=lane),
        )

        assert torch.equal(
            routed[:, 0],
            torch.arange(5, 28, dtype=torch.float32),
        )
        assert global_group.calls[0][1] == tuple(
            count
            for cp_count in spec.per_rank_tokens
            for count in (cp_count, 0)
        )


def test_logits_processor_accepts_prepruned_cp_last_hidden():
    class _ExtendMode:
        @staticmethod
        def is_decode_or_idle():
            return False

        @staticmethod
        def is_target_verify():
            return False

        @staticmethod
        def is_draft_extend_v2():
            return False

        @staticmethod
        def is_extend():
            return True

    hidden = torch.tensor([[1.0, 2.0]])
    metadata = SimpleNamespace(
        forward_mode=_ExtendMode(),
        welm_kv_mirror_contracted=False,
        prefill_cp_pruned=True,
        extend_return_logprob=False,
        padded_static_len=-1,
        extend_seq_lens=torch.tensor([32]),
    )
    processor = object.__new__(LogitsProcessor)

    pruned = processor._get_pruned_states(
        hidden,
        None,
        None,
        metadata,
    )

    assert pruned[0] is hidden
    assert pruned[3] is None
    assert pruned[4] is None


class _CaptureLast:
    @staticmethod
    def is_full():
        return False

    @staticmethod
    def need_capture():
        return False


class _CaptureFull:
    @staticmethod
    def is_full():
        return True


class _CaptureStoredLast:
    @staticmethod
    def is_full():
        return False

    @staticmethod
    def is_last():
        return True

    @staticmethod
    def need_capture():
        return True


@pytest.mark.parametrize(
    ("batch_overrides", "capture_mode", "message"),
    [
        ({"multi_item_delimiter_indices": [torch.tensor([1])]}, _CaptureLast(), "multi-item"),
        ({"multi_item_delimiter_indices": None}, _CaptureFull(), "full hidden-state"),
    ],
)
def test_welm_unsupported_normal_logits_modes_fail_before_routing(
    monkeypatch,
    batch_overrides,
    capture_mode,
    message,
):
    runtime = _runtime(cp_rank=0, owner_rotation=0, contracted=False)
    metadata = SimpleNamespace(
        extend_return_logprob=False,
        capture_hidden_mode=capture_mode,
    )
    monkeypatch.setattr(
        welmv4_model.LogitsMetadata,
        "from_forward_batch",
        lambda _batch: metadata,
    )
    monkeypatch.setattr(
        welmv4_model,
        "route_cp_prefill_hidden_states",
        lambda *_args, **_kwargs: pytest.fail("unsupported mode must fail first"),
    )
    batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=runtime,
        **batch_overrides,
    )

    with pytest.raises(NotImplementedError, match=message):
        welmv4_model._welm_prepare_cp_prefill_logits_states(
            runtime.local_logical_indices.to(torch.float32).view(-1, 1),
            None,
            batch,
        )


def test_welm_routes_final_hidden_and_aux_before_logits(monkeypatch):
    runtime = _runtime(cp_rank=0, owner_rotation=0, contracted=False)
    hidden = runtime.local_logical_indices.to(torch.float32).view(-1, 1)
    aux = [hidden + 100]
    metadata = SimpleNamespace(
        extend_return_logprob=False,
        capture_hidden_mode=_CaptureLast(),
        prefill_cp_pruned=False,
    )
    monkeypatch.setattr(
        welmv4_model.LogitsMetadata,
        "from_forward_batch",
        lambda _batch: metadata,
    )
    calls = []

    def fake_route(tensor, passed_runtime, *, logical_start, token_count):
        calls.append((tensor, passed_runtime, logical_start, token_count))
        return tensor.new_tensor([[float(logical_start)]])

    monkeypatch.setattr(
        welmv4_model, "route_cp_prefill_hidden_states", fake_route
    )

    prepared = welmv4_model._welm_prepare_cp_prefill_logits_states(
        hidden,
        aux,
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=runtime,
            multi_item_delimiter_indices=None,
        ),
    )

    assert prepared.hidden_states.tolist() == [[47.0]]
    assert prepared.aux_hidden_states[0].tolist() == [[47.0]]
    assert prepared.logits_metadata is metadata
    assert prepared.chunk_states_loader is None
    assert metadata.prefill_cp_pruned
    assert len(calls) == 2
    assert all(call[1] is runtime for call in calls)
    assert all(call[2:] == (47, 1) for call in calls)


def test_welm_no_q_chunk_skips_logits_routing(monkeypatch):
    runtime = contract_cp_prefill_runtime_to_last_q(
        _runtime(cp_rank=0, owner_rotation=0, contracted=False),
        has_active_q=False,
    )
    metadata = SimpleNamespace(
        extend_return_logprob=False,
        capture_hidden_mode=_CaptureLast(),
        prefill_cp_pruned=False,
    )
    monkeypatch.setattr(
        welmv4_model.LogitsMetadata,
        "from_forward_batch",
        lambda _batch: metadata,
    )
    monkeypatch.setattr(
        welmv4_model,
        "route_cp_prefill_hidden_states",
        lambda *_args, **_kwargs: pytest.fail("no-Q chunk must not route hidden rows"),
    )
    empty = torch.empty((0, 1))

    prepared = welmv4_model._welm_prepare_cp_prefill_logits_states(
        empty,
        None,
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=runtime,
            multi_item_delimiter_indices=None,
        ),
    )

    assert prepared.hidden_states is empty
    assert prepared.aux_hidden_states is None
    assert prepared.logits_metadata is metadata
    assert prepared.chunk_states_loader is None
    assert metadata.prefill_cp_pruned


def test_welm_prompt_logprobs_build_logical_chunk_loader(monkeypatch):
    spec = build_cp_prefill_split_spec(
        extend_start=5,
        extend_len=23,
        cp_size=4,
        page_size=4,
        owner_rotation=3,
        leading_page_owner=2,
    )
    runtime = _runtime_from_spec(spec, cp_rank=2)
    hidden = runtime.local_logical_indices.to(torch.float32).view(-1, 1)
    metadata = SimpleNamespace(
        extend_return_logprob=True,
        capture_hidden_mode=_CaptureStoredLast(),
        extend_seq_lens_cpu=[23],
        extend_logprob_start_lens_cpu=[5],
        prefill_cp_pruned=False,
    )
    monkeypatch.setattr(
        welmv4_model.LogitsMetadata,
        "from_forward_batch",
        lambda _batch: metadata,
    )

    calls = []

    def fake_route(tensor, passed_runtime, *, logical_start, token_count):
        calls.append((tensor, passed_runtime, logical_start, token_count))
        return torch.arange(
            logical_start,
            logical_start + token_count,
            dtype=tensor.dtype,
        ).view(-1, 1)

    monkeypatch.setattr(
        welmv4_model, "route_cp_prefill_hidden_states", fake_route
    )

    prepared = welmv4_model._welm_prepare_cp_prefill_logits_states(
        hidden,
        None,
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=runtime,
            multi_item_delimiter_indices=None,
        ),
    )

    assert prepared.hidden_states.shape == (23, 0)
    assert prepared.logits_metadata is metadata
    assert not metadata.prefill_cp_pruned
    assert prepared.hidden_states_to_store.tolist() == [[27.0]]
    assert prepared.chunk_states_loader(0, 4).tolist() == [
        [10.0],
        [11.0],
        [12.0],
        [13.0],
    ]
    assert prepared.chunk_states_loader(4, 18).tolist() == [
        [float(i)] for i in range(14, 28)
    ]
    assert [(start, count) for _, _, start, count in calls] == [
        (27, 1),
        (10, 4),
        (14, 14),
    ]


def test_streamed_logprobs_loader_matches_dense_reference():
    class _ExtendMode:
        @staticmethod
        def is_decode_or_idle():
            return False

        @staticmethod
        def is_target_verify():
            return False

        @staticmethod
        def is_draft_extend_v2():
            return False

        @staticmethod
        def is_extend():
            return True

    logits = torch.tensor(
        [
            [2.0, 1.0, -1.0],
            [0.0, 3.0, 1.0],
            [1.5, -0.5, 2.5],
            [-1.0, 0.5, 2.0],
            [4.0, 1.0, 0.0],
        ]
    )
    target_ids = torch.tensor([0, 1, 2, 1, 0])
    metadata = SimpleNamespace(
        forward_mode=_ExtendMode(),
        welm_kv_mirror_contracted=False,
        prefill_cp_pruned=False,
        extend_return_logprob=True,
        padded_static_len=-1,
        extend_seq_lens=torch.tensor([5]),
        extend_seq_lens_cpu=[5],
        extend_logprob_start_lens_cpu=[0],
        extend_logprob_pruned_lens_cpu=[5],
        extend_return_top_logprob=False,
        extend_token_ids_logprob=False,
        temp_scaled_logprobs=False,
        temperature=None,
        top_p_normalized_logprobs=False,
        top_p=None,
        extend_input_logprob_token_ids_gpu=target_ids,
        mm_input_embeds=None,
    )
    processor = object.__new__(LogitsProcessor)
    processor.logprobs_chunk_size = 2
    processor._get_logits = lambda states, _head, _metadata: states
    calls = []
    last_chunk_ref = None

    def loader(start_idx, end_idx):
        nonlocal last_chunk_ref
        if last_chunk_ref is not None:
            assert last_chunk_ref() is None
        calls.append((start_idx, end_idx))
        chunk = logits[start_idx:end_idx].clone()
        last_chunk_ref = weakref.ref(chunk)
        return chunk

    stored_hidden = torch.tensor([[99.0]])
    output = processor.forward_input_logprobs_by_chunk_loader(
        torch.empty((5, 0)),
        object(),
        metadata,
        chunk_states_loader=loader,
        hidden_states_to_store=stored_hidden,
    )

    reference = torch.log_softmax(logits, dim=-1)
    assert calls == [(0, 2), (2, 4), (4, 5)]
    assert last_chunk_ref() is None
    assert max(end - start for start, end in calls) == 2
    assert torch.allclose(
        output.input_token_logprobs,
        reference[torch.arange(5), target_ids],
    )
    assert torch.equal(output.next_token_logits, logits[-1:])
    assert output.hidden_states is stored_hidden


def test_welm_no_q_chunk_skips_lm_head_and_logits_collectives():
    metadata = SimpleNamespace(
        prefill_cp_pruned=True,
        capture_hidden_mode=_CaptureLast(),
        mm_input_embeds=None,
    )
    empty = torch.empty((0, 4), dtype=torch.bfloat16)

    output = welmv4_model._welm_compute_logits_output(
        lambda *_args, **_kwargs: pytest.fail(
            "no-Q chunk must not invoke the logits processor"
        ),
        torch.empty((0,), dtype=torch.int64),
        object(),
        welmv4_model._WelmPreparedLogits(
            hidden_states=empty,
            aux_hidden_states=None,
            logits_metadata=metadata,
        ),
        vocab_size=32,
    )

    assert output.next_token_logits.shape == (0, 32)
    assert output.next_token_logits.dtype == torch.float32
    assert output.hidden_states is None


def test_welm_split_prefill_spec_fails_without_runtime_layout():
    model = object.__new__(welmv4_model.WeLMV4MoeForCausalLM)

    with pytest.raises(NotImplementedError, match="split-prefill execution"):
        model.forward_split_prefill(
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0,), dtype=torch.int32),
            SimpleNamespace(
                attn_cp_prefill_runtime_layout=None,
                attn_cp_prefill_split_specs=(object(),),
            ),
            (0, 1),
        )
