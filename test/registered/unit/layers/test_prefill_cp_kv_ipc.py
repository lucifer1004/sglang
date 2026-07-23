from __future__ import annotations

import pytest
import torch

import sglang.jit_kernel.prefill_cp_kv_source_push as source_push
import sglang.srt.layers.attention.prefill_cp_kv_ipc as kv_ipc
from sglang.srt.layers.attention.prefill_cp_kv_ipc import (
    CPKVIPCArenaLayout,
    CPKVIPCEpochTracker,
    CPKVIPCForwardEpochSequencer,
    CPKVIPCSourcePushTransport,
)
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def test_arena_layout_separates_payload_and_cacheline_signals():
    layout = CPKVIPCArenaLayout.build(
        max_rows=262144,
        row_width=256,
        cp_size=4,
    )

    assert layout.k_offset_bytes == 0
    assert layout.v_offset_bytes == 262144 * 256 * 2
    assert layout.ready_offset_bytes >= 2 * layout.v_offset_bytes
    assert layout.consumed_offset_bytes >= layout.ready_offset_bytes + 4 * 128
    assert layout.arena_bytes % (2 * 1024 * 1024) == 0
    assert layout.signal_stride_bytes == 128


def test_arena_layout_rejects_rows_beyond_capacity():
    layout = CPKVIPCArenaLayout.build(max_rows=32, row_width=256, cp_size=4)

    layout.require_capacity(32)
    with pytest.raises(RuntimeError, match="capacity"):
        layout.require_capacity(33)


def test_arena_initialization_finishes_before_ipc_handle_export(monkeypatch):
    events = []
    arena = object()
    device = torch.device("cuda", 3)

    def fake_zeros(*args, **kwargs):
        events.append(("zeros", args, kwargs))
        return arena

    def fake_synchronize(sync_device):
        events.append(("synchronize", sync_device))

    monkeypatch.setattr(kv_ipc.torch, "zeros", fake_zeros)
    monkeypatch.setattr(kv_ipc.torch.cuda, "synchronize", fake_synchronize)

    result = kv_ipc._allocate_zeroed_cuda_arena(
        arena_bytes=4096,
        device=device,
    )

    assert result is arena
    assert events == [
        (
            "zeros",
            (4096,),
            {"dtype": torch.uint8, "device": device},
        ),
        ("synchronize", device),
    ]


def test_ipc_handle_exchange_bypasses_src0_only_message_queue(monkeypatch):
    class FakeCPGroup:
        ranks = (0, 2, 4, 6)
        cpu_group = object()

        def broadcast_object(self, obj, src=0):
            raise AssertionError("message-queue broadcast only supports src=0")

    transport = object.__new__(CPKVIPCSourcePushTransport)
    transport.cp_group = FakeCPGroup()
    transport.cp_size = 4
    transport.cp_rank = 2
    local_handle = ("local-handle",)
    calls = []

    def fake_broadcast_object_list(objects, *, src, group):
        assert group is transport.cp_group.cpu_group
        source_rank = transport.cp_group.ranks.index(src)
        if source_rank == transport.cp_rank:
            assert objects == [local_handle]
        else:
            assert objects == [None]
        objects[0] = (f"handle-{source_rank}",)
        calls.append(src)

    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )

    handles = transport._exchange_storage_handles(local_handle)

    assert handles == [(f"handle-{rank}",) for rank in range(4)]
    assert calls == list(transport.cp_group.ranks)


def test_epoch_tracker_requires_local_consumer_release_before_next_layer():
    tracker = CPKVIPCEpochTracker(cp_size=4, local_rank=2)

    first = tracker.begin(source_mask=0b1111, destination_mask=0b0100)

    assert first.epoch == 1
    assert first.previous_consumed_epoch == 0
    assert first.requires_local_release
    with pytest.raises(RuntimeError, match="release"):
        tracker.begin(source_mask=0b1111, destination_mask=0b1111)

    tracker.release(first)
    second = tracker.begin(source_mask=0b1111, destination_mask=0b1111)

    assert second.epoch == 2
    assert second.previous_consumed_epoch == 1
    assert second.previous_destination_mask == 0b0100


def test_epoch_tracker_non_consumer_can_advance_without_release():
    tracker = CPKVIPCEpochTracker(cp_size=4, local_rank=0)

    first = tracker.begin(source_mask=0b0011, destination_mask=0b1000)
    second = tracker.begin(source_mask=0b1111, destination_mask=0b0010)

    assert not first.requires_local_release
    assert second.previous_consumed_epoch == first.epoch


def test_epoch_tracker_retains_consumption_wait_across_zero_source_layer():
    tracker = CPKVIPCEpochTracker(cp_size=4, local_rank=0)

    first = tracker.begin(source_mask=0b0001, destination_mask=0b1000)
    skipped = tracker.begin(source_mask=0b0010, destination_mask=0b1000)
    resumed = tracker.begin(source_mask=0b0001, destination_mask=0b1000)

    assert not skipped.is_local_source
    assert not skipped.waits_for_previous_consumption
    assert resumed.waits_for_previous_consumption
    assert resumed.previous_consumed_epoch == first.epoch
    assert resumed.previous_destination_mask == first.destination_mask


def test_epoch_tracker_rejects_stale_or_duplicate_release():
    tracker = CPKVIPCEpochTracker(cp_size=4, local_rank=1)
    ticket = tracker.begin(source_mask=0b1111, destination_mask=0b0010)

    tracker.release(ticket)
    with pytest.raises(RuntimeError, match="already released"):
        tracker.release(ticket)


@pytest.mark.parametrize(
    ("source_mask", "destination_mask"),
    [(0, 1), (1, 0), (0b10000, 1), (1, 0b10000)],
)
def test_epoch_tracker_rejects_invalid_masks(source_mask, destination_mask):
    tracker = CPKVIPCEpochTracker(cp_size=4, local_rank=0)

    with pytest.raises(ValueError, match="mask"):
        tracker.begin(
            source_mask=source_mask,
            destination_mask=destination_mask,
        )


def test_forward_epoch_matches_after_one_rank_skips_earlier_forwards():
    always_active = CPKVIPCForwardEpochSequencer(max_calls_per_forward=64)
    newly_active = CPKVIPCForwardEpochSequencer(max_calls_per_forward=64)

    for forward_iter in range(1, 26):
        always_active.next_epoch(forward_iter=forward_iter, operation_index=0)

    assert always_active.next_epoch(
        forward_iter=26, operation_index=0
    ) == newly_active.next_epoch(
        forward_iter=26, operation_index=0
    )


def test_forward_epoch_preserves_call_order_within_one_forward():
    left = CPKVIPCForwardEpochSequencer(max_calls_per_forward=64)
    right = CPKVIPCForwardEpochSequencer(max_calls_per_forward=64)

    assert [
        left.next_epoch(forward_iter=7, operation_index=operation_index)
        for operation_index in range(3)
    ] == [
        right.next_epoch(forward_iter=7, operation_index=operation_index)
        for operation_index in range(3)
    ]


def test_forward_epoch_matches_when_one_rank_skips_an_earlier_operation():
    always_active = CPKVIPCForwardEpochSequencer(max_calls_per_forward=64)
    selectively_active = CPKVIPCForwardEpochSequencer(max_calls_per_forward=64)

    always_active.next_epoch(forward_iter=7, operation_index=0)

    assert always_active.next_epoch(
        forward_iter=7, operation_index=5
    ) == selectively_active.next_epoch(forward_iter=7, operation_index=5)


def test_forward_epoch_supports_values_beyond_uint32():
    sequencer = CPKVIPCForwardEpochSequencer(max_calls_per_forward=64)
    first_forward_beyond_uint32 = (1 << 32) // 64 + 1

    epoch = sequencer.next_epoch(
        forward_iter=first_forward_beyond_uint32,
        operation_index=0,
    )

    assert epoch == (1 << 32) + 1


def test_epoch_tracker_supports_values_beyond_uint32():
    tracker = CPKVIPCEpochTracker(cp_size=4, local_rank=0)

    ticket = tracker.begin(
        source_mask=0b0001,
        destination_mask=0b0010,
        epoch=(1 << 32) + 1,
    )

    assert ticket.epoch == (1 << 32) + 1


def test_signal_metadata_supports_values_beyond_uint32():
    source_push._validate_signal_metadata(
        signal_offset_bytes=0,
        signal_stride_bytes=128,
        epoch=(1 << 32) + 1,
    )
