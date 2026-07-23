from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.jit_kernel.prefill_cp_kv_source_push import (
    _jit_source_push_module,
    publish_epoch,
    source_push_indexed,
    wait_ready,
)
from sglang.srt.layers.attention.cp_sharded_kv import (
    CPKVSourcePushSegmentPlan,
    CPPrefillKVSourcePushPlan,
)


_SIGNAL_STRIDE_BYTES = 128
_ARENA_ALIGNMENT_BYTES = 2 * 1024 * 1024
_SUPPORTED_CP_SIZE = 4
_SUPPORTED_ROW_WIDTH = 256
_ROWS_PER_BLOCK = 16
_NUM_THREADS = 128
_INT64_MAX = (1 << 63) - 1


def _align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("arena alignment inputs must be positive")
    return (value + alignment - 1) // alignment * alignment


def _allocate_zeroed_cuda_arena(
    *, arena_bytes: int, device: torch.device
) -> torch.Tensor:
    arena = torch.zeros(arena_bytes, dtype=torch.uint8, device=device)
    # Peer mappings become visible immediately after handle export. Complete
    # the one-time memset first so it cannot race a peer's first payload write.
    torch.cuda.synchronize(device)
    return arena


@dataclass(frozen=True)
class CPKVIPCArenaLayout:
    max_rows: int
    row_width: int
    cp_size: int
    row_bytes: int
    k_offset_bytes: int
    v_offset_bytes: int
    ready_offset_bytes: int
    consumed_offset_bytes: int
    signal_stride_bytes: int
    arena_bytes: int

    @classmethod
    def build(
        cls, *, max_rows: int, row_width: int, cp_size: int
    ) -> "CPKVIPCArenaLayout":
        if max_rows <= 0 or row_width <= 0 or cp_size <= 0:
            raise ValueError("arena dimensions must be positive")
        row_bytes = row_width * torch.bfloat16.itemsize
        k_offset_bytes = 0
        v_offset_bytes = max_rows * row_bytes
        ready_offset_bytes = _align_up(
            v_offset_bytes + max_rows * row_bytes, 16
        )
        consumed_offset_bytes = ready_offset_bytes + cp_size * _SIGNAL_STRIDE_BYTES
        arena_bytes = _align_up(
            consumed_offset_bytes + cp_size * _SIGNAL_STRIDE_BYTES,
            _ARENA_ALIGNMENT_BYTES,
        )
        return cls(
            max_rows=max_rows,
            row_width=row_width,
            cp_size=cp_size,
            row_bytes=row_bytes,
            k_offset_bytes=k_offset_bytes,
            v_offset_bytes=v_offset_bytes,
            ready_offset_bytes=ready_offset_bytes,
            consumed_offset_bytes=consumed_offset_bytes,
            signal_stride_bytes=_SIGNAL_STRIDE_BYTES,
            arena_bytes=arena_bytes,
        )

    def require_capacity(self, rows: int) -> None:
        if rows < 0 or rows > self.max_rows:
            raise RuntimeError(
                f"K/V IPC arena capacity is {self.max_rows} rows, requested {rows}"
            )


@dataclass(frozen=True)
class CPKVIPCTransferTicket:
    epoch: int
    source_mask: int
    destination_mask: int
    previous_consumed_epoch: int
    previous_destination_mask: int
    is_local_source: bool
    requires_local_release: bool
    waits_for_previous_consumption: bool


class CPKVIPCForwardEpochSequencer:
    """Derive IPC epochs from stable forward and attention operation ids."""

    def __init__(self, *, max_calls_per_forward: int):
        if max_calls_per_forward <= 0:
            raise ValueError("max_calls_per_forward must be positive")
        self._max_calls_per_forward = max_calls_per_forward
        self._epoch_stride = 1 << (max_calls_per_forward - 1).bit_length()
        self._forward_iter: int | None = None
        self._last_operation_index = -1

    def next_epoch(self, *, forward_iter: int, operation_index: int) -> int:
        if forward_iter <= 0:
            raise ValueError("forward_iter must be positive")
        if operation_index < 0 or operation_index >= self._max_calls_per_forward:
            raise ValueError("K/V IPC operation index is outside forward capacity")
        if self._forward_iter is None or forward_iter > self._forward_iter:
            self._forward_iter = forward_iter
            self._last_operation_index = -1
        elif forward_iter < self._forward_iter:
            raise RuntimeError("K/V IPC forward_iter moved backwards")

        if operation_index <= self._last_operation_index:
            raise RuntimeError(
                "K/V IPC operation ids must increase within one forward"
            )
        epoch = (
            (forward_iter - 1) * self._epoch_stride + operation_index + 1
        )
        if epoch > _INT64_MAX:
            raise RuntimeError("K/V IPC epoch exhausted int64 capacity")
        self._last_operation_index = operation_index
        return epoch


class CPKVIPCEpochTracker:
    """Host-side sequencing for one reusable source-push arena slot."""

    def __init__(self, *, cp_size: int, local_rank: int):
        if cp_size <= 0 or cp_size > 31:
            raise ValueError("CP size must fit in a positive int32 rank mask")
        if local_rank < 0 or local_rank >= cp_size:
            raise ValueError("local rank is outside the CP group")
        self._cp_size = cp_size
        self._local_rank = local_rank
        self._next_epoch = 1
        self._pending_source_consumption: CPKVIPCTransferTicket | None = None
        self._pending_local_release: CPKVIPCTransferTicket | None = None
        self._last_released_epoch = 0

    def _validate_mask(self, name: str, mask: int) -> None:
        valid_mask = (1 << self._cp_size) - 1
        if mask <= 0 or mask & ~valid_mask:
            raise ValueError(f"{name} mask must select ranks inside the CP group")

    def begin(
        self,
        *,
        source_mask: int,
        destination_mask: int,
        epoch: int | None = None,
    ) -> CPKVIPCTransferTicket:
        if self._pending_local_release is not None:
            raise RuntimeError(
                "the previous local K/V IPC consumer must release its arena lease"
            )
        self._validate_mask("source", source_mask)
        self._validate_mask("destination", destination_mask)
        if epoch is None:
            epoch = self._next_epoch
        elif epoch < self._next_epoch:
            raise RuntimeError("K/V IPC epoch must increase monotonically")
        if epoch > _INT64_MAX:
            raise RuntimeError("K/V IPC epoch exhausted int64 capacity")

        previous_source = self._pending_source_consumption
        local_bit = 1 << self._local_rank
        is_local_source = bool(source_mask & local_bit)
        ticket = CPKVIPCTransferTicket(
            epoch=epoch,
            source_mask=source_mask,
            destination_mask=destination_mask,
            previous_consumed_epoch=(
                previous_source.epoch if previous_source is not None else 0
            ),
            previous_destination_mask=(
                previous_source.destination_mask
                if previous_source is not None
                else 0
            ),
            is_local_source=is_local_source,
            requires_local_release=bool(destination_mask & local_bit),
            waits_for_previous_consumption=bool(
                is_local_source and previous_source is not None
            ),
        )
        self._next_epoch = epoch + 1
        if is_local_source:
            self._pending_source_consumption = ticket
        if ticket.requires_local_release:
            self._pending_local_release = ticket
        return ticket

    def release(self, ticket: CPKVIPCTransferTicket) -> None:
        if ticket.epoch == self._last_released_epoch:
            raise RuntimeError("K/V IPC arena lease was already released")
        if self._pending_local_release is None:
            raise RuntimeError("no local K/V IPC arena lease is pending")
        if self._pending_local_release is not ticket:
            raise RuntimeError("stale K/V IPC arena lease cannot be released")
        self._pending_local_release = None
        self._last_released_epoch = ticket.epoch


class CPKVIPCArenaLease:
    def __init__(
        self,
        *,
        transport: "CPKVIPCSourcePushTransport",
        ticket: CPKVIPCTransferTicket,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        self._transport = transport
        self._ticket = ticket
        self.key = key
        self.value = value
        self._released = False

    def release(self) -> None:
        if self._released:
            raise RuntimeError("K/V IPC arena lease was already released")
        self._transport.release(self._ticket)
        self._released = True


class CPKVIPCSourcePushTransport:
    """Single-slot CP4 CUDA IPC transport for final-logical Prefill K/V."""

    def __init__(
        self,
        *,
        cp_group,
        device: torch.device | str,
        max_rows: int,
        row_width: int,
        max_calls_per_forward: int = 256,
    ):
        self.cp_group = cp_group
        self.cp_size = int(cp_group.world_size)
        self.cp_rank = int(cp_group.rank_in_group)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Prefill CP K/V IPC requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.cp_size != _SUPPORTED_CP_SIZE:
            raise ValueError(
                f"Prefill CP K/V IPC requires CP4, got CP{self.cp_size}"
            )
        if row_width != _SUPPORTED_ROW_WIDTH:
            raise ValueError(
                "Prefill CP K/V IPC requires one BF16 K/V row of width 256"
            )
        self.layout = CPKVIPCArenaLayout.build(
            max_rows=max_rows,
            row_width=row_width,
            cp_size=self.cp_size,
        )
        self._arena = _allocate_zeroed_cuda_arena(
            arena_bytes=self.layout.arena_bytes,
            device=self.device,
        )
        if self._arena.data_ptr() % 16:
            raise RuntimeError("K/V IPC arena is not 16-byte aligned")
        self._key = self._payload_view(self.layout.k_offset_bytes)
        self._value = self._payload_view(self.layout.v_offset_bytes)
        self._peer_tensors = self._open_peer_arenas()
        self._peer_bases = torch.tensor(
            [tensor.data_ptr() for tensor in self._peer_tensors],
            dtype=torch.int64,
            device="cpu",
        )
        self._completion = torch.zeros(
            1, dtype=torch.int32, device=self.device
        )
        self._epochs = CPKVIPCEpochTracker(
            cp_size=self.cp_size,
            local_rank=self.cp_rank,
        )
        self._forward_epochs = CPKVIPCForwardEpochSequencer(
            max_calls_per_forward=max_calls_per_forward
        )
        self._load_jit_across_group()

    def _payload_view(self, offset_bytes: int) -> torch.Tensor:
        payload_bytes = self.layout.max_rows * self.layout.row_bytes
        return (
            self._arena.narrow(0, offset_bytes, payload_bytes)
            .view(torch.bfloat16)
            .view(self.layout.max_rows, self.layout.row_width)
        )

    def _exchange_storage_handles(self, local_handle) -> list[tuple]:
        handles = []
        for source_rank in range(self.cp_size):
            objects = [local_handle if source_rank == self.cp_rank else None]
            # GroupCoordinator may route object broadcasts through a src-0-only
            # message queue. IPC handles must be broadcast by every CP rank.
            torch.distributed.broadcast_object_list(
                objects,
                src=self.cp_group.ranks[source_rank],
                group=self.cp_group.cpu_group,
            )
            received = objects[0]
            if received is None:
                raise RuntimeError("K/V IPC handle exchange returned an empty handle")
            handles.append(tuple(received))
        return handles

    def _open_peer_arenas(self) -> list[torch.Tensor]:
        local_handle = tuple(self._arena.untyped_storage()._share_cuda_())
        handles = self._exchange_storage_handles(local_handle)
        peer_tensors = []
        local_device_index = self.device.index
        if local_device_index is None:
            local_device_index = torch.cuda.current_device()
        for peer_rank, handle in enumerate(handles):
            if peer_rank == self.cp_rank:
                peer_tensors.append(self._arena)
                continue
            source_device_index = int(handle[0])
            if not torch.cuda.can_device_access_peer(
                local_device_index, source_device_index
            ):
                raise RuntimeError(
                    f"CUDA device {local_device_index} cannot access CP peer "
                    f"device {source_device_index}"
                )
            redirected_handle = (local_device_index,) + handle[1:]
            storage = torch.UntypedStorage._new_shared_cuda(*redirected_handle)
            peer = torch.empty(
                0, dtype=torch.uint8, device=self.device
            ).set_(
                storage,
                storage_offset=0,
                size=(self.layout.arena_bytes,),
                stride=(1,),
            )
            peer_tensors.append(peer)
        return peer_tensors

    def _load_jit_across_group(self) -> None:
        if self.cp_rank == 0:
            _jit_source_push_module()
        self.cp_group.barrier()
        if self.cp_rank != 0:
            _jit_source_push_module()
        self.cp_group.barrier()

    def _validate_source_rows(
        self,
        *,
        name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        segment: CPKVSourcePushSegmentPlan,
    ) -> None:
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise RuntimeError(f"{name} K/V must use BF16")
        if key.device != self.device or value.device != self.device:
            raise RuntimeError(f"{name} K/V must be on the IPC transport device")
        if key.ndim != 2 or value.ndim != 2:
            raise RuntimeError(f"{name} K/V must be two-dimensional rows")
        if key.shape != value.shape or key.shape[1] != self.layout.row_width:
            raise RuntimeError(f"{name} K/V shape does not match the IPC row width")
        if not key.is_contiguous() or not value.is_contiguous():
            raise RuntimeError(f"{name} K/V rows must be contiguous")
        if segment.source_rows.device != self.device:
            raise RuntimeError(f"{name} source rows must be on the IPC device")
        if segment.destination_rows.device != self.device:
            raise RuntimeError(f"{name} destination rows must be on the IPC device")
        if segment.source_rows.numel() != segment.destination_rows.numel():
            raise RuntimeError(f"{name} source and destination row counts differ")

    @staticmethod
    def _destination_mask(destination_ranks, cp_size: int) -> int:
        mask = 0
        for rank in destination_ranks:
            rank = int(rank)
            if rank < 0 or rank >= cp_size:
                raise ValueError("K/V IPC destination is outside the CP group")
            mask |= 1 << rank
        if mask == 0:
            raise ValueError("K/V IPC requires at least one destination")
        return mask

    def push(
        self,
        *,
        plan: CPPrefillKVSourcePushPlan,
        prefix_key_rows: torch.Tensor,
        prefix_value_rows: torch.Tensor,
        extend_key_rows: torch.Tensor,
        extend_value_rows: torch.Tensor,
        destination_ranks,
        forward_iter: int | None = None,
        operation_index: int | None = None,
    ) -> CPKVIPCArenaLease | None:
        self.layout.require_capacity(plan.logical_token_count)
        destination_mask = self._destination_mask(
            destination_ranks, self.cp_size
        )
        local_source_rows = (
            plan.prefix.source_rows.numel() + plan.extend.source_rows.numel()
        )
        local_is_source = bool(plan.source_mask & (1 << self.cp_rank))
        if local_is_source != (local_source_rows > 0):
            raise RuntimeError("K/V IPC source mask disagrees with local rows")

        self._validate_source_rows(
            name="Prefix",
            key=prefix_key_rows,
            value=prefix_value_rows,
            segment=plan.prefix,
        )
        self._validate_source_rows(
            name="Extend",
            key=extend_key_rows,
            value=extend_value_rows,
            segment=plan.extend,
        )
        if extend_key_rows.shape[0] != plan.extend.source_rows.numel():
            raise RuntimeError(
                "Extend K/V rows do not match the owner-local source plan"
            )
        if forward_iter is None:
            if operation_index is not None:
                raise ValueError(
                    "K/V IPC operation index requires a scheduler forward_iter"
                )
            epoch = None
        else:
            if operation_index is None:
                raise ValueError(
                    "K/V IPC production epochs require a stable operation index"
                )
            epoch = self._forward_epochs.next_epoch(
                forward_iter=forward_iter,
                operation_index=operation_index,
            )
        ticket = self._epochs.begin(
            source_mask=plan.source_mask,
            destination_mask=destination_mask,
            epoch=epoch,
        )

        if ticket.waits_for_previous_consumption:
            wait_ready(
                local_arena_base=self._arena.data_ptr(),
                source_mask=ticket.previous_destination_mask,
                signal_offset_bytes=self.layout.consumed_offset_bytes,
                signal_stride_bytes=self.layout.signal_stride_bytes,
                epoch=ticket.previous_consumed_epoch,
            )

        segments = []
        if plan.prefix.source_rows.numel() != 0:
            segments.append(
                (prefix_key_rows, prefix_value_rows, plan.prefix)
            )
        if plan.extend.source_rows.numel() != 0:
            segments.append((extend_key_rows, extend_value_rows, plan.extend))
        for segment_index, (key, value, segment) in enumerate(segments):
            source_push_indexed(
                key,
                value,
                segment.source_rows,
                segment.destination_rows,
                self._peer_bases,
                destination_mask=destination_mask,
                k_offset_bytes=self.layout.k_offset_bytes,
                v_offset_bytes=self.layout.v_offset_bytes,
                signal_offset_bytes=self.layout.ready_offset_bytes,
                signal_stride_bytes=self.layout.signal_stride_bytes,
                completion=self._completion,
                source_rank=self.cp_rank,
                epoch=ticket.epoch,
                publish_signal=segment_index == len(segments) - 1,
                rows_per_block=_ROWS_PER_BLOCK,
                num_threads=_NUM_THREADS,
            )

        if not ticket.requires_local_release:
            return None
        wait_ready(
            local_arena_base=self._arena.data_ptr(),
            source_mask=ticket.source_mask,
            signal_offset_bytes=self.layout.ready_offset_bytes,
            signal_stride_bytes=self.layout.signal_stride_bytes,
            epoch=ticket.epoch,
        )
        return CPKVIPCArenaLease(
            transport=self,
            ticket=ticket,
            key=self._key.narrow(0, 0, plan.logical_token_count),
            value=self._value.narrow(0, 0, plan.logical_token_count),
        )

    def release(self, ticket: CPKVIPCTransferTicket) -> None:
        publish_epoch(
            self._peer_bases,
            destination_mask=ticket.source_mask,
            signal_offset_bytes=self.layout.consumed_offset_bytes,
            signal_stride_bytes=self.layout.signal_stride_bytes,
            publisher_rank=self.cp_rank,
            epoch=ticket.epoch,
        )
        self._epochs.release(ticket)
