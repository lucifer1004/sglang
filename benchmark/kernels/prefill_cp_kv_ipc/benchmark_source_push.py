#!/usr/bin/env python3
"""Independent microbenchmark for the Prefill CP K/V IPC source-push kernel."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class CopyRun:
    src_start: int
    dst_start: int
    row_count: int

    def __post_init__(self) -> None:
        if self.src_start < 0:
            raise ValueError("src_start must be non-negative")
        if self.dst_start < 0:
            raise ValueError("dst_start must be non-negative")
        if self.row_count <= 0:
            raise ValueError("row_count must be positive")


def _zigzag_owner(block_index: int, cp_size: int, rotation: int) -> int:
    owner = block_index if block_index < cp_size else 2 * cp_size - block_index - 1
    return (owner + rotation) % cp_size


def build_rotated_zigzag_runs(
    *,
    total_rows: int,
    cp_size: int,
    rank: int,
    rotation: int,
) -> list[CopyRun]:
    """Build rank-local packed-to-logical runs for equal 2*CP zigzag blocks."""
    if cp_size <= 0:
        raise ValueError("cp_size must be positive")
    if rank < 0 or rank >= cp_size:
        raise ValueError("rank must be inside the CP group")
    if rotation < 0 or rotation >= cp_size:
        raise ValueError("rotation must be inside the CP group")
    block_count = 2 * cp_size
    if total_rows <= 0 or total_rows % block_count:
        raise ValueError("total_rows must be positive and divisible by 2 * cp_size")

    block_rows = total_rows // block_count
    local_start = 0
    runs: list[CopyRun] = []
    for block_index in range(block_count):
        if _zigzag_owner(block_index, cp_size, rotation) != rank:
            continue
        runs.append(
            CopyRun(
                src_start=local_start,
                dst_start=block_index * block_rows,
                row_count=block_rows,
            )
        )
        local_start += block_rows
    return runs


def build_fragmented_prefix_runs(
    *, physical_slots: Sequence[int], logical_start: int
) -> list[CopyRun]:
    """Coalesce adjacent physical slots while preserving logical row order."""
    if logical_start < 0:
        raise ValueError("logical_start must be non-negative")
    if not physical_slots:
        return []
    slots = [int(slot) for slot in physical_slots]
    if any(slot < 0 for slot in slots):
        raise ValueError("physical slots must be non-negative")

    runs: list[CopyRun] = []
    run_src = slots[0]
    run_dst = logical_start
    run_count = 1
    for logical_offset, slot in enumerate(slots[1:], start=1):
        if slot == slots[logical_offset - 1] + 1:
            run_count += 1
            continue
        runs.append(CopyRun(run_src, run_dst, run_count))
        run_src = slot
        run_dst = logical_start + logical_offset
        run_count = 1
    runs.append(CopyRun(run_src, run_dst, run_count))
    return runs


def build_fragmented_zigzag_runs(
    *,
    total_rows: int,
    cp_size: int,
    rank: int,
    rotation: int,
    physical_slots: Sequence[int],
) -> list[CopyRun]:
    """Map fragmented rank-local source rows into global zigzag destinations."""
    logical_runs = build_rotated_zigzag_runs(
        total_rows=total_rows,
        cp_size=cp_size,
        rank=rank,
        rotation=rotation,
    )
    expected_rows = sum(run.row_count for run in logical_runs)
    if len(physical_slots) != expected_rows:
        raise ValueError(
            "physical_slots must contain exactly the rank-local zigzag rows"
        )

    runs: list[CopyRun] = []
    local_offset = 0
    for logical_run in logical_runs:
        local_slots = physical_slots[
            local_offset : local_offset + logical_run.row_count
        ]
        runs.extend(
            build_fragmented_prefix_runs(
                physical_slots=local_slots,
                logical_start=logical_run.dst_start,
            )
        )
        local_offset += logical_run.row_count
    return runs


def build_fragmented_slots(*, row_count: int, fragment_rows: int) -> list[int]:
    """Build stable page-like source runs separated by physical gaps."""
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if fragment_rows <= 0:
        raise ValueError("fragment_rows must be positive")
    slots: list[int] = []
    cursor = 0
    fragment_index = 0
    while len(slots) < row_count:
        count = min(fragment_rows, row_count - len(slots))
        slots.extend(range(cursor, cursor + count))
        cursor += count + fragment_index % 3 + 1
        fragment_index += 1
    return slots


def build_rank_packed_to_logical(
    *,
    total_rows: int,
    cp_size: int,
    rotation: int,
    logical_offset: int = 0,
) -> torch.Tensor:
    """Return the logical row for each row in rank-concatenated gather output."""
    if logical_offset < 0:
        raise ValueError("logical_offset must be non-negative")
    if total_rows == 0:
        return torch.empty((0,), dtype=torch.int64)
    parts = [
        torch.arange(
            logical_offset + run.dst_start,
            logical_offset + run.dst_start + run.row_count,
            dtype=torch.int64,
        )
        for rank in range(cp_size)
        for run in build_rotated_zigzag_runs(
            total_rows=total_rows,
            cp_size=cp_size,
            rank=rank,
            rotation=rotation,
        )
    ]
    return torch.cat(parts)


def build_copy_tiles(runs: Iterable[CopyRun], *, tile_rows: int) -> torch.Tensor:
    """Split contiguous copy runs into fixed-row CUDA work descriptors."""
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    tiles: list[tuple[int, int, int]] = []
    for run in runs:
        offset = 0
        while offset < run.row_count:
            count = min(tile_rows, run.row_count - offset)
            tiles.append((run.src_start + offset, run.dst_start + offset, count))
            offset += count
    if not tiles:
        return torch.empty((0, 3), dtype=torch.int32)
    return torch.tensor(tiles, dtype=torch.int32)


def expand_copy_runs_to_indices(
    runs: Iterable[CopyRun],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand runs into arbitrary source and final logical row mappings."""
    source_parts = []
    destination_parts = []
    for run in runs:
        source_parts.append(
            torch.arange(
                run.src_start,
                run.src_start + run.row_count,
                dtype=torch.int32,
            )
        )
        destination_parts.append(
            torch.arange(
                run.dst_start,
                run.dst_start + run.row_count,
                dtype=torch.int32,
            )
        )
    if not source_parts:
        empty = torch.empty((0,), dtype=torch.int32)
        return empty, empty.clone()
    return torch.cat(source_parts), torch.cat(destination_parts)


CP_SIZE = 4
TOTAL_ROWS = 32768
ROW_WIDTH = 256
PREFIX_HIT_ROWS = 32256
PREFIX_HIT_EXTEND_ROWS = TOTAL_ROWS - PREFIX_HIT_ROWS
PREFIX_FRAGMENT_ROWS = 16
SIGNAL_STRIDE_BYTES = 128
IPC_ALIGNMENT_BYTES = 2 * 1024 * 1024


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return (value + alignment - 1) // alignment * alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--tile-rows", type=int, nargs="+", default=[8])
    parser.add_argument("--threads", type=int, nargs="+", default=[256])
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("cold", "prefix-hit"),
        default=["cold", "prefix-hit"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("dense", "mirror"),
        default=["dense", "mirror"],
    )
    args = parser.parse_args()
    if args.rotation < 0 or args.rotation >= CP_SIZE:
        parser.error(f"--rotation must be in [0, {CP_SIZE})")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations positive")
    if any(tile_rows <= 0 for tile_rows in args.tile_rows):
        parser.error("all --tile-rows values must be positive")
    allowed_threads = {64, 128, 256, 512}
    if any(num_threads not in allowed_threads for num_threads in args.threads):
        parser.error("--threads values must be one of 64, 128, 256, 512")
    return args


def init_distributed() -> tuple[int, int, torch.device]:
    missing = [
        name for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if name not in os.environ
    ]
    if missing:
        raise RuntimeError(f"run with torchrun; missing variables: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != CP_SIZE:
        raise RuntimeError(f"the first microbenchmark requires CP4, got {world_size}")
    return rank, world_size, device


def make_rank_kv(
    rank: int,
    rows: int,
    row_width: int,
    device: torch.device,
    *,
    salt: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(rows, dtype=torch.int32, device=device).view(-1, 1)
    col = torch.arange(row_width, dtype=torch.int32, device=device).view(1, -1)
    key = ((row * 3 + col * 5 + rank * 17 + salt * 23) % 127).to(
        torch.bfloat16
    )
    value = -((row * 7 + col * 11 + rank * 19 + salt * 29) % 113).to(
        torch.bfloat16
    )
    return key.contiguous(), value.contiguous()


@dataclass
class IpcArena:
    local_base: int
    peer_bases: tuple[int, ...]
    arena_bytes: int
    k_offset_bytes: int
    v_offset_bytes: int
    signal_offset_bytes: int
    consumed_signal_offset_bytes: int
    cuda_rt: object

    def close(self, rank: int) -> None:
        close_fn = self.cuda_rt.lib.cudaIpcCloseMemHandle
        close_fn.restype = ctypes.c_int
        close_fn.argtypes = [ctypes.c_void_p]
        for peer_rank, pointer in enumerate(self.peer_bases):
            if peer_rank == rank:
                continue
            self.cuda_rt.CUDART_CHECK(close_fn(ctypes.c_void_p(pointer)))
        dist.barrier()
        self.cuda_rt.cudaFree(ctypes.c_void_p(self.local_base))


def create_ipc_arena(rank: int, world_size: int) -> IpcArena:
    from sglang.srt.distributed.device_communicators.cuda_wrapper import (
        CudaRTLibrary,
        cudaIpcMemHandle_t,
    )

    row_bytes = ROW_WIDTH * 2
    k_offset_bytes = 0
    v_offset_bytes = TOTAL_ROWS * row_bytes
    signal_offset_bytes = v_offset_bytes + TOTAL_ROWS * row_bytes
    consumed_signal_offset_bytes = (
        signal_offset_bytes + world_size * SIGNAL_STRIDE_BYTES
    )
    arena_bytes = align_up(
        consumed_signal_offset_bytes + world_size * SIGNAL_STRIDE_BYTES,
        IPC_ALIGNMENT_BYTES,
    )
    cuda_rt = CudaRTLibrary()
    local_pointer = cuda_rt.cudaMalloc(arena_bytes)
    cuda_rt.cudaMemset(local_pointer, 0, arena_bytes)
    handle = cuda_rt.cudaIpcGetMemHandle(local_pointer)
    handle_bytes = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
    gathered_handles: list[bytes | None] = [None] * world_size
    dist.all_gather_object(gathered_handles, handle_bytes)

    peer_bases: list[int] = []
    for peer_rank, raw_handle in enumerate(gathered_handles):
        if raw_handle is None:
            raise RuntimeError("IPC handle exchange returned an empty handle")
        if peer_rank == rank:
            peer_bases.append(int(local_pointer.value))
            continue
        handle_obj = cudaIpcMemHandle_t()
        ctypes.memmove(
            ctypes.addressof(handle_obj),
            raw_handle,
            min(len(raw_handle), ctypes.sizeof(handle_obj)),
        )
        peer_bases.append(int(cuda_rt.cudaIpcOpenMemHandle(handle_obj).value))
    dist.barrier()
    return IpcArena(
        local_base=int(local_pointer.value),
        peer_bases=tuple(peer_bases),
        arena_bytes=arena_bytes,
        k_offset_bytes=k_offset_bytes,
        v_offset_bytes=v_offset_bytes,
        signal_offset_bytes=signal_offset_bytes,
        consumed_signal_offset_bytes=consumed_signal_offset_bytes,
        cuda_rt=cuda_rt,
    )


def read_local_arena(
    arena: IpcArena, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.empty((TOTAL_ROWS, ROW_WIDTH), dtype=torch.bfloat16, device=device)
    value = torch.empty_like(key)
    tensor_bytes = key.numel() * key.element_size()
    arena.cuda_rt.cudaMemcpy(
        ctypes.c_void_p(key.data_ptr()),
        ctypes.c_void_p(arena.local_base + arena.k_offset_bytes),
        tensor_bytes,
    )
    arena.cuda_rt.cudaMemcpy(
        ctypes.c_void_p(value.data_ptr()),
        ctypes.c_void_p(arena.local_base + arena.v_offset_bytes),
        tensor_bytes,
    )
    return key, value


@dataclass
class BenchmarkLayout:
    name: str
    prefix_rows: int
    extend_rows: int
    rotation: int
    prefix_pool_k: torch.Tensor
    prefix_pool_v: torch.Tensor
    prefix_slots: torch.Tensor
    prefix_runs: tuple[CopyRun, ...]
    extend_k: torch.Tensor
    extend_v: torch.Tensor
    extend_runs: tuple[CopyRun, ...]
    prefix_sizes: tuple[int, ...]
    extend_sizes: tuple[int, ...]
    prefix_rank_packed_to_logical: torch.Tensor
    extend_rank_packed_to_logical: torch.Tensor

    def copy_tiles(
        self, *, tile_rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            build_copy_tiles(self.prefix_runs, tile_rows=tile_rows).to(device),
            build_copy_tiles(self.extend_runs, tile_rows=tile_rows).to(device),
        )

    def copy_indices(
        self, *, device: torch.device
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]:
        prefix = expand_copy_runs_to_indices(self.prefix_runs)
        extend = expand_copy_runs_to_indices(self.extend_runs)
        return (
            (prefix[0].to(device), prefix[1].to(device)),
            (extend[0].to(device), extend[1].to(device)),
        )


@dataclass
class NcclBuffers:
    local_prefix_k: torch.Tensor
    local_prefix_v: torch.Tensor
    packed_prefix_k: torch.Tensor | None
    packed_prefix_v: torch.Tensor | None
    packed_extend_k: torch.Tensor | None
    packed_extend_v: torch.Tensor | None
    logical_prefix_k: torch.Tensor | None
    logical_prefix_v: torch.Tensor | None
    logical_extend_k: torch.Tensor | None
    logical_extend_v: torch.Tensor | None
    final_k: torch.Tensor | None
    final_v: torch.Tensor | None


def _segment_sizes(*, total_rows: int, cp_size: int, rotation: int) -> tuple[int, ...]:
    if total_rows == 0:
        return (0,) * cp_size
    return tuple(
        sum(
            run.row_count
            for run in build_rotated_zigzag_runs(
                total_rows=total_rows,
                cp_size=cp_size,
                rank=rank,
                rotation=rotation,
            )
        )
        for rank in range(cp_size)
    )


def prepare_layout(
    *,
    name: str,
    prefix_rows: int,
    extend_rows: int,
    rank: int,
    rotation: int,
    device: torch.device,
) -> BenchmarkLayout:
    if prefix_rows + extend_rows != TOTAL_ROWS:
        raise ValueError("layout must cover TOTAL_ROWS")

    prefix_sizes = _segment_sizes(
        total_rows=prefix_rows, cp_size=CP_SIZE, rotation=rotation
    )
    extend_sizes = _segment_sizes(
        total_rows=extend_rows, cp_size=CP_SIZE, rotation=rotation
    )

    local_prefix_rows = prefix_sizes[rank]
    physical_slots = build_fragmented_slots(
        row_count=local_prefix_rows,
        fragment_rows=PREFIX_FRAGMENT_ROWS,
    )
    prefix_slots = torch.tensor(physical_slots, dtype=torch.int64, device=device)
    prefix_pool_rows = max(physical_slots, default=0) + 1
    prefix_pool_k = torch.zeros(
        (prefix_pool_rows, ROW_WIDTH), dtype=torch.bfloat16, device=device
    )
    prefix_pool_v = torch.zeros_like(prefix_pool_k)
    if local_prefix_rows:
        packed_prefix_k, packed_prefix_v = make_rank_kv(
            rank, local_prefix_rows, ROW_WIDTH, device, salt=1
        )
        prefix_pool_k.index_copy_(0, prefix_slots, packed_prefix_k)
        prefix_pool_v.index_copy_(0, prefix_slots, packed_prefix_v)
        prefix_runs = tuple(
            build_fragmented_zigzag_runs(
                total_rows=prefix_rows,
                cp_size=CP_SIZE,
                rank=rank,
                rotation=rotation,
                physical_slots=physical_slots,
            )
        )
    else:
        prefix_runs = ()

    local_extend_rows = extend_sizes[rank]
    extend_k, extend_v = make_rank_kv(
        rank, local_extend_rows, ROW_WIDTH, device, salt=2
    )
    extend_runs = tuple(
        CopyRun(
            src_start=run.src_start,
            dst_start=prefix_rows + run.dst_start,
            row_count=run.row_count,
        )
        for run in (
            build_rotated_zigzag_runs(
                total_rows=extend_rows,
                cp_size=CP_SIZE,
                rank=rank,
                rotation=rotation,
            )
            if extend_rows
            else []
        )
    )

    return BenchmarkLayout(
        name=name,
        prefix_rows=prefix_rows,
        extend_rows=extend_rows,
        rotation=rotation,
        prefix_pool_k=prefix_pool_k,
        prefix_pool_v=prefix_pool_v,
        prefix_slots=prefix_slots,
        prefix_runs=prefix_runs,
        extend_k=extend_k,
        extend_v=extend_v,
        extend_runs=extend_runs,
        prefix_sizes=prefix_sizes,
        extend_sizes=extend_sizes,
        prefix_rank_packed_to_logical=build_rank_packed_to_logical(
            total_rows=prefix_rows,
            cp_size=CP_SIZE,
            rotation=rotation,
        ).to(device),
        extend_rank_packed_to_logical=build_rank_packed_to_logical(
            total_rows=extend_rows,
            cp_size=CP_SIZE,
            rotation=rotation,
        ).to(device),
    )


def build_expected_layout(
    layout: BenchmarkLayout, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_k = torch.empty(
        (TOTAL_ROWS, ROW_WIDTH), dtype=torch.bfloat16, device=device
    )
    expected_v = torch.empty_like(expected_k)
    for source_rank in range(CP_SIZE):
        prefix_source_k, prefix_source_v = make_rank_kv(
            source_rank,
            layout.prefix_sizes[source_rank],
            ROW_WIDTH,
            device,
            salt=1,
        )
        if layout.prefix_rows:
            for run in build_rotated_zigzag_runs(
                total_rows=layout.prefix_rows,
                cp_size=CP_SIZE,
                rank=source_rank,
                rotation=layout.rotation,
            ):
                expected_k.narrow(0, run.dst_start, run.row_count).copy_(
                    prefix_source_k.narrow(0, run.src_start, run.row_count)
                )
                expected_v.narrow(0, run.dst_start, run.row_count).copy_(
                    prefix_source_v.narrow(0, run.src_start, run.row_count)
                )

        extend_source_k, extend_source_v = make_rank_kv(
            source_rank,
            layout.extend_sizes[source_rank],
            ROW_WIDTH,
            device,
            salt=2,
        )
        if layout.extend_rows:
            for run in build_rotated_zigzag_runs(
                total_rows=layout.extend_rows,
                cp_size=CP_SIZE,
                rank=source_rank,
                rotation=layout.rotation,
            ):
                destination = layout.prefix_rows + run.dst_start
                expected_k.narrow(0, destination, run.row_count).copy_(
                    extend_source_k.narrow(0, run.src_start, run.row_count)
                )
                expected_v.narrow(0, destination, run.row_count).copy_(
                    extend_source_v.narrow(0, run.src_start, run.row_count)
                )
    return expected_k, expected_v


def create_nccl_buffers(
    layout: BenchmarkLayout,
    *,
    is_destination: bool,
) -> NcclBuffers:
    local_prefix_k = layout.prefix_pool_k.index_select(0, layout.prefix_slots)
    local_prefix_v = layout.prefix_pool_v.index_select(0, layout.prefix_slots)
    if not is_destination:
        return NcclBuffers(
            local_prefix_k=local_prefix_k,
            local_prefix_v=local_prefix_v,
            packed_prefix_k=None,
            packed_prefix_v=None,
            packed_extend_k=None,
            packed_extend_v=None,
            logical_prefix_k=None,
            logical_prefix_v=None,
            logical_extend_k=None,
            logical_extend_v=None,
            final_k=None,
            final_v=None,
        )
    packed_prefix_k = torch.empty(
        (layout.prefix_rows, ROW_WIDTH),
        dtype=torch.bfloat16,
        device=layout.extend_k.device,
    )
    packed_prefix_v = torch.empty_like(packed_prefix_k)
    packed_extend_k = torch.empty(
        (layout.extend_rows, ROW_WIDTH),
        dtype=torch.bfloat16,
        device=layout.extend_k.device,
    )
    packed_extend_v = torch.empty_like(packed_extend_k)
    logical_prefix_k = torch.empty_like(packed_prefix_k)
    logical_prefix_v = torch.empty_like(packed_prefix_v)
    logical_extend_k = torch.empty_like(packed_extend_k)
    logical_extend_v = torch.empty_like(packed_extend_v)
    final_k = torch.empty(
        (TOTAL_ROWS, ROW_WIDTH),
        dtype=torch.bfloat16,
        device=layout.extend_k.device,
    )
    final_v = torch.empty_like(final_k)
    return NcclBuffers(
        local_prefix_k=local_prefix_k,
        local_prefix_v=local_prefix_v,
        packed_prefix_k=packed_prefix_k,
        packed_prefix_v=packed_prefix_v,
        packed_extend_k=packed_extend_k,
        packed_extend_v=packed_extend_v,
        logical_prefix_k=logical_prefix_k,
        logical_prefix_v=logical_prefix_v,
        logical_extend_k=logical_extend_k,
        logical_extend_v=logical_extend_v,
        final_k=final_k,
        final_v=final_v,
    )


def exchange_nccl_segment(
    *,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    packed_k: torch.Tensor | None,
    packed_v: torch.Tensor | None,
    sizes: tuple[int, ...],
    destination_ranks: tuple[int, ...],
    rank: int,
    communicator,
) -> None:
    total_rows = sum(sizes)
    if total_rows == 0:
        return
    is_destination = rank in destination_ranks
    if len(destination_ranks) == CP_SIZE:
        if packed_k is None or packed_v is None:
            raise RuntimeError("dense NCCL destination buffers are missing")
        communicator.group_start()
        communicator.all_gather(packed_k, local_k)
        communicator.all_gather(packed_v, local_v)
        communicator.group_end()
        return

    if is_destination:
        if packed_k is None or packed_v is None:
            raise RuntimeError("mirror NCCL destination buffers are missing")
        local_offset = sum(sizes[:rank])
        if sizes[rank]:
            packed_k.narrow(0, local_offset, sizes[rank]).copy_(local_k)
            packed_v.narrow(0, local_offset, sizes[rank]).copy_(local_v)

    source_offsets = [0]
    for size in sizes:
        source_offsets.append(source_offsets[-1] + size)
    communicator.group_start()
    for local, packed in ((local_k, packed_k), (local_v, packed_v)):
        if is_destination:
            for source, size in enumerate(sizes):
                if source == rank or size == 0:
                    continue
                communicator.recv(
                    packed.narrow(0, source_offsets[source], size), source
                )
        if sizes[rank]:
            for destination in destination_ranks:
                if destination != rank:
                    communicator.send(local, destination)
    communicator.group_end()


def run_nccl_path(
    *,
    layout: BenchmarkLayout,
    buffers: NcclBuffers,
    destination_ranks: tuple[int, ...],
    rank: int,
    communicator,
    materialize_prefix: bool,
    transfer: bool,
    reorder: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if materialize_prefix:
        local_prefix_k = layout.prefix_pool_k.index_select(0, layout.prefix_slots)
        local_prefix_v = layout.prefix_pool_v.index_select(0, layout.prefix_slots)
    else:
        local_prefix_k = buffers.local_prefix_k
        local_prefix_v = buffers.local_prefix_v

    if transfer:
        exchange_nccl_segment(
            local_k=local_prefix_k,
            local_v=local_prefix_v,
            packed_k=buffers.packed_prefix_k,
            packed_v=buffers.packed_prefix_v,
            sizes=layout.prefix_sizes,
            destination_ranks=destination_ranks,
            rank=rank,
            communicator=communicator,
        )
        exchange_nccl_segment(
            local_k=layout.extend_k,
            local_v=layout.extend_v,
            packed_k=buffers.packed_extend_k,
            packed_v=buffers.packed_extend_v,
            sizes=layout.extend_sizes,
            destination_ranks=destination_ranks,
            rank=rank,
            communicator=communicator,
        )

    if rank not in destination_ranks or not reorder:
        return None
    assert buffers.final_k is not None and buffers.final_v is not None
    if layout.prefix_rows:
        assert buffers.packed_prefix_k is not None
        assert buffers.packed_prefix_v is not None
        assert buffers.logical_prefix_k is not None
        assert buffers.logical_prefix_v is not None
        buffers.logical_prefix_k.index_copy_(
            0,
            layout.prefix_rank_packed_to_logical,
            buffers.packed_prefix_k,
        )
        buffers.logical_prefix_v.index_copy_(
            0,
            layout.prefix_rank_packed_to_logical,
            buffers.packed_prefix_v,
        )
    if layout.extend_rows:
        assert buffers.packed_extend_k is not None
        assert buffers.packed_extend_v is not None
        assert buffers.logical_extend_k is not None
        assert buffers.logical_extend_v is not None
        buffers.logical_extend_k.index_copy_(
            0,
            layout.extend_rank_packed_to_logical,
            buffers.packed_extend_k,
        )
        buffers.logical_extend_v.index_copy_(
            0,
            layout.extend_rank_packed_to_logical,
            buffers.packed_extend_v,
        )
    if layout.prefix_rows:
        buffers.final_k.narrow(0, 0, layout.prefix_rows).copy_(
            buffers.logical_prefix_k
        )
        buffers.final_v.narrow(0, 0, layout.prefix_rows).copy_(
            buffers.logical_prefix_v
        )
    if layout.extend_rows:
        buffers.final_k.narrow(
            0, layout.prefix_rows, layout.extend_rows
        ).copy_(buffers.logical_extend_k)
        buffers.final_v.narrow(
            0, layout.prefix_rows, layout.extend_rows
        ).copy_(buffers.logical_extend_v)
    return buffers.final_k, buffers.final_v


def run_ipc_path(
    *,
    layout: BenchmarkLayout,
    prefix_work,
    extend_work,
    indexed: bool,
    rows_per_block: int,
    arena: IpcArena,
    peer_bases: torch.Tensor,
    completion: torch.Tensor,
    destination_mask: int,
    rank: int,
    epoch: int,
    ready_to_consume: bool,
    complete_lifecycle: bool,
    num_threads: int,
) -> None:
    from sglang.jit_kernel.prefill_cp_kv_source_push import (
        publish_epoch,
        source_push,
        source_push_indexed,
        wait_ready,
    )

    segments = []
    if indexed:
        if prefix_work[0].shape[0]:
            segments.append(
                (layout.prefix_pool_k, layout.prefix_pool_v, prefix_work)
            )
        if extend_work[0].shape[0]:
            segments.append((layout.extend_k, layout.extend_v, extend_work))
    else:
        if prefix_work.shape[0]:
            segments.append(
                (layout.prefix_pool_k, layout.prefix_pool_v, prefix_work)
            )
        if extend_work.shape[0]:
            segments.append((layout.extend_k, layout.extend_v, extend_work))

    for segment_index, (key, value, work) in enumerate(segments):
        common = dict(
            destination_mask=destination_mask,
            k_offset_bytes=arena.k_offset_bytes,
            v_offset_bytes=arena.v_offset_bytes,
            signal_offset_bytes=arena.signal_offset_bytes,
            signal_stride_bytes=SIGNAL_STRIDE_BYTES,
            completion=completion,
            source_rank=rank,
            epoch=epoch,
            publish_signal=(
                ready_to_consume and segment_index == len(segments) - 1
            ),
            num_threads=num_threads,
        )
        if indexed:
            source_push_indexed(
                key,
                value,
                work[0],
                work[1],
                peer_bases,
                rows_per_block=rows_per_block,
                **common,
            )
        else:
            source_push(key, value, work, peer_bases, **common)
    source_mask = sum(
        1 << source
        for source in range(CP_SIZE)
        if layout.prefix_sizes[source] + layout.extend_sizes[source] > 0
    )
    if ready_to_consume:
        if destination_mask & (1 << rank):
            wait_ready(
                local_arena_base=arena.local_base,
                source_mask=source_mask,
                signal_offset_bytes=arena.signal_offset_bytes,
                signal_stride_bytes=SIGNAL_STRIDE_BYTES,
                epoch=epoch,
            )
            if complete_lifecycle:
                publish_epoch(
                    peer_bases,
                    destination_mask=source_mask,
                    signal_offset_bytes=arena.consumed_signal_offset_bytes,
                    signal_stride_bytes=SIGNAL_STRIDE_BYTES,
                    publisher_rank=rank,
                    epoch=epoch,
                )
        if complete_lifecycle and source_mask & (1 << rank):
            wait_ready(
                local_arena_base=arena.local_base,
                source_mask=destination_mask,
                signal_offset_bytes=arena.consumed_signal_offset_bytes,
                signal_stride_bytes=SIGNAL_STRIDE_BYTES,
                epoch=epoch,
            )


def measure_cuda_operation(
    operation: Callable[[int], object],
    *,
    rank: int,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    critical_samples: list[float] = []
    for step in range(warmup + iterations):
        dist.barrier()
        start.record()
        keepalive = operation(step)
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        per_rank_ms: list[float | None] = [None] * CP_SIZE
        dist.all_gather_object(per_rank_ms, elapsed_ms)
        if rank == 0 and step >= warmup:
            critical_samples.append(max(float(value) for value in per_rank_ms))
        del keepalive
    return critical_samples


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_samples(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    values = sorted(samples)
    return {
        "p10_ms": _percentile(values, 0.10),
        "median_ms": _percentile(values, 0.50),
        "p90_ms": _percentile(values, 0.90),
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def count_mismatches(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> int:
    actual_k, actual_v = actual
    expected_k, expected_v = expected
    mismatch_count = int(torch.count_nonzero(actual_k != expected_k).item())
    mismatch_count += int(torch.count_nonzero(actual_v != expected_v).item())
    return mismatch_count


def gather_correctness(mismatch_count: int) -> dict[str, object]:
    gathered: list[int | None] = [None] * CP_SIZE
    dist.all_gather_object(gathered, mismatch_count)
    return {
        "mismatches_per_rank": gathered,
        "passed": all(count == 0 for count in gathered),
    }


def validate_ipc_layout(
    *,
    layout: BenchmarkLayout,
    prefix_work,
    extend_work,
    indexed: bool,
    rows_per_block: int,
    arena: IpcArena,
    peer_bases: torch.Tensor,
    completion: torch.Tensor,
    destination_mask: int,
    rank: int,
    device: torch.device,
    epoch: int,
    num_threads: int,
) -> dict[str, object]:
    run_ipc_path(
        layout=layout,
        prefix_work=prefix_work,
        extend_work=extend_work,
        indexed=indexed,
        rows_per_block=rows_per_block,
        arena=arena,
        peer_bases=peer_bases,
        completion=completion,
        destination_mask=destination_mask,
        rank=rank,
        epoch=epoch,
        ready_to_consume=True,
        complete_lifecycle=True,
        num_threads=num_threads,
    )
    torch.cuda.synchronize(device)
    dist.barrier()
    mismatch_count = 0
    if destination_mask & (1 << rank):
        mismatch_count = count_mismatches(
            read_local_arena(arena, device),
            build_expected_layout(layout, device),
        )
    correctness = gather_correctness(mismatch_count)
    correctness.update(
        {
            "epoch": epoch,
            "prefix_work_items": int(
                prefix_work[0].shape[0] if indexed else prefix_work.shape[0]
            ),
            "extend_work_items": int(
                extend_work[0].shape[0] if indexed else extend_work.shape[0]
            ),
        }
    )
    return correctness


def validate_nccl_layout(
    *,
    layout: BenchmarkLayout,
    buffers: NcclBuffers,
    destination_ranks: tuple[int, ...],
    rank: int,
    device: torch.device,
    communicator,
) -> dict[str, object]:
    actual = run_nccl_path(
        layout=layout,
        buffers=buffers,
        destination_ranks=destination_ranks,
        rank=rank,
        communicator=communicator,
        materialize_prefix=True,
        transfer=True,
        reorder=True,
    )
    torch.cuda.synchronize(device)
    dist.barrier()
    mismatch_count = 0
    if rank in destination_ranks:
        if actual is None:
            raise RuntimeError("NCCL destination did not produce final K/V")
        mismatch_count = count_mismatches(
            actual, build_expected_layout(layout, device)
        )
    return gather_correctness(mismatch_count)


def add_effective_bandwidth(
    summary: dict[str, float], logical_bytes: int
) -> dict[str, float]:
    result = dict(summary)
    median_ms = result.get("median_ms")
    if median_ms:
        result["effective_gbps"] = logical_bytes / (median_ms * 1e-3) / 1e9
    return result


def run_benchmark_mode(
    *,
    layout: BenchmarkLayout,
    mode: str,
    tile_rows_values: Sequence[int],
    thread_values: Sequence[int],
    arena: IpcArena,
    peer_bases: torch.Tensor,
    completion: torch.Tensor,
    rank: int,
    device: torch.device,
    communicator,
    warmup: int,
    iterations: int,
    epoch_start: int,
) -> tuple[dict[str, object], int]:
    destination_ranks = (
        tuple(range(CP_SIZE)) if mode == "dense" else (CP_SIZE - 1,)
    )
    destination_mask = sum(1 << destination for destination in destination_ranks)
    logical_bytes = (
        TOTAL_ROWS
        * ROW_WIDTH
        * 2
        * 2
        * len(destination_ranks)
    )
    remote_bytes = logical_bytes * (CP_SIZE - 1) // CP_SIZE
    buffers = create_nccl_buffers(
        layout, is_destination=rank in destination_ranks
    )
    baseline_correctness = validate_nccl_layout(
        layout=layout,
        buffers=buffers,
        destination_ranks=destination_ranks,
        rank=rank,
        device=device,
        communicator=communicator,
    )
    if not baseline_correctness["passed"]:
        raise AssertionError(f"{layout.name} {mode} NCCL baseline is incorrect")

    nccl_transfer = measure_cuda_operation(
        lambda _step: run_nccl_path(
            layout=layout,
            buffers=buffers,
            destination_ranks=destination_ranks,
            rank=rank,
            communicator=communicator,
            materialize_prefix=False,
            transfer=True,
            reorder=False,
        ),
        rank=rank,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    reorder_only = measure_cuda_operation(
        lambda _step: run_nccl_path(
            layout=layout,
            buffers=buffers,
            destination_ranks=destination_ranks,
            rank=rank,
            communicator=communicator,
            materialize_prefix=False,
            transfer=False,
            reorder=True,
        ),
        rank=rank,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    baseline_total = measure_cuda_operation(
        lambda _step: run_nccl_path(
            layout=layout,
            buffers=buffers,
            destination_ranks=destination_ranks,
            rank=rank,
            communicator=communicator,
            materialize_prefix=True,
            transfer=True,
            reorder=True,
        ),
        rank=rank,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    baseline = {
        "correctness": baseline_correctness,
        "nccl_transfer": add_effective_bandwidth(
            summarize_samples(nccl_transfer), logical_bytes
        ),
        "reorder_only": summarize_samples(reorder_only),
        "ready_to_consume": add_effective_bandwidth(
            summarize_samples(baseline_total), logical_bytes
        ),
    }

    configs = []
    epoch = epoch_start
    for tile_rows in tile_rows_values:
        prefix_tiles, extend_tiles = layout.copy_tiles(
            tile_rows=tile_rows, device=device
        )
        prefix_indices, extend_indices = layout.copy_indices(device=device)
        for num_threads in thread_values:
            for variant, prefix_work, extend_work in (
                ("runs", prefix_tiles, extend_tiles),
                ("indexed", prefix_indices, extend_indices),
            ):
                indexed = variant == "indexed"
                correctness = validate_ipc_layout(
                    layout=layout,
                    prefix_work=prefix_work,
                    extend_work=extend_work,
                    indexed=indexed,
                    rows_per_block=tile_rows,
                    arena=arena,
                    peer_bases=peer_bases,
                    completion=completion,
                    destination_mask=destination_mask,
                    rank=rank,
                    device=device,
                    epoch=epoch,
                    num_threads=num_threads,
                )
                epoch += 1
                if not correctness["passed"]:
                    raise AssertionError(
                        f"{layout.name} {mode} {variant} IPC failed for "
                        f"tile_rows={tile_rows} threads={num_threads}"
                    )

                def run_payload(
                    _step,
                    pw=prefix_work,
                    ew=extend_work,
                    idx=indexed,
                ):
                    return run_ipc_path(
                        layout=layout,
                        prefix_work=pw,
                        extend_work=ew,
                        indexed=idx,
                        rows_per_block=tile_rows,
                        arena=arena,
                        peer_bases=peer_bases,
                        completion=completion,
                        destination_mask=destination_mask,
                        rank=rank,
                        epoch=1,
                        ready_to_consume=False,
                        complete_lifecycle=False,
                        num_threads=num_threads,
                    )

                payload_only = measure_cuda_operation(
                    run_payload,
                    rank=rank,
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                )
                ready_epoch_base = epoch

                def run_ready(
                    step,
                    base=ready_epoch_base,
                    pw=prefix_work,
                    ew=extend_work,
                    idx=indexed,
                ):
                    return run_ipc_path(
                        layout=layout,
                        prefix_work=pw,
                        extend_work=ew,
                        indexed=idx,
                        rows_per_block=tile_rows,
                        arena=arena,
                        peer_bases=peer_bases,
                        completion=completion,
                        destination_mask=destination_mask,
                        rank=rank,
                        epoch=base + step,
                        ready_to_consume=True,
                        complete_lifecycle=False,
                        num_threads=num_threads,
                    )

                ready_to_consume = measure_cuda_operation(
                    run_ready,
                    rank=rank,
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                )
                epoch += warmup + iterations
                lifecycle_epoch_base = epoch

                def run_lifecycle(
                    step,
                    base=lifecycle_epoch_base,
                    pw=prefix_work,
                    ew=extend_work,
                    idx=indexed,
                ):
                    return run_ipc_path(
                        layout=layout,
                        prefix_work=pw,
                        extend_work=ew,
                        indexed=idx,
                        rows_per_block=tile_rows,
                        arena=arena,
                        peer_bases=peer_bases,
                        completion=completion,
                        destination_mask=destination_mask,
                        rank=rank,
                        epoch=base + step,
                        ready_to_consume=True,
                        complete_lifecycle=True,
                        num_threads=num_threads,
                    )

                ready_and_consumed = measure_cuda_operation(
                    run_lifecycle,
                    rank=rank,
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                )
                epoch += warmup + iterations
                ready_summary = add_effective_bandwidth(
                    summarize_samples(ready_to_consume), logical_bytes
                )
                baseline_median = baseline["ready_to_consume"].get("median_ms")
                ipc_median = ready_summary.get("median_ms")
                speedup = (
                    baseline_median / ipc_median
                    if baseline_median and ipc_median
                    else None
                )
                configs.append(
                    {
                        "variant": variant,
                        "tile_rows": tile_rows,
                        "threads": num_threads,
                        "correctness": correctness,
                        "payload_only": add_effective_bandwidth(
                            summarize_samples(payload_only), logical_bytes
                        ),
                        "ready_to_consume": ready_summary,
                        "ready_and_consumed": add_effective_bandwidth(
                            summarize_samples(ready_and_consumed), logical_bytes
                        ),
                        "speedup_vs_baseline": speedup,
                    }
                )

    return (
        {
            "layout": layout.name,
            "mode": mode,
            "destination_ranks": destination_ranks,
            "logical_bytes": logical_bytes,
            "remote_bytes": remote_bytes,
            "baseline": baseline,
            "ipc_configs": configs,
        },
        epoch,
    )


def print_summary(results: Sequence[dict[str, object]]) -> None:
    for result in results:
        baseline = result["baseline"]["ready_to_consume"]
        for config in result["ipc_configs"]:
            ready = config["ready_to_consume"]
            lifecycle = config["ready_and_consumed"]
            print(
                "RESULT "
                f"layout={result['layout']} mode={result['mode']} "
                f"variant={config['variant']} "
                f"tile_rows={config['tile_rows']} threads={config['threads']} "
                f"baseline_ms={baseline.get('median_ms', float('nan')):.4f} "
                f"ipc_ms={ready.get('median_ms', float('nan')):.4f} "
                f"lifecycle_ms={lifecycle.get('median_ms', float('nan')):.4f} "
                f"speedup={config['speedup_vs_baseline']:.3f}",
                flush=True,
            )


def load_jit_once_across_ranks(rank: int) -> None:
    from sglang.jit_kernel.prefill_cp_kv_source_push import (
        _jit_source_push_module,
    )

    if rank == 0:
        _jit_source_push_module()
    dist.barrier()
    if rank != 0:
        _jit_source_push_module()
    dist.barrier()


class _TorchrunCPGroup:
    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size
        self.ranks = tuple(range(world_size))
        self.cpu_group = dist.group.WORLD

    def broadcast_object(self, obj, src: int = 0):
        values = [obj if self.rank_in_group == src else None]
        dist.broadcast_object_list(values, src=src)
        return values[0]

    def barrier(self):
        dist.barrier()


def validate_production_transport(
    *,
    layouts: Sequence[BenchmarkLayout],
    modes: Sequence[str],
    rank: int,
    world_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    from sglang.srt.layers.attention.cp_sharded_kv import (
        CPKVSourcePushSegmentPlan,
        CPPrefillKVSourcePushPlan,
    )
    from sglang.srt.layers.attention.prefill_cp_kv_ipc import (
        CPKVIPCSourcePushTransport,
    )

    transport = CPKVIPCSourcePushTransport(
        cp_group=_TorchrunCPGroup(rank, world_size),
        device=device,
        max_rows=TOTAL_ROWS,
        row_width=ROW_WIDTH,
    )
    cases = [(layout, mode) for layout in layouts for mode in modes]
    if len(cases) == 1:
        cases = cases * 2
    results = []
    for step, (layout, mode) in enumerate(cases):
        prefix_indices, extend_indices = layout.copy_indices(device=device)
        source_mask = sum(
            1 << source
            for source in range(CP_SIZE)
            if layout.prefix_sizes[source] + layout.extend_sizes[source] > 0
        )
        plan = CPPrefillKVSourcePushPlan(
            prefix=CPKVSourcePushSegmentPlan(
                source_rows=prefix_indices[0],
                destination_rows=prefix_indices[1],
            ),
            extend=CPKVSourcePushSegmentPlan(
                source_rows=extend_indices[0],
                destination_rows=extend_indices[1],
            ),
            source_mask=source_mask,
            logical_token_count=TOTAL_ROWS,
        )
        destination_ranks = (
            tuple(range(CP_SIZE)) if mode == "dense" else (CP_SIZE - 1,)
        )
        lease = transport.push(
            plan=plan,
            prefix_key_rows=layout.prefix_pool_k,
            prefix_value_rows=layout.prefix_pool_v,
            extend_key_rows=layout.extend_k,
            extend_value_rows=layout.extend_v,
            destination_ranks=destination_ranks,
        )
        mismatch_count = 0
        if rank in destination_ranks:
            if lease is None:
                raise RuntimeError("production transport destination has no lease")
            mismatch_count = count_mismatches(
                (lease.key, lease.value),
                build_expected_layout(layout, device),
            )
            lease.release()
        elif lease is not None:
            raise RuntimeError("production transport non-destination received a lease")
        correctness = gather_correctness(mismatch_count)
        correctness.update(
            {
                "step": step,
                "layout": layout.name,
                "mode": mode,
            }
        )
        if not correctness["passed"]:
            raise AssertionError(
                f"production transport failed for {layout.name} {mode}"
            )
        results.append(correctness)

    empty_rows = torch.empty(
        (0, ROW_WIDTH), dtype=torch.bfloat16, device=device
    )
    empty_indices = torch.empty((0,), dtype=torch.int32, device=device)
    expected_key, expected_value = make_rank_kv(
        0, 1, ROW_WIDTH, device, salt=97
    )
    local_is_source = rank == 0
    short_plan = CPPrefillKVSourcePushPlan(
        prefix=CPKVSourcePushSegmentPlan(
            source_rows=empty_indices,
            destination_rows=empty_indices,
        ),
        extend=CPKVSourcePushSegmentPlan(
            source_rows=(
                torch.zeros((1,), dtype=torch.int32, device=device)
                if local_is_source
                else empty_indices
            ),
            destination_rows=(
                torch.zeros((1,), dtype=torch.int32, device=device)
                if local_is_source
                else empty_indices
            ),
        ),
        source_mask=0b0001,
        logical_token_count=1,
    )
    lease = transport.push(
        plan=short_plan,
        prefix_key_rows=empty_rows,
        prefix_value_rows=empty_rows,
        extend_key_rows=expected_key if local_is_source else empty_rows,
        extend_value_rows=expected_value if local_is_source else empty_rows,
        destination_ranks=tuple(range(CP_SIZE)),
    )
    if lease is None:
        raise RuntimeError("short-owner-gap destination has no IPC lease")
    mismatch_count = count_mismatches(
        (lease.key, lease.value),
        (expected_key, expected_value),
    )
    lease.release()
    correctness = gather_correctness(mismatch_count)
    correctness.update(
        {
            "step": len(cases),
            "layout": "short-owner-gap",
            "mode": "dense",
        }
    )
    if not correctness["passed"]:
        raise AssertionError("production transport failed for short-owner-gap")
    results.append(correctness)
    torch.cuda.synchronize(device)
    dist.barrier()
    return results


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_distributed()
    arena = create_ipc_arena(rank, world_size)
    try:
        load_jit_once_across_ranks(rank)
        peer_bases = torch.tensor(arena.peer_bases, dtype=torch.int64)
        completion = torch.zeros(1, dtype=torch.int32, device=device)
        layout_specs = {
            "cold": (0, TOTAL_ROWS),
            "prefix-hit": (PREFIX_HIT_ROWS, PREFIX_HIT_EXTEND_ROWS),
        }
        layouts = [
            prepare_layout(
                name=name,
                prefix_rows=layout_specs[name][0],
                extend_rows=layout_specs[name][1],
                rank=rank,
                rotation=args.rotation,
                device=device,
            )
            for name in args.layouts
        ]

        if args.correctness_only:
            correctness_results = []
            epoch = 1
            for layout in layouts:
                for mode in args.modes:
                    destination_mask = (
                        (1 << CP_SIZE) - 1
                        if mode == "dense"
                        else 1 << (CP_SIZE - 1)
                    )
                    for tile_rows in args.tile_rows:
                        prefix_tiles, extend_tiles = layout.copy_tiles(
                            tile_rows=tile_rows, device=device
                        )
                        prefix_indices, extend_indices = layout.copy_indices(
                            device=device
                        )
                        for num_threads in args.threads:
                            for variant, prefix_work, extend_work in (
                                ("runs", prefix_tiles, extend_tiles),
                                ("indexed", prefix_indices, extend_indices),
                            ):
                                correctness = validate_ipc_layout(
                                    layout=layout,
                                    prefix_work=prefix_work,
                                    extend_work=extend_work,
                                    indexed=variant == "indexed",
                                    rows_per_block=tile_rows,
                                    arena=arena,
                                    peer_bases=peer_bases,
                                    completion=completion,
                                    destination_mask=destination_mask,
                                    rank=rank,
                                    device=device,
                                    epoch=epoch,
                                    num_threads=num_threads,
                                )
                                epoch += 1
                                correctness_results.append(
                                    {
                                        "layout": layout.name,
                                        "mode": mode,
                                        "variant": variant,
                                        "tile_rows": tile_rows,
                                        "threads": num_threads,
                                        **correctness,
                                    }
                                )
            production_transport_results = validate_production_transport(
                layouts=layouts,
                modes=args.modes,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "config": {
                                "cp_size": CP_SIZE,
                                "total_rows": TOTAL_ROWS,
                                "row_width": ROW_WIDTH,
                                "rotation": args.rotation,
                                "arena_bytes": arena.arena_bytes,
                            },
                            "correctness": correctness_results,
                            "production_transport": production_transport_results,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
            if not all(result["passed"] for result in correctness_results):
                raise AssertionError("source-push correctness failed")
            return

        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        communicator = PyNcclCommunicator(dist.group.WORLD, device=device)
        benchmark_results = []
        epoch = 1
        with communicator.change_state(enable=True):
            for layout in layouts:
                for mode in args.modes:
                    result, epoch = run_benchmark_mode(
                        layout=layout,
                        mode=mode,
                        tile_rows_values=args.tile_rows,
                        thread_values=args.threads,
                        arena=arena,
                        peer_bases=peer_bases,
                        completion=completion,
                        rank=rank,
                        device=device,
                        communicator=communicator,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        epoch_start=epoch,
                    )
                    benchmark_results.append(result)
        if rank == 0:
            print_summary(benchmark_results)
            print(
                json.dumps(
                    {
                        "config": {
                            "cp_size": CP_SIZE,
                            "total_rows": TOTAL_ROWS,
                            "row_width": ROW_WIDTH,
                            "rotation": args.rotation,
                            "tile_rows": args.tile_rows,
                            "threads": args.threads,
                            "warmup": args.warmup,
                            "iterations": args.iterations,
                            "arena_bytes": arena.arena_bytes,
                        },
                        "results": benchmark_results,
                    },
                    indent=2,
                ),
                flush=True,
            )
    finally:
        dist.barrier()
        arena.close(rank)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
