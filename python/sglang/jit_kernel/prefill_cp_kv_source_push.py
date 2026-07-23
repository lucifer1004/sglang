from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_CP_SIZE = 4
_ROW_WIDTH = 256
_VECTOR_BYTES = 16
_SIGNAL_BYTES = 8
_INT64_MAX = (1 << 63) - 1
_ALLOWED_THREAD_COUNTS = (64, 128, 256, 512)


@cache_once
def _jit_source_push_module() -> Module:
    args = make_cpp_args(torch.bfloat16, _CP_SIZE, _ROW_WIDTH)
    return load_jit(
        "prefill_cp_kv_source_push",
        *args,
        cuda_files=["distributed/prefill_cp_kv_source_push.cuh"],
        cuda_wrappers=[
            ("source_push", f"prefill_cp_kv_source_push<{args}>"),
            (
                "source_push_indexed",
                f"prefill_cp_kv_source_push_indexed<{args}>",
            ),
            ("wait_ready", f"prefill_cp_kv_wait_ready<{_CP_SIZE}>"),
            ("publish_epoch", f"prefill_cp_kv_publish_epoch<{_CP_SIZE}>"),
        ],
    )


def _require_cuda_bf16_rows(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must use torch.bfloat16")
    if tensor.ndim != 2 or tensor.shape[1] != _ROW_WIDTH:
        raise ValueError(f"{name} must have shape [rows, {_ROW_WIDTH}]")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % _VECTOR_BYTES:
        raise ValueError(f"{name} must be {_VECTOR_BYTES}-byte aligned")


def _require_non_negative_aligned(name: str, value: int, alignment: int) -> None:
    if value < 0 or value % alignment:
        raise ValueError(
            f"{name} must be non-negative and {alignment}-byte aligned"
        )


def _validate_peer_bases(peer_bases: torch.Tensor) -> None:
    if peer_bases.device.type != "cpu":
        raise ValueError("peer_bases must be a CPU tensor")
    if peer_bases.dtype != torch.int64 or peer_bases.numel() != _CP_SIZE:
        raise ValueError(f"peer_bases must contain {_CP_SIZE} int64 pointers")
    if not peer_bases.is_contiguous():
        raise ValueError("peer_bases must be contiguous")
    if any(pointer <= 0 for pointer in peer_bases.tolist()):
        raise ValueError("peer_bases must contain non-null pointers")


def _validate_signal_metadata(
    *, signal_offset_bytes: int, signal_stride_bytes: int, epoch: int
) -> None:
    if epoch <= 0 or epoch > _INT64_MAX:
        raise ValueError("epoch must fit in a positive int64")
    _require_non_negative_aligned(
        "signal_offset_bytes", signal_offset_bytes, _SIGNAL_BYTES
    )
    if signal_stride_bytes < _SIGNAL_BYTES:
        raise ValueError("signal_stride_bytes must fit one int64 signal")
    _require_non_negative_aligned(
        "signal_stride_bytes", signal_stride_bytes, _SIGNAL_BYTES
    )


def _validate_launch_metadata(
    *,
    key: torch.Tensor,
    peer_bases: torch.Tensor,
    destination_mask: int,
    k_offset_bytes: int,
    v_offset_bytes: int,
    signal_offset_bytes: int,
    signal_stride_bytes: int,
    completion: torch.Tensor,
    source_rank: int,
    epoch: int,
    num_threads: int,
) -> None:
    _validate_peer_bases(peer_bases)
    if not completion.is_cuda or completion.device != key.device:
        raise ValueError("completion must be a CUDA tensor on the K/V device")
    if completion.dtype != torch.int32 or completion.numel() != 1:
        raise ValueError("completion must contain one int32 counter")
    if not completion.is_contiguous():
        raise ValueError("completion must be contiguous")

    valid_destination_mask = (1 << _CP_SIZE) - 1
    if destination_mask <= 0 or destination_mask & ~valid_destination_mask:
        raise ValueError("destination_mask must select at least one CP4 rank")
    if source_rank < 0 or source_rank >= _CP_SIZE:
        raise ValueError("source_rank must be inside the CP4 group")
    if num_threads not in _ALLOWED_THREAD_COUNTS:
        raise ValueError(
            f"num_threads must be one of {_ALLOWED_THREAD_COUNTS}, got {num_threads}"
        )

    _require_non_negative_aligned("k_offset_bytes", k_offset_bytes, _VECTOR_BYTES)
    _require_non_negative_aligned("v_offset_bytes", v_offset_bytes, _VECTOR_BYTES)
    _validate_signal_metadata(
        signal_offset_bytes=signal_offset_bytes,
        signal_stride_bytes=signal_stride_bytes,
        epoch=epoch,
    )


def source_push(
    key: torch.Tensor,
    value: torch.Tensor,
    tiles: torch.Tensor,
    peer_bases: torch.Tensor,
    *,
    destination_mask: int,
    k_offset_bytes: int,
    v_offset_bytes: int,
    signal_offset_bytes: int,
    signal_stride_bytes: int,
    completion: torch.Tensor,
    source_rank: int,
    epoch: int,
    publish_signal: bool,
    num_threads: int = 256,
) -> None:
    """Push packed source K/V rows into selected peer IPC arenas.

    This first microbenchmark kernel intentionally supports only CP4, BF16,
    and one 256-element K/V row per token. ``tiles`` contains
    ``(src_start, dst_start, row_count)`` descriptors.
    """
    _require_cuda_bf16_rows("key", key)
    _require_cuda_bf16_rows("value", value)
    if key.shape != value.shape or key.device != value.device:
        raise ValueError("key and value must have identical shape and device")

    if not tiles.is_cuda or tiles.device != key.device:
        raise ValueError("tiles must be a CUDA tensor on the K/V device")
    if tiles.dtype != torch.int32 or tiles.ndim != 2 or tiles.shape[1] != 3:
        raise ValueError("tiles must have shape [num_tiles, 3] and dtype int32")
    if not tiles.is_contiguous():
        raise ValueError("tiles must be contiguous")
    if tiles.shape[0] == 0:
        return

    _validate_launch_metadata(
        key=key,
        peer_bases=peer_bases,
        destination_mask=destination_mask,
        k_offset_bytes=k_offset_bytes,
        v_offset_bytes=v_offset_bytes,
        signal_offset_bytes=signal_offset_bytes,
        signal_stride_bytes=signal_stride_bytes,
        completion=completion,
        source_rank=source_rank,
        epoch=epoch,
        num_threads=num_threads,
    )

    module = _jit_source_push_module()
    module.source_push(
        key,
        value,
        tiles,
        peer_bases,
        destination_mask,
        k_offset_bytes,
        v_offset_bytes,
        signal_offset_bytes,
        signal_stride_bytes,
        completion,
        source_rank,
        epoch,
        publish_signal,
        num_threads,
    )


def source_push_indexed(
    key: torch.Tensor,
    value: torch.Tensor,
    source_rows: torch.Tensor,
    destination_rows: torch.Tensor,
    peer_bases: torch.Tensor,
    *,
    destination_mask: int,
    k_offset_bytes: int,
    v_offset_bytes: int,
    signal_offset_bytes: int,
    signal_stride_bytes: int,
    completion: torch.Tensor,
    source_rank: int,
    epoch: int,
    publish_signal: bool,
    rows_per_block: int = 16,
    num_threads: int = 128,
) -> None:
    """Push arbitrary source K/V rows into final logical destination rows."""
    _require_cuda_bf16_rows("key", key)
    _require_cuda_bf16_rows("value", value)
    if key.shape != value.shape or key.device != value.device:
        raise ValueError("key and value must have identical shape and device")

    for name, rows in (
        ("source_rows", source_rows),
        ("destination_rows", destination_rows),
    ):
        if not rows.is_cuda or rows.device != key.device:
            raise ValueError(f"{name} must be a CUDA tensor on the K/V device")
        if rows.dtype != torch.int32 or rows.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional int32 tensor")
        if not rows.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if source_rows.shape != destination_rows.shape:
        raise ValueError("source_rows and destination_rows must have equal length")
    if source_rows.numel() == 0:
        return
    if rows_per_block <= 0:
        raise ValueError("rows_per_block must be positive")

    _validate_launch_metadata(
        key=key,
        peer_bases=peer_bases,
        destination_mask=destination_mask,
        k_offset_bytes=k_offset_bytes,
        v_offset_bytes=v_offset_bytes,
        signal_offset_bytes=signal_offset_bytes,
        signal_stride_bytes=signal_stride_bytes,
        completion=completion,
        source_rank=source_rank,
        epoch=epoch,
        num_threads=num_threads,
    )

    module = _jit_source_push_module()
    module.source_push_indexed(
        key,
        value,
        source_rows,
        destination_rows,
        peer_bases,
        destination_mask,
        k_offset_bytes,
        v_offset_bytes,
        signal_offset_bytes,
        signal_stride_bytes,
        completion,
        source_rank,
        epoch,
        publish_signal,
        rows_per_block,
        num_threads,
    )


def wait_ready(
    *,
    local_arena_base: int,
    source_mask: int,
    signal_offset_bytes: int,
    signal_stride_bytes: int,
    epoch: int,
) -> None:
    """Wait on the current CUDA stream until all selected sources publish."""
    if local_arena_base <= 0 or local_arena_base % _VECTOR_BYTES:
        raise ValueError("local_arena_base must be a non-null 16-byte-aligned pointer")
    valid_source_mask = (1 << _CP_SIZE) - 1
    if source_mask <= 0 or source_mask & ~valid_source_mask:
        raise ValueError("source_mask must select at least one CP4 rank")
    _validate_signal_metadata(
        signal_offset_bytes=signal_offset_bytes,
        signal_stride_bytes=signal_stride_bytes,
        epoch=epoch,
    )

    module = _jit_source_push_module()
    module.wait_ready(
        local_arena_base,
        source_mask,
        signal_offset_bytes,
        signal_stride_bytes,
        epoch,
        torch.cuda.current_device(),
    )


def publish_epoch(
    peer_bases: torch.Tensor,
    *,
    destination_mask: int,
    signal_offset_bytes: int,
    signal_stride_bytes: int,
    publisher_rank: int,
    epoch: int,
) -> None:
    """Publish an epoch from one rank into selected peer signal slots."""
    _validate_peer_bases(peer_bases)
    valid_destination_mask = (1 << _CP_SIZE) - 1
    if destination_mask <= 0 or destination_mask & ~valid_destination_mask:
        raise ValueError("destination_mask must select at least one CP4 rank")
    if publisher_rank < 0 or publisher_rank >= _CP_SIZE:
        raise ValueError("publisher_rank must be inside the CP4 group")
    _validate_signal_metadata(
        signal_offset_bytes=signal_offset_bytes,
        signal_stride_bytes=signal_stride_bytes,
        epoch=epoch,
    )

    module = _jit_source_push_module()
    module.publish_epoch(
        peer_bases,
        destination_mask,
        signal_offset_bytes,
        signal_stride_bytes,
        publisher_rank,
        epoch,
        torch.cuda.current_device(),
    )
