"""GPU row materialization for sharded-KV context-parallel prefill."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.context_parallel.prefill_layout import CPPrefillSplitSpec


_DUMMY_PHYSICAL_SLOT = 0


@dataclass(frozen=True)
class CPQueryBlock:
    local_start: int
    token_count: int
    logical_start: int
    visible_kv_end: int
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor

    @property
    def local_end(self) -> int:
        return self.local_start + self.token_count


@dataclass(frozen=True)
class CPPrefillRuntimeLayout:
    spec: CPPrefillSplitSpec
    cp_rank: int
    local_logical_indices: torch.Tensor
    local_extend_indices: torch.Tensor
    local_input_ids: torch.Tensor
    local_positions: torch.Tensor
    local_out_cache_loc: torch.Tensor
    q_blocks: tuple[CPQueryBlock, ...]
    active_local_tokens: int
    active_per_rank_tokens: tuple[int, ...]
    kv_local_tokens: int
    kv_per_rank_tokens: tuple[int, ...]
    q_is_contracted: bool

    def active_tokens_per_cp_rank(self) -> tuple[int, ...]:
        return self.active_per_rank_tokens

    def local_index_for_logical(self, logical_index: int) -> int | None:
        for block in self.q_blocks:
            offset = logical_index - block.logical_start
            if 0 <= offset < block.token_count:
                return block.local_start + offset
        return None


def materialize_cp_prefill_runtime_layout(
    *,
    spec: CPPrefillSplitSpec,
    cp_rank: int,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> CPPrefillRuntimeLayout:
    """Select this CP rank's model rows without changing logical batch state."""
    cp_size = len(spec.per_rank_tokens)
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    _validate_token_axis(input_ids, spec.extend_len, "input_ids", dim=0)
    _validate_token_axis(positions, spec.extend_len, "positions", dim=0)
    _validate_token_axis(out_cache_loc, spec.extend_len, "out_cache_loc", dim=0)
    if not (input_ids.device == positions.device == out_cache_loc.device):
        raise ValueError("prefill runtime tensors must be on the same device")

    device = input_ids.device
    local_blocks = spec.local_blocks(cp_rank)
    index_parts = tuple(
        torch.arange(
            block.logical_start - spec.extend_start,
            block.logical_start - spec.extend_start + block.token_count,
            dtype=torch.int64,
            device=device,
        )
        for block in local_blocks
    )
    local_extend_indices = (
        torch.cat(index_parts)
        if index_parts
        else torch.empty((0,), dtype=torch.int64, device=device)
    )
    local_logical_indices = local_extend_indices + spec.extend_start

    q_blocks = []
    local_start = 0
    for block in local_blocks:
        q_blocks.append(
            CPQueryBlock(
                local_start=local_start,
                token_count=block.token_count,
                logical_start=block.logical_start,
                visible_kv_end=block.logical_start + block.token_count,
                cu_seqlens_q=torch.tensor(
                    [0, block.token_count], dtype=torch.int32, device=device
                ),
                cu_seqlens_k=torch.tensor(
                    [0, block.logical_start + block.token_count],
                    dtype=torch.int32,
                    device=device,
                ),
            )
        )
        local_start += block.token_count
    active_local_tokens = spec.per_rank_tokens[cp_rank]
    if local_start != active_local_tokens:
        raise ValueError("local CP blocks do not match per-rank token count")

    local_out_cache_loc = out_cache_loc.index_select(0, local_extend_indices)
    _assert_no_dummy_write_slot(local_out_cache_loc)
    return CPPrefillRuntimeLayout(
        spec=spec,
        cp_rank=cp_rank,
        local_logical_indices=local_logical_indices,
        local_extend_indices=local_extend_indices,
        local_input_ids=input_ids.index_select(0, local_extend_indices),
        local_positions=positions.index_select(0, local_extend_indices),
        local_out_cache_loc=local_out_cache_loc,
        q_blocks=tuple(q_blocks),
        active_local_tokens=active_local_tokens,
        active_per_rank_tokens=spec.per_rank_tokens,
        kv_local_tokens=active_local_tokens,
        kv_per_rank_tokens=spec.per_rank_tokens,
        q_is_contracted=False,
    )


def contract_cp_prefill_runtime_to_last_q(
    runtime: CPPrefillRuntimeLayout,
    *,
    has_active_q: bool = True,
) -> CPPrefillRuntimeLayout:
    """Contract Q/hidden rows while retaining the original owner-local K/V rows."""
    if runtime.q_is_contracted:
        return runtime
    if runtime.spec.extend_len <= 0:
        raise ValueError("KV mirror contraction requires a non-empty extend")

    last_logical = runtime.spec.extend_start + runtime.spec.extend_len - 1
    last_owner = runtime.spec.blocks[-1].owner_rank
    active_per_rank_tokens = tuple(
        1 if has_active_q and rank == last_owner else 0
        for rank in range(len(runtime.spec.per_rank_tokens))
    )
    local_index = (
        runtime.local_index_for_logical(last_logical)
        if has_active_q and runtime.cp_rank == last_owner
        else None
    )
    if has_active_q and runtime.cp_rank == last_owner and local_index is None:
        raise RuntimeError("last-Q owner cannot resolve its local logical row")

    if local_index is None:
        active_indices = torch.empty(
            (0,), dtype=torch.int64, device=runtime.local_positions.device
        )
        q_blocks = ()
    else:
        active_indices = torch.tensor(
            [local_index], dtype=torch.int64, device=runtime.local_positions.device
        )
        q_blocks = (
            CPQueryBlock(
                local_start=0,
                token_count=1,
                logical_start=last_logical,
                visible_kv_end=last_logical + 1,
                cu_seqlens_q=torch.tensor(
                    [0, 1], dtype=torch.int32, device=runtime.local_positions.device
                ),
                cu_seqlens_k=torch.tensor(
                    [0, last_logical + 1],
                    dtype=torch.int32,
                    device=runtime.local_positions.device,
                ),
            ),
        )

    return CPPrefillRuntimeLayout(
        spec=runtime.spec,
        cp_rank=runtime.cp_rank,
        local_logical_indices=runtime.local_logical_indices.index_select(
            0, active_indices
        ),
        local_extend_indices=runtime.local_extend_indices.index_select(
            0, active_indices
        ),
        local_input_ids=runtime.local_input_ids.index_select(0, active_indices),
        local_positions=runtime.local_positions.index_select(0, active_indices),
        local_out_cache_loc=runtime.local_out_cache_loc,
        q_blocks=q_blocks,
        active_local_tokens=active_per_rank_tokens[runtime.cp_rank],
        active_per_rank_tokens=active_per_rank_tokens,
        kv_local_tokens=runtime.kv_local_tokens,
        kv_per_rank_tokens=runtime.kv_per_rank_tokens,
        q_is_contracted=True,
    )


def _validate_token_axis(
    tensor: torch.Tensor,
    expected_tokens: int,
    name: str,
    *,
    dim: int,
) -> None:
    if tensor.ndim <= dim or tensor.shape[dim] != expected_tokens:
        shape = tuple(tensor.shape)
        raise ValueError(
            f"{name} must contain {expected_tokens} global extend rows, got {shape}"
        )


def _assert_no_dummy_write_slot(local_out_cache_loc: torch.Tensor) -> None:
    if local_out_cache_loc.numel() == 0:
        return
    has_dummy = torch.any(local_out_cache_loc == _DUMMY_PHYSICAL_SLOT)
    if local_out_cache_loc.device.type == "cpu":
        if bool(has_dummy):
            raise ValueError("CP-local token resolved to a dummy physical write slot")
        return
    torch._assert_async(
        torch.logical_not(has_dummy),
        "CP-local token resolved to a dummy physical write slot",
    )
