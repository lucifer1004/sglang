"""CPU contract tests for the Phase 2 context-parallel prefill split plan."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _layout_module():
    return import_module("sglang.srt.context_parallel.prefill_layout")


def _runtime_module():
    return import_module("sglang.srt.context_parallel.prefill_runtime")


def _plan(
    extend_start: int,
    extend_len: int,
    *,
    cp_size: int = 4,
    page_size: int = 16,
    owner_rotation: int = 0,
    leading_page_owner: int | None = None,
):
    return _layout_module().build_cp_prefill_split_spec(
        extend_start=extend_start,
        extend_len=extend_len,
        cp_size=cp_size,
        page_size=page_size,
        owner_rotation=owner_rotation,
        leading_page_owner=leading_page_owner,
    )


def test_cp4_32k_uses_eight_page_aligned_balanced_blocks():
    plan = _plan(extend_start=0, extend_len=32 * 1024)

    assert [(block.logical_start, block.token_count) for block in plan.blocks] == [
        (index * 4096, 4096) for index in range(8)
    ]
    assert [block.owner_rank for block in plan.blocks] == [0, 1, 2, 3, 3, 2, 1, 0]
    assert plan.per_rank_tokens == (8192, 8192, 8192, 8192)
    assert plan.page_demand(page_size=16) == (512, 512, 512, 512)


@pytest.mark.parametrize("owner_rotation", range(4))
def test_rotation_changes_only_block_owners(owner_rotation: int):
    plan = _plan(
        extend_start=0,
        extend_len=32 * 1024,
        owner_rotation=owner_rotation,
    )

    assert [(block.logical_start, block.token_count) for block in plan.blocks] == [
        (index * 4096, 4096) for index in range(8)
    ]
    assert [block.owner_rank for block in plan.blocks] == [
        (owner + owner_rotation) % 4 for owner in (0, 1, 2, 3, 3, 2, 1, 0)
    ]


def test_blocks_cover_each_scheduled_token_once():
    plan = _plan(
        extend_start=5,
        extend_len=91,
        owner_rotation=2,
        leading_page_owner=3,
    )

    covered_tokens = [
        token
        for block in plan.blocks
        for token in range(block.logical_start, block.logical_start + block.token_count)
    ]
    assert covered_tokens == list(range(plan.extend_start, plan.extend_start + plan.extend_len))


def test_per_rank_tokens_equal_the_sum_of_local_blocks():
    plan = _plan(
        extend_start=5,
        extend_len=91,
        owner_rotation=1,
        leading_page_owner=3,
    )

    assert tuple(
        sum(block.token_count for block in plan.local_blocks(cp_rank))
        for cp_rank in range(4)
    ) == plan.per_rank_tokens


def test_short_extend_allows_zero_token_ranks():
    plan = _plan(extend_start=0, extend_len=3)

    assert [(block.logical_start, block.token_count, block.owner_rank) for block in plan.blocks] == [
        (0, 3, 0)
    ]
    assert plan.per_rank_tokens == (3, 0, 0, 0)
    assert plan.local_blocks(1) == ()
    assert plan.page_demand(page_size=16) == (1, 0, 0, 0)


def test_chunk_continuation_reuses_the_request_rotation():
    first_chunk = _plan(extend_start=0, extend_len=48, owner_rotation=2)
    continuation = _plan(extend_start=48, extend_len=48, owner_rotation=2)

    assert first_chunk.owner_rotation == continuation.owner_rotation == 2
    assert [block.owner_rank for block in first_chunk.blocks] == [2, 3, 0]
    assert [block.owner_rank for block in continuation.blocks] == [2, 3, 0]


def test_leading_resident_partial_page_keeps_its_owner_and_is_not_demanded():
    plan = _plan(
        extend_start=5,
        extend_len=43,
        owner_rotation=1,
        leading_page_owner=3,
    )

    assert [(block.logical_start, block.token_count, block.owner_rank) for block in plan.blocks] == [
        (5, 11, 3),
        (16, 16, 1),
        (32, 16, 2),
    ]
    assert plan.page_demand(page_size=16) == (0, 1, 1, 0)


def test_trailing_partial_page_owner_is_reused_by_the_next_append():
    first_append = _plan(extend_start=0, extend_len=17, owner_rotation=2)
    trailing_owner = first_append.blocks[-1].owner_rank
    continuation = _plan(
        extend_start=17,
        extend_len=31,
        owner_rotation=2,
        leading_page_owner=trailing_owner,
    )

    assert first_append.blocks[-1].token_count == 1
    assert trailing_owner == 3
    assert continuation.blocks[0].logical_start == 17
    assert continuation.blocks[0].token_count == 15
    assert continuation.blocks[0].owner_rank == trailing_owner
    assert continuation.page_demand(page_size=16) == (0, 0, 1, 0)


def test_manual_spec_rejects_in_range_owner_outside_zigzag_order():
    module = _layout_module()
    spec = module.CPPrefillSplitSpec(
        0,
        24,
        0,
        (
            module.CPBlock(0, 8, 0),
            module.CPBlock(8, 8, 0),
            module.CPBlock(16, 8, 1),
        ),
        (16, 8),
    )

    with pytest.raises(ValueError, match="zigzag"):
        spec.validate(cp_size=2, page_size=8)


def test_manual_spec_rejects_leading_block_larger_than_resident_fragment():
    module = _layout_module()
    spec = module.CPPrefillSplitSpec(
        1,
        31,
        0,
        (module.CPBlock(1, 31, 1),),
        (0, 31),
    )

    with pytest.raises(ValueError, match="resident fragment"):
        spec.validate(cp_size=2, page_size=8)


@pytest.mark.parametrize("owner_rotation", [-1, 2])
def test_manual_spec_rejects_noncanonical_owner_rotation(owner_rotation: int):
    module = _layout_module()
    spec = module.CPPrefillSplitSpec(
        0,
        16,
        owner_rotation,
        (module.CPBlock(0, 8, 0), module.CPBlock(8, 8, 1)),
        (8, 8),
    )

    with pytest.raises(ValueError, match="owner_rotation"):
        spec.validate(cp_size=2, page_size=8)


def test_manual_spec_rejects_more_than_one_zigzag_period_of_new_blocks():
    module = _layout_module()
    spec = module.CPPrefillSplitSpec(
        0,
        40,
        0,
        (
            module.CPBlock(0, 8, 0),
            module.CPBlock(8, 8, 1),
            module.CPBlock(16, 8, 1),
            module.CPBlock(24, 8, 0),
            module.CPBlock(32, 8, 1),
        ),
        (16, 24),
    )

    with pytest.raises(ValueError, match=r"2 \* cp_size"):
        spec.validate(cp_size=2, page_size=8)


@pytest.mark.parametrize(
    "spec_factory",
    [
        lambda module: module.CPPrefillSplitSpec(
            0,
            16,
            0,
            (module.CPBlock(0, 8, 0), module.CPBlock(7, 8, 0)),
            (16, 0),
        ),
        lambda module: module.CPPrefillSplitSpec(
            0,
            16,
            0,
            (module.CPBlock(0, 8, 0), module.CPBlock(9, 7, 0)),
            (15, 0),
        ),
        lambda module: module.CPPrefillSplitSpec(
            0, 16, 0, (module.CPBlock(0, 16, 2),), (0, 16)
        ),
        lambda module: module.CPPrefillSplitSpec(
            0,
            16,
            0,
            (module.CPBlock(0, 7, 0), module.CPBlock(7, 9, 0)),
            (16, 0),
        ),
        lambda module: module.CPPrefillSplitSpec(
            0, 16, 0, (module.CPBlock(0, 15, 0),), (15, 0)
        ),
    ],
    ids=["overlap", "gap", "owner", "page_alignment", "length"],
)
def test_invalid_manual_specs_raise_immediately(spec_factory):
    with pytest.raises(ValueError):
        spec_factory(_layout_module()).validate(cp_size=2, page_size=8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"extend_start": -1, "extend_len": 1},
        {"extend_start": 0, "extend_len": -1},
        {"extend_start": 0, "extend_len": 1, "cp_size": 0},
        {"extend_start": 0, "extend_len": 1, "page_size": 0},
        {
            "extend_start": 5,
            "extend_len": 1,
            "leading_page_owner": 4,
        },
    ],
    ids=["negative_start", "negative_length", "cp_size", "page_size", "leading_owner"],
)
def test_invalid_planner_inputs_raise_immediately(kwargs):
    with pytest.raises(ValueError):
        _plan(**kwargs)


def _physical_slots_for_rank(plan, cp_rank: int) -> torch.Tensor:
    slots = torch.zeros(plan.extend_len, dtype=torch.int64)
    for block in plan.local_blocks(cp_rank):
        start = block.logical_start - plan.extend_start
        end = start + block.token_count
        slots[start:end] = torch.arange(start, end, dtype=torch.int64) + 100
    return slots


def test_runtime_materializes_three_discontiguous_local_query_blocks():
    plan = _plan(
        extend_start=5,
        extend_len=139,
        owner_rotation=0,
        leading_page_owner=0,
    )
    input_ids = torch.arange(plan.extend_len, dtype=torch.int64) + 1000
    positions = torch.arange(
        plan.extend_start,
        plan.extend_start + plan.extend_len,
        dtype=torch.int32,
    )
    out_cache_loc = _physical_slots_for_rank(plan, cp_rank=0)

    runtime = _runtime_module().materialize_cp_prefill_runtime_layout(
        spec=plan,
        cp_rank=0,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
    )

    expected_extend_indices = torch.cat(
        (
            torch.arange(0, 11),
            torch.arange(11, 27),
            torch.arange(123, 139),
        )
    )
    assert torch.equal(runtime.local_extend_indices, expected_extend_indices)
    assert torch.equal(
        runtime.local_logical_indices, expected_extend_indices + plan.extend_start
    )
    assert torch.equal(runtime.local_input_ids, input_ids[expected_extend_indices])
    assert torch.equal(runtime.local_positions, positions[expected_extend_indices])
    assert torch.equal(
        runtime.local_out_cache_loc, out_cache_loc[expected_extend_indices]
    )
    assert runtime.active_local_tokens == 43
    assert [
        (
            block.local_start,
            block.token_count,
            block.logical_start,
            block.visible_kv_end,
        )
        for block in runtime.q_blocks
    ] == [
        (0, 11, 5, 16),
        (11, 16, 16, 32),
        (27, 16, 128, 144),
    ]
    assert [block.cu_seqlens_q.tolist() for block in runtime.q_blocks] == [
        [0, 11],
        [0, 16],
        [0, 16],
    ]
    assert [block.cu_seqlens_k.tolist() for block in runtime.q_blocks] == [
        [0, 16],
        [0, 32],
        [0, 144],
    ]
    assert runtime.local_index_for_logical(5) == 0
    assert runtime.local_index_for_logical(31) == 26
    assert runtime.local_index_for_logical(128) == 27
    assert runtime.local_index_for_logical(32) is None


def test_runtime_materializes_empty_tensors_for_zero_token_rank():
    plan = _plan(extend_start=0, extend_len=3)
    input_ids = torch.arange(3, dtype=torch.int64)
    positions = torch.arange(3, dtype=torch.int32)
    out_cache_loc = _physical_slots_for_rank(plan, cp_rank=1)

    runtime = _runtime_module().materialize_cp_prefill_runtime_layout(
        spec=plan,
        cp_rank=1,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
    )

    assert runtime.active_local_tokens == 0
    assert runtime.q_blocks == ()
    assert runtime.local_extend_indices.shape == (0,)
    assert runtime.local_logical_indices.shape == (0,)
    assert runtime.local_input_ids.shape == (0,)
    assert runtime.local_positions.shape == (0,)
    assert runtime.local_out_cache_loc.shape == (0,)


@pytest.mark.parametrize("owner_rotation", range(4))
def test_kv_mirror_runtime_contracts_q_to_last_owner_and_preserves_kv_rows(
    owner_rotation: int,
):
    plan = _plan(
        extend_start=16,
        extend_len=128,
        owner_rotation=owner_rotation,
    )
    input_ids = torch.arange(plan.extend_len, dtype=torch.int64) + 1000
    positions = torch.arange(
        plan.extend_start,
        plan.extend_start + plan.extend_len,
        dtype=torch.int32,
    )
    last_logical = plan.extend_start + plan.extend_len - 1
    last_owner = plan.blocks[-1].owner_rank
    expected_active_counts = tuple(
        1 if rank == last_owner else 0 for rank in range(4)
    )

    for cp_rank in range(4):
        runtime = _runtime_module().materialize_cp_prefill_runtime_layout(
            spec=plan,
            cp_rank=cp_rank,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=_physical_slots_for_rank(plan, cp_rank),
        )
        original_cache_loc = runtime.local_out_cache_loc.clone()
        original_kv_rows = runtime.active_local_tokens
        original_local_index = runtime.local_index_for_logical(last_logical)

        contracted = _runtime_module().contract_cp_prefill_runtime_to_last_q(
            runtime
        )

        assert contracted.active_per_rank_tokens == expected_active_counts
        assert contracted.kv_per_rank_tokens == plan.per_rank_tokens
        assert contracted.kv_local_tokens == original_kv_rows
        assert torch.equal(contracted.local_out_cache_loc, original_cache_loc)
        if cp_rank == last_owner:
            assert contracted.active_local_tokens == 1
            assert original_local_index is not None
            assert contracted.local_logical_indices.tolist() == [last_logical]
            assert contracted.local_positions.tolist() == [last_logical]
            assert len(contracted.q_blocks) == 1
            assert contracted.q_blocks[0].logical_start == last_logical
            assert contracted.q_blocks[0].visible_kv_end == last_logical + 1
        else:
            assert contracted.active_local_tokens == 0
            assert original_local_index is None
            assert contracted.local_logical_indices.numel() == 0
            assert contracted.local_positions.numel() == 0
            assert contracted.q_blocks == ()


def test_runtime_rejects_dummy_slot_for_locally_owned_token():
    plan = _plan(extend_start=0, extend_len=17)
    input_ids = torch.arange(17, dtype=torch.int64)
    positions = torch.arange(17, dtype=torch.int32)
    out_cache_loc = _physical_slots_for_rank(plan, cp_rank=0)
    out_cache_loc[0] = 0

    with pytest.raises(ValueError, match="dummy physical write slot"):
        _runtime_module().materialize_cp_prefill_runtime_layout(
            spec=plan,
            cp_rank=0,
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
        )


@pytest.mark.parametrize("field", ["input_ids", "positions", "out_cache_loc"])
def test_runtime_rejects_global_row_length_mismatch(field: str):
    plan = _plan(extend_start=0, extend_len=17)
    tensors = {
        "input_ids": torch.arange(17, dtype=torch.int64),
        "positions": torch.arange(17, dtype=torch.int32),
        "out_cache_loc": _physical_slots_for_rank(plan, cp_rank=0),
    }
    tensors[field] = tensors[field][:-1]

    with pytest.raises(ValueError, match=field):
        _runtime_module().materialize_cp_prefill_runtime_layout(
            spec=plan,
            cp_rank=0,
            **tensors,
        )


def test_sharded_kv_cp_split_requires_prefill_runtime_layout():
    cp_utils = import_module("sglang.srt.layers.utils.cp_utils")
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
        seq_lens_cpu=torch.tensor([8]),
        attn_cp_prefill_runtime_layout=object(),
    )

    with patch.object(
        cp_utils, "is_prefill_context_parallel_enabled", return_value=True
    ), patch.object(cp_utils, "is_cp_kv_sharded", return_value=True):
        assert cp_utils.can_cp_split(3, 4, forward_batch)
        forward_batch.attn_cp_prefill_runtime_layout = None
        assert not cp_utils.can_cp_split(3, 4, forward_batch)


def test_cp_attention_uses_explicit_unequal_query_blocks():
    cp_utils = import_module("sglang.srt.layers.utils.cp_utils")
    plan = _plan(
        extend_start=5,
        extend_len=139,
        owner_rotation=0,
        leading_page_owner=0,
    )
    runtime = _runtime_module().materialize_cp_prefill_runtime_layout(
        spec=plan,
        cp_rank=0,
        input_ids=torch.arange(plan.extend_len),
        positions=torch.arange(5, 144, dtype=torch.int32),
        out_cache_loc=_physical_slots_for_rank(plan, cp_rank=0),
    )
    q = torch.arange(runtime.active_local_tokens, dtype=torch.float32).view(-1, 1)
    calls = []

    def attn_fn(q_block, cu_seqlens_q, cache_seqlens, max_seqlen_q):
        calls.append(
            (
                q_block.clone(),
                cu_seqlens_q.tolist(),
                cache_seqlens.tolist(),
                max_seqlen_q,
            )
        )
        return q_block + 1

    result = cp_utils.cp_attn_forward_extend(
        SimpleNamespace(attn_cp_prefill_runtime_layout=runtime),
        q,
        torch.device("cpu"),
        attn_fn,
    )

    assert torch.equal(result, q + 1)
    assert [call[0].shape[0] for call in calls] == [11, 16, 16]
    assert [call[1] for call in calls] == [[0, 11], [0, 16], [0, 16]]
    assert [call[2] for call in calls] == [[16], [32], [144]]
    assert [call[3] for call in calls] == [11, 16, 16]


def test_cp_attention_accepts_empty_local_query_layout():
    cp_utils = import_module("sglang.srt.layers.utils.cp_utils")
    plan = _plan(extend_start=0, extend_len=3)
    runtime = _runtime_module().materialize_cp_prefill_runtime_layout(
        spec=plan,
        cp_rank=1,
        input_ids=torch.arange(3),
        positions=torch.arange(3, dtype=torch.int32),
        out_cache_loc=_physical_slots_for_rank(plan, cp_rank=1),
    )

    result = cp_utils.cp_attn_forward_extend(
        SimpleNamespace(attn_cp_prefill_runtime_layout=runtime),
        torch.empty((0, 2), dtype=torch.float32),
        torch.device("cpu"),
        lambda *_args: pytest.fail("empty local Q must not issue an attention call"),
    )

    assert result.shape == (0, 2)
