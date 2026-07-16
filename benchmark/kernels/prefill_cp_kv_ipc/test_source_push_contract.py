from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_source_push import (  # noqa: E402
    CopyRun,
    build_copy_tiles,
    build_fragmented_slots,
    build_fragmented_prefix_runs,
    build_fragmented_zigzag_runs,
    build_rank_packed_to_logical,
    build_rotated_zigzag_runs,
    expand_copy_runs_to_indices,
)


def test_rotated_zigzag_runs_cover_two_owner_blocks() -> None:
    runs = build_rotated_zigzag_runs(
        total_rows=32768,
        cp_size=4,
        rank=1,
        rotation=0,
    )

    assert runs == [
        CopyRun(src_start=0, dst_start=4096, row_count=4096),
        CopyRun(src_start=4096, dst_start=24576, row_count=4096),
    ]


def test_rotated_zigzag_runs_apply_owner_rotation() -> None:
    runs = build_rotated_zigzag_runs(
        total_rows=32768,
        cp_size=4,
        rank=0,
        rotation=1,
    )

    assert runs == [
        CopyRun(src_start=0, dst_start=12288, row_count=4096),
        CopyRun(src_start=4096, dst_start=16384, row_count=4096),
    ]


def test_copy_tiles_split_runs_and_preserve_short_tail() -> None:
    tiles = build_copy_tiles(
        [
            CopyRun(src_start=10, dst_start=100, row_count=5),
            CopyRun(src_start=30, dst_start=300, row_count=2),
        ],
        tile_rows=4,
    )

    torch.testing.assert_close(
        tiles,
        torch.tensor(
            [
                [10, 100, 4],
                [14, 104, 1],
                [30, 300, 2],
            ],
            dtype=torch.int32,
        ),
    )


def test_copy_tiles_accept_empty_runs_without_placeholder() -> None:
    tiles = build_copy_tiles([], tile_rows=8)

    assert tiles.dtype == torch.int32
    assert tuple(tiles.shape) == (0, 3)


def test_indexed_rows_preserve_fragmented_source_and_logical_destination() -> None:
    source_rows, destination_rows = expand_copy_runs_to_indices(
        [
            CopyRun(src_start=32, dst_start=200, row_count=3),
            CopyRun(src_start=80, dst_start=203, row_count=2),
            CopyRun(src_start=17, dst_start=205, row_count=1),
        ]
    )

    assert source_rows.dtype == torch.int32
    assert destination_rows.dtype == torch.int32
    assert source_rows.tolist() == [32, 33, 34, 80, 81, 17]
    assert destination_rows.tolist() == [200, 201, 202, 203, 204, 205]


def test_indexed_rows_accept_empty_segments() -> None:
    source_rows, destination_rows = expand_copy_runs_to_indices([])

    assert source_rows.dtype == torch.int32
    assert destination_rows.dtype == torch.int32
    assert tuple(source_rows.shape) == (0,)
    assert tuple(destination_rows.shape) == (0,)


def test_fragmented_prefix_runs_coalesce_only_adjacent_physical_slots() -> None:
    runs = build_fragmented_prefix_runs(
        physical_slots=[32, 33, 34, 80, 81, 17],
        logical_start=200,
    )

    assert runs == [
        CopyRun(src_start=32, dst_start=200, row_count=3),
        CopyRun(src_start=80, dst_start=203, row_count=2),
        CopyRun(src_start=17, dst_start=205, row_count=1),
    ]


def test_fragmented_zigzag_runs_keep_global_logical_destinations() -> None:
    runs = build_fragmented_zigzag_runs(
        total_rows=16,
        cp_size=2,
        rank=0,
        rotation=0,
        physical_slots=[10, 11, 30, 31, 50, 70, 71, 72],
    )

    assert runs == [
        CopyRun(src_start=10, dst_start=0, row_count=2),
        CopyRun(src_start=30, dst_start=2, row_count=2),
        CopyRun(src_start=50, dst_start=12, row_count=1),
        CopyRun(src_start=70, dst_start=13, row_count=3),
    ]


def test_fragmented_slots_preserve_rows_but_insert_physical_gaps() -> None:
    slots = build_fragmented_slots(row_count=10, fragment_rows=4)

    assert slots == [0, 1, 2, 3, 5, 6, 7, 8, 11, 12]


def test_rank_packed_to_logical_matches_collective_rank_order() -> None:
    mapping = build_rank_packed_to_logical(
        total_rows=16,
        cp_size=2,
        rotation=0,
        logical_offset=100,
    )

    torch.testing.assert_close(
        mapping,
        torch.tensor(
            [
                100,
                101,
                102,
                103,
                112,
                113,
                114,
                115,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
            ],
            dtype=torch.int64,
        ),
    )


def test_invalid_tile_rows_fail_explicitly() -> None:
    try:
        build_copy_tiles([CopyRun(0, 0, 1)], tile_rows=0)
    except ValueError as exc:
        assert "tile_rows must be positive" in str(exc)
    else:
        raise AssertionError("zero tile_rows did not fail")
