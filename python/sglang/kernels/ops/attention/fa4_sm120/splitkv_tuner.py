# Copyright (c) 2026, SGLang Team.
"""Startup-only measurement engine for calibrated SM120 FA4 SplitKV routing."""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import torch

from sglang.kernels.ops.attention.fa4_sm120.splitkv_fit import (
    SplitKvObservation,
    fit_splitkv_calibration,
)
from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
    SplitKvCalibration,
    SplitKvWorkload,
    ceil_div,
    effective_partitions,
    near_optimal_partitions,
    select_partition,
)
from sglang.kernels.ops.attention.fa4_sm120.splitkv_router import (
    SplitKvProbeSpec,
    SplitKvRouteSpec,
)

_CALIBRATION_N_TILE_GRID = (2, 4, 8, 16, 64, 128)
_DEFAULT_POOL_MIB = 512
_MIN_POOL_MIB = 256
_WARMUP = 2
_TIMED_BATCHES = 5
_COMBINE_CALLS_PER_BATCH = 32
_REFINE_WINDOW = 6
_MAX_MODEL_TO_ORACLE_RATIO = 1.10


class SplitKvTuningError(RuntimeError):
    """Raised when startup measurements cannot produce a safe route."""


@dataclass
class _Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    out: torch.Tensor
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None
    page_tables: list[torch.Tensor]


def _pool_mib() -> int:
    raw = os.getenv("SGLANG_FA4_SPLITKV_CALIBRATION_POOL_MIB")
    if raw is None:
        return _DEFAULT_POOL_MIB
    try:
        value = int(raw)
    except ValueError as error:
        raise SplitKvTuningError(
            "SGLANG_FA4_SPLITKV_CALIBRATION_POOL_MIB must be an integer"
        ) from error
    if value < _MIN_POOL_MIB:
        raise SplitKvTuningError(
            "SM120 FA4 SplitKV calibration pool must be at least "
            f"{_MIN_POOL_MIB} MiB"
        )
    return value


def _capture_graph_batch(
    calls: Sequence[Callable[[], None]],
) -> torch.cuda.CUDAGraph:
    if not calls:
        raise ValueError("at least one call is required")
    for _ in range(_WARMUP):
        for call in calls:
            call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    torch.cuda.synchronize()
    return graph


def _median_graph_replay_per_call_s(
    graph: torch.cuda.CUDAGraph,
    *,
    calls_per_replay: int,
) -> float:
    if calls_per_replay <= 0:
        raise ValueError("calls_per_replay must be positive")
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(_TIMED_BATCHES):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0 / calls_per_replay)
    return statistics.median(samples)


def _measure_combine_s(
    *,
    num_splits: int,
    output_rows: int,
    head_dim_v: int,
    device: torch.device,
) -> float:
    from sglang.kernels.ops.attention.flash_attn.cute.interface import (
        _flash_attn_fwd_combine,
    )

    out_partial = torch.empty(
        num_splits,
        1,
        output_rows,
        head_dim_v,
        device=device,
        dtype=torch.float32,
    )
    lse_partial = torch.empty(
        num_splits, 1, output_rows, device=device, dtype=torch.float32
    )
    out = torch.empty(
        1, output_rows, head_dim_v, device=device, dtype=torch.bfloat16
    )
    cu_seqlens = torch.tensor([0, 1], device=device, dtype=torch.int32)

    def call() -> None:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            cu_seqlens=cu_seqlens,
        )

    calls = [call] * _COMBINE_CALLS_PER_BATCH
    graph = _capture_graph_batch(calls)
    elapsed = _median_graph_replay_per_call_s(
        graph,
        calls_per_replay=len(calls),
    )
    del graph
    _ = (out_partial, lse_partial, out, cu_seqlens)
    return elapsed


def _dtype_for_route(route: SplitKvRouteSpec) -> torch.dtype:
    if route.compute != "bf16":
        raise SplitKvTuningError(
            f"startup tuning does not support compute type {route.compute!r}"
        )
    if route.kv_storage == "bf16":
        return torch.bfloat16
    if route.kv_storage == "fp8e4m3":
        return torch.float8_e4m3fn
    raise SplitKvTuningError(
        f"startup tuning does not support KV storage {route.kv_storage!r}"
    )


def _validate_probe_route(
    *,
    route: SplitKvRouteSpec,
    probe: SplitKvProbeSpec,
    kv_length: int,
    device: torch.device,
) -> None:
    from sglang.kernels.ops.attention.fa4_sm120.runtime import sm120_forward_host

    if route.page_size != probe.page_size:
        raise SplitKvTuningError(
            f"probe page size {probe.page_size} does not match route {route.page_size}"
        )
    config = sm120_forward_host.select_config(
        head_dim=route.head_dim,
        head_dim_v=route.head_dim_v,
        tile_mn=None,
        has_bias=False,
        total_q_rows=(
            probe.batch_size
            * probe.max_seqlen_q
            * probe.num_head_kv
            * probe.qhead_per_kvhead
        ),
        num_sms=torch.cuda.get_device_properties(device).multi_processor_count,
        num_batch=probe.batch_size,
        seqlen_q=probe.max_seqlen_q,
        seqlen_k=kv_length,
        num_head_kv=probe.num_head_kv,
        qhead_per_kvhead=probe.qhead_per_kvhead,
        is_causal=probe.causal,
        is_local=False,
        window_size_left=None,
        window_size_right=None,
        pack_gqa=True,
        paged_kv=True,
    )
    split_qk_n = sm120_forward_host.use_paged_decode_split_qk_n(
        head_dim=route.head_dim,
        head_dim_v=route.head_dim_v,
        paged_kv=True,
        max_seqlen_q=probe.max_seqlen_q,
        has_compact_q_groups=True,
        packed_q_rows=probe.max_seqlen_q * probe.qhead_per_kvhead,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        num_n_blocks=ceil_div(kv_length, config.tile_n),
        is_causal=probe.causal,
        is_local=False,
        has_score_or_mask_mod=False,
    )
    packed_q_rows = probe.max_seqlen_q * probe.qhead_per_kvhead
    direct_uniform_batch = sm120_forward_host.use_direct_uniform_batch(
        batch_size=probe.batch_size,
        paged_kv=True,
        has_cu_seqlens_q=True,
        has_seqused_q=False,
        total_q=probe.batch_size * probe.max_seqlen_q,
        max_seqlen_q=probe.max_seqlen_q,
        num_m_blocks=ceil_div(packed_q_rows, config.tile_m),
    )
    actual = (
        config.tile_m,
        config.tile_n,
        direct_uniform_batch,
        split_qk_n,
    )
    expected = (
        route.tile_m,
        route.tile_n,
        route.direct_uniform_batch,
        route.split_qk_n,
    )
    if actual != expected:
        raise SplitKvTuningError(
            f"probe resolves route {actual}, expected calibrated family {expected}"
        )


def _allocate_inputs(
    *,
    route: SplitKvRouteSpec,
    probe: SplitKvProbeSpec,
    kv_length: int,
    device: torch.device,
) -> _Inputs:
    props = torch.cuda.get_device_properties(device)
    l2_bytes = int(getattr(props, "L2_cache_size", 0) or 0)
    pool_bytes = _pool_mib() << 20
    if pool_bytes <= l2_bytes:
        raise SplitKvTuningError("calibration KV pool must be larger than L2")
    kv_dtype = _dtype_for_route(route)
    element_size = torch.empty((), dtype=kv_dtype).element_size()
    bytes_per_page_pair = (
        probe.page_size
        * probe.num_head_kv
        * (route.head_dim + route.head_dim_v)
        * element_size
    )
    num_pages = pool_bytes // bytes_per_page_pair
    pages_per_sequence = ceil_div(kv_length, probe.page_size)
    pages_per_call = probe.batch_size * pages_per_sequence
    footprint_bytes = (
        probe.batch_size
        * probe.num_head_kv
        * kv_length
        * (route.head_dim + route.head_dim_v)
        * element_size
    )
    available_sets = num_pages // pages_per_call
    required_sets = max(8, l2_bytes // max(footprint_bytes, 1) + 2)
    if available_sets < required_sets:
        raise SplitKvTuningError(
            "calibration pool cannot provide an L2-cold page-table ring: "
            f"available={available_sets}, required={required_sets}"
        )
    k_shape = (
        num_pages,
        probe.page_size,
        probe.num_head_kv,
        route.head_dim,
    )
    v_shape = (*k_shape[:-1], route.head_dim_v)
    k_cache = torch.zeros(k_shape, device=device, dtype=kv_dtype)
    v_cache = torch.zeros(v_shape, device=device, dtype=kv_dtype)
    permutation = torch.randperm(num_pages, device=device, dtype=torch.int64)
    page_tables = []
    for index in range(required_sets):
        begin = index * pages_per_call
        page_tables.append(
            permutation[begin : begin + pages_per_call]
            .to(torch.int32)
            .view(probe.batch_size, pages_per_sequence)
        )
    total_q = probe.batch_size * probe.max_seqlen_q
    num_heads = probe.num_head_kv * probe.qhead_per_kvhead
    q = torch.empty(
        total_q,
        num_heads,
        route.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    out = torch.empty(
        total_q,
        num_heads,
        route.head_dim_v,
        device=device,
        dtype=torch.bfloat16,
    )
    cache_seqlens = torch.full(
        (probe.batch_size,), kv_length, device=device, dtype=torch.int32
    )
    cu_seqlens_q = (
        torch.arange(probe.batch_size + 1, device=device, dtype=torch.int32)
        * probe.max_seqlen_q
    )
    scales = (
        torch.ones(
            probe.batch_size,
            probe.num_head_kv,
            device=device,
            dtype=torch.float32,
        )
        if kv_dtype == torch.float8_e4m3fn
        else None
    )
    return _Inputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        out=out,
        k_descale=scales,
        v_descale=scales,
        page_tables=page_tables,
    )


def _make_call(
    *,
    route: SplitKvRouteSpec,
    probe: SplitKvProbeSpec,
    inputs: _Inputs,
    page_table: torch.Tensor,
    kv_length: int,
    grain: int,
) -> Callable[[], None]:
    from sglang.kernels.ops.attention.fa4_sm120.runtime import (
        splitkv_calibration_partition,
    )

    if route.kv_storage == "fp8e4m3":
        from sglang.kernels.ops.attention.fa4_sm120.fp8_kv import (
            flash_attn_fp8_kv_sm120,
        )

        def launch() -> None:
            flash_attn_fp8_kv_sm120(
                inputs.q,
                inputs.k_cache,
                inputs.v_cache,
                page_table=page_table,
                cache_seqlens=inputs.cache_seqlens,
                cu_seqlens_q=inputs.cu_seqlens_q,
                max_seqlen_q=probe.max_seqlen_q,
                max_seqlen_k=kv_length,
                k_descale=inputs.k_descale,
                v_descale=inputs.v_descale,
                softmax_scale=route.head_dim**-0.5,
                causal=probe.causal,
                num_splits=0,
                pack_gqa=True,
                out=inputs.out,
            )

    else:
        from sglang.kernels.ops.attention.flash_attention_v4_sm120 import (
            flash_attn_with_kvcache,
        )

        def launch() -> None:
            flash_attn_with_kvcache(
                inputs.q,
                inputs.k_cache,
                inputs.v_cache,
                page_table=page_table,
                cache_seqlens=inputs.cache_seqlens,
                cu_seqlens_q=inputs.cu_seqlens_q,
                max_seqlen_q=probe.max_seqlen_q,
                max_seqlen_k=kv_length,
                softmax_scale=route.head_dim**-0.5,
                causal=probe.causal,
                num_splits=0,
                pack_gqa=True,
                out=inputs.out,
            )

    def call() -> None:
        with splitkv_calibration_partition(grain):
            launch()

    return call


def _measure_grain_s(
    *,
    route: SplitKvRouteSpec,
    probe: SplitKvProbeSpec,
    inputs: _Inputs,
    kv_length: int,
    grain: int,
) -> float:
    calls = [
        _make_call(
            route=route,
            probe=probe,
            inputs=inputs,
            page_table=page_table,
            kv_length=kv_length,
            grain=grain,
        )
        for page_table in inputs.page_tables
    ]
    graph = _capture_graph_batch(calls)
    elapsed = _median_graph_replay_per_call_s(
        graph,
        calls_per_replay=len(calls),
    )
    del graph
    return elapsed


def _calibration_grains(num_n_blocks: int) -> tuple[int, ...]:
    if num_n_blocks <= 16:
        return tuple(
            grain
            for grain, _ in effective_partitions(
                num_n_blocks,
                max_splits=128,
            )
        )
    targets = (
        num_n_blocks,
        ceil_div(num_n_blocks, 4),
        ceil_div(num_n_blocks, 8),
        ceil_div(num_n_blocks, 16),
        4,
        2,
        1,
    )
    return tuple(
        sorted(
            {
                grain
                for grain in targets
                if 1 <= grain <= num_n_blocks
                and ceil_div(num_n_blocks, grain) <= 128
            },
            reverse=True,
        )
    )


def calibrate_route_family(
    *,
    route: SplitKvRouteSpec,
    workload: SplitKvWorkload,
    probe: SplitKvProbeSpec,
    device: torch.device,
) -> SplitKvCalibration:
    """Fit route-family constants from a fixed DRAM-faithful probe grid."""
    if torch.cuda.is_current_stream_capturing():
        raise SplitKvTuningError("calibration cannot run during CUDA Graph capture")
    device = torch.device(device)
    # Family constants describe the kernel, not the capture batch that first
    # discovered it.  A single batch/KV-head group keeps the fixed probe pool
    # bounded and exposes both latency- and bandwidth-limited split counts.
    family_probe = replace(probe, batch_size=1, num_head_kv=1)
    packed_q_rows = family_probe.max_seqlen_q * family_probe.qhead_per_kvhead
    calibration_workloads = {
        num_n_blocks: SplitKvWorkload(
            total_mblocks=ceil_div(packed_q_rows, route.tile_m),
            num_n_blocks=num_n_blocks,
            main_bytes_per_kv_tile=workload.main_bytes_per_kv_tile,
            output_rows=packed_q_rows,
            head_dim_v=workload.head_dim_v,
        )
        for num_n_blocks in _CALIBRATION_N_TILE_GRID
    }
    combine_cache: dict[tuple[int, int], float] = {}

    def combine(num_splits: int, output_rows: int) -> float:
        key = (num_splits, output_rows)
        if key not in combine_cache:
            combine_cache[key] = _measure_combine_s(
                num_splits=num_splits,
                output_rows=output_rows,
                head_dim_v=route.head_dim_v,
                device=device,
            )
        return combine_cache[key]

    observations = []
    measured_totals: dict[int, dict[int, float]] = {}
    for num_n_blocks, calibration_workload in calibration_workloads.items():
        kv_length = route.tile_n * num_n_blocks
        _validate_probe_route(
            route=route,
            probe=family_probe,
            kv_length=kv_length,
            device=device,
        )
        inputs = _allocate_inputs(
            route=route,
            probe=family_probe,
            kv_length=kv_length,
            device=device,
        )
        scale_measurements = measured_totals.setdefault(num_n_blocks, {})
        for grain in _calibration_grains(num_n_blocks):
            num_splits = ceil_div(num_n_blocks, grain)
            total_s = _measure_grain_s(
                route=route,
                probe=family_probe,
                inputs=inputs,
                kv_length=kv_length,
                grain=grain,
            )
            scale_measurements[grain] = total_s
            combine_s = (
                0.0
                if num_splits == 1
                else combine(num_splits, calibration_workload.output_rows)
            )
            main_s = total_s - combine_s
            if main_s <= 0:
                raise SplitKvTuningError(
                    "non-positive main estimate at "
                    f"n_tiles={num_n_blocks}, grain={grain}: "
                    f"total={total_s}, combine={combine_s}"
                )
            observations.append(
                SplitKvObservation("main", calibration_workload, grain, main_s)
            )
            observations.append(
                SplitKvObservation(
                    "total",
                    calibration_workload,
                    grain,
                    total_s,
                )
            )
        del inputs
        torch.cuda.empty_cache()

    combine_rows = sorted(
        {
            8,
            16,
            32,
            64,
            128,
            workload.output_rows,
            packed_q_rows,
        }
    )
    combine_splits = sorted(
        {
            2,
            8,
            32,
            64,
            *(
                ceil_div(num_n_blocks, grain)
                for num_n_blocks in _CALIBRATION_N_TILE_GRID
                for grain in _calibration_grains(num_n_blocks)
            ),
        }
        - {1}
    )
    for output_rows in combine_rows:
        if output_rows <= 0:
            continue
        for num_splits in combine_splits:
            combine_workload = SplitKvWorkload(
                total_mblocks=1,
                num_n_blocks=64,
                main_bytes_per_kv_tile=workload.main_bytes_per_kv_tile,
                output_rows=output_rows,
                head_dim_v=route.head_dim_v,
            )
            observations.append(
                SplitKvObservation(
                    "combine",
                    combine_workload,
                    ceil_div(64, num_splits),
                    combine(num_splits, output_rows),
                )
            )
    props = torch.cuda.get_device_properties(device)
    fit = fit_splitkv_calibration(
        observations,
        sm_slots=int(props.multi_processor_count),
        l2_cache_bytes=int(getattr(props, "L2_cache_size", 0) or 0),
    )
    for num_n_blocks, calibration_workload in calibration_workloads.items():
        selected = select_partition(calibration_workload, fit.constants)
        scale_measurements = measured_totals[num_n_blocks]
        if selected.kv_tiles_per_cta not in scale_measurements:
            kv_length = route.tile_n * num_n_blocks
            inputs = _allocate_inputs(
                route=route,
                probe=family_probe,
                kv_length=kv_length,
                device=device,
            )
            scale_measurements[selected.kv_tiles_per_cta] = _measure_grain_s(
                route=route,
                probe=family_probe,
                inputs=inputs,
                kv_length=kv_length,
                grain=selected.kv_tiles_per_cta,
            )
            del inputs
            torch.cuda.empty_cache()
        ratio = scale_measurements[selected.kv_tiles_per_cta] / min(
            scale_measurements.values()
        )
        measured_best_grain = min(
            scale_measurements,
            key=lambda grain: (scale_measurements[grain], -grain),
        )
        near_optimal_grains = {
            prediction.kv_tiles_per_cta
            for prediction in near_optimal_partitions(
                calibration_workload,
                fit.constants,
                relative_tolerance=_MAX_MODEL_TO_ORACLE_RATIO - 1.0,
            )
        }
        if (
            ratio > _MAX_MODEL_TO_ORACLE_RATIO
            and measured_best_grain not in near_optimal_grains
        ):
            raise SplitKvTuningError(
                "calibrated model misses its measured probe oracle: "
                f"n_tiles={num_n_blocks}, "
                f"selected_grain={selected.kv_tiles_per_cta}, "
                f"measured_best_grain={measured_best_grain}, "
                f"measurements={scale_measurements}, "
                f"all_measurements={measured_totals}, "
                f"constants={fit.constants}, ratio={ratio:.3f}, "
                f"limit={_MAX_MODEL_TO_ORACLE_RATIO:.3f}"
            )
    return fit.constants


def refine_route_workload(
    *,
    route: SplitKvRouteSpec,
    workload: SplitKvWorkload,
    probe: SplitKvProbeSpec,
    constants: SplitKvCalibration,
    device: torch.device,
) -> int:
    """Measure a bounded exact-workload window around the model proposal."""
    if torch.cuda.is_current_stream_capturing():
        raise SplitKvTuningError("refinement cannot run during CUDA Graph capture")
    centers = {
        prediction.kv_tiles_per_cta
        for prediction in near_optimal_partitions(
            workload,
            constants,
            relative_tolerance=_MAX_MODEL_TO_ORACLE_RATIO - 1.0,
        )
    }
    _validate_probe_route(
        route=route,
        probe=probe,
        kv_length=probe.max_seqlen_k,
        device=torch.device(device),
    )
    candidates = tuple(
        grain
        for grain, _ in effective_partitions(
            workload.num_n_blocks, max_splits=workload.max_splits
        )
        if any(abs(grain - center) <= _REFINE_WINDOW for center in centers)
    )
    if not candidates:
        return select_partition(workload, constants).kv_tiles_per_cta
    inputs = _allocate_inputs(
        route=route,
        probe=probe,
        kv_length=probe.max_seqlen_k,
        device=torch.device(device),
    )
    measured = {
        grain: _measure_grain_s(
            route=route,
            probe=probe,
            inputs=inputs,
            kv_length=probe.max_seqlen_k,
            grain=grain,
        )
        for grain in candidates
    }
    return min(candidates, key=lambda grain: (measured[grain], -grain))
