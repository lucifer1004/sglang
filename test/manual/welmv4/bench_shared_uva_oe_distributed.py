#!/usr/bin/env python3
"""Distributed prefill/decode benchmark for shared host-resident WeLM OE weights."""

from __future__ import annotations

import argparse
import ctypes
import errno
import gc
import json
import mmap
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from bench_shared_uva_oe_numa import (
    OeWeightSpec,
    load_oe_specs,
    load_projection_weight,
    parse_int_list,
    query_gpu_numa_nodes,
    query_sampled_numa_nodes,
    run_lookup_concat,
    shard_bounds,
    summarize_samples,
)


@dataclass(frozen=True)
class Topology:
    attn_tp_size: int
    world_size: int = 8
    gpu_numa_nodes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.attn_tp_size <= 0 or self.world_size <= 0:
            raise ValueError("attention TP and world size must be positive")
        if self.world_size % self.attn_tp_size != 0:
            raise ValueError("attention TP size must divide world size")
        if self.gpu_numa_nodes and len(self.gpu_numa_nodes) != self.world_size:
            raise ValueError("GPU NUMA map must contain every world rank")

    @property
    def cp_size(self) -> int:
        return self.world_size // self.attn_tp_size

    @property
    def group_ranks(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(range(start, start + self.attn_tp_size))
            for start in range(0, self.world_size, self.attn_tp_size)
        )

    def tp_rank(self, global_rank: int) -> int:
        self._validate_rank(global_rank)
        return global_rank % self.attn_tp_size

    def cp_rank(self, global_rank: int) -> int:
        self._validate_rank(global_rank)
        return global_rank // self.attn_tp_size

    def gpu_numa(self, global_rank: int) -> int:
        self._validate_rank(global_rank)
        if self.gpu_numa_nodes:
            return self.gpu_numa_nodes[global_rank]
        return 0 if global_rank < self.world_size // 2 else 1

    def shard_consumers(self, tp_rank: int) -> tuple[int, ...]:
        if tp_rank < 0 or tp_rank >= self.attn_tp_size:
            raise ValueError("TP rank is out of range")
        return tuple(range(tp_rank, self.world_size, self.attn_tp_size))

    def _validate_rank(self, global_rank: int) -> None:
        if global_rank < 0 or global_rank >= self.world_size:
            raise ValueError("global rank is out of range")


def physical_shard_keys(
    topology: Topology, placement: str
) -> tuple[tuple[int, int], ...]:
    """Return unique ``(tp_rank, numa_node)`` physical shard copies."""
    if placement in {"global-numa0", "global-numa1"}:
        node = 0 if placement.endswith("0") else 1
        return tuple((tp_rank, node) for tp_rank in range(topology.attn_tp_size))
    if placement in {"paired-numa", "paired-global"}:
        return tuple(
            (tp_rank, numa_node)
            for tp_rank in range(topology.attn_tp_size)
            for numa_node in (0, 1)
        )
    if placement != "local-numa":
        raise ValueError(f"unknown placement policy {placement!r}")

    keys = {
        (tp_rank, topology.gpu_numa(consumer))
        for tp_rank in range(topology.attn_tp_size)
        for consumer in topology.shard_consumers(tp_rank)
    }
    return tuple(sorted(keys))


def physical_storage_bytes(
    topology: Topology,
    placement: str,
    shard_bytes: int,
) -> int:
    if shard_bytes <= 0:
        raise ValueError("shard_bytes must be positive")
    return len(physical_shard_keys(topology, placement)) * shard_bytes


def worker_shard_key(
    topology: Topology, placement: str, global_rank: int
) -> tuple[int, int]:
    tp_rank = topology.tp_rank(global_rank)
    if placement == "global-numa0":
        return tp_rank, 0
    if placement == "global-numa1":
        return tp_rank, 1
    if placement == "local-numa":
        return tp_rank, topology.gpu_numa(global_rank)
    raise ValueError(f"unknown placement policy {placement!r}")


def worker_access_shard_keys(
    topology: Topology, placement: str, global_rank: int
) -> dict[str, tuple[int, int]]:
    if placement not in {"paired-numa", "paired-global"}:
        return {"default": worker_shard_key(topology, placement, global_rank)}

    tp_rank = topology.tp_rank(global_rank)
    if placement == "paired-global":
        return {"numa0": (tp_rank, 0), "numa1": (tp_rank, 1)}

    local_numa = topology.gpu_numa(global_rank)
    return {
        "local": (tp_rank, local_numa),
        "remote": (tp_rank, 1 - local_numa),
    }


def aggregate_group_samples(
    topology: Topology,
    rank_samples: Mapping[int, Sequence[float]],
) -> dict[int, tuple[float, ...]]:
    """Aggregate synchronized rank samples into per-group critical-path samples."""
    if set(rank_samples) != set(range(topology.world_size)):
        raise ValueError("rank_samples must contain every global rank")
    sample_counts = {len(values) for values in rank_samples.values()}
    if len(sample_counts) != 1:
        raise ValueError("every rank must have the same number of samples")

    return {
        cp_rank: tuple(
            max(rank_samples[rank][index] for rank in ranks)
            for index in range(next(iter(sample_counts)))
        )
        for cp_rank, ranks in enumerate(topology.group_ranks)
    }


def iteration_access_order(
    access_names: Sequence[str], iteration: int
) -> tuple[str, ...]:
    ordered = tuple(access_names)
    return ordered if iteration % 2 == 0 else tuple(reversed(ordered))


class SharedHostTensor:
    """Own one process-local CUDA registration of a shared file mapping."""

    def __init__(
        self,
        *,
        path: Path,
        shape: tuple[int, ...],
        dtype: Any,
        device: int,
        fd: int,
        mapping: mmap.mmap,
        cpu_tensor: Any,
        cuda_tensor: Any,
        host_ptr: int,
        nbytes: int,
    ) -> None:
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._fd = fd
        self._mapping = mapping
        self._cpu_tensor = cpu_tensor
        self.cuda_tensor = cuda_tensor
        self._host_ptr = host_ptr
        self.nbytes = nbytes
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        path: Path,
        shape: Sequence[int],
        dtype: Any,
        device: int,
    ) -> "SharedHostTensor":
        import torch
        import cuda.bindings.runtime as cuda_rt

        from sglang.jit_kernel.memory_allocator import _make_tensor_from_ptr
        from sglang.srt.utils import check_cuda_result

        path = Path(path)
        shape = tuple(int(value) for value in shape)
        element_size = torch.empty((), dtype=dtype).element_size()
        nbytes = element_size
        for dimension in shape:
            nbytes *= dimension
        if nbytes <= 0:
            raise ValueError("shared tensor must not be empty")
        if path.stat().st_size != nbytes:
            raise ValueError(
                f"shared file size {path.stat().st_size} does not match {nbytes}"
            )

        fd = os.open(path, os.O_RDWR)
        mapping = mmap.mmap(
            fd,
            nbytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        cpu_tensor = torch.frombuffer(mapping, dtype=dtype).view(shape)
        host_ptr = int(cpu_tensor.data_ptr())
        torch.cuda.set_device(device)
        check_cuda_result(
            cuda_rt.cudaHostRegister(
                host_ptr,
                nbytes,
                cuda_rt.cudaHostRegisterPortable | cuda_rt.cudaHostRegisterMapped,
            )
        )
        try:
            device_ptr = int(
                check_cuda_result(cuda_rt.cudaHostGetDevicePointer(host_ptr, 0))[0]
            )
            cuda_tensor = _make_tensor_from_ptr(
                device_ptr,
                shape,
                dtype,
                torch.device("cuda", device),
                lambda _ptr: None,
            )
        except BaseException:
            check_cuda_result(cuda_rt.cudaHostUnregister(host_ptr))
            del cpu_tensor
            mapping.close()
            os.close(fd)
            raise

        return cls(
            path=path,
            shape=shape,
            dtype=dtype,
            device=device,
            fd=fd,
            mapping=mapping,
            cpu_tensor=cpu_tensor,
            cuda_tensor=cuda_tensor,
            host_ptr=host_ptr,
            nbytes=nbytes,
        )

    def close(self) -> None:
        if self._closed:
            return
        import torch
        import cuda.bindings.runtime as cuda_rt

        from sglang.srt.utils import check_cuda_result

        torch.cuda.synchronize(self.device)
        self.cuda_tensor = None
        gc.collect()
        check_cuda_result(cuda_rt.cudaHostUnregister(self._host_ptr))
        self._cpu_tensor = None
        gc.collect()
        self._mapping.close()
        os.close(self._fd)
        self._closed = True

    def __enter__(self) -> "SharedHostTensor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class SharedShardFile:
    branch_index: int
    tp_rank: int
    numa_node: int
    path: Path
    shape: tuple[int, int]
    global_rows: int
    shard_start: int
    shard_end: int
    source_name: str
    source_path: Path

    @property
    def nbytes(self) -> int:
        return self.shape[0] * self.shape[1] * 2


@dataclass(frozen=True)
class WorkerConfig:
    checkpoint: Path
    topology: Topology
    placement: str
    shard_files: tuple[SharedShardFile, ...]
    projection_spec: OeWeightSpec
    weight_source: str
    id_distribution: str
    prefill_sizes: tuple[int, ...]
    decode_sizes: tuple[int, ...]
    reduce_orders: tuple[str, ...]
    warmups: int
    repeats: int
    master_port: int
    correctness_atol: float
    correctness_rtol: float


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError(f"empty CPU list {value!r}")
    return cpus


def _bind_process_to_numa(numa_node: int) -> None:
    cpu_list_path = Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
    os.sched_setaffinity(0, _parse_cpu_list(cpu_list_path.read_text()))


def _bind_memory_range_to_numa(data_ptr: int, nbytes: int, numa_node: int) -> None:
    """Apply an explicit MPOL_BIND policy before first-touching an mmap."""
    if data_ptr <= 0 or nbytes <= 0 or numa_node < 0:
        raise ValueError("invalid memory range or NUMA node")
    bits_per_word = ctypes.sizeof(ctypes.c_ulong) * 8
    required_bits = numa_node + 1
    word_count = (required_bits + bits_per_word - 1) // bits_per_word
    maxnode = word_count * bits_per_word
    nodemask = (ctypes.c_ulong * word_count)()
    nodemask[numa_node // bits_per_word] |= 1 << (numa_node % bits_per_word)

    libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    mbind = libnuma.mbind
    mbind.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_uint,
    ]
    mbind.restype = ctypes.c_long
    result = mbind(data_ptr, nbytes, 2, nodemask, maxnode, 0)  # MPOL_BIND
    if result == -1:
        error = ctypes.get_errno()
        if error == errno.EPERM:
            raise PermissionError(error, "mbind NUMA policy is not permitted")
        raise OSError(error, os.strerror(error))


def create_shared_shard_files(
    *,
    embedding_specs: Sequence[OeWeightSpec],
    topology: Topology,
    placement: str,
    root: Path,
    max_shard_rows: int,
) -> tuple[SharedShardFile, ...]:
    root.mkdir(parents=True, exist_ok=False)
    files: list[SharedShardFile] = []
    try:
        for tp_rank, numa_node in physical_shard_keys(topology, placement):
            for branch_index, spec in enumerate(embedding_specs):
                shard_start, real_end, partition_rows = shard_bounds(
                    spec.shape[0], topology.attn_tp_size, tp_rank
                )
                allocation_rows = (
                    min(partition_rows, max_shard_rows)
                    if max_shard_rows > 0
                    else partition_rows
                )
                shard_end = min(real_end, shard_start + allocation_rows)
                if shard_end <= shard_start:
                    raise ValueError(
                        f"empty branch {branch_index} TP shard {tp_rank}"
                    )
                path = root / (
                    f"branch{branch_index}_tp{tp_rank}_numa{numa_node}.bf16"
                )
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                try:
                    os.ftruncate(fd, allocation_rows * spec.shape[1] * 2)
                finally:
                    os.close(fd)
                files.append(
                    SharedShardFile(
                        branch_index=branch_index,
                        tp_rank=tp_rank,
                        numa_node=numa_node,
                        path=path,
                        shape=(allocation_rows, spec.shape[1]),
                        global_rows=spec.shape[0],
                        shard_start=shard_start,
                        shard_end=shard_end,
                        source_name=spec.name,
                        source_path=spec.path,
                    )
                )
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return tuple(files)


def _populate_shard_files_worker(
    files: tuple[SharedShardFile, ...],
    numa_node: int,
    weight_source: str,
    status_queue,
) -> None:
    try:
        _bind_process_to_numa(numa_node)
        import torch
        from safetensors import safe_open

        torch.set_num_threads(max(1, min(16, len(os.sched_getaffinity(0)))))
        for file_spec in files:
            fd = os.open(file_spec.path, os.O_RDWR)
            mapping = mmap.mmap(
                fd,
                file_spec.nbytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            target = torch.frombuffer(mapping, dtype=torch.bfloat16).view(
                file_spec.shape
            )
            _bind_memory_range_to_numa(
                int(target.data_ptr()), file_spec.nbytes, numa_node
            )
            if weight_source == "synthetic":
                target.fill_(file_spec.branch_index + 1)
            elif weight_source == "checkpoint":
                with safe_open(
                    file_spec.source_path, framework="pt", device="cpu"
                ) as handle:
                    source = handle.get_slice(file_spec.source_name)[
                        file_spec.shard_start : file_spec.shard_end
                    ]
                    target[: source.shape[0]].copy_(source)
                    if source.shape[0] < target.shape[0]:
                        target[source.shape[0] :].zero_()
                    del source
            else:
                raise ValueError(f"unknown weight source {weight_source!r}")
            mapping.flush()
            del target
            gc.collect()
            mapping.close()
            os.close(fd)
        status_queue.put({"numa_node": numa_node, "ok": True})
    except BaseException:
        status_queue.put(
            {
                "numa_node": numa_node,
                "ok": False,
                "error": traceback.format_exc(),
            }
        )


def populate_shared_shard_files(
    files: Sequence[SharedShardFile], weight_source: str
) -> None:
    context = mp.get_context("spawn")
    by_node: dict[int, list[SharedShardFile]] = {}
    for file_spec in files:
        by_node.setdefault(file_spec.numa_node, []).append(file_spec)

    for numa_node, node_files in sorted(by_node.items()):
        status_queue = context.Queue()
        process = context.Process(
            target=_populate_shard_files_worker,
            args=(tuple(node_files), numa_node, weight_source, status_queue),
        )
        process.start()
        status = status_queue.get(timeout=3600)
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join()
            raise RuntimeError("NUMA shard loader did not exit")
        if not status["ok"] or process.exitcode:
            detail = status.get("error", "loader exited abnormally")
            raise RuntimeError(f"shared OE shard population failed:\n{detail}")


def sample_shared_file_placements(
    files: Sequence[SharedShardFile], max_samples: int = 16
) -> dict[str, dict[int, int]]:
    import torch

    placements = {}
    for file_spec in files:
        fd = os.open(file_spec.path, os.O_RDWR)
        mapping = mmap.mmap(
            fd,
            file_spec.nbytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        tensor = torch.frombuffer(mapping, dtype=torch.bfloat16)
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = max(1, (file_spec.nbytes + page_size - 1) // page_size)
        sample_count = min(page_count, max_samples)
        page_indices = {
            min(page_count - 1, index * page_count // sample_count)
            for index in range(sample_count)
        }
        for page_index in page_indices:
            mapping[page_index * page_size]
        placements[str(file_spec.path)] = query_sampled_numa_nodes(
            tensor.data_ptr(), file_spec.nbytes, max_samples=max_samples
        )
        del tensor
        mapping.close()
        os.close(fd)
    return placements


def _files_for_worker_accesses(
    config: WorkerConfig, global_rank: int
) -> dict[str, tuple[SharedShardFile, ...]]:
    selected_by_access = {}
    for access, key in worker_access_shard_keys(
        config.topology, config.placement, global_rank
    ).items():
        selected = tuple(
            sorted(
                (
                    file_spec
                    for file_spec in config.shard_files
                    if (file_spec.tp_rank, file_spec.numa_node) == key
                ),
                key=lambda file_spec: file_spec.branch_index,
            )
        )
        if len(selected) != 4:
            raise RuntimeError(
                f"worker {global_rank} access {access!r} resolved "
                f"{len(selected)} OE files"
            )
        selected_by_access[access] = selected
    return selected_by_access


def _make_worker_hashed_batches(
    *,
    device: int,
    topology: Topology,
    shard_files: Sequence[SharedShardFile],
    token_count: int,
    batch_count: int,
    distribution: str,
    seed: int,
) -> list[list[Any]]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches = []
    for _ in range(batch_count):
        branches = []
        for branch_file in shard_files:
            if distribution == "global":
                values = torch.randint(
                    0,
                    branch_file.global_rows,
                    (token_count,),
                    generator=generator,
                    dtype=torch.int64,
                )
            elif distribution == "covered-shards":
                values = torch.empty(token_count, dtype=torch.int64)
                # Choose ownership independently for every OE branch, matching
                # independent n-gram hashes while guaranteeing that truncated
                # smoke-test shards still serve every generated lookup.
                owners = torch.randint(
                    0,
                    topology.attn_tp_size,
                    (token_count,),
                    generator=generator,
                    dtype=torch.int64,
                )
                for tp_rank in range(topology.attn_tp_size):
                    start, real_end, partition_rows = shard_bounds(
                        branch_file.global_rows, topology.attn_tp_size, tp_rank
                    )
                    rows = min(branch_file.shape[0], partition_rows, real_end - start)
                    mask = owners == tp_rank
                    values[mask] = torch.randint(
                        start,
                        start + rows,
                        (int(mask.sum()),),
                        generator=generator,
                        dtype=torch.int64,
                    )
            else:
                raise ValueError(f"unknown ID distribution {distribution!r}")
            branches.append(values.to(device, non_blocking=False))
        batches.append(branches)
    return batches


def _time_distributed_operation(
    *,
    device: int,
    barrier,
    inputs: Sequence[Any],
    warmups: int,
    operation,
) -> dict[str, tuple[float, ...] | tuple[int, ...]]:
    import torch

    samples = []
    wall_start_ns = []
    wall_end_ns = []
    with torch.cuda.device(device):
        for index, value in enumerate(inputs):
            barrier.wait()
            host_start = time.perf_counter_ns()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = operation(value)
            end.record()
            end.synchronize()
            host_end = time.perf_counter_ns()
            if index >= warmups:
                samples.append(float(start.elapsed_time(end)))
                wall_start_ns.append(host_start)
                wall_end_ns.append(host_end)
            del output
            barrier.wait()
    return {
        "samples_ms": tuple(samples),
        "wall_start_ns": tuple(wall_start_ns),
        "wall_end_ns": tuple(wall_end_ns),
    }


def _time_distributed_operations_by_access(
    *,
    device: int,
    barrier,
    inputs: Sequence[Any],
    warmups: int,
    operations: Mapping[str, Any],
) -> dict[str, dict[str, tuple[float, ...] | tuple[int, ...]]]:
    import torch

    samples = {access: [] for access in operations}
    wall_start_ns = {access: [] for access in operations}
    wall_end_ns = {access: [] for access in operations}
    access_names = tuple(operations)
    with torch.cuda.device(device):
        for index, value in enumerate(inputs):
            for access in iteration_access_order(access_names, index):
                barrier.wait()
                host_start = time.perf_counter_ns()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = operations[access](value)
                end.record()
                end.synchronize()
                host_end = time.perf_counter_ns()
                if index >= warmups:
                    samples[access].append(float(start.elapsed_time(end)))
                    wall_start_ns[access].append(host_start)
                    wall_end_ns[access].append(host_end)
                del output
                barrier.wait()
    return {
        access: {
            "samples_ms": tuple(samples[access]),
            "wall_start_ns": tuple(wall_start_ns[access]),
            "wall_end_ns": tuple(wall_end_ns[access]),
        }
        for access in operations
    }


def _distributed_worker(global_rank: int, config: WorkerConfig, barrier, result_queue):
    shared_tensors: list[SharedHostTensor] = []
    try:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(global_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{config.master_port}",
            rank=global_rank,
            world_size=config.topology.world_size,
            device_id=torch.device("cuda", global_rank),
        )
        groups = [
            dist.new_group(list(ranks), backend="nccl")
            for ranks in config.topology.group_ranks
        ]
        process_group = groups[config.topology.cp_rank(global_rank)]

        worker_files_by_access = _files_for_worker_accesses(config, global_rank)
        modules_by_access = {}
        for access, worker_files in worker_files_by_access.items():
            access_shared_tensors = [
                SharedHostTensor.open(
                    path=file_spec.path,
                    shape=file_spec.shape,
                    dtype=torch.bfloat16,
                    device=global_rank,
                )
                for file_spec in worker_files
            ]
            shared_tensors.extend(access_shared_tensors)
            modules_by_access[access] = [
                SimpleNamespace(
                    weight=shared.cuda_tensor,
                    shard_indices=SimpleNamespace(
                        org_vocab_start_index=file_spec.shard_start,
                        org_vocab_end_index=file_spec.shard_end,
                    ),
                )
                for shared, file_spec in zip(access_shared_tensors, worker_files)
            ]
        representative_files = next(iter(worker_files_by_access.values()))
        projection_weight = load_projection_weight(
            config.projection_spec, global_rank, config.weight_source
        )

        records = []
        correctness = []
        total_batches = config.warmups + config.repeats
        for mode, sizes in (
            ("prefill", config.prefill_sizes),
            ("decode", config.decode_sizes),
        ):
            for token_count in sizes:
                batches = _make_worker_hashed_batches(
                    device=global_rank,
                    topology=config.topology,
                    shard_files=representative_files,
                    token_count=token_count,
                    batch_count=total_batches,
                    distribution=config.id_distribution,
                    seed=(1 if mode == "decode" else 2) * 1_000_003 + token_count,
                )
                projection_input = torch.randn(
                    (token_count, projection_weight.shape[1]),
                    dtype=torch.bfloat16,
                    device=global_rank,
                )
                collective_input = torch.zeros_like(projection_input)

                barrier.wait()
                reference_partial = None
                for access, modules in modules_by_access.items():
                    local_partial = run_lookup_concat(modules, batches[0])
                    if reference_partial is None:
                        reference_partial = local_partial.clone()
                    else:
                        copy_max_abs = float(
                            (reference_partial.float() - local_partial.float())
                            .abs()
                            .max()
                            .item()
                        )
                        copy_equivalent = torch.equal(reference_partial, local_partial)
                        correctness.append(
                            {
                                "mode": mode,
                                "token_count": token_count,
                                "access": access,
                                "check": "numa_copy",
                                "max_abs": copy_max_abs,
                                "max_ref": float(
                                    reference_partial.float().abs().max().item()
                                ),
                                "relative": 0.0,
                                "equivalent": bool(copy_equivalent),
                            }
                        )
                        if not copy_equivalent:
                            raise RuntimeError(
                                f"NUMA OE copies differ: mode={mode} "
                                f"tokens={token_count} max_abs={copy_max_abs}"
                            )

                    pre_output = local_partial.clone()
                    dist.all_reduce(pre_output, group=process_group)
                    pre_output = torch.nn.functional.linear(
                        pre_output, projection_weight
                    )
                    post_output = torch.nn.functional.linear(
                        local_partial, projection_weight
                    )
                    dist.all_reduce(post_output, group=process_group)
                    max_abs = float(
                        (pre_output.float() - post_output.float()).abs().max().item()
                    )
                    max_ref = float(pre_output.float().abs().max().item())
                    # BF16 changes rounding depending on whether TP reduction
                    # happens before or after GEMM. Compare the worst error
                    # against the output's global scale instead of applying an
                    # elementwise relative tolerance near zero.
                    equivalent = max_abs <= (
                        config.correctness_atol
                        + config.correctness_rtol * max_ref
                    )
                    correctness.append(
                        {
                            "mode": mode,
                            "token_count": token_count,
                            "access": access,
                            "check": "reduce_order",
                            "max_abs": max_abs,
                            "max_ref": max_ref,
                            "relative": max_abs / max_ref if max_ref else 0.0,
                            "equivalent": bool(equivalent),
                        }
                    )
                    del local_partial, pre_output, post_output
                    if not equivalent:
                        raise RuntimeError(
                            f"pre/post projection reduce mismatch: mode={mode} "
                            f"tokens={token_count} access={access} "
                            f"max_abs={max_abs} max_ref={max_ref}"
                        )
                del reference_partial
                barrier.wait()

                access_operations: dict[str, dict[str, Any]] = {
                    "lookup": {},
                }
                if "pre-proj" in config.reduce_orders:
                    access_operations["oe_total_pre_proj_reduce"] = {}
                if "post-proj" in config.reduce_orders:
                    access_operations["oe_total_post_proj_reduce"] = {}
                for access, modules in modules_by_access.items():
                    access_operations["lookup"][access] = (
                        lambda value, modules=modules: run_lookup_concat(modules, value)
                    )
                    if "pre-proj" in config.reduce_orders:
                        access_operations["oe_total_pre_proj_reduce"][access] = (
                            lambda value, modules=modules: _run_pre_proj_reduce(
                                modules, value, projection_weight, process_group
                            )
                        )
                    if "post-proj" in config.reduce_orders:
                        access_operations["oe_total_post_proj_reduce"][access] = (
                            lambda value, modules=modules: _run_post_proj_reduce(
                                modules, value, projection_weight, process_group
                            )
                        )

                for scope, operations in access_operations.items():
                    timings_by_access = _time_distributed_operations_by_access(
                        device=global_rank,
                        barrier=barrier,
                        inputs=batches,
                        warmups=config.warmups,
                        operations=operations,
                    )
                    for access, timing in timings_by_access.items():
                        records.append(
                            {
                                "mode": mode,
                                "token_count": token_count,
                                "scope": scope,
                                "access": access,
                                **timing,
                            }
                        )

                shared_operations = {
                    "all_reduce": lambda _value: _run_all_reduce(
                        collective_input, process_group
                    ),
                    "projection": lambda _value: torch.nn.functional.linear(
                        projection_input, projection_weight
                    ),
                }
                for scope, operation in shared_operations.items():
                    timing = _time_distributed_operation(
                        device=global_rank,
                        barrier=barrier,
                        inputs=batches,
                        warmups=config.warmups,
                        operation=operation,
                    )
                    records.append(
                        {
                            "mode": mode,
                            "token_count": token_count,
                            "scope": scope,
                            "access": "shared",
                            **timing,
                        }
                    )
                del batches, projection_input, collective_input

        dist.barrier()
        result_queue.put(
            {
                "rank": global_rank,
                "tp_rank": config.topology.tp_rank(global_rank),
                "cp_rank": config.topology.cp_rank(global_rank),
                "gpu_numa": config.topology.gpu_numa(global_rank),
                "access_shard_numa": {
                    access: key[1]
                    for access, key in worker_access_shard_keys(
                        config.topology, config.placement, global_rank
                    ).items()
                },
                "records": records,
                "correctness": correctness,
            }
        )
    except BaseException:
        try:
            barrier.abort()
        except BaseException:
            pass
        result_queue.put({"rank": global_rank, "error": traceback.format_exc()})
    finally:
        for shared in shared_tensors:
            try:
                shared.close()
            except BaseException:
                pass
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except BaseException:
            pass


def _run_all_reduce(tensor, process_group):
    import torch.distributed as dist

    dist.all_reduce(tensor, group=process_group)
    return tensor


def _run_pre_proj_reduce(modules, hashed_inputs, projection_weight, process_group):
    import torch

    local = run_lookup_concat(modules, hashed_inputs)
    _run_all_reduce(local, process_group)
    return torch.nn.functional.linear(local, projection_weight)


def _run_post_proj_reduce(modules, hashed_inputs, projection_weight, process_group):
    import torch

    local = run_lookup_concat(modules, hashed_inputs)
    output = torch.nn.functional.linear(local, projection_weight)
    _run_all_reduce(output, process_group)
    return output


def parse_reduce_orders(value: str) -> list[str]:
    orders = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"pre-proj", "post-proj"}
    if not orders or len(set(orders)) != len(orders) or any(
        order not in allowed for order in orders
    ):
        raise ValueError(f"reduce orders must be unique values from {sorted(allowed)}")
    return orders


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attn-tp-size", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument(
        "--placement",
        choices=(
            "global-numa0",
            "global-numa1",
            "local-numa",
            "paired-numa",
            "paired-global",
        ),
        default="global-numa0",
    )
    parser.add_argument(
        "--prefill-sizes", type=parse_int_list, default=[256, 1024, 4096, 16384]
    )
    parser.add_argument(
        "--decode-sizes", type=parse_int_list, default=[1, 8, 32, 128]
    )
    parser.add_argument(
        "--reduce-orders",
        type=parse_reduce_orders,
        default=["pre-proj", "post-proj"],
    )
    parser.add_argument(
        "--weight-source", choices=("checkpoint", "synthetic"), default="checkpoint"
    )
    parser.add_argument(
        "--id-distribution",
        choices=("global", "covered-shards"),
        default="global",
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument(
        "--max-shard-rows",
        type=int,
        default=0,
        help="Truncate rank shards for smoke tests; 0 uses checkpoint shapes.",
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("oe-bench-results/unfused-distributed"),
    )
    parser.add_argument("--correctness-atol", type=float, default=0.5)
    parser.add_argument("--correctness-rtol", type=float, default=0.02)
    parser.add_argument("--skip-numa-placement-check", action="store_true")
    return parser


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def query_gpu_state() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    result = []
    for line in output.splitlines():
        index, used, free, utilization = (
            int(value.strip()) for value in line.split(",")
        )
        result.append(
            {
                "index": index,
                "memory_used_mib": used,
                "memory_free_mib": free,
                "utilization_percent": utilization,
            }
        )
    return result


def run_distributed_workers(config: WorkerConfig) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    barrier = context.Barrier(config.topology.world_size)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, config, barrier, result_queue),
        )
        for rank in range(config.topology.world_size)
    ]
    for process in processes:
        process.start()

    results = []
    try:
        for _ in processes:
            results.append(result_queue.get(timeout=7200))
    finally:
        for process in processes:
            process.join(timeout=60)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

    errors = [result for result in results if "error" in result]
    bad_exits = [process.exitcode for process in processes if process.exitcode]
    if errors or bad_exits:
        detail = errors[0]["error"] if errors else f"worker exits={bad_exits}"
        raise RuntimeError(f"distributed OE worker failed:\n{detail}")
    return sorted(results, key=lambda result: result["rank"])


def aggregate_worker_results(
    topology: Topology, worker_results: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_case: dict[tuple[str, int, str, str], dict[int, dict[str, Any]]] = {}
    for worker in worker_results:
        rank = int(worker["rank"])
        for record in worker["records"]:
            key = (
                record["mode"],
                int(record["token_count"]),
                record["scope"],
                record.get("access", "default"),
            )
            by_case.setdefault(key, {})[rank] = record

    aggregated = []
    for (mode, token_count, scope, access), rank_records in sorted(by_case.items()):
        rank_samples = {
            rank: record["samples_ms"] for rank, record in rank_records.items()
        }
        group_samples = aggregate_group_samples(topology, rank_samples)
        sample_count = len(next(iter(rank_samples.values())))
        global_samples = tuple(
            max(rank_samples[rank][index] for rank in range(topology.world_size))
            for index in range(sample_count)
        )
        flattened_group_samples = tuple(
            sample
            for cp_rank in range(topology.cp_size)
            for sample in group_samples[cp_rank]
        )
        global_summary = summarize_samples(global_samples)
        group_summary = summarize_samples(flattened_group_samples)
        group_wall_samples = {
            cp_rank: tuple(
                (
                    max(rank_records[rank]["wall_end_ns"][index] for rank in ranks)
                    - min(
                        rank_records[rank]["wall_start_ns"][index] for rank in ranks
                    )
                )
                / 1_000_000.0
                for index in range(sample_count)
            )
            for cp_rank, ranks in enumerate(topology.group_ranks)
        }
        global_wall_samples = tuple(
            (
                max(
                    rank_records[rank]["wall_end_ns"][index]
                    for rank in range(topology.world_size)
                )
                - min(
                    rank_records[rank]["wall_start_ns"][index]
                    for rank in range(topology.world_size)
                )
            )
            / 1_000_000.0
            for index in range(sample_count)
        )
        flattened_group_wall_samples = tuple(
            sample
            for cp_rank in range(topology.cp_size)
            for sample in group_wall_samples[cp_rank]
        )
        global_wall_summary = summarize_samples(global_wall_samples)
        group_wall_summary = summarize_samples(flattened_group_wall_samples)
        requests_per_second = (
            topology.cp_size * 1000.0 / global_wall_summary["median_ms"]
        )
        tokens_per_second = requests_per_second * token_count
        aggregated.append(
            {
                "type": "aggregate",
                "mode": mode,
                "token_count": token_count,
                "scope": scope,
                "access": access,
                "group_critical_path": group_summary,
                "global_critical_path": global_summary,
                "group_wall": group_wall_summary,
                "global_wall": global_wall_summary,
                "per_group": {
                    str(cp_rank): summarize_samples(samples)
                    for cp_rank, samples in group_samples.items()
                },
                "per_group_wall": {
                    str(cp_rank): summarize_samples(samples)
                    for cp_rank, samples in group_wall_samples.items()
                },
                "requests_per_second": requests_per_second,
                "tokens_per_second": tokens_per_second,
            }
        )
    return aggregated


def render_markdown(
    *, metadata: dict[str, Any], aggregated: Sequence[dict[str, Any]]
) -> str:
    lines = [
        "# WeLM Distributed Shared-UVA OE Results",
        "",
        f"- Topology: attention-TP{metadata['attn_tp_size']} + CP{metadata['cp_size']}",
        f"- Placement: `{metadata['placement']}`",
        f"- Physical OE bytes: {metadata['physical_weight_bytes'] / 1024**3:.2f} GiB",
        f"- Weight source: `{metadata['weight_source']}`",
        f"- OE path: `{metadata['oe_path']}`",
        "",
        "| Mode | Tokens | Scope | Access | Group CUDA ms | Global CUDA ms | "
        "Global wall ms | Wall p99 ms | Requests/s |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = {"prefill": 0, "decode": 1}
    for record in sorted(
        aggregated,
        key=lambda item: (
            order[item["mode"]],
            item["token_count"],
            item["scope"],
            item["access"],
        ),
    ):
        group = record["group_critical_path"]
        global_path = record["global_critical_path"]
        global_wall = record["global_wall"]
        lines.append(
            f"| {record['mode']} | {record['token_count']} | {record['scope']} | "
            f"{record['access']} | "
            f"{group['median_ms']:.6f} | {global_path['median_ms']:.6f} | "
            f"{global_wall['median_ms']:.6f} | {global_wall['p99_ms']:.6f} | "
            f"{record['requests_per_second']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _jsonable_placements(
    placements: Mapping[str, Mapping[int, int]]
) -> dict[str, dict[str, int]]:
    return {
        path: {str(node): count for node, count in nodes.items()}
        for path, nodes in placements.items()
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.warmups < 1 or args.repeats < 1:
        raise ValueError("warmups and repeats must be positive")
    if args.max_shard_rows < 0:
        raise ValueError("max-shard-rows must be non-negative")
    if args.max_shard_rows > 0 and args.id_distribution == "global":
        raise ValueError(
            "truncated shards require --id-distribution covered-shards; "
            "global IDs would benchmark mostly zero-fill misses"
        )

    import torch

    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"benchmark needs {args.world_size} GPUs, found {torch.cuda.device_count()}"
        )
    gpu_numa_nodes = query_gpu_numa_nodes(args.world_size)
    topology = Topology(args.attn_tp_size, args.world_size, gpu_numa_nodes)
    embedding_specs, projection_spec = load_oe_specs(args.checkpoint)
    gpu_state_before = query_gpu_state()
    run_id = f"welm_oe_{os.getpid()}_{int(time.time())}"
    shared_root = Path("/dev/shm") / run_id
    setup_start = time.perf_counter()
    shard_files = create_shared_shard_files(
        embedding_specs=embedding_specs,
        topology=topology,
        placement=args.placement,
        root=shared_root,
        max_shard_rows=args.max_shard_rows,
    )
    try:
        populate_shared_shard_files(shard_files, args.weight_source)
        placements = sample_shared_file_placements(shard_files)
        if not args.skip_numa_placement_check:
            for file_spec in shard_files:
                placement = placements[str(file_spec.path)]
                if set(placement) != {file_spec.numa_node}:
                    raise RuntimeError(
                        f"{file_spec.path} pages are not on NUMA{file_spec.numa_node}: "
                        f"{placement}"
                    )
        setup_seconds = time.perf_counter() - setup_start

        config = WorkerConfig(
            checkpoint=args.checkpoint,
            topology=topology,
            placement=args.placement,
            shard_files=shard_files,
            projection_spec=projection_spec,
            weight_source=args.weight_source,
            id_distribution=args.id_distribution,
            prefill_sizes=tuple(args.prefill_sizes),
            decode_sizes=tuple(args.decode_sizes),
            reduce_orders=tuple(args.reduce_orders),
            warmups=args.warmups,
            repeats=args.repeats,
            master_port=args.master_port or _find_free_port(),
            correctness_atol=args.correctness_atol,
            correctness_rtol=args.correctness_rtol,
        )
        worker_results = run_distributed_workers(config)
        aggregated = aggregate_worker_results(topology, worker_results)

        metadata = {
            "type": "metadata",
            "checkpoint": str(args.checkpoint),
            "attn_tp_size": topology.attn_tp_size,
            "cp_size": topology.cp_size,
            "world_size": topology.world_size,
            "gpu_numa_nodes": gpu_numa_nodes,
            "placement": args.placement,
            "weight_source": args.weight_source,
            "id_distribution": args.id_distribution,
            "max_shard_rows": args.max_shard_rows,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "physical_weight_bytes": sum(file.nbytes for file in shard_files),
            "full_checkpoint_oe_bytes": sum(
                spec.shape[0] * spec.shape[1] * 2 for spec in embedding_specs
            ),
            "setup_seconds": setup_seconds,
            "gpu_state_before": gpu_state_before,
            "gpu_state_after": query_gpu_state(),
            "sampled_page_placements": _jsonable_placements(placements),
            "numa_placement_check_skipped": args.skip_numa_placement_check,
            "oe_path": "unfused_prehashed_lookup_linear_collective",
            "shared_files": [
                {
                    "branch_index": file.branch_index,
                    "tp_rank": file.tp_rank,
                    "numa_node": file.numa_node,
                    "shape": list(file.shape),
                    "shard_start": file.shard_start,
                    "shard_end": file.shard_end,
                    "bytes": file.nbytes,
                }
                for file in shard_files
            ],
            "correctness": {
                str(worker["rank"]): worker["correctness"]
                for worker in worker_results
            },
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"distributed_oe_attntp{topology.attn_tp_size}_cp{topology.cp_size}_"
            f"{args.placement}_{timestamp}"
        )
        jsonl_path = args.output_dir / f"{stem}.jsonl"
        markdown_path = args.output_dir / f"{stem}.md"
        with jsonl_path.open("w") as output:
            output.write(json.dumps(metadata) + "\n")
            for worker in worker_results:
                output.write(json.dumps({"type": "worker", **worker}) + "\n")
            for record in aggregated:
                output.write(json.dumps(record) + "\n")
        markdown = render_markdown(metadata=metadata, aggregated=aggregated)
        markdown_path.write_text(markdown)
        print(markdown)
        print(f"JSONL: {jsonl_path}")
        print(f"Markdown: {markdown_path}")
    finally:
        shutil.rmtree(shared_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
