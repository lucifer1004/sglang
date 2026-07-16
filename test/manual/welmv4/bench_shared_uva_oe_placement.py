#!/usr/bin/env python3
"""Benchmark full shared WeLM base/OE embeddings under NUMA placement."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import mmap
import multiprocessing as mp
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from bench_shared_uva_oe_distributed import query_gpu_state
from bench_shared_uva_oe_numa import (
    OeWeightSpec,
    load_oe_specs,
    load_projection_weight,
    parse_int_list,
    query_gpu_numa_nodes,
    run_lookup_concat,
    summarize_samples,
)
from sglang.srt.model_loader.welm_shared_embedding_weights import (
    LinuxNumaPlacementAdapter,
    NumaPlacementMode,
    WeLMSharedEmbeddingPolicy,
    WeLMSharedEmbeddingRegistry,
    build_welm_shared_embedding_dry_run_report,
    calculate_welm_embedding_byte_counts,
    discover_welm_shared_embedding_checkpoint_tensors,
    launch_welm_embedding_arena_process,
    load_welm_embedding_arena_manifest,
    plan_welm_embedding_replicas,
)
from sglang.srt.utils.shared_uva_tensor import (
    SharedTensorFileSpec,
    SharedUVATensorView,
)


BASE_EMBEDDING_KEY = "model.embed_tokens.weight"
OE_EMBEDDING_KEYS = tuple(
    f"model.oe_embed.{index}.weight" for index in range(4)
)
SHARED_EMBEDDING_KEYS = (BASE_EMBEDDING_KEY, *OE_EMBEDDING_KEYS)


@dataclass(frozen=True)
class FullVocabTopology:
    input_tp_size: int
    world_size: int = 8
    gpu_numa_nodes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.input_tp_size <= 0 or self.world_size <= 0:
            raise ValueError("input TP and world size must be positive")
        if self.world_size % self.input_tp_size != 0:
            raise ValueError("input TP size must divide world size")
        if self.gpu_numa_nodes and len(self.gpu_numa_nodes) != self.world_size:
            raise ValueError("GPU NUMA map must contain every world rank")

    @property
    def cp_size(self) -> int:
        return self.world_size // self.input_tp_size

    @property
    def input_tp_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(range(start, start + self.input_tp_size))
            for start in range(0, self.world_size, self.input_tp_size)
        )

    def cp_rank(self, global_rank: int) -> int:
        self._validate_rank(global_rank)
        return global_rank // self.input_tp_size

    def input_tp_rank(self, global_rank: int) -> int:
        self._validate_rank(global_rank)
        return global_rank % self.input_tp_size

    def gpu_numa(self, global_rank: int) -> int:
        self._validate_rank(global_rank)
        if self.gpu_numa_nodes:
            return self.gpu_numa_nodes[global_rank]
        return 0 if global_rank < self.world_size // 2 else 1

    def _validate_rank(self, global_rank: int) -> None:
        if global_rank < 0 or global_rank >= self.world_size:
            raise ValueError("global rank is out of range")


@dataclass(frozen=True)
class PlacementPolicy:
    name: str
    mode: str
    nodes: tuple[int, ...]

    @classmethod
    def parse(cls, value: str) -> "PlacementPolicy":
        policies = {
            "bind-numa0": cls("bind-numa0", "bind", (0,)),
            "bind-numa1": cls("bind-numa1", "bind", (1,)),
            "interleave": cls("interleave", "interleave", (0, 1)),
            "replicate-numa": cls(
                "replicate-numa", "replicate", (0, 1)
            ),
        }
        try:
            return policies[value]
        except KeyError as exc:
            raise ValueError(f"unknown placement policy {value!r}") from exc

    @property
    def production_policy(self) -> WeLMSharedEmbeddingPolicy:
        if self.mode == "replicate":
            return WeLMSharedEmbeddingPolicy.REPLICATE_NUMA
        return WeLMSharedEmbeddingPolicy(self.mode)

    @property
    def bind_node(self) -> int | None:
        return self.nodes[0] if self.mode == "bind" else None


def local_prefill_tokens(global_tokens: int, cp_size: int) -> int:
    if global_tokens <= 0 or cp_size <= 0:
        raise ValueError("global tokens and CP size must be positive")
    if global_tokens % cp_size != 0:
        raise ValueError("global prefill tokens must be divisible by CP size")
    return global_tokens // cp_size


def worker_hash_seed(
    topology: FullVocabTopology,
    global_rank: int,
    mode: str,
    token_count: int,
) -> int:
    mode_seed = {"prefill": 1, "decode": 2}.get(mode)
    if mode_seed is None:
        raise ValueError(f"unknown mode {mode!r}")
    return mode_seed * 1_000_003 + token_count * 257 + topology.cp_rank(global_rank)


def full_table_storage_bytes(
    shapes: Sequence[tuple[int, int]], element_size: int = 2
) -> int:
    if element_size <= 0:
        raise ValueError("element size must be positive")
    return sum(rows * width * element_size for rows, width in shapes)


def critical_path_samples(
    topology: FullVocabTopology,
    rank_samples: Mapping[int, Sequence[float]],
) -> tuple[dict[int, tuple[float, ...]], tuple[float, ...]]:
    if set(rank_samples) != set(range(topology.world_size)):
        raise ValueError("rank samples must contain every rank")
    sample_counts = {len(samples) for samples in rank_samples.values()}
    if len(sample_counts) != 1:
        raise ValueError("every rank must contain the same sample count")
    sample_count = next(iter(sample_counts))
    pair_samples = {
        cp_rank: tuple(
            max(rank_samples[rank][index] for rank in ranks)
            for index in range(sample_count)
        )
        for cp_rank, ranks in enumerate(topology.input_tp_groups)
    }
    global_samples = tuple(
        max(rank_samples[rank][index] for rank in range(topology.world_size))
        for index in range(sample_count)
    )
    return pair_samples, global_samples


@dataclass(frozen=True)
class FullTableFile:
    table_index: int
    key: str
    path: Path
    shape: tuple[int, int]
    global_rows: int
    source_key: str
    source_path: Path

    @property
    def nbytes(self) -> int:
        return self.shape[0] * self.shape[1] * 2


@dataclass(frozen=True)
class TableReplica:
    replica_id: str
    numa_node: int | None
    numa_nodes: tuple[int, ...]
    table_files: tuple[FullTableFile, ...]


@dataclass(frozen=True)
class WorkerConfig:
    topology: FullVocabTopology
    arena_root: Path
    manifest_path: str | None
    table_replicas: tuple[TableReplica, ...]
    projection_spec: OeWeightSpec
    weight_source: str
    prefill_sizes: tuple[int, ...]
    decode_sizes: tuple[int, ...]
    warmups: int
    repeats: int
    source_row_checksums: Mapping[str, str]


def create_full_table_files(
    *,
    embedding_specs: Sequence[Any],
    root: Path,
    max_rows: int,
) -> tuple[FullTableFile, ...]:
    if tuple(spec.key for spec in embedding_specs) != SHARED_EMBEDDING_KEYS:
        raise ValueError("WeLM shared embedding specs must contain base then OE 0-3")
    if max_rows < 0:
        raise ValueError("max rows must be non-negative")
    root.mkdir(parents=True, exist_ok=False)
    files: list[FullTableFile] = []
    try:
        for table_index, spec in enumerate(embedding_specs):
            rows = min(spec.shape[0], max_rows) if max_rows else spec.shape[0]
            shape = (rows, spec.shape[1])
            path = root / f"table{table_index}_full.bf16"
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(fd, rows * spec.shape[1] * 2)
            finally:
                os.close(fd)
            files.append(
                FullTableFile(
                    table_index=table_index,
                    key=spec.key,
                    path=path,
                    shape=shape,
                    global_rows=spec.shape[0],
                    source_key=spec.key,
                    source_path=spec.path,
                )
            )
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return tuple(files)


def create_table_replicas(
    *,
    embedding_specs: Sequence[Any],
    root: Path,
    max_rows: int,
    policy: PlacementPolicy,
) -> tuple[TableReplica, ...]:
    if policy.mode != "replicate":
        return (
            TableReplica(
                replica_id=policy.name,
                numa_node=None,
                numa_nodes=policy.nodes,
                table_files=create_full_table_files(
                    embedding_specs=embedding_specs,
                    root=root,
                    max_rows=max_rows,
                ),
            ),
        )

    root.mkdir(parents=True, exist_ok=False)
    replicas = []
    try:
        for node in policy.nodes:
            replicas.append(
                TableReplica(
                    replica_id=f"replica-numa-{node}",
                    numa_node=node,
                    numa_nodes=(node,),
                    table_files=create_full_table_files(
                        embedding_specs=embedding_specs,
                        root=root / f"numa{node}",
                        max_rows=max_rows,
                    ),
                )
            )
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return tuple(replicas)


def table_replicas_from_manifest(
    manifest, embedding_specs: Sequence[Any]
) -> tuple[TableReplica, ...]:
    sources = {spec.key: spec for spec in embedding_specs}
    replicas = []
    for replica in manifest.replicas:
        files = []
        for table_index, tensor in enumerate(replica.tensors):
            source = sources[tensor.key]
            files.append(
                FullTableFile(
                    table_index=table_index,
                    key=tensor.key,
                    path=Path(tensor.path),
                    shape=tensor.shape,
                    global_rows=source.shape[0],
                    source_key=source.key,
                    source_path=source.path,
                )
            )
        replicas.append(
            TableReplica(
                replica_id=replica.replica_id,
                numa_node=(
                    replica.numa_nodes[0]
                    if manifest.policy == WeLMSharedEmbeddingPolicy.REPLICATE_NUMA.value
                    else None
                ),
                numa_nodes=replica.numa_nodes,
                table_files=tuple(files),
            )
        )
    return tuple(replicas)


def _tensor_bytes_digest(tensor) -> str:
    import torch

    byte_view = tensor.contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(byte_view)).hexdigest()


def checkpoint_row_checksums(embedding_specs: Sequence[Any]) -> dict[str, str]:
    import torch
    from safetensors import safe_open

    checksums = {}
    for source in embedding_specs:
        indices = (0, source.shape[0] // 2, source.shape[0] - 1)
        with safe_open(source.path, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(source.key)
            rows = torch.cat(
                [tensor_slice[index : index + 1] for index in indices], dim=0
            )
        checksums[source.key] = _tensor_bytes_digest(rows)
    return checksums


def synthetic_row_checksums(
    table_files: Sequence[FullTableFile],
) -> dict[str, str]:
    import torch

    return {
        file.key: _tensor_bytes_digest(
            torch.full(
                (3, file.shape[1]),
                file.table_index + 1,
                dtype=torch.bfloat16,
            )
        )
        for file in table_files
    }


def measure_resident_file_bytes(
    replicas: Sequence[TableReplica],
) -> tuple[int, int, dict[str, int]]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    mincore = ctypes.CDLL(None, use_errno=True).mincore
    mincore.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    mincore.restype = ctypes.c_int
    resident_by_path = {}
    allocated_bytes = 0
    seen_paths = set()
    for replica in replicas:
        for file in replica.table_files:
            path = file.path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            file_stat = path.stat()
            allocated_bytes += file_stat.st_blocks * 512
            fd = os.open(path, os.O_RDONLY)
            mapping = mmap.mmap(
                fd,
                file.nbytes,
                flags=mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            try:
                page_count = math.ceil(file.nbytes / page_size)
                residency = (ctypes.c_ubyte * page_count)()
                data_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                if mincore(data_ptr, file.nbytes, residency) == -1:
                    error = ctypes.get_errno()
                    raise OSError(error, os.strerror(error))
                resident_by_path[str(path)] = (
                    sum(1 for value in residency if value & 1) * page_size
                )
            finally:
                mapping.close()
                os.close(fd)
    return sum(resident_by_path.values()), allocated_bytes, resident_by_path


def select_table_files_for_numa(
    replicas: Sequence[TableReplica], numa_node: int
) -> tuple[FullTableFile, ...]:
    return select_table_replica_for_numa(replicas, numa_node).table_files


def select_table_replica_for_numa(
    replicas: Sequence[TableReplica], numa_node: int
) -> TableReplica:
    if len(replicas) == 1 and replicas[0].numa_node is None:
        return replicas[0]
    for replica in replicas:
        if replica.numa_node == numa_node:
            return replica
    raise RuntimeError(f"no OE table replica is available on NUMA{numa_node}")


def _write_tensor_chunk(mapping, *, byte_start: int, chunk) -> None:
    import torch

    byte_view = chunk.contiguous().view(torch.uint8).numpy()
    raw = memoryview(byte_view).cast("B")
    try:
        mapping[byte_start : byte_start + raw.nbytes] = raw
    finally:
        raw.release()
        del raw, byte_view


def _populate_full_table_files_worker(
    files: tuple[FullTableFile, ...],
    policy: PlacementPolicy,
    weight_source: str,
    copy_chunk_rows: int,
    status_queue,
) -> None:
    try:
        import torch
        from safetensors import safe_open

        torch.set_num_threads(max(1, min(16, len(os.sched_getaffinity(0)))))
        adapter = LinuxNumaPlacementAdapter()
        mode = (
            NumaPlacementMode.BIND
            if policy.mode == "bind"
            else NumaPlacementMode.INTERLEAVE
        )
        for file_spec in files:
            fd = os.open(file_spec.path, os.O_RDWR)
            mapping = mmap.mmap(
                fd,
                file_spec.nbytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            try:
                data_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                previous_policy = adapter.apply(
                    data_ptr=data_ptr,
                    nbytes=file_spec.nbytes,
                    mode=mode,
                    numa_nodes=policy.nodes,
                )
                try:
                    row_nbytes = file_spec.shape[1] * 2
                    if weight_source == "synthetic":
                        for start in range(0, file_spec.shape[0], copy_chunk_rows):
                            end = min(start + copy_chunk_rows, file_spec.shape[0])
                            chunk = torch.full(
                                (end - start, file_spec.shape[1]),
                                file_spec.table_index + 1,
                                dtype=torch.bfloat16,
                            )
                            _write_tensor_chunk(
                                mapping,
                                byte_start=start * row_nbytes,
                                chunk=chunk,
                            )
                            del chunk
                    elif weight_source == "checkpoint":
                        with safe_open(
                            file_spec.source_path, framework="pt", device="cpu"
                        ) as handle:
                            source = handle.get_slice(file_spec.source_key)
                            for start in range(
                                0, file_spec.shape[0], copy_chunk_rows
                            ):
                                end = min(
                                    start + copy_chunk_rows, file_spec.shape[0]
                                )
                                chunk = source[start:end]
                                _write_tensor_chunk(
                                    mapping,
                                    byte_start=start * row_nbytes,
                                    chunk=chunk,
                                )
                                del chunk
                    else:
                        raise ValueError(f"unknown weight source {weight_source!r}")
                finally:
                    adapter.reset(previous_policy)
                mapping.flush()
                gc.collect()
            finally:
                mapping.close()
                os.close(fd)
        status_queue.put({"ok": True})
    except BaseException:
        status_queue.put({"ok": False, "error": traceback.format_exc()})


def populate_full_table_files(
    files: Sequence[FullTableFile],
    policy: PlacementPolicy,
    weight_source: str,
    copy_chunk_rows: int,
) -> None:
    if copy_chunk_rows <= 0:
        raise ValueError("copy chunk rows must be positive")
    context = mp.get_context("spawn")
    status_queue = context.Queue()
    process = context.Process(
        target=_populate_full_table_files_worker,
        args=(tuple(files), policy, weight_source, copy_chunk_rows, status_queue),
    )
    started = False
    try:
        process.start()
        started = True
        status = status_queue.get(timeout=3600)
        process.join(timeout=60)
        if process.is_alive():
            raise RuntimeError("full-table loader did not exit")
    finally:
        if started and process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
            if process.is_alive():
                raise RuntimeError("full-table loader could not be stopped")
    if not status["ok"] or process.exitcode:
        raise RuntimeError(
            f"shared embedding full-table population failed:\n"
            f"{status.get('error', 'loader exited abnormally')}"
        )


def populate_table_replicas(
    replicas: Sequence[TableReplica],
    policy: PlacementPolicy,
    weight_source: str,
    copy_chunk_rows: int,
) -> None:
    for replica in replicas:
        replica_policy = policy
        if replica.numa_node is not None:
            replica_policy = PlacementPolicy(
                name=f"bind-numa{replica.numa_node}",
                mode="bind",
                nodes=(replica.numa_node,),
            )
        populate_full_table_files(
            replica.table_files,
            replica_policy,
            weight_source,
            copy_chunk_rows,
        )


def sample_full_table_placements(
    files: Sequence[FullTableFile], max_samples: int = 127
) -> dict[str, dict[int, int]]:
    adapter = LinuxNumaPlacementAdapter()
    placements = {}
    for file_spec in files:
        fd = os.open(file_spec.path, os.O_RDWR)
        mapping = mmap.mmap(
            fd,
            file_spec.nbytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        try:
            fault_sampled_pages(
                mapping,
                file_spec.nbytes,
                max_samples=max_samples,
            )
            data_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            placements[str(file_spec.path)] = adapter.sample(
                data_ptr=data_ptr,
                nbytes=file_spec.nbytes,
                max_samples=max_samples,
            )
        finally:
            mapping.close()
            os.close(fd)
    return placements


def sample_replica_placements(
    replicas: Sequence[TableReplica], max_samples: int = 127
) -> dict[str, dict[int, int]]:
    placements = {}
    for replica in replicas:
        placements.update(
            sample_full_table_placements(
                replica.table_files, max_samples=max_samples
            )
        )
    return placements


def fault_sampled_pages(
    mapping,
    nbytes: int,
    max_samples: int = 127,
    page_size: int | None = None,
) -> None:
    if nbytes <= 0 or max_samples <= 0:
        raise ValueError("nbytes and max samples must be positive")
    page_size = page_size or os.sysconf("SC_PAGE_SIZE")
    page_count = max(1, (nbytes + page_size - 1) // page_size)
    sample_count = min(page_count, max_samples)
    page_indices = sorted(
        {
            min(page_count - 1, index * page_count // sample_count)
            for index in range(sample_count)
        }
    )
    for page_index in page_indices:
        mapping[page_index * page_size]


def validate_sampled_placements(
    policy: PlacementPolicy, placements: Mapping[str, Mapping[int, int]]
) -> None:
    if not placements:
        raise RuntimeError("no sampled placements were recorded")
    if policy.mode == "bind":
        expected = policy.nodes[0]
        for path, nodes in placements.items():
            if set(nodes) != {expected}:
                raise RuntimeError(f"{path} pages are not all on NUMA{expected}: {nodes}")
        return
    for path, nodes in placements.items():
        if not set(policy.nodes).issubset(nodes):
            raise RuntimeError(f"{path} pages are not interleaved: {nodes}")


def validate_replica_placements(
    policy: PlacementPolicy,
    replicas: Sequence[TableReplica],
    placements: Mapping[str, Mapping[int, int]],
) -> None:
    if policy.mode != "replicate":
        validate_sampled_placements(policy, placements)
        return
    if {replica.numa_node for replica in replicas} != set(policy.nodes):
        raise RuntimeError("replica NUMA nodes do not match the placement policy")
    for replica in replicas:
        replica_placements = {
            str(file_spec.path): placements[str(file_spec.path)]
            for file_spec in replica.table_files
        }
        node = replica.numa_node
        validate_sampled_placements(
            PlacementPolicy(
                name=f"bind-numa{node}", mode="bind", nodes=(node,)
            ),
            replica_placements,
        )


def _make_embedding_batches(
    *,
    device: int,
    table_files: Sequence[FullTableFile],
    token_count: int,
    batch_count: int,
    seed: int,
) -> list[list[Any]]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if tuple(file.key for file in table_files) != SHARED_EMBEDDING_KEYS:
        raise ValueError("worker requires canonical base plus four OE tables")
    base_file, *oe_files = table_files
    batches = []
    for _ in range(batch_count):
        batches.append(
            {
                "base_ids": torch.randint(
                    0,
                    base_file.shape[0],
                    (token_count,),
                    generator=generator,
                    dtype=torch.int64,
                ).to(device),
                "oe_ids": [
                torch.randint(
                    0,
                    file_spec.shape[0],
                    (token_count,),
                    generator=generator,
                    dtype=torch.int64,
                ).to(device)
                    for file_spec in oe_files
                ],
            }
        )
    return batches


def _time_operation(
    *,
    device: int,
    barrier,
    inputs: Sequence[Any],
    warmups: int,
    operation,
) -> tuple[float, ...]:
    import torch

    samples = []
    with torch.cuda.device(device):
        for index, value in enumerate(inputs):
            barrier.wait()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = operation(value)
            end.record()
            end.synchronize()
            if index >= warmups:
                samples.append(float(start.elapsed_time(end)))
            del output
            barrier.wait()
    return tuple(samples)


def _worker(global_rank: int, config: WorkerConfig, barrier, result_queue) -> None:
    shared_views: list[SharedUVATensorView] = []
    registry = None
    production_registry = None
    base_module = None
    oe_modules = None
    projection_weight = None
    result = None
    try:
        import torch
        from sglang.srt.layers.full_vocab_shared_embedding import (
            FullVocabSharedEmbedding,
        )

        torch.cuda.set_device(global_rank)
        replica = select_table_replica_for_numa(
            config.table_replicas, config.topology.gpu_numa(global_rank)
        )
        table_files = replica.table_files
        if config.manifest_path is not None:
            production_registry = WeLMSharedEmbeddingRegistry.from_manifest(
                config.manifest_path,
                gpu_id=global_rank,
                gpu_numa_node=config.topology.gpu_numa(global_rank),
            )
            registry = production_registry
        else:
            for file_spec in table_files:
                shared_spec = SharedTensorFileSpec(
                    key=file_spec.key,
                    path=str(file_spec.path),
                    shape=file_spec.shape,
                    dtype="bfloat16",
                    nbytes=file_spec.nbytes,
                    replica_id=replica.replica_id,
                    numa_nodes=replica.numa_nodes,
                    inode=file_spec.path.stat().st_ino,
                )
                shared_views.append(
                    SharedUVATensorView.open(
                        shared_spec,
                        global_rank,
                        arena_root=config.arena_root,
                    )
                )

            class _Registry:
                def __init__(self, views, files):
                    self.weights = {
                        file.key: view.cuda_tensor
                        for view, file in zip(views, files)
                    }

                def get(self, key):
                    return self.weights[key]

            registry = _Registry(shared_views, table_files)
        base_file, *oe_files = table_files
        base_module = FullVocabSharedEmbedding(
            key=base_file.key,
            num_embeddings=base_file.shape[0],
            embedding_dim=base_file.shape[1],
            registry=registry,
        )
        oe_modules = [
            FullVocabSharedEmbedding(
                key=file_spec.key,
                num_embeddings=file_spec.shape[0],
                embedding_dim=file_spec.shape[1],
                registry=registry,
            )
            for file_spec in oe_files
        ]

        projection_weight = load_projection_weight(
            config.projection_spec, global_rank, config.weight_source
        )
        output_width = projection_weight.shape[0]
        if output_width != base_file.shape[1]:
            raise RuntimeError("base embedding and OE projection widths must match")

        observed_source_checksums = {}
        for module, file in zip((base_module, *oe_modules), table_files):
            indices = torch.tensor(
                [0, file.shape[0] // 2, file.shape[0] - 1],
                device=global_rank,
                dtype=torch.int64,
            )
            observed = _tensor_bytes_digest(module(indices))
            expected = config.source_row_checksums[file.key]
            if observed != expected:
                raise RuntimeError(
                    f"checkpoint row checksum mismatch for {file.key}: "
                    f"observed={observed}, expected={expected}"
                )
            observed_source_checksums[file.key] = observed

        records = []
        checksums = []
        total_batches = config.warmups + config.repeats
        workload_sizes = [
            (
                "prefill",
                global_tokens,
                local_prefill_tokens(global_tokens, config.topology.cp_size),
            )
            for global_tokens in config.prefill_sizes
        ] + [
            ("decode", batch_size, batch_size) for batch_size in config.decode_sizes
        ]
        for mode, global_tokens, local_tokens in workload_sizes:
            batches = _make_embedding_batches(
                device=global_rank,
                table_files=table_files,
                token_count=local_tokens,
                batch_count=total_batches,
                seed=worker_hash_seed(
                    config.topology, global_rank, mode, global_tokens
                ),
            )
            barrier.wait()
            first_batch = batches[0]
            base_reference = base_module(first_batch["base_ids"])
            oe_reference = run_lookup_concat(oe_modules, first_batch["oe_ids"])
            oe_projected = torch.nn.functional.linear(
                oe_reference, projection_weight
            )
            combined_reference = (base_reference + oe_projected) / 2
            checksum_rows = min(64, local_tokens)
            if not torch.isfinite(combined_reference).all():
                raise RuntimeError("combined embedding produced non-finite values")
            checksums.append(
                {
                    "mode": mode,
                    "global_tokens": global_tokens,
                    "local_tokens": local_tokens,
                    "base_lookup_checksum": float(
                        base_reference[:checksum_rows].float().sum().item()
                    ),
                    "oe_lookup_checksum": float(
                        oe_reference[:checksum_rows].float().sum().item()
                    ),
                    "combined_checksum": float(
                        combined_reference[:checksum_rows].float().sum().item()
                    ),
                }
            )
            del base_reference, oe_reference, oe_projected, combined_reference
            barrier.wait()

            operations = {
                "base_lookup": lambda value: base_module(value["base_ids"]),
                "oe_total": lambda value: torch.nn.functional.linear(
                    run_lookup_concat(oe_modules, value["oe_ids"]),
                    projection_weight,
                ),
                "combined": lambda value: (
                    base_module(value["base_ids"])
                    + torch.nn.functional.linear(
                        run_lookup_concat(oe_modules, value["oe_ids"]),
                        projection_weight,
                    )
                )
                / 2,
            }
            for scope, operation in operations.items():
                samples = _time_operation(
                    device=global_rank,
                    barrier=barrier,
                    inputs=batches,
                    warmups=config.warmups,
                    operation=operation,
                )
                records.append(
                    {
                        "mode": mode,
                        "global_tokens": global_tokens,
                        "local_tokens": local_tokens,
                        "scope": scope,
                        "samples_ms": samples,
                    }
                )
            del batches

        result = {
            "rank": global_rank,
            "cp_rank": config.topology.cp_rank(global_rank),
            "input_tp_rank": config.topology.input_tp_rank(global_rank),
            "gpu_numa": config.topology.gpu_numa(global_rank),
            "records": records,
            "checksums": checksums,
            "source_row_checksums": observed_source_checksums,
        }
    except BaseException:
        try:
            barrier.abort()
        except BaseException:
            pass
        result = {"rank": global_rank, "error": traceback.format_exc()}
    finally:
        cleanup_errors = []
        projection_weight = None
        base_module = None
        oe_modules = None
        registry = None
        module = None
        operations = None
        operation = None
        gc.collect()
        if production_registry is not None:
            try:
                production_registry.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        for shared in reversed(shared_views):
            try:
                shared.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            result = {
                "rank": global_rank,
                "error": repr(
                    BaseExceptionGroup("shared UVA worker cleanup failed", cleanup_errors)
                ),
            }
    result_queue.put(result)


def run_workers(config: WorkerConfig) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    barrier = context.Barrier(config.topology.world_size)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_worker,
            args=(rank, config, barrier, result_queue),
        )
        for rank in range(config.topology.world_size)
    ]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        results = [result_queue.get(timeout=7200) for _ in processes]
        for process in started:
            process.join(timeout=120)
    except BaseException as original_error:
        cleanup_errors = _stop_processes(started)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "worker execution and cleanup failed",
                [original_error, *cleanup_errors],
            )
        raise
    finally:
        cleanup_errors = _stop_processes(started)
        if cleanup_errors:
            raise BaseExceptionGroup("worker cleanup failed", cleanup_errors)
    errors = [result for result in results if "error" in result]
    if errors:
        detail = "\n".join(
            f"rank {result['rank']}:\n{result['error']}" for result in errors
        )
        raise RuntimeError(f"full-vocab embedding worker failed:\n{detail}")
    if any(process.exitcode for process in processes):
        raise RuntimeError(
            f"worker exit codes: {[process.exitcode for process in processes]}"
        )
    return sorted(results, key=lambda result: result["rank"])


def _stop_processes(processes) -> list[BaseException]:
    errors = []
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
            if process.is_alive():
                raise RuntimeError(f"worker process {process.pid} did not stop")
        except BaseException as exc:
            errors.append(exc)
    return errors


def validate_pair_checksums(
    topology: FullVocabTopology,
    worker_results: Sequence[dict[str, Any]],
) -> None:
    checksums_by_rank = {
        result["rank"]: {
            (
                record["mode"],
                record["global_tokens"],
                record["local_tokens"],
            ): (
                float(record["base_lookup_checksum"]),
                float(record["oe_lookup_checksum"]),
                float(record["combined_checksum"]),
            )
            for record in result["checksums"]
        }
        for result in worker_results
    }
    if set(checksums_by_rank) != set(range(topology.world_size)):
        raise ValueError("checksum results must contain every rank")
    for cp_rank, ranks in enumerate(topology.input_tp_groups):
        reference = checksums_by_rank[ranks[0]]
        for rank in ranks[1:]:
            if checksums_by_rank[rank] != reference:
                raise RuntimeError(
                    f"CP pair {cp_rank} embedding checksums differ: "
                    f"rank {ranks[0]}={reference}, rank {rank}={checksums_by_rank[rank]}"
                )


def aggregate_worker_results(
    topology: FullVocabTopology,
    worker_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if {result["rank"] for result in worker_results} != set(
        range(topology.world_size)
    ):
        raise ValueError("worker results must contain every rank")
    records_by_rank = {
        result["rank"]: {
            (
                record["mode"],
                record["global_tokens"],
                record["local_tokens"],
                record["scope"],
            ): record
            for record in result["records"]
        }
        for result in worker_results
    }
    keys = sorted(next(iter(records_by_rank.values())))
    if any(set(records) != set(keys) for records in records_by_rank.values()):
        raise ValueError("worker record keys do not match")

    aggregated = []
    for key in keys:
        rank_samples = {
            rank: tuple(records_by_rank[rank][key]["samples_ms"])
            for rank in range(topology.world_size)
        }
        pair_samples, global_samples = critical_path_samples(topology, rank_samples)
        mode, global_tokens, local_tokens, scope = key
        aggregated.append(
            {
                "type": "aggregate",
                "mode": mode,
                "global_tokens": global_tokens,
                "local_tokens": local_tokens,
                "scope": scope,
                "global_critical_path": summarize_samples(global_samples),
                "pair_critical_paths": {
                    str(cp_rank): summarize_samples(samples)
                    for cp_rank, samples in pair_samples.items()
                },
                "rank_summaries": {
                    str(rank): summarize_samples(samples)
                    for rank, samples in rank_samples.items()
                },
            }
        )
    return aggregated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input-tp-size", type=int, choices=(1, 2, 4, 8), default=2
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument(
        "--placement",
        choices=(
            "bind-numa0",
            "bind-numa1",
            "interleave",
            "replicate-numa",
        ),
        default="interleave",
    )
    parser.add_argument(
        "--prefill-sizes", type=parse_int_list, default=[16384, 65536]
    )
    parser.add_argument("--decode-sizes", type=parse_int_list, default=[1, 32])
    parser.add_argument(
        "--weight-source", choices=("checkpoint", "synthetic"), default="checkpoint"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Truncate every full table for synthetic smoke tests; 0 uses checkpoint shapes.",
    )
    parser.add_argument("--copy-chunk-rows", type=int, default=65536)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("oe-bench-results/shared-uva-production"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-numa-placement-check", action="store_true")
    return parser


def _jsonable_placements(
    placements: Mapping[str, Mapping[int, int]]
) -> dict[str, dict[str, int]]:
    return {
        path: {str(node): count for node, count in nodes.items()}
        for path, nodes in placements.items()
    }


def render_markdown(
    metadata: Mapping[str, Any], aggregated: Sequence[Mapping[str, Any]]
) -> str:
    lines = [
        "# WeLM Full-Vocab Shared-UVA Placement Benchmark",
        "",
        f"- Placement: `{metadata['placement']}`",
        f"- Topology: CP{metadata['cp_size']} + InputTP{metadata['input_tp_size']}",
        f"- Base bytes: {metadata['base_weight_bytes'] / 1024**3:.2f} GiB",
        f"- OE bytes: {metadata['oe_weight_bytes'] / 1024**3:.2f} GiB",
        f"- Physical embedding bytes: {metadata['physical_weight_bytes'] / 1024**3:.2f} GiB",
        f"- Sampled pages: `{metadata['sampled_page_placements']}`",
        f"- Warmups/repeats: {metadata['warmups']}/{metadata['repeats']}",
        "",
        "| Mode | Global size | Local rank tokens | Scope | Median ms | P90 ms | P99 ms |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for record in aggregated:
        summary = record["global_critical_path"]
        lines.append(
            f"| {record['mode']} | {record['global_tokens']} | "
            f"{record['local_tokens']} | {record['scope']} | "
            f"{summary['median_ms']:.4f} | {summary['p90_ms']:.4f} | "
            f"{summary['p99_ms']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.warmups <= 0 or args.repeats <= 0:
        raise ValueError("warmups and repeats must be positive")
    if args.max_rows < 0 or args.copy_chunk_rows <= 0:
        raise ValueError("max rows must be non-negative and copy chunk rows positive")

    import torch

    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"benchmark needs {args.world_size} GPUs, found {torch.cuda.device_count()}"
        )
    gpu_numa_nodes = query_gpu_numa_nodes(args.world_size)
    if args.dry_run:
        print(
            json.dumps(
                build_welm_shared_embedding_dry_run_report(
                    checkpoint=args.checkpoint,
                    gpu_numa_nodes=gpu_numa_nodes,
                ),
                indent=2,
            )
        )
        return 0
    topology = FullVocabTopology(
        input_tp_size=args.input_tp_size,
        world_size=args.world_size,
        gpu_numa_nodes=gpu_numa_nodes,
    )
    for size in args.prefill_sizes:
        local_prefill_tokens(size, topology.cp_size)

    policy = PlacementPolicy.parse(args.placement)
    embedding_specs, checkpoint_identity = (
        discover_welm_shared_embedding_checkpoint_tensors(args.checkpoint)
    )
    _oe_specs, projection_spec = load_oe_specs(args.checkpoint)
    gpu_state_before = query_gpu_state()
    shared_root = (
        Path("/dev/shm") / f"welm_full_embedding_{os.getpid()}_{int(time.time())}"
    )
    setup_start = time.perf_counter()
    arena_handle = None
    try:
        if args.weight_source == "checkpoint":
            if args.max_rows:
                raise ValueError("checkpoint mode does not support --max-rows")
            plans = plan_welm_embedding_replicas(
                policy.production_policy,
                topology.gpu_numa_nodes,
                policy.bind_node,
            )
            arena_handle = launch_welm_embedding_arena_process(
                checkpoint=args.checkpoint,
                root=shared_root,
                plans=plans,
                timeout=3600,
            )
            manifest = load_welm_embedding_arena_manifest(
                arena_handle.manifest_path
            )
            table_replicas = table_replicas_from_manifest(
                manifest, embedding_specs
            )
            placements = {
                tensor.path: dict(tensor.sampled_numa_nodes)
                for replica in manifest.replicas
                for tensor in replica.tensors
            }
            manifest_path = arena_handle.manifest_path
            source_row_checksums = checkpoint_row_checksums(embedding_specs)
        else:
            table_replicas = create_table_replicas(
                embedding_specs=embedding_specs,
                root=shared_root,
                max_rows=args.max_rows,
                policy=policy,
            )
            populate_table_replicas(
                table_replicas,
                policy,
                args.weight_source,
                args.copy_chunk_rows,
            )
            placements = sample_replica_placements(table_replicas)
            if not args.skip_numa_placement_check:
                validate_replica_placements(policy, table_replicas, placements)
            manifest_path = None
            source_row_checksums = synthetic_row_checksums(
                table_replicas[0].table_files
            )

        resident_bytes, allocated_bytes, resident_by_path = (
            measure_resident_file_bytes(table_replicas)
        )
        page_size = os.sysconf("SC_PAGE_SIZE")
        expected_resident_bytes = sum(
            math.ceil(file.nbytes / page_size) * page_size
            for replica in table_replicas
            for file in replica.table_files
        )
        if resident_bytes != expected_resident_bytes:
            raise RuntimeError(
                f"shared embedding resident bytes mismatch: "
                f"observed={resident_bytes}, expected={expected_resident_bytes}"
            )
        setup_seconds = time.perf_counter() - setup_start
        config = WorkerConfig(
            topology=topology,
            arena_root=shared_root,
            manifest_path=manifest_path,
            table_replicas=table_replicas,
            projection_spec=projection_spec,
            weight_source=args.weight_source,
            prefill_sizes=tuple(args.prefill_sizes),
            decode_sizes=tuple(args.decode_sizes),
            warmups=args.warmups,
            repeats=args.repeats,
            source_row_checksums=source_row_checksums,
        )
        worker_results = run_workers(config)
        validate_pair_checksums(topology, worker_results)
        aggregated = aggregate_worker_results(topology, worker_results)
        actual_shapes = [file.shape for file in table_replicas[0].table_files]
        actual_bytes = calculate_welm_embedding_byte_counts(
            base_shape=actual_shapes[0],
            oe_shapes=actual_shapes[1:],
            element_size=2,
            replica_count=len(table_replicas),
        )
        checkpoint_bytes = calculate_welm_embedding_byte_counts(
            base_shape=embedding_specs[0].shape,
            oe_shapes=[spec.shape for spec in embedding_specs[1:]],
            element_size=2,
            replica_count=len(table_replicas),
        )
        metadata = {
            "type": "metadata",
            "checkpoint": str(args.checkpoint),
            "checkpoint_identity": checkpoint_identity,
            "placement": policy.name,
            "placement_mode": policy.mode,
            "requested_numa_nodes": list(policy.nodes),
            "input_tp_size": topology.input_tp_size,
            "cp_size": topology.cp_size,
            "world_size": topology.world_size,
            "gpu_numa_nodes": list(topology.gpu_numa_nodes),
            "weight_source": args.weight_source,
            "max_rows": args.max_rows,
            "replica_count": len(table_replicas),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "base_weight_bytes": actual_bytes.base_bytes,
            "oe_weight_bytes": actual_bytes.oe_bytes,
            "logical_weight_bytes": actual_bytes.logical_bytes,
            "physical_weight_bytes": actual_bytes.physical_bytes,
            "resident_physical_bytes": resident_bytes,
            "filesystem_allocated_bytes": allocated_bytes,
            "resident_bytes_by_path": resident_by_path,
            "full_checkpoint_base_bytes": checkpoint_bytes.base_bytes,
            "full_checkpoint_oe_bytes": checkpoint_bytes.oe_bytes,
            "full_checkpoint_logical_bytes": checkpoint_bytes.logical_bytes,
            "sampled_page_placements": _jsonable_placements(placements),
            "setup_seconds": setup_seconds,
            "gpu_state_before": gpu_state_before,
            "gpu_state_after": query_gpu_state(),
            "embedding_path": "full_vocab_shared_uva_base_oe_combined",
            "checksums": {
                str(worker["rank"]): worker["checksums"]
                for worker in worker_results
            },
            "source_row_checksums": source_row_checksums,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = (
            f"full_vocab_embedding_inputtp{topology.input_tp_size}_cp{topology.cp_size}_"
            f"{policy.name}_{timestamp}"
        )
        jsonl_path = args.output_dir / f"{stem}.jsonl"
        markdown_path = args.output_dir / f"{stem}.md"
        with jsonl_path.open("w") as output:
            output.write(json.dumps(metadata) + "\n")
            for worker in worker_results:
                output.write(json.dumps({"type": "worker", **worker}) + "\n")
            for record in aggregated:
                output.write(json.dumps(record) + "\n")
        markdown = render_markdown(metadata, aggregated)
        markdown_path.write_text(markdown)
        print(markdown)
        print(f"JSONL: {jsonl_path}")
        print(f"Markdown: {markdown_path}")
    finally:
        if arena_handle is not None:
            arena_handle.close(timeout=60)
        else:
            shutil.rmtree(shared_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
