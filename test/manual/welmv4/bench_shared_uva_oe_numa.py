#!/usr/bin/env python3
"""Microbenchmark local versus remote NUMA access to host-resident WeLM OE weights."""

from __future__ import annotations

import argparse
import math
import statistics
import ctypes
import errno
import json
import os
import gc
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


@dataclass(frozen=True)
class BenchmarkCase:
    mode: str
    token_count: int
    placement: str
    gpu: int
    gpu_numa: int
    weight_numa: int
    tp_size: int
    tp_rank: int
    weight_source: str
    concurrency: int = 1


@dataclass(frozen=True)
class OeWeightSpec:
    name: str
    path: Path
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class BenchmarkRecord:
    mode: str
    token_count: int
    scope: str
    placement: str
    device: int
    gpu_numa: int
    weight_numa: int
    tp_size: int
    tp_rank: int
    weight_source: str
    id_distribution: str
    useful_host_bytes: int
    cuda_samples_ms: tuple[float, ...]
    wall_samples_ms: tuple[float, ...]
    concurrency: int = 1

    def as_dict(self) -> dict[str, Any]:
        cuda_summary = summarize_samples(self.cuda_samples_ms)
        wall_summary = summarize_samples(self.wall_samples_ms)
        median_seconds = cuda_summary["median_ms"] / 1000.0
        useful_gib_per_s = (
            self.useful_host_bytes / (1024**3) / median_seconds
            if self.useful_host_bytes and median_seconds
            else 0.0
        )
        return {
            "mode": self.mode,
            "token_count": self.token_count,
            "scope": self.scope,
            "placement": self.placement,
            "device": self.device,
            "gpu_numa": self.gpu_numa,
            "weight_numa": self.weight_numa,
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "weight_source": self.weight_source,
            "id_distribution": self.id_distribution,
            "useful_host_bytes": self.useful_host_bytes,
            "useful_gib_per_s": useful_gib_per_s,
            "cuda": cuda_summary,
            "wall": wall_summary,
            "cuda_samples_ms": list(self.cuda_samples_ms),
            "wall_samples_ms": list(self.wall_samples_ms),
            "concurrency": self.concurrency,
        }


class HostWeightOwner:
    """Own one pinned-host allocation and expose non-owning CUDA/UVA views."""

    def __init__(self, tensor: Any):
        self._tensor = tensor
        self._views: dict[int, Any] = {tensor.device.index: tensor}

    @classmethod
    def allocate(cls, shape, dtype, owner_device: int) -> "HostWeightOwner":
        import torch

        if owner_device < 0 or owner_device >= torch.cuda.device_count():
            raise ValueError(f"invalid CUDA device {owner_device}")
        from sglang.jit_kernel.memory_allocator import custom_empty

        return cls(custom_empty(tuple(shape), dtype=dtype, device_id=owner_device))

    def __copy__(self):
        raise TypeError("HostWeightOwner cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("HostWeightOwner cannot be copied")

    @property
    def tensor(self):
        if self._tensor is None:
            raise RuntimeError("HostWeightOwner is closed")
        return self._tensor

    @property
    def data_ptr(self) -> int:
        return int(self.tensor.data_ptr())

    @property
    def nbytes(self) -> int:
        return int(self.tensor.numel() * self.tensor.element_size())

    def device_view(self, device: int):
        import torch

        if self._tensor is None:
            raise RuntimeError("HostWeightOwner is closed")
        if device < 0 or device >= torch.cuda.device_count():
            raise ValueError(f"invalid CUDA device {device}")
        if device not in self._views:
            from sglang.jit_kernel.memory_allocator import _make_tensor_from_ptr

            self._views[device] = _make_tensor_from_ptr(
                self.data_ptr,
                tuple(self.tensor.shape),
                self.tensor.dtype,
                torch.device("cuda", device),
                lambda _ptr: None,
            )
        return self._views[device]

    def close(self) -> None:
        if self._tensor is None:
            return
        import torch

        for device in self._views:
            torch.cuda.synchronize(device)
        self._views.clear()
        self._tensor = None


def query_sampled_numa_nodes(
    data_ptr: int,
    nbytes: int,
    max_samples: int = 64,
) -> dict[int, int]:
    """Query NUMA placement for sampled resident pages with Linux move_pages."""
    if data_ptr <= 0 or nbytes <= 0 or max_samples <= 0:
        raise ValueError("data_ptr, nbytes, and max_samples must be positive")

    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = max(1, math.ceil(nbytes / page_size))
    sample_count = min(page_count, max_samples)
    page_indices = {
        min(page_count - 1, index * page_count // sample_count)
        for index in range(sample_count)
    }
    addresses_list = [data_ptr + index * page_size for index in sorted(page_indices)]
    count = len(addresses_list)
    addresses = (ctypes.c_void_p * count)(*addresses_list)
    status = (ctypes.c_int * count)()

    libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    move_pages = libnuma.move_pages
    move_pages.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    move_pages.restype = ctypes.c_long
    result = move_pages(0, count, addresses, None, status, 0)
    if result == -1:
        error = ctypes.get_errno()
        if error == errno.EACCES:
            raise PermissionError(error, "move_pages placement query is not permitted")
        raise OSError(error, os.strerror(error))

    placement: dict[int, int] = {}
    for node in status:
        if node < 0:
            raise OSError(-node, os.strerror(-node))
        placement[node] = placement.get(node, 0) + 1
    return placement


def query_gpu_numa_nodes(device_count: int) -> tuple[int, ...]:
    """Resolve visible NVIDIA GPU indices to NUMA nodes through PCI sysfs."""
    import torch

    if device_count <= 0:
        raise ValueError("device_count must be positive")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    nodes_by_uuid: dict[str, int] = {}
    for line in output.splitlines():
        uuid_text, pci_text = (part.strip() for part in line.split(",", 1))
        domain, bus, slot = pci_text.split(":")
        pci_id = f"{int(domain, 16) & 0xFFFF:04x}:{bus.lower()}:{slot.lower()}"
        node = int((Path("/sys/bus/pci/devices") / pci_id / "numa_node").read_text())
        if node < 0:
            raise RuntimeError(f"GPU {index_text} PCI {pci_id} has no NUMA node")
        nodes_by_uuid[uuid_text.removeprefix("GPU-").lower()] = node
    result = []
    for device in range(device_count):
        uuid = str(torch.cuda.get_device_properties(device).uuid).lower()
        try:
            result.append(nodes_by_uuid[uuid])
        except KeyError as exc:
            raise RuntimeError(
                f"missing NUMA mapping for visible GPU {device} {uuid}"
            ) from exc
    return tuple(result)


def shard_bounds(
    rows: int,
    tp_size: int,
    tp_rank: int,
    padding: int = 64,
) -> tuple[int, int, int]:
    """Return the real row range and padded partition size for one vocab shard."""
    if rows <= 0:
        raise ValueError("rows must be positive")
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError("tp_rank must be in [0, tp_size)")
    if padding <= 0:
        raise ValueError("padding must be positive")

    padded_rows = math.ceil(rows / padding) * padding
    if padded_rows % tp_size != 0:
        raise ValueError(
            f"padded rows {padded_rows} are not divisible by tp_size {tp_size}"
        )
    partition_rows = padded_rows // tp_size
    start = partition_rows * tp_rank
    end = min(start + partition_rows, rows)
    if end <= start:
        raise ValueError("the selected TP shard has no real vocab rows")
    return start, end, partition_rows


def parse_int_list(value: str) -> list[int]:
    """Parse a non-empty comma-separated list of positive integers."""
    if not value:
        raise ValueError("integer list must not be empty")
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError(f"invalid integer list: {value!r}")
    try:
        result = [int(part.strip()) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid integer list: {value!r}") from exc
    if any(item <= 0 for item in result):
        raise ValueError("integer list values must be positive")
    return result


def parse_device_list(value: str) -> list[int]:
    """Parse a non-empty comma-separated list of unique non-negative devices."""
    if not value:
        raise ValueError("device list must not be empty")
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError(f"invalid device list: {value!r}")
    try:
        result = [int(part.strip()) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid device list: {value!r}") from exc
    if any(item < 0 for item in result) or len(set(result)) != len(result):
        raise ValueError("devices must be non-negative and unique")
    return result


def _linear_percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def summarize_samples(samples_ms: Sequence[float]) -> dict[str, float]:
    """Summarize non-empty millisecond samples using linear percentiles."""
    if not samples_ms:
        raise ValueError("samples must not be empty")
    values = sorted(float(value) for value in samples_ms)
    return {
        "count": len(values),
        "min_ms": values[0],
        "max_ms": values[-1],
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": _linear_percentile(values, 0.90),
        "p99_ms": _linear_percentile(values, 0.99),
        "stddev_ms": statistics.pstdev(values),
    }


def load_oe_specs(checkpoint: Path) -> tuple[list[OeWeightSpec], OeWeightSpec]:
    """Read WeLM OE tensor metadata without materializing checkpoint values."""
    checkpoint = Path(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    weight_map = json.loads(index_path.read_text())["weight_map"]
    embedding_names = [f"model.oe_embed.{index}.weight" for index in range(4)]
    projection_name = "model.oe_up_proj.weight"

    from safetensors import safe_open

    def read_spec(name: str) -> OeWeightSpec:
        try:
            path = checkpoint / weight_map[name]
        except KeyError as exc:
            raise KeyError(f"missing checkpoint weight {name}") from exc
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(name)
            return OeWeightSpec(
                name=name,
                path=path,
                shape=tuple(int(value) for value in tensor_slice.get_shape()),
                dtype=str(tensor_slice.get_dtype()),
            )

    embeddings = [read_spec(name) for name in embedding_names]
    projection = read_spec(projection_name)
    if any(len(spec.shape) != 2 or spec.shape[1] != 512 for spec in embeddings):
        raise ValueError("expected four [vocab, 512] OE embedding tables")
    if any(spec.dtype != "BF16" for spec in embeddings):
        raise ValueError("expected BF16 OE embedding tables")
    if projection.shape != (2048, 2048) or projection.dtype != "BF16":
        raise ValueError("expected BF16 [2048, 2048] OE projection")
    return embeddings, projection


def build_dry_run_report(
    *,
    checkpoint: Path,
    embedding_specs: Sequence[OeWeightSpec],
    projection_spec: OeWeightSpec,
    tp_size: int,
    tp_rank: int,
    prefill_sizes: Sequence[int],
    decode_sizes: Sequence[int],
) -> dict[str, Any]:
    """Build exact allocation and shape accounting without allocating CUDA memory."""
    branches = []
    embedding_pinned_bytes = 0
    for spec in embedding_specs:
        start, end, partition_rows = shard_bounds(
            spec.shape[0], tp_size=tp_size, tp_rank=tp_rank
        )
        nbytes = partition_rows * spec.shape[1] * 2
        embedding_pinned_bytes += nbytes
        branches.append(
            {
                "name": spec.name,
                "source_shape": list(spec.shape),
                "real_row_start": start,
                "real_row_end": end,
                "allocation_shape": [partition_rows, spec.shape[1]],
                "bytes": nbytes,
            }
        )
    projection_bytes = math.prod(projection_spec.shape) * 2
    return {
        "checkpoint": str(checkpoint),
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "branches": branches,
        "embedding_pinned_bytes": embedding_pinned_bytes,
        "projection_bytes": projection_bytes,
        "prefill_sizes": list(prefill_sizes),
        "decode_sizes": list(decode_sizes),
    }


def make_lookup_modules(
    *,
    owners: Sequence[HostWeightOwner],
    device: int,
    shard_starts: Sequence[int],
    shard_ends: Sequence[int],
) -> list[Any]:
    """Build the minimal descriptors consumed by the production OE lookup kernel."""
    if not (len(owners) == len(shard_starts) == len(shard_ends) == 4):
        raise ValueError("WeLM OE lookup requires exactly four branches")
    modules = []
    for owner, start, end in zip(owners, shard_starts, shard_ends):
        if start < 0 or end <= start:
            raise ValueError(f"invalid shard range [{start}, {end})")
        modules.append(
            SimpleNamespace(
                weight=owner.device_view(device),
                shard_indices=SimpleNamespace(
                    org_vocab_start_index=int(start),
                    org_vocab_end_index=int(end),
                ),
            )
        )
    return modules


def run_lookup_concat(modules: Sequence[Any], hashed_inputs: Sequence[Any]):
    """Invoke the production specialized WeLM prehashed lookup/concat kernel."""
    if len(modules) != 4 or len(hashed_inputs) != 4:
        raise ValueError("WeLM OE lookup requires exactly four branches")
    import torch

    device = hashed_inputs[0].device
    if not all(tensor.device == device for tensor in hashed_inputs):
        raise ValueError("all hashed inputs must be on the same CUDA device")
    from sglang.srt.models.welm_perf_opt import (
        _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512,
    )

    with torch.cuda.device(device):
        return _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512(
            hashed_inputs=hashed_inputs,
            oe_embed_modules=modules,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--weight-source", choices=("synthetic", "checkpoint"), default="synthetic"
    )
    parser.add_argument(
        "--id-distribution", choices=("global", "local-valid"), default="global"
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--owner-device", type=int, default=0)
    parser.add_argument("--local-device", type=int, default=0)
    parser.add_argument("--remote-device", type=int, default=4)
    parser.add_argument("--weight-numa", type=int, default=0)
    parser.add_argument("--local-gpu-numa", type=int, default=0)
    parser.add_argument("--remote-gpu-numa", type=int, default=1)
    parser.add_argument(
        "--prefill-sizes",
        type=parse_int_list,
        default=[256, 1024, 4096, 16384],
    )
    parser.add_argument(
        "--decode-sizes",
        type=parse_int_list,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--max-shard-rows",
        type=int,
        default=0,
        help="Truncate each rank-local shard for smoke tests; 0 uses real shapes.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--concurrent-devices",
        type=parse_device_list,
        default=[],
        help="Physical/visible CUDA devices that concurrently consume one host shard.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-numa-placement-check", action="store_true")
    return parser


def allocate_oe_shard_owners(
    *,
    specs: Sequence[OeWeightSpec],
    tp_size: int,
    tp_rank: int,
    owner_device: int,
    weight_source: str,
    max_shard_rows: int,
) -> tuple[list[HostWeightOwner], list[int], list[int], float]:
    """Allocate and initialize one physical host copy of four OE vocab shards."""
    import torch

    owners: list[HostWeightOwner] = []
    starts: list[int] = []
    ends: list[int] = []
    setup_start = time.perf_counter()
    for branch_index, spec in enumerate(specs):
        start, real_end, partition_rows = shard_bounds(
            spec.shape[0], tp_size=tp_size, tp_rank=tp_rank
        )
        allocation_rows = (
            min(partition_rows, max_shard_rows)
            if max_shard_rows > 0
            else partition_rows
        )
        selected_end = min(real_end, start + allocation_rows)
        if selected_end <= start:
            raise ValueError(f"empty selected shard for {spec.name}")
        owner = HostWeightOwner.allocate(
            shape=(allocation_rows, spec.shape[1]),
            dtype=torch.bfloat16,
            owner_device=owner_device,
        )
        with torch.cuda.device(owner_device):
            if weight_source == "synthetic":
                owner.tensor.fill_(branch_index + 1)
            elif weight_source == "checkpoint":
                from safetensors import safe_open

                with safe_open(spec.path, framework="pt", device="cpu") as handle:
                    source = handle.get_slice(spec.name)[start:selected_end]
                    owner.tensor[: source.shape[0]].copy_(source)
                    if source.shape[0] < allocation_rows:
                        owner.tensor[source.shape[0] :].zero_()
                    del source
                gc.collect()
            else:
                raise ValueError(f"unknown weight source {weight_source!r}")
            torch.cuda.synchronize(owner_device)
        owners.append(owner)
        starts.append(start)
        ends.append(selected_end)
    return owners, starts, ends, time.perf_counter() - setup_start


def load_projection_weight(spec: OeWeightSpec, device: int, source: str):
    import torch

    with torch.cuda.device(device):
        if source == "checkpoint":
            from safetensors import safe_open

            with safe_open(spec.path, framework="pt", device="cpu") as handle:
                weight = handle.get_tensor(spec.name).to(device)
        else:
            generator = torch.Generator(device=device).manual_seed(17)
            weight = torch.randn(
                spec.shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
    return weight


def make_hashed_input_batches(
    *,
    device: int,
    token_count: int,
    batch_count: int,
    global_rows: Sequence[int],
    shard_starts: Sequence[int],
    shard_ends: Sequence[int],
    distribution: str,
    seed: int,
) -> tuple[list[list[Any]], list[int]]:
    import torch

    # Generate on CPU so the same seed produces bitwise-identical batches for
    # local and remote GPUs. Transfers happen before the timed region.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches: list[list[Any]] = []
    useful_bytes: list[int] = []
    for _ in range(batch_count):
        branch_inputs = []
        valid_count = 0
        for rows, start, end in zip(global_rows, shard_starts, shard_ends):
            low, high = (start, end) if distribution == "local-valid" else (0, rows)
            hashed_cpu = torch.randint(
                low,
                high,
                (token_count,),
                dtype=torch.int64,
                generator=generator,
            )
            valid_count += int(
                ((hashed_cpu >= start) & (hashed_cpu < end)).sum().item()
            )
            branch_inputs.append(hashed_cpu.to(device, non_blocking=False))
        batches.append(branch_inputs)
        useful_bytes.append(valid_count * 512 * 2)
    return batches, useful_bytes


def alternating_order(names: Sequence[str], iteration: int) -> tuple[str, ...]:
    """Return an ABBA-style order for paired local/remote measurements."""
    ordered = tuple(names)
    return ordered if iteration % 2 == 0 else tuple(reversed(ordered))


def time_paired_cuda_operations(
    *,
    devices: dict[str, int],
    inputs_by_name: dict[str, Sequence[Any]],
    operations_by_name: dict[str, Any],
    warmups: int,
    repeats: int,
) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    """Time isolated operations in alternating order on two or more GPUs."""
    import torch

    names = tuple(devices)
    if len(names) < 2 or set(names) != set(inputs_by_name) or set(names) != set(
        operations_by_name
    ):
        raise ValueError(
            "paired timing needs matching names for devices and operations"
        )
    iteration_count = warmups + repeats
    if any(len(inputs_by_name[name]) != iteration_count for name in names):
        raise ValueError("every paired input needs warmups + repeats batches")

    cuda_samples = {name: [] for name in names}
    wall_samples = {name: [] for name in names}
    for index in range(iteration_count):
        for name in alternating_order(names, index):
            device = devices[name]
            with torch.cuda.device(device):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start_event.record()
                output = operations_by_name[name](inputs_by_name[name][index])
                end_event.record()
                end_event.synchronize()
                if index >= warmups:
                    cuda_samples[name].append(
                        float(start_event.elapsed_time(end_event))
                    )
                    wall_samples[name].append(
                        (time.perf_counter() - wall_start) * 1000.0
                    )
                del output
    return (
        {name: tuple(values) for name, values in cuda_samples.items()},
        {name: tuple(values) for name, values in wall_samples.items()},
    )


def time_cuda_sequence(
    *,
    device: int,
    inputs: Sequence[Any],
    warmups: int,
    repeats: int,
    operation,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    import torch

    if len(inputs) != warmups + repeats:
        raise ValueError("input sequence length must equal warmups + repeats")
    with torch.cuda.device(device):
        for value in inputs[:warmups]:
            operation(value)
        torch.cuda.synchronize(device)
        cuda_samples = []
        wall_samples = []
        for value in inputs[warmups:]:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start_event.record()
            output = operation(value)
            end_event.record()
            end_event.synchronize()
            wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
            cuda_samples.append(float(start_event.elapsed_time(end_event)))
            del output
    return tuple(cuda_samples), tuple(wall_samples)


def time_concurrent_operations(
    *,
    devices: Sequence[int],
    inputs_by_device: dict[int, Sequence[Any]],
    operations_by_device: dict[int, Any],
    warmups: int,
    repeats: int,
) -> tuple[dict[int, tuple[float, ...]], tuple[float, ...]]:
    """Launch one operation per GPU concurrently using a host-thread barrier."""
    import torch

    devices = list(devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be non-empty and unique")
    iteration_count = warmups + repeats
    for device in devices:
        if len(inputs_by_device[device]) != iteration_count:
            raise ValueError("every device needs warmups + repeats input batches")

    barrier = threading.Barrier(len(devices) + 1)
    sample_lists: dict[int, list[float]] = {device: [] for device in devices}
    errors: list[BaseException] = []
    error_lock = threading.Lock()

    def worker(device: int) -> None:
        try:
            with torch.cuda.device(device):
                for index, value in enumerate(inputs_by_device[device]):
                    barrier.wait()
                    if index < warmups:
                        operations_by_device[device](value)
                        torch.cuda.synchronize(device)
                    else:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        output = operations_by_device[device](value)
                        end_event.record()
                        end_event.synchronize()
                        sample_lists[device].append(
                            float(start_event.elapsed_time(end_event))
                        )
                        del output
                    barrier.wait()
        except BaseException as exc:
            with error_lock:
                errors.append(exc)
            barrier.abort()

    threads = [
        threading.Thread(target=worker, args=(device,), daemon=True) for device in devices
    ]
    for thread in threads:
        thread.start()

    wall_samples = []
    try:
        for index in range(iteration_count):
            wall_start = time.perf_counter()
            barrier.wait()
            barrier.wait()
            if index >= warmups:
                wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
    finally:
        for thread in threads:
            thread.join()
    if errors:
        raise RuntimeError("concurrent CUDA worker failed") from errors[0]
    return (
        {device: tuple(samples) for device, samples in sample_lists.items()},
        tuple(wall_samples),
    )


def run_device_matrix(
    *,
    mode: str,
    sizes: Sequence[int],
    placement: str,
    device: int,
    gpu_numa: int,
    weight_numa: int,
    owners: Sequence[HostWeightOwner],
    shard_starts: Sequence[int],
    shard_ends: Sequence[int],
    global_rows: Sequence[int],
    projection_weight,
    tp_size: int,
    tp_rank: int,
    weight_source: str,
    id_distribution: str,
    warmups: int,
    repeats: int,
) -> list[BenchmarkRecord]:
    import torch

    modules = make_lookup_modules(
        owners=owners,
        device=device,
        shard_starts=shard_starts,
        shard_ends=shard_ends,
    )
    records = []
    for token_count in sizes:
        batches, useful_bytes_per_batch = make_hashed_input_batches(
            device=device,
            token_count=token_count,
            batch_count=warmups + repeats,
            global_rows=global_rows,
            shard_starts=shard_starts,
            shard_ends=shard_ends,
            distribution=id_distribution,
            seed=10_000 + token_count,
        )
        timed_useful_bytes = useful_bytes_per_batch[warmups:]
        useful_host_bytes = round(statistics.fmean(timed_useful_bytes))

        lookup_cuda, lookup_wall = time_cuda_sequence(
            device=device,
            inputs=batches,
            warmups=warmups,
            repeats=repeats,
            operation=lambda value: run_lookup_concat(modules, value),
        )
        records.append(
            BenchmarkRecord(
                mode=mode,
                token_count=token_count,
                scope="lookup_concat",
                placement=placement,
                device=device,
                gpu_numa=gpu_numa,
                weight_numa=weight_numa,
                tp_size=tp_size,
                tp_rank=tp_rank,
                weight_source=weight_source,
                id_distribution=id_distribution,
                useful_host_bytes=useful_host_bytes,
                cuda_samples_ms=lookup_cuda,
                wall_samples_ms=lookup_wall,
            )
        )

        total_cuda, total_wall = time_cuda_sequence(
            device=device,
            inputs=batches,
            warmups=warmups,
            repeats=repeats,
            operation=lambda value: torch.nn.functional.linear(
                run_lookup_concat(modules, value), projection_weight
            ),
        )
        records.append(
            BenchmarkRecord(
                mode=mode,
                token_count=token_count,
                scope="oe_total_unfused",
                placement=placement,
                device=device,
                gpu_numa=gpu_numa,
                weight_numa=weight_numa,
                tp_size=tp_size,
                tp_rank=tp_rank,
                weight_source=weight_source,
                id_distribution=id_distribution,
                useful_host_bytes=useful_host_bytes,
                cuda_samples_ms=total_cuda,
                wall_samples_ms=total_wall,
            )
        )
    return records


def run_paired_device_matrix(
    *,
    mode: str,
    sizes: Sequence[int],
    local_device: int,
    remote_device: int,
    local_gpu_numa: int,
    remote_gpu_numa: int,
    weight_numa: int,
    owners: Sequence[HostWeightOwner],
    shard_starts: Sequence[int],
    shard_ends: Sequence[int],
    global_rows: Sequence[int],
    projection_weights: dict[int, Any],
    tp_size: int,
    tp_rank: int,
    weight_source: str,
    id_distribution: str,
    warmups: int,
    repeats: int,
) -> list[BenchmarkRecord]:
    """Measure local and remote GPUs against identical inputs in ABBA order."""
    import torch

    names = ("local", "remote")
    devices = {"local": local_device, "remote": remote_device}
    gpu_numa = {"local": local_gpu_numa, "remote": remote_gpu_numa}
    modules = {
        name: make_lookup_modules(
            owners=owners,
            device=device,
            shard_starts=shard_starts,
            shard_ends=shard_ends,
        )
        for name, device in devices.items()
    }
    records = []
    for token_count in sizes:
        inputs_by_name = {}
        useful_by_name = {}
        for name, device in devices.items():
            batches, useful = make_hashed_input_batches(
                device=device,
                token_count=token_count,
                batch_count=warmups + repeats,
                global_rows=global_rows,
                shard_starts=shard_starts,
                shard_ends=shard_ends,
                distribution=id_distribution,
                seed=10_000 + token_count,
            )
            inputs_by_name[name] = batches
            useful_by_name[name] = useful
        if useful_by_name["local"] != useful_by_name["remote"]:
            raise RuntimeError("paired local/remote inputs are not identical")
        useful_host_bytes = round(
            statistics.fmean(useful_by_name["local"][warmups:])
        )

        lookup_operations = {
            name: (
                lambda value, lookup_modules=modules[name]: run_lookup_concat(
                    lookup_modules, value
                )
            )
            for name in names
        }
        lookup_cuda, lookup_wall = time_paired_cuda_operations(
            devices=devices,
            inputs_by_name=inputs_by_name,
            operations_by_name=lookup_operations,
            warmups=warmups,
            repeats=repeats,
        )
        total_operations = {
            name: (
                lambda value,
                lookup_modules=modules[name],
                projection=projection_weights[
                    devices[name]
                ]: torch.nn.functional.linear(
                    run_lookup_concat(lookup_modules, value), projection
                )
            )
            for name in names
        }
        total_cuda, total_wall = time_paired_cuda_operations(
            devices=devices,
            inputs_by_name=inputs_by_name,
            operations_by_name=total_operations,
            warmups=warmups,
            repeats=repeats,
        )

        for name in names:
            for scope, cuda_samples, wall_samples in (
                ("lookup_concat", lookup_cuda[name], lookup_wall[name]),
                ("oe_total_unfused", total_cuda[name], total_wall[name]),
            ):
                records.append(
                    BenchmarkRecord(
                        mode=mode,
                        token_count=token_count,
                        scope=scope,
                        placement=name,
                        device=devices[name],
                        gpu_numa=gpu_numa[name],
                        weight_numa=weight_numa,
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                        weight_source=weight_source,
                        id_distribution=id_distribution,
                        useful_host_bytes=useful_host_bytes,
                        cuda_samples_ms=cuda_samples,
                        wall_samples_ms=wall_samples,
                    )
                )
    return records


def run_concurrent_matrix(
    *,
    mode: str,
    sizes: Sequence[int],
    devices: Sequence[int],
    gpu_numa_nodes: Sequence[int],
    weight_numa: int,
    owners: Sequence[HostWeightOwner],
    shard_starts: Sequence[int],
    shard_ends: Sequence[int],
    global_rows: Sequence[int],
    projection_weights: dict[int, Any],
    tp_size: int,
    tp_rank: int,
    weight_source: str,
    id_distribution: str,
    warmups: int,
    repeats: int,
) -> list[BenchmarkRecord]:
    import torch

    devices = list(devices)
    modules_by_device = {
        device: make_lookup_modules(
            owners=owners,
            device=device,
            shard_starts=shard_starts,
            shard_ends=shard_ends,
        )
        for device in devices
    }
    records = []
    for token_count in sizes:
        inputs_by_device = {}
        useful_by_device = {}
        for device in devices:
            batches, useful = make_hashed_input_batches(
                device=device,
                token_count=token_count,
                batch_count=warmups + repeats,
                global_rows=global_rows,
                shard_starts=shard_starts,
                shard_ends=shard_ends,
                distribution=id_distribution,
                seed=20_000 + token_count * 17 + device,
            )
            inputs_by_device[device] = batches
            useful_by_device[device] = round(statistics.fmean(useful[warmups:]))

        lookup_operations = {
            device: (
                lambda value, modules=modules_by_device[device]: run_lookup_concat(
                    modules, value
                )
            )
            for device in devices
        }
        lookup_samples, lookup_wall = time_concurrent_operations(
            devices=devices,
            inputs_by_device=inputs_by_device,
            operations_by_device=lookup_operations,
            warmups=warmups,
            repeats=repeats,
        )
        total_operations = {
            device: (
                lambda value,
                modules=modules_by_device[device],
                projection=projection_weights[device]: torch.nn.functional.linear(
                    run_lookup_concat(modules, value), projection
                )
            )
            for device in devices
        }
        total_samples, total_wall = time_concurrent_operations(
            devices=devices,
            inputs_by_device=inputs_by_device,
            operations_by_device=total_operations,
            warmups=warmups,
            repeats=repeats,
        )

        for device in devices:
            gpu_numa = int(gpu_numa_nodes[device])
            relation = "local" if gpu_numa == weight_numa else "remote"
            for scope, samples, wall in (
                ("lookup_concat_concurrent", lookup_samples[device], lookup_wall),
                ("oe_total_unfused_concurrent", total_samples[device], total_wall),
            ):
                records.append(
                    BenchmarkRecord(
                        mode=mode,
                        token_count=token_count,
                        scope=scope,
                        placement=f"concurrent_{relation}_gpu{device}",
                        device=device,
                        gpu_numa=gpu_numa,
                        weight_numa=weight_numa,
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                        weight_source=weight_source,
                        id_distribution=id_distribution,
                        useful_host_bytes=useful_by_device[device],
                        cuda_samples_ms=samples,
                        wall_samples_ms=wall,
                        concurrency=len(devices),
                    )
                )
    return records


def write_results(
    *,
    output_dir: Path,
    records: Sequence[BenchmarkRecord],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"shared_uva_oe_numa_{timestamp}.jsonl"
    markdown_path = output_dir / f"shared_uva_oe_numa_{timestamp}.md"
    dict_records = [record.as_dict() for record in records]
    with jsonl_path.open("w") as output:
        output.write(json.dumps({"type": "metadata", **metadata}) + "\n")
        for record in dict_records:
            output.write(json.dumps({"type": "record", **record}) + "\n")

    keyed = {
        (record["mode"], record["token_count"], record["scope"], record["placement"]): record
        for record in dict_records
    }
    lines = [
        "# WeLM Shared-UVA OE NUMA Results",
        "",
        "| Mode | Tokens | Scope | Local ms | Remote ms | Remote/Local | Local GiB/s | Remote GiB/s |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    base_keys = sorted({key[:3] for key in keyed})
    for mode, token_count, scope in base_keys:
        local = keyed.get((mode, token_count, scope, "local"))
        remote = keyed.get((mode, token_count, scope, "remote"))
        if local is None or remote is None:
            continue
        local_ms = local["cuda"]["median_ms"]
        remote_ms = remote["cuda"]["median_ms"]
        lines.append(
            f"| {mode} | {token_count} | {scope} | {local_ms:.6f} | "
            f"{remote_ms:.6f} | {remote_ms / local_ms:.3f} | "
            f"{local['useful_gib_per_s']:.3f} | {remote['useful_gib_per_s']:.3f} |"
        )
    concurrent_records = [record for record in dict_records if record["concurrency"] > 1]
    if concurrent_records:
        lines.extend(
            [
                "",
                "| Mode | Tokens | Scope | Device | Relation | Median ms | p90 ms | p99 ms | GiB/s |",
                "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in sorted(
            concurrent_records,
            key=lambda item: (
                item["mode"],
                item["token_count"],
                item["scope"],
                item["device"],
            ),
        ):
            relation = "local" if record["gpu_numa"] == record["weight_numa"] else "remote"
            lines.append(
                f"| {record['mode']} | {record['token_count']} | {record['scope']} | "
                f"{record['device']} | {relation} | {record['cuda']['median_ms']:.6f} | "
                f"{record['cuda']['p90_ms']:.6f} | {record['cuda']['p99_ms']:.6f} | "
                f"{record['useful_gib_per_s']:.3f} |"
            )
    markdown_path.write_text("\n".join(lines) + "\n")
    return jsonl_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    embedding_specs, projection_spec = load_oe_specs(args.checkpoint)
    dry_report = build_dry_run_report(
        checkpoint=args.checkpoint,
        embedding_specs=embedding_specs,
        projection_spec=projection_spec,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        prefill_sizes=args.prefill_sizes,
        decode_sizes=args.decode_sizes,
    )
    if args.dry_run:
        print(json.dumps(dry_report, indent=2))
        return 0
    if args.warmups < 1 or args.repeats < 1:
        raise ValueError("warmups and repeats must be positive")
    if args.max_shard_rows < 0:
        raise ValueError("max-shard-rows must be non-negative")

    import torch

    required_device = max(args.owner_device, args.local_device, args.remote_device)
    if torch.cuda.device_count() <= required_device:
        raise RuntimeError(
            f"benchmark needs CUDA device {required_device}, found {torch.cuda.device_count()}"
        )
    owners, shard_starts, shard_ends, setup_seconds = allocate_oe_shard_owners(
        specs=embedding_specs,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        owner_device=args.owner_device,
        weight_source=args.weight_source,
        max_shard_rows=args.max_shard_rows,
    )
    placements = [
        query_sampled_numa_nodes(owner.data_ptr, owner.nbytes) for owner in owners
    ]
    if not args.skip_numa_placement_check:
        for placement in placements:
            if set(placement) != {args.weight_numa}:
                raise RuntimeError(
                    f"weight pages are not on NUMA{args.weight_numa}: {placement}"
                )

    required_projection_devices = {
        args.local_device,
        args.remote_device,
        *args.concurrent_devices,
    }
    projection_weights = {
        device: load_projection_weight(projection_spec, device, args.weight_source)
        for device in sorted(required_projection_devices)
    }
    gpu_numa_nodes = query_gpu_numa_nodes(torch.cuda.device_count())
    if gpu_numa_nodes[args.local_device] != args.local_gpu_numa:
        raise RuntimeError(
            f"local device {args.local_device} is on NUMA"
            f"{gpu_numa_nodes[args.local_device]}, not NUMA{args.local_gpu_numa}"
        )
    if gpu_numa_nodes[args.remote_device] != args.remote_gpu_numa:
        raise RuntimeError(
            f"remote device {args.remote_device} is on NUMA"
            f"{gpu_numa_nodes[args.remote_device]}, not NUMA{args.remote_gpu_numa}"
        )
    global_rows = [spec.shape[0] for spec in embedding_specs]
    records = []
    for mode, sizes in (
        ("prefill", args.prefill_sizes),
        ("decode", args.decode_sizes),
    ):
        records.extend(
            run_paired_device_matrix(
                mode=mode,
                sizes=sizes,
                local_device=args.local_device,
                remote_device=args.remote_device,
                local_gpu_numa=args.local_gpu_numa,
                remote_gpu_numa=args.remote_gpu_numa,
                weight_numa=args.weight_numa,
                owners=owners,
                shard_starts=shard_starts,
                shard_ends=shard_ends,
                global_rows=global_rows,
                projection_weights=projection_weights,
                tp_size=args.tp_size,
                tp_rank=args.tp_rank,
                weight_source=args.weight_source,
                id_distribution=args.id_distribution,
                warmups=args.warmups,
                repeats=args.repeats,
            )
        )
        if args.concurrent_devices:
            records.extend(
                run_concurrent_matrix(
                    mode=mode,
                    sizes=sizes,
                    devices=args.concurrent_devices,
                    gpu_numa_nodes=gpu_numa_nodes,
                    weight_numa=args.weight_numa,
                    owners=owners,
                    shard_starts=shard_starts,
                    shard_ends=shard_ends,
                    global_rows=global_rows,
                    projection_weights=projection_weights,
                    tp_size=args.tp_size,
                    tp_rank=args.tp_rank,
                    weight_source=args.weight_source,
                    id_distribution=args.id_distribution,
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
            )
    metadata = {
        **dry_report,
        "weight_source": args.weight_source,
        "id_distribution": args.id_distribution,
        "owner_device": args.owner_device,
        "local_device": args.local_device,
        "remote_device": args.remote_device,
        "weight_numa": args.weight_numa,
        "max_shard_rows": args.max_shard_rows,
        "allocated_embedding_bytes": sum(owner.nbytes for owner in owners),
        "setup_seconds": setup_seconds,
        "sampled_page_placements": placements,
        "numa_placement_check_skipped": args.skip_numa_placement_check,
        "oe_path": "unfused_prehashed_lookup_linear",
        "torch_version": torch.__version__,
        "concurrent_devices": args.concurrent_devices,
        "gpu_numa_nodes": gpu_numa_nodes,
    }
    jsonl_path, markdown_path = write_results(
        output_dir=args.output_dir,
        records=records,
        metadata=metadata,
    )
    print(markdown_path.read_text())
    print(f"JSONL: {jsonl_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
