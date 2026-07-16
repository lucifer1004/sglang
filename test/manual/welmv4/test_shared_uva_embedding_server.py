#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import torch

from sglang.srt.model_loader.welm_shared_embedding_weights import (
    WeLMSharedEmbeddingPolicy,
    WeLMSharedEmbeddingRegistry,
    _query_local_gpu_numa_nodes,
    launch_welm_embedding_arena_process,
    plan_welm_embedding_replicas,
)


def _parse_devices(value: str) -> tuple[int, ...]:
    devices = tuple(int(item) for item in value.split(","))
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise argparse.ArgumentTypeError("devices must be unique non-negative IDs")
    return devices


def _lookup_checksums(manifest_path: str, devices: tuple[int, ...]):
    gpu_numa_nodes = _query_local_gpu_numa_nodes(
        argparse.Namespace(numa_node=None)
    )
    results = []
    for device in devices:
        started = time.perf_counter()
        registry = WeLMSharedEmbeddingRegistry.from_manifest(
            manifest_path,
            gpu_id=device,
            gpu_numa_node=gpu_numa_nodes[device],
        )
        try:
            checksums = {}
            for key in sorted(registry.externally_owned_names()):
                weight = registry.get(key)
                indices = torch.tensor(
                    [0, weight.shape[0] // 2, weight.shape[0] - 1],
                    dtype=torch.int64,
                    device=f"cuda:{device}",
                )
                rows = weight.index_select(0, indices)
                checksums[key] = float(rows.float().sum().cpu().item())
                del rows, indices, weight
            torch.cuda.synchronize(device)
            results.append(
                {
                    "device": device,
                    "gpu_numa_node": gpu_numa_nodes[device],
                    "attach_and_lookup_seconds": time.perf_counter() - started,
                    "checksums": checksums,
                    "diagnostics": registry.diagnostics(),
                }
            )
        finally:
            registry.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("bind", "interleave", "replicate-numa"),
        default="interleave",
    )
    parser.add_argument("--bind-node", type=int)
    parser.add_argument("--devices", type=_parse_devices, default=(0, 4))
    parser.add_argument("--arena-parent", type=Path, default=Path("/dev/shm"))
    args = parser.parse_args()

    gpu_numa_nodes = _query_local_gpu_numa_nodes(
        argparse.Namespace(numa_node=None)
    )
    policy = WeLMSharedEmbeddingPolicy(args.policy)
    plans = plan_welm_embedding_replicas(
        policy,
        gpu_numa_nodes,
        args.bind_node,
    )
    root = args.arena_parent / f"sglang-welm-embedding-smoke-{uuid.uuid4().hex}"
    started = time.perf_counter()
    handle = launch_welm_embedding_arena_process(
        checkpoint=args.checkpoint,
        root=root,
        plans=plans,
        timeout=3600,
    )
    try:
        from sglang.srt.model_loader.welm_shared_embedding_weights import (
            load_welm_embedding_arena_manifest,
        )

        manifest = load_welm_embedding_arena_manifest(handle.manifest_path)
        arena_seconds = time.perf_counter() - started
        consumers = _lookup_checksums(handle.manifest_path, args.devices)
        checksum_sets = [item["checksums"] for item in consumers]
        if any(checksums != checksum_sets[0] for checksums in checksum_sets[1:]):
            raise RuntimeError("consumer checksums differ across GPUs")
        print(
            json.dumps(
                {
                    "policy": manifest.policy,
                    "arena_id": manifest.arena_id,
                    "arena_seconds": arena_seconds,
                    "logical_weight_bytes": manifest.logical_weight_bytes,
                    "physical_weight_bytes": manifest.physical_weight_bytes,
                    "replicas": [
                        {
                            "replica_id": replica.replica_id,
                            "numa_nodes": replica.numa_nodes,
                            "inodes": [tensor.inode for tensor in replica.tensors],
                            "sampled_numa_nodes": {
                                tensor.key: tensor.sampled_numa_nodes
                                for tensor in replica.tensors
                            },
                        }
                        for replica in manifest.replicas
                    ],
                    "consumers": consumers,
                },
                indent=2,
            )
        )
    finally:
        handle.close(timeout=60)


if __name__ == "__main__":
    main()
