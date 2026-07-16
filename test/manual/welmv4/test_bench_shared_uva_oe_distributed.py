#!/usr/bin/env python3

import mmap
import tempfile
import unittest
from pathlib import Path

from bench_shared_uva_oe_distributed import (
    SharedHostTensor,
    Topology,
    aggregate_group_samples,
    aggregate_worker_results,
    build_arg_parser,
    iteration_access_order,
    physical_shard_keys,
    physical_storage_bytes,
    worker_access_shard_keys,
    worker_shard_key,
)


def _has_cuda_devices(count: int) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.device_count() >= count


class TopologyTest(unittest.TestCase):
    def test_tp2_cp4_contiguous_groups_and_shard_consumers(self):
        topology = Topology(attn_tp_size=2, world_size=8)

        self.assertEqual(topology.cp_size, 4)
        self.assertEqual(topology.group_ranks, ((0, 1), (2, 3), (4, 5), (6, 7)))
        self.assertEqual(topology.tp_rank(6), 0)
        self.assertEqual(topology.cp_rank(6), 3)
        self.assertEqual(topology.shard_consumers(0), (0, 2, 4, 6))
        self.assertEqual(topology.shard_consumers(1), (1, 3, 5, 7))

    def test_tp4_cp2_and_tp8_cp1(self):
        tp4 = Topology(attn_tp_size=4, world_size=8)
        tp8 = Topology(attn_tp_size=8, world_size=8)

        self.assertEqual(tp4.group_ranks, ((0, 1, 2, 3), (4, 5, 6, 7)))
        self.assertEqual(tp4.shard_consumers(3), (3, 7))
        self.assertEqual(tp8.group_ranks, (tuple(range(8)),))
        self.assertEqual(tp8.shard_consumers(5), (5,))

    def test_rejects_non_divisible_topology(self):
        with self.assertRaisesRegex(ValueError, "divide"):
            Topology(attn_tp_size=3, world_size=8)


class PlacementPolicyTest(unittest.TestCase):
    def test_global_policy_has_one_copy_per_tp_rank(self):
        topology = Topology(attn_tp_size=2, world_size=8)

        self.assertEqual(
            physical_shard_keys(topology, "global-numa0"),
            ((0, 0), (1, 0)),
        )
        self.assertEqual(physical_storage_bytes(topology, "global-numa0", 100), 200)

    def test_local_numa_duplicates_only_shared_cross_numa_shards(self):
        tp2 = Topology(attn_tp_size=2, world_size=8)
        tp4 = Topology(attn_tp_size=4, world_size=8)
        tp8 = Topology(attn_tp_size=8, world_size=8)

        self.assertEqual(
            physical_shard_keys(tp2, "local-numa"),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
        )
        self.assertEqual(physical_storage_bytes(tp2, "local-numa", 100), 400)
        self.assertEqual(physical_storage_bytes(tp4, "local-numa", 100), 800)
        self.assertEqual(physical_storage_bytes(tp8, "local-numa", 100), 800)

    def test_worker_selects_global_or_local_numa_copy(self):
        topology = Topology(attn_tp_size=2, world_size=8)

        self.assertEqual(worker_shard_key(topology, "global-numa0", 6), (0, 0))
        self.assertEqual(worker_shard_key(topology, "global-numa1", 1), (1, 1))
        self.assertEqual(worker_shard_key(topology, "local-numa", 2), (0, 0))
        self.assertEqual(worker_shard_key(topology, "local-numa", 6), (0, 1))

    def test_paired_numa_materializes_both_nodes_for_every_tp_shard(self):
        for attn_tp_size in (2, 4, 8):
            topology = Topology(attn_tp_size=attn_tp_size, world_size=8)

            self.assertEqual(
                physical_shard_keys(topology, "paired-numa"),
                tuple(
                    (tp_rank, numa_node)
                    for tp_rank in range(attn_tp_size)
                    for numa_node in (0, 1)
                ),
            )
            self.assertEqual(
                physical_storage_bytes(topology, "paired-numa", 100),
                attn_tp_size * 2 * 100,
            )

    def test_paired_numa_exposes_local_and_remote_copy_to_each_worker(self):
        topology = Topology(attn_tp_size=2, world_size=8)

        self.assertEqual(
            worker_access_shard_keys(topology, "paired-numa", 2),
            {"local": (0, 0), "remote": (0, 1)},
        )
        self.assertEqual(
            worker_access_shard_keys(topology, "paired-numa", 6),
            {"local": (0, 1), "remote": (0, 0)},
        )
        self.assertEqual(
            worker_access_shard_keys(topology, "global-numa0", 6),
            {"default": (0, 0)},
        )

    def test_paired_global_exposes_both_single_owner_placements(self):
        topology = Topology(attn_tp_size=2, world_size=8)

        self.assertEqual(
            physical_shard_keys(topology, "paired-global"),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
        )
        self.assertEqual(
            worker_access_shard_keys(topology, "paired-global", 6),
            {"numa0": (0, 0), "numa1": (0, 1)},
        )


class AggregationTest(unittest.TestCase):
    def test_group_latency_is_iteration_wise_slowest_rank(self):
        topology = Topology(attn_tp_size=2, world_size=8)
        rank_samples = {
            rank: [float(rank + 1), float(10 - rank)] for rank in range(8)
        }

        group_samples = aggregate_group_samples(topology, rank_samples)

        self.assertEqual(group_samples[0], (2.0, 10.0))
        self.assertEqual(group_samples[1], (4.0, 8.0))
        self.assertEqual(group_samples[2], (6.0, 6.0))
        self.assertEqual(group_samples[3], (8.0, 4.0))

    def test_worker_aggregation_keeps_local_and_remote_samples_separate(self):
        topology = Topology(attn_tp_size=2, world_size=8)
        worker_results = [
            {
                "rank": rank,
                "records": [
                    {
                        "mode": "decode",
                        "token_count": 1,
                        "scope": "lookup",
                        "access": "local",
                        "samples_ms": [1.0 + rank],
                        "wall_start_ns": [1_000_000_000 + rank * 100_000],
                        "wall_end_ns": [
                            1_000_000_000 + rank * 100_000 + int((1.0 + rank) * 1e6)
                        ],
                    },
                    {
                        "mode": "decode",
                        "token_count": 1,
                        "scope": "lookup",
                        "access": "remote",
                        "samples_ms": [11.0 + rank],
                        "wall_start_ns": [2_000_000_000 + rank * 100_000],
                        "wall_end_ns": [
                            2_000_000_000 + rank * 100_000 + int((11.0 + rank) * 1e6)
                        ],
                    },
                ],
            }
            for rank in range(8)
        ]

        aggregated = aggregate_worker_results(topology, worker_results)

        self.assertEqual([record["access"] for record in aggregated], ["local", "remote"])
        self.assertEqual(
            [record["global_critical_path"]["median_ms"] for record in aggregated],
            [8.0, 18.0],
        )

    def test_paired_access_order_alternates_each_iteration(self):
        self.assertEqual(
            iteration_access_order(("local", "remote"), 0),
            ("local", "remote"),
        )
        self.assertEqual(
            iteration_access_order(("local", "remote"), 1),
            ("remote", "local"),
        )


class CliTest(unittest.TestCase):
    def test_defaults_cover_prefill_decode_and_both_reduce_orders(self):
        args = build_arg_parser().parse_args(["--checkpoint", "/tmp/model"])

        self.assertEqual(args.prefill_sizes, [256, 1024, 4096, 16384])
        self.assertEqual(args.decode_sizes, [1, 8, 32, 128])
        self.assertEqual(args.attn_tp_size, 2)
        self.assertEqual(args.placement, "global-numa0")
        self.assertEqual(args.reduce_orders, ["pre-proj", "post-proj"])
        self.assertEqual(args.weight_source, "checkpoint")
        self.assertEqual(args.warmups, 10)
        self.assertEqual(args.repeats, 100)


@unittest.skipUnless(_has_cuda_devices(1), "requires a CUDA device")
class SharedHostTensorCudaTest(unittest.TestCase):
    def test_registered_mmap_is_directly_readable_by_gpu(self):
        import torch

        rows, width = 64, 16
        nbytes = rows * width * torch.bfloat16.itemsize
        with tempfile.NamedTemporaryFile(dir="/dev/shm") as handle:
            handle.truncate(nbytes)
            with mmap.mmap(handle.fileno(), nbytes, access=mmap.ACCESS_WRITE) as mapping:
                source = torch.frombuffer(mapping, dtype=torch.bfloat16).view(rows, width)
                source.copy_(torch.arange(rows, dtype=torch.bfloat16).view(-1, 1))

            shared = SharedHostTensor.open(
                path=Path(handle.name),
                shape=(rows, width),
                dtype=torch.bfloat16,
                device=0,
            )
            try:
                ids = torch.tensor([0, 7, 63], dtype=torch.int64, device="cuda:0")
                output = torch.nn.functional.embedding(ids, shared.cuda_tensor).cpu()
            finally:
                shared.close()

        expected = torch.tensor([0, 7, 63], dtype=torch.bfloat16).view(-1, 1)
        self.assertTrue(torch.equal(output, expected.expand(-1, width)))


if __name__ == "__main__":
    unittest.main()
