#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import bench_shared_uva_oe_placement as benchmark
from bench_shared_uva_oe_placement import (
    FullVocabTopology,
    PlacementPolicy,
    aggregate_worker_results,
    build_arg_parser,
    critical_path_samples,
    create_full_table_files,
    fault_sampled_pages,
    full_table_storage_bytes,
    local_prefill_tokens,
    validate_pair_checksums,
    validate_sampled_placements,
    worker_hash_seed,
)


class FullVocabTopologyTest(unittest.TestCase):
    def test_cp4_input_tp2_rank_layout(self):
        topology = FullVocabTopology(input_tp_size=2, world_size=8)

        self.assertEqual(topology.cp_size, 4)
        self.assertEqual(
            topology.input_tp_groups,
            ((0, 1), (2, 3), (4, 5), (6, 7)),
        )
        self.assertEqual(topology.cp_rank(6), 3)
        self.assertEqual(topology.input_tp_rank(6), 0)
        self.assertEqual(topology.input_tp_rank(7), 1)

    def test_rejects_non_divisible_world_size(self):
        with self.assertRaisesRegex(ValueError, "divide"):
            FullVocabTopology(input_tp_size=3, world_size=8)

    def test_pair_lanes_share_hash_seed_but_cp_shards_do_not(self):
        topology = FullVocabTopology(input_tp_size=2, world_size=8)

        seeds = [
            worker_hash_seed(topology, rank, "prefill", 16384)
            for rank in range(topology.world_size)
        ]

        self.assertEqual(seeds[0], seeds[1])
        self.assertEqual(seeds[2], seeds[3])
        self.assertNotEqual(seeds[0], seeds[2])


class PlacementPolicyTest(unittest.TestCase):
    def test_sampling_faults_new_mapping_before_move_pages_query(self):
        events = []

        class RecordingAdapter:
            def sample(self, **kwargs):
                events.append("sample")
                return {0: 1}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.bf16"
            path.write_bytes(bytes(8192))
            file_spec = SimpleNamespace(path=path, nbytes=8192)
            with mock.patch.object(
                benchmark,
                "LinuxNumaPlacementAdapter",
                return_value=RecordingAdapter(),
            ), mock.patch.object(
                benchmark,
                "fault_sampled_pages",
                side_effect=lambda *args, **kwargs: events.append("fault"),
            ):
                benchmark.sample_full_table_placements((file_spec,))

        self.assertEqual(events, ["fault", "sample"])

    def test_faults_the_same_evenly_spaced_pages_used_for_sampling(self):
        class RecordingMapping:
            def __init__(self):
                self.offsets = []

            def __getitem__(self, offset):
                self.offsets.append(offset)
                return 0

        mapping = RecordingMapping()

        fault_sampled_pages(mapping, nbytes=10 * 4096, max_samples=4, page_size=4096)

        self.assertEqual(mapping.offsets, [0, 8192, 20480, 28672])

    def test_default_sampling_does_not_alias_two_node_page_interleave(self):
        class RecordingMapping:
            def __init__(self):
                self.offsets = []

            def __getitem__(self, offset):
                self.offsets.append(offset)
                return 0

        mapping = RecordingMapping()

        fault_sampled_pages(mapping, nbytes=1024 * 4096, page_size=4096)

        sampled_page_parity = {offset // 4096 % 2 for offset in mapping.offsets}
        self.assertEqual(sampled_page_parity, {0, 1})

    def test_supported_placement_policies(self):
        self.assertEqual(
            PlacementPolicy.parse("bind-numa0"),
            PlacementPolicy(name="bind-numa0", mode="bind", nodes=(0,)),
        )
        self.assertEqual(
            PlacementPolicy.parse("bind-numa1"),
            PlacementPolicy(name="bind-numa1", mode="bind", nodes=(1,)),
        )
        self.assertEqual(
            PlacementPolicy.parse("interleave"),
            PlacementPolicy(name="interleave", mode="interleave", nodes=(0, 1)),
        )
        self.assertEqual(
            PlacementPolicy.parse("replicate-numa"),
            PlacementPolicy(
                name="replicate-numa", mode="replicate", nodes=(0, 1)
            ),
        )

    def test_rejects_unknown_placement(self):
        with self.assertRaisesRegex(ValueError, "unknown placement"):
            PlacementPolicy.parse("automatic")

    def test_placement_validation_distinguishes_requested_policies(self):
        sampled = {
            "table0": {0: 8, 1: 8},
            "table1": {0: 7, 1: 9},
        }

        validate_sampled_placements(PlacementPolicy.parse("interleave"), sampled)
        with self.assertRaisesRegex(RuntimeError, "NUMA0"):
            validate_sampled_placements(PlacementPolicy.parse("bind-numa0"), sampled)

    def test_interleave_requires_both_nodes_in_every_table(self):
        with self.assertRaisesRegex(RuntimeError, "interleaved"):
            validate_sampled_placements(
                PlacementPolicy.parse("interleave"), {"table0": {0: 16}}
            )


class WorkloadAccountingTest(unittest.TestCase):
    def test_global_prefill_is_divided_across_cp_ranks(self):
        self.assertEqual(local_prefill_tokens(16384, cp_size=4), 4096)
        self.assertEqual(local_prefill_tokens(65536, cp_size=4), 16384)

    def test_prefill_requires_balanced_divisible_split(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            local_prefill_tokens(16385, cp_size=4)

    def test_full_table_storage_counts_one_physical_copy(self):
        shapes = ((16_000_008, 512), (16_000_016, 512), (16_000_024, 512))

        expected = sum(rows * width * 2 for rows, width in shapes)

        self.assertEqual(full_table_storage_bytes(shapes), expected)

    def test_resident_measurement_counts_faulted_file_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resident.bin"
            path.write_bytes(bytes([1]) * 8192)
            replicas = (
                SimpleNamespace(
                    table_files=(SimpleNamespace(path=path, nbytes=8192),)
                ),
            )

            resident, allocated, by_path = benchmark.measure_resident_file_bytes(
                replicas
            )

            self.assertEqual(resident, 8192)
            self.assertGreaterEqual(allocated, 8192)
            self.assertEqual(by_path[str(path.resolve())], 8192)

    def test_checkpoint_row_checksums_use_first_middle_and_last_rows(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            tensor = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
            save_file({"weight": tensor}, path)
            specs = (
                SimpleNamespace(key="weight", path=path, shape=(5, 4)),
            )

            checksums = benchmark.checkpoint_row_checksums(specs)

            expected = benchmark._tensor_bytes_digest(tensor[[0, 2, 4]])
            self.assertEqual(checksums, {"weight": expected})

    def test_full_table_files_create_exactly_one_file_per_branch(self):
        keys = (
            "model.embed_tokens.weight",
            "model.oe_embed.0.weight",
            "model.oe_embed.1.weight",
            "model.oe_embed.2.weight",
            "model.oe_embed.3.weight",
        )
        sources = tuple(
            SimpleNamespace(
                shape=(rows, 8),
                key=key,
                path=Path(f"checkpoint-{index}.safetensors"),
            )
            for index, (key, rows) in enumerate(zip(keys, (9, 10, 11, 12, 13)))
        )
        with tempfile.TemporaryDirectory() as directory:
            files = create_full_table_files(
                embedding_specs=sources,
                root=Path(directory) / "arena",
                max_rows=0,
            )

            self.assertEqual(len(files), 5)
            self.assertEqual([file.key for file in files], list(keys))
            self.assertEqual(
                [file.shape for file in files],
                [(9, 8), (10, 8), (11, 8), (12, 8), (13, 8)],
            )
            self.assertEqual(sum(file.nbytes for file in files), 55 * 8 * 2)
            self.assertTrue(
                all(file.path.stat().st_size == file.nbytes for file in files)
            )

    def test_replicated_layout_creates_one_full_copy_per_numa(self):
        keys = ("model.embed_tokens.weight",) + tuple(
            f"model.oe_embed.{index}.weight" for index in range(4)
        )
        sources = tuple(
            SimpleNamespace(
                shape=(rows, 8),
                key=key,
                path=Path(f"checkpoint-{index}.safetensors"),
            )
            for index, (key, rows) in enumerate(zip(keys, (9, 10, 11, 12, 13)))
        )
        with tempfile.TemporaryDirectory() as directory:
            replicas = benchmark.create_table_replicas(
                embedding_specs=sources,
                root=Path(directory) / "arena",
                max_rows=0,
                policy=PlacementPolicy.parse("replicate-numa"),
            )

            self.assertEqual([replica.numa_node for replica in replicas], [0, 1])
            self.assertTrue(all(len(replica.table_files) == 5 for replica in replicas))
            self.assertEqual(
                sum(
                    file.nbytes
                    for replica in replicas
                    for file in replica.table_files
                ),
                2 * 55 * 8 * 2,
            )
            self.assertEqual(
                benchmark.select_table_files_for_numa(replicas, 0),
                replicas[0].table_files,
            )
            self.assertEqual(
                benchmark.select_table_files_for_numa(replicas, 1),
                replicas[1].table_files,
            )

    def test_single_copy_layout_is_shared_by_both_numa_nodes(self):
        replica = SimpleNamespace(numa_node=None, table_files=("shared",))

        self.assertEqual(
            benchmark.select_table_files_for_numa((replica,), 0), ("shared",)
        )
        self.assertEqual(
            benchmark.select_table_files_for_numa((replica,), 1), ("shared",)
        )

    def test_replicated_placement_validation_checks_each_local_copy(self):
        replicas = (
            SimpleNamespace(
                numa_node=0,
                table_files=(SimpleNamespace(path=Path("numa0/table")),),
            ),
            SimpleNamespace(
                numa_node=1,
                table_files=(SimpleNamespace(path=Path("numa1/table")),),
            ),
        )
        placements = {
            "numa0/table": {0: 127},
            "numa1/table": {1: 127},
        }

        benchmark.validate_replica_placements(
            PlacementPolicy.parse("replicate-numa"), replicas, placements
        )
        placements["numa1/table"] = {0: 127}
        with self.assertRaisesRegex(RuntimeError, "NUMA1"):
            benchmark.validate_replica_placements(
                PlacementPolicy.parse("replicate-numa"), replicas, placements
            )

    def test_cli_defaults_match_focused_full_vocab_matrix(self):
        args = build_arg_parser().parse_args(["--checkpoint", "/tmp/model"])

        self.assertEqual(args.input_tp_size, 2)
        self.assertEqual(args.world_size, 8)
        self.assertEqual(args.placement, "interleave")
        self.assertEqual(args.prefill_sizes, [16384, 65536])
        self.assertEqual(args.decode_sizes, [1, 32])
        self.assertEqual(args.warmups, 10)
        self.assertEqual(args.repeats, 50)
        self.assertEqual(
            args.output_dir,
            Path("oe-bench-results/shared-uva-production"),
        )
        self.assertFalse(args.dry_run)


class AggregationTest(unittest.TestCase):
    def test_critical_path_keeps_pair_and_global_maxima(self):
        topology = FullVocabTopology(input_tp_size=2, world_size=8)
        rank_samples = {
            rank: (float(rank + 1), float(10 - rank)) for rank in range(8)
        }

        pair_samples, global_samples = critical_path_samples(topology, rank_samples)

        self.assertEqual(pair_samples[0], (2.0, 10.0))
        self.assertEqual(pair_samples[1], (4.0, 8.0))
        self.assertEqual(pair_samples[2], (6.0, 6.0))
        self.assertEqual(pair_samples[3], (8.0, 4.0))
        self.assertEqual(global_samples, (8.0, 10.0))

    def test_critical_path_requires_every_rank(self):
        topology = FullVocabTopology(input_tp_size=2, world_size=8)

        with self.assertRaisesRegex(ValueError, "every rank"):
            critical_path_samples(topology, {0: (1.0,)})

    def test_worker_aggregation_uses_global_critical_path(self):
        topology = FullVocabTopology(input_tp_size=2, world_size=8)
        worker_results = [
            {
                "rank": rank,
                "records": [
                    {
                        "mode": "prefill",
                        "global_tokens": 16384,
                        "local_tokens": 4096,
                        "scope": scope,
                        "samples_ms": [float(rank + 1), float(10 - rank)],
                    }
                    for scope in ("base_lookup", "oe_total", "combined")
                ],
            }
            for rank in range(8)
        ]

        aggregated = aggregate_worker_results(topology, worker_results)

        self.assertEqual(
            [record["scope"] for record in aggregated],
            ["base_lookup", "combined", "oe_total"],
        )
        for record in aggregated:
            self.assertEqual(record["global_critical_path"]["median_ms"], 9.0)
            self.assertEqual(record["pair_critical_paths"]["0"]["median_ms"], 6.0)

    def test_pair_lanes_must_read_identical_lookup_values(self):
        topology = FullVocabTopology(input_tp_size=2, world_size=4)
        worker_results = [
            {
                "rank": rank,
                "checksums": [
                    {
                        "mode": "prefill",
                        "global_tokens": 16384,
                        "local_tokens": 8192,
                        "base_lookup_checksum": 10.0 + rank // 2,
                        "oe_lookup_checksum": 20.0 + rank // 2,
                        "combined_checksum": 30.0 + rank // 2,
                    }
                ],
            }
            for rank in range(4)
        ]

        validate_pair_checksums(topology, worker_results)
        worker_results[1]["checksums"][0]["combined_checksum"] = 99.0
        with self.assertRaisesRegex(RuntimeError, "CP pair 0"):
            validate_pair_checksums(topology, worker_results)


if __name__ == "__main__":
    unittest.main()
