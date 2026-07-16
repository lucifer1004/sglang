import math
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_shared_uva_oe_numa import (  # noqa: E402
    HostWeightOwner,
    alternating_order,
    build_arg_parser,
    make_lookup_modules,
    parse_int_list,
    run_lookup_concat,
    shard_bounds,
    summarize_samples,
    time_concurrent_operations,
)


def _has_cuda_devices(count: int) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.device_count() >= count


class BenchmarkHelpersTest(unittest.TestCase):
    def test_shard_bounds_match_real_tp2_oe_tables(self):
        cases = [
            (16_000_008, 0, (0, 8_000_032, 8_000_032)),
            (16_000_008, 1, (8_000_032, 16_000_008, 8_000_032)),
            (16_000_016, 0, (0, 8_000_032, 8_000_032)),
            (16_000_016, 1, (8_000_032, 16_000_016, 8_000_032)),
            (16_000_024, 0, (0, 8_000_032, 8_000_032)),
            (16_000_024, 1, (8_000_032, 16_000_024, 8_000_032)),
            (16_000_032, 0, (0, 8_000_032, 8_000_032)),
            (16_000_032, 1, (8_000_032, 16_000_032, 8_000_032)),
        ]
        for rows, rank, expected in cases:
            with self.subTest(rows=rows, rank=rank):
                self.assertEqual(
                    shard_bounds(rows, tp_size=2, tp_rank=rank), expected
                )

    def test_shard_bounds_reject_invalid_layout(self):
        cases = [
            {"rows": 0, "tp_size": 2, "tp_rank": 0},
            {"rows": 10, "tp_size": 0, "tp_rank": 0},
            {"rows": 10, "tp_size": 2, "tp_rank": -1},
            {"rows": 10, "tp_size": 2, "tp_rank": 2},
            {"rows": 10, "tp_size": 2, "tp_rank": 0, "padding": 0},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                shard_bounds(**kwargs)

    def test_summarize_samples_reports_linear_percentiles(self):
        summary = summarize_samples([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 4.0)
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertEqual(summary["median_ms"], 2.5)
        self.assertAlmostEqual(summary["p90_ms"], 3.7)
        self.assertAlmostEqual(summary["p99_ms"], 3.97)
        self.assertAlmostEqual(summary["stddev_ms"], math.sqrt(1.25))

    def test_summarize_samples_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "samples"):
            summarize_samples([])

    def test_parse_int_list_normalizes_values(self):
        self.assertEqual(parse_int_list("1, 2,32, 128"), [1, 2, 32, 128])

    def test_parse_int_list_rejects_invalid_values(self):
        for value in ["", "1,,2", "0,1", "-1,2", "1,two"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_int_list(value)

    def test_cli_defaults_cover_prefill_and_decode_matrix(self):
        args = build_arg_parser().parse_args(["--checkpoint", "/tmp/model"])

        self.assertEqual(args.prefill_sizes, [256, 1024, 4096, 16384])
        self.assertEqual(args.decode_sizes, [1, 2, 4, 8, 16, 32, 64, 128])
        self.assertEqual(args.tp_size, 2)
        self.assertEqual(args.tp_rank, 0)
        self.assertEqual(args.local_device, 0)
        self.assertEqual(args.remote_device, 4)

    def test_cli_accepts_gpu_zero_in_concurrent_devices(self):
        args = build_arg_parser().parse_args(
            [
                "--checkpoint",
                "/tmp/model",
                "--concurrent-devices",
                "0,2,4,6",
            ]
        )

        self.assertEqual(args.concurrent_devices, [0, 2, 4, 6])

    def test_paired_order_alternates_local_and_remote(self):
        self.assertEqual(
            alternating_order(("local", "remote"), 0), ("local", "remote")
        )
        self.assertEqual(
            alternating_order(("local", "remote"), 1), ("remote", "local")
        )


@unittest.skipUnless(_has_cuda_devices(5), "requires at least five CUDA devices")
class HostWeightOwnerCudaTest(unittest.TestCase):
    def test_one_owner_supports_local_and_remote_device_views(self):
        import torch

        self.assertGreaterEqual(torch.cuda.device_count(), 5)
        owner = HostWeightOwner.allocate(
            shape=(1024, 512), dtype=torch.bfloat16, owner_device=0
        )
        owner.tensor.fill_(1)
        torch.cuda.synchronize(0)

        local = owner.device_view(0)
        remote = owner.device_view(4)
        ids0 = torch.tensor([1, 7, 13], dtype=torch.int64, device=0)
        ids4 = ids0.to(4)
        local_output = torch.nn.functional.embedding(ids0, local).cpu()
        remote_output = torch.nn.functional.embedding(ids4, remote).cpu()

        torch.testing.assert_close(local_output, remote_output, rtol=0, atol=0)
        self.assertIs(owner.device_view(0), local)
        with self.assertRaises(TypeError):
            copy.copy(owner)

    def test_closed_owner_rejects_new_views(self):
        import torch

        owner = HostWeightOwner.allocate(
            shape=(16, 512), dtype=torch.bfloat16, owner_device=0
        )
        owner.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            owner.device_view(0)

    def test_production_lookup_matches_for_local_and_remote_views(self):
        import torch

        owners = [
            HostWeightOwner.allocate(
                shape=(4096, 512), dtype=torch.bfloat16, owner_device=0
            )
            for _ in range(4)
        ]
        for index, owner in enumerate(owners):
            owner.tensor.fill_(index + 1)
        torch.cuda.synchronize(0)

        local_modules = make_lookup_modules(
            owners=owners,
            device=0,
            shard_starts=[0, 0, 0, 0],
            shard_ends=[4096, 4096, 4096, 4096],
        )
        remote_modules = make_lookup_modules(
            owners=owners,
            device=4,
            shard_starts=[0, 0, 0, 0],
            shard_ends=[4096, 4096, 4096, 4096],
        )
        generator = torch.Generator(device=0).manual_seed(7)
        hashed0 = [
            torch.randint(0, 4096, (257,), device=0, generator=generator)
            for _ in range(4)
        ]
        hashed4 = [tensor.to(4) for tensor in hashed0]

        local_output = run_lookup_concat(local_modules, hashed0).cpu()
        remote_output = run_lookup_concat(remote_modules, hashed4).cpu()

        self.assertEqual(tuple(local_output.shape), (257, 2048))
        self.assertEqual(local_output.dtype, torch.bfloat16)
        torch.testing.assert_close(local_output, remote_output, rtol=0, atol=0)

    def test_concurrent_timer_runs_local_and_remote_operations(self):
        import torch

        owners = [
            HostWeightOwner.allocate(
                shape=(4096, 512), dtype=torch.bfloat16, owner_device=0
            )
            for _ in range(4)
        ]
        for index, owner in enumerate(owners):
            owner.tensor.fill_(index + 1)
        torch.cuda.synchronize(0)
        operations = {}
        inputs = {}
        for device in (0, 4):
            modules = make_lookup_modules(
                owners=owners,
                device=device,
                shard_starts=[0, 0, 0, 0],
                shard_ends=[4096, 4096, 4096, 4096],
            )
            batches = [
                [
                    torch.randint(0, 4096, (257,), device=device)
                    for _ in range(4)
                ]
                for _ in range(4)
            ]
            operations[device] = lambda value, modules=modules: run_lookup_concat(
                modules, value
            )
            inputs[device] = batches

        device_samples, wall_samples = time_concurrent_operations(
            devices=[0, 4],
            inputs_by_device=inputs,
            operations_by_device=operations,
            warmups=1,
            repeats=3,
        )

        self.assertEqual(len(device_samples[0]), 3)
        self.assertEqual(len(device_samples[4]), 3)
        self.assertEqual(len(wall_samples), 3)
        self.assertTrue(all(value > 0 for value in wall_samples))


if __name__ == "__main__":
    unittest.main()
