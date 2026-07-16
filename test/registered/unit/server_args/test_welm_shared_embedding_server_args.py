import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs, prepare_server_args


class TestWelmSharedEmbeddingArgs(unittest.TestCase):
    def test_cli_defaults_to_disabled(self):
        args = prepare_server_args(["--model-path", "dummy"])

        self.assertEqual(args.welm_shared_embedding_policy, "disabled")
        self.assertIsNone(args.welm_shared_embedding_numa_node)

    def test_cli_parses_bind_policy_and_node(self):
        args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--enable-over-encoding",
                "--welm-shared-embedding-policy",
                "bind",
                "--welm-shared-embedding-numa-node",
                "1",
            ]
        )

        self.assertEqual(args.welm_shared_embedding_policy, "bind")
        self.assertEqual(args.welm_shared_embedding_numa_node, 1)

    def test_shared_policy_requires_over_encoding(self):
        with self.assertRaisesRegex(ValueError, "--enable-over-encoding"):
            ServerArgs(
                model_path="dummy",
                welm_shared_embedding_policy="interleave",
            )

    def test_cli_rejects_first_touch_policy(self):
        with self.assertRaises(SystemExit):
            prepare_server_args(
                [
                    "--model-path",
                    "dummy",
                    "--enable-over-encoding",
                    "--welm-shared-embedding-policy",
                    "first-touch",
                ]
            )

    def test_bind_policy_requires_numa_node(self):
        with self.assertRaisesRegex(
            ValueError, "--welm-shared-embedding-numa-node"
        ):
            ServerArgs(
                model_path="dummy",
                enable_over_encoding=True,
                welm_shared_embedding_policy="bind",
            )

    def test_non_bind_policy_rejects_numa_node(self):
        with self.assertRaisesRegex(ValueError, "only valid"):
            ServerArgs(
                model_path="dummy",
                enable_over_encoding=True,
                welm_shared_embedding_policy="interleave",
                welm_shared_embedding_numa_node=0,
            )

    def test_bind_policy_rejects_negative_numa_node(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ServerArgs(
                model_path="dummy",
                enable_over_encoding=True,
                welm_shared_embedding_policy="bind",
                welm_shared_embedding_numa_node=-1,
            )

    def test_shared_policy_rejects_incompatible_host_weight_owner(self):
        incompatible = [
            ["--cpu-offload-gb", "1"],
            ["--offload-group-size", "1"],
            ["--enable-weights-cpu-backup"],
            ["--enable-draft-weights-cpu-backup"],
            ["--kt-weight-path", "/tmp/weights"],
            ["--use-ray"],
            ["--weight-loader-disable-mmap"],
            ["--load-format", "remote_instance"],
            ["--load-format", "gguf"],
            ["--load-format", "pt"],
            ["--load-format", "npcache"],
            ["--load-format", "fastsafetensors"],
            ["--load-format", "remote"],
            ["--load-format", "runai_streamer"],
        ]

        for extra_args in incompatible:
            with self.subTest(extra_args=extra_args):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    prepare_server_args(
                        [
                            "--model-path",
                            "dummy",
                            "--enable-over-encoding",
                            "--welm-shared-embedding-policy",
                            "interleave",
                            *extra_args,
                        ]
                    )

    def test_auto_detected_gguf_is_rejected_after_load_format_resolution(self):
        args = ServerArgs(
            model_path="dummy",
            enable_over_encoding=True,
            welm_shared_embedding_policy="interleave",
        )
        args.model_path = "model.gguf"

        with (
            patch("sglang.srt.server_args.check_gguf_file", return_value=True),
            self.assertRaisesRegex(ValueError, "incompatible"),
        ):
            args._handle_load_format()


if __name__ == "__main__":
    unittest.main()
