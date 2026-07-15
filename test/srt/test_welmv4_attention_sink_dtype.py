import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.welmv4 import _get_welm_attention_sink_dtype


class TestWelmV4AttentionSinkDtype(unittest.TestCase):
    def _get_dtype(self, **backends):
        server_args = SimpleNamespace(
            attention_backend=backends.get("attention_backend"),
            prefill_attention_backend=backends.get("prefill_attention_backend"),
            decode_attention_backend=backends.get("decode_attention_backend"),
        )
        with patch(
            "sglang.srt.models.welmv4.get_global_server_args",
            return_value=server_args,
        ):
            return _get_welm_attention_sink_dtype()

    def test_trtllm_mha_uses_float32_sink(self):
        for backend_field in (
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
        ):
            with self.subTest(backend_field=backend_field):
                self.assertEqual(
                    self._get_dtype(**{backend_field: "trtllm_mha"}),
                    torch.float32,
                )

    def test_other_backends_preserve_default_dtype(self):
        self.assertIsNone(
            self._get_dtype(
                attention_backend="fa3",
                prefill_attention_backend="fa3",
                decode_attention_backend="fa3",
            )
        )


if __name__ == "__main__":
    unittest.main()
