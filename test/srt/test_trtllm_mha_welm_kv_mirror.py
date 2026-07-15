import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMHAAttnBackend,
    TRTLLMMHAMetadata,
)


class TestTRTLLMMHAWeLMKVMirror(unittest.TestCase):
    def _get_context_metadata(self, metadata, q, forward_batch, page_table):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.forward_metadata = metadata
        return backend._get_context_attention_metadata(q, forward_batch, page_table)

    def test_uncontracted_query_keeps_extend_metadata(self):
        metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=torch.tensor([4, 7], dtype=torch.int32),
            max_seq_len_q=3,
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 4, 11], dtype=torch.int32),
        )
        q = torch.empty(5, 12, 256)
        page_table = torch.arange(8).reshape(2, 4)
        forward_batch = SimpleNamespace(welm_kv_mirror_contracted=False)

        result = self._get_context_metadata(metadata, q, forward_batch, page_table)

        self.assertIs(result[0], page_table)
        self.assertIs(result[1], metadata.cache_seqlens_int32)
        self.assertEqual(result[2], 3)
        self.assertIs(result[3], metadata.cu_seqlens_q)
        self.assertIs(result[4], metadata.cu_seqlens_k)

    def test_contracted_query_uses_one_query_per_active_request(self):
        metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=torch.tensor([10, 20, 30], dtype=torch.int32),
            max_seq_len_q=8,
            cu_seqlens_q=torch.tensor([0, 4, 7, 15], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 10, 30, 60], dtype=torch.int32),
            mirror_cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        )
        q = torch.empty(2, 12, 256)
        page_table = torch.arange(12).reshape(3, 4)
        forward_batch = SimpleNamespace(
            welm_kv_mirror_contracted=True,
            custom_last_index=torch.tensor([3, 14]),
            kv_mirror_active_batch_indices=torch.tensor([0, 2]),
        )

        result = self._get_context_metadata(metadata, q, forward_batch, page_table)

        torch.testing.assert_close(result[0], page_table[[0, 2]])
        torch.testing.assert_close(
            result[1], torch.tensor([10, 30], dtype=torch.int32)
        )
        self.assertEqual(result[2], 1)
        torch.testing.assert_close(
            result[3], torch.tensor([0, 1, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(
            result[4], torch.tensor([0, 10, 40], dtype=torch.int32)
        )

    def test_contracted_query_rejects_missing_mirror_metadata(self):
        metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=torch.tensor([10], dtype=torch.int32),
            max_seq_len_q=10,
            cu_seqlens_q=torch.tensor([0, 10], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 10], dtype=torch.int32),
        )
        forward_batch = SimpleNamespace(
            welm_kv_mirror_contracted=True,
            custom_last_index=torch.tensor([9]),
        )

        with self.assertRaisesRegex(RuntimeError, "missing TRT-LLM context metadata"):
            self._get_context_metadata(
                metadata,
                torch.empty(1, 12, 256),
                forward_batch,
                torch.zeros(1, 2),
            )


if __name__ == "__main__":
    unittest.main()
