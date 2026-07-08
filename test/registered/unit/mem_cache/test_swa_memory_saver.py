import unittest

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SuffixKVPool


class _RecordingKVPool:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def get_kv_size_bytes(self):
        return 0, 0


class TestSWAMemorySaver(unittest.TestCase):
    def setUp(self):
        _RecordingKVPool.instances.clear()

    def test_swa_kv_pool_passes_memory_saver_to_inner_pools(self):
        SWAKVPool(
            size=8,
            size_swa=4,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=8,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=_RecordingKVPool,
            enable_memory_saver=True,
        )

        self.assertEqual(len(_RecordingKVPool.instances), 2)
        self.assertTrue(
            _RecordingKVPool.instances[0].kwargs["enable_memory_saver"]
        )
        self.assertTrue(
            _RecordingKVPool.instances[1].kwargs["enable_memory_saver"]
        )

    def test_swa_kv_pool_defaults_memory_saver_to_false(self):
        SWAKVPool(
            size=8,
            size_swa=4,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=8,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=_RecordingKVPool,
        )

        self.assertEqual(len(_RecordingKVPool.instances), 2)
        self.assertFalse(
            _RecordingKVPool.instances[0].kwargs["enable_memory_saver"]
        )
        self.assertFalse(
            _RecordingKVPool.instances[1].kwargs["enable_memory_saver"]
        )

    def test_suffix_kv_pool_passes_memory_saver_to_inner_pools(self):
        SuffixKVPool(
            size=8,
            size_suffix=4,
            size_swa=2,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=8,
            suffix_attention_layer_ids=[2],
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0, 3],
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=_RecordingKVPool,
            enable_memory_saver=True,
        )

        self.assertEqual(len(_RecordingKVPool.instances), 3)
        for inner_pool in _RecordingKVPool.instances:
            self.assertTrue(inner_pool.kwargs["enable_memory_saver"])


if __name__ == "__main__":
    unittest.main()
