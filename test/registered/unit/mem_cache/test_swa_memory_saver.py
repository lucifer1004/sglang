import unittest

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SuffixKVPool


class _RecordingKVPool:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.__class__.instances.append(self)

    def get_kv_size_bytes(self):
        return 0, 0

    def set_kv_buffer(
        self,
        layer,
        loc,
        cache_k,
        cache_v,
        k_scale=1.0,
        v_scale=1.0,
        layer_id_override=None,
    ):
        self.calls.append((loc.clone(), layer_id_override))


class _Layer:
    layer_id = 1


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

    def test_swa_kv_pool_ignores_stale_precomputed_loc_for_short_custom_loc(self):
        pool = SWAKVPool(
            size=8,
            size_swa=8,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=8,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=_RecordingKVPool,
        )
        pool.register_mapping(torch.arange(16, dtype=torch.int32) + 100)
        pool.set_swa_loc(torch.tensor([10, 11, 12, 13], dtype=torch.int32))

        loc = torch.tensor([2, 3], dtype=torch.int64)
        pool.set_kv_buffer(_Layer(), loc, torch.empty(0), torch.empty(0))

        inner_swa_pool = _RecordingKVPool.instances[0]
        self.assertEqual(len(inner_swa_pool.calls), 1)
        actual_loc, layer_id_override = inner_swa_pool.calls[0]
        self.assertEqual(layer_id_override, 0)
        torch.testing.assert_close(
            actual_loc, torch.tensor([102, 103], dtype=torch.int32)
        )

    def test_swa_kv_pool_uses_precomputed_loc_when_lengths_match(self):
        pool = SWAKVPool(
            size=8,
            size_swa=8,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=8,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=_RecordingKVPool,
        )
        precomputed_loc = torch.tensor([10, 11], dtype=torch.int32)
        pool.register_mapping(torch.arange(16, dtype=torch.int32) + 100)
        pool.set_swa_loc(precomputed_loc)

        pool.set_kv_buffer(
            _Layer(),
            torch.tensor([2, 3], dtype=torch.int64),
            torch.empty(0),
            torch.empty(0),
        )

        inner_swa_pool = _RecordingKVPool.instances[0]
        self.assertEqual(len(inner_swa_pool.calls), 1)
        actual_loc, layer_id_override = inner_swa_pool.calls[0]
        self.assertEqual(layer_id_override, 0)
        torch.testing.assert_close(actual_loc, precomputed_loc)


if __name__ == "__main__":
    unittest.main()
