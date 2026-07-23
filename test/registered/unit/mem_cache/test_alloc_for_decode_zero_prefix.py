from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def test_paged_decode_uses_first_page_sentinel_for_zero_length_row():
    req_to_token = torch.tensor(
        [
            [0, 0, 0, 99],
            [4, 7, 0, 0],
        ],
        dtype=torch.int32,
    )
    req_to_token_pool = SimpleNamespace(
        req_to_token=req_to_token,
        write=MagicMock(),
    )
    tree_cache = SimpleNamespace(
        page_size=16,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=16),
    )
    batch = SimpleNamespace(
        maybe_evict_swa=lambda: None,
        seq_lens=torch.tensor([0, 2], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([0, 2], dtype=torch.int64),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        req_to_token_pool=req_to_token_pool,
        tree_cache=tree_cache,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        device="cpu",
    )
    captured = {}

    def fake_alloc_paged_token_slots_decode(**kwargs):
        captured["last_loc"] = kwargs["last_loc"].clone()
        return torch.tensor([10, 11], dtype=torch.int64)

    with patch(
        "sglang.srt.mem_cache.common.alloc_paged_token_slots_decode",
        side_effect=fake_alloc_paged_token_slots_decode,
    ):
        alloc_for_decode(batch, token_per_req=1)

    assert captured["last_loc"].tolist() == [-1, 7]
