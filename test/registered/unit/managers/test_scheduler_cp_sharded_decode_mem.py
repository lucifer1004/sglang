import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestCPShardedDecodeMemoryCheck(unittest.TestCase):
    def test_uses_replicated_owner_ledger_without_collective(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_size = 4
        scheduler.tp_cpu_group = object()
        scheduler._use_cp_sharded_decode_mem_check = MagicMock(return_value=True)

        demand_by_start_and_rotation = {
            (15, 1): (1, 0),
            (31, 0): (0, 2),
        }
        allocator = SimpleNamespace(
            cp_size=2,
            alloc_size_per_rank_for_range=MagicMock(
                side_effect=lambda start, _length, rotation: (
                    demand_by_start_and_rotation[(start, rotation)]
                )
            ),
        )
        tree_cache = object()
        batch = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            reqs=[
                SimpleNamespace(kv_committed_len=15, attn_cp_owner_rotation=1),
                SimpleNamespace(kv_committed_len=31, attn_cp_owner_rotation=0),
            ],
            tree_cache=tree_cache,
            _get_scale_seq_factor=MagicMock(return_value=1),
        )

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "all_reduce") as all_reduce,
            patch(
                "sglang.srt.managers.scheduler.ensure_cp_sharded_kv_capacity",
                return_value=True,
            ) as ensure_capacity,
        ):
            result = Scheduler._check_decode_mem(scheduler, batch)

        self.assertTrue(result)
        all_reduce.assert_not_called()
        ensure_capacity.assert_called_once_with(
            allocator=allocator,
            tree_cache=tree_cache,
            demand_tokens=(1, 2),
        )
        self.assertEqual(
            allocator.alloc_size_per_rank_for_range.call_args_list,
            [
                call(15, 1, 1),
                call(31, 1, 0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
