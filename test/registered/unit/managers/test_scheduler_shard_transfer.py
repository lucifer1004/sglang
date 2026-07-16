import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    CheckWeightsReqInput,
    InstallShardTransferPlanReqInput,
    UpdateWeightsFromShardTransferReqInput,
    UpdateWeightsFromShardTransferReqOutput,
)
from sglang.srt.managers.scheduler_update_weights_mixin import (  # noqa: E402
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.managers.tokenizer_control_mixin import (  # noqa: E402
    TokenizerControlMixin,
)
from sglang.srt.utils.common import SafeUnpickler  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestSchedulerShardTransfer(unittest.TestCase):
    def _scheduler(self):
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(model_runner=MagicMock()),
            tp_cpu_group=object(),
            tp_rank=0,
            flush_cache_after_weight_update=MagicMock(),
        )
        scheduler._get_draft_model_runner = MagicMock(return_value=None)
        return scheduler

    def test_check_weights_keeps_legacy_main_payload_by_default(self):
        scheduler = self._scheduler()
        main_payload = {
            "checksums": {"weight": "abc"},
            "parallelism_info": {"tp_rank": 0},
        }
        scheduler.tp_worker.model_runner.check_weights.return_value = main_payload
        draft = MagicMock()
        scheduler._get_draft_model_runner.return_value = draft

        output = SchedulerUpdateWeightsMixin.check_weights(
            scheduler, CheckWeightsReqInput(action="checksum")
        )

        self.assertTrue(output.success)
        self.assertIs(output.payload, main_payload)
        draft.check_weights.assert_not_called()

    def test_check_weights_includes_draft_only_when_requested(self):
        scheduler = self._scheduler()
        main_payload = {"checksums": {"main": "abc"}}
        draft_payload = {"checksums": {"draft": "def"}}
        scheduler.tp_worker.model_runner.check_weights.return_value = main_payload
        draft = MagicMock()
        draft.check_weights.return_value = draft_payload
        scheduler._get_draft_model_runner.return_value = draft

        output = SchedulerUpdateWeightsMixin.check_weights(
            scheduler,
            CheckWeightsReqInput(action="checksum", include_draft=True),
        )

        self.assertTrue(output.success)
        self.assertEqual(
            output.payload,
            {"main": main_payload, "draft": draft_payload},
        )

    def test_report_treats_backend_payload_as_opaque(self):
        scheduler = self._scheduler()
        backend = MagicMock()
        backend.report_targets.return_value = "shard-transfer-pickle-v1:targets"

        with patch(
            "sglang.srt.managers.shard_transfer_backend.get_shard_transfer_backend",
            return_value=backend,
        ):
            output = SchedulerUpdateWeightsMixin.report_shard_transfer_target(
                scheduler, SimpleNamespace()
            )

        self.assertTrue(output.success)
        self.assertEqual(
            output.serialized_targets,
            "shard-transfer-pickle-v1:targets",
        )
        backend.report_targets.assert_called_once_with(
            {"main": scheduler.tp_worker.model_runner}
        )

    def test_global_safe_unpickler_has_no_backend_specific_prefixes(self):
        self.assertFalse(
            any(
                prefix.startswith("slime.")
                for prefix in SafeUnpickler.ALLOWED_MODULE_PREFIXES
            )
        )

    @staticmethod
    def _inject_remote(remote):
        def gather(results, local, group):
            results[:] = [local, remote]

        return gather

    def test_install_propagates_remote_rank_failure(self):
        scheduler = self._scheduler()
        backend = MagicMock()
        backend.install_plan.return_value = "installed local plan"
        remote = {"success": False, "message": "rank-local install failed"}

        with (
            patch(
                "sglang.srt.managers.shard_transfer_backend.get_shard_transfer_backend",
                return_value=backend,
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=self._inject_remote(remote),
            ) as gather,
        ):
            output = SchedulerUpdateWeightsMixin.install_shard_transfer_plan(
                scheduler,
                InstallShardTransferPlanReqInput(
                    serialized_plan="shard-transfer-pickle-v1:plan"
                ),
            )

        self.assertFalse(output.success)
        self.assertIn("rank-local install failed", output.message)
        backend.install_plan.assert_called_once_with("shard-transfer-pickle-v1:plan", 0)
        gather.assert_called_once()

    def test_update_propagates_remote_rank_failure(self):
        scheduler = self._scheduler()
        backend = MagicMock()
        backend.update_weights.return_value = {
            "entries": 2,
            "bytes": 20,
            "seconds": 1.0,
        }
        remote = {
            "success": False,
            "message": "rank-local pull failed",
            "per_target": {},
        }

        with (
            patch(
                "sglang.srt.managers.shard_transfer_backend.get_shard_transfer_backend",
                return_value=backend,
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=self._inject_remote(remote),
            ) as gather,
            patch("torch.distributed.barrier") as barrier,
        ):
            output = SchedulerUpdateWeightsMixin.update_weights_from_shard_transfer(
                scheduler,
                UpdateWeightsFromShardTransferReqInput(
                    serialized_source="shard-transfer-pickle-v1:source",
                    flush_cache=False,
                ),
            )

        self.assertFalse(output.success)
        self.assertIn("rank-local pull failed", output.message)
        backend.update_weights.assert_called_once_with(
            "main",
            scheduler.tp_worker.model_runner,
            "shard-transfer-pickle-v1:source",
        )
        gather.assert_called_once()
        barrier.assert_not_called()

    def test_update_aggregates_stats_across_tp_ranks(self):
        scheduler = self._scheduler()
        backend = MagicMock()
        backend.update_weights.return_value = {
            "entries": 2,
            "bytes": 20,
            "seconds": 1.0,
        }
        remote = {
            "success": True,
            "message": "Success.",
            "per_target": {"main": {"entries": 3, "bytes": 30, "seconds": 2.0}},
        }

        with (
            patch(
                "sglang.srt.managers.shard_transfer_backend.get_shard_transfer_backend",
                return_value=backend,
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=self._inject_remote(remote),
            ) as gather,
            patch("torch.distributed.barrier") as barrier,
        ):
            output = SchedulerUpdateWeightsMixin.update_weights_from_shard_transfer(
                scheduler,
                UpdateWeightsFromShardTransferReqInput(
                    serialized_source="shard-transfer-pickle-v1:source",
                    flush_cache=False,
                ),
            )

        self.assertTrue(output.success)
        self.assertEqual(output.entries, 5)
        self.assertEqual(output.bytes, 50)
        self.assertEqual(output.seconds, 2.0)
        self.assertEqual(
            output.per_target,
            {"main": {"entries": 5, "bytes": 50, "seconds": 2.0}},
        )
        gather.assert_called_once()
        barrier.assert_not_called()

    def test_update_propagates_remote_flush_failure(self):
        scheduler = self._scheduler()
        backend = MagicMock()
        backend.update_weights.return_value = {
            "entries": 2,
            "bytes": 20,
            "seconds": 1.0,
        }
        remote = {
            "success": False,
            "message": "flush: rank-local flush failed",
            "per_target": {},
        }

        with (
            patch(
                "sglang.srt.managers.shard_transfer_backend.get_shard_transfer_backend",
                return_value=backend,
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=self._inject_remote(remote),
            ) as gather,
        ):
            output = SchedulerUpdateWeightsMixin.update_weights_from_shard_transfer(
                scheduler,
                UpdateWeightsFromShardTransferReqInput(
                    serialized_source="shard-transfer-pickle-v1:source",
                    flush_cache=True,
                ),
            )

        self.assertFalse(output.success)
        self.assertIn("rank 1", output.message)
        self.assertIn("rank-local flush failed", output.message)
        gather.assert_called_once()

    def test_tokenizer_commits_version_only_after_successful_update(self):
        class AsyncLock:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        output = UpdateWeightsFromShardTransferReqOutput(
            success=True,
            message="Success.",
            entries=2,
            bytes=20,
            seconds=1.0,
            per_target={"main": {"entries": 2, "bytes": 20, "seconds": 1.0}},
        )
        manager = SimpleNamespace(
            auto_create_handle_loop=MagicMock(),
            model_update_lock=SimpleNamespace(writer_lock=AsyncLock()),
            update_weights_from_shard_transfer_communicator=AsyncMock(
                return_value=[output]
            ),
            _update_weight_version_if_provided=MagicMock(),
        )
        request = UpdateWeightsFromShardTransferReqInput(
            serialized_source="shard-transfer-pickle-v1:source",
            weight_version="shard-transfer-7",
        )

        success, message, stats = asyncio.run(
            TokenizerControlMixin.update_weights_from_shard_transfer(
                manager,
                request,
            )
        )

        self.assertTrue(success)
        self.assertIn("shard-transfer-7", message)
        self.assertEqual(stats["entries"], 2)
        manager._update_weight_version_if_provided.assert_called_once_with(
            "shard-transfer-7"
        )


if __name__ == "__main__":
    unittest.main()
