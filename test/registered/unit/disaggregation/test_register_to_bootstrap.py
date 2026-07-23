"""Unit tests for srt/disaggregation/common/conn — register_to_bootstrap retry logic."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import sglang.srt.disaggregation.common.conn as common_conn
from sglang.srt.disaggregation.common.welm_deferred_protocol import (
    build_welm_deferred_mirror_capability,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.models.welm_deferred_mirror import WelmDeferredMirrorPlan
from sglang.test.test_utils import CustomTestCase


class TestRegisterToBootstrap(CustomTestCase):
    """Tests for CommonKVManager.register_to_bootstrap retry/backoff behavior."""

    def test_decode_sharded_kv_transfer_uses_global_tp_and_sharded_cp(self):
        with (
            patch.object(common_conn, "get_attention_tp_rank", return_value=1),
            patch.object(common_conn, "get_attention_tp_size", return_value=2),
            patch.object(common_conn, "get_attention_cp_rank", return_value=0),
            patch.object(common_conn, "get_attention_cp_size", return_value=2),
            patch.object(
                common_conn, "get_tensor_model_parallel_rank", return_value=3
            ),
            patch.object(
                common_conn,
                "get_tensor_model_parallel_world_size",
                return_value=4,
            ),
            patch.object(
                common_conn,
                "get_sharded_kv_context_model_parallel_rank",
                return_value=1,
            ),
            patch.object(
                common_conn,
                "get_sharded_kv_context_model_parallel_world_size",
                return_value=2,
            ),
        ):
            decode = common_conn._resolve_kv_transfer_parallel_info(
                DisaggregationMode.DECODE, is_cp_sharded_kv=True
            )
            prefill = common_conn._resolve_kv_transfer_parallel_info(
                DisaggregationMode.PREFILL, is_cp_sharded_kv=True
            )

        self.assertEqual(
            (decode.attn_tp_rank, decode.attn_tp_size),
            (3, 4),
        )
        self.assertEqual(
            (decode.attn_cp_rank, decode.attn_cp_size),
            (1, 2),
        )
        self.assertTrue(decode.attn_tp_rank_includes_cp)
        self.assertEqual(
            (prefill.attn_tp_rank, prefill.attn_tp_size),
            (1, 2),
        )
        self.assertEqual(
            (prefill.attn_cp_rank, prefill.attn_cp_size),
            (1, 2),
        )
        self.assertFalse(prefill.attn_tp_rank_includes_cp)

    def test_sharded_kv_routing_uses_runtime_attention_coordinates(self):
        from sglang.srt.disaggregation.common.conn import CommonKVManager

        mgr = object.__new__(CommonKVManager)
        mgr.attn_tp_size = 2
        mgr.attn_tp_rank = 1
        mgr.attn_cp_size = 4
        mgr.attn_cp_rank = 3

        mgr._init_attention_routing_topology()

        self.assertEqual(mgr.routing_attn_tp_size, 2)
        self.assertEqual(mgr.routing_attn_tp_rank, 1)
        self.assertEqual(mgr.routing_attn_cp_size, 4)
        self.assertEqual(mgr.routing_attn_cp_rank, 3)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_succeeds_on_first_attempt(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_put.return_value = mock_response

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        mock_put.assert_called_once()
        mock_time.sleep.assert_not_called()

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_succeeds_after_retries(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [fail_resp, fail_resp, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 3)
        self.assertEqual(mock_time.sleep.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_all_retries_exhausted(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 5)
        # Sleep is only called between attempts, not after the final failure
        self.assertEqual(mock_time.sleep.call_count, 4)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_exception_with_nested_cause(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0

        root_exc = ConnectionRefusedError("connection refused")
        inner_exc = OSError("os error")
        inner_exc.__cause__ = root_exc
        outer_exc = Exception("wrapped")
        outer_exc.__cause__ = inner_exc

        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [outer_exc, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_exception_with_no_cause(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0

        exc = ConnectionError("plain connection error")
        exc.__cause__ = None

        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [exc, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_backoff_delay_exponential(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        # With monotonic() = 0.0, jitter factor = 0.75 + 0.25 * (0.0 % 1) = 0.75
        # delay = min(1.0 * 2^attempt, 30.0) * 0.75
        # Sleep happens only between attempts (attempt 0..3), not after the final failure
        expected_calls = [call(0.75), call(1.5), call(3.0), call(6.0)]
        self.assertEqual(mock_time.sleep.call_args_list, expected_calls)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_jitter_never_exceeds_max_delay(self, mock_put, mock_time):
        """Guard against operator-precedence regressions in the jitter factor.

        The jitter factor must stay in [0.75, 1.0), so a delay capped at
        max_delay must never exceed max_delay after applying jitter.
        """
        # monotonic() returns a value whose fractional part is close to 1.
        # If the parentheses around `time.monotonic() % 1` were dropped, the
        # jitter factor could grow up to ~1.75 and blow past max_delay.
        mock_time.monotonic.return_value = 999.9999
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        max_delay = 30.0
        for sleep_call in mock_time.sleep.call_args_list:
            actual_delay = sleep_call[0][0]
            self.assertLess(actual_delay, max_delay)
            self.assertGreaterEqual(actual_delay, 0.75)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_payload_contains_required_fields(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        call_kwargs = mock_put.call_args
        payload = call_kwargs[1]["json"]
        required_fields = [
            "attn_tp_size",
            "attn_tp_rank",
            "attn_cp_size",
            "attn_cp_rank",
            "attn_dp_size",
            "attn_dp_rank",
            "pp_size",
            "pp_rank",
            "system_dp_size",
            "system_dp_rank",
            "rank_ip",
            "rank_port",
            "page_size",
            "kv_cache_dtype",
        ]
        for field in required_fields:
            self.assertIn(field, payload)

        self.assertNotIn("welm_deferred_mirror", payload)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_payload_includes_deferred_capability_only_when_enabled(
        self, mock_put, mock_time
    ):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp
        mgr = self._make_manager()
        mgr.welm_deferred_mirror_capability = self._make_capability()

        mgr.register_to_bootstrap()

        payload = mock_put.call_args[1]["json"]
        self.assertEqual(
            payload["welm_deferred_mirror"],
            mgr.welm_deferred_mirror_capability.to_wire(),
        )

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_capability_conflict_fails_without_retry_for_legacy_rank(
        self, mock_put, mock_time
    ):
        conflict = MagicMock()
        conflict.status_code = 409
        conflict.text = "Inconsistent WeLM deferred capability"
        mock_put.return_value = conflict
        mgr = self._make_manager()

        with self.assertRaisesRegex(RuntimeError, "capability conflict"):
            mgr.register_to_bootstrap()

        mock_put.assert_called_once()
        mock_time.sleep.assert_not_called()

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    def test_decode_rejects_capability_mismatch_before_rank_mapping(self, mock_get):
        mgr = object.__new__(common_conn.CommonKVManager)
        mgr.prefill_info_table = {}
        mgr.kv_args = MagicMock(page_size=16)
        mgr.server_args = MagicMock(kv_cache_dtype="auto", enable_hisparse=False)
        mgr.welm_deferred_mirror_capability = self._make_capability()
        mgr._resolve_rank_mapping = MagicMock()

        remote = self._make_capability().to_wire()
        remote["mirror_fingerprint"] = "different"
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "attn_tp_size": 1,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 16,
            "kv_cache_dtype": "auto",
            "follow_bootstrap_room": True,
            "welm_deferred_mirror": remote,
        }
        mock_get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "local=.*remote="):
            mgr.try_ensure_parallel_info("127.0.0.1:8998")

        mgr._resolve_rank_mapping.assert_not_called()
        self.assertEqual(mgr.prefill_info_table, {})

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    def test_decode_rejects_malformed_capability_before_rank_mapping(self, mock_get):
        mgr = object.__new__(common_conn.CommonKVManager)
        mgr.prefill_info_table = {}
        mgr.kv_args = MagicMock(page_size=16)
        mgr.server_args = MagicMock(kv_cache_dtype="auto", enable_hisparse=False)
        mgr.welm_deferred_mirror_capability = self._make_capability()
        mgr._resolve_rank_mapping = MagicMock()

        malformed = self._make_capability().to_wire()
        del malformed["protocol_version"]
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "attn_tp_size": 1,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 16,
            "kv_cache_dtype": "auto",
            "follow_bootstrap_room": True,
            "welm_deferred_mirror": malformed,
        }
        mock_get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "Invalid Prefill P/D capability"):
            mgr.try_ensure_parallel_info("127.0.0.1:8998")

        mgr._resolve_rank_mapping.assert_not_called()
        self.assertEqual(mgr.prefill_info_table, {})

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_payload_uses_sharded_kv_routing_topology(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.attn_tp_size = 2
        mgr.attn_tp_rank = 1
        mgr.routing_attn_tp_size = 2
        mgr.routing_attn_tp_rank = 1
        mgr.routing_attn_cp_size = 4
        mgr.routing_attn_cp_rank = 3

        mgr.register_to_bootstrap()

        payload = mock_put.call_args[1]["json"]
        self.assertEqual(payload["attn_tp_size"], 2)
        self.assertEqual(payload["attn_tp_rank"], 1)
        self.assertEqual(payload["attn_cp_size"], 4)
        self.assertEqual(payload["attn_cp_rank"], 3)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_payload_advertises_all_cp_rank_transfer(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.enable_all_cp_ranks_for_transfer = True

        mgr.register_to_bootstrap()

        payload = mock_put.call_args[1]["json"]
        self.assertIn("all_cp_ranks_transfer", payload)
        self.assertTrue(payload["all_cp_ranks_transfer"])

    def test_remote_sharded_kv_routes_decode_to_every_owner_rank(self):
        from sglang.srt.disaggregation.common.conn import (
            CommonKVManager,
            PrefillServerInfo,
        )

        mgr = MagicMock()
        mgr.attn_tp_size = 4
        mgr.attn_tp_rank = 0
        mgr.attn_cp_size = 1
        mgr.attn_cp_rank = 0
        mgr.enable_all_cp_ranks_for_transfer = False
        mgr.is_mla_backend = False
        mgr.pp_size = 1
        mgr.pp_rank = 0
        mgr.kv_args.engine_rank = 0

        info = PrefillServerInfo(
            attn_tp_size=2,
            attn_cp_size=2,
            dp_size=1,
            pp_size=1,
            page_size=16,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
            all_cp_ranks_transfer=True,
        )

        CommonKVManager._resolve_rank_mapping(mgr, info)

        self.assertEqual(info.target_cp_ranks, [0, 1])
        self.assertEqual(info.required_prefill_response_num, 2)

    def test_equal_decode_cp_routes_each_rank_to_every_prefill_owner(self):
        from sglang.srt.disaggregation.common.conn import (
            CommonKVManager,
            PrefillServerInfo,
        )

        for decode_cp_rank, engine_rank in enumerate((0, 2)):
            with self.subTest(decode_cp_rank=decode_cp_rank):
                mgr = MagicMock()
                mgr.attn_tp_size = 2
                mgr.attn_tp_rank = 0
                mgr.attn_cp_size = 2
                mgr.attn_cp_rank = decode_cp_rank
                mgr.enable_all_cp_ranks_for_transfer = True
                mgr.is_cp_sharded_kv = True
                mgr.is_mla_backend = False
                mgr.pp_size = 1
                mgr.pp_rank = 0
                mgr.kv_args.engine_rank = engine_rank

                info = PrefillServerInfo(
                    attn_tp_size=2,
                    attn_cp_size=2,
                    dp_size=1,
                    pp_size=1,
                    page_size=16,
                    kv_cache_dtype="auto",
                    follow_bootstrap_room=True,
                    all_cp_ranks_transfer=True,
                )

                CommonKVManager._resolve_rank_mapping(mgr, info)

                self.assertEqual(info.target_tp_ranks, [0])
                self.assertEqual(info.target_cp_ranks, [0, 1])
                self.assertEqual(info.required_dst_info_num, 2)
                self.assertEqual(info.required_prefill_response_num, 2)

    def test_tp_prefill_maps_one_replica_to_each_decode_cp_owner(self):
        from sglang.srt.disaggregation.common.conn import (
            CommonKVManager,
            PrefillServerInfo,
        )

        for engine_rank, (attn_tp_rank, cp_rank, prefill_tp_rank) in enumerate(
            ((0, 0, 0), (1, 0, 2), (0, 1, 1), (1, 1, 3))
        ):
            with self.subTest(engine_rank=engine_rank):
                mgr = MagicMock()
                mgr.attn_tp_size = 2
                mgr.attn_tp_rank = attn_tp_rank
                mgr.attn_cp_size = 2
                mgr.attn_cp_rank = cp_rank
                mgr.enable_all_cp_ranks_for_transfer = True
                mgr.is_cp_sharded_kv = True
                mgr.is_mla_backend = False
                mgr.pp_size = 1
                mgr.pp_rank = 0
                mgr.kv_args.engine_rank = attn_tp_rank
                mgr.kv_args.total_kv_head_num = 2
                info = PrefillServerInfo(
                    attn_tp_size=4,
                    attn_cp_size=1,
                    dp_size=1,
                    pp_size=1,
                    page_size=16,
                    kv_cache_dtype="auto",
                    follow_bootstrap_room=True,
                    all_cp_ranks_transfer=False,
                )

                CommonKVManager._resolve_rank_mapping(mgr, info)

                self.assertEqual(info.target_tp_rank, prefill_tp_rank)
                self.assertEqual(info.target_tp_ranks, [prefill_tp_rank])
                self.assertEqual(info.target_cp_ranks, [0])
                self.assertEqual(info.required_dst_info_num, 1)
                self.assertEqual(info.required_prefill_response_num, 1)

    def test_decode_cp_routes_from_every_prefill_cp_owner_in_same_lane(self):
        from sglang.srt.disaggregation.common.conn import (
            CommonKVManager,
            PrefillServerInfo,
        )

        mgr = MagicMock()
        mgr.attn_tp_size = 2
        mgr.attn_tp_rank = 0
        mgr.attn_cp_size = 2
        mgr.attn_cp_rank = 0
        mgr.enable_all_cp_ranks_for_transfer = True
        mgr.is_cp_sharded_kv = True
        mgr.is_mla_backend = False
        mgr.pp_size = 1
        mgr.pp_rank = 0
        mgr.kv_args.engine_rank = 3
        info = PrefillServerInfo(
            attn_tp_size=2,
            attn_cp_size=4,
            dp_size=1,
            pp_size=1,
            page_size=16,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
            all_cp_ranks_transfer=True,
        )

        CommonKVManager._resolve_rank_mapping(mgr, info)

        self.assertEqual(info.target_tp_ranks, [0])
        self.assertEqual(info.target_cp_ranks, [0, 1, 2, 3])
        self.assertEqual(info.required_dst_info_num, 2)
        self.assertEqual(info.required_prefill_response_num, 4)

    def test_decode_cp_does_not_collapse_distinct_prefill_kv_shards(self):
        from sglang.srt.disaggregation.common.conn import (
            CommonKVManager,
            PrefillServerInfo,
        )

        mgr = MagicMock()
        mgr.attn_tp_size = 2
        mgr.attn_tp_rank = 0
        mgr.attn_cp_size = 2
        mgr.attn_cp_rank = 0
        mgr.enable_all_cp_ranks_for_transfer = True
        mgr.is_cp_sharded_kv = True
        mgr.is_mla_backend = False
        mgr.pp_size = 1
        mgr.pp_rank = 0
        mgr.kv_args.engine_rank = 0
        mgr.kv_args.total_kv_head_num = 8
        info = PrefillServerInfo(
            attn_tp_size=4,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=16,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
            all_cp_ranks_transfer=False,
        )

        CommonKVManager._resolve_rank_mapping(mgr, info)

        self.assertEqual(info.target_tp_ranks, [0, 1])
        self.assertEqual(info.target_cp_ranks, [0])
        self.assertEqual(info.required_dst_info_num, 2)
        self.assertEqual(info.required_prefill_response_num, 2)

    def test_phase2_tp8_decode_maps_to_attntp2_cp4_prefill(self):
        from sglang.srt.disaggregation.common.conn import (
            CommonKVManager,
            PrefillServerInfo,
        )

        for decode_rank in range(8):
            with self.subTest(decode_rank=decode_rank):
                mgr = MagicMock()
                mgr.attn_tp_size = 8
                mgr.attn_tp_rank = decode_rank
                mgr.attn_cp_size = 1
                mgr.attn_cp_rank = 0
                mgr.enable_all_cp_ranks_for_transfer = False
                mgr.is_mla_backend = False
                mgr.pp_size = 1
                mgr.pp_rank = 0
                mgr.kv_args.engine_rank = decode_rank

                info = PrefillServerInfo(
                    attn_tp_size=2,
                    attn_cp_size=4,
                    dp_size=1,
                    pp_size=1,
                    page_size=16,
                    kv_cache_dtype="auto",
                    follow_bootstrap_room=True,
                    all_cp_ranks_transfer=True,
                )

                CommonKVManager._resolve_rank_mapping(mgr, info)

                expected_prefill_tp_rank = decode_rank // 4
                self.assertEqual(info.target_tp_rank, expected_prefill_tp_rank)
                self.assertEqual(
                    info.target_tp_ranks, [expected_prefill_tp_rank]
                )
                self.assertEqual(info.target_cp_ranks, [0, 1, 2, 3])
                self.assertEqual(info.required_dst_info_num, 4)
                self.assertEqual(info.required_prefill_response_num, 4)

    def test_phase2_tp4_decode_cp2_maps_to_attntp2_cp2_prefill(self):
        from sglang.srt.disaggregation.common.conn import (
            CommonKVManager,
            PrefillServerInfo,
        )

        for decode_rank in range(4):
            with self.subTest(decode_rank=decode_rank):
                mgr = MagicMock()
                mgr.attn_tp_size = 4
                mgr.attn_tp_rank = decode_rank
                mgr.attn_cp_size = 2
                mgr.attn_cp_rank = decode_rank % 2
                mgr.attn_tp_rank_includes_cp = True
                mgr.enable_all_cp_ranks_for_transfer = True
                mgr.is_cp_sharded_kv = True
                mgr.is_mla_backend = False
                mgr.pp_size = 1
                mgr.pp_rank = 0
                mgr.kv_args.engine_rank = decode_rank
                mgr.kv_args.total_kv_head_num = 2

                info = PrefillServerInfo(
                    attn_tp_size=2,
                    attn_cp_size=2,
                    dp_size=1,
                    pp_size=1,
                    page_size=16,
                    kv_cache_dtype="auto",
                    follow_bootstrap_room=True,
                    all_cp_ranks_transfer=True,
                )

                CommonKVManager._resolve_rank_mapping(mgr, info)

                expected_prefill_tp_rank = decode_rank // 2
                self.assertEqual(info.target_tp_rank, expected_prefill_tp_rank)
                self.assertEqual(
                    info.target_tp_ranks, [expected_prefill_tp_rank]
                )
                self.assertEqual(info.target_cp_ranks, [0, 1])
                # Decode global TP already contains the CP replica axis: each
                # Prefill (TP lane, CP owner) receives two Decode destinations,
                # not another TP-ratio x CP-size Cartesian product.
                self.assertEqual(info.required_dst_info_num, 2)
                self.assertEqual(info.required_prefill_response_num, 2)

    def test_deferred_capability_does_not_change_kv_rank_mapping(self):
        from sglang.srt.disaggregation.common.conn import (
            CommonKVManager,
            PrefillServerInfo,
        )

        capability = build_welm_deferred_mirror_capability(
            model_identity="/models/welm",
            plan=WelmDeferredMirrorPlan(
                num_hidden_layers=48,
                execution_end_layer=33,
                pairs=(),
                fingerprint="fingerprint",
            ),
        )

        def make_manager(deferred_capability):
            return SimpleNamespace(
                attn_tp_size=2,
                attn_tp_rank=0,
                attn_dp_size=4,
                attn_dp_rank=2,
                attn_cp_size=1,
                attn_cp_rank=0,
                attn_tp_rank_includes_cp=False,
                enable_all_cp_ranks_for_transfer=False,
                is_cp_sharded_kv=False,
                is_mla_backend=False,
                pp_size=1,
                pp_rank=0,
                kv_args=SimpleNamespace(
                    engine_rank=0,
                    total_kv_head_num=2,
                ),
                welm_deferred_mirror_capability=deferred_capability,
            )

        def make_prefill_info(deferred_capability):
            return PrefillServerInfo(
                attn_tp_size=8,
                attn_cp_size=1,
                dp_size=1,
                pp_size=1,
                page_size=16,
                kv_cache_dtype="auto",
                follow_bootstrap_room=True,
                all_cp_ranks_transfer=False,
                welm_deferred_mirror=deferred_capability,
            )

        generic_info = make_prefill_info(None)
        deferred_info = make_prefill_info(capability)
        CommonKVManager._resolve_rank_mapping(make_manager(None), generic_info)
        CommonKVManager._resolve_rank_mapping(
            make_manager(capability), deferred_info
        )

        generic_mapping = (
            generic_info.target_tp_rank,
            generic_info.target_tp_ranks,
            generic_info.target_cp_ranks,
            generic_info.target_pp_ranks,
            generic_info.required_dst_info_num,
            generic_info.required_prefill_response_num,
        )
        deferred_mapping = (
            deferred_info.target_tp_rank,
            deferred_info.target_tp_ranks,
            deferred_info.target_cp_ranks,
            deferred_info.target_pp_ranks,
            deferred_info.required_dst_info_num,
            deferred_info.required_prefill_response_num,
        )
        self.assertEqual(deferred_mapping, generic_mapping)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_url_with_dist_init_addr(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager(dist_init_addr="10.0.0.1:12345")
        mgr.register_to_bootstrap()

        url_used = mock_put.call_args[0][0]
        self.assertIn("10.0.0.1", url_used)

    def _make_manager(self, dist_init_addr=None):
        """Create a lightweight mock manager that has the attributes needed
        by register_to_bootstrap, without going through CommonKVManager.__init__
        (which requires zmq, ServerArgs model resolution, etc.)."""
        from sglang.srt.disaggregation.common.conn import CommonKVManager

        mgr = MagicMock(spec=CommonKVManager)
        # Bind the real method to the mock
        mgr.register_to_bootstrap = CommonKVManager.register_to_bootstrap.__get__(
            mgr, CommonKVManager
        )

        # Set attributes that register_to_bootstrap reads
        mgr.dist_init_addr = dist_init_addr
        mgr.bootstrap_host = "127.0.0.1"
        mgr.bootstrap_port = 8765
        mgr.attn_tp_size = 1
        mgr.attn_tp_rank = 0
        mgr.attn_cp_size = 1
        mgr.attn_cp_rank = 0
        mgr.routing_attn_tp_size = 1
        mgr.routing_attn_tp_rank = 0
        mgr.routing_attn_cp_size = 1
        mgr.routing_attn_cp_rank = 0
        mgr.enable_all_cp_ranks_for_transfer = False
        mgr.attn_dp_size = 1
        mgr.attn_dp_rank = 0
        mgr.pp_size = 1
        mgr.pp_rank = 0
        mgr.system_dp_size = 1
        mgr.system_dp_rank = 0
        mgr.local_ip = "127.0.0.1"
        mgr.rank_port = 12345

        mgr.kv_args = MagicMock()
        mgr.kv_args.page_size = 16

        mgr.server_args = MagicMock()
        mgr.server_args.kv_cache_dtype = "auto"
        mgr.server_args.load_balance_method = "follow_bootstrap_room"
        mgr.welm_deferred_mirror_capability = None

        return mgr

    @staticmethod
    def _make_capability():
        return build_welm_deferred_mirror_capability(
            model_identity="/models/welm",
            plan=WelmDeferredMirrorPlan(
                num_hidden_layers=48,
                execution_end_layer=33,
                pairs=(),
                fingerprint="fingerprint",
            ),
        )


if __name__ == "__main__":
    unittest.main()
