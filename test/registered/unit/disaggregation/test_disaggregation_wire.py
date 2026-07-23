import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.common.conn import (
    PrefillServerInfo,
    prefill_server_info_to_wire,
)
from sglang.srt.disaggregation.common.welm_deferred_protocol import (
    WelmDeferredCompletion,
    build_welm_deferred_mirror_capability,
)
from sglang.srt.disaggregation.common.utils import (
    pack_int_lists,
    pack_list_of_buffers,
    unpack_int_lists,
    unpack_list_of_buffers,
)
from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.models.welm_deferred_mirror import WelmDeferredMirrorPlan
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestDisaggregationWire(unittest.TestCase):
    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)

    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)

    def test_legacy_prefill_info_wire_has_exact_original_fields(self):
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

        self.assertEqual(
            prefill_server_info_to_wire(info),
            {
                "attn_tp_size": 2,
                "attn_cp_size": 4,
                "dp_size": 1,
                "pp_size": 1,
                "page_size": 16,
                "kv_cache_dtype": "auto",
                "follow_bootstrap_room": True,
                "all_cp_ranks_transfer": True,
                "target_tp_rank": None,
                "target_tp_ranks": None,
                "target_cp_ranks": None,
                "target_pp_ranks": None,
                "required_dst_info_num": None,
                "required_prefill_response_num": None,
            },
        )

    def test_deferred_prefill_info_roundtrips_nested_capability(self):
        capability = build_welm_deferred_mirror_capability(
            model_identity="/models/welm",
            plan=WelmDeferredMirrorPlan(
                num_hidden_layers=48,
                execution_end_layer=33,
                pairs=(),
                fingerprint="fingerprint",
            ),
        )
        info = PrefillServerInfo(
            attn_tp_size=2,
            attn_cp_size=4,
            dp_size=1,
            pp_size=1,
            page_size=16,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
            welm_deferred_mirror=capability,
        )

        wire = prefill_server_info_to_wire(info)

        self.assertEqual(wire["welm_deferred_mirror"], capability.to_wire())
        self.assertEqual(PrefillServerInfo(**wire).welm_deferred_mirror, capability)

    def test_deferred_completion_uses_dedicated_optional_aux_buffer(self):
        buffers = MetadataBuffers(
            size=2,
            hidden_size=16,
            hidden_states_dtype=torch.float32,
        )

        legacy_infos = buffers.get_buf_infos()
        deferred_infos = buffers.get_buf_infos(
            include_welm_deferred_completion=True
        )

        self.assertEqual(len(legacy_infos[0]), 10)
        self.assertEqual(len(deferred_infos[0]), 11)
        self.assertEqual(deferred_infos[2][-1], 16 * np.dtype(np.int32).itemsize)
        completion = WelmDeferredCompletion(
            committed_kv_len=7,
            seed_position=7,
            seed_token_id=123,
        )
        buffers.set_welm_deferred_completion(1, completion)
        self.assertEqual(buffers.get_welm_deferred_completion(1), completion)
        self.assertEqual(len(buffers.get_buf_infos()[0]), 10)

    def test_cached_token_stats_roundtrip_independently_of_output_metadata(self):
        buffers = MetadataBuffers(
            size=2,
            hidden_size=16,
            hidden_states_dtype=torch.float32,
        )
        req = SimpleNamespace(
            metadata_buffer_index=1,
            cached_tokens=31,
            cached_tokens_device=17,
            cached_tokens_host=9,
            cached_tokens_storage=5,
        )

        buffers.set_cached_token_stats(req)

        self.assertEqual(
            buffers.get_cached_token_stats(1)[:4].tolist(),
            [31, 17, 9, 5],
        )


if __name__ == "__main__":
    unittest.main()
