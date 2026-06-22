from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers import schedule_policy
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder


def _make_prefill_adder(*, nsa_in_seq_cp: bool) -> PrefillAdder:
    tree_cache = SimpleNamespace(
        disable=True,
        supports_mamba=lambda: False,
    )
    running_batch = SimpleNamespace(reqs=[])

    with patch.object(
        schedule_policy,
        "is_nsa_prefill_cp_in_seq_split",
        return_value=nsa_in_seq_cp,
    ):
        return PrefillAdder(
            page_size=1,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=object(),
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=4096,
            rem_chunk_tokens=1024,
        )


def test_nsa_in_seq_cp_keeps_single_request_prefill_limit():
    adder = _make_prefill_adder(nsa_in_seq_cp=True)
    adder.can_run_list.append(object())
    req = SimpleNamespace(sampling_params=SimpleNamespace(ignore_eos=True))

    with patch.object(
        PrefillAdder, "add_one_req_ignore_eos", return_value=AddReqResult.CONTINUE
    ) as add_ignore_eos:
        assert adder.add_one_req(req, False, None) == AddReqResult.OTHER
        add_ignore_eos.assert_not_called()


def test_sharded_kv_cp_does_not_force_single_request_prefill():
    # Sharded-KV CP keeps full Q rows, so it must not collapse prefill to a
    # single request the way NSA in-seq-split CP does.
    adder = _make_prefill_adder(nsa_in_seq_cp=False)
    adder.can_run_list.append(object())
    req = SimpleNamespace(sampling_params=SimpleNamespace(ignore_eos=True))

    with patch.object(
        PrefillAdder, "add_one_req_ignore_eos", return_value=AddReqResult.CONTINUE
    ) as add_ignore_eos:
        assert adder.add_one_req(req, False, None) == AddReqResult.CONTINUE
        add_ignore_eos.assert_called_once_with(req)
