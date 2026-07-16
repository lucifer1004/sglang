from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.tp_worker import _phase2_no_q_prefill_token_ids
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def _forward_batch(*, has_global_q: bool, batch_size: int = 1):
    return SimpleNamespace(
        attn_cp_prefill_runtime_layout=object(),
        welm_cp_prefill_kv_mirror_has_global_q=has_global_q,
        batch_size=batch_size,
        input_ids=torch.arange(4),
    )


def test_phase2_no_q_prefill_builds_batch_sized_overlap_placeholder():
    token_ids = _phase2_no_q_prefill_token_ids(
        SimpleNamespace(next_token_logits=torch.empty((0, 32))),
        _forward_batch(has_global_q=False, batch_size=1),
    )

    assert token_ids.shape == (1,)
    assert token_ids.dtype == torch.long
    assert token_ids.device.type == "cpu"
    assert token_ids.tolist() == [0]


def test_phase2_global_q_prefill_does_not_build_placeholder():
    token_ids = _phase2_no_q_prefill_token_ids(
        SimpleNamespace(next_token_logits=torch.empty((1, 32))),
        _forward_batch(has_global_q=True),
    )

    assert token_ids is None


def test_phase2_no_q_prefill_rejects_nonempty_logits():
    with pytest.raises(RuntimeError, match="no-Q prefill produced logits"):
        _phase2_no_q_prefill_token_ids(
            SimpleNamespace(next_token_logits=torch.empty((1, 32))),
            _forward_batch(has_global_q=False),
        )


def test_non_phase2_empty_logits_do_not_build_placeholder():
    token_ids = _phase2_no_q_prefill_token_ids(
        SimpleNamespace(next_token_logits=torch.empty((0, 32))),
        SimpleNamespace(
            attn_cp_prefill_runtime_layout=None,
            welm_cp_prefill_kv_mirror_has_global_q=False,
            batch_size=1,
            input_ids=torch.arange(4),
        ),
    )

    assert token_ids is None
