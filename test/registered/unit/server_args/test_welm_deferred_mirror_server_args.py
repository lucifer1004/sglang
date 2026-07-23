import pytest

from sglang.srt.server_args import prepare_server_args


def test_welm_mirror_pd_mode_defaults_to_legacy():
    args = prepare_server_args(["--model-path", "dummy"])

    assert args.welm_kv_mirror_pd_mode == "legacy"


def test_enable_kv_mirror_deferred_selects_deferred_last_prompt_mode():
    args = prepare_server_args(
        [
            "--model-path",
            "dummy",
            "--enable-kv-mirror-deferred",
        ]
    )

    assert args.welm_kv_mirror_pd_mode == "deferred-last-prompt"


def test_removed_welm_mirror_pd_mode_argument_is_rejected():
    with pytest.raises(SystemExit):
        prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--welm-kv-mirror-pd-mode",
                "deferred-last-prompt",
            ]
        )
