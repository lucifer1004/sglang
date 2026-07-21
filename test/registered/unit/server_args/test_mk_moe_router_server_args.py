from sglang.srt.server_args import prepare_server_args


def test_mk_moe_router_is_disabled_by_default():
    args = prepare_server_args(["--model-path", "dummy"])

    assert not args.enable_welm_v45_80a3_mk_moe_router


def test_mk_moe_router_flag_is_parsed():
    args = prepare_server_args(
        ["--model-path", "dummy", "--enable-welm-v45-80a3-mk-moe-router"]
    )

    assert args.enable_welm_v45_80a3_mk_moe_router
