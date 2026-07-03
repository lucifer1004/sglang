import math
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.mk_decode_attention_backend import (
    MkDecodeAttentionBackend,
)
from sglang.srt.configs.model_config import yarn_get_mscale
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_cuda_ci


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

register_cuda_ci(est_time=30, stage="stage-b", runner_config="1-gpu-large")


class _FakeKVPool:
    _SWA_PAGE_OFFSET = 32

    def __init__(
        self,
        layer_num: int,
        size: int,
        page_size: int,
        device: str,
        swa_layer_ids=None,
    ):
        self.page_size = page_size
        self.dtype = torch.bfloat16
        self.swa_layer_ids = set(swa_layer_ids or [])
        self.swa_translate_calls = 0
        allocated_size = size + (self._SWA_PAGE_OFFSET + 1) * page_size
        self.k = [
            torch.randn(
                allocated_size, 1, 256, device=device, dtype=torch.bfloat16
            )
            for _ in range(layer_num)
        ]
        self.v = [
            torch.randn(
                allocated_size, 1, 256, device=device, dtype=torch.bfloat16
            )
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id: int):
        return self.k[layer_id]

    def get_value_buffer(self, layer_id: int):
        return self.v[layer_id]

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args):
        if self.is_swa_layer(layer.layer_id):
            loc = self.translate_loc_from_full_to_swa(loc)
        self.k[layer.layer_id][loc] = cache_k.view(-1, 1, 256)
        self.v[layer.layer_id][loc] = cache_v.view(-1, 1, 256)

    def translate_loc_from_full_to_swa(self, loc):
        self.swa_translate_calls += 1
        return loc + self._SWA_PAGE_OFFSET * self.page_size

    def is_swa_layer(self, layer_id: int):
        return layer_id in self.swa_layer_ids


def _yarn_sm_scale(rope_scaling=None):
    scale = 1 / math.sqrt(256)
    if (
        isinstance(rope_scaling, dict)
        and rope_scaling.get("type") == "yarn"
        and rope_scaling.get("apply_softmax_scale", False)
        and rope_scaling.get("mscale_all_dim", False)
    ):
        mscale = yarn_get_mscale(
            float(rope_scaling["factor"]),
            float(rope_scaling["mscale_all_dim"]),
        )
        scale *= mscale * mscale
    return scale


def _make_backend(*, seq_lens_cpu, layerwise_windows, context_len=None, rope_scaling=None):
    page_size = 16
    bs = int(seq_lens_cpu.numel())
    max_seq = max(int(seq_lens_cpu.max().item()), 1)
    context_len = int(context_len or max_seq)
    max_pages = (max(max_seq, context_len) + page_size - 1) // page_size
    size = bs * max_pages * page_size
    layer_num = len(layerwise_windows)
    device = "cuda"
    req_to_token = torch.zeros(
        (bs + 1, max_pages * page_size), dtype=torch.int32, device=device
    )
    for row, seq_len in enumerate(seq_lens_cpu.tolist(), start=1):
        base = (row - 1) * max_pages * page_size
        req_to_token[row, :seq_len] = torch.arange(
            base, base + int(seq_len), dtype=torch.int32, device=device
        )

    swa_layer_ids = [
        layer_id
        for layer_id, window in enumerate(layerwise_windows)
        if window > 0 and window + 1 < context_len
    ]
    kv_pool = _FakeKVPool(layer_num, size, page_size, device, swa_layer_ids)
    runner = SimpleNamespace(
        device=device,
        server_args=SimpleNamespace(page_size=page_size),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=kv_pool,
        is_hybrid_swa=any(window > 0 for window in layerwise_windows),
        sliding_window_size=max([w for w in layerwise_windows if w > 0], default=None),
        tp_size=4,
        model_config=SimpleNamespace(
            num_attention_heads=24,
            head_dim=256,
            context_len=context_len,
            num_hidden_layers=layer_num,
            hf_config=SimpleNamespace(
                sliding_window_size_layerwise=layerwise_windows,
                rope_scaling=rope_scaling,
            ),
            get_num_kv_heads=lambda tp: max(1, 4 // tp),
        ),
    )
    backend = MkDecodeAttentionBackend(runner)
    req_pool_indices = torch.arange(1, bs + 1, dtype=torch.int64, device=device)
    out_cache_loc = torch.tensor(
        [
            (row * max_pages * page_size) + int(seq_len) - 1
            for row, seq_len in enumerate(seq_lens_cpu.tolist())
        ],
        dtype=torch.int64,
        device=device,
    )
    forward_batch = SimpleNamespace(
        batch_size=bs,
        seq_lens_cpu=seq_lens_cpu,
        req_pool_indices=req_pool_indices,
        forward_mode=ForwardMode.DECODE,
        token_to_kv_pool=kv_pool,
        out_cache_loc=out_cache_loc,
    )
    return backend, runner, forward_batch


def _make_layer(layer_id: int, window_left: int, *, scaling=None):
    return SimpleNamespace(
        layer_id=layer_id,
        sliding_window_size=window_left,
        tp_q_head_num=6,
        tp_k_head_num=1,
        qk_head_dim=256,
        v_head_dim=256,
        logit_cap=0.0,
        scaling=1 / math.sqrt(256) if scaling is None else float(scaling),
        k_scale=None,
        v_scale=None,
    )


def _reference_output(q, sinks, runner, forward_batch, layer, window_left: int):
    scale = float(layer.scaling)
    req_to_token = runner.req_to_token_pool.req_to_token
    kv_pool = runner.token_to_kv_pool
    ref = []
    for batch_idx, seq_len in enumerate(forward_batch.seq_lens_cpu.tolist()):
        seq_len = int(seq_len)
        begin = max(seq_len - int(window_left) - 1, 0) if window_left > 0 else 0
        slots = req_to_token[batch_idx + 1, begin:seq_len].long()
        if kv_pool.is_swa_layer(layer.layer_id):
            slots = kv_pool.translate_loc_from_full_to_swa(slots)
        kk = kv_pool.k[layer.layer_id][slots, 0, :].float()
        vv = kv_pool.v[layer.layer_id][slots, 0, :].float()
        qb = q[batch_idx].view(6, 256).float()
        scores = qb @ kk.T * scale
        weights = torch.softmax(
            torch.cat([scores, sinks.float().view(6, 1)], dim=1), dim=1
        )[:, : slots.numel()]
        ref.append((weights @ vv).to(torch.bfloat16))
    return torch.stack(ref).reshape(q.shape)


def test_mk_decode_attention_is_registered_decode_backend():
    assert "mk_decode_attention" in ATTENTION_BACKENDS
    assert "mk_decode_attention" in ATTENTION_BACKEND_CHOICES


@pytest.mark.parametrize(
    "layerwise_windows, layer_id, window_left, seq_lens",
    [
        ([-1], 0, -1, [17, 33, 64]),
        ([-1, 35], 1, 35, [50, 64]),
    ],
)
def test_mk_decode_attention_matches_reference(
    layerwise_windows, layer_id, window_left, seq_lens
):
    pytest.importorskip("mk")
    torch.manual_seed(0)
    seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
    backend, runner, forward_batch = _make_backend(
        seq_lens_cpu=seq_lens_cpu,
        layerwise_windows=layerwise_windows,
    )
    backend.init_forward_metadata(forward_batch)
    layer = _make_layer(layer_id, window_left)
    q = torch.randn(
        len(seq_lens), 6 * 256, device="cuda", dtype=torch.bfloat16
    )
    kv_pool = runner.token_to_kv_pool
    k = kv_pool.k[layer.layer_id][forward_batch.out_cache_loc].reshape(
        len(seq_lens), 256
    )
    v = kv_pool.v[layer.layer_id][forward_batch.out_cache_loc].reshape(
        len(seq_lens), 256
    )
    sinks = torch.zeros((6,), device="cuda", dtype=torch.bfloat16)

    out = backend.forward_decode(q, k, v, layer, forward_batch, sinks=sinks)
    ref = _reference_output(q, sinks, runner, forward_batch, layer, window_left)

    torch.testing.assert_close(out, ref, atol=8e-3, rtol=8e-3)


def test_mk_decode_attention_uses_yarn_softmax_scale():
    pytest.importorskip("mk")
    torch.manual_seed(11)
    rope_scaling = {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "apply_softmax_scale": True,
        "mscale_all_dim": 1.0,
    }
    seq_lens_cpu = torch.tensor([33, 48], dtype=torch.int32, device="cpu")
    backend, runner, forward_batch = _make_backend(
        seq_lens_cpu=seq_lens_cpu,
        layerwise_windows=[-1],
        rope_scaling=rope_scaling,
    )
    backend.init_forward_metadata(forward_batch)
    layer = _make_layer(0, -1, scaling=_yarn_sm_scale(rope_scaling))
    q = torch.randn(2, 6 * 256, device="cuda", dtype=torch.bfloat16)
    kv_pool = runner.token_to_kv_pool
    k = kv_pool.k[layer.layer_id][forward_batch.out_cache_loc].reshape(2, 256)
    v = kv_pool.v[layer.layer_id][forward_batch.out_cache_loc].reshape(2, 256)
    sinks = torch.zeros((6,), device="cuda", dtype=torch.bfloat16)

    out = backend.forward_decode(q, k, v, layer, forward_batch, sinks=sinks)
    ref = _reference_output(q, sinks, runner, forward_batch, layer, -1)

    torch.testing.assert_close(out, ref, atol=8e-3, rtol=8e-3)


def test_mk_decode_attention_idle_metadata_clears_stale_state():
    seq_lens_cpu = torch.tensor([17], dtype=torch.int32, device="cpu")
    backend, _runner, forward_batch = _make_backend(
        seq_lens_cpu=seq_lens_cpu,
        layerwise_windows=[-1],
    )
    backend.init_forward_metadata(forward_batch)
    assert backend.forward_metadata is not None

    forward_batch.forward_mode = ForwardMode.IDLE
    backend.init_forward_metadata(forward_batch)

    assert backend.forward_metadata is None


def test_mk_decode_attention_eager_init_does_not_reuse_incompatible_workspace():
    pytest.importorskip("mk")
    seq_lens_cpu = torch.tensor([32, 32], dtype=torch.int32, device="cpu")
    backend, _runner, forward_batch = _make_backend(
        seq_lens_cpu=seq_lens_cpu,
        layerwise_windows=[-1],
    )
    backend.init_forward_metadata(forward_batch)
    first_workspace = backend.forward_metadata.workspace_by_window_left[-1]

    forward_batch.seq_lens_cpu = torch.tensor([17, 32], dtype=torch.int32, device="cpu")
    backend.init_forward_metadata(forward_batch)

    assert backend.forward_metadata.workspace_by_window_left[-1] is not first_workspace


def test_mk_decode_attention_cuda_graph_replay_matches_eager():
    pytest.importorskip("mk")
    torch.manual_seed(1)
    seq_lens_cpu = torch.tensor([17, 32], dtype=torch.int32, device="cpu")
    backend, runner, forward_batch = _make_backend(
        seq_lens_cpu=seq_lens_cpu,
        layerwise_windows=[-1],
        context_len=128,
    )
    layer = _make_layer(0, -1)
    q = torch.randn(2, 6 * 256, device="cuda", dtype=torch.bfloat16)
    kv_pool = runner.token_to_kv_pool
    k = kv_pool.k[layer.layer_id][forward_batch.out_cache_loc].reshape(2, 256)
    v = kv_pool.v[layer.layer_id][forward_batch.out_cache_loc].reshape(2, 256)
    sinks = torch.zeros((6,), device="cuda", dtype=torch.bfloat16)

    backend._cuda_graph_seq_len_fill_value = backend.get_cuda_graph_seq_len_fill_value()
    backend.init_forward_metadata_capture_cuda_graph(
        2,
        2,
        forward_batch.req_pool_indices,
        torch.empty(2, dtype=torch.int32, device="cuda"),
        None,
        ForwardMode.DECODE,
        None,
    )
    backend._cuda_graph_seq_len_fill_value = None
    for _ in range(3):
        backend.forward_decode(q, k, v, layer, forward_batch, sinks=sinks)
    torch.cuda.synchronize()
    backend.on_after_cuda_graph_warmup()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = backend.forward_decode(q, k, v, layer, forward_batch, sinks=sinks)
    torch.cuda.synchronize()

    q.copy_(torch.randn_like(q))
    backend.init_forward_metadata(forward_batch)
    eager_out = backend.forward_decode(q, k, v, layer, forward_batch, sinks=sinks).clone()

    backend._cuda_graph_seq_len_fill_value = backend.get_cuda_graph_seq_len_fill_value()
    try:
        backend.init_forward_metadata_replay_cuda_graph(
            2,
            forward_batch.req_pool_indices,
            torch.empty(2, dtype=torch.int32, device="cuda"),
            int(seq_lens_cpu.sum().item()),
            None,
            ForwardMode.DECODE,
            None,
            seq_lens_cpu=seq_lens_cpu,
        )
    finally:
        backend._cuda_graph_seq_len_fill_value = None
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)
    assert backend.forward_metadata.page_ids_by_window_left[-1].shape[1] == 8


def test_mk_decode_attention_cuda_graph_workspace_buffer_is_persistent(monkeypatch):
    pytest.importorskip("mk")
    monkeypatch.setenv("SGLANG_MK_DECODE_CUDA_GRAPH_WORKSPACE_BYTES", "1m")
    seq_lens_cpu = torch.tensor([17, 32], dtype=torch.int32, device="cpu")
    backend, _runner, forward_batch = _make_backend(
        seq_lens_cpu=seq_lens_cpu,
        layerwise_windows=[-1],
        context_len=128,
    )

    backend._cuda_graph_seq_len_fill_value = backend.get_cuda_graph_seq_len_fill_value()
    backend.init_cuda_graph_state(2, 2)
    backend.init_forward_metadata_capture_cuda_graph(
        2,
        2,
        forward_batch.req_pool_indices,
        torch.empty(2, dtype=torch.int32, device="cuda"),
        None,
        ForwardMode.DECODE,
        None,
    )
    backend._cuda_graph_seq_len_fill_value = None

    buffer = backend._cuda_graph_workspace_buffer
    assert buffer is not None
    assert buffer.numel() == 1024 * 1024
    workspace = backend.forward_metadata.workspace_by_window_left[-1]
    buffer_begin = int(buffer.data_ptr())
    buffer_end = buffer_begin + int(buffer.numel())
    capture_workspace_ptr = int(workspace.workspace.data_ptr())
    assert buffer_begin <= capture_workspace_ptr < buffer_end

    backend.on_after_cuda_graph_warmup()
    warmup_workspace = backend.forward_metadata.workspace_by_window_left[-1]
    warmup_workspace_ptr = int(warmup_workspace.workspace.data_ptr())
    assert warmup_workspace_ptr == capture_workspace_ptr

    backend._cuda_graph_seq_len_fill_value = backend.get_cuda_graph_seq_len_fill_value()
    try:
        backend.init_forward_metadata_replay_cuda_graph(
            2,
            forward_batch.req_pool_indices,
            torch.empty(2, dtype=torch.int32, device="cuda"),
            int(seq_lens_cpu.sum().item()),
            None,
            ForwardMode.DECODE,
            None,
            seq_lens_cpu=seq_lens_cpu,
        )
    finally:
        backend._cuda_graph_seq_len_fill_value = None

    replay_workspace = backend.forward_metadata.workspace_by_window_left[-1]
    assert int(replay_workspace.workspace.data_ptr()) == capture_workspace_ptr


def test_mk_decode_attention_cuda_graph_frozen_resize_keeps_pointer(monkeypatch):
    pytest.importorskip("mk")
    monkeypatch.setenv("SGLANG_MK_DECODE_CUDA_GRAPH_WORKSPACE_BYTES", "1m")
    seq_lens_cpu = torch.tensor([17, 32], dtype=torch.int32, device="cpu")
    backend, _runner, forward_batch = _make_backend(
        seq_lens_cpu=seq_lens_cpu,
        layerwise_windows=[-1],
        context_len=128,
    )

    backend._cuda_graph_seq_len_fill_value = backend.get_cuda_graph_seq_len_fill_value()
    backend.init_cuda_graph_state(2, 2)
    backend.init_forward_metadata_capture_cuda_graph(
        2,
        2,
        forward_batch.req_pool_indices,
        torch.empty(2, dtype=torch.int32, device="cuda"),
        None,
        ForwardMode.DECODE,
        None,
    )
    backend._cuda_graph_seq_len_fill_value = None
    backend.on_after_cuda_graph_warmup()

    workspace = backend.forward_metadata.workspace_by_window_left[-1]
    capture_workspace_ptr = int(workspace.workspace.data_ptr())

    import mk.kernels.decode_attention_welmv45 as welmv45

    real_init_workspace = welmv45.decode_attention_welmv45_init_workspace

    def resize_if_allowed(*args, **kwargs):
        existing_workspace = kwargs.get("workspace")
        if existing_workspace is None:
            return real_init_workspace(*args, **kwargs)
        if kwargs.get("allow_resize") is False:
            raise RuntimeError("blocked frozen workspace resize")
        existing_workspace.workspace_size += 256
        existing_workspace.workspace = torch.empty(
            (existing_workspace.workspace.numel() + 256,),
            dtype=torch.uint8,
            device="cuda",
        )
        return existing_workspace

    monkeypatch.setattr(
        welmv45,
        "decode_attention_welmv45_init_workspace",
        resize_if_allowed,
    )

    backend._cuda_graph_seq_len_fill_value = backend.get_cuda_graph_seq_len_fill_value()
    try:
        with pytest.raises(RuntimeError, match="blocked frozen workspace resize"):
            backend.init_forward_metadata_replay_cuda_graph(
                2,
                forward_batch.req_pool_indices,
                torch.empty(2, dtype=torch.int32, device="cuda"),
                int(seq_lens_cpu.sum().item()),
                None,
                ForwardMode.DECODE,
                None,
                seq_lens_cpu=seq_lens_cpu,
            )
    finally:
        backend._cuda_graph_seq_len_fill_value = None

    assert int(workspace.workspace.data_ptr()) == capture_workspace_ptr


def test_mk_decode_attention_cuda_graph_workspace_capacity_exceeded(monkeypatch):
    pytest.importorskip("mk")
    monkeypatch.setenv("SGLANG_MK_DECODE_CUDA_GRAPH_WORKSPACE_BYTES", "1024")
    seq_lens_cpu = torch.tensor([17, 32], dtype=torch.int32, device="cpu")
    backend, _runner, forward_batch = _make_backend(
        seq_lens_cpu=seq_lens_cpu,
        layerwise_windows=[-1],
        context_len=128,
    )

    backend._cuda_graph_seq_len_fill_value = backend.get_cuda_graph_seq_len_fill_value()
    with pytest.raises(
        RuntimeError, match="SGLANG_MK_DECODE_CUDA_GRAPH_WORKSPACE_BYTES"
    ):
        backend.init_forward_metadata_capture_cuda_graph(
            2,
            2,
            forward_batch.req_pool_indices,
            torch.empty(2, dtype=torch.int32, device="cuda"),
            None,
            ForwardMode.DECODE,
            None,
        )
    backend._cuda_graph_seq_len_fill_value = None


def test_mk_decode_attention_only_translates_actual_swa_layers():
    pytest.importorskip("mk")
    seq_lens_cpu = torch.tensor([32], dtype=torch.int32, device="cpu")
    backend, runner, forward_batch = _make_backend(
        seq_lens_cpu=seq_lens_cpu,
        layerwise_windows=[63, 31],
        context_len=64,
    )

    backend.init_forward_metadata(forward_batch)

    assert sorted(backend.forward_metadata.page_ids_by_window_left) == [-1, 31]
    assert runner.token_to_kv_pool.swa_translate_calls == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
