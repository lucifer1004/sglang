"""
End-to-end test for commit 212cc70bc: fix welmv4 RL CUDA graph + weight update.

This test instantiates REAL Qwen2MoeAttention objects from welmv4.py and uses
the REAL LayerManager.post_init. Only the SGLang distributed runtime and
RadixAttention are mocked (they are not relevant to the KV-mirror / CUDA graph
fix being tested).

The CUDA graph test captures F.linear(x, self.qkv_proj_weight, ...) — the
exact call path in Qwen2MoeAttention.forward — and verifies that after RL
weight updates + post_init, replay output matches eager output.

Usage (requires SGLang installed + CUDA):
    python test/rl_test/test_kv_mirror_cuda_graph_e2e.py
"""

import contextlib
import gc
import inspect
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.models.welmv4 import LayerManager, Qwen2MoeAttention


# ---------------------------------------------------------------------------
# Distributed-environment mock: the fix under test has nothing to do with
# TP/PP/RadixAttention, so we mock these to allow single-GPU unit testing.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def mock_sglang_runtime():
    """Patch distributed / runtime dependencies so Qwen2MoeAttention can be
    instantiated without a full SGLang server."""
    server_args = MagicMock()
    server_args.speculative_algorithm = None

    patches = [
        patch("sglang.srt.models.welmv4.get_attention_tp_rank", return_value=0),
        patch("sglang.srt.models.welmv4.get_attention_tp_size", return_value=1),
        patch(
            "sglang.srt.models.welmv4.get_global_server_args",
            return_value=server_args,
        ),
        patch("sglang.srt.models.welmv4.RadixAttention", MagicMock),
        patch("sglang.srt.models.welmv4.get_rope", return_value=MagicMock()),
        # ColumnParallelLinear (used by gate_proj) reads TP rank/size if not
        # explicitly passed.
        patch(
            "sglang.srt.layers.linear.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "sglang.srt.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# Thin wrapper: LayerManager.post_init accesses layer.self_attn, but we
# create Qwen2MoeAttention directly (not via the full Qwen2MoeDecoderLayer
# which pulls in MoE, LayerNorm, etc.).
# ---------------------------------------------------------------------------


class DecoderLayerStub(nn.Module):
    def __init__(self, attention: Qwen2MoeAttention):
        super().__init__()
        self.self_attn = attention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_attention(
    hidden_size,
    num_heads,
    num_kv_heads,
    head_dim,
    layer_idx,
    kv_mirror_layers,
    kv_mirror_imitated_layers,
    bias=True,
    total_layers=2,
):
    """Build a real Qwen2MoeAttention (must be called inside mock_sglang_runtime)."""
    LayerManager.num_target_layers = total_layers
    return Qwen2MoeAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        layer_id=layer_idx,
        layer_idx=layer_idx,
        qkv_bias=bias,
        qk_norm=False,
        k_norm=False,
        o_norm=False,
        kv_mirror_layers=kv_mirror_layers,
        kv_mirror_imitated_layers=kv_mirror_imitated_layers,
        sliding_window_size_layerwise=[],
        enable_attn_sink_layerwise=[],
        total_layer_num=total_layers,
    )


def _setup_layer_manager(layers: dict):
    LayerManager.decoder_layer = dict(layers)


def _teardown_layer_manager():
    LayerManager.decoder_layer = dict()
    LayerManager.num_nextn_predict_layer_idx = []


def _simulate_weight_update(*attns):
    """In-place randomize nn.Parameter data, as
    default_weight_loader(param, loaded_weight) -> param.data.copy_() does."""
    with torch.no_grad():
        for attn in attns:
            attn.qkv_proj.weight.data.normal_()
            if attn.qkv_proj.bias is not None:
                attn.qkv_proj.bias.data.normal_()


def _qkv_forward(attn, x):
    """The F.linear call path used by Qwen2MoeAttention.forward for both
    imitated and mirror layers when qkv_proj_weight is set."""
    return F.linear(x, attn.qkv_proj_weight, attn.qkv_proj_bias)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestLayerManagerPostInitE2E(unittest.TestCase):
    """End-to-end tests using REAL Qwen2MoeAttention + REAL LayerManager.post_init."""

    HIDDEN = 64
    NUM_HEADS = 4
    NUM_KV_HEADS = 2
    HEAD_DIM = 16
    BS = 4

    @property
    def q_size(self):
        return self.NUM_HEADS * self.HEAD_DIM

    @property
    def kv_size(self):
        return self.NUM_KV_HEADS * self.HEAD_DIM

    def setUp(self):
        _teardown_layer_manager()

    def tearDown(self):
        _teardown_layer_manager()

    def _build_mirror_imitated_pair(self, bias=True):
        """Build a (mirror, imitated) pair of real Qwen2MoeAttention objects,
        wrap them in DecoderLayerStub, and register in LayerManager."""
        mirror_id, imitated_id = 1, 0
        kv_mirror_layers = [mirror_id]
        kv_mirror_imitated_layers = [imitated_id]

        with mock_sglang_runtime():
            mirror_attn = _build_attention(
                self.HIDDEN, self.NUM_HEADS, self.NUM_KV_HEADS, self.HEAD_DIM,
                layer_idx=mirror_id,
                kv_mirror_layers=kv_mirror_layers,
                kv_mirror_imitated_layers=kv_mirror_imitated_layers,
                bias=bias,
            )
            imitated_attn = _build_attention(
                self.HIDDEN, self.NUM_HEADS, self.NUM_KV_HEADS, self.HEAD_DIM,
                layer_idx=imitated_id,
                kv_mirror_layers=kv_mirror_layers,
                kv_mirror_imitated_layers=kv_mirror_imitated_layers,
                bias=bias,
            )

        mirror = DecoderLayerStub(mirror_attn).cuda()
        imitated = DecoderLayerStub(imitated_attn).cuda()
        _setup_layer_manager({mirror_id: mirror, imitated_id: imitated})
        return mirror_id, imitated_id, mirror, imitated

    # ----------------------------------------------------------------
    # Source-level guards: detect if welmv4.py changes in ways that
    # would invalidate this test's assumptions.
    # ----------------------------------------------------------------

    def test_forward_source_uses_qkv_proj_weight(self):
        """Qwen2MoeAttention.forward must still use F.linear with
        self.qkv_proj_weight. If this fails, welmv4.py's forward path
        has changed and _qkv_forward() needs updating."""
        source = inspect.getsource(Qwen2MoeAttention.forward)
        self.assertIn(
            "self.qkv_proj_weight",
            source,
            "Qwen2MoeAttention.forward no longer references self.qkv_proj_weight",
        )
        self.assertIn(
            "F.linear",
            source,
            "Qwen2MoeAttention.forward no longer uses F.linear",
        )

    def test_attention_uses_real_qkv_parallel_linear(self):
        """Verify the real Qwen2MoeAttention creates a QKVParallelLinear,
        not a plain nn.Linear."""
        from sglang.srt.layers.linear import QKVParallelLinear

        mid, iid, mirror, imitated = self._build_mirror_imitated_pair()
        self.assertIsInstance(mirror.self_attn.qkv_proj, QKVParallelLinear)
        self.assertIsInstance(imitated.self_attn.qkv_proj, QKVParallelLinear)

    # ----------------------------------------------------------------
    # Core: address preservation
    # ----------------------------------------------------------------

    def test_post_init_creates_derived_tensors(self):
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair()
        LayerManager.post_init([mid], [iid])

        self.assertIsNotNone(mirror.self_attn.qkv_proj_weight)
        self.assertIsNotNone(imitated.self_attn.qkv_proj_weight)
        self.assertIsNotNone(mirror.self_attn.qkv_proj_bias)
        self.assertIsNotNone(imitated.self_attn.qkv_proj_bias)

    def test_post_init_preserves_addresses_on_second_call(self):
        """Core invariant: tensor addresses must not change across weight
        updates + post_init calls."""
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair()

        LayerManager.post_init([mid], [iid])
        addrs = {
            "mw": mirror.self_attn.qkv_proj_weight.data_ptr(),
            "iw": imitated.self_attn.qkv_proj_weight.data_ptr(),
            "mb": mirror.self_attn.qkv_proj_bias.data_ptr(),
            "ib": imitated.self_attn.qkv_proj_bias.data_ptr(),
        }

        _simulate_weight_update(mirror.self_attn, imitated.self_attn)
        LayerManager.post_init([mid], [iid])

        self.assertEqual(addrs["mw"], mirror.self_attn.qkv_proj_weight.data_ptr(),
                         "mirror qkv_proj_weight address changed")
        self.assertEqual(addrs["iw"], imitated.self_attn.qkv_proj_weight.data_ptr(),
                         "imitated qkv_proj_weight address changed")
        self.assertEqual(addrs["mb"], mirror.self_attn.qkv_proj_bias.data_ptr(),
                         "mirror qkv_proj_bias address changed")
        self.assertEqual(addrs["ib"], imitated.self_attn.qkv_proj_bias.data_ptr(),
                         "imitated qkv_proj_bias address changed")

    def test_post_init_preserves_addresses_no_bias(self):
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair(bias=False)

        LayerManager.post_init([mid], [iid])
        self.assertIsNone(mirror.self_attn.qkv_proj_bias)

        m_w_addr = mirror.self_attn.qkv_proj_weight.data_ptr()
        i_w_addr = imitated.self_attn.qkv_proj_weight.data_ptr()

        _simulate_weight_update(mirror.self_attn, imitated.self_attn)
        LayerManager.post_init([mid], [iid])

        self.assertEqual(m_w_addr, mirror.self_attn.qkv_proj_weight.data_ptr())
        self.assertEqual(i_w_addr, imitated.self_attn.qkv_proj_weight.data_ptr())

    def test_post_init_addresses_stable_across_many_updates(self):
        """Addresses remain stable across 10 simulated RL iterations."""
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair()

        LayerManager.post_init([mid], [iid])
        addrs = {
            "mw": mirror.self_attn.qkv_proj_weight.data_ptr(),
            "iw": imitated.self_attn.qkv_proj_weight.data_ptr(),
            "mb": mirror.self_attn.qkv_proj_bias.data_ptr(),
            "ib": imitated.self_attn.qkv_proj_bias.data_ptr(),
        }

        for i in range(10):
            _simulate_weight_update(mirror.self_attn, imitated.self_attn)
            LayerManager.post_init([mid], [iid])
            self.assertEqual(addrs["mw"], mirror.self_attn.qkv_proj_weight.data_ptr(),
                             f"mirror weight addr changed at iteration {i}")
            self.assertEqual(addrs["iw"], imitated.self_attn.qkv_proj_weight.data_ptr(),
                             f"imitated weight addr changed at iteration {i}")
            self.assertEqual(addrs["mb"], mirror.self_attn.qkv_proj_bias.data_ptr(),
                             f"mirror bias addr changed at iteration {i}")
            self.assertEqual(addrs["ib"], imitated.self_attn.qkv_proj_bias.data_ptr(),
                             f"imitated bias addr changed at iteration {i}")

    # ----------------------------------------------------------------
    # Core: derived weight values
    # ----------------------------------------------------------------

    def test_derived_weight_values_mirror(self):
        """mirror.qkv_proj_weight = mirror.qkv_proj.weight[:q_size, :]"""
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair(bias=False)

        with torch.no_grad():
            mirror.self_attn.qkv_proj.weight.fill_(1.0)
            imitated.self_attn.qkv_proj.weight.fill_(2.0)

        LayerManager.post_init([mid], [iid])

        expected = mirror.self_attn.qkv_proj.weight[: self.q_size, :]
        self.assertTrue(
            torch.equal(mirror.self_attn.qkv_proj_weight, expected),
            "mirror qkv_proj_weight != qkv_proj.weight[:q_size, :]",
        )

    def test_derived_weight_values_imitated(self):
        """imitated.qkv_proj_weight = cat([imitated.qkv_proj.weight,
                                           mirror.qkv_proj.weight[q_size:, :]])"""
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair(bias=False)

        with torch.no_grad():
            mirror.self_attn.qkv_proj.weight.fill_(1.0)
            imitated.self_attn.qkv_proj.weight.fill_(2.0)

        LayerManager.post_init([mid], [iid])

        expected = torch.concat([
            imitated.self_attn.qkv_proj.weight,
            mirror.self_attn.qkv_proj.weight[self.q_size:, :],
        ], dim=0)
        self.assertTrue(
            torch.equal(imitated.self_attn.qkv_proj_weight, expected),
            "imitated qkv_proj_weight values incorrect",
        )

    def test_derived_values_updated_after_weight_change(self):
        """After weight update + post_init, derived values reflect new weights."""
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair(bias=True)

        LayerManager.post_init([mid], [iid])

        with torch.no_grad():
            mirror.self_attn.qkv_proj.weight.fill_(5.0)
            imitated.self_attn.qkv_proj.weight.fill_(7.0)
            mirror.self_attn.qkv_proj.bias.fill_(0.5)
            imitated.self_attn.qkv_proj.bias.fill_(0.7)

        LayerManager.post_init([mid], [iid])

        self.assertTrue(torch.all(mirror.self_attn.qkv_proj_weight == 5.0))
        self.assertTrue(torch.all(mirror.self_attn.qkv_proj_bias == 0.5))

        q_kv_kv = self.q_size + self.kv_size + self.kv_size
        self.assertTrue(
            torch.all(imitated.self_attn.qkv_proj_weight[:q_kv_kv] == 7.0),
        )
        self.assertTrue(
            torch.all(imitated.self_attn.qkv_proj_weight[q_kv_kv:] == 5.0),
        )

    # ----------------------------------------------------------------
    # Core: CUDA graph correctness
    # ----------------------------------------------------------------

    def test_cuda_graph_replay_after_weight_update(self):
        """Capture CUDA graph -> update weights -> real post_init -> replay
        -> verify output matches eager."""
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair()

        LayerManager.post_init([mid], [iid])

        x_static = torch.randn(self.BS, self.HIDDEN, device="cuda")
        _ = _qkv_forward(imitated.self_attn, x_static)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_static = _qkv_forward(imitated.self_attn, x_static)

        _simulate_weight_update(mirror.self_attn, imitated.self_attn)
        LayerManager.post_init([mid], [iid])

        gc.collect()
        torch.cuda.empty_cache()

        x_new = torch.randn(self.BS, self.HIDDEN, device="cuda")
        x_static.copy_(x_new)
        graph.replay()
        graph_output = out_static.clone()

        eager_output = _qkv_forward(imitated.self_attn, x_new)
        self.assertTrue(
            torch.allclose(graph_output, eager_output, atol=1e-5),
            f"CUDA graph replay != eager after weight update. "
            f"Max diff: {(graph_output - eager_output).abs().max().item():.6e}",
        )

    def test_cuda_graph_replay_mirror_layer(self):
        """CUDA graph correctness for the mirror layer's forward path."""
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair()

        LayerManager.post_init([mid], [iid])

        x_static = torch.randn(self.BS, self.HIDDEN, device="cuda")
        _ = _qkv_forward(mirror.self_attn, x_static)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_static = _qkv_forward(mirror.self_attn, x_static)

        _simulate_weight_update(mirror.self_attn, imitated.self_attn)
        LayerManager.post_init([mid], [iid])

        gc.collect()
        torch.cuda.empty_cache()

        x_new = torch.randn(self.BS, self.HIDDEN, device="cuda")
        x_static.copy_(x_new)
        graph.replay()
        graph_output = out_static.clone()

        eager_output = _qkv_forward(mirror.self_attn, x_new)
        self.assertTrue(
            torch.allclose(graph_output, eager_output, atol=1e-5),
            f"Mirror layer: CUDA graph replay != eager. "
            f"Max diff: {(graph_output - eager_output).abs().max().item():.6e}",
        )

    def test_cuda_graph_correctness_over_multiple_rl_iterations(self):
        """Capture graph once, then 5 RL update cycles, each verified."""
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair()

        LayerManager.post_init([mid], [iid])

        x_static = torch.randn(self.BS, self.HIDDEN, device="cuda")
        _ = _qkv_forward(imitated.self_attn, x_static)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_static = _qkv_forward(imitated.self_attn, x_static)

        for i in range(5):
            _simulate_weight_update(mirror.self_attn, imitated.self_attn)
            LayerManager.post_init([mid], [iid])

            x_new = torch.randn(self.BS, self.HIDDEN, device="cuda")
            x_static.copy_(x_new)
            graph.replay()
            graph_out = out_static.clone()

            eager_out = _qkv_forward(imitated.self_attn, x_new)
            self.assertTrue(
                torch.allclose(graph_out, eager_out, atol=1e-5),
                f"RL iteration {i}: max diff "
                f"{(graph_out - eager_out).abs().max().item():.6e}",
            )

    # ----------------------------------------------------------------
    # Multiple kv_mirror pairs
    # ----------------------------------------------------------------

    def test_multiple_mirror_pairs(self):
        """2 (mirror, imitated) pairs, address stability after update."""
        kv_mirror_layers = [2, 3]
        kv_mirror_imitated_layers = [0, 1]
        total_layers = 4

        layers = {}
        with mock_sglang_runtime():
            for lid in range(total_layers):
                attn = _build_attention(
                    self.HIDDEN, self.NUM_HEADS, self.NUM_KV_HEADS, self.HEAD_DIM,
                    layer_idx=lid,
                    kv_mirror_layers=kv_mirror_layers,
                    kv_mirror_imitated_layers=kv_mirror_imitated_layers,
                    total_layers=total_layers,
                )
                layers[lid] = DecoderLayerStub(attn).cuda()

        _setup_layer_manager(layers)
        LayerManager.post_init(kv_mirror_layers, kv_mirror_imitated_layers)

        addrs = {}
        for lid in range(total_layers):
            attn = layers[lid].self_attn
            addrs[f"{lid}_w"] = attn.qkv_proj_weight.data_ptr()
            addrs[f"{lid}_b"] = attn.qkv_proj_bias.data_ptr()

        for lid in range(total_layers):
            _simulate_weight_update(layers[lid].self_attn)

        LayerManager.post_init(kv_mirror_layers, kv_mirror_imitated_layers)

        for lid in range(total_layers):
            attn = layers[lid].self_attn
            self.assertEqual(addrs[f"{lid}_w"], attn.qkv_proj_weight.data_ptr(),
                             f"Layer {lid}: weight address changed")
            self.assertEqual(addrs[f"{lid}_b"], attn.qkv_proj_bias.data_ptr(),
                             f"Layer {lid}: bias address changed")

    # ----------------------------------------------------------------
    # Shape correctness
    # ----------------------------------------------------------------

    def test_derived_tensor_shapes(self):
        """Verify derived tensor shapes match WelmV4Attention.forward expectations."""
        mid, iid, mirror, imitated = self._build_mirror_imitated_pair()
        LayerManager.post_init([mid], [iid])

        self.assertEqual(
            mirror.self_attn.qkv_proj_weight.shape,
            (self.q_size, self.HIDDEN),
        )
        # imitated: original qkv output + mirror's KV part
        self.assertEqual(
            imitated.self_attn.qkv_proj_weight.shape,
            (self.q_size + 4 * self.kv_size, self.HIDDEN),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
