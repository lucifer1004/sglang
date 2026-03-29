"""
Precision tests for WelmV4 fused ops (welmv4_op.py).

Each fused op is compared against the pre-fusion calling pattern used in welmv4.py,
using SGLang's own native implementations as the reference.

Model config parameters (both configs share the same values):
  hidden_size=2048, head_dim=256, num_attention_heads=24, num_key_value_heads=2,
  qk_rope_head_dim=64, qk_norm=False, k_norm=True, rms_norm_eps=1e-5,
  gated_self_attention_headwise=True, ppln=True

Tolerances: atol=2e-2 (bf16 has ~0.0078 per-ULP precision).
"""

import unittest

import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.layers.welmv4_op import (
    WelmV4FusedRMSNorm,
    WelmV4InplaceRotaryEmbedding,
    inplace_sigmoid_mul,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

torch.random.manual_seed(1)

# bf16 has ~0.0078 per-ULP precision.
# Triton's tree-reduction for sum vs PyTorch's sequential accumulation can cause
# up to ~0.5 ULP difference in the final bf16 result, so we use 5e-3 (< 1 ULP).
ATOL = 1e-3
RTOL = 1e-3
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Model config parameters (shared by both configs)
HIDDEN_SIZE = 2048
HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_K_HEADS = 2
QK_ROPE_HEAD_DIM = 64
RMS_NORM_EPS = 1e-5

num_tokens_list = [127]


def setUpModule():
    """Set global server args required by RotaryEmbedding init."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


# =====================================================================
# Test 1: WelmV4FusedRMSNorm vs RMSNorm.forward_native
# =====================================================================
class TestWelmV4FusedRMSNorm(CustomTestCase):
    """
    Pre-fusion code used RMSNorm from sglang.srt.layers.layernorm.
    Calling patterns (from the git diff):

    DecoderLayer (hidden_size=2048, eps=1e-5):
      - input_layernorm:
          Before:  if residual is None:
                       residual = hidden_states
                       hidden_states = self.input_layernorm(hidden_states)
                   else:
                       hidden_states, residual = self.input_layernorm(hidden_states, residual)
          After:   hidden_states, residual = self.input_layernorm(
                       hidden_states, residual, residual_after_layernorm=...)

      - post_attention_layernorm:
          Before:  hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
          After:   hidden_states, residual, hidden_states_fp32 = self.post_attention_layernorm(
                       hidden_states, residual, clone_fp32_out=True)

    Attention k_norm (hidden_size=head_dim=256, eps=1e-5, k_norm=True, qk_norm=False):
      Before:  k_by_head = self.k_norm.forward_native(k_by_head)
      After:   k_by_head, _ = self.k_norm(k_by_head)
    """

    def _make_ref_and_fused(self, hidden_size, eps=RMS_NORM_EPS):
        ref = RMSNorm(hidden_size, eps=eps).to(device=DEVICE, dtype=DTYPE)
        fused = WelmV4FusedRMSNorm(hidden_size, eps=eps).to(device=DEVICE, dtype=DTYPE)
        fused.weight.data.copy_(ref.weight.data)
        return ref, fused

    # --- Case (a): no residual (first layer, residual=None) ---
    def test_no_residual(self):
        ref, fused = self._make_ref_and_fused(HIDDEN_SIZE)
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                x = torch.randn(num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                ref_out = ref.forward_native(x.clone())
                ref_residual = x.clone()
                fused_out, fused_residual = fused.forward_cuda(x.clone())
                torch.testing.assert_close(fused_out, ref_out, atol=ATOL, rtol=RTOL)
                torch.testing.assert_close(
                    fused_residual, ref_residual, atol=ATOL, rtol=RTOL
                )

    # --- Case (b): with residual ---
    def test_with_residual(self):
        ref, fused = self._make_ref_and_fused(HIDDEN_SIZE)
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                x = torch.randn(num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                residual = torch.randn(
                    num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
                )
                ref_out, ref_residual = ref.forward_native(
                    x.clone(), residual=residual.clone()
                )
                fused_out, fused_residual = fused.forward_cuda(
                    x.clone(), residual=residual.clone()
                )
                torch.testing.assert_close(fused_out, ref_out, atol=ATOL, rtol=RTOL)
                torch.testing.assert_close(
                    fused_residual, ref_residual, atol=ATOL, rtol=RTOL
                )

    # --- Case (c): residual_after_layernorm=True (ppln=True) ---
    # Pre-fusion: hidden_states, residual = layernorm(h, r); residual = hidden_states.clone()
    # Fused: hidden_states, residual = layernorm(h, r, residual_after_layernorm=True)
    def test_residual_after_layernorm(self):
        ref, fused = self._make_ref_and_fused(HIDDEN_SIZE)
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                x = torch.randn(num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                residual = torch.randn(
                    num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
                )
                ref_out, _ = ref.forward_native(x.clone(), residual=residual.clone())
                ref_residual = ref_out.clone()
                fused_out, fused_residual = fused.forward_cuda(
                    x.clone(),
                    residual=residual.clone(),
                    residual_after_layernorm=True,
                )
                torch.testing.assert_close(fused_out, ref_out, atol=ATOL, rtol=RTOL)
                torch.testing.assert_close(
                    fused_residual, ref_residual, atol=ATOL, rtol=RTOL
                )

    # --- Case (d): clone_fp32_out=True (post_attention_layernorm for MoE router) ---
    def test_clone_fp32_out(self):
        ref, fused = self._make_ref_and_fused(HIDDEN_SIZE)
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                x = torch.randn(num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                residual = torch.randn(
                    num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
                )
                ref_out, ref_residual = ref.forward_native(
                    x.clone(), residual=residual.clone()
                )
                fused_out, fused_residual, fused_fp32 = fused.forward_cuda(
                    x.clone(), residual=residual.clone(), clone_fp32_out=True
                )
                torch.testing.assert_close(fused_out, ref_out, atol=ATOL, rtol=RTOL)
                torch.testing.assert_close(
                    fused_residual, ref_residual, atol=ATOL, rtol=RTOL
                )
                # fp32_out cast to bf16 should exactly match bf16 output
                torch.testing.assert_close(
                    fused_fp32.to(DTYPE), fused_out, atol=0, rtol=0
                )
                # fp32_out vs pre-fusion pattern (bf16 output cast to fp32)
                ref_fp32 = ref_out.to(torch.float32)
                torch.testing.assert_close(fused_fp32, ref_fp32, atol=2e-2, rtol=2e-2)

    # --- Case (e): per-head k_norm (head_dim=256, k_norm=True, qk_norm=False) ---
    # Before: k_by_head = self.k_norm.forward_native(k_by_head)
    #         k_by_head shape: (num_tokens, num_kv_heads, head_dim)
    # After:  k_by_head, _ = self.k_norm(k_by_head)
    # Reshape to 2D to match kernel's stride assumption.
    def test_per_head_k_norm(self):
        ref, fused = self._make_ref_and_fused(HEAD_DIM)
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                k_by_head = torch.randn(
                    num_tokens, NUM_K_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                ref_out = ref.forward_native(k_by_head.clone())
                x_2d = k_by_head.clone().reshape(-1, HEAD_DIM)
                fused_out_2d, _ = fused.forward_cuda(x_2d)
                fused_out = fused_out_2d.view(num_tokens, NUM_K_HEADS, HEAD_DIM)
                torch.testing.assert_close(fused_out, ref_out, atol=ATOL, rtol=RTOL)


# =====================================================================
# Test 2: inplace_sigmoid_mul vs torch.sigmoid(gate) * attn_output
# =====================================================================
# NOTE: 和之前的实现无法对齐，应为 triton 中的 sigmoid 算子强制限制在fp32或以上。
class TestInplaceSigmoidMul(CustomTestCase):
    """
    Pre-fusion code (gated_self_attention_headwise=True):
        gate = self.gate_proj(hidden_states)[0].unsqueeze(-1)  # (bs*seq_len, num_heads, 1)
        attn_output = attn_output.view(attn_shape[0], self.num_heads, -1)
        attn_output = attn_output * torch.sigmoid(gate)

    After fusion:
        inplace_sigmoid_mul(gate, attn_output)

    Triton kernel computes sigmoid in fp32. Pre-fusion code computed in bf16:
        attn_output = attn_output * torch.sigmoid(gate)
    Reference uses the pre-fusion bf16 path.
    Model params: num_attention_heads=24, head_dim=256.
    """

    def test_basic(self):
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                gate = torch.randn(
                    num_tokens, NUM_Q_HEADS, 1, device=DEVICE, dtype=DTYPE
                )
                attn_output = torch.randn(
                    num_tokens, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                # Reference: pre-fusion PyTorch code (bf16 path)
                ref_out = attn_output.clone() * torch.sigmoid(gate.clone())
                fused_attn = attn_output.clone()
                inplace_sigmoid_mul(gate.clone(), fused_attn)
                torch.testing.assert_close(fused_attn, ref_out, atol=ATOL, rtol=RTOL)


# =====================================================================
# Test 3: WelmV4InplaceRotaryEmbedding
# =====================================================================
class TestWelmV4InplaceRotaryEmbedding(CustomTestCase):
    """
    Pre-fusion code (on CUDA, via CustomOp dispatch -> forward_cuda):
        # self.rotary_emb was RotaryEmbedding(head_size=qk_rope_head_dim, rotary_dim=qk_rope_head_dim)
        # which on CUDA dispatches to forward_cuda -> sgl_kernel.apply_rope_with_cos_sin_cache_inplace
        q_nope, q_pe = q.view(q_by_head_shape).split([nope_dim, rope_dim], dim=-1)
        k_nope, k_pe = k.view(k_by_head_shape).split([nope_dim, rope_dim], dim=-1)
        q_pe = q_pe.reshape((*q_shape[:-1], q_shape[-1] // head_dim * rope_dim))
        k_pe = k_pe.reshape((*k_shape[:-1], k_shape[-1] // head_dim * rope_dim))
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)  # -> forward_cuda on CUDA
        q[..., nope_dim:] = q_pe;  k[..., nope_dim:] = k_pe

    After fusion:
        # self.rotary_emb is WelmV4InplaceRotaryEmbedding(head_size=head_dim, rotary_dim=qk_rope_head_dim)
        self.rotary_emb.forward_cuda(positions, q, k)

    Model params: head_dim=256, qk_rope_head_dim=64 (partial_rotary_factor=0.25),
    num_attention_heads=24, num_key_value_heads=2, is_neox_style=True.
    """

    def _make_ref_and_fused(self, max_pos=32768, base=100000):
        """
        ref: RotaryEmbedding(head_size=QK_ROPE_HEAD_DIM, rotary_dim=QK_ROPE_HEAD_DIM)
             — matches the pre-fusion rotary_emb, on CUDA uses sgl_kernel rope.
        fused: WelmV4InplaceRotaryEmbedding(head_size=HEAD_DIM, rotary_dim=QK_ROPE_HEAD_DIM)
             — the fused replacement, applies rope in-place on the tail of each head.
        """
        ref = RotaryEmbedding(
            head_size=QK_ROPE_HEAD_DIM,
            rotary_dim=QK_ROPE_HEAD_DIM,
            max_position_embeddings=max_pos,
            base=base,
            is_neox_style=True,
            dtype=DTYPE,
        ).to(DEVICE)

        fused = WelmV4InplaceRotaryEmbedding(
            head_size=HEAD_DIM,
            rotary_dim=QK_ROPE_HEAD_DIM,
            max_position_embeddings=max_pos,
            base=base,
            is_neox_style=True,
            dtype=DTYPE,
        ).to(DEVICE)

        return ref, fused

    def _ref_rope_on_tail(self, ref, positions, data, num_heads):
        """
        Reference: exactly replicate the pre-fusion calling pattern.
        Split out the rope portion, flatten to 2D, call ref.forward_cuda, put back.
        """
        num_tokens = positions.shape[0]
        nope_dim = HEAD_DIM - QK_ROPE_HEAD_DIM

        data = data.view(num_tokens, num_heads, HEAD_DIM).clone()

        # Split: extract rope portion from tail of each head
        data_pe = data[..., nope_dim:].contiguous()  # (tokens, heads, rope_dim)

        # Flatten to 2D as the pre-fusion code did:
        #   q_pe = q_pe.reshape((*q_shape[:-1], num_heads * rope_dim))
        data_pe_flat = data_pe.reshape(num_tokens, num_heads * QK_ROPE_HEAD_DIM)

        # We need a dummy counterpart for the other argument (ref expects both q and k)
        dummy = torch.zeros_like(data_pe_flat)

        # Call RotaryEmbedding.forward_cuda (sgl_kernel rope) — same as pre-fusion
        data_pe_out, _ = ref.forward_cuda(positions, data_pe_flat, dummy)

        # Reshape back and put into original data
        data_pe_out = data_pe_out.reshape(num_tokens, num_heads, QK_ROPE_HEAD_DIM)
        data[..., nope_dim:] = data_pe_out

        return data.view(num_tokens, num_heads * HEAD_DIM)

    # --- Case (a): basic partial rotation (head_dim=256, rope_dim=64) ---
    def test_basic_partial_rotation(self):
        ref, fused = self._make_ref_and_fused()
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                positions = torch.randint(
                    0, 4096, (num_tokens,), device=DEVICE, dtype=torch.int64
                )
                q = torch.randn(
                    num_tokens, NUM_Q_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                k = torch.randn(
                    num_tokens, NUM_K_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                q_ref = self._ref_rope_on_tail(ref, positions, q.clone(), NUM_Q_HEADS)
                k_ref = self._ref_rope_on_tail(ref, positions, k.clone(), NUM_K_HEADS)
                q_fused = q.clone()
                k_fused = k.clone()
                fused.forward_cuda(positions, q_fused, k_fused)
                torch.testing.assert_close(q_fused, q_ref, atol=ATOL, rtol=RTOL)
                torch.testing.assert_close(k_fused, k_ref, atol=ATOL, rtol=RTOL)

    # --- Case (b): verify nope portion unchanged ---
    def test_nope_unchanged(self):
        _, fused = self._make_ref_and_fused()
        nope_dim = HEAD_DIM - QK_ROPE_HEAD_DIM
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                positions = torch.randint(
                    0, 4096, (num_tokens,), device=DEVICE, dtype=torch.int64
                )
                q = torch.randn(
                    num_tokens, NUM_Q_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                k = torch.randn(
                    num_tokens, NUM_K_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                q_orig = q.clone().view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
                k_orig = k.clone().view(num_tokens, NUM_K_HEADS, HEAD_DIM)
                q_fused = q.clone()
                k_fused = k.clone()
                fused.forward_cuda(positions, q_fused, k_fused)
                q_fused_h = q_fused.view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
                k_fused_h = k_fused.view(num_tokens, NUM_K_HEADS, HEAD_DIM)
                # Non-rope part (first nope_dim elements per head) must be unchanged
                torch.testing.assert_close(
                    q_fused_h[..., :nope_dim], q_orig[..., :nope_dim], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    k_fused_h[..., :nope_dim], k_orig[..., :nope_dim], atol=0, rtol=0
                )

    # --- Case (c): with last_index (KV mirror mode) ---
    # Pre-fusion:
    #   _, k_pe = self.rotary_emb(positions, q_pe_proxy, k_pe)
    #   q_pe, _ = self.rotary_emb(positions[custom_last_index], q_pe, k_pe_proxy)
    # After fusion:
    #   self.rotary_emb.forward_cuda(positions, q, k, last_index=custom_last_index)
    def test_with_last_index(self):
        ref, fused = self._make_ref_and_fused()
        num_tokens = 64
        bs = 4
        last_index = torch.tensor([15, 31, 47, 63], device=DEVICE, dtype=torch.int64)
        positions = torch.randint(
            0, 4096, (num_tokens,), device=DEVICE, dtype=torch.int64
        )
        q = torch.randn(num_tokens, NUM_Q_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
        k = torch.randn(num_tokens, NUM_K_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
        nope_dim = HEAD_DIM - QK_ROPE_HEAD_DIM

        # Reference: k with all positions (all tokens)
        k_ref = self._ref_rope_on_tail(ref, positions, k.clone(), NUM_K_HEADS)

        # Reference: q with positions[last_index] for first bs tokens only
        # Pre-fusion: q_pe, _ = self.rotary_emb(positions[last_index], q_pe, k_pe_proxy)
        q_ref_heads = q.clone().view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
        q_pe_bs = q_ref_heads[:bs, :, nope_dim:].contiguous()
        q_pe_bs_flat = q_pe_bs.reshape(bs, NUM_Q_HEADS * QK_ROPE_HEAD_DIM)
        dummy = torch.zeros(
            bs, NUM_K_HEADS * QK_ROPE_HEAD_DIM, device=DEVICE, dtype=DTYPE
        )
        q_pe_out, _ = ref.forward_cuda(positions[last_index], q_pe_bs_flat, dummy)
        q_pe_out = q_pe_out.reshape(bs, NUM_Q_HEADS, QK_ROPE_HEAD_DIM)
        q_ref_heads[:bs, :, nope_dim:] = q_pe_out

        # Fused
        q_fused = q.clone()
        k_fused = k.clone()
        fused.forward_cuda(positions, q_fused, k_fused, last_index=last_index)

        # Compare k (all tokens)
        torch.testing.assert_close(k_fused, k_ref, atol=ATOL, rtol=RTOL)
        # Compare q (first bs tokens)
        q_fused_heads = q_fused.view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
        torch.testing.assert_close(
            q_fused_heads[:bs], q_ref_heads[:bs], atol=ATOL, rtol=RTOL
        )
        # q tokens beyond bs should be unchanged
        q_orig_heads = q.view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
        torch.testing.assert_close(
            q_fused_heads[bs:], q_orig_heads[bs:], atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
