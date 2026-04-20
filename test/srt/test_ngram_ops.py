"""Unit tests for the JIT-compiled n-gram ops used by WeLM v4 MTP.

Tests three CUDA kernels exposed via ``sglang.jit_kernel.ngram_ops``:

- ``build_ngram_with_tree``
- ``build_ngram_with_target_verify``
- ``assign_ngram_input_ids_draft_extend_after_decode``

Each kernel is compared against a pure-PyTorch reference implementation
that mirrors the CUDA logic in ``python/sglang/jit_kernel/csrc/ngram_ops.cuh``.

Run with::

    python -m pytest test/srt/test_ngram_ops.py -v
"""

from __future__ import annotations

import unittest

import torch

from sglang.jit_kernel.ngram_ops import (
    assign_ngram_input_ids_draft_extend_after_decode,
    build_ngram_with_target_verify,
    build_ngram_with_tree,
)
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# Reference implementations (mirror the CUDA kernels exactly)
# ---------------------------------------------------------------------------


def ref_build_ngram_with_tree(
    parent_list: torch.Tensor,
    token_list: torch.Tensor,
    current_parrent_list: torch.Tensor,
    buffer: torch.Tensor,
    buffer_size: int,
    gram_n: int,
    topk: int,
    i: int,
) -> torch.Tensor:
    bs = parent_list.size(0)
    parent_list_stride = parent_list.stride(0)
    token_list_stride = token_list.stride(0)

    parent_list_flat = parent_list.flatten()
    token_list_flat = token_list.flatten()
    out = torch.zeros(bs * topk, dtype=torch.int64, device=parent_list.device)

    gram = gram_n - 1 - i
    for bid in range(bs):
        for tid in range(topk):
            if gram > 0:
                out[bid * topk + tid] = buffer[(bid + 1) * buffer_size - gram]
                continue

            current_pos = int(current_parrent_list[bid, tid].item())
            ii = i
            parent_token = 0
            for _ in range(gram_n - 1):
                pre_layer_num_node = topk + topk * topk * (ii - 1)
                cur_layer_pos = current_pos - pre_layer_num_node
                parent_layer_pos = cur_layer_pos // topk
                parent_offset = 1 + topk * (ii - 1)
                parent_pos = parent_layer_pos + parent_offset
                parent_pos = int(
                    parent_list_flat[bid * parent_list_stride + parent_pos].item()
                )
                parent_token = int(
                    token_list_flat[bid * token_list_stride + parent_pos].item()
                )
                current_pos = parent_pos
                ii -= 1
            out[bid * topk + tid] = parent_token
    return out


def ref_build_ngram_with_target_verify(
    buffer: torch.Tensor,
    draft_token_ids: torch.Tensor,
    tree_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    gram_n: int,
    draft_token_num: int,
    buffer_size: int,
) -> torch.Tensor:
    bs = seq_lens.numel()
    out = torch.zeros(
        bs * draft_token_num, dtype=torch.int64, device=draft_token_ids.device
    )

    seq_lens_cpu = seq_lens.cpu().tolist()
    draft_cpu = draft_token_ids.cpu().tolist()
    buffer_cpu = buffer.cpu().tolist()
    mask_cpu = tree_mask.cpu().tolist()

    for bid in range(bs * draft_token_num):
        seq_id = bid // draft_token_num
        mask_offset = 0
        for j in range(seq_id):
            mask_offset += draft_token_num * (seq_lens_cpu[j] + draft_token_num)
        seq_len = seq_lens_cpu[seq_id]
        mask_len = seq_len + draft_token_num
        mask_offset += (bid % draft_token_num) * mask_len

        target_gram = gram_n
        res = 0
        for k in range(seq_len + draft_token_num - 1, seq_len - 1, -1):
            if mask_cpu[mask_offset + k]:
                target_gram -= 1
                if target_gram == 0:
                    res = draft_cpu[seq_id * draft_token_num + (k - seq_len)]
                    break
        if target_gram != 0:
            res = buffer_cpu[(seq_id + 1) * buffer_size - target_gram - 1]
        out[bid] = res
    return out


def ref_assign_ngram_input_ids_draft_extend_after_decode(
    input_ids: torch.Tensor,
    buffer: torch.Tensor,
    accept_length: torch.Tensor,
    gram_n: int,
    buffer_size: int,
) -> torch.Tensor:
    bs = accept_length.numel()
    accept_cpu = accept_length.cpu().tolist()
    total = sum(accept_cpu)
    out = torch.zeros(total, dtype=torch.int64, device=input_ids.device)

    input_ids_cpu = input_ids.cpu().tolist()
    buffer_cpu = buffer.cpu().tolist()
    gram = gram_n - 1
    accum = 0
    for bid in range(bs):
        curr = accept_cpu[bid]
        for tid in range(curr):
            if tid >= gram:
                out[accum + tid] = input_ids_cpu[accum + tid - gram]
            else:
                out[accum + tid] = buffer_cpu[
                    bid * buffer_size + buffer_size - (gram - tid)
                ]
        accum += curr
    return out


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for ngram ops")
class TestBuildNgramWithTree(CustomTestCase):
    def _run(self, bs: int, topk: int, gram_n: int, buffer_size: int, i: int):
        device = torch.device("cuda")
        torch.manual_seed(0)

        # Layout matches the original prc_custom_ops convention used in
        # WeLM v4: parent_list/token_list grow with the tree depth.
        # Use a conservative shape that is large enough for any valid index.
        layer_widths = [topk] + [topk * topk] * max(i, 0)
        max_nodes = sum(layer_widths) + topk * topk * 4 + 8

        parent_list = torch.randint(
            0, max(1, max_nodes), (bs, max_nodes), dtype=torch.int64, device=device
        )
        token_list = torch.randint(
            0, 50000, (bs, max_nodes), dtype=torch.int64, device=device
        )
        # current_parrent_list values must point past the previous layers.
        pre_layer_num_node = topk + topk * topk * max(i - 1, 0)
        current_parrent_list = torch.arange(
            pre_layer_num_node,
            pre_layer_num_node + topk,
            dtype=torch.int64,
            device=device,
        ).repeat(bs, 1)
        buffer = torch.randint(
            0, 50000, (bs * buffer_size,), dtype=torch.int64, device=device
        )

        out_jit = torch.empty(bs * topk, dtype=torch.int64, device=device)
        build_ngram_with_tree(
            out_jit,
            parent_list,
            token_list,
            current_parrent_list,
            buffer,
            buffer_size,
            gram_n,
            topk,
            i,
        )
        torch.cuda.synchronize()

        out_ref = ref_build_ngram_with_tree(
            parent_list,
            token_list,
            current_parrent_list,
            buffer,
            buffer_size,
            gram_n,
            topk,
            i,
        )
        self.assertTrue(
            torch.equal(out_jit.cpu(), out_ref.cpu()),
            f"Mismatch:\n  jit={out_jit}\n  ref={out_ref}",
        )

    def test_gram2_first_layer(self):
        self._run(bs=2, topk=2, gram_n=2, buffer_size=4, i=0)

    def test_gram3_mid_layer(self):
        self._run(bs=3, topk=4, gram_n=3, buffer_size=4, i=1)

    def test_gram4_deep_layer(self):
        self._run(bs=2, topk=2, gram_n=4, buffer_size=4, i=3)

    def test_large_batch(self):
        self._run(bs=8, topk=8, gram_n=3, buffer_size=4, i=1)

    def test_mtp_test_example(self):
        # Hardcoded expected value from mtp_test.py (lines 40-57):
        #   i=2, topk=2, gram_n=3, buffer_size=4
        #   parent_list = [[-1, 0, 1], [5, 3], [7, 9]] -> cat = [-1,0,1,5,3,7,9]
        #   token_list  = [[12,13],   [14,15,16,17], [18,19,20,21]]
        #                            -> cat = [12,13,14,15,16,17,18,19,20,21]
        #   current_parrent_list = parent_list[i] = [[7, 9]]
        #   buffer = [1..8]
        #   Expected (per the comment in mtp_test.py): [13, 12]
        device = torch.device("cuda")
        parent_tensor = torch.tensor(
            [[-1, 0, 1, 5, 3, 7, 9]], dtype=torch.int64, device=device
        )
        token_tensor = torch.tensor(
            [[12, 13, 14, 15, 16, 17, 18, 19, 20, 21]],
            dtype=torch.int64,
            device=device,
        )
        current_parrent_list = torch.tensor([[7, 9]], dtype=torch.int64, device=device)
        buffer = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device=device
        )
        out_jit = torch.empty(2, dtype=torch.int64, device=device)
        build_ngram_with_tree(
            out_jit,
            parent_tensor,
            token_tensor,
            current_parrent_list,
            buffer,
            buffer_size=4,
            gram_n=3,
            topk=2,
            i=2,
        )
        torch.cuda.synchronize()
        expected = torch.tensor([13, 12], dtype=torch.int64)
        self.assertTrue(
            torch.equal(out_jit.cpu(), expected),
            f"Mismatch:\n  jit={out_jit.cpu()}\n  expected={expected}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for ngram ops")
class TestBuildNgramWithTargetVerify(CustomTestCase):
    def _build_inputs(
        self,
        seq_lens_list,
        draft_token_num: int,
        gram_n: int,
        buffer_size: int,
        seed: int = 0,
    ):
        device = torch.device("cuda")
        torch.manual_seed(seed)
        bs = len(seq_lens_list)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)

        draft_token_ids = torch.randint(
            0, 50000, (bs * draft_token_num,), dtype=torch.int64, device=device
        )
        buffer = torch.randint(
            0, 50000, (bs * buffer_size,), dtype=torch.int64, device=device
        )
        positions = torch.zeros(bs * draft_token_num, dtype=torch.int64, device=device)

        # tree_mask is a flattened concatenation of per-sequence masks of
        # shape [draft_token_num, seq_len + draft_token_num].
        masks = []
        for sl in seq_lens_list:
            m = torch.randint(
                0, 2, (draft_token_num, sl + draft_token_num), device=device
            ).bool()
            # Always make the diagonal-tail entry True so we can hit edge cases.
            for r in range(draft_token_num):
                m[r, sl + r] = True
            masks.append(m.flatten())
        tree_mask = torch.cat(masks)
        return seq_lens, draft_token_ids, buffer, positions, tree_mask

    def _run(self, seq_lens_list, draft_token_num, gram_n, buffer_size):
        device = torch.device("cuda")
        seq_lens, draft_token_ids, buffer, positions, tree_mask = self._build_inputs(
            seq_lens_list, draft_token_num, gram_n, buffer_size
        )
        bs = len(seq_lens_list)

        out_jit = torch.empty(bs * draft_token_num, dtype=torch.int64, device=device)
        build_ngram_with_target_verify(
            out_jit,
            buffer,
            draft_token_ids,
            tree_mask,
            positions,
            seq_lens,
            gram_n,
            draft_token_num,
            buffer_size,
        )
        torch.cuda.synchronize()

        out_ref = ref_build_ngram_with_target_verify(
            buffer,
            draft_token_ids,
            tree_mask,
            seq_lens,
            gram_n,
            draft_token_num,
            buffer_size,
        )
        self.assertTrue(
            torch.equal(out_jit.cpu(), out_ref.cpu()),
            f"Mismatch:\n  jit={out_jit}\n  ref={out_ref}",
        )

    def test_basic(self):
        self._run([7, 6], draft_token_num=4, gram_n=4, buffer_size=4)

    def test_gram2(self):
        self._run([5, 8, 3], draft_token_num=4, gram_n=2, buffer_size=4)

    def test_gram3(self):
        self._run([10, 4, 6, 9], draft_token_num=4, gram_n=3, buffer_size=4)

    def test_single_seq(self):
        self._run([12], draft_token_num=4, gram_n=4, buffer_size=4)

    def test_mtp_test_example(self):
        # Hardcoded case from mtp_test.py (lines 75-98):
        #   bs=2, draft_token_num=4, gram_n=4, buffer_size=4
        #   seq_lens         = [7, 6]
        #   draft_token_ids  = [12, 13, 14, 15, 16, 17, 18, 19]
        #   tree_mask flat (84 bools) — see mtp_test.py for the layout:
        #     seq 0 (44 bools = 4 rows of 11): per-row masks for the
        #     [seq_len + draft_token_num] = 11 token window
        #     seq 1 (40 bools = 4 rows of 10).
        #
        # NOTE: mtp_test.py uses ``buffer = [1..6]`` which is undersized
        # (kernel needs ``bs * buffer_size = 8`` elements and may read up
        # to index 6). We extend it to [1..8]; everything else is identical.
        # Expected output, traced manually through the kernel logic:
        #   bid=0 -> target_gram=3, res=buffer[0]=1
        #   bid=1 -> target_gram=2, res=buffer[1]=2
        #   bid=2 -> target_gram=2, res=buffer[1]=2
        #   bid=3 -> target_gram=1, res=buffer[2]=3
        #   bid=4 -> target_gram=3, res=buffer[4]=5
        #   bid=5 -> target_gram=2, res=buffer[5]=6
        #   bid=6 -> target_gram=2, res=buffer[5]=6
        #   bid=7 -> target_gram=1, res=buffer[6]=7
        device = torch.device("cuda")
        draft_token_ids = torch.tensor(
            [12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int64, device=device
        )
        tree_mask = torch.tensor(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
            ],
            dtype=torch.bool,
            device=device,
        )
        positions = torch.tensor(
            [7, 8, 8, 9, 6, 7, 7, 8], dtype=torch.int64, device=device
        )
        buffer = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device=device
        )
        seq_lens = torch.tensor([7, 6], dtype=torch.int64, device=device)

        out_jit = torch.empty(8, dtype=torch.int64, device=device)
        build_ngram_with_target_verify(
            out_jit,
            buffer,
            draft_token_ids,
            tree_mask,
            positions,
            seq_lens,
            gram_n=4,
            draft_token_num=4,
            buffer_size=4,
        )
        torch.cuda.synchronize()
        expected = torch.tensor([1, 2, 2, 3, 5, 6, 6, 7], dtype=torch.int64)
        # Cross-check our manual trace against the Python reference too.
        out_ref = ref_build_ngram_with_target_verify(
            buffer, draft_token_ids, tree_mask, seq_lens, 4, 4, 4
        )
        self.assertTrue(
            torch.equal(out_ref.cpu(), expected),
            f"ref disagrees with manual trace: ref={out_ref.cpu()} expected={expected}",
        )
        self.assertTrue(
            torch.equal(out_jit.cpu(), expected),
            f"jit disagrees with manual trace: jit={out_jit.cpu()} expected={expected}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for ngram ops")
class TestAssignNgramInputIdsDraftExtendAfterDecode(CustomTestCase):
    def _run(self, accept_lens, gram_n, buffer_size, seed=0):
        device = torch.device("cuda")
        torch.manual_seed(seed)
        bs = len(accept_lens)
        total = sum(accept_lens)

        accept_length = torch.tensor(accept_lens, dtype=torch.int32, device=device)
        input_ids = torch.randint(0, 50000, (total,), dtype=torch.int64, device=device)
        buffer = torch.randint(
            0, 50000, (bs * buffer_size,), dtype=torch.int64, device=device
        )
        out_jit = torch.empty(total, dtype=torch.int64, device=device)

        assign_ngram_input_ids_draft_extend_after_decode(
            input_ids,
            buffer,
            out_jit,
            accept_length,
            gram_n,
            buffer_size,
            update_buffer=False,
        )
        torch.cuda.synchronize()

        out_ref = ref_assign_ngram_input_ids_draft_extend_after_decode(
            input_ids, buffer, accept_length, gram_n, buffer_size
        )
        self.assertTrue(
            torch.equal(out_jit.cpu(), out_ref.cpu()),
            f"Mismatch:\n  jit={out_jit}\n  ref={out_ref}",
        )

    def test_gram2_basic(self):
        # Mirrors the example at the bottom of test.py:
        #   draft_token_ids = [12, 13, 14, 15, 17]
        #   buffer = [1..8], accept = [2, 3], gram_n=2, buffer_size=4
        #   expected input_ids_gram = [4, 12, 8, 14, 15]
        device = torch.device("cuda")
        input_ids = torch.tensor([12, 13, 14, 15, 17], dtype=torch.int64, device=device)
        buffer = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device=device
        )
        accept_length = torch.tensor([2, 3], dtype=torch.int32, device=device)
        out_jit = torch.empty(5, dtype=torch.int64, device=device)
        assign_ngram_input_ids_draft_extend_after_decode(
            input_ids,
            buffer,
            out_jit,
            accept_length,
            gram_n=2,
            buffer_size=4,
            update_buffer=False,
        )
        torch.cuda.synchronize()
        expected = torch.tensor([4, 12, 8, 14, 15], dtype=torch.int64)
        self.assertTrue(torch.equal(out_jit.cpu(), expected))

    def test_gram3(self):
        self._run([3, 4, 2], gram_n=3, buffer_size=4, seed=1)

    def test_gram4(self):
        self._run([5, 1, 3, 4], gram_n=4, buffer_size=4, seed=2)

    def test_uniform(self):
        self._run([4] * 8, gram_n=2, buffer_size=4, seed=3)


if __name__ == "__main__":
    unittest.main()
