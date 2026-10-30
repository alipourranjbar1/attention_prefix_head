#!/usr/bin/env python3
"""CPU mask/SDPA checks; requires PyTorch, no models or NPU required."""
import importlib.util
from pathlib import Path
import unittest

import torch
from torch.nn.attention.flex_attention import create_mask
from torch.nn.functional import scaled_dot_product_attention

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'dflash_mask_under_test', ROOT / 'src/speculators/models/dflash/attention.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def mask(causal=False, **kwargs):
    # Two packed documents, one padding position, and two independent anchors.
    mod, q_len, kv_len = module.create_anchor_block_mask_mod(
        document_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, -1]),
        total_seq_len=8, anchor_positions=torch.tensor([2, 5]),
        block_size=4, causal_block=causal, **kwargs)
    return create_mask(mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len,
                       device='cpu')


class CausalMaskTests(unittest.TestCase):
    def test_visibility(self):
        for causal in (False, True):
            actual = mask(causal)[0, 0]
            for q in range(8):
                block, offset = divmod(q, 4)
                expected = {0, 1} if block == 0 else {4}
                end = offset + 1 if causal else 4
                expected |= set(range(8 + block * 4, 8 + block * 4 + end))
                self.assertEqual(set(torch.where(actual[q])[0].tolist()), expected)

    def test_context_unchanged(self):
        self.assertTrue(torch.equal(mask(False)[..., :8], mask(True)[..., :8]))
        self.assertTrue(mask(True).any(dim=-1).all())

    def test_sliding_behavior_preserved(self):
        # A causal override must not accidentally re-enable future draft positions.
        actual = mask(True, sliding_window=1, sliding_window_non_causal=True)
        self.assertFalse(actual[0, 0, 0, 9])
        self.assertTrue(actual[0, 0, 0, 1])
        self.assertFalse(actual[0, 0, 0, 0])
        default = mask(False, sliding_window=1)
        self.assertTrue(torch.equal(default, actual))
        bidir = mask(False, sliding_window=1, sliding_window_non_causal=True)
        self.assertTrue(bidir[0, 0, 0, 9])

    def test_sdpa_future_is_blocked_and_backward_is_finite(self):
        torch.manual_seed(42)
        q = torch.randn(1, 2, 8, 8, requires_grad=True)
        k = torch.randn(1, 2, 16, 8, requires_grad=True)
        v = torch.randn(1, 2, 16, 8, requires_grad=True)
        causal = mask(True)
        out = scaled_dot_product_attention(q, k, v, attn_mask=causal)
        changed = v.detach().clone()
        changed[:, :, 11, :] += 10  # future position in the first block
        out_changed = scaled_dot_product_attention(q, k, changed, attn_mask=causal)
        torch.testing.assert_close(out[:, :, :3], out_changed[:, :, :3])
        self.assertFalse(torch.allclose(out[:, :, 3], out_changed[:, :, 3]))
        torch.testing.assert_close(out[:, :, 4:], out_changed[:, :, 4:])
        bidir = scaled_dot_product_attention(q, k, v, attn_mask=mask(False))
        bidir_changed = scaled_dot_product_attention(q, k, changed, attn_mask=mask(False))
        self.assertFalse(torch.allclose(bidir[:, :, 0], bidir_changed[:, :, 0]))
        out.square().mean().backward()
        for tensor in (q, k, v):
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == '__main__':
    unittest.main()
