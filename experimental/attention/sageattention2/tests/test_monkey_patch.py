"""
Unit tests for SageAttention2 SDPA monkey-patch utility (P2-8).
Validates replace_sdpa, restore_sdpa, sdpa_patch context manager,
numerical consistency with direct sageattn calls, and fallback behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

current_file = Path(__file__).resolve()
OP_ROOT = current_file.parent.parent
REPO_ROOT = current_file.parents[5]

sys.path.insert(0, str(current_file.parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OP_ROOT) not in sys.path:
    sys.path.insert(0, str(OP_ROOT))
if str(OP_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(OP_ROOT / "python"))

from _golden_utils import make_qkv
from sageattention2 import (
    is_sdpa_patched,
    replace_sdpa,
    restore_sdpa,
    sdpa_patch,
    sageattn,
)
from sageattention2.monkey_patch import _ORIGINAL_SDPA


class TestSdpaMonkeyPatch(unittest.TestCase):
    """Test suite for SDPA monkey-patch hot replacement tools."""

    @classmethod
    def setUpClass(cls):
        os.environ["SAGEATTN_ALLOW_CPU_STUB"] = "1"

    def tearDown(self):
        # Always restore to clean unpatched state after each test
        restore_sdpa(_ORIGINAL_SDPA)

    _make_qkv = staticmethod(make_qkv)

    def test_replace_and_restore(self):
        """Verify replace_sdpa replaces F.scaled_dot_product_attention and restore_sdpa reverts it."""
        self.assertFalse(is_sdpa_patched())
        self.assertIs(F.scaled_dot_product_attention, _ORIGINAL_SDPA)

        prev = replace_sdpa()
        self.assertTrue(is_sdpa_patched())
        self.assertIsNot(F.scaled_dot_product_attention, _ORIGINAL_SDPA)
        self.assertIs(prev, _ORIGINAL_SDPA)

        restore_sdpa()
        self.assertFalse(is_sdpa_patched())
        self.assertIs(F.scaled_dot_product_attention, _ORIGINAL_SDPA)

    def test_context_manager_scope(self):
        """Verify sdpa_patch context manager scopes replacement and restores upon exit/exception."""
        self.assertFalse(is_sdpa_patched())

        with sdpa_patch():
            self.assertTrue(is_sdpa_patched())

        self.assertFalse(is_sdpa_patched())
        self.assertIs(F.scaled_dot_product_attention, _ORIGINAL_SDPA)

        # Exception safety
        try:
            with sdpa_patch():
                self.assertTrue(is_sdpa_patched())
                raise RuntimeError("Intentional error inside context")
        except RuntimeError:
            pass

        self.assertFalse(is_sdpa_patched())
        self.assertIs(F.scaled_dot_product_attention, _ORIGINAL_SDPA)

    def test_nested_context_manager(self):
        """Verify nested sdpa_patch context managers properly restore outer configurations."""
        self.assertFalse(is_sdpa_patched())

        with sdpa_patch(pv_accum_dtype="fp32"):
            self.assertTrue(is_sdpa_patched())
            with sdpa_patch(pv_accum_dtype="fp32+fp16"):
                self.assertTrue(is_sdpa_patched())
            # Outer context still active
            self.assertTrue(is_sdpa_patched())

        self.assertFalse(is_sdpa_patched())
        self.assertIs(F.scaled_dot_product_attention, _ORIGINAL_SDPA)

    def test_numerical_consistency(self):
        """Verify output of patched SDPA matches direct sageattn call and is close to PyTorch SDPA."""
        q, k, v = self._make_qkv(b=2, hq=4, hkv=4, s=64, d=64)

        expected = sageattn(q, k, v)

        with sdpa_patch():
            actual = F.scaled_dot_product_attention(q, k, v)

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

        # Check parity with original PyTorch SDPA
        ref = _ORIGINAL_SDPA(q.float(), k.float(), v.float()).half()
        max_diff = (actual.float() - ref.float()).abs().max().item()
        self.assertLess(max_diff, 0.15)

    def test_causal_and_scale_passthrough(self):
        """Verify is_causal and scale parameters are accurately routed to SageAttention2."""
        q, k, v = self._make_qkv(b=1, hq=2, hkv=2, s=64, d=64)
        custom_scale = 0.088

        expected = sageattn(q, k, v, is_causal=True, sm_scale=custom_scale)

        with sdpa_patch():
            actual = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=custom_scale)

        self.assertTrue(torch.equal(actual, expected))

    def test_gqa_passthrough(self):
        """Verify Grouped Query Attention (Hq % Hkv == 0) routes through patched SDPA cleanly."""
        q, k, v = self._make_qkv(b=1, hq=8, hkv=2, s=64, d=64)

        expected = sageattn(q, k, v)

        with sdpa_patch():
            actual = F.scaled_dot_product_attention(q, k, v)

        self.assertEqual(actual.shape, (1, 8, 64, 64))
        self.assertTrue(torch.equal(actual, expected))

    def test_fallback_to_original_when_unsupported(self):
        """Verify unsupported features (e.g. attn_mask, dropout > 0) fall back to original SDPA."""
        q, k, v = self._make_qkv(b=1, hq=2, hkv=2, s=32, d=64)
        mask = torch.ones(32, 32, dtype=torch.bool).tril()

        with sdpa_patch(fallback_to_original=True):
            # Non-None attn_mask triggers fallback to PyTorch SDPA without crashing
            out_mask = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            self.assertEqual(out_mask.shape, q.shape)

            # dropout_p > 0 triggers fallback
            out_drop = F.scaled_dot_product_attention(q, k, v, dropout_p=0.2)
            self.assertEqual(out_drop.shape, q.shape)

        # When fallback_to_original=False, must raise ValueError
        with sdpa_patch(fallback_to_original=False):
            with self.assertRaises(ValueError) as ctx:
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            self.assertIn("attn_mask is not supported", str(ctx.exception))

    def test_requires_grad_fallback_behavior(self):
        """Verify requires_grad=True triggers automatic fallback to PyTorch SDPA when fallback_to_original=True."""
        q = torch.randn(1, 2, 32, 64, dtype=torch.float32, requires_grad=True)
        k = torch.randn(1, 2, 32, 64, dtype=torch.float32, requires_grad=True)
        v = torch.randn(1, 2, 32, 64, dtype=torch.float32, requires_grad=True)

        with sdpa_patch(fallback_to_original=True):
            out = F.scaled_dot_product_attention(q, k, v)
            loss = out.sum()
            loss.backward()
            self.assertIsNotNone(q.grad)
            self.assertIsNotNone(k.grad)
            self.assertIsNotNone(v.grad)

        # When fallback_to_original=False, must raise ValueError
        q2 = torch.randn(1, 2, 32, 64, dtype=torch.float16, requires_grad=True)
        k2 = torch.randn(1, 2, 32, 64, dtype=torch.float16)
        v2 = torch.randn(1, 2, 32, 64, dtype=torch.float16)
        with sdpa_patch(fallback_to_original=False):
            with self.assertRaises(ValueError) as ctx:
                F.scaled_dot_product_attention(q2, k2, v2)
            self.assertIn("requires_grad", str(ctx.exception))

    def test_direct_sdpa_plug_and_play(self):
        """Verify direct replacement F.scaled_dot_product_attention = sageattn behaves accurately."""
        q, k, v = self._make_qkv(b=1, hq=2, hkv=2, s=32, d=64)
        orig_sdpa = F.scaled_dot_product_attention
        try:
            F.scaled_dot_product_attention = sageattn
            # Test with scale, is_causal, dropout_p=0.0, attn_mask=None
            custom_scale = 0.125
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=custom_scale
            )
            expected = sageattn(q, k, v, is_causal=True, sm_scale=custom_scale)
            self.assertTrue(torch.equal(out, expected))

            # Test dropout_p > 0 rejected
            with self.assertRaises(ValueError):
                F.scaled_dot_product_attention(q, k, v, dropout_p=0.5)

            # Test non-None attn_mask rejected
            with self.assertRaises(ValueError):
                F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(32, 32))
        finally:
            F.scaled_dot_product_attention = orig_sdpa


if __name__ == "__main__":
    unittest.main()
