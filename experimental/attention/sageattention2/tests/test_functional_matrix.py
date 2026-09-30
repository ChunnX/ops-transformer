import sys
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from sageattention2 import replace_sdpa, run
from sageattention2._ops import _PLAN_CACHE


class TestFunctionalMatrix(unittest.TestCase):
    def setUp(self):
        _PLAN_CACHE.clear()
        torch.manual_seed(42)

    def tearDown(self):
        _PLAN_CACHE.clear()

    def test_tc01_basic_functionality_and_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            for layout in ("HND", "NHD"):
                if layout == "HND":
                    q = torch.randn(1, 4, 128, 64, dtype=dtype)
                    k = torch.randn(1, 4, 128, 64, dtype=dtype)
                    v = torch.randn(1, 4, 128, 64, dtype=dtype)
                else:
                    q = torch.randn(1, 128, 4, 64, dtype=dtype)
                    k = torch.randn(1, 128, 4, 64, dtype=dtype)
                    v = torch.randn(1, 128, 4, 64, dtype=dtype)
                out = run(q, k, v, tensor_layout=layout)
                self.assertEqual(out.shape, q.shape)
                self.assertEqual(out.dtype, dtype)
                self.assertFalse(torch.isnan(out).any())

    def test_tc02_quant_gran_and_accum_dtypes(self):
        q = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        k = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        v = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        grans = ("per_thread", "per_warp")
        accums = ("fp32", "fp32+fp32", "fp32+fp16")
        for g in grans:
            for acc in accums:
                out = run(q, k, v, qk_quant_gran=g, pv_accum_dtype=acc)
                self.assertEqual(out.shape, q.shape)
                self.assertFalse(torch.isnan(out).any())

    def test_tc03_smooth_k_and_smooth_v_behavior(self):
        q = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        k = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        v = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        out_sk = run(q, k, v, smooth_k=True)
        out_nosk = run(q, k, v, smooth_k=False)
        self.assertEqual(out_sk.shape, q.shape)
        self.assertEqual(out_nosk.shape, q.shape)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            run(q, k, v, pv_accum_dtype="fp32+fp16", smooth_v=True)
            self.assertTrue(any("smooth_v" in str(w.message) for w in recorded))

    def test_tc04_lse_export_and_shapes(self):
        b, hq, sq, d = 2, 4, 128, 64
        q = torch.randn(b, hq, sq, d, dtype=torch.float16)
        k = torch.randn(b, hq, sq, d, dtype=torch.float16)
        v = torch.randn(b, hq, sq, d, dtype=torch.float16)
        out, lse = run(q, k, v, return_lse=True)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(lse.shape, (b, hq, sq))
        self.assertEqual(lse.dtype, torch.float32)
        self.assertFalse(torch.isnan(lse).any())

    def test_tc05_head_dim_auto_pad_and_crop(self):
        for raw_d in (48, 96):
            q = torch.randn(1, 2, 64, raw_d, dtype=torch.float16)
            k = torch.randn(1, 2, 64, raw_d, dtype=torch.float16)
            v = torch.randn(1, 2, 64, raw_d, dtype=torch.float16)
            out = run(q, k, v)
            self.assertEqual(out.shape, (1, 2, 64, raw_d))
            self.assertEqual(out.stride(-1), 1)

    def test_tc06_gqa_and_asymmetric_sequence_lengths(self):
        b, sq, sk, d = 1, 128, 256, 64
        hq, hk = 4, 2
        q = torch.randn(b, hq, sq, d, dtype=torch.float16)
        k = torch.randn(b, hk, sk, d, dtype=torch.float16)
        v = torch.randn(b, hk, sk, d, dtype=torch.float16)
        out = run(q, k, v, is_causal=False)
        self.assertEqual(out.shape, (b, hq, sq, d))
        self.assertFalse(torch.isnan(out).any())

    def test_tc07_causal_attention(self):
        b, h, s, d = 1, 2, 128, 64
        q = torch.randn(b, h, s, d, dtype=torch.float16)
        k = torch.randn(b, h, s, d, dtype=torch.float16)
        v = torch.randn(b, h, s, d, dtype=torch.float16)
        out = run(q, k, v, is_causal=True)
        self.assertEqual(out.shape, q.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_tc08_strict_exception_interception(self):
        q = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        k = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        v = torch.randn(1, 2, 64, 64, dtype=torch.float16)
        with self.assertRaises(ValueError):
            run(torch.randn(1, 2, 64, 160, dtype=torch.float16), k, v)
        with self.assertRaises(ValueError):
            run(q, k, v, is_causal=True, tensor_layout="HND")
            run(torch.randn(1, 2, 64, 64, dtype=torch.float16), torch.randn(1, 2, 128, 64, dtype=torch.float16), v, is_causal=True)
        with self.assertRaises(ValueError):
            run(q, k, v, attn_mask=torch.ones(1, 1, 64, 64))
        with self.assertRaises(ValueError):
            run(q, k, v, qk_quant_gran="per_tensor")
        with self.assertRaises(ValueError):
            run(q, k, v, pv_accum_dtype="int32")
        with self.assertRaises(ValueError):
            run(q, k, v, tensor_layout="BNSD")

    def test_tc09_long_sequence_handling(self):
        q = torch.randn(1, 1, 1024, 64, dtype=torch.float16)
        k = torch.randn(1, 1, 1024, 64, dtype=torch.float16)
        v = torch.randn(1, 1, 1024, 64, dtype=torch.float16)
        out = run(q, k, v)
        self.assertEqual(out.shape, q.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_tc10_sdpa_seamless_substitution(self):
        orig_sdpa = F.scaled_dot_product_attention
        try:
            replace_sdpa()
            self.assertNotEqual(F.scaled_dot_product_attention, orig_sdpa)
            q = torch.randn(1, 2, 64, 64, dtype=torch.float16)
            k = torch.randn(1, 2, 64, 64, dtype=torch.float16)
            v = torch.randn(1, 2, 64, 64, dtype=torch.float16)
            out = F.scaled_dot_product_attention(q, k, v)
            self.assertEqual(out.shape, q.shape)
        finally:
            F.scaled_dot_product_attention = orig_sdpa


if __name__ == "__main__":
    unittest.main()
