import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from sageattention2 import run
from sageattention2.reference import attention as ref_attention


def compute_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item()


class TestPrecisionDualLevel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_la_precision_gate_fp16(self):
        b, h, s, d = 1, 2, 256, 64
        q = torch.randn(b, h, s, d, dtype=torch.float16)
        k = torch.randn(b, h, s, d, dtype=torch.float16)
        v = torch.randn(b, h, s, d, dtype=torch.float16)
        sm_scale = 1.0 / math.sqrt(d)

        out_cand = run(q, k, v, sm_scale=sm_scale, pv_accum_dtype="fp32+fp16")
        out_gold = ref_attention(q, k, v, sm_scale=sm_scale, pv_accum_dtype="fp32+fp16")

        diff = (out_cand.float() - out_gold.float()).abs()
        max_abs = diff.max().item()
        tol = 2.0 ** -9
        matched = (diff <= (tol + tol * out_gold.float().abs())).float().mean().item()

        self.assertLessEqual(max_abs, 0.1, f"FP16 max abs error {max_abs} exceeds 0.1")
        self.assertGreaterEqual(matched, 0.99, f"FP16 match ratio {matched} below 0.99")

    def test_la_precision_gate_bf16(self):
        b, h, s, d = 1, 2, 256, 64
        q = torch.randn(b, h, s, d, dtype=torch.bfloat16)
        k = torch.randn(b, h, s, d, dtype=torch.bfloat16)
        v = torch.randn(b, h, s, d, dtype=torch.bfloat16)
        sm_scale = 1.0 / math.sqrt(d)

        out_cand = run(q, k, v, sm_scale=sm_scale, pv_accum_dtype="fp32+fp16")
        out_gold = ref_attention(q, k, v, sm_scale=sm_scale, pv_accum_dtype="fp32+fp16")

        diff = (out_cand.float() - out_gold.float()).abs()
        max_abs = diff.max().item()
        tol = 2.0 ** -6
        matched = (diff <= (tol + tol * out_gold.float().abs())).float().mean().item()

        self.assertLessEqual(max_abs, 1.0, f"BF16 max abs error {max_abs} exceeds 1.0")
        self.assertGreaterEqual(matched, 0.99, f"BF16 match ratio {matched} below 0.99")

    def test_lb_precision_sdpa_error_ratio(self):
        b, h, s, d = 1, 2, 256, 64
        q = torch.randn(b, h, s, d, dtype=torch.float16)
        k = torch.randn(b, h, s, d, dtype=torch.float16)
        v = torch.randn(b, h, s, d, dtype=torch.float16)
        sm_scale = 1.0 / math.sqrt(d)

        out_fp32_sdpa = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), scale=sm_scale
        )
        out_npu = run(q, k, v, sm_scale=sm_scale, pv_accum_dtype="fp32+fp16").float()
        out_gpu = ref_attention(q, k, v, sm_scale=sm_scale, pv_accum_dtype="fp32+fp16").float()

        err_npu = (out_npu - out_fp32_sdpa).abs()
        err_gpu = (out_gpu - out_fp32_sdpa).abs()

        eps = 1e-7
        ratio_max = (err_npu.max() / (err_gpu.max() + eps)).item()
        ratio_mean = (err_npu.mean() / (err_gpu.mean() + eps)).item()

        rmse_npu = compute_rmse(out_npu, out_fp32_sdpa)
        rmse_gpu = compute_rmse(out_gpu, out_fp32_sdpa)
        ratio_rmse = rmse_npu / (rmse_gpu + eps)

        self.assertLessEqual(ratio_max, 2.0, f"Max error ratio {ratio_max} > 2.0")
        self.assertLessEqual(ratio_mean, 1.2, f"Mean error ratio {ratio_mean} > 1.2")
        self.assertLessEqual(ratio_rmse, 1.2, f"RMSE error ratio {ratio_rmse} > 1.2")


if __name__ == "__main__":
    unittest.main()
