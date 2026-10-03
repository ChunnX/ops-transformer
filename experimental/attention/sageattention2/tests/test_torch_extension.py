"""Unit tests for PyTorch 2.x C++ extension bridge, custom op schema, FakeTensor and torch.compile."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

# 本文件位于 <算子目录>/tests/：算子包 python 目录即 ../python（自包含，
# 不依赖其他测试模块先执行注入 sys.path）
OPS_TRANSFORMER_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(OPS_TRANSFORMER_PYTHON) not in sys.path:
    sys.path.insert(0, str(OPS_TRANSFORMER_PYTHON))

import sageattention2 as sage2
from sageattention2 import _extension


class TestTorchExtension(unittest.TestCase):
    """Test PyTorch 2.x C++ extension bridge and dispatcher metadata."""

    def setUp(self):
        os.environ["SAGEATTN_ALLOW_CPU_STUB"] = "1"

    def test_meta_layout_hnd_and_nhd(self):
        """Verify Meta backend returns accurate shapes for both HND and NHD layouts."""
        for layout in ("HND", "NHD"):
            for dtype in (torch.float16, torch.bfloat16):
                for return_lse in (False, True):
                    shape_q = (2, 4, 33, 64) if layout == "HND" else (2, 33, 4, 64)
                    shape_k = (2, 2, 65, 64) if layout == "HND" else (2, 65, 2, 64)

                    q = torch.empty(shape_q, device="meta", dtype=dtype)
                    k = torch.empty(shape_k, device="meta", dtype=dtype)
                    v = torch.empty(shape_k, device="meta", dtype=dtype)

                    out, lse = torch.ops.sageattention2.forward(
                        q,
                        k,
                        v,
                        tensor_layout=layout,
                        qk_quant_gran="per_thread",
                        pv_accum_dtype="fp32+fp16",
                        return_lse=return_lse,
                    )

                    self.assertEqual(out.shape, q.shape)
                    self.assertEqual(out.dtype, q.dtype)
                    self.assertEqual(out.device.type, "meta")

                    if return_lse:
                        self.assertEqual(lse.shape, (2, 4, 33))
                        self.assertEqual(lse.dtype, torch.float32)
                        self.assertEqual(lse.device.type, "meta")
                    else:
                        self.assertIsNone(lse)

    def test_fake_tensor_does_not_compile_extension(self):
        """Verify FakeTensorMode uses the Meta backend and never triggers JIT extension compilation."""
        with patch.object(_extension, "load_extension", side_effect=RuntimeError("load_extension invoked unexpectedly!")) as mock_load:
            with FakeTensorMode():
                q = torch.empty(1, 4, 33, 64, dtype=torch.float16)
                k = torch.empty(1, 2, 65, 64, dtype=torch.float16)
                v = torch.empty(1, 2, 65, 64, dtype=torch.float16)

                out, lse = sage2.sageattn(
                    q, k, v, qk_quant_gran="per_thread", pv_accum_dtype="fp32+fp16", return_lse=True
                )
                self.assertEqual(out.shape, q.shape)
                self.assertEqual(lse.shape, (1, 4, 33))
                self.assertEqual(lse.dtype, torch.float32)

            mock_load.assert_not_called()

    def test_dispatcher_schema_and_opcheck(self):
        """Verify operator registration schema and FakeTensor contracts via torch.library.opcheck."""
        q = torch.randn(2, 4, 32, 64, dtype=torch.float16)
        k = torch.randn(2, 2, 32, 64, dtype=torch.float16)
        v = torch.randn(2, 2, 32, 64, dtype=torch.float16)

        # Validate with both return_lse True and False
        torch.library.opcheck(
            torch.ops.sageattention2.forward.default,
            (q, k, v),
            kwargs={"return_lse": True, "qk_quant_gran": "per_thread", "pv_accum_dtype": "fp32+fp16"},
            test_utils=("test_schema", "test_faketensor"),
        )
        torch.library.opcheck(
            torch.ops.sageattention2.forward.default,
            (q, k, v),
            kwargs={"return_lse": False, "qk_quant_gran": "per_warp", "pv_accum_dtype": "fp32"},
            test_utils=("test_schema", "test_faketensor"),
        )

    def test_torch_compile_fullgraph_eager_hnd_and_nhd(self):
        """Verify torch.compile(backend='eager', fullgraph=True) succeeds without graph breaks."""
        def attn_hnd(q, k, v):
            return sage2.sageattn(
                q, k, v, tensor_layout="HND", qk_quant_gran="per_thread", pv_accum_dtype="fp32+fp16", return_lse=True
            )

        def attn_nhd(q, k, v):
            return sage2.sageattn(
                q, k, v, tensor_layout="NHD", qk_quant_gran="per_warp", pv_accum_dtype="fp32", return_lse=True
            )

        compiled_hnd = torch.compile(attn_hnd, backend="eager", fullgraph=True)
        compiled_nhd = torch.compile(attn_nhd, backend="eager", fullgraph=True)

        # HND compilation run
        q_hnd = torch.randn(1, 4, 32, 64, dtype=torch.float16)
        k_hnd = torch.randn(1, 2, 32, 64, dtype=torch.float16)
        v_hnd = torch.randn(1, 2, 32, 64, dtype=torch.float16)
        out_hnd, lse_hnd = compiled_hnd(q_hnd, k_hnd, v_hnd)
        self.assertEqual(out_hnd.shape, (1, 4, 32, 64))
        self.assertEqual(lse_hnd.shape, (1, 4, 32))
        self.assertTrue(torch.isfinite(out_hnd).all())

        # NHD compilation run
        q_nhd = torch.randn(1, 32, 4, 64, dtype=torch.float16)
        k_nhd = torch.randn(1, 32, 2, 64, dtype=torch.float16)
        v_nhd = torch.randn(1, 32, 2, 64, dtype=torch.float16)
        out_nhd, lse_nhd = compiled_nhd(q_nhd, k_nhd, v_nhd)
        self.assertEqual(out_nhd.shape, (1, 32, 4, 64))
        self.assertEqual(lse_nhd.shape, (1, 4, 32))
        self.assertTrue(torch.isfinite(out_nhd).all())

    def test_real_parameters_passthrough_no_hardcoding(self):
        """Verify qk_quant_gran and pv_accum_dtype are preserved faithfully without hardcoding or downgrading."""
        q = torch.randn(1, 2, 16, 64, dtype=torch.float16)
        k = torch.randn(1, 2, 16, 64, dtype=torch.float16)
        v = torch.randn(1, 2, 16, 64, dtype=torch.float16)

        # Valid quantization granularities (per_thread, per_warp) must be accepted
        for gran in ("per_thread", "per_warp"):
            out = sage2.sageattn(q, k, v, qk_quant_gran=gran)
            self.assertEqual(out.shape, q.shape)

        # All 3 valid accumulation datatypes must be accepted
        for accum in ("fp32", "fp32+fp32", "fp32+fp16"):
            out = sage2.sageattn(q, k, v, pv_accum_dtype=accum)
            self.assertEqual(out.shape, q.shape)

        # Invalid granularity must be rejected
        with self.assertRaises(ValueError):
            sage2.sageattn(q, k, v, qk_quant_gran="per_token")
        with self.assertRaises(ValueError):
            sage2.sageattn(q, k, v, qk_quant_gran="invalid_gran")

        # Invalid accum dtype must be rejected
        with self.assertRaises(ValueError):
            sage2.sageattn(q, k, v, pv_accum_dtype="int32")


if __name__ == "__main__":
    unittest.main()
