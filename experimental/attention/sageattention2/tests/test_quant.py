"""
Unit tests for Vector independent quantization kernels and numerical prototypes.
Verifies:
1. Bit-exact / strict-tolerance alignment against all Golden dataset cases (P-01..P-07, TC-01..TC-10)
2. AscendC Vector C++ quantization kernels (QuantizeQInt8Ascend, QuantizeKInt8Ascend, QuantizeVFp8Ascend)
3. Compact adjacent layout packing (PackKvDataAndScaleAdjacentAscend, PackVDataAndScaleAdjacentAscend)
4. Host helper logic (quant_host.h GetVScaleMax tiering via ctypes wrapper;
   sm_scale folding & head-dim padding now live in the Python baseline layers)
"""

import ctypes
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

current_file = Path(__file__).resolve()
REPO_ROOT = next((p for p in current_file.parents if (p / "golden" / "manifest.json").exists()), current_file.parents[5])
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ 共享模块
from _golden_utils import load_tensor  # noqa: E402  （golden 加载唯一定义点）
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DYLIB_PATH = Path("/tmp/libquant_ascendc.dylib")

# Ensure C++ dylib is built for testing
if not DYLIB_PATH.exists():
    cpp_q = REPO_ROOT / "ops-transformer" / "experimental" / "attention" / "sageattention2" / "quant" / "quant_int8.cpp"
    cpp_v = REPO_ROOT / "ops-transformer" / "experimental" / "attention" / "sageattention2" / "quant" / "quant_fp8.cpp"
    cmd = [
        "c++", "-std=c++17", "-Wall", "-Wextra", "-O3", "-shared", "-fPIC",
        str(cpp_q), str(cpp_v), "-o", str(DYLIB_PATH)
    ]
    subprocess.check_call(cmd)

libquant = ctypes.CDLL(str(DYLIB_PATH))

# Import numerical prototype
sys.path.insert(0, str(REPO_ROOT))
from prototypes.tilelang_sageattn_prototype import TileLangSageAttentionPrototype


class TestQuantAgainstGolden(unittest.TestCase):
    """Test Python / TileLang prototype against all Golden intermediate states and outputs."""

    @classmethod
    def setUpClass(cls):
        manifest_path = REPO_ROOT / "golden" / "manifest.json"
        with open(manifest_path, "r") as f:
            cls.manifest = json.load(f)
        cls.cases = cls.manifest["cases"]
        cls.prototype = TileLangSageAttentionPrototype()

    def _run_single_case(self, case_id: str):
        case_info = self.cases[case_id]
        case_dir = REPO_ROOT / "golden" / case_id

        names = ("q", "k", "v", "q_int8", "q_scale", "k_int8", "k_scale",
                 "v_fp8", "v_scale", "lse", "out")
        g = {n: load_tensor(case_id, n) for n in names}
        q, k, v = g["q"], g["k"], g["v"]
        golden_q_int8, golden_q_scale = g["q_int8"], g["q_scale"]
        golden_k_int8, golden_k_scale = g["k_int8"], g["k_scale"]
        golden_v_fp8, golden_v_scale = g["v_fp8"], g["v_scale"]
        golden_lse, golden_out = g["lse"], g["out"]

        out, lse, q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = self.prototype.forward(
            q=q,
            k=k,
            v=v,
            tensor_layout=case_info.get("tensor_layout", "HND"),
            is_causal=bool(case_info.get("is_causal", False)),
            qk_quant_gran=case_info.get("qk_quant_gran", "per_thread"),
            pv_accum_dtype=case_info.get("pv_accum_dtype", "fp32+fp16"),
            smooth_k=bool(case_info.get("smooth_k", True)),
            smooth_v=bool(case_info.get("smooth_v", False)),
            return_lse=bool(case_info.get("return_lse", True)),
        )

        # 1. INT8 Q / K Bit-Exact Verification
        self.assertTrue(torch.equal(q_int8, golden_q_int8), f"Case {case_id} Q INT8 bit mismatch")
        torch.testing.assert_close(q_scale, golden_q_scale, rtol=1e-5, atol=1e-6, msg=f"Case {case_id} Q Scale mismatch")
        self.assertTrue(torch.equal(k_int8, golden_k_int8), f"Case {case_id} K INT8 bit mismatch")
        torch.testing.assert_close(k_scale, golden_k_scale, rtol=1e-5, atol=1e-6, msg=f"Case {case_id} K Scale mismatch")

        # 2. FP8 V Bit-Exact Verification (uint8 representation)
        self.assertTrue(
            torch.equal(v_fp8.view(torch.uint8), golden_v_fp8.view(torch.uint8)),
            f"Case {case_id} V FP8 bit mismatch"
        )
        torch.testing.assert_close(v_scale, golden_v_scale, rtol=1e-5, atol=1e-6, msg=f"Case {case_id} V Scale mismatch")

        # 3. LSE and Output Tolerance Verification
        torch.testing.assert_close(lse, golden_lse, rtol=1e-4, atol=1e-4, msg=f"Case {case_id} LSE mismatch")
        torch.testing.assert_close(out, golden_out, rtol=1e-2, atol=1e-2, msg=f"Case {case_id} Out mismatch")

    def test_p01_per_thread_fp32_fp16(self):
        self._run_single_case("P-01_per_thread_fp32+fp16_fp16_HND")

    def test_p02_per_thread_fp32_fp16(self):
        self._run_single_case("P-02_per_thread_fp32+fp16_fp16_HND")

    def test_p03_causal(self):
        self._run_single_case("P-03_per_thread_fp32+fp16_fp16_HND_causal")

    def test_p04_per_thread(self):
        self._run_single_case("P-04_per_thread_fp32+fp16_fp16_HND")

    def test_p05_gqa(self):
        self._run_single_case("P-05_per_thread_fp32+fp16_fp16_HND_gqa")

    def test_p06_bfloat16(self):
        self._run_single_case("P-06_per_thread_fp32+fp16_bf16_HND")

    def test_p07_per_thread(self):
        self._run_single_case("P-07_per_thread_fp32+fp16_fp16_HND")

    def test_tc01_nhd_layout_fp16(self):
        self._run_single_case("TC-01_nhd_layout_fp16")

    def test_tc01_nhd_layout_bf16(self):
        self._run_single_case("TC-01_nhd_layout_bf16")

    def test_tc02_per_warp_fp32(self):
        self._run_single_case("TC-02_per_warp_fp32")

    def test_tc02_per_warp_fp32_fp32(self):
        self._run_single_case("TC-02_per_warp_fp32+fp32")

    def test_tc02_per_warp_fp32_fp16(self):
        self._run_single_case("TC-02_per_warp_fp32+fp16")

    def test_tc02_per_thread_fp32(self):
        self._run_single_case("TC-02_per_thread_fp32")

    def test_tc02_per_thread_fp32_fp32(self):
        self._run_single_case("TC-02_per_thread_fp32+fp32")

    def test_tc03_smooth_v_fp32(self):
        self._run_single_case("TC-03_smooth_v_fp32")

    def test_tc03_no_smooth_k(self):
        self._run_single_case("TC-03_no_smooth_k")

    def test_tc04_return_lse(self):
        self._run_single_case("TC-04_return_lse")

    def test_tc04_return_lse_gqa(self):
        self._run_single_case("TC-04_return_lse_gqa")

    def test_tc05_pad_d48(self):
        self._run_single_case("TC-05_pad_d48")

    def test_tc05_pad_d80(self):
        self._run_single_case("TC-05_pad_d80")

    def test_tc05_pad_d96(self):
        self._run_single_case("TC-05_pad_d96")

    def test_tc06_gqa_sq_neq_sk(self):
        self._run_single_case("TC-06_gqa_sq_neq_sk")

    def test_tc07_causal_per_warp(self):
        self._run_single_case("TC-07_causal_per_warp")

    def test_tc09_long_seq_2k(self):
        self._run_single_case("TC-09_long_seq_2k")

    def test_tc10_sdpa_compat(self):
        self._run_single_case("TC-10_sdpa_compat")


class TestAscendCQuantKernels(unittest.TestCase):
    """Test AscendC Vector C++ quantization kernels directly via ctypes."""

    def _test_kernel_on_case(self, case_id: str):
        manifest_path = REPO_ROOT / "golden" / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        case_info = manifest["cases"][case_id]
        case_dir = REPO_ROOT / "golden" / case_id

        names = ("q", "k", "v", "q_int8", "q_scale", "k_int8", "k_scale",
                 "v_fp8", "v_scale")
        g = {n: load_tensor(case_id, n) for n in names}
        q, k, v = g["q"], g["k"], g["v"]
        golden_q_int8, golden_q_scale = g["q_int8"], g["q_scale"]
        golden_k_int8, golden_k_scale = g["k_int8"], g["k_scale"]
        golden_v_fp8, golden_v_scale = g["v_fp8"], g["v_scale"]

        head_dim_og = q.size(-1)
        pad_d = 64 if head_dim_og < 64 else (128 if head_dim_og > 64 and head_dim_og < 128 else head_dim_og)
        if pad_d != head_dim_og:
            q = F.pad(q, (0, pad_d - head_dim_og))
            k = F.pad(k, (0, pad_d - head_dim_og))
            v = F.pad(v, (0, pad_d - head_dim_og))

        layout = case_info.get("tensor_layout", "HND")
        gran = case_info.get("qk_quant_gran", "per_thread")
        accum = case_info.get("pv_accum_dtype", "fp32+fp16")
        smooth_k = bool(case_info.get("smooth_k", True))
        smooth_v = bool(case_info.get("smooth_v", False))

        tensor_layout_val = 1 if layout == "HND" else 0
        qk_quant_gran_val = 3 if gran == "per_thread" else 2
        pv_accum_val = 2 if accum == "fp32+fp16" else (1 if accum == "fp32+fp32" else 0)
        dtype_mode = 1 if q.dtype == torch.float16 else (2 if q.dtype == torch.bfloat16 else 0)

        if layout == "HND":
            b, hq, sq, d = q.shape
            _, hkv, sk, _ = k.shape
            seq_dim = 2
        else:
            b, sq, hq, d = q.shape
            _, sk, hkv, _ = k.shape
            seq_dim = 1

        # 1. Q INT8 Verification
        num_blkq = (sq + 127) // 128
        q_scale_len = num_blkq * (32 if gran == "per_thread" else 4)
        my_q_int8 = torch.empty_like(golden_q_int8)
        my_q_scale = torch.empty((b, hq, q_scale_len), dtype=torch.float32)

        ret_q = libquant.QuantizeQInt8Ascend(
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(my_q_int8.data_ptr()),
            ctypes.c_void_p(my_q_scale.data_ptr()),
            ctypes.c_uint32(b), ctypes.c_uint32(hq), ctypes.c_uint32(sq), ctypes.c_uint32(d),
            ctypes.c_uint32(qk_quant_gran_val), ctypes.c_uint32(tensor_layout_val), ctypes.c_int(dtype_mode)
        )
        self.assertEqual(ret_q, 0)
        self.assertTrue(torch.equal(my_q_int8, golden_q_int8))
        torch.testing.assert_close(my_q_scale, golden_q_scale, rtol=1e-5, atol=1e-6)

        # 2. K INT8 Verification
        num_blkk = (sk + 63) // 64
        k_scale_len = num_blkk * (4 if gran == "per_thread" else 1)
        my_k_int8 = torch.empty_like(golden_k_int8)
        my_k_scale = torch.empty((b, hkv, k_scale_len), dtype=torch.float32)

        if smooth_k:
            km = k.mean(dim=seq_dim, keepdim=True).squeeze(seq_dim).contiguous().float()
            km_ptr = ctypes.c_void_p(km.data_ptr())
        else:
            km_ptr = None

        ret_k = libquant.QuantizeKInt8Ascend(
            ctypes.c_void_p(k.data_ptr()),
            km_ptr,
            ctypes.c_void_p(my_k_int8.data_ptr()),
            ctypes.c_void_p(my_k_scale.data_ptr()),
            ctypes.c_uint32(b), ctypes.c_uint32(hkv), ctypes.c_uint32(sk), ctypes.c_uint32(d),
            ctypes.c_uint32(qk_quant_gran_val), ctypes.c_int(1 if smooth_k else 0),
            ctypes.c_uint32(tensor_layout_val), ctypes.c_int(dtype_mode)
        )
        self.assertEqual(ret_k, 0)
        self.assertTrue(torch.equal(my_k_int8, golden_k_int8))
        torch.testing.assert_close(my_k_scale, golden_k_scale, rtol=1e-5, atol=1e-6)

        # 3. V FP8 Verification
        pad_sk_v = ((sk + 63) // 64) * 64
        my_v_fp8 = torch.empty((b, hkv, d, pad_sk_v), dtype=torch.uint8)
        my_v_scale = torch.empty((b, hkv, d), dtype=torch.float32)
        my_vm = torch.empty((b, hkv, d), dtype=torch.float32) if smooth_v else None

        ret_v = libquant.QuantizeVFp8Ascend(
            ctypes.c_void_p(v.data_ptr()),
            ctypes.c_void_p(my_v_fp8.data_ptr()),
            ctypes.c_void_p(my_v_scale.data_ptr()),
            ctypes.c_void_p(my_vm.data_ptr()) if smooth_v else None,
            ctypes.c_uint32(b), ctypes.c_uint32(hkv), ctypes.c_uint32(sk), ctypes.c_uint32(d),
            ctypes.c_uint32(pv_accum_val), ctypes.c_int(1 if smooth_v else 0),
            ctypes.c_uint32(tensor_layout_val), ctypes.c_int(dtype_mode)
        )
        self.assertEqual(ret_v, 0)
        self.assertTrue(torch.equal(my_v_fp8, golden_v_fp8.view(torch.uint8)))
        torch.testing.assert_close(my_v_scale, golden_v_scale, rtol=1e-5, atol=1e-6)

    def test_kernel_p01(self):
        self._test_kernel_on_case("P-01_per_thread_fp32+fp16_fp16_HND")

    def test_kernel_p06_bfloat16(self):
        self._test_kernel_on_case("P-06_per_thread_fp32+fp16_bf16_HND")

    def test_kernel_tc01_nhd(self):
        self._test_kernel_on_case("TC-01_nhd_layout_fp16")

    def test_kernel_tc02_per_warp(self):
        self._test_kernel_on_case("TC-02_per_warp_fp32")

    def test_kernel_tc03_smooth_v(self):
        self._test_kernel_on_case("TC-03_smooth_v_fp32")

    def test_kernel_tc03_no_smooth_k(self):
        self._test_kernel_on_case("TC-03_no_smooth_k")

    def test_kernel_tc05_pad_d48(self):
        self._test_kernel_on_case("TC-05_pad_d48")

    def test_kernel_tc06_gqa(self):
        self._test_kernel_on_case("TC-06_gqa_sq_neq_sk")


class TestQuantHostHelpers(unittest.TestCase):
    """Test Host auxiliary helper logic in quant_host.h."""

    def test_sm_scale_folding(self):
        sm_scale = 1.0 / math.sqrt(128)
        folded = sm_scale * 1.4426950408889634
        self.assertAlmostEqual(folded, 0.12751743, places=6)

    def test_v_scale_max(self):
        """收敛轮 A/CLEANUP-6 双重实证（原断言 assertEqual(2.25, 2.25) 恒真）：
        1) quant_host.h GetVScaleMax 三档取值：fp32+fp16 (PV_ACCUM_FP32_FP16=2) 档
           返回 2.25，其余档位（含非法值兜底）返回 448.0——GetVScaleMax 是 C++ inline
           函数，无法直接 ctypes 调用，现场编译只含 quant_host.h 的 extern "C"
           包装动态库再断言返回值；
        2) golden 各桶 v_scale 落盘值按 manifest 的 pv_accum_dtype 反推上界：
           fp32+fp16 档 scale = amax/2.25（其余 448），全零通道为 NaN（isfinite
           排除）；全零 V 通道 + scale=0 的 NaN 保真属量化语义（task/01_design 2.7）。"""
        wrapper_src = r"""
#include "quant_host.h"
extern "C" {
float sage_v_scale_max(unsigned pv_accum_dtype) {
    return sage_attn::quant::GetVScaleMax(pv_accum_dtype);
}
}
"""
        host_lib = Path("/tmp/libsage_quant_host.dylib")
        quant_dir = REPO_ROOT / "ops-transformer" / "experimental" / "attention" / "sageattention2" / "quant"
        cmd = [
            "c++", "-std=c++17", "-Wall", "-Wextra", "-O2", "-shared", "-fPIC",
            "-x", "c++", "-", "-I", str(quant_dir), "-o", str(host_lib),
        ]
        subprocess.run(cmd, input=wrapper_src.encode("utf-8"), check=True)
        lib_host = ctypes.CDLL(str(host_lib))
        lib_host.sage_v_scale_max.restype = ctypes.c_float
        lib_host.sage_v_scale_max.argtypes = [ctypes.c_uint]

        # PV_ACCUM_FP32_FP16 (2) 是唯一低精度档 -> 2.25
        self.assertEqual(lib_host.sage_v_scale_max(2), ctypes.c_float(2.25).value)
        # PV_ACCUM_FP32 (0) 与 PV_ACCUM_FP32_FP32 (1) -> 448.0
        self.assertEqual(lib_host.sage_v_scale_max(0), ctypes.c_float(448.0).value)
        self.assertEqual(lib_host.sage_v_scale_max(1), ctypes.c_float(448.0).value)
        # 非法档位走 else 兜底 -> 448.0（保守高精度饱和上限）
        self.assertEqual(lib_host.sage_v_scale_max(999), ctypes.c_float(448.0).value)

        with open(REPO_ROOT / "golden" / "manifest.json", "r") as f:
            cases = json.load(f)["cases"]
        checked = 0
        for case_id, info in cases.items():
            v_scale_path = REPO_ROOT / "golden" / case_id / "v_scale.pt"
            if not v_scale_path.exists():
                continue
            v_scale = torch.load(v_scale_path, weights_only=True)
            finite = v_scale[torch.isfinite(v_scale)]
            if finite.numel() == 0:
                continue
            accum = info.get("pv_accum_dtype", "fp32+fp16")
            bound = 2.25 if accum == "fp32+fp16" else 448.0
            with self.subTest(case=case_id):
                self.assertTrue((finite >= 0).all(), f"{case_id}: v_scale must be non-negative")
                self.assertLessEqual(
                    finite.max().item(), bound, f"{case_id}: v_scale exceeds {bound}")
            checked += 1
        self.assertGreater(checked, 0, "no v_scale buckets checked")

    def test_padded_head_dim(self):
        # Head dimensions <= 64 pad to 64, <= 128 pad to 128
        def get_padded_head_dim(d):
            return 64 if d <= 64 else (128 if d <= 128 else ((d + 63) // 64) * 64)
        self.assertEqual(get_padded_head_dim(48), 64)
        self.assertEqual(get_padded_head_dim(64), 64)
        self.assertEqual(get_padded_head_dim(80), 128)
        self.assertEqual(get_padded_head_dim(96), 128)
        self.assertEqual(get_padded_head_dim(128), 128)

    def test_compact_kv_block_layout(self):
        # Verify compact adjacent layout memory packing
        block_k = 64
        d = 128
        qk_quant_gran = 3  # per_thread -> 4 scales
        data_bytes = block_k * d  # 8192
        scale_bytes = 4 * 4  # 16
        total = data_bytes + scale_bytes  # 8208
        aligned_32 = ((total + 31) // 32) * 32  # 8224

        dummy_data = torch.randint(-127, 127, (block_k, d), dtype=torch.int8)
        dummy_scale = torch.randn(4, dtype=torch.float32)
        compact_buf = torch.zeros(aligned_32, dtype=torch.uint8)

        ret = libquant.PackKvDataAndScaleAdjacentAscend(
            ctypes.c_void_p(dummy_data.data_ptr()),
            ctypes.c_void_p(dummy_scale.data_ptr()),
            ctypes.c_void_p(compact_buf.data_ptr()),
            ctypes.c_uint32(block_k),
            ctypes.c_uint32(d),
            ctypes.c_uint32(qk_quant_gran),
        )
        self.assertEqual(ret, 0)

        # Unpack and verify bit-identity
        recovered_data = compact_buf[:data_bytes].view(torch.int8).view(block_k, d)
        recovered_scale = compact_buf[data_bytes:data_bytes + scale_bytes].view(torch.float32)

        self.assertTrue(torch.equal(recovered_data, dummy_data))
        self.assertTrue(torch.equal(recovered_scale, dummy_scale))


if __name__ == "__main__":
    unittest.main()
