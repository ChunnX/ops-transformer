"""宿主孪生执行器与 kernel 宿主串行分支 vs golden 的功能交叉验证（R9 数值门）。

被测对象（两条宿主数值路径，均从未在本文档之前对 golden 落盘数据执行过）：
1. CPU 孪生全流水执行器：sageattn_fp8/ascendc/twin_reference.{h,cpp} 的
   RunSequential（经 C-ABI SageAttnTwinRunForward）——量化 → QK^T（INT32 精确点积
   + 单点反量化）→ online softmax（448· 放大域逐块锚定）→ FP8 PV（两级累加）→
   归一化与 LSE 的完整数值链；
2. kernel 宿主串行分支：op_kernel/sageattention2.cpp 的 SageAttnKernel
   （SAGE_ATTN_DEVICE_COMPILE=0 路径，SimMemory 模拟 L1/UB/L0）——经 extern "C"
   sage_attention2 入口直驱，Host Tiling 同款 TilingData（core_num_used=1 + 关闭
   Die 亲和串行覆盖全部任务），验证 V0 量化 prologue / V1 softmax 与 PV 的核内
   数值路径与 GM 出入口（含 d_pad pad/crop、NHD 原生布局、smooth_k LSE 校正、
   smooth_v 的 V 减均值量化与 PV +vm）。

比对基线：golden/manifest.json 全部 25 桶的落盘 out/lse（CPU 高保真参考轨迹）。
容差口径（与 tests/test_golden_precision.py 的 L-A/L-B 一致）：
- L-A：matched_ratio >= 0.99（atol=rtol=2^-9 fp16 / 2^-6 bf16）、max_abs <= 0.1/1.0、
  LSE |diff| <= 1e-3、输出全部有限；
- L-B：对 out_sdpa_fp32 的 max/mean/rmse 误差比 <= 2.0/1.2/1.2（分母取 manifest metrics）。

覆盖范围与如实标注：
- 全流水孪生：25 桶整桶执行（含 P-01 主档 / TC-05_pad_d48 / TC-07_causal_per_warp
  三个代表桶的专属用例与量化中间态位级比对）；TC-03_smooth_v_fp32 经 smooth_v
  正路径（V 减均值量化 + PV +vm）整桶执行；
  P-05/P-07/TC-09 三个大桶（saved_shape 序列长 >= 512/1024/2048）按"非因果行的行级
  独立性"取 128 对齐行窗口执行（窗口内 Q 量化组按 32-token 组边界整组包含，与整桶
  golden 量化轨迹逐位同源），测试名以 row_window 如实标注；
- kernel 宿主串行分支：TC-05（pad d48/d96 补零+裁剪，覆盖 pad64/pad128 两域）、
  TC-07（causal + per_warp）、TC-01（NHD 原生布局 fp16/bf16）、TC-04（smooth_k LSE
  校正）、TC-03（smooth_v 正路径）、P-01 主档共 8 桶（宿主标量路径每任务重摆 KV，
  代价高于孪生，取代表性覆盖）。

lse_correction 喂法（开源契约）：km 与 corr = q·km_broadcast^T（fp32 matmul）均在
上层算好传入——孪生经 C-ABI lse_correction 形参、kernel 经 GM 入参 5，两条路径
消费同一 `_lse_correction` 产出（本文件唯一定义）。

运行方式：pytest tests/test_kernel_twin.py 或 python tests/test_kernel_twin.py；
动态库现场编译（首次约 1 s，按源码 mtime 缓存），无需 NPU 与预置产物。
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

current_file = Path(__file__).resolve()
OP_ROOT = current_file.parents[1]
sys.path.insert(0, str(current_file.parent))  # tests/ 内共享模块（tiling_mirror）
REPO_ROOT = next((p for p in current_file.parents if (p / "golden" / "manifest.json").exists()),
                 OP_ROOT.parent.parent.parent.parent)

# 与 tiling.h / quant_common.h 的编码唯一定义一致
# 编码镜像唯一定义点：tests/tiling_mirror.py（锚定 tiling.h 枚举）
from tiling_mirror import (  # noqa: E402
    ACCUM_CODE,
    ACCUM_FP32,
    ACCUM_FP32_FP16,
    GRAN_PER_THREAD,
    GRAN_PER_WARP,
    LAYOUT_HND,
    LAYOUT_NHD,
)
LOG2E = 1.4426950408889634
S_FP8_OFFSET = 8.807354922057604


def _lse_correction(q: torch.Tensor, k: torch.Tensor, hnd: bool) -> torch.Tensor:
    """smooth_k 行级 lse_correction（[b, hq, sq] fp32，开源契约 core.py:593-597）。"""
    if not hnd:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
    km = k.mean(dim=2, keepdim=True)
    q_per_kv = q.shape[1] // k.shape[1]
    if q_per_kv > 1:
        km = km.repeat_interleave(q_per_kv, dim=1)
    return torch.matmul(q.float(), km.float().transpose(2, 3)).squeeze(-1).contiguous()


def _effective_smooth_v(case: dict) -> bool:
    """smooth_v 生效判据（host tiling 同式：smoothV && pv_accum==fp32）。"""
    return bool(case.get("smooth_v", False)) and case.get("pv_accum_dtype", "fp32+fp16") == "fp32"
# 大桶行窗口（非因果行独立性；窗口起点 128 对齐保证 per_thread/per_warp Q 量化组
# 与整桶 golden 同组同值——QScaleIndex 以 128-token 块内相位寻址；K/V 按头切片）
TWIN_ROW_WINDOWS = {
    "P-05_per_thread_fp32+fp16_fp16_HND_gqa": [(0, 0, 64), (0, 384, 448)],
    "P-07_per_thread_fp32+fp16_fp16_HND": [(1, 0, 64), (1, 896, 960)],
    "TC-09_long_seq_2k": [(0, 0, 128), (0, 1920, 2048)],
}

_DYLIB_PATH = OP_ROOT / "build" / ("libsage_kernel_twin.so" if sys.platform.startswith("linux")
                                   else "libsage_kernel_twin.dylib")
_TWIN_SOURCES = [
    OP_ROOT / "op_kernel" / "sageattention2.cpp",
    OP_ROOT / "sageattn_fp8" / "ascendc" / "twin_reference.cpp",
    OP_ROOT / "quant" / "quant_int8.cpp",
    OP_ROOT / "quant" / "quant_fp8.cpp",
]


def _build_dylib() -> Path:
    """现场编译孪生 + kernel 宿主分支动态库（按源码 mtime 缓存）。"""
    if _DYLIB_PATH.exists() and not os.environ.get("SAGE_TWIN_FORCE_BUILD"):
        lib_mtime = _DYLIB_PATH.stat().st_mtime
        if all(src.stat().st_mtime <= lib_mtime for src in _TWIN_SOURCES):
            return _DYLIB_PATH
    _DYLIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.environ.get("CXX", "c++"), "-std=c++17", "-O3", "-shared", "-fPIC",
        *[str(s) for s in _TWIN_SOURCES],
        "-I", str(OP_ROOT / "sageattn_fp8" / "ascendc"),
        "-I", str(OP_ROOT / "sageattn_fp8" / "catlass"),
        "-I", str(OP_ROOT / "quant"),
        "-I", str(REPO_ROOT / "git" / "catlass" / "include"),
        "-o", str(_DYLIB_PATH),
    ]
    subprocess.check_call(cmd)
    return _DYLIB_PATH


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_build_dylib()))
    lib.SageAttnTwinRunForward.restype = ctypes.c_int
    lib.SageAttnTwinRunForward.argtypes = (
        [ctypes.c_void_p] * 7 + [ctypes.c_uint] * 12 + [ctypes.c_int, ctypes.c_float]
    )
    lib.SageAttnTwinQuantDump.restype = None
    lib.SageAttnTwinQuantDump.argtypes = (
        [ctypes.POINTER(ctypes.POINTER(ctypes.c_char))] * 6
        + [ctypes.POINTER(ctypes.c_uint)] * 2
    )
    lib.sage_attention2.restype = None
    lib.sage_attention2.argtypes = [ctypes.c_void_p] * 9
    return lib


# 镜像唯一定义点：tests/tiling_mirror.py（逐字段偏移由 tests/test_tiling_struct.py 锁定）
from tiling_mirror import SageAttention2TilingData

sys.path.insert(0, str(OP_ROOT / "python"))  # 算子包（scale_max 分档唯一定义点）
from sageattention2.dispatch import v_scale_max_for  # noqa: E402
from _golden_utils import (  # noqa: E402  （golden 加载与 L-A/L-B 判据唯一定义点）
    load_tensor,
    l_a_metrics as _l_a_metrics,
    l_b_ratios as _l_b_ratios,
)


_LIB = _load_lib()


# golden 加载与 L-A/L-B 判据唯一定义点：tests/_golden_utils.py（纯等价搬迁，
# 带进程内缓存以保持既有跨用例复用行为）
_TENSOR_CACHE = {}


def _load_tensor(case_id: str, name: str) -> "torch.Tensor":
    key = (case_id, name)
    if key not in _TENSOR_CACHE:
        _TENSOR_CACHE[key] = load_tensor(case_id, name)
    return _TENSOR_CACHE[key]


def _dtype_mode(t: torch.Tensor) -> int:
    if t.dtype == torch.float16:
        return 1
    if t.dtype == torch.bfloat16:
        return 2
    return 0


def _run_twin(case: dict, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """调孪生全流水执行器（零拷贝直传 torch 位型缓冲），返回 (out, lse)。"""
    shape = case["saved_shape"]
    b, hq, hkv = shape["batch_size"], shape["num_qo_heads"], shape["num_kv_heads"]
    sq, sk, d = shape["qo_len"], shape["kv_len"], shape["head_dim"]
    hnd = case.get("tensor_layout", "HND") == "HND"
    layout = LAYOUT_HND if hnd else LAYOUT_NHD
    gran = GRAN_PER_THREAD if case.get("qk_quant_gran", "per_thread") == "per_thread" else GRAN_PER_WARP
    accum = ACCUM_CODE[case.get("pv_accum_dtype", "fp32+fp16")]
    smooth_k = 1 if case.get("smooth_k", True) else 0
    smooth_v = 1 if _effective_smooth_v(case) else 0
    out = torch.empty_like(q)
    lse = torch.zeros(b * hq * sq, dtype=torch.float32)
    # smooth_k 校正项由上层算好传入（开源契约，与设备核 GM 入参同式）
    corr = _lse_correction(q, k, hnd) if smooth_k else None
    rc = _LIB.SageAttnTwinRunForward(
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()),
        None,
        ctypes.c_void_p(corr.data_ptr()) if corr is not None else None,
        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(lse.data_ptr()),
        b, hq, hkv, sq, sk, d, layout, gran, accum,
        1 if case.get("is_causal") else 0,
        smooth_k, smooth_v,
        _dtype_mode(q), 1.0 / math.sqrt(d),
    )
    assert rc == 0, f"twin refused config (rc={rc})"
    return out, lse.view(b, hq, sq)


def _run_kernel_host(case: dict, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """直驱 op_kernel/sageattention2.cpp 的宿主串行分支（SimMemory 标量基线）。

    入出口契约：Q/K/V/OUT 的 GM 视图按 d_pad 步长寻址（d_og < d_pad 时调用方补零
    pad、出参裁剪，与设备包装层职责一致）；key_mean 为 fp32 [b, hkv, d_pad]；
    lse_correction 为 fp32 [b, hq, sq]（smooth_k 时非空，核 V1 Finalize 按
    (b, hq, token) 全局下标直读——与设备 GM 入参同契约）；TilingData 与
    aclnn/sage_attn_host_tiling.h ComputeHostTiling 同式填充，
    core_num_used=1 + 关闭 Die 亲和使单核串行覆盖全部任务。
    """
    shape = case["saved_shape"]
    b, hq, hkv = shape["batch_size"], shape["num_qo_heads"], shape["num_kv_heads"]
    sq, sk, d_og = shape["qo_len"], shape["kv_len"], shape["head_dim"]
    d_pad = 64 if d_og <= 64 else 128
    hnd = case.get("tensor_layout", "HND") == "HND"
    layout = LAYOUT_HND if hnd else LAYOUT_NHD
    gran = GRAN_PER_THREAD if case.get("qk_quant_gran", "per_thread") == "per_thread" else GRAN_PER_WARP
    accum = ACCUM_CODE[case.get("pv_accum_dtype", "fp32+fp16")]
    causal = 1 if case.get("is_causal") else 0
    smooth_k = 1 if case.get("smooth_k", True) else 0
    smooth_v = 1 if _effective_smooth_v(case) else 0
    gqa = hq // hkv
    assert gqa == 1, "kernel host tier covers gqa=1 buckets (rows = gqa*64 <= 128)"

    def pad(t: torch.Tensor, heads: int, seq: int) -> torch.Tensor:
        if d_og == d_pad:
            return t.contiguous()
        if hnd:
            p = torch.zeros((b, heads, seq, d_pad), dtype=t.dtype)
        else:
            p = torch.zeros((b, seq, heads, d_pad), dtype=t.dtype)
        p[..., :d_og] = t
        return p.contiguous()

    qp, kp, vp = pad(q, hq, sq), pad(k, hkv, sk), pad(v, hkv, sk)
    km = None
    if smooth_k:
        km = kp.mean(dim=2 if hnd else 1).float().contiguous()
        corr = _lse_correction(qp, kp, hnd).contiguous().view(-1)
    else:
        corr = None
    out_pad = torch.zeros_like(qp)
    lse = torch.zeros(b * hq * sq, dtype=torch.float32)

    sm_scale = 1.0 / math.sqrt(d_og)
    t = SageAttention2TilingData()
    t.b, t.hq, t.hkv, t.sq, t.sk, t.d_og, t.d_pad = b, hq, hkv, sq, sk, d_og, d_pad
    t.is_causal, t.tensor_layout, t.qk_quant_gran, t.pv_accum_dtype = causal, layout, gran, accum
    t.smooth_k, t.smooth_v = smooth_k, smooth_v
    # 输入位型分派（tiling.h §8，ComputeHostTiling 同式）：核内 ReadInputBits/
    # km 减法按该字段走 fp16/bf16 分支，0 会被核内归一为 fp16
    t.input_dtype = _dtype_mode(q)
    t.sm_scale_log2e = sm_scale * LOG2E
    t.v_scale_max = v_scale_max_for(case.get("pv_accum_dtype", "fp32+fp16"))
    t.s_fp8_offset = S_FP8_OFFSET
    t.m_per_core, t.block_n, t.k_step = 64, 64, 16
    t.core_num_used = 1
    t.q_blk_num = (sq + t.m_per_core - 1) // t.m_per_core
    t.kv_blk_num = (sk + t.block_n - 1) // t.block_n
    t.split_kv_num, t.is_merge = 1, 0
    t.l1_q_offset = 0
    t.l1_q_scale_offset = 16 * 1024
    t.l1_kv_offset = 256 * 1024
    t.l1_kv_scale_offset = 256 * 1024 + 16 * 1024
    t.ub_acc_offset = 0
    t.ub_softmax_offset = 64 * 1024
    t.ub_quant_offset = 96 * 1024
    t.workspace_lse_offset, t.workspace_vm_offset = 0, 0
    t.enable_die_affinity, t.die_num = 0, 1
    t.cores_per_die, t.tasks_per_die = 1, b * hkv * t.q_blk_num

    km_ptr = ctypes.c_void_p(km.data_ptr()) if km is not None else None
    corr_ptr = ctypes.c_void_p(corr.data_ptr()) if corr is not None else None
    _LIB.sage_attention2(
        ctypes.c_void_p(qp.data_ptr()), ctypes.c_void_p(kp.data_ptr()), ctypes.c_void_p(vp.data_ptr()),
        km_ptr, corr_ptr,
        ctypes.c_void_p(out_pad.data_ptr()), ctypes.c_void_p(lse.data_ptr()),
        None, ctypes.cast(ctypes.byref(t), ctypes.c_void_p),
    )
    if d_og == d_pad:
        out = out_pad
    else:
        out = out_pad[..., :d_og].contiguous()
    return out, lse.view(b, hq, sq)


class GoldenTwinTestBase(unittest.TestCase):
    """公共断言：L-A/L-B 口径与 subTest 断言消息中的比对数字。"""

    @classmethod
    def setUpClass(cls):
        with open(REPO_ROOT / "golden" / "manifest.json", encoding="utf-8") as f:
            cls.manifest = json.load(f)["cases"]

    def assert_bucket(self, out: torch.Tensor, lse: torch.Tensor, case_id: str,
                      ref_out: torch.Tensor = None, ref_lse: torch.Tensor = None,
                      ref_sdpa: torch.Tensor = None):
        case = self.manifest[case_id]
        is_bf16 = case.get("dtype") in ("bfloat16", "bf16")
        ref_out = _load_tensor(case_id, "out") if ref_out is None else ref_out
        ref_lse = _load_tensor(case_id, "lse") if ref_lse is None else ref_lse

        m = _l_a_metrics(out, ref_out, is_bf16)
        self.assertTrue(torch.isfinite(out).all(), f"[{case_id}] non-finite output")
        self.assertGreaterEqual(
            m["matched_ratio"], 0.99,
            f"[{case_id}] L-A matched_ratio {m['matched_ratio']:.6f} < 0.99 "
            f"(atol={m['atol']:.6g}, max_abs={m['max_abs_error']:.6g})")
        self.assertLessEqual(
            m["max_abs_error"], m["max_abs_limit"],
            f"[{case_id}] L-A max_abs {m['max_abs_error']:.6g} > {m['max_abs_limit']}")
        lse_diff = (lse.float() - ref_lse.float()).abs().max().item()
        self.assertLessEqual(
            lse_diff, 1e-3, f"[{case_id}] LSE max diff {lse_diff:.3e} > 1e-3")

        sdpa = _load_tensor(case_id, "out_sdpa_fp32") if ref_sdpa is None else ref_sdpa
        r = _l_b_ratios(out, sdpa, case)
        self.assertLessEqual(r["ratio_max"], 2.0 + 1e-5,
                             f"[{case_id}] L-B ratio_max {r['ratio_max']:.4f} > 2.0")
        self.assertLessEqual(r["ratio_mean"], 1.2 + 1e-5,
                             f"[{case_id}] L-B ratio_mean {r['ratio_mean']:.4f} > 1.2")
        self.assertLessEqual(r["ratio_rmse"], 1.2 + 1e-5,
                             f"[{case_id}] L-B ratio_rmse {r['ratio_rmse']:.4f} > 1.2")
        return {**m, **r, "lse_max_diff": lse_diff}


class TestTwinFullPipelineVsGolden(GoldenTwinTestBase):
    """孪生全流水执行器（twin_reference RunSequential）vs golden。"""

    def _run_full_bucket(self, case_id: str) -> dict:
        q = _load_tensor(case_id, "q")
        k = _load_tensor(case_id, "k")
        v = _load_tensor(case_id, "v")
        out, lse = _run_twin(self.manifest[case_id], q, k, v)
        return self.assert_bucket(out, lse, case_id)

    def test_p01_main_bucket(self):
        """P-01 主档整桶：per_thread + fp32+fp16 + smooth_k，另附量化中间态位级比对。"""
        case_id = "P-01_per_thread_fp32+fp16_fp16_HND"
        stats = self._run_full_bucket(case_id)
        print(f"\n[twin] {case_id}: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e} "
              f"lb=({stats['ratio_max']:.3f}/{stats['ratio_mean']:.3f}/{stats['ratio_rmse']:.3f})")

        # 量化中间态位级比对（孪生经 quant 层唯一定义产出，与 golden 落盘逐字节对照）
        q = _load_tensor(case_id, "q")
        k = _load_tensor(case_id, "k")
        v = _load_tensor(case_id, "v")
        _run_twin(self.manifest[case_id], q, k, v)
        q8p, k8p, v8p, qsp, ksp, vsp = (ctypes.POINTER(ctypes.c_char)() for _ in range(6))
        d_pad, pad_sk_v = ctypes.c_uint(), ctypes.c_uint()
        _LIB.SageAttnTwinQuantDump(ctypes.byref(q8p), ctypes.byref(k8p), ctypes.byref(v8p),
                                   ctypes.byref(qsp), ctypes.byref(ksp), ctypes.byref(vsp),
                                   ctypes.byref(d_pad), ctypes.byref(pad_sk_v))
        shape = self.manifest[case_id]["saved_shape"]
        b, hq, hkv = shape["batch_size"], shape["num_qo_heads"], shape["num_kv_heads"]
        sq, sk = shape["qo_len"], shape["kv_len"]
        self.assertEqual(d_pad.value, 128)
        self.assertEqual(pad_sk_v.value, (sk + 63) // 64 * 64)
        my_q8 = torch.frombuffer(bytearray(ctypes.string_at(q8p, b * hq * sq * 128)),
                                 dtype=torch.int8).view(b, hq, sq, 128)
        my_k8 = torch.frombuffer(bytearray(ctypes.string_at(k8p, b * hkv * sk * 128)),
                                 dtype=torch.int8).view(b, hkv, sk, 128)
        my_v8 = torch.frombuffer(bytearray(ctypes.string_at(v8p, b * hkv * 128 * pad_sk_v.value)),
                                 dtype=torch.uint8).view(b, hkv, 128, pad_sk_v.value)
        self.assertTrue(torch.equal(my_q8, _load_tensor(case_id, "q_int8")), "q_int8 bit mismatch")
        self.assertTrue(torch.equal(my_k8, _load_tensor(case_id, "k_int8")), "k_int8 bit mismatch")
        self.assertTrue(torch.equal(my_v8, _load_tensor(case_id, "v_fp8").view(torch.uint8)),
                        "v_fp8 bit mismatch")
        for name, ptr, cnt in (("q_scale", qsp, b * hq * (sq // 128 * 32)),
                               ("k_scale", ksp, b * hkv * (sk // 64 * 4)),
                               ("v_scale", vsp, b * hkv * 128)):
            my = torch.frombuffer(bytearray(ctypes.string_at(ptr, cnt * 4)), dtype=torch.float32)
            ref = _load_tensor(case_id, name).float().reshape(-1)
            self.assertEqual(my.numel(), ref.numel(), f"{name} scale count mismatch")
            self.assertTrue(torch.allclose(my, ref, rtol=1e-5, atol=1e-6), f"{name} mismatch")

    def test_tc05_pad_d48_bucket(self):
        """TC-05_pad_d48 整桶：d48→d_pad64 补零与 d_og 裁剪全链路。"""
        stats = self._run_full_bucket("TC-05_pad_d48")
        print(f"\n[twin] TC-05_pad_d48: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e}")

    def test_tc07_causal_per_warp_bucket(self):
        """TC-07_causal_per_warp 整桶：causal 掩码 + per_warp RNE 量化。"""
        stats = self._run_full_bucket("TC-07_causal_per_warp")
        print(f"\n[twin] TC-07_causal_per_warp: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e}")

    def test_p_bucket_matrix(self):
        """P-02/P-03(causal)/P-04(batch2)/P-06(bf16) 整桶 L-A/L-B。"""
        for case_id in ("P-02_per_thread_fp32+fp16_fp16_HND",
                        "P-03_per_thread_fp32+fp16_fp16_HND_causal",
                        "P-04_per_thread_fp32+fp16_fp16_HND",
                        "P-06_per_thread_fp32+fp16_bf16_HND"):
            with self.subTest(case=case_id):
                self._run_full_bucket(case_id)

    def test_tc_bucket_matrix(self):
        """TC 行为矩阵整桶：NHD/bf16、per_warp 与三档累加、smooth_v 正路径、no_smooth_k、
        GQA、pad d80/d96、LSE。"""
        for case_id in ("TC-01_nhd_layout_fp16", "TC-01_nhd_layout_bf16",
                        "TC-02_per_thread_fp32", "TC-02_per_thread_fp32+fp32",
                        "TC-02_per_warp_fp32", "TC-02_per_warp_fp32+fp16", "TC-02_per_warp_fp32+fp32",
                        "TC-03_smooth_v_fp32", "TC-03_no_smooth_k", "TC-04_return_lse", "TC-04_return_lse_gqa",
                        "TC-05_pad_d80", "TC-05_pad_d96",
                        "TC-06_gqa_sq_neq_sk", "TC-10_sdpa_compat"):
            with self.subTest(case=case_id):
                self._run_full_bucket(case_id)

    def test_large_buckets_row_windows(self):
        """P-05/P-07/TC-09 大桶按 128 对齐行窗口执行（非因果行独立性，如实标注）。

        窗口内 Q 量化组按 32-token 组边界整组包含，K/V 全序列量化与整桶 golden
        同源，故窗口轨迹 == 整桶 golden 轨迹的对应行切片。
        """
        for case_id, windows in TWIN_ROW_WINDOWS.items():
            case = self.manifest[case_id]
            self.assertFalse(case.get("is_causal"), f"{case_id} 行窗口口径要求非因果")
            self.assertEqual(case.get("tensor_layout", "HND"), "HND",
                             f"{case_id} 行窗口口径当前仅覆盖 HND")
            q_full = _load_tensor(case_id, "q")
            k_full = _load_tensor(case_id, "k")
            v_full = _load_tensor(case_id, "v")
            ref_out = _load_tensor(case_id, "out")
            ref_lse = _load_tensor(case_id, "lse")
            shape = case["saved_shape"]
            gqa = shape["num_qo_heads"] // shape["num_kv_heads"]
            for head, w_beg, w_end in windows:
                with self.subTest(case=case_id, head=head, rows=f"{w_beg}:{w_end}"):
                    q_win = q_full[0:1, head:head + 1, w_beg:w_end].contiguous()
                    kh = head // gqa  # GQA：Q 头所属 KV 头，K/V 按头切片零拷贝
                    k_win = k_full[0:1, kh:kh + 1].contiguous()
                    v_win = v_full[0:1, kh:kh + 1].contiguous()
                    win_case = dict(case)
                    win_case["saved_shape"] = {**shape, "num_qo_heads": 1,
                                               "num_kv_heads": 1, "qo_len": w_end - w_beg}
                    out, lse = _run_twin(win_case, q_win, k_win, v_win)
                    stats = self.assert_bucket(
                        out, lse, case_id,
                        ref_out[0, head, w_beg:w_end].unsqueeze(0),
                        ref_lse[0, head, w_beg:w_end].unsqueeze(0),
                        _load_tensor(case_id, "out_sdpa_fp32")[0, head, w_beg:w_end].unsqueeze(0))
                    print(f"\n[twin row_window] {case_id}[h{head},{w_beg}:{w_end}]: "
                          f"matched={stats['matched_ratio']:.6f} max_abs={stats['max_abs_error']:.6g} "
                          f"lse={stats['lse_max_diff']:.3e}")

    def test_manifest_bucket_coverage(self):
        """门禁自检：25 桶全部被本文件覆盖（整桶或行窗口；smooth_v 正路径已入基线）。"""
        covered = {"P-01_per_thread_fp32+fp16_fp16_HND", "TC-05_pad_d48", "TC-07_causal_per_warp"}
        covered |= set(TWIN_ROW_WINDOWS)
        for case_id in self.manifest:
            if case_id.startswith("P-") and case_id not in covered:
                if case_id in ("P-02_per_thread_fp32+fp16_fp16_HND",
                               "P-03_per_thread_fp32+fp16_fp16_HND_causal",
                               "P-04_per_thread_fp32+fp16_fp16_HND",
                               "P-06_per_thread_fp32+fp16_bf16_HND"):
                    covered.add(case_id)
            elif case_id.startswith("TC-") and case_id not in covered:
                covered.add(case_id)
        missing = [c for c in self.manifest if c not in covered]
        self.assertEqual(missing, [], f"twin coverage hole: {missing}")


class TestKernelHostSerialBranchVsGolden(GoldenTwinTestBase):
    """kernel 宿主串行分支（op_kernel SageAttnKernel + SimMemory）vs golden。

    以 TilingData 直驱 extern "C" sage_attention2 入口（SAGE_ATTN_DEVICE_COMPILE=0
    路径），验证 V0 量化 prologue（含 km 减法与 smooth_v 的 vm 预扫）、V1 块内 logit/
    softmax/P 量化/PV 与 GM 出入口（lse_correction 按 GM 入参直读）；KV ping-pong 槽的
    宿主整段时序约束由 RestageKvBlockHost 保证，本测试即其 golden 级回归。
    """

    def _run_kernel_host_bucket(self, case_id: str) -> dict:
        q = _load_tensor(case_id, "q")
        k = _load_tensor(case_id, "k")
        v = _load_tensor(case_id, "v")
        out, lse = _run_kernel_host(self.manifest[case_id], q, k, v)
        return self.assert_bucket(out, lse, case_id)

    def test_tc03_smooth_v_kernel_host(self):
        """TC-03 smooth_v 正路径（fp32 档）：V 减均值量化 + PV 反量化 +vm 全链路。"""
        stats = self._run_kernel_host_bucket("TC-03_smooth_v_fp32")
        print(f"\n[kernel-host] TC-03_smooth_v_fp32: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e}")

    def test_tc05_pad_d48_kernel_host(self):
        stats = self._run_kernel_host_bucket("TC-05_pad_d48")
        print(f"\n[kernel-host] TC-05_pad_d48: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e}")

    def test_tc05_pad_d96_kernel_host(self):
        """TC-05_pad_d96 kernel 宿主分支：d96→d_pad128 补零与裁剪（pad128 域，
        与 d48 的 pad64 域互补；C++ 宿主仿真 tests/test_kernel_host_sim.cpp 的
        pad_d48/pad_d96 用例为同域位级判据）。"""
        stats = self._run_kernel_host_bucket("TC-05_pad_d96")
        print(f"\n[kernel-host] TC-05_pad_d96: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e}")

    def test_tc07_causal_per_warp_kernel_host(self):
        stats = self._run_kernel_host_bucket("TC-07_causal_per_warp")
        print(f"\n[kernel-host] TC-07_causal_per_warp: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e}")

    def test_tc01_nhd_kernel_host(self):
        stats = self._run_kernel_host_bucket("TC-01_nhd_layout_fp16")
        print(f"\n[kernel-host] TC-01_nhd_layout_fp16: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e}")

    def test_tc04_smooth_lse_kernel_host(self):
        stats = self._run_kernel_host_bucket("TC-04_return_lse")
        print(f"\n[kernel-host] TC-04_return_lse: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e}")

    def test_p01_main_bucket_kernel_host(self):
        stats = self._run_kernel_host_bucket("P-01_per_thread_fp32+fp16_fp16_HND")
        print(f"\n[kernel-host] P-01 主档: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e}")

    def test_tc01_nhd_bf16_kernel_host(self):
        """TC-01 NHD bf16 桶：核内 input_dtype==2 分支（bf16 读写与 km 差值位型舍回）。"""
        stats = self._run_kernel_host_bucket("TC-01_nhd_layout_bf16")
        print(f"\n[kernel-host] TC-01_nhd_layout_bf16: matched={stats['matched_ratio']:.6f} "
              f"max_abs={stats['max_abs_error']:.6g} lse={stats['lse_max_diff']:.3e}")


if __name__ == "__main__":
    unittest.main()
