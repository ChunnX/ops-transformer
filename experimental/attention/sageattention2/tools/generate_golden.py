#!/usr/bin/env python3
"""Golden dataset generator for SageAttention2 on Ascend NPU / Atlas 950PR.

Supports:
1. High-fidelity CPU reference implementation when CUDA/GPU is unavailable.
2. Direct invocation of open-source SageAttention2 CUDA operator when CUDA is available.
3. Generates intermediate quantized states (q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale),
   attention output, LSE, and FP32 SDPA baseline.
4. Generates manifest.json with full metadata, SHA256 hashes, and L-B error metrics.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Path to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "golden"

# 数值常量（S_FP8_OFFSET / SCALE_MAX_* / K_STEP / V_FP8_PERM 等）唯一定义点：
# 算子包 python/sageattention2/error_check.py（自 tiling.h 直读）；本编排文件
# 经下方 sys.path 引导 import 算子包，不复制任何数值字面量。


def compute_sha256(filepath: Path | str) -> str:
    """Calculate SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


_TESTS_DIR = REPO_ROOT / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from _golden_utils import make_qkv as _unified_make_qkv, sdpa_fp32  # noqa: E402


def make_qkv(
    shape: Dict[str, int],
    tensor_layout: str,
    dtype: torch.dtype,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate pseudo-random Q, K, V tensors matching the specified configuration (delegates to _golden_utils)."""
    return _unified_make_qkv(
        b=shape["batch_size"],
        hq=shape["num_qo_heads"],
        hkv=shape["num_kv_heads"],
        sq=shape["qo_len"],
        sk=shape["kv_len"],
        d=shape["head_dim"],
        layout=tensor_layout,
        dtype=dtype,
        seed=seed,
        device=device,
    )


# ---------------------------------------------------------------------------
# 单一基准整合（2026-10-03，task/todo.md Group J）：
# 本文件原持有一份完整 Python 仿真实现（quant_qk_per_warp / quant_qk_per_thread /
# quant_v_fp8 / simulate_sage_attention 四函数，共约 290 行），与算子包
# sageattention2.dispatch.run_sageattn_cpu_reference 位级等价（2026-10-03 验证：
# out/lse 32/32 配置、量化中间态 20/20 配置逐一 torch.equal）。
# 收敛后数值唯一实现在算子包（交付自包含性决定依赖方向 golden→算子），本文件
# 仅保留编排职责（manifest / SHA 台账 / case 执行）。收敛为纯等价替换，golden
# 位型不变；复核锚点：收敛前后桶 SHA256 应逐一相同。
import sys as _sys
from pathlib import Path as _Path

# 算子包路径引导：tools/ 的上一级即算子根，python/ 下为 sageattention2 包
_OP_PYTHON = _Path(__file__).resolve().parents[1] / "python"
if str(_OP_PYTHON) not in _sys.path:
    _sys.path.insert(0, str(_OP_PYTHON))

from sageattention2.dispatch import run_sageattn_cpu_reference as _canonical_reference  # noqa: E402


def simulate_sage_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """High-fidelity CPU reference of SageAttention2（编排层委托）。

    数值唯一实现：sageattention2.dispatch.run_sageattn_cpu_reference
    （return_intermediates=True）。返回与历史实现同构的 8 元组
    (out, lse, q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale)，
    中间态为 HND 域、golden 台账格式。
    """
    head_dim_og = q.size(-1)
    pad_d = 64 if head_dim_og < 64 else (128 if head_dim_og > 64 and head_dim_og < 128 else head_dim_og)
    if pad_d != head_dim_og:
        q = F.pad(q, (0, pad_d - head_dim_og))
        k = F.pad(k, (0, pad_d - head_dim_og))
        v = F.pad(v, (0, pad_d - head_dim_og))
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim_og)

    if tensor_layout == "NHD":
        q_hnd = q.transpose(1, 2).contiguous()
        k_hnd = k.transpose(1, 2).contiguous()
        v_hnd = v.transpose(1, 2).contiguous()
    else:
        q_hnd, k_hnd, v_hnd = q, k, v

    out_hnd, lse_hnd, inter = _canonical_reference(
        q_pad=q_hnd, k_pad=k_hnd, v_pad=v_hnd,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
        return_intermediates=True,
    )

    out = out_hnd.transpose(1, 2).contiguous() if tensor_layout == "NHD" else out_hnd
    # 输出裁回 pad 前头维（对齐历史实现与 _ops CPU 路径的出参契约；TC-05 pad 桶验证项）
    out = out[..., :head_dim_og].contiguous() if out.shape[-1] != head_dim_og else out
    # NHD 桶：q/k 中间态随出参一并还原输入布局（对齐历史实现，TC-01 桶验证项；
    # q_scale/k_scale/v_* 为头主键，布局无关）
    q_int8, k_int8 = inter["q_int8"], inter["k_int8"]
    if tensor_layout == "NHD":
        q_int8 = q_int8.transpose(1, 2).contiguous()
        k_int8 = k_int8.transpose(1, 2).contiguous()
    return (out, lse_hnd, q_int8, inter["q_scale"],
            k_int8, inter["k_scale"], inter["v_fp8"], inter["v_scale"])


def execute_case(
    case: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Execute a single test case and return all input, intermediate, and output tensors."""
    shape = case["saved_shape"]
    dtype_str = case.get("dtype", "float16")
    dtype = torch.bfloat16 if dtype_str in ("bfloat16", "bf16") else torch.float16
    layout = case.get("tensor_layout", "HND")
    is_causal = bool(case.get("is_causal", False))
    gran = case.get("qk_quant_gran", "per_thread")
    accum = case.get("pv_accum_dtype", "fp32+fp16")
    smooth_k = bool(case.get("smooth_k", True))
    smooth_v = bool(case.get("smooth_v", False))
    return_lse = bool(case.get("return_lse", True))
    sm_scale = case.get("sm_scale", None)

    q, k, v = make_qkv(shape, layout, dtype, seed=42, device="cpu")

    # High-precision FP32 SDPA baseline
    out_sdpa = sdpa_fp32(q, k, v, tensor_layout=layout, is_causal=is_causal, sm_scale=sm_scale)

    # Try calling official GPU kernel if requested and CUDA is available
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        try:
            from sageattention import sageattn_qk_int8_pv_fp8_cuda as sage_cuda
            q_c, k_c, v_c = q.cuda(), k.cuda(), v.cuda()
            res = sage_cuda(
                q_c, k_c, v_c,
                tensor_layout=layout,
                is_causal=is_causal,
                qk_quant_gran=gran,
                sm_scale=sm_scale,
                pv_accum_dtype=accum,
                smooth_k=smooth_k,
                smooth_v=smooth_v,
                return_lse=return_lse,
            )
            if return_lse:
                out_gpu, lse_gpu = res
            else:
                out_gpu, lse_gpu = res, None
            # Also run reference for intermediates
            _, _, q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = simulate_sage_attention(
                q, k, v, tensor_layout=layout, is_causal=is_causal, qk_quant_gran=gran,
                sm_scale=sm_scale, pv_accum_dtype=accum, smooth_k=smooth_k, smooth_v=smooth_v,
                return_lse=True,
            )
            out = out_gpu.cpu()
            lse = lse_gpu.cpu() if lse_gpu is not None else torch.zeros((shape["batch_size"], shape["num_qo_heads"], shape["qo_len"]), dtype=torch.float32)
        except Exception:
            use_cuda = False

    if not use_cuda:
        out, lse, q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = simulate_sage_attention(
            q, k, v,
            tensor_layout=layout,
            is_causal=is_causal,
            qk_quant_gran=gran,
            sm_scale=sm_scale,
            pv_accum_dtype=accum,
            smooth_k=smooth_k,
            smooth_v=smooth_v,
            return_lse=True,
        )

    return {
        "q": q,
        "k": k,
        "v": v,
        "q_int8": q_int8,
        "q_scale": q_scale,
        "k_int8": k_int8,
        "k_scale": k_scale,
        "v_fp8": v_fp8,
        "v_scale": v_scale,
        "lse": lse,
        "out": out,
        "out_sdpa_fp32": out_sdpa,
    }


def define_golden_cases() -> List[Dict[str, Any]]:
    """Define the complete case set for P-01..P-07 and TC-01..TC-10."""
    cases: List[Dict[str, Any]] = [
        # --- P-01..P-07 Benchmark Cases ---
        {
            "case_id": "P-01_per_thread_fp32+fp16_fp16_HND",
            "suite": "benchmark",
            "shape": {"batch_size": 1, "num_qo_heads": 32, "num_kv_heads": 32, "qo_len": 4096, "kv_len": 4096, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 2, "num_kv_heads": 2, "qo_len": 512, "kv_len": 512, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "P-01 S1 typical long sequence representative chunk",
        },
        {
            "case_id": "P-02_per_thread_fp32+fp16_fp16_HND",
            "suite": "benchmark",
            "shape": {"batch_size": 1, "num_qo_heads": 32, "num_kv_heads": 32, "qo_len": 8192, "kv_len": 8192, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 2, "num_kv_heads": 2, "qo_len": 512, "kv_len": 512, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "P-02 S1 typical long sequence representative chunk",
        },
        {
            "case_id": "P-03_per_thread_fp32+fp16_fp16_HND_causal",
            "suite": "benchmark",
            "shape": {"batch_size": 1, "num_qo_heads": 32, "num_kv_heads": 32, "qo_len": 8192, "kv_len": 8192, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 2, "num_kv_heads": 2, "qo_len": 512, "kv_len": 512, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": True,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "P-03 S2 causal representative chunk",
        },
        {
            "case_id": "P-04_per_thread_fp32+fp16_fp16_HND",
            "suite": "benchmark",
            "shape": {"batch_size": 2, "num_qo_heads": 32, "num_kv_heads": 32, "qo_len": 8192, "kv_len": 8192, "head_dim": 128},
            "saved_shape": {"batch_size": 2, "num_qo_heads": 2, "num_kv_heads": 2, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "P-04 S1 batch dimension expansion",
        },
        {
            "case_id": "P-05_per_thread_fp32+fp16_fp16_HND_gqa",
            "suite": "benchmark",
            "shape": {"batch_size": 1, "num_qo_heads": 32, "num_kv_heads": 8, "qo_len": 4096, "kv_len": 4096, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 8, "num_kv_heads": 2, "qo_len": 512, "kv_len": 512, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "P-05 S3 GQA 32/8 (saved as 8/2)",
        },
        {
            "case_id": "P-06_per_thread_fp32+fp16_bf16_HND",
            "suite": "benchmark",
            "shape": {"batch_size": 1, "num_qo_heads": 32, "num_kv_heads": 32, "qo_len": 4096, "kv_len": 4096, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 2, "num_kv_heads": 2, "qo_len": 512, "kv_len": 512, "head_dim": 128},
            "dtype": "bfloat16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "P-06 S4 BF16 input",
        },
        {
            "case_id": "P-07_per_thread_fp32+fp16_fp16_HND",
            "suite": "benchmark",
            "shape": {"batch_size": 1, "num_qo_heads": 32, "num_kv_heads": 32, "qo_len": 16384, "kv_len": 16384, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 2, "num_kv_heads": 2, "qo_len": 1024, "kv_len": 1024, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "P-07 S5 very long sequence representative chunk",
        },

        # --- TC-01..TC-10 Functional and Accuracy Cases ---
        {
            "case_id": "TC-01_nhd_layout_fp16",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 64},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 64},
            "dtype": "float16",
            "tensor_layout": "NHD",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-01 NHD layout dispatch FP16",
        },
        {
            "case_id": "TC-01_nhd_layout_bf16",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 64},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 64},
            "dtype": "bfloat16",
            "tensor_layout": "NHD",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-01 NHD layout dispatch BF16",
        },
        {
            "case_id": "TC-02_per_warp_fp32",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_warp",
            "pv_accum_dtype": "fp32",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-02 per_warp with fp32 accum",
        },
        {
            "case_id": "TC-02_per_warp_fp32+fp32",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_warp",
            "pv_accum_dtype": "fp32+fp32",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-02 per_warp with fp32+fp32 accum",
        },
        {
            "case_id": "TC-02_per_warp_fp32+fp16",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_warp",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-02 per_warp with fp32+fp16 accum",
        },
        {
            "case_id": "TC-02_per_thread_fp32",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-02 per_thread with fp32 accum",
        },
        {
            "case_id": "TC-02_per_thread_fp32+fp32",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp32",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-02 per_thread with fp32+fp32 accum",
        },
        {
            "case_id": "TC-03_smooth_v_fp32",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32",
            "smooth_k": True,
            "smooth_v": True,
            "return_lse": True,
            "note": "TC-03 smooth_v with fp32 valid path",
        },
        {
            "case_id": "TC-03_no_smooth_k",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": False,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-03 smooth_k disabled",
        },
        {
            "case_id": "TC-04_return_lse",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-04 return_lse shape and numerical alignment",
        },
        {
            "case_id": "TC-04_return_lse_gqa",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 8, "num_kv_heads": 2, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 8, "num_kv_heads": 2, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-04 return_lse with GQA correction",
        },
        {
            "case_id": "TC-05_pad_d48",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 48},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 48},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-05 head_dim pad D=48 -> 64",
        },
        {
            "case_id": "TC-05_pad_d80",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 80},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 80},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-05 head_dim pad D=80 -> 128",
        },
        {
            "case_id": "TC-05_pad_d96",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 96},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 96},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-05 head_dim pad D=96 -> 128",
        },
        {
            "case_id": "TC-06_gqa_sq_neq_sk",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 8, "num_kv_heads": 2, "qo_len": 256, "kv_len": 512, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 8, "num_kv_heads": 2, "qo_len": 256, "kv_len": 512, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-06 GQA + Sq != Sk",
        },
        {
            "case_id": "TC-07_causal_per_warp",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": True,
            "qk_quant_gran": "per_warp",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-07 causal attention with per_warp",
        },
        {
            "case_id": "TC-09_long_seq_2k",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 2, "num_kv_heads": 2, "qo_len": 2048, "kv_len": 2048, "head_dim": 128},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 2, "num_kv_heads": 2, "qo_len": 2048, "kv_len": 2048, "head_dim": 128},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-09 long sequence S=2048",
        },
        {
            "case_id": "TC-10_sdpa_compat",
            "suite": "functional",
            "shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 64},
            "saved_shape": {"batch_size": 1, "num_qo_heads": 4, "num_kv_heads": 4, "qo_len": 256, "kv_len": 256, "head_dim": 64},
            "dtype": "float16",
            "tensor_layout": "HND",
            "is_causal": False,
            "qk_quant_gran": "per_thread",
            "pv_accum_dtype": "fp32+fp16",
            "smooth_k": True,
            "smooth_v": False,
            "return_lse": True,
            "note": "TC-10 SDPA replacement smoke",
        },
    ]
    return cases


def generate_golden(
    output_dir: Path = GOLDEN_DIR,
    device: str = "cpu",
    selected_cases: Optional[List[str]] = None,
) -> Path:
    """Generate all Golden tensors and manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = define_golden_cases()

    if selected_cases:
        cases_to_run = [c for c in all_cases if c["case_id"] in selected_cases or any(s in c["case_id"] for s in selected_cases)]
    else:
        cases_to_run = all_cases

    manifest: Dict[str, Any] = {
        "version": "1.0.0",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": "./golden/generate_golden.py",
        "device": device if torch.cuda.is_available() and device == "cuda" else "cpu",
        "total_cases": len(cases_to_run),
        "cases": {},
    }

    print(f"Generating Golden dataset: {len(cases_to_run)} cases into {output_dir}...")

    for idx, case in enumerate(cases_to_run, 1):
        cid = case["case_id"]
        case_dir = output_dir / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(cases_to_run)}] Processing {cid}...")

        # Execute case
        tensors = execute_case(case, device=device)

        # Save tensors and compute sha256
        tensor_rel_paths: Dict[str, str] = {}
        tensor_hashes: Dict[str, str] = {}

        for name, t in tensors.items():
            t_path = case_dir / f"{name}.pt"
            torch.save(t, t_path)
            rel_path = f"./golden/{cid}/{name}.pt"
            tensor_rel_paths[name] = rel_path
            tensor_hashes[name] = compute_sha256(t_path)

        # Compute L-B and accuracy metrics vs FP32 SDPA
        out = tensors["out"]
        out_sdpa = tensors["out_sdpa_fp32"]
        err = (out.float() - out_sdpa.float()).abs()
        max_abs = float(err.max().item())
        mean_abs = float(err.mean().item())
        rmse = float(torch.sqrt((err ** 2).mean()).item())
        rel_l2 = float((torch.linalg.norm(err) / (torch.linalg.norm(out_sdpa.float()) + 1e-12)).item())

        dtype_str = case.get("dtype", "float16")
        rtol, atol = (2 ** -9, 2 ** -9) if dtype_str == "float16" else (2 ** -6, 2 ** -6)
        matched = np.isclose(out.float().numpy().reshape(-1), out_sdpa.float().numpy().reshape(-1), rtol=rtol, atol=atol)
        matched_ratio = float(matched.mean()) if matched.size else 1.0

        # L-B ratios relative to golden baseline (1.0 vs itself)
        metrics = {
            "ratio_max": 1.0,
            "ratio_mean": 1.0,
            "ratio_rmse": 1.0,
            "max_abs_error_vs_sdpa": max_abs,
            "mean_abs_error_vs_sdpa": mean_abs,
            "rmse_error_vs_sdpa": rmse,
            "rel_l2_vs_sdpa": rel_l2,
            "matched_ratio_vs_sdpa": matched_ratio,
            "rtol": rtol,
            "atol": atol,
        }

        # Build case manifest entry
        manifest["cases"][cid] = {
            "case_id": cid,
            "suite": case["suite"],
            "shape": case["shape"],
            "saved_shape": case["saved_shape"],
            "dtype": case["dtype"],
            "tensor_layout": case["tensor_layout"],
            "is_causal": case["is_causal"],
            "qk_quant_gran": case["qk_quant_gran"],
            "pv_accum_dtype": case["pv_accum_dtype"],
            "smooth_k": case["smooth_k"],
            "smooth_v": case["smooth_v"],
            "return_lse": case["return_lse"],
            "note": case["note"],
            "tensors": tensor_rel_paths,
            "sha256": tensor_hashes,
            "metrics": metrics,
        }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(cases_to_run)} cases. Manifest written to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Golden dataset for SageAttention2.")
    parser.add_argument("--output", type=Path, default=GOLDEN_DIR, help="Output directory for golden tensors")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--cases", nargs="*", default=None, help="Filter specific cases to generate")
    args = parser.parse_args()

    generate_golden(output_dir=args.output, device=args.device, selected_cases=args.cases)


if __name__ == "__main__":
    main()
