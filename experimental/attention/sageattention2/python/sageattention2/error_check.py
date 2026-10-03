"""
Input validation, error checking, and preprocessing for SageAttention2 on Ascend NPU.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F

def _load_cpp_constants():
    """从 C++ 唯一定义点直读数值常量（sageattn_fp8/ascendc/tiling.h）与默认档字符串
    （aclnn/sage_attn_proto.h）。

    本模块是 python 侧数值常量的唯一定义点：dispatch / golden 生成编排一律
    `from .error_check import ...` 引用，禁止再复制数值字面量。
    """
    _dir = os.path.dirname(__file__)
    _tiling_h = os.path.join(_dir, "..", "..", "sageattn_fp8", "ascendc", "tiling.h")
    _proto_h = os.path.join(_dir, "..", "..", "aclnn", "sage_attn_proto.h")

    if not (os.path.isfile(_tiling_h) and os.path.isfile(_proto_h)):
        # 降级兜底：当以纯 python 包部署在 site-packages 时，C++ 源码树不存在，
        # 返回与 tiling.h / proto.h 严格一致的标准常量
        return (
            8.807354922057604,
            2.25,
            448.0,
            127.0,
            1e-5,
            16,
            "HND",
            "per_thread",
            "fp32+fp16",
        )

    with open(_tiling_h, "r") as f:
        t_content = f.read()
    with open(_proto_h, "r") as f:
        p_content = f.read()

    def _float_const(name: str) -> float:
        return float(re.search(
            r"constexpr\s+float\s+" + name + r"\s*=\s*([0-9.eE+-]+)f", t_content).group(1))

    s_fp8 = _float_const("S_FP8_OFFSET")
    scale_max_fp16 = _float_const("SCALE_MAX_FP16_ACCUM")
    s_fp8_scale_max = _float_const("FP8_E4M3_MAX")
    int8_quant_max = _float_const("INT8_QUANT_MAX")
    scale_epsilon = _float_const("SCALE_EPSILON")
    k_step = int(re.search(
        r"constexpr\s+uint32_t\s+DEFAULT_K_STEP\s*=\s*(\d+)\s*;", t_content).group(1))

    layout = re.search(r'kDefaultLayout\s*=\s*"([^"]+)"', p_content).group(1)
    gran = re.search(r'kDefaultQkQuantGran\s*=\s*"([^"]+)"', p_content).group(1)
    accum = re.search(r'kDefaultPvAccumDtype\s*=\s*"([^"]+)"', p_content).group(1)

    return (s_fp8, scale_max_fp16, s_fp8_scale_max, int8_quant_max,
            scale_epsilon, k_step, layout, gran, accum)

(S_FP8_OFFSET, SCALE_MAX_FP16_ACCUM, S_FP8_SCALE_MAX, INT8_QUANT_MAX, SCALE_EPSILON,
 K_STEP, DEFAULT_TENSOR_LAYOUT, DEFAULT_QK_QUANT_GRAN, DEFAULT_PV_ACCUM_DTYPE) = _load_cpp_constants()

# exp2 域换算系数：golden 冻结截断精度（勿"修正"为全精度 1.4426950408889634——
# 位级对齐依赖；tiling.h 的 LOG2E 为全精度 C++ 常量，与本值为两个不同常量）
LOG2E = 1.44269504

# FP8 V MMA 布局 16-token 段内 permute（C++ 唯一定义 quant/quant_common.h
# V_PERM_PATTERN/V_PERM_INV_PATTERN；python 侧唯一定义点即此处）
V_FP8_PERM = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
V_FP8_INV_PERM = [0] * 16
for _i, _p in enumerate(V_FP8_PERM):
    V_FP8_INV_PERM[_p] = _i

VALID_LAYOUTS = {"HND", "NHD"}
VALID_QUANT_GRANS = {"per_warp", "per_thread"}
VALID_PV_ACCUM_DTYPES = {"fp32", "fp32+fp32", "fp32+fp16"}

def validate_common_parameters(
    tensor_layout: str,
    qk_quant_gran: str,
    pv_accum_dtype: str,
    kwargs: Optional[Dict[str, Any]] = None,
    attn_mask: Optional[Any] = None,
) -> None:
    if tensor_layout not in VALID_LAYOUTS:
        raise ValueError(f"Invalid tensor_layout '{tensor_layout}'")
    if qk_quant_gran not in VALID_QUANT_GRANS:
        raise ValueError(f"qk_quant_gran must be 'per_warp' or 'per_thread'")
    if pv_accum_dtype not in VALID_PV_ACCUM_DTYPES:
        raise ValueError(f"pv_accum_dtype must be one of {VALID_PV_ACCUM_DTYPES}")
    if attn_mask is not None:
        raise ValueError("attn_mask is not supported")
    if kwargs is not None and kwargs.get("attn_mask") is not None:
        raise ValueError("attn_mask is not supported")

def handle_smooth_v(pv_accum_dtype: str, smooth_v: bool) -> bool:
    if pv_accum_dtype in ("fp32+fp32", "fp32+fp16") and smooth_v:
        warnings.warn(f"smooth_v will be ignored when pv_accum_dtype='{pv_accum_dtype}'", UserWarning)
        return False
    return smooth_v

def validate_tensors(q: Any, k: Any, v: Any, tensor_layout: str, is_causal: bool) -> None:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4: raise ValueError("Input tensors q, k, v must all be 4D tensors")
    if tensor_layout == "HND":
        b_q, h_q, s_q, d_q = q.shape
        b_k, h_k, s_k, d_k = k.shape
        b_v, h_v, s_v, d_v = v.shape
    else:
        b_q, s_q, h_q, d_q = q.shape
        b_k, s_k, h_k, d_k = k.shape
        b_v, s_v, h_v, d_v = v.shape
    if b_q != b_k or b_q != b_v: raise ValueError("Batch size mismatch across tensors")
    if h_k != h_v or s_k != s_v or d_k != d_v: raise ValueError("Key and value tensor shapes must match")
    if d_q != d_k: raise ValueError("Head dimension mismatch")
    if d_q <= 0 or d_q > 128: raise ValueError("Unsupported head_dim")
    if h_q % h_k != 0: raise ValueError("GQA constraint violated")
    if is_causal and s_q != s_k: raise ValueError("is_causal=True requires qo_len == kv_len")
    if q.device != k.device or q.device != v.device: raise AssertionError("All tensors (q, k, v) must be on the same device")
    valid_dtypes = (torch.float16, torch.bfloat16)
    if q.dtype not in valid_dtypes: raise AssertionError("Input tensors must be in dtype of torch.float16 or torch.bfloat16")
    if q.dtype != k.dtype or q.dtype != v.dtype: raise AssertionError("All tensors (q, k, v) must have the same dtype")
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."


def pad_head_dim(q: Any, k: Any, v: Any) -> Tuple[Any, Any, Any, int]:
    head_dim_og = q.size(-1)
    if head_dim_og <= 0:
        raise ValueError("Invalid head_dim")
    if head_dim_og < 64:
        pad_size = 64 - head_dim_og
        q = F.pad(q, (0, pad_size))
        k = F.pad(k, (0, pad_size))
        v = F.pad(v, (0, pad_size))
    elif 64 < head_dim_og < 128:
        pad_size = 128 - head_dim_og
        q = F.pad(q, (0, pad_size))
        k = F.pad(k, (0, pad_size))
        v = F.pad(v, (0, pad_size))
    elif head_dim_og > 128:
        raise ValueError("Unsupported head_dim")

    return q, k, v, head_dim_og

def compute_sm_scale(head_dim_og: int, sm_scale: Optional[float]) -> float:
    if sm_scale is None:
        return 1.0 / math.sqrt(head_dim_og)
    scale = float(sm_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("sm_scale must be > 0")
    return scale
