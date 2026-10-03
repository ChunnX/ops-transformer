"""
Input validation, error checking, and preprocessing for SageAttention2 on Ascend NPU.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Optional, Tuple
import warnings

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock()
    F = MagicMock()

def _load_cpp_constants():
    _dir = os.path.dirname(__file__)
    _tiling_h = os.path.join(_dir, "..", "..", "sageattn_fp8", "ascendc", "tiling.h")
    _proto_h = os.path.join(_dir, "..", "..", "aclnn", "sage_attn_proto.h")
    
    with open(_tiling_h, "r") as f:
        t_content = f.read()
    with open(_proto_h, "r") as f:
        p_content = f.read()
        
    s_fp8 = float(re.search(r"constexpr\s+float\s+S_FP8_OFFSET\s*=\s*([0-9.]+)f", t_content).group(1))
    scale_max = float(re.search(r"constexpr\s+float\s+SCALE_MAX_FP16_ACCUM\s*=\s*([0-9.]+)f", t_content).group(1))
    
    layout = re.search(r'kDefaultLayout\s*=\s*"([^"]+)"', p_content).group(1)
    gran = re.search(r'kDefaultQkQuantGran\s*=\s*"([^"]+)"', p_content).group(1)
    accum = re.search(r'kDefaultPvAccumDtype\s*=\s*"([^"]+)"', p_content).group(1)
    
    return s_fp8, scale_max, layout, gran, accum

S_FP8_OFFSET, SCALE_MAX_FP16_ACCUM, DEFAULT_TENSOR_LAYOUT, DEFAULT_QK_QUANT_GRAN, DEFAULT_PV_ACCUM_DTYPE = _load_cpp_constants()

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
    if not hasattr(q, "dim") or not hasattr(k, "dim") or not hasattr(v, "dim"): return
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
    if h_q % h_k != 0: raise ValueError("GQA constraint violated")
    if is_causal and s_q != s_k: raise ValueError("is_causal=True requires qo_len == kv_len")
    if hasattr(q, "device"):
        if q.device != k.device or q.device != v.device: raise AssertionError("All tensors (q, k, v) must be on the same device")
    if hasattr(q, "dtype"):
        valid_dtypes = (torch.float16, torch.bfloat16)
        if q.dtype not in valid_dtypes: raise AssertionError("Input tensors must be in dtype of torch.float16 or torch.bfloat16")
        if q.dtype != k.dtype or q.dtype != v.dtype: raise AssertionError("All tensors (q, k, v) must have the same dtype")
    if hasattr(q, "stride") and hasattr(k, "stride") and hasattr(v, "stride"):
        assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."


def pad_head_dim(q: Any, k: Any, v: Any) -> Tuple[Any, Any, Any, int]:
    if not hasattr(q, "size"):
        return q, k, v, 0
    if hasattr(q, "stride") and hasattr(k, "stride") and hasattr(v, "stride"):
        assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
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
