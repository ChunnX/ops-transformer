"""
Input validation, error checking, and preprocessing for SageAttention2 on Ascend NPU.
Aligned with task/01_design.md 1.1 to 1.3 and git/SageAttention/sageattention/core.py.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import warnings

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock()
    F = MagicMock()


VALID_LAYOUTS = {"HND", "NHD"}
VALID_QUANT_GRANS = {"per_warp", "per_thread"}
VALID_PV_ACCUM_DTYPES = {"fp32", "fp32+fp32", "fp32+fp16"}


def validate_common_parameters(
    tensor_layout: str,
    qk_quant_gran: str,
    pv_accum_dtype: str,
    kwargs: Dict[str, Any],
) -> None:
    """Validate string hyperparameters and kwargs."""
    if tensor_layout not in VALID_LAYOUTS:
        raise ValueError(
            f"Invalid tensor_layout '{tensor_layout}', expected one of {sorted(VALID_LAYOUTS)}"
        )

    if qk_quant_gran not in VALID_QUANT_GRANS:
        raise ValueError(
            f"qk_quant_gran must be either 'per_warp' or 'per_thread', got '{qk_quant_gran}'"
        )

    if pv_accum_dtype not in VALID_PV_ACCUM_DTYPES:
        raise ValueError(
            f"pv_accum_dtype must be one of {sorted(VALID_PV_ACCUM_DTYPES)}, got '{pv_accum_dtype}'"
        )

    if kwargs.get("attn_mask") is not None:
        raise ValueError(
            "attn_mask is not supported in SageAttention2; only is_causal is supported."
        )


def handle_smooth_v(pv_accum_dtype: str, smooth_v: bool) -> bool:
    """
    Handle smooth_v semantic rules.
    If pv_accum_dtype is 'fp32+fp32' or 'fp32+fp16' and smooth_v is True,
    emit a warning and automatically set smooth_v to False.
    """
    if pv_accum_dtype in ("fp32+fp32", "fp32+fp16") and smooth_v:
        warnings.warn(
            f"pv_accum_dtype is '{pv_accum_dtype}', smooth_v will be ignored.",
            UserWarning,
            stacklevel=3,
        )
        return False
    return smooth_v


def validate_tensors(
    q: Any,
    k: Any,
    v: Any,
    tensor_layout: str,
    is_causal: bool,
) -> None:
    """Validate tensor shapes, dimensions, dtypes, and device attributes."""
    if not hasattr(q, "dim") or not hasattr(k, "dim") or not hasattr(v, "dim"):
        return

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"Input tensors q, k, v must all be 4D tensors, got dimensions "
            f"q:{q.dim()}, k:{k.dim()}, v:{v.dim()}"
        )

    # Layout parsing
    if tensor_layout == "HND":
        b_q, h_q, s_q, d_q = q.shape
        b_k, h_k, s_k, d_k = k.shape
        b_v, h_v, s_v, d_v = v.shape
    else:  # NHD
        b_q, s_q, h_q, d_q = q.shape
        b_k, s_k, h_k, d_k = k.shape
        b_v, s_v, h_v, d_v = v.shape

    # Batch size matching
    if b_q != b_k or b_q != b_v:
        raise ValueError(
            f"Batch size mismatch across tensors: q({b_q}), k({b_k}), v({b_v})"
        )

    # KV shapes matching
    if h_k != h_v or s_k != s_v or d_k != d_v:
        raise ValueError(
            f"Key and value tensor shapes must match: k({k.shape}) vs v({v.shape})"
        )

    # Head dimension matching
    if d_q != d_k:
        raise ValueError(
            f"Head dimension mismatch between query and key: q({d_q}) vs k({d_k})"
        )

    # GQA ratio constraint
    if h_q % h_k != 0:
        raise ValueError(
            f"GQA constraint violated: num_qo_heads ({h_q}) must be divisible by "
            f"num_kv_heads ({h_k})"
        )

    # Causal constraint
    if is_causal and s_q != s_k:
        raise ValueError(
            f"is_causal=True requires qo_len == kv_len, got qo_len={s_q} and kv_len={s_k}"
        )

    # Device consistency
    if hasattr(q, "device") and hasattr(k, "device") and hasattr(v, "device"):
        if q.device != k.device or q.device != v.device:
            raise AssertionError(
                f"All tensors (q, k, v) must be on the same device, got "
                f"q:{q.device}, k:{k.device}, v:{v.device}"
            )

    # Dtype consistency
    if hasattr(q, "dtype") and hasattr(k, "dtype") and hasattr(v, "dtype"):
        valid_dtypes = (torch.float16, torch.bfloat16)
        if q.dtype not in valid_dtypes:
            raise AssertionError(
                f"Input tensors must be in dtype of torch.float16 or torch.bfloat16, got {q.dtype}"
            )
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise AssertionError(
                f"All tensors (q, k, v) must have the same dtype, got "
                f"q:{q.dtype}, k:{k.dtype}, v:{v.dtype}"
            )


def pad_head_dim(q: Any, k: Any, v: Any) -> Tuple[Any, Any, Any, int]:
    """
    Pad head_dim to 64 or 128 if needed, strictly aligning with core.py lines 750-764.
    If D < 64: pad to 64.
    If 64 < D < 128: pad to 128.
    If D > 128: raise ValueError.
    Also asserts stride(-1) == 1 for contiguity.
    Returns (q_pad, k_pad, v_pad, head_dim_og).
    """
    if not hasattr(q, "size"):
        return q, k, v, 0

    head_dim_og = q.size(-1)

    if head_dim_og <= 0:
        raise ValueError(f"Invalid head_dim: {head_dim_og}, must be > 0")

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
        raise ValueError(f"Unsupported head_dim: {head_dim_og}, max supported head_dim is 128")

    # Contiguity check after padding (matches core.py line 764)
    if hasattr(q, "stride") and hasattr(k, "stride") and hasattr(v, "stride"):
        assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, (
            "Last dim of qkv must be contiguous."
        )

    return q, k, v, head_dim_og


def compute_sm_scale(head_dim_og: int, sm_scale: Optional[float]) -> float:
    """Compute sm_scale default: 1.0 / sqrt(head_dim_og) when sm_scale is None.

    Frozen semantics (task/01_design 1.3): the default only applies to None;
    an explicit sm_scale must be a finite positive number, otherwise raise.
    """
    if sm_scale is None:
        return 1.0 / math.sqrt(head_dim_og)
    scale = float(sm_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "sm_scale must be a finite positive number when explicitly provided; "
            "omit it for the default 1/sqrt(head_dim)"
        )
    return scale
