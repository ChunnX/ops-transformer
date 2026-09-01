# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import math
import os
import warnings
from typing import Optional, Tuple, Union

import torch

from . import _C  # noqa: F401


_VALID_LAYOUTS = {"HND", "NHD"}
_VALID_GRAN = {"per_warp", "per_thread"}
_VALID_ACCUM = {"fp32", "fp32+fp32", "fp32+fp16"}


def _env_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def sageattn_qk_int8_pv_fp8_asc(
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
    return_lse: bool = False,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Ascend SageAttention2 INT8-QK + FP8-PV forward path."""
    attn_mask = kwargs.pop("attn_mask", None)
    if attn_mask is not None:
        raise ValueError("attn_mask is not supported by the SageAttention2 NPU path")
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword arguments: {names}")
    if tensor_layout not in _VALID_LAYOUTS:
        raise ValueError(f"tensor_layout must be HND or NHD, got {tensor_layout!r}")
    if qk_quant_gran not in _VALID_GRAN:
        raise ValueError(f"qk_quant_gran must be per_warp or per_thread, got {qk_quant_gran!r}")
    if pv_accum_dtype not in _VALID_ACCUM:
        raise ValueError(
            f"pv_accum_dtype must be fp32, fp32+fp32, or fp32+fp16, got {pv_accum_dtype!r}"
        )
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must all be rank 4")
    if q.dtype not in (torch.float16, torch.bfloat16) or q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same float16 or bfloat16 dtype")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, and v must be on the same device")
    if smooth_v and pv_accum_dtype != "fp32":
        warnings.warn(
            f"smooth_v is ignored when pv_accum_dtype={pv_accum_dtype!r}",
            stacklevel=2,
        )
        smooth_v = False
    scale = 0.0 if sm_scale is None else float(sm_scale)
    if sm_scale is not None and (not math.isfinite(scale) or scale <= 0.0):
        raise ValueError("sm_scale must be finite and positive")
    output, lse = torch.ops.sageattention2.sage_attention2(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
    )
    return (output, lse) if return_lse else output


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Automatic Ascend entry. NPU tensors always dispatch to the FP8 SageAttention2 path."""
    # Environment defaults are intentionally applied only when the corresponding kwarg is absent.
    env_accum = os.getenv("SAGEATTN_NPU_PV_ACCUM_DTYPE")
    env_gran = os.getenv("SAGEATTN_NPU_QK_QUANT_GRAN")
    env_smooth_k = _env_bool("SAGEATTN_NPU_SMOOTH_K")
    env_smooth_v = _env_bool("SAGEATTN_NPU_SMOOTH_V")
    if env_accum is not None:
        kwargs.setdefault("pv_accum_dtype", env_accum)
    if env_gran is not None:
        kwargs.setdefault("qk_quant_gran", env_gran)
    if env_smooth_k is not None:
        kwargs.setdefault("smooth_k", env_smooth_k)
    if env_smooth_v is not None:
        kwargs.setdefault("smooth_v", env_smooth_v)
    return sageattn_qk_int8_pv_fp8_asc(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        return_lse=return_lse,
        **kwargs,
    )


__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_asc"]
