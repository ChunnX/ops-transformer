# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyTorch dispatcher interface for the fused SageAttention2 operator."""

from __future__ import annotations

import math
import warnings

import torch
from torch.library import impl


def _validate(q, k, v, layout, causal, gran, kwargs, *, allow_meta=False):
    if layout not in ("HND", "NHD"):
        raise ValueError("tensor_layout must be HND or NHD")
    if gran not in ("per_token", "per_thread", "per_warp"):
        raise ValueError("qk_quant_gran must be per_token, per_thread or per_warp")
    if kwargs.pop("attn_mask", None) is not None:
        raise ValueError("attn_mask is unsupported")
    if kwargs.pop("dropout_p", 0.0) != 0:
        raise ValueError("dropout is unsupported")
    if kwargs:
        raise TypeError(f"unsupported arguments: {sorted(kwargs)}")
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("q/k/v must have rank 4")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(
        t.dtype != q.dtype for t in (k, v)
    ):
        raise ValueError("q/k/v must share FP16 or BF16 dtype")
    # Meta is allowed only by the dispatcher shape-inference implementation.
    # Fake NPU tensors report an NPU device and remain valid public inputs.
    if q.device.type != "npu" and not (allow_meta and q.device.type == "meta"):
        raise ValueError("SageAttention2 requires NPU inputs (q/k/v)")
    if any(t.device != q.device for t in (k, v)):
        raise ValueError("q/k/v must share a device")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("last dimension must be contiguous")
    if any(any(n <= 0 for n in t.shape) for t in (q, k, v)):
        raise ValueError("empty dimensions are unsupported")
    if any(any(n >= 2**32 - 64 for n in t.shape) for t in (q, k, v)):
        raise ValueError("dimensions exceed the kernel uint32 range")
    qc, kc, vc = (
        (q, k, v) if layout == "HND" else tuple(t.transpose(1, 2) for t in (q, k, v))
    )
    b, h, s, d = qc.shape
    if not 0 < d <= 128:
        raise ValueError("head_dim must be in [1,128]")
    if kc.shape != vc.shape or kc.shape[0] != b or kc.shape[-1] != d:
        raise ValueError("incompatible q/k/v shapes")
    if h % kc.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    if b * kc.shape[1] >= 2**32:
        raise ValueError("batch times KV heads exceeds the kernel uint32 range")
    if causal and s != kc.shape[2]:
        raise ValueError("causal requires Sq == Sk")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("forward-only operator does not support autograd")
    return qc, kc, vc


_library = torch.library.Library("sageattention2", "DEF")
_library.define(
    'forward(Tensor q, Tensor k, Tensor v, *, str tensor_layout="HND", '
    "bool is_causal=False, float? sm_scale=None, bool smooth_k=True, "
    "bool return_lse=False) -> (Tensor, Tensor?)"
)


@impl(_library, "forward", "Meta")
def _meta(
    q,
    k,
    v,
    *,
    tensor_layout="HND",
    is_causal=False,
    sm_scale=None,
    smooth_k=True,
    return_lse=False,
):
    qc, _, _ = _validate(
        q, k, v, tensor_layout, is_causal, "per_token", {}, allow_meta=True
    )
    if sm_scale is not None and (
        not math.isfinite(sm_scale) or abs(sm_scale) > 3.4028234663852886e38
    ):
        raise ValueError("invalid FP32 sm_scale")
    out = q.new_empty(q.shape)
    lse = q.new_empty(qc.shape[:3], dtype=torch.float32) if return_lse else None
    return out, lse


@impl(_library, "forward", "PrivateUse1")
def _npu(
    q,
    k,
    v,
    *,
    tensor_layout="HND",
    is_causal=False,
    sm_scale=None,
    smooth_k=True,
    return_lse=False,
):
    from ._extension import load_extension

    return load_extension().forward(
        q, k, v, tensor_layout, is_causal, sm_scale, smooth_k, return_lse
    )


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_token",
    sm_scale: float | None = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run forward SageAttention2 with per-token INT8 QK and MXFP8 PV.

    Args:
        q: FP16/BF16 query, [B,Hq,Sq,D] in HND or [B,Sq,Hq,D] in NHD.
        k: Key with the same dtype/device; [B,Hkv,Sk,D] or [B,Sk,Hkv,D].
        v: Value with the same shape, dtype and device as k.
        tensor_layout: HND or NHD. Last dimension must be contiguous, 1 <= D <= 128.
        is_causal: Apply a causal mask; requires Sq == Sk.
        qk_quant_gran: per_token; per_thread/per_warp warn and use per_token.
        sm_scale: Softmax scale, defaults to 1/sqrt(D).
        pv_accum_dtype: fp32+fp32, with fp32 as an alias; other values warn and fall back.
        smooth_k: Subtract the sequence mean of K before quantization.
        smooth_v: Compatibility flag; True warns and is ignored.
        return_lse: Also return FP32 logsumexp with shape [B,Hq,Sq].
        **kwargs: Only attn_mask=None and dropout_p=0 are accepted.

    Returns:
        Output with q's shape/dtype, or (output, lse) when return_lse=True.
        Hq must be divisible by Hkv. This operator is forward-only.
    """
    _validate(q, k, v, tensor_layout, is_causal, qk_quant_gran, kwargs)
    if sm_scale is not None:
        sm_scale = float(sm_scale)
        if not math.isfinite(sm_scale) or abs(sm_scale) > 3.4028234663852886e38:
            raise ValueError("invalid FP32 sm_scale")
    if qk_quant_gran != "per_token":
        warnings.warn(
            f"qk_quant_gran={qk_quant_gran!r} falls back to per_token on Ascend 950.",
            UserWarning,
            stacklevel=2,
        )
    if pv_accum_dtype not in ("fp32+fp32", "fp32"):
        warnings.warn(
            f"pv_accum_dtype={pv_accum_dtype!r} falls back to fp32+fp32 on Ascend 950.",
            UserWarning,
            stacklevel=2,
        )
    if smooth_v:
        warnings.warn(
            "smooth_v is ignored: SageAttention2 uses MXFP8 PV without value centering.",
            UserWarning,
            stacklevel=2,
        )
    out, lse = torch.ops.sageattention2.forward(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        smooth_k=smooth_k,
        return_lse=return_lse,
    )
    return (out, lse) if return_lse else out
