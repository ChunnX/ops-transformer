# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SageAttention2 Ascend 950: per-token INT8 QK, FP8 PV, FP32 accumulation."""

import torch


def _validate(q, k, v, layout, causal, gran, kwargs):
    if layout not in ("HND", "NHD"):
        raise ValueError("tensor_layout must be HND or NHD")
    if gran != "per_token":
        raise ValueError("only per_token Q/K quantization is supported")
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


def sageattn_qk_int8_pv_fp8_asc(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_gran="per_token",
    sm_scale=None,
    pv_accum_dtype="fp32",
    smooth_k=True,
    smooth_v=False,
    return_lse=False,
    **kwargs,
):
    from .aclnn import run

    return run(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
        **kwargs,
    )


def sageattn(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    sm_scale=None,
    return_lse=False,
    **kwargs,
):
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
