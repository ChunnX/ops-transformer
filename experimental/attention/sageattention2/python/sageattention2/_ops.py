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
from typing import Any, Optional, Tuple, Union
import warnings

import torch
from torch.library import impl

__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_asc", "run"]


def _validate(q, k, v, layout, causal, gran, accum, kwargs):
    if layout not in ("HND", "NHD"):
        raise ValueError("tensor_layout must be HND or NHD")
    if gran not in ("per_thread", "per_warp", "per_token"):
        raise ValueError("qk_quant_gran must be per_thread, per_warp or per_token")
    if accum not in ("fp32+fp16", "fp32+fp32", "fp32"):
        raise ValueError("pv_accum_dtype must be fp32+fp16, fp32+fp32 or fp32")
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


_library = torch.library.Library("sageattention2", "DEF")
_library.define(
    'forward(Tensor q, Tensor k, Tensor v, *, str tensor_layout="HND", '
    'bool is_causal=False, str qk_quant_gran="per_thread", float? sm_scale=None, '
    'str pv_accum_dtype="fp32+fp16", bool smooth_k=True, '
    'bool return_lse=False) -> (Tensor, Tensor?)'
)


@impl(_library, "forward", "Meta")
def _meta(
    q,
    k,
    v,
    *,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_gran="per_thread",
    sm_scale=None,
    pv_accum_dtype="fp32+fp16",
    smooth_k=True,
    return_lse=False,
):
    qc, _, _ = _validate(
        q, k, v, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype, {}
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
    qk_quant_gran="per_thread",
    sm_scale=None,
    pv_accum_dtype="fp32+fp16",
    smooth_k=True,
    return_lse=False,
):
    from ._extension import load_extension

    return load_extension().forward(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        qk_quant_gran,
        sm_scale,
        pv_accum_dtype,
        smooth_k,
        return_lse,
    )


@impl(_library, "forward", "CPU")
def _cpu(
    q,
    k,
    v,
    *,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_gran="per_thread",
    sm_scale=None,
    pv_accum_dtype="fp32+fp16",
    smooth_k=True,
    return_lse=False,
):
    from .reference import attention

    if return_lse:
        out, ref_lse = attention(
            q,
            k,
            v,
            sm_scale=sm_scale,
            is_causal=is_causal,
            smooth_k=smooth_k,
            smooth_v=False,
            tensor_layout=tensor_layout,
            return_lse=True,
        )
        scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else float(sm_scale)
        if smooth_k:
            if tensor_layout == "NHD":
                seq_dim, nh_dim = 1, 2
                km = k.mean(dim=seq_dim, keepdim=True)
                nqheads = q.size(nh_dim)
                nkheads = k.size(nh_dim)
                q_per_kv_heads = nqheads // nkheads
                km_bc = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim) if q_per_kv_heads > 1 else km
                q_hnd = q.transpose(1, 2)
                km_hnd = km_bc.transpose(1, 2)
                lse_corr = torch.matmul(q_hnd, km_hnd.transpose(-2, -1)).squeeze(-1).to(torch.float32)
            else:
                seq_dim, nh_dim = 2, 1
                km = k.mean(dim=seq_dim, keepdim=True)
                nqheads = q.size(nh_dim)
                nkheads = k.size(nh_dim)
                q_per_kv_heads = nqheads // nkheads
                km_bc = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim) if q_per_kv_heads > 1 else km
                lse_corr = torch.matmul(q, km_bc.transpose(-2, -1)).squeeze(-1).to(torch.float32)
            raw_lse = (ref_lse - lse_corr * scale) * 1.44269504
        else:
            raw_lse = ref_lse * 1.44269504
        return out, raw_lse
    else:
        out = attention(
            q,
            k,
            v,
            sm_scale=sm_scale,
            is_causal=is_causal,
            smooth_k=smooth_k,
            smooth_v=False,
            tensor_layout=tensor_layout,
            return_lse=False,
        )
        return out, None


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
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run forward SageAttention2 with INT8 QK and MXFP8 PV on Ascend NPU.

    Args:
        q: FP16/BF16 query, [B, Hq, Sq, D] in HND or [B, Sq, Hq, D] in NHD.
        k: Key with the same dtype and device; [B, Hkv, Sk, D] or [B, Sk, Hkv, D].
        v: Value with the same shape, dtype and device as k.
        tensor_layout: Tensor layout: "HND" (default) or "NHD".
        is_causal: Whether to apply causal mask. Only applicable when Sq == Sk.
        qk_quant_gran: Q/K INT8 quantization granularity: "per_thread" (default) or "per_warp".
        sm_scale: Softmax scale, defaults to 1.0 / sqrt(head_dim).
        pv_accum_dtype: PV accumulation precision: "fp32+fp16" (default), "fp32+fp32", or "fp32".
        smooth_k: Subtract key sequence mean before quantization. Default: True.
        smooth_v: Compatibility flag; True with fp32+fp16 or fp32+fp32 warns and is ignored.
        return_lse: Whether to return logsumexp normalization factor. Default: False.
        **kwargs: Only attn_mask=None and dropout_p=0 are accepted.

    Returns:
        Output tensor, or (output, lse) if return_lse is True.
        When return_lse is True, lse is converted via:
        lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504.
    """
    _validate(q, k, v, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype, kwargs)
    d_orig = q.shape[-1]
    scale = (
        1.0 / math.sqrt(d_orig)
        if sm_scale is None
        else float(sm_scale)
    )
    if not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("invalid FP32 sm_scale")

    if smooth_v and pv_accum_dtype in ("fp32+fp16", "fp32+fp32"):
        warnings.warn(
            f"smooth_v is not supported when pv_accum_dtype is {pv_accum_dtype}, smooth_v is ignored and set to False",
            UserWarning,
            stacklevel=2,
        )
        smooth_v = False

    lse_correction = None
    if smooth_k and return_lse:
        if tensor_layout == "NHD":
            # q: [B, Sq, Hq, D], k: [B, Sk, Hkv, D]
            seq_dim, nh_dim = 1, 2
            km = k.mean(dim=seq_dim, keepdim=True)
            nqheads = q.size(nh_dim)
            nkheads = k.size(nh_dim)
            q_per_kv_heads = nqheads // nkheads
            if q_per_kv_heads > 1:
                km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
            else:
                km_broadcast = km
            lse_correction = torch.matmul(
                q.transpose(1, 2),
                km_broadcast.transpose(1, 2).transpose(2, 3),
            ).squeeze(-1).to(torch.float32)
        else:
            # q: [B, Hq, Sq, D], k: [B, Hkv, Sk, D]
            seq_dim, nh_dim = 2, 1
            km = k.mean(dim=seq_dim, keepdim=True)
            nqheads = q.size(nh_dim)
            nkheads = k.size(nh_dim)
            q_per_kv_heads = nqheads // nkheads
            if q_per_kv_heads > 1:
                km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
            else:
                km_broadcast = km
            lse_correction = torch.matmul(
                q,
                km_broadcast.transpose(2, 3),
            ).squeeze(-1).to(torch.float32)

    # Head dimension auto-padding (D <= 64 -> 64, 64 < D <= 128 -> 128)
    target_d = 64 if d_orig <= 64 else 128
    need_pad = d_orig < target_d
    if need_pad:
        pad_size = target_d - d_orig
        q_in = torch.nn.functional.pad(q, (0, pad_size))
        k_in = torch.nn.functional.pad(k, (0, pad_size))
        v_in = torch.nn.functional.pad(v, (0, pad_size))
    else:
        q_in, k_in, v_in = q, k, v

    out, lse = torch.ops.sageattention2.forward(
        q_in,
        k_in,
        v_in,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        return_lse=return_lse,
    )

    if need_pad:
        out = out[..., :d_orig]

    if return_lse:
        lse_final = (
            lse / 1.44269504 + lse_correction * scale
            if smooth_k
            else lse / 1.44269504
        )
        return out, lse_final
    else:
        return out


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Automatically selects the appropriate SageAttention implementation on NPU (fixed to FP8 PV).

    Parameters and defaults match open-source sageattn interface with NPU backend routing.
    """
    qk_quant_gran = kwargs.pop("qk_quant_gran", "per_thread")
    pv_accum_dtype = kwargs.pop("pv_accum_dtype", "fp32+fp16")
    smooth_k = kwargs.pop("smooth_k", True)
    smooth_v = kwargs.pop("smooth_v", False)
    return sageattn_qk_int8_pv_fp8_asc(
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


_orig_sdpa = None


def replace_sdpa(enable: bool = True) -> None:
    """Monkey-patch torch.nn.functional.scaled_dot_product_attention with sageattn.

    When enable=True, replaces torch.nn.functional.scaled_dot_product_attention with a wrapper
    that routes standard attention calls to sageattn.
    When enable=False, restores the original implementation.
    """
    global _orig_sdpa
    if enable:
        if _orig_sdpa is None:
            _orig_sdpa = torch.nn.functional.scaled_dot_product_attention

        def _sage_sdpa_wrapper(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: Optional[float] = None,
            enable_gqa: bool = False,
        ) -> torch.Tensor:
            if attn_mask is None and dropout_p == 0.0:
                return sageattn(
                    query,
                    key,
                    value,
                    tensor_layout="HND",
                    is_causal=is_causal,
                    sm_scale=scale,
                )
            return _orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        torch.nn.functional.scaled_dot_product_attention = _sage_sdpa_wrapper
    else:
        if _orig_sdpa is not None:
            torch.nn.functional.scaled_dot_product_attention = _orig_sdpa
            _orig_sdpa = None


run = sageattn_qk_int8_pv_fp8_asc

