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
from typing import Any, Dict, Optional, Tuple, Union
import warnings

import torch
from torch.library import impl

__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_asc", "replace_sdpa", "run"]


class _FastPathPlan:
    """Pre-validated execution plan caching static shapes, layouts, and scaling."""

    __slots__ = (
        "q_shape",
        "k_shape",
        "v_shape",
        "dtype",
        "device",
        "tensor_layout",
        "is_causal",
        "qk_quant_gran",
        "pv_accum_dtype",
        "smooth_k",
        "return_lse",
        "sm_scale",
        "scale",
        "d_orig",
        "need_pad",
        "pad_size",
        "b",
        "hq",
        "hk",
        "sq",
        "is_npu",
    )

    def __init__(
        self,
        q_shape: Tuple[int, ...],
        k_shape: Tuple[int, ...],
        v_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        tensor_layout: str,
        is_causal: bool,
        qk_quant_gran: str,
        pv_accum_dtype: str,
        smooth_k: bool,
        return_lse: bool,
        sm_scale: Optional[float],
        scale: float,
        d_orig: int,
        need_pad: bool,
        pad_size: int,
        b: int,
        hq: int,
        hk: int,
        sq: int,
    ):
        self.q_shape = q_shape
        self.k_shape = k_shape
        self.v_shape = v_shape
        self.dtype = dtype
        self.device = device
        self.tensor_layout = tensor_layout
        self.is_causal = is_causal
        self.qk_quant_gran = qk_quant_gran
        self.pv_accum_dtype = pv_accum_dtype
        self.smooth_k = smooth_k
        self.return_lse = return_lse
        self.sm_scale = sm_scale
        self.scale = scale
        self.d_orig = d_orig
        self.need_pad = need_pad
        self.pad_size = pad_size
        self.b = b
        self.hq = hq
        self.hk = hk
        self.sq = sq
        self.is_npu = device.type in ("privateuseone", "npu")


_LAST_PLAN: Optional[_FastPathPlan] = None
_PLAN_CACHE: Dict[Tuple[Any, ...], _FastPathPlan] = {}
_MAX_CACHE_SIZE: int = 128
_EXT_MODULE = None
_FORWARD_FN = None


def _get_extension():
    global _EXT_MODULE, _FORWARD_FN
    if _EXT_MODULE is None:
        from ._extension import load_extension

        _EXT_MODULE = load_extension()
        _FORWARD_FN = _EXT_MODULE.forward
    return _EXT_MODULE


def _compute_lse_correction(
    q: torch.Tensor,
    k: torch.Tensor,
    tensor_layout: str,
    b: int,
    hq: int,
    hk: int,
    sq: int,
    d: int,
) -> torch.Tensor:
    """Compute GQA LSE correction factor without repeat_interleave memory copies."""
    if tensor_layout == "NHD":
        # q: [B, Sq, Hq, D], k: [B, Sk, Hk, D]
        km = k.mean(dim=1, keepdim=True)  # [B, 1, Hk, D]
        q_hnd = q.transpose(1, 2)  # [B, Hq, Sq, D]
        km_hnd = km.transpose(1, 2)  # [B, Hk, 1, D]
        if hq != hk:
            g = hq // hk
            return torch.matmul(
                q_hnd.reshape(b, hk, g * sq, d),
                km_hnd.transpose(2, 3),
            ).view(b, hq, sq).to(torch.float32)
        return torch.matmul(q_hnd, km_hnd.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        # q: [B, Hq, Sq, D], k: [B, Hk, Sk, D]
        km = k.mean(dim=2, keepdim=True)  # [B, Hk, 1, D]
        if hq != hk:
            g = hq // hk
            return torch.matmul(
                q.reshape(b, hk, g * sq, d),
                km.transpose(2, 3),
            ).view(b, hq, sq).to(torch.float32)
        return torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)


def _validate(q, k, v, layout, causal, gran, accum, kwargs):
    """Validate input tensors and configurations. Returns canonical (HND) views."""
    if layout != "HND" and layout != "NHD":
        raise ValueError("tensor_layout must be HND or NHD")
    if gran != "per_thread" and gran != "per_warp" and gran != "per_token":
        raise ValueError("qk_quant_gran must be per_thread, per_warp or per_token")
    if accum != "fp32+fp16" and accum != "fp32+fp32" and accum != "fp32":
        raise ValueError("pv_accum_dtype must be fp32+fp16, fp32+fp32 or fp32")
    if kwargs:
        if kwargs.pop("attn_mask", None) is not None:
            raise ValueError("attn_mask is unsupported")
        if kwargs.pop("dropout_p", 0.0) != 0:
            raise ValueError("dropout is unsupported")
        if kwargs:
            raise TypeError(f"unsupported arguments: {sorted(kwargs)}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must have rank 4")
    q_dtype = q.dtype
    if (
        q_dtype is not torch.float16 and q_dtype is not torch.bfloat16
    ) or k.dtype is not q_dtype or v.dtype is not q_dtype:
        raise ValueError("q/k/v must share FP16 or BF16 dtype")
    q_dev = q.device
    if k.device != q_dev or v.device != q_dev:
        raise ValueError("q/k/v must share a device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("last dimension must be contiguous")
    qs = q.shape
    ks = k.shape
    vs = v.shape
    for n in qs:
        if n <= 0:
            raise ValueError("empty dimensions are unsupported")
        if n >= 4294967232:  # 2**32 - 64
            raise ValueError("dimensions exceed the kernel uint32 range")
    for n in ks:
        if n <= 0:
            raise ValueError("empty dimensions are unsupported")
        if n >= 4294967232:
            raise ValueError("dimensions exceed the kernel uint32 range")
    for n in vs:
        if n <= 0:
            raise ValueError("empty dimensions are unsupported")
        if n >= 4294967232:
            raise ValueError("dimensions exceed the kernel uint32 range")

    if layout == "HND":
        b, h, s, d = qs
        kb, kh, ks_len, kd = ks
    else:
        b, s, h, d = qs
        kb, ks_len, kh, kd = ks

    if d > 128:
        raise ValueError("Only head_dim <= 128 is supported")
    if d <= 0:
        raise ValueError("head_dim must be in [1,128]")
    if ks != vs or kb != b or kd != d:
        raise ValueError("incompatible q/k/v shapes")
    if h % kh != 0:
        raise ValueError("Hq must be divisible by Hkv")
    if b * kh >= 4294967296:  # 2**32
        raise ValueError("batch times KV heads exceeds the kernel uint32 range")
    if causal and s != ks_len:
        raise ValueError("causal requires Sq == Sk")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("forward-only operator does not support autograd")
    return (q, k, v) if layout == "HND" else (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))


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
    return _get_extension().forward(
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
            tensor_layout=tensor_layout,
            return_lse=True,
        )
        scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else float(sm_scale)
        if smooth_k:
            if tensor_layout == "NHD":
                b, sq, hq, d = q.shape
                hk = k.shape[2]
            else:
                b, hq, sq, d = q.shape
                hk = k.shape[1]
            lse_corr = _compute_lse_correction(q, k, tensor_layout, b, hq, hk, sq, d)
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
            tensor_layout=tensor_layout,
            return_lse=False,
        )
        return out, None


_forward_op = torch.ops.sageattention2.forward


def _build_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    qk_quant_gran: str,
    sm_scale: Optional[float],
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
    kwargs: Optional[Dict[str, Any]],
) -> _FastPathPlan:
    """Run full validation, calculate scale and padding, and return a compiled _FastPathPlan."""
    _validate(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        qk_quant_gran,
        pv_accum_dtype,
        kwargs if kwargs else {},
    )
    d_orig = q.shape[-1]
    scale = 1.0 / math.sqrt(d_orig) if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("invalid FP32 sm_scale")

    if smooth_v and pv_accum_dtype in ("fp32+fp16", "fp32+fp32"):
        warnings.warn(
            f"smooth_v is not supported when pv_accum_dtype is {pv_accum_dtype}, smooth_v is ignored and set to False",
            UserWarning,
            stacklevel=3,
        )

    target_d = 64 if d_orig <= 64 else 128
    need_pad = d_orig < target_d
    pad_size = target_d - d_orig if need_pad else 0

    if tensor_layout == "HND":
        b, hq, sq, _ = q.shape
        _, hk, _, _ = k.shape
    else:
        b, sq, hq, _ = q.shape
        _, _, hk, _ = k.shape

    return _FastPathPlan(
        q_shape=q.shape,
        k_shape=k.shape,
        v_shape=v.shape,
        dtype=q.dtype,
        device=q.device,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        return_lse=return_lse,
        sm_scale=sm_scale,
        scale=scale,
        d_orig=d_orig,
        need_pad=need_pad,
        pad_size=pad_size,
        b=b,
        hq=hq,
        hk=hk,
        sq=sq,
    )


def _dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    qk_quant_gran: str,
    sm_scale: Optional[float],
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
    kwargs: Optional[Dict[str, Any]],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Ultra-low latency dispatch with fast-path caching for inference loops."""
    global _LAST_PLAN, _PLAN_CACHE

    plan = _LAST_PLAN
    # Fast path: check last plan (typical in layer-by-layer transformer loops)
    if (
        plan is not None
        and not kwargs
        and not smooth_v
        and q.shape == plan.q_shape
        and k.shape == plan.k_shape
        and q.dtype is plan.dtype
        and tensor_layout == plan.tensor_layout
        and is_causal == plan.is_causal
        and return_lse == plan.return_lse
        and sm_scale == plan.sm_scale
        and smooth_k == plan.smooth_k
        and qk_quant_gran == plan.qk_quant_gran
        and pv_accum_dtype == plan.pv_accum_dtype
        and not q.requires_grad
    ):
        pass  # Fast-path hit
    else:
        # Cache lookup or full build
        if not kwargs and not smooth_v:
            key = (
                q.shape,
                k.shape,
                v.shape,
                q.dtype,
                q.device,
                tensor_layout,
                is_causal,
                qk_quant_gran,
                pv_accum_dtype,
                smooth_k,
                return_lse,
                sm_scale,
            )
            plan = _PLAN_CACHE.get(key)
            if (
                plan is not None
                and k.dtype is plan.dtype
                and v.dtype is plan.dtype
                and k.device == plan.device
                and v.device == plan.device
                and q.stride(-1) == 1
                and k.stride(-1) == 1
                and v.stride(-1) == 1
                and not (q.requires_grad or k.requires_grad or v.requires_grad)
            ):
                _LAST_PLAN = plan
            else:
                plan = _build_plan(
                    q,
                    k,
                    v,
                    tensor_layout,
                    is_causal,
                    qk_quant_gran,
                    sm_scale,
                    pv_accum_dtype,
                    smooth_k,
                    smooth_v,
                    return_lse,
                    kwargs,
                )
                if len(_PLAN_CACHE) >= _MAX_CACHE_SIZE:
                    _PLAN_CACHE.clear()
                _PLAN_CACHE[key] = plan
                _LAST_PLAN = plan
        else:
            plan = _build_plan(
                q,
                k,
                v,
                tensor_layout,
                is_causal,
                qk_quant_gran,
                sm_scale,
                pv_accum_dtype,
                smooth_k,
                smooth_v,
                return_lse,
                kwargs,
            )

    d_orig = plan.d_orig
    scale = plan.scale

    if plan.need_pad:
        pad = (0, plan.pad_size)
        q_in = torch.nn.functional.pad(q, pad)
        k_in = torch.nn.functional.pad(k, pad)
        v_in = torch.nn.functional.pad(v, pad)
    else:
        q_in, k_in, v_in = q, k, v

    if plan.is_npu:
        if _FORWARD_FN is None:
            _get_extension()
        out, lse = _FORWARD_FN(
            q_in,
            k_in,
            v_in,
            tensor_layout,
            is_causal,
            qk_quant_gran,
            scale,
            pv_accum_dtype,
            smooth_k,
            return_lse,
        )
    else:
        out, lse = _forward_op(
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

    if plan.need_pad:
        out = out[..., :d_orig]

    if return_lse:
        if smooth_k:
            lse_correction = _compute_lse_correction(
                q, k, tensor_layout, plan.b, plan.hq, plan.hk, plan.sq, d_orig
            )
            lse_final = lse / 1.44269504 + lse_correction * scale
        else:
            lse_final = lse / 1.44269504
        return out, lse_final
    return out


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
    return _dispatch(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        qk_quant_gran,
        sm_scale,
        pv_accum_dtype,
        smooth_k,
        smooth_v,
        return_lse,
        kwargs if kwargs else None,
    )


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
    if not kwargs:
        return _dispatch(
            q,
            k,
            v,
            tensor_layout,
            is_causal,
            "per_thread",
            sm_scale,
            "fp32+fp16",
            True,
            False,
            return_lse,
            None,
        )
    qk_quant_gran = kwargs.pop("qk_quant_gran", "per_thread")
    pv_accum_dtype = kwargs.pop("pv_accum_dtype", "fp32+fp16")
    smooth_k = kwargs.pop("smooth_k", True)
    smooth_v = kwargs.pop("smooth_v", False)
    return _dispatch(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        qk_quant_gran,
        sm_scale,
        pv_accum_dtype,
        smooth_k,
        smooth_v,
        return_lse,
        kwargs,
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
                return _dispatch(
                    query,
                    key,
                    value,
                    "HND",
                    is_causal,
                    "per_thread",
                    scale,
                    "fp32+fp16",
                    True,
                    False,
                    False,
                    None,
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
