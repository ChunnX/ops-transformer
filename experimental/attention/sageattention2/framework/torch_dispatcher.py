from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple, Union
import warnings

import torch
from torch.library import impl

from .fast_path_cache import (
    _FastPathPlan,
    _env_override,
    _check_sm_scale,
    _UPSTREAM_LOG2E,
    _QK_QUANT_GRAN_ENV,
    _PV_ACCUM_DTYPE_ENV,
    get_cached_plan,
    set_cached_plan,
)

__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_asc", "replace_sdpa", "run"]

_EXT_MODULE = None
_FORWARD_FN = None
_LAST_PLAN: Optional[_FastPathPlan] = None


def _get_extension():
    global _EXT_MODULE, _FORWARD_FN
    if _EXT_MODULE is None:
        try:
            from ..python.sageattention2._extension import load_extension
            _EXT_MODULE = load_extension()
            _FORWARD_FN = _EXT_MODULE.forward
        except Exception:
            _EXT_MODULE = None
            _FORWARD_FN = None
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
    if tensor_layout == "NHD":
        km = k.mean(dim=1, keepdim=True)
        q_hnd = q.transpose(1, 2)
        km_hnd = km.transpose(1, 2)
        if hq != hk:
            g = hq // hk
            return torch.matmul(
                q_hnd.reshape(b, hk, g * sq, d),
                km_hnd.transpose(2, 3),
            ).view(b, hq, sq).to(torch.float32)
        return torch.matmul(q_hnd, km_hnd.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = k.mean(dim=2, keepdim=True)
        if hq != hk:
            g = hq // hk
            return torch.matmul(
                q.reshape(b, hk, g * sq, d),
                km.transpose(2, 3),
            ).view(b, hq, sq).to(torch.float32)
        return torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)


def _validate(q, k, v, layout, causal, gran, accum, kwargs):
    if layout != "HND" and layout != "NHD":
        raise ValueError("tensor_layout must be HND or NHD")
    if gran != "per_thread" and gran != "per_warp":
        raise ValueError("qk_quant_gran must be per_thread or per_warp")
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
    for t in (qs, ks, vs):
        for n in t:
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
    if b * kh >= 4294967296:
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
    'str pv_accum_dtype="fp32+fp16", bool smooth_k=True, bool smooth_v=False, '
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
    smooth_v=False,
    return_lse=False,
):
    qc, _, _ = _validate(
        q, k, v, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype, {}
    )
    if sm_scale is not None:
        _check_sm_scale(sm_scale)
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
    smooth_v=False,
    return_lse=False,
):
    ext = _get_extension()
    if ext is not None:
        return ext.forward(
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
        )
    return _cpu(
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
    smooth_v=False,
    return_lse=False,
):
    from ..reference.cpu_golden_reference import attention

    if return_lse:
        out, ref_lse = attention(
            q,
            k,
            v,
            sm_scale=sm_scale,
            is_causal=is_causal,
            smooth_k=smooth_k,
            smooth_v=smooth_v,
            qk_quant_gran=qk_quant_gran,
            pv_accum_dtype=pv_accum_dtype,
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
            raw_lse = (ref_lse - lse_corr * scale) * _UPSTREAM_LOG2E
        else:
            raw_lse = ref_lse * _UPSTREAM_LOG2E
        return out, raw_lse
    else:
        out = attention(
            q,
            k,
            v,
            sm_scale=sm_scale,
            is_causal=is_causal,
            smooth_k=smooth_k,
            smooth_v=smooth_v,
            qk_quant_gran=qk_quant_gran,
            pv_accum_dtype=pv_accum_dtype,
            tensor_layout=tensor_layout,
            return_lse=False,
        )
        return out, None


_forward_op = torch.ops.sageattention2.forward


def _effective_smooth_v(smooth_v: bool, pv_accum_dtype: str) -> bool:
    return smooth_v and pv_accum_dtype == "fp32"


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
    scale = _check_sm_scale(1.0 / math.sqrt(d_orig) if sm_scale is None else float(sm_scale))

    effective_smooth_v = _effective_smooth_v(smooth_v, pv_accum_dtype)
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
        smooth_v=effective_smooth_v,
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


def _plan_matches(
    plan: _FastPathPlan,
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
) -> bool:
    return (
        not (q.requires_grad or k.requires_grad or v.requires_grad)
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and q.dtype is plan.dtype
        and k.dtype is plan.dtype
        and v.dtype is plan.dtype
        and q.device == plan.device
        and k.device == plan.device
        and v.device == plan.device
        and q.shape == plan.q_shape
        and k.shape == plan.k_shape
        and v.shape == plan.v_shape
        and tensor_layout == plan.tensor_layout
        and is_causal == plan.is_causal
        and return_lse == plan.return_lse
        and sm_scale == plan.sm_scale
        and smooth_k == plan.smooth_k
        and _effective_smooth_v(smooth_v, pv_accum_dtype) == plan.smooth_v
        and qk_quant_gran == plan.qk_quant_gran
        and pv_accum_dtype == plan.pv_accum_dtype
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
    global _LAST_PLAN

    if smooth_v and pv_accum_dtype in ("fp32+fp16", "fp32+fp32"):
        warnings.warn(
            f'smooth_v is ignored unless pv_accum_dtype is "fp32", got {pv_accum_dtype}',
            UserWarning,
            stacklevel=3,
        )

    cacheable = not kwargs
    plan = _LAST_PLAN
    matched = (
        cacheable
        and plan is not None
        and _plan_matches(
            plan,
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
        )
    )
    if not matched:
        if cacheable:
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
                _effective_smooth_v(smooth_v, pv_accum_dtype),
                return_lse,
                sm_scale,
            )
            plan = get_cached_plan(key)
            if not (
                plan is not None
                and _plan_matches(
                    plan,
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
                )
            ):
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
                set_cached_plan(key, plan)
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
        global _FORWARD_FN
        if _FORWARD_FN is None:
            _get_extension()
        if _FORWARD_FN is not None:
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
                plan.smooth_v,
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
                smooth_v=plan.smooth_v,
                return_lse=return_lse,
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
            smooth_v=plan.smooth_v,
            return_lse=return_lse,
        )

    if plan.need_pad:
        out = out[..., :d_orig]

    if return_lse:
        if smooth_k:
            lse_correction = _compute_lse_correction(
                q, k, tensor_layout, plan.b, plan.hq, plan.hk, plan.sq, d_orig
            )
            lse_final = lse / _UPSTREAM_LOG2E + lse_correction * scale
        else:
            lse_final = lse / _UPSTREAM_LOG2E
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
    return _dispatch(
        q,
        k,
        v,
        tensor_layout,
        is_causal,
        _env_override(_QK_QUANT_GRAN_ENV, kwargs.pop("qk_quant_gran", None), "per_thread"),
        sm_scale,
        _env_override(_PV_ACCUM_DTYPE_ENV, kwargs.pop("pv_accum_dtype", None), "fp32+fp16"),
        kwargs.pop("smooth_k", True),
        kwargs.pop("smooth_v", False),
        return_lse,
        kwargs or None,
    )


_orig_sdpa = None


def replace_sdpa(enable: bool = True) -> None:
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
