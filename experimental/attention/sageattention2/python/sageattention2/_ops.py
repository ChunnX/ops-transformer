# -*- coding: utf-8 -*-
"""SageAttention2 PyTorch dispatcher and user API interfaces.

Provides NPU-accelerated forward dispatching for SageAttention2 with INT8 QK
matrix multiplication and FP8 PV accumulation.
"""

from __future__ import annotations

import math, os, warnings
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import torch
from torch.library import impl

__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_asc", "replace_sdpa", "run"]

_FP32_LIMIT, _LOG2E_RECIPROCAL = 3.4028234663852886e38, 1.44269504
_ENV_NAME_PV_DTYPE, _ENV_NAME_QK_GRAN = "SAGEATTN_PV_ACCUM_DTYPE", "SAGEATTN_QK_QUANT_GRAN"

_SUPPORTED_LAYOUTS = frozenset({"HND", "NHD"})
_SUPPORTED_QUANT_GRAN = frozenset({"per_thread", "per_warp"})
_SUPPORTED_PV_ACCUM = frozenset({"fp32+fp16", "fp32+fp32", "fp32"})
_VALID_TENSOR_TYPES = frozenset({torch.float16, torch.bfloat16})
_MAX_DIM_THRESHOLD, _MAX_BATCH_HEAD_PROD = 4294967232, 4294967296


def _require(predicate: bool, error_msg: str, exc_cls=ValueError) -> None:
    if not predicate: raise exc_cls(error_msg)


def _env_override(var_name: str, direct_val: Optional[str], fallback: str) -> str:
    if direct_val is not None: return direct_val
    configured = os.environ.get(var_name)
    return fallback if (configured is None or configured == "") else configured


def _check_sm_scale(sm_scale: float) -> float:
    _require(math.isfinite(sm_scale) and abs(sm_scale) <= _FP32_LIMIT, "invalid FP32 sm_scale")
    return sm_scale


_ExecutionPlan = NamedTuple(
    "_ExecutionPlan",
    [
        ("q_shape", tuple), ("k_shape", tuple), ("v_shape", tuple), ("dtype", torch.dtype),
        ("device", torch.device), ("tensor_layout", str), ("is_causal", bool),
        ("qk_quant_gran", str), ("pv_accum_dtype", str), ("smooth_k", bool),
        ("smooth_v", bool), ("return_lse", bool), ("sm_scale", Optional[float]),
        ("scale", float), ("d_orig", int), ("need_pad", bool), ("pad_size", int),
        ("b", int), ("hq", int), ("hk", int), ("sq", int), ("is_npu", bool)
    ]
)

_FastPathPlan = _ExecutionPlan

_ACTIVE_PLAN: Optional[_ExecutionPlan] = None
_PLAN_CACHE: Dict[Tuple[Any, ...], _ExecutionPlan] = {}
_CACHE_LIMIT: int = 128
_EXT_MODULE, _FORWARD_FN = None, None


def _get_extension():
    global _EXT_MODULE, _FORWARD_FN
    if _EXT_MODULE is None:
        from ._extension import load_extension
        _EXT_MODULE = load_extension()
        _FORWARD_FN = _EXT_MODULE.forward
    return _EXT_MODULE


def _effective_smooth_v(smooth_v: bool, accum_type: str) -> bool:
    return bool(smooth_v and accum_type == "fp32")


def _compute_lse_correction(query: torch.Tensor, key: torch.Tensor, layout: str, batch: int, hq: int, hk: int, sq: int, dim: int) -> torch.Tensor:
    nhd_mode = (layout == "NHD")
    k_mean = key.mean(dim=(1 if nhd_mode else 2), keepdim=True)
    q_mat = query.transpose(1, 2) if nhd_mode else query
    k_mat = k_mean.transpose(1, 2) if nhd_mode else k_mean

    if hq == hk:
        delta = torch.matmul(q_mat, k_mat.transpose(2, 3)).squeeze(-1)
    else:
        group_span = hq // hk
        delta = torch.matmul(q_mat.reshape(batch, hk, group_span * sq, dim), k_mat.transpose(2, 3)).view(batch, hq, sq)
    return delta.to(torch.float32)


def _validate(q, k, v, layout, causal, gran, accum, kwargs):
    _require(layout in _SUPPORTED_LAYOUTS, "tensor_layout must be HND or NHD")
    _require(gran in _SUPPORTED_QUANT_GRAN, "qk_quant_gran must be per_thread or per_warp")
    _require(accum in _SUPPORTED_PV_ACCUM, "pv_accum_dtype must be fp32+fp16, fp32+fp32 or fp32")

    if kwargs:
        _require(kwargs.pop("attn_mask", None) is None, "attn_mask is unsupported")
        _require(kwargs.pop("dropout_p", 0.0) == 0, "dropout is unsupported")
        _require(not kwargs, f"unsupported arguments: {sorted(kwargs)}", TypeError)

    tensor_tuple = (q, k, v)
    _require(not any(t.ndim != 4 for t in tensor_tuple), "q/k/v must have rank 4")

    lead_dt = q.dtype
    _require(lead_dt in _VALID_TENSOR_TYPES and k.dtype == lead_dt and v.dtype == lead_dt, "q/k/v must share FP16 or BF16 dtype")
    _require(k.device == q.device and v.device == q.device, "q/k/v must share a device")
    _require(not any(t.stride(-1) != 1 for t in tensor_tuple), "last dimension must be contiguous")

    shapes = (q.shape, k.shape, v.shape)
    for shp in shapes:
        for extent in shp:
            _require(extent > 0, "empty dimensions are unsupported")
            _require(extent < _MAX_DIM_THRESHOLD, "dimensions exceed the kernel uint32 range")

    (b, h, s, d) = q.shape if layout == "HND" else (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
    (kb, kh, ks_len, kd) = k.shape if layout == "HND" else (k.shape[0], k.shape[2], k.shape[1], k.shape[3])

    _require(d <= 128, "Only head_dim <= 128 is supported")
    _require(d > 0, "head_dim must be in [1,128]")
    _require(shapes[1] == shapes[2] and kb == b and kd == d, "incompatible q/k/v shapes")
    _require(h % kh == 0, "Hq must be divisible by Hkv")
    _require(b * kh < _MAX_BATCH_HEAD_PROD, "batch times KV heads exceeds the kernel uint32 range")
    _require(not causal or s == ks_len, "causal requires Sq == Sk")
    _require(not any(t.requires_grad for t in tensor_tuple), "forward-only operator does not support autograd")

    return (q, k, v) if layout == "HND" else (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))


_library = torch.library.Library("sageattention2", "DEF")
_library.define(
    'forward(Tensor q, Tensor k, Tensor v, *, str tensor_layout="HND", '
    'bool is_causal=False, str qk_quant_gran="per_thread", float? sm_scale=None, '
    'str pv_accum_dtype="fp32+fp16", bool smooth_k=True, bool smooth_v=False, '
    'bool return_lse=False) -> (Tensor, Tensor?)'
)


@impl(_library, "forward", "Meta")
def _meta(q, k, v, *, tensor_layout="HND", is_causal=False, qk_quant_gran="per_thread",
          sm_scale=None, pv_accum_dtype="fp32+fp16", smooth_k=True, smooth_v=False, return_lse=False):
    canon_q, _, _ = _validate(q, k, v, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype, {})
    if sm_scale is not None: _check_sm_scale(sm_scale)
    out_tensor = q.new_empty(q.shape)
    lse_tensor = q.new_empty(canon_q.shape[:3], dtype=torch.float32) if return_lse else None
    return out_tensor, lse_tensor


@impl(_library, "forward", "PrivateUse1")
def _npu(q, k, v, *, tensor_layout="HND", is_causal=False, qk_quant_gran="per_thread",
         sm_scale=None, pv_accum_dtype="fp32+fp16", smooth_k=True, smooth_v=False, return_lse=False):
    return _get_extension().forward(
        q, k, v, tensor_layout, is_causal, qk_quant_gran, sm_scale, pv_accum_dtype, smooth_k, smooth_v, return_lse
    )


@impl(_library, "forward", "CPU")
def _cpu(q, k, v, *, tensor_layout="HND", is_causal=False, qk_quant_gran="per_thread",
         sm_scale=None, pv_accum_dtype="fp32+fp16", smooth_k=True, smooth_v=False, return_lse=False):
    from .reference import attention
    if not return_lse:
        return attention(
            q, k, v, sm_scale=sm_scale, is_causal=is_causal, smooth_k=smooth_k, smooth_v=smooth_v,
            qk_quant_gran=qk_quant_gran, pv_accum_dtype=pv_accum_dtype, tensor_layout=tensor_layout, return_lse=False
        ), None

    res, lse_ref = attention(
        q, k, v, sm_scale=sm_scale, is_causal=is_causal, smooth_k=smooth_k, smooth_v=smooth_v,
        qk_quant_gran=qk_quant_gran, pv_accum_dtype=pv_accum_dtype, tensor_layout=tensor_layout, return_lse=True
    )
    scale_val = (1.0 / math.sqrt(q.shape[-1])) if sm_scale is None else float(sm_scale)
    if smooth_k:
        is_nhd = (tensor_layout == "NHD")
        (batch_cnt, q_seq, h_q, d_elem) = q.shape if is_nhd else (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
        h_k = k.shape[2] if is_nhd else k.shape[1]
        correction = _compute_lse_correction(q, k, tensor_layout, batch_cnt, h_q, h_k, q_seq, d_elem)
        raw_lse = (lse_ref - correction * scale_val) * _LOG2E_RECIPROCAL
    else:
        raw_lse = lse_ref * _LOG2E_RECIPROCAL
    return res, raw_lse


_forward_op = torch.ops.sageattention2.forward


def _construct_plan(q, k, v, layout, causal, gran, sm_scale, accum, smooth_k, smooth_v, ret_lse, extra_kw) -> _ExecutionPlan:
    _validate(q, k, v, layout, causal, gran, accum, extra_kw or {})
    head_dim_input = q.shape[-1]
    scale_coeff = _check_sm_scale((1.0 / math.sqrt(head_dim_input)) if sm_scale is None else float(sm_scale))
    v_effective = _effective_smooth_v(smooth_v, accum)

    pad_boundary = 64 if head_dim_input <= 64 else 128
    pad_extent = max(0, pad_boundary - head_dim_input)

    nhd_view = (layout == "NHD")
    batch_num, heads_q, seq_q = (q.shape[0], q.shape[2], q.shape[1]) if nhd_view else (q.shape[0], q.shape[1], q.shape[2])
    heads_k = k.shape[2] if nhd_view else k.shape[1]

    return _ExecutionPlan(
        q.shape, k.shape, v.shape, q.dtype, q.device, layout, causal, gran, accum,
        smooth_k, v_effective, ret_lse, sm_scale, scale_coeff, head_dim_input,
        (pad_extent > 0), pad_extent, batch_num, heads_q, heads_k, seq_q,
        (q.device.type in ("privateuseone", "npu"))
    )


def _matches_plan(p: _ExecutionPlan, q, k, v, layout, causal, gran, sm_scale, accum, smooth_k, smooth_v, ret_lse) -> bool:
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1: return False
    if q.requires_grad or k.requires_grad or v.requires_grad: return False
    if q.dtype is not p.dtype or k.dtype is not p.dtype or v.dtype is not p.dtype: return False
    if q.device != p.device or k.device != p.device or v.device != p.device: return False
    if q.shape != p.q_shape or k.shape != p.k_shape or v.shape != p.v_shape: return False
    return (
        layout == p.tensor_layout and causal == p.is_causal and ret_lse == p.return_lse
        and sm_scale == p.sm_scale and smooth_k == p.smooth_k
        and _effective_smooth_v(smooth_v, accum) == p.smooth_v
        and gran == p.qk_quant_gran and accum == p.pv_accum_dtype
    )


def _dispatch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str,
    is_causal: bool, qk_quant_gran: str, sm_scale: Optional[float], pv_accum_dtype: str,
    smooth_k: bool, smooth_v: bool, return_lse: bool, kwargs: Optional[Dict[str, Any]],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    global _ACTIVE_PLAN, _PLAN_CACHE

    if smooth_v and pv_accum_dtype in ("fp32+fp16", "fp32+fp32"):
        warnings.warn(f'smooth_v is ignored unless pv_accum_dtype is "fp32", got {pv_accum_dtype}', UserWarning, stacklevel=3)

    allow_caching = not kwargs
    cached_plan = _ACTIVE_PLAN
    if allow_caching and cached_plan is not None and _matches_plan(
        cached_plan, q, k, v, tensor_layout, is_causal, qk_quant_gran, sm_scale, pv_accum_dtype, smooth_k, smooth_v, return_lse
    ):
        target_plan = cached_plan
    elif allow_caching:
        cache_id = (
            q.shape, k.shape, v.shape, q.dtype, q.device, tensor_layout, is_causal,
            qk_quant_gran, pv_accum_dtype, smooth_k, _effective_smooth_v(smooth_v, pv_accum_dtype),
            return_lse, sm_scale
        )
        target_plan = _PLAN_CACHE.get(cache_id)
        if target_plan is None or not _matches_plan(
            target_plan, q, k, v, tensor_layout, is_causal, qk_quant_gran, sm_scale, pv_accum_dtype, smooth_k, smooth_v, return_lse
        ):
            target_plan = _construct_plan(
                q, k, v, tensor_layout, is_causal, qk_quant_gran, sm_scale,
                pv_accum_dtype, smooth_k, smooth_v, return_lse, kwargs
            )
            if len(_PLAN_CACHE) >= _CACHE_LIMIT: _PLAN_CACHE.clear()
            _PLAN_CACHE[cache_id] = target_plan
        _ACTIVE_PLAN = target_plan
    else:
        target_plan = _construct_plan(
            q, k, v, tensor_layout, is_causal, qk_quant_gran, sm_scale,
            pv_accum_dtype, smooth_k, smooth_v, return_lse, kwargs
        )

    raw_dim, scalar_sm = target_plan.d_orig, target_plan.scale

    if target_plan.need_pad:
        pad_tup = (0, target_plan.pad_size)
        q_feed, k_feed, v_feed = torch.nn.functional.pad(q, pad_tup), torch.nn.functional.pad(k, pad_tup), torch.nn.functional.pad(v, pad_tup)
    else:
        q_feed, k_feed, v_feed = q, k, v

    if target_plan.is_npu:
        if _FORWARD_FN is None: _get_extension()
        output_t, lse_t = _FORWARD_FN(
            q_feed, k_feed, v_feed, tensor_layout, is_causal, qk_quant_gran,
            scalar_sm, pv_accum_dtype, smooth_k, target_plan.smooth_v, return_lse
        )
    else:
        output_t, lse_t = _forward_op(
            q_feed, k_feed, v_feed, tensor_layout=tensor_layout, is_causal=is_causal,
            qk_quant_gran=qk_quant_gran, sm_scale=scalar_sm, pv_accum_dtype=pv_accum_dtype,
            smooth_k=smooth_k, smooth_v=target_plan.smooth_v, return_lse=return_lse
        )

    if target_plan.need_pad: output_t = output_t[..., :raw_dim]
    if not return_lse: return output_t

    if smooth_k:
        corr_bias = _compute_lse_correction(q, k, tensor_layout, target_plan.b, target_plan.hq, target_plan.hk, target_plan.sq, raw_dim)
        lse_final = lse_t / _LOG2E_RECIPROCAL + corr_bias * scalar_sm
    else:
        lse_final = lse_t / _LOG2E_RECIPROCAL
    return output_t, lse_final


def sageattn_qk_int8_pv_fp8_asc(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str = "HND",
    is_causal: bool = False, qk_quant_gran: str = "per_thread", sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16", smooth_k: bool = True, smooth_v: bool = False,
    return_lse: bool = False, **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return _dispatch(q, k, v, tensor_layout, is_causal, qk_quant_gran, sm_scale, pv_accum_dtype, smooth_k, smooth_v, return_lse, kwargs or None)


def sageattn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str = "HND",
    is_causal: bool = False, sm_scale: Optional[float] = None, return_lse: bool = False, **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return _dispatch(
        q, k, v, tensor_layout, is_causal,
        _env_override(_ENV_NAME_QK_GRAN, kwargs.pop("qk_quant_gran", None), "per_thread"),
        sm_scale,
        _env_override(_ENV_NAME_PV_DTYPE, kwargs.pop("pv_accum_dtype", None), "fp32+fp16"),
        kwargs.pop("smooth_k", True), kwargs.pop("smooth_v", False), return_lse, kwargs or None
    )


_orig_sdpa = None


def replace_sdpa(enable: bool = True) -> None:
    global _orig_sdpa
    if enable:
        if _orig_sdpa is None: _orig_sdpa = torch.nn.functional.scaled_dot_product_attention

        def _patched_sdpa(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0,
            is_causal: bool = False, scale: Optional[float] = None, enable_gqa: bool = False,
        ) -> torch.Tensor:
            if attn_mask is None and dropout_p == 0.0:
                return _dispatch(query, key, value, "HND", is_causal, "per_thread", scale, "fp32+fp16", True, False, False, None)
            return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)

        torch.nn.functional.scaled_dot_product_attention = _patched_sdpa
    else:
        if _orig_sdpa is not None:
            torch.nn.functional.scaled_dot_product_attention = _orig_sdpa
            _orig_sdpa = None


run = sageattn_qk_int8_pv_fp8_asc
