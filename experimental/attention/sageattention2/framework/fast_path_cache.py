from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple
import torch

_FP32_MAX = 3.4028234663852886e38
_UPSTREAM_LOG2E = 1.44269504

_PV_ACCUM_DTYPE_ENV = "SAGEATTN_PV_ACCUM_DTYPE"
_QK_QUANT_GRAN_ENV = "SAGEATTN_QK_QUANT_GRAN"


def _env_override(name: str, explicit: Optional[str], default: str) -> str:
    if explicit is not None:
        return explicit
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _check_sm_scale(scale: float) -> float:
    if not math.isfinite(scale) or abs(scale) > _FP32_MAX:
        raise ValueError("invalid FP32 sm_scale")
    return scale


class _FastPathPlan:
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
        "smooth_v",
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
        smooth_v: bool,
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
        self.smooth_v = smooth_v
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


def get_cached_plan(key: Tuple[Any, ...]) -> Optional[_FastPathPlan]:
    return _PLAN_CACHE.get(key)


def set_cached_plan(key: Tuple[Any, ...], plan: _FastPathPlan) -> None:
    if len(_PLAN_CACHE) >= _MAX_CACHE_SIZE:
        _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    _PLAN_CACHE[key] = plan
