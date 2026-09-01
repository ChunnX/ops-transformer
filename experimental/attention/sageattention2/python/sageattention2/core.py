"""sageattention2 NPU 包 — PyTorch 适配层（Host 侧逻辑）。

对标开源 https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py
的 `sageattn` 与 `sageattn_qk_int8_pv_fp8_cuda`（本任务改名 *_asc）。

本文件只包含 Host 侧逻辑（参数校验 / head_dim pad / sm_scale / smooth_k /
lse_correction / smooth_v 警告 / 分发 / 输出裁回 / LSE 换算）；
NPU kernel 经 backend.py 的注册后端调用；无 NPU 后端时回退 CPU 参考实现
（仅流程调试，不作验收）。
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Union

import torch

from . import backend
from .quant import quant_v_fp8, quant_qk_int8

__all__ = ["sageattn", "sageattn_qk_int8_pv_fp8_asc"]

_VALID_LAYOUTS = ("HND", "NHD")
_VALID_GRANS = ("per_warp", "per_thread")
_VALID_ACCUMS = ("fp32", "fp32+fp32", "fp32+fp16")
_LOG2E = 1.44269504


def _validate_inputs(q, k, v, tensor_layout, qk_quant_gran, pv_accum_dtype, attn_mask) -> None:
    if tensor_layout not in _VALID_LAYOUTS:
        raise ValueError(f"tensor_layout must be one of {_VALID_LAYOUTS}, got {tensor_layout}")
    if qk_quant_gran not in _VALID_GRANS:
        raise ValueError(f"qk_quant_gran must be one of {_VALID_GRANS}, got {qk_quant_gran}")
    if pv_accum_dtype not in _VALID_ACCUMS:
        raise ValueError(f"pv_accum_dtype must be one of {_VALID_ACCUMS}, got {pv_accum_dtype}")
    if attn_mask is not None:
        raise ValueError("attn_mask is not supported by sageattn_qk_int8_pv_fp8_asc")
    dtype = q.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q/k/v must be torch.float16 or torch.bfloat16")
    if not (k.dtype == v.dtype == dtype):
        raise TypeError("q/k/v must have the same dtype")
    if not (q.device == k.device == v.device):
        raise RuntimeError("q/k/v must be on the same device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("last dim of q/k/v must be contiguous (stride(-1)==1)")

    if tensor_layout == "HND":
        _, hq, sq, d = q.shape
        _, hkv, sk, _ = k.shape
    else:
        _, sq, hq, d = q.shape
        _, sk, hkv, _ = k.shape
    if hq % hkv != 0:
        raise ValueError(f"GQA requires num_qo_heads % num_kv_heads == 0, got {hq}/{hkv}")
    if not (0 < d <= 128):
        raise ValueError(f"Unsupported head_dim: {d}")


def _pad_head_dim(q, k, v) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    head_dim_og = q.size(-1)
    if head_dim_og < 64:
        pad = 64 - head_dim_og
    elif 64 < head_dim_og < 128:
        pad = 128 - head_dim_og
    else:  # == 64 或 == 128：不 pad（对齐开源 core.py 语义）
        pad = 0
    if pad:
        q = torch.nn.functional.pad(q, (0, pad))
        k = torch.nn.functional.pad(k, (0, pad))
        v = torch.nn.functional.pad(v, (0, pad))
    return q, k, v, head_dim_og


def _smooth_k_and_correction(q, k, tensor_layout, return_lse, sm_scale):
    """对标 core.py：km = k.mean(seq)；GQA repeat_interleave；lse_correction = q@km^T。"""
    seq_dim = 2 if tensor_layout == "HND" else 1
    nh_dim = 1 if tensor_layout == "HND" else 2
    km = k.mean(dim=seq_dim, keepdim=True)
    hq = q.size(nh_dim)
    hkv = k.size(nh_dim)
    q_per_kv_heads = hq // hkv
    if q_per_kv_heads > 1:
        km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
    else:
        km_broadcast = km
    lse_correction = None
    if return_lse:
        if tensor_layout == "NHD":
            lse_correction = torch.matmul(
                q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)
            ).squeeze(-1).to(torch.float32)
        else:
            lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    return km, lse_correction


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
    """INT8 QK + FP8 PV 量化注意力（NPU 入口，对标 *_cuda）。"""
    attn_mask = kwargs.pop("attn_mask", None)
    _validate_inputs(q, k, v, tensor_layout, qk_quant_gran, pv_accum_dtype, attn_mask)

    head_dim_og = q.size(-1)
    q, k, v, head_dim_og = _pad_head_dim(q, k, v)
    if sm_scale is None:
        sm_scale = head_dim_og ** -0.5

    km, lse_correction = (None, None)
    if smooth_k:
        km, lse_correction = _smooth_k_and_correction(q, k, tensor_layout, return_lse, sm_scale)

    # 对齐开源 core.py L797-803：仅 fp32+fp32 / fp32+fp16 忽略 smooth_v；fp32 下 smooth_v 生效
    if pv_accum_dtype in ("fp32+fp32", "fp32+fp16") and smooth_v:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
        smooth_v = False

    # --- 量化（Host 侧预处理；NPU 上可与 Attn 主循环融合，见设计文档 §3.2.4）---
    q_int8, q_scale, k_int8, k_scale = quant_qk_int8(
        q, k, km, tensor_layout=tensor_layout, qk_quant_gran=qk_quant_gran
    )
    v_fp8, v_scale, vm = quant_v_fp8(
        v, tensor_layout=tensor_layout,
        scale_max=2.25 if pv_accum_dtype == "fp32+fp16" else 448.0,
        smooth_v=smooth_v,
    )

    o = torch.empty(q.size(), dtype=q.dtype, device=q.device)
    lse = backend.run_attn(
        q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm,
        tensor_layout=tensor_layout, is_causal=is_causal,
        qk_quant_gran=qk_quant_gran, sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype, return_lse=return_lse,
        head_dim_og=head_dim_og,
    )

    o = o[..., :head_dim_og]
    if return_lse:
        lse_out = lse / _LOG2E
        if smooth_k:
            lse_out = lse_out + lse_correction * sm_scale
        return o, lse_out
    return o


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
    """自动分发：NPU 上固定走 FP8 PV（默认 pv_accum_dtype=fp32+fp16）。"""
    pv_accum_dtype = kwargs.pop("pv_accum_dtype", "fp32+fp16")
    qk_quant_gran = kwargs.pop("qk_quant_gran", "per_thread")
    smooth_k = kwargs.pop("smooth_k", True)
    smooth_v = kwargs.pop("smooth_v", False)
    return sageattn_qk_int8_pv_fp8_asc(
        q, k, v,
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
