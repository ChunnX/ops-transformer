"""PyTorch 2.x dispatcher interface, schema definition, and FakeTensor/Meta registration for SageAttention2."""

from __future__ import annotations

import math
from typing import Any
import warnings

import torch
from torch.library import impl

# 接口字符串协议唯一来源：error_check.py 的 VALID_* 集合（禁止在此复制第二套字面量表）
from .error_check import VALID_LAYOUTS, VALID_PV_ACCUM_DTYPES, VALID_QUANT_GRANS


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    causal: bool,
    gran: str,
    pv_accum_dtype: str,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate tensors and parameters against SageAttention2 specifications."""
    if layout not in VALID_LAYOUTS:
        raise ValueError("tensor_layout must be HND or NHD")
    if gran not in VALID_QUANT_GRANS:
        raise ValueError(f"qk_quant_gran must be 'per_thread' or 'per_warp', got '{gran}'")
    if pv_accum_dtype not in VALID_PV_ACCUM_DTYPES:
        raise ValueError("pv_accum_dtype must be fp32, fp32+fp32, or fp32+fp16")
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


# Register custom operator schema under sageattention2 namespace using FRAGMENT to allow multiple module imports
_library = torch.library.Library("sageattention2", "FRAGMENT")

if not hasattr(getattr(torch.ops, "sageattention2", None), "forward"):
    _library.define(
        'forward(Tensor q, Tensor k, Tensor v, *, str tensor_layout="HND", '
        'bool is_causal=False, str qk_quant_gran="per_thread", float? sm_scale=None, '
        'str pv_accum_dtype="fp32+fp16", bool smooth_k=True, bool smooth_v=False, '
        'bool return_lse=False) -> (Tensor, Tensor?)'
    )

    @impl(_library, "forward", "Meta")
    def _meta(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        tensor_layout: str = "HND",
        is_causal: bool = False,
        qk_quant_gran: str = "per_thread",
        sm_scale: float | None = None,
        pv_accum_dtype: str = "fp32+fp16",
        smooth_k: bool = True,
        smooth_v: bool = False,
        return_lse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """FakeTensor / Meta implementation for shape deduction in torch.compile graph mode."""
        qc, _, _ = _validate(
            q, k, v, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype, {}
        )
        if sm_scale is not None and (
            not math.isfinite(sm_scale) or abs(sm_scale) > 3.4028234663852886e38
        ):
            raise ValueError("invalid FP32 sm_scale")
        # 冻结语义：显式 sm_scale 必须为正；默认 1/sqrt(D) 仅在 None 时生效
        if sm_scale is not None and sm_scale <= 0:
            raise ValueError(
                "sm_scale must be > 0 when explicitly provided; omit it for the default 1/sqrt(head_dim)"
            )

        out = q.new_empty(q.shape)
        lse = q.new_empty(qc.shape[:3], dtype=torch.float32) if return_lse else None
        return out, lse

    @impl(_library, "forward", "PrivateUse1")
    def _npu(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        tensor_layout: str = "HND",
        is_causal: bool = False,
        qk_quant_gran: str = "per_thread",
        sm_scale: float | None = None,
        pv_accum_dtype: str = "fp32+fp16",
        smooth_k: bool = True,
        smooth_v: bool = False,
        return_lse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Execute via the compiled C++ bridge on Ascend NPU hardware."""
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
            smooth_v,
            return_lse,
        )

    @impl(_library, "forward", "CPU")
    def _cpu(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        tensor_layout: str = "HND",
        is_causal: bool = False,
        qk_quant_gran: str = "per_thread",
        sm_scale: float | None = None,
        pv_accum_dtype: str = "fp32+fp16",
        smooth_k: bool = True,
        smooth_v: bool = False,
        return_lse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """High-fidelity CPU reference fallback for local tests and non-NPU environments."""
        from .dispatch import run_sageattn_cpu_reference
        from .error_check import compute_sm_scale, pad_head_dim

        q_pad, k_pad, v_pad, head_dim_og = pad_head_dim(q, k, v)
        sm_scale_val = compute_sm_scale(head_dim_og, sm_scale)
        out_pad, lse = run_sageattn_cpu_reference(
            q_pad=q_pad,
            k_pad=k_pad,
            v_pad=v_pad,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            qk_quant_gran=qk_quant_gran,
            sm_scale=sm_scale_val,
            pv_accum_dtype=pv_accum_dtype,
            smooth_k=smooth_k,
            smooth_v=smooth_v,
            return_lse=return_lse,
        )
        out = out_pad[..., :head_dim_og].contiguous()
        return out, lse


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: float | None = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run SageAttention2 forward through PyTorch custom operator dispatcher.

    Args:
        q: FP16/BF16 query, [B,Hq,Sq,D] in HND or [B,Sq,Hq,D] in NHD.
        k: Key with the same dtype/device; [B,Hkv,Sk,D] or [B,Sk,Hkv,D].
        v: Value with the same shape, dtype and device as k.
        tensor_layout: HND or NHD. Last dimension must be contiguous, 1 <= D <= 128.
        is_causal: Apply a causal mask; requires Sq == Sk.
        qk_quant_gran: per_thread or per_warp.
        sm_scale: Softmax scale; defaults to 1/sqrt(D) only when omitted (None).
            Explicit 0/negative values are rejected.
        pv_accum_dtype: fp32, fp32+fp32, or fp32+fp16. Fully preserved and passed through.
        smooth_k: Subtract the sequence mean of K before quantization.
        smooth_v: Center V around mean (warns and ignored unless pv_accum_dtype='fp32').
        return_lse: Also return FP32 logsumexp with shape [B,Hq,Sq].
        **kwargs: Unused parameters.

    Returns:
        Output tensor with q's shape/dtype, or (output, lse) when return_lse=True.
    """
    _validate(q, k, v, tensor_layout, is_causal, qk_quant_gran, pv_accum_dtype, kwargs)
    if sm_scale is not None:
        sm_scale = float(sm_scale)
        if not math.isfinite(sm_scale) or abs(sm_scale) > 3.4028234663852886e38:
            raise ValueError("invalid FP32 sm_scale")
        # 冻结语义：显式 sm_scale 必须为正；默认 1/sqrt(D) 仅在 None 时生效
        if sm_scale <= 0:
            raise ValueError(
                "sm_scale must be > 0 when explicitly provided; omit it for the default 1/sqrt(head_dim)"
            )

    if smooth_v and pv_accum_dtype != "fp32":
        warnings.warn(
            f"smooth_v will be ignored when pv_accum_dtype='{pv_accum_dtype}' "
            "(FP8 PV does not support value centering; use pv_accum_dtype='fp32' to enable smooth_v).",
            UserWarning,
            stacklevel=2,
        )

    out, lse = torch.ops.sageattention2.forward(
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
    return (out, lse) if return_lse else out
