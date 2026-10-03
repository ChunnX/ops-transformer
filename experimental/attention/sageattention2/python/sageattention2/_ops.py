"""PyTorch 2.x dispatcher interface, schema definition, and FakeTensor/Meta registration for SageAttention2."""

from __future__ import annotations

import math
from typing import Any
import warnings

import torch
from torch.library import impl

from .error_check import (
    DEFAULT_PV_ACCUM_DTYPE,
    DEFAULT_QK_QUANT_GRAN,
    DEFAULT_TENSOR_LAYOUT,
    VALID_LAYOUTS,
    VALID_PV_ACCUM_DTYPES,
    VALID_QUANT_GRANS,
    validate_common_parameters,
    validate_tensors,
)


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
    validate_common_parameters(
        tensor_layout=layout,
        qk_quant_gran=gran,
        pv_accum_dtype=pv_accum_dtype,
        kwargs=kwargs,
    )
    validate_tensors(q, k, v, tensor_layout=layout, is_causal=causal)
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("forward-only operator does not support autograd")
    if kwargs.pop("dropout_p", 0.0) != 0:
        raise ValueError("dropout is unsupported")
    if kwargs:
        raise TypeError(f"unsupported arguments: {sorted(kwargs)}")
    qc, kc, vc = (
        (q, k, v) if layout == "HND" else tuple(t.transpose(1, 2) for t in (q, k, v))
    )
    if not 0 < qc.shape[-1] <= 128:
        raise ValueError("head_dim must be in [1,128]")
    return qc, kc, vc


# Register custom operator schema under sageattention2 namespace using FRAGMENT to allow multiple module imports
_library = torch.library.Library("sageattention2", "FRAGMENT")

# schema 字符串中的默认值（"HND"/"per_thread"/"fp32+fp16"）为文本，须与
# error_check.DEFAULT_TENSOR_LAYOUT / DEFAULT_QK_QUANT_GRAN / DEFAULT_PV_ACCUM_DTYPE 一致
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
        tensor_layout: str = DEFAULT_TENSOR_LAYOUT,
        is_causal: bool = False,
        qk_quant_gran: str = DEFAULT_QK_QUANT_GRAN,
        sm_scale: float | None = None,
        pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
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
        tensor_layout: str = DEFAULT_TENSOR_LAYOUT,
        is_causal: bool = False,
        qk_quant_gran: str = DEFAULT_QK_QUANT_GRAN,
        sm_scale: float | None = None,
        pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
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
        tensor_layout: str = DEFAULT_TENSOR_LAYOUT,
        is_causal: bool = False,
        qk_quant_gran: str = DEFAULT_QK_QUANT_GRAN,
        sm_scale: float | None = None,
        pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
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
    tensor_layout: str = DEFAULT_TENSOR_LAYOUT,
    is_causal: bool = False,
    qk_quant_gran: str = DEFAULT_QK_QUANT_GRAN,
    sm_scale: float | None = None,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
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
