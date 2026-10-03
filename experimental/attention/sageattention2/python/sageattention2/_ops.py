"""
PyTorch custom operator registration for SageAttention2 on Ascend NPU.
"""

from __future__ import annotations

import os
import warnings

import torch
from torch.library import impl

from .error_check import (
    DEFAULT_PV_ACCUM_DTYPE,
    DEFAULT_QK_QUANT_GRAN,
    DEFAULT_TENSOR_LAYOUT,
    compute_sm_scale,
    handle_smooth_v,
    pad_head_dim,
    validate_common_parameters,
    validate_tensors,
)

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
        tensor_layout: str = DEFAULT_TENSOR_LAYOUT,
        is_causal: bool = False,
        qk_quant_gran: str = DEFAULT_QK_QUANT_GRAN,
        sm_scale: float | None = None,
        pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
        smooth_k: bool = True,
        smooth_v: bool = False,
        return_lse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        validate_common_parameters(tensor_layout, qk_quant_gran, pv_accum_dtype)
        validate_tensors(q, k, v, tensor_layout, is_causal)
        compute_sm_scale(q.shape[-1], sm_scale)
        smooth_v = handle_smooth_v(pv_accum_dtype, smooth_v)

        out = q.new_empty(q.shape)
        if return_lse:
            qc = q if tensor_layout == "HND" else q.transpose(1, 2)
            lse = q.new_empty(qc.shape[:3], dtype=torch.float32)
        else:
            lse = None
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
        from ._extension import load_extension

        validate_common_parameters(tensor_layout, qk_quant_gran, pv_accum_dtype)
        validate_tensors(q, k, v, tensor_layout, is_causal)
        smooth_v = handle_smooth_v(pv_accum_dtype, smooth_v)
        q_pad, k_pad, v_pad, head_dim_og = pad_head_dim(q, k, v)
        sm_scale_val = compute_sm_scale(head_dim_og, sm_scale)
        
        out_pad, lse = load_extension().forward(
            q_pad,
            k_pad,
            v_pad,
            tensor_layout,
            is_causal,
            qk_quant_gran,
            sm_scale_val,
            pv_accum_dtype,
            smooth_k,
            smooth_v,
            return_lse,
        )
        out = out_pad[..., :head_dim_og].contiguous()
        return out, lse

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
        from .dispatch import run_sageattn_cpu_reference

        validate_common_parameters(tensor_layout, qk_quant_gran, pv_accum_dtype)
        validate_tensors(q, k, v, tensor_layout, is_causal)
        smooth_v = handle_smooth_v(pv_accum_dtype, smooth_v)

        # 允许通过环境变量启用 stub warning
        allow_cpu_stub = os.environ.get("SAGEATTN_ALLOW_CPU_STUB", "0").lower() in ("1", "true", "yes")
        if allow_cpu_stub:
            warnings.warn(
                "SageAttention2 is running the high-fidelity CPU reference stub "
                "(SAGEATTN_ALLOW_CPU_STUB opt-in); results are NOT valid NPU precision/performance evidence.",
                UserWarning,
                stacklevel=3,
            )

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
