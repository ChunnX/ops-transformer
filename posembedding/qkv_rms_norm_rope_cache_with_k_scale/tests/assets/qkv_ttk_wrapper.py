#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.

"""Thin PTA wrappers used by TTK E2E cases.

Each wrapper calls exactly one public cann_ops_transformer operator.  M-RoPE's
None q_scale is filtered so the common TestSpec comparison hook can run.
"""

from typing import List, Optional, Tuple

import torch


def _torch_dtype(value, is_mrope):
    if is_mrope:
        return torch.bfloat16
    if value is None or value in (36, "36", "float8_e4m3fn", "torch.float8_e4m3fn"):
        return torch.float8_e4m3fn
    if value in (27, "27", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    return value


def _kwargs(
    rotation, v_scale, layout_qkv, layout_q_out, epsilon,
    mrope_position, mrope_section, q_quant_mode, q_out_dtype,
):
    return {
        "rotation": rotation,
        "v_scale": v_scale,
        "layout_qkv": layout_qkv,
        "layout_q_out": layout_q_out,
        "epsilon": epsilon,
        "mrope_position": mrope_position,
        "mrope_section": mrope_section,
        "q_quant_mode": q_quant_mode,
        "q_out_dtype": _torch_dtype(q_out_dtype, str(q_quant_mode) == "NoQuant"),
    }


def pta_functional(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    head_nums: List[int],
    *,
    rotation: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    layout_qkv: str = "TND",
    layout_q_out: str = "NTD",
    epsilon: float = 1e-6,
    mrope_position: Optional[torch.Tensor] = None,
    mrope_section: Optional[List[int]] = None,
    q_quant_mode: str = "PerTokenPerHead",
    q_out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, ...]:
    from cann_ops_transformer import qkv_rms_norm_rope_cache_with_k_scale

    result = qkv_rms_norm_rope_cache_with_k_scale(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, head_nums,
        **_kwargs(rotation, v_scale, layout_qkv, layout_q_out, epsilon,
                  mrope_position, mrope_section, q_quant_mode, q_out_dtype),
    )
    q_out, q_scale, k_out, v_out, k_scale_out = result
    if q_scale is None:
        return q_out, k_out, v_out, k_scale_out
    return q_out, q_scale, k_out, v_out, k_scale_out


def pta_inplace(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    head_nums: List[int],
    *,
    rotation: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    layout_qkv: str = "TND",
    layout_q_out: str = "NTD",
    epsilon: float = 1e-6,
    mrope_position: Optional[torch.Tensor] = None,
    mrope_section: Optional[List[int]] = None,
    q_quant_mode: str = "PerTokenPerHead",
    q_out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, ...]:
    from cann_ops_transformer import qkv_rms_norm_rope_cache_with_k_scale_

    q_out, q_scale = qkv_rms_norm_rope_cache_with_k_scale_(
        qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
        k_scale_cache, query_start_loc, seq_lens, head_nums,
        **_kwargs(rotation, v_scale, layout_qkv, layout_q_out, epsilon,
                  mrope_position, mrope_section, q_quant_mode, q_out_dtype),
    )
    if q_scale is None:
        return q_out
    return q_out, q_scale
