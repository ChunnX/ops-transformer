# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


OP_NAME = "qkv_rms_norm_rope_cache_with_k_scale"
INPLACE_OP_NAME = "qkv_rms_norm_rope_cache_with_k_scale_"
QKV_LAYOUT_TND = "TND"
QKV_LAYOUT_NTD = "NTD"
GE_DTYPE_BF16 = 27
GE_DTYPE_FLOAT8_E4M3FN = 36
Q_QUANT_PER_TOKEN_PER_HEAD = "PerTokenPerHead"
Q_QUANT_NO_QUANT = "NoQuant"


class QkvRmsNormRopeCacheWithKScaleOpBuilder(OpBuilder):
    def __init__(self):
        super(QkvRmsNormRopeCacheWithKScaleOpBuilder, self).__init__(OP_NAME)

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/qkv_rms_norm_rope_cache_with_k_scale.cpp"]

    def schema(self) -> List[str]:
        """PyTorch operator signatures."""
        return [
            "qkv_rms_norm_rope_cache_with_k_scale_("
            "Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, "
            "Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor(c!) k_scale_cache, "
            "Tensor? query_start_loc, Tensor? seq_lens, int[] head_nums, "
            "*, "
            "Tensor? rotation=None, Tensor? v_scale=None, "
            'str? layout_qkv="TND", str? layout_q_out="NTD", '
            "float epsilon=0.000001, Tensor? mrope_position=None, int[]? mrope_section=None, "
            'str q_quant_mode="PerTokenPerHead", int q_out_dtype=36) -> (Tensor, Tensor?)',
            "qkv_rms_norm_rope_cache_with_k_scale("
            "Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, "
            "Tensor k_cache, Tensor v_cache, Tensor k_scale_cache, "
            "Tensor? query_start_loc, Tensor? seq_lens, int[] head_nums, "
            "*, "
            "Tensor? rotation=None, Tensor? v_scale=None, "
            'str? layout_qkv="TND", str? layout_q_out="NTD", '
            "float epsilon=0.000001, Tensor? mrope_position=None, int[]? mrope_section=None, "
            'str q_quant_mode="PerTokenPerHead", int q_out_dtype=36) -> '
            "(Tensor, Tensor?, Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(AS_LIBRARY, INPLACE_OP_NAME, "Meta")
        def qkv_rms_norm_rope_cache_with_k_scale_inplace_meta(
            qkv,
            q_gamma,
            k_gamma,
            cos_sin,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale_cache,
            query_start_loc,
            seq_lens,
            head_nums,
            *,
            rotation=None,
            v_scale=None,
            layout_qkv=QKV_LAYOUT_TND,
            layout_q_out=QKV_LAYOUT_NTD,
            epsilon=1e-6,
            mrope_position=None,
            mrope_section=None,
            q_quant_mode=Q_QUANT_PER_TOKEN_PER_HEAD,
            q_out_dtype=GE_DTYPE_FLOAT8_E4M3FN,
        ):
            return _qkv_rms_norm_rope_cache_with_k_scale_meta_outputs(
                qkv,
                head_nums,
                layout_qkv,
                layout_q_out,
                query_start_loc,
                seq_lens,
                mrope_position,
                mrope_section,
                q_quant_mode,
                q_out_dtype,
                k_cache,
                v_cache,
                k_scale_cache,
            )

        @impl(AS_LIBRARY, OP_NAME, "Meta")
        def qkv_rms_norm_rope_cache_with_k_scale_meta(
            qkv,
            q_gamma,
            k_gamma,
            cos_sin,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale_cache,
            query_start_loc,
            seq_lens,
            head_nums,
            *,
            rotation=None,
            v_scale=None,
            layout_qkv=QKV_LAYOUT_TND,
            layout_q_out=QKV_LAYOUT_NTD,
            epsilon=1e-6,
            mrope_position=None,
            mrope_section=None,
            q_quant_mode=Q_QUANT_PER_TOKEN_PER_HEAD,
            q_out_dtype=GE_DTYPE_FLOAT8_E4M3FN,
        ):
            q_out, q_scale = _qkv_rms_norm_rope_cache_with_k_scale_meta_outputs(
                qkv,
                head_nums,
                layout_qkv,
                layout_q_out,
                query_start_loc,
                seq_lens,
                mrope_position,
                mrope_section,
                q_quant_mode,
                q_out_dtype,
                k_cache,
                v_cache,
                k_scale_cache,
            )
            return (
                q_out,
                q_scale,
                torch.empty_like(k_cache),
                torch.empty_like(v_cache),
                torch.empty_like(k_scale_cache),
            )


def _get_q_head_num_for_meta(head_nums):
    if head_nums is None:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: head_nums must not be None"
        )
    if len(head_nums) < 1:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: head_nums must contain q head num"
        )

    n_q = head_nums[0]
    if n_q <= 0:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: head_nums[0] must be greater than 0"
        )
    return n_q


def _normalize_layout_for_meta(layout, default_layout, attr_name):
    layout = default_layout if layout is None or layout == "" else layout
    if layout not in (QKV_LAYOUT_TND, QKV_LAYOUT_NTD):
        raise RuntimeError(
            f"qkv_rms_norm_rope_cache_with_k_scale: {attr_name} must be TND or NTD"
        )
    return layout


def _normalize_meta_layouts(layout_qkv, layout_q_out):
    return (
        _normalize_layout_for_meta(layout_qkv, QKV_LAYOUT_TND, "layout_qkv"),
        _normalize_layout_for_meta(layout_q_out, QKV_LAYOUT_NTD, "layout_q_out"),
    )


def _normalize_q_quant_mode_for_meta(q_quant_mode):
    if q_quant_mode == "":
        return Q_QUANT_PER_TOKEN_PER_HEAD
    if q_quant_mode not in (Q_QUANT_PER_TOKEN_PER_HEAD, Q_QUANT_NO_QUANT):
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: q_quant_mode must be "
            "PerTokenPerHead or NoQuant"
        )
    return q_quant_mode


def _get_qkv_shape_for_meta(qkv, layout_qkv, n_q):
    if qkv.dim() < 3:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: qkv must be at least 3D"
        )

    is_ntd = layout_qkv == QKV_LAYOUT_NTD
    head_axis = 0 if is_ntd else 1
    token_axis = 1 if is_ntd else 0
    token_num = qkv.size(token_axis)
    head_size = qkv.size(2)
    if token_num <= 0 or head_size <= 0:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: qkv token and head dimensions must be positive"
        )
    if qkv.size(head_axis) < n_q:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: qkv head dimension must be greater than or equal to head_nums[0]"
        )
    return token_num, head_size


def _make_meta_output_tensors(
    n_q, token_num, head_size, layout_q_out, q_quant_mode, q_out_dtype
):
    if layout_q_out == QKV_LAYOUT_NTD:
        q_out_shape = (n_q, token_num, head_size)
        q_scale_shape = (n_q, token_num)
    else:
        q_out_shape = (token_num, n_q, head_size)
        q_scale_shape = (token_num, n_q)

    if q_quant_mode == Q_QUANT_PER_TOKEN_PER_HEAD and q_out_dtype == GE_DTYPE_FLOAT8_E4M3FN:
        q_out_dtype_torch = torch.float8_e4m3fn
        q_scale = torch.empty(q_scale_shape, dtype=torch.float32, device="meta")
    elif q_quant_mode == Q_QUANT_NO_QUANT and q_out_dtype == GE_DTYPE_BF16:
        q_out_dtype_torch = torch.bfloat16
        q_scale = None
    else:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: unsupported q_quant_mode/q_out_dtype combination"
        )
    q_out = torch.empty(q_out_shape, dtype=q_out_dtype_torch, device="meta")
    return q_out, q_scale


def _validate_meta_scene(
    k_cache,
    v_cache,
    k_scale_cache,
    token_num,
    head_size,
    layout_qkv,
    layout_q_out,
    query_start_loc,
    seq_lens,
    mrope_position,
    mrope_section,
    q_quant_mode,
    q_out_dtype,
):
    if v_cache.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: v_cache must be FP8 E4M3FN"
        )
    if k_scale_cache.dtype != torch.float32:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: k_scale_cache must be FP32"
        )
    query_source_present = query_start_loc is not None or seq_lens is not None
    query_source_complete = query_start_loc is not None and seq_lens is not None
    mrope_source_present = mrope_position is not None or mrope_section is not None
    mrope_source_complete = mrope_position is not None and mrope_section is not None
    if query_source_present != query_source_complete:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: query_start_loc and seq_lens must be provided together"
        )
    if mrope_source_present != mrope_source_complete:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: mrope_position and mrope_section must be provided together"
        )
    if query_source_complete == mrope_source_complete:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: exactly one position source must be provided"
        )

    if query_source_complete:
        if q_quant_mode != Q_QUANT_PER_TOKEN_PER_HEAD or q_out_dtype != GE_DTYPE_FLOAT8_E4M3FN:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: legacy mode requires "
                "PerTokenPerHead and FP8 E4M3FN"
            )
        if layout_qkv == QKV_LAYOUT_NTD and layout_q_out != QKV_LAYOUT_NTD:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: legacy NTD input only supports NTD output"
            )
        if query_start_loc.dim() != 1 or query_start_loc.size(0) < 2:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: query_start_loc must be 1D with length >= 2"
            )
        if seq_lens.dim() != 1 or seq_lens.size(0) != query_start_loc.size(0) - 1:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: seq_lens must match query_start_loc batches"
            )
        if query_start_loc.dtype != torch.int32 or seq_lens.dtype != torch.int32:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: legacy position tensors must be int32"
            )
        if k_cache.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: legacy k_cache must be FP8 E4M3FN"
            )
    else:
        if q_quant_mode != Q_QUANT_NO_QUANT or q_out_dtype != GE_DTYPE_BF16:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: M-RoPE mode requires NoQuant and BF16"
            )
        if layout_qkv != QKV_LAYOUT_TND or layout_q_out != QKV_LAYOUT_TND:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: M-RoPE mode only supports TND to TND"
            )
        if head_size != 128:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: M-RoPE mode requires head dimension 128"
            )
        if mrope_position.dim() != 2 or mrope_position.size(0) != 3 or mrope_position.size(1) != token_num:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: mrope_position must have shape [3, T]"
            )
        if mrope_position.dtype != torch.int32:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: mrope_position must be int32"
            )
        if len(mrope_section) != 3:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: mrope_section must have exactly 3 elements"
            )
        section_caps = (22, 21, 21)
        if any(
            not isinstance(value, int) or value < 0 or value > cap
            for value, cap in zip(mrope_section, section_caps)
        ):
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: mrope_section exceeds the D128 limits"
            )
        if k_cache.dtype != torch.int8:
            raise RuntimeError(
                "qkv_rms_norm_rope_cache_with_k_scale: M-RoPE k_cache must be int8"
            )


def _qkv_rms_norm_rope_cache_with_k_scale_meta_outputs(
    qkv,
    head_nums,
    layout_qkv,
    layout_q_out,
    query_start_loc,
    seq_lens,
    mrope_position,
    mrope_section,
    q_quant_mode,
    q_out_dtype,
    k_cache,
    v_cache,
    k_scale_cache,
):
    n_q = _get_q_head_num_for_meta(head_nums)
    layout_qkv, layout_q_out = _normalize_meta_layouts(layout_qkv, layout_q_out)
    q_quant_mode = _normalize_q_quant_mode_for_meta(q_quant_mode)
    token_num, head_size = _get_qkv_shape_for_meta(qkv, layout_qkv, n_q)
    _validate_meta_scene(
        k_cache,
        v_cache,
        k_scale_cache,
        token_num,
        head_size,
        layout_qkv,
        layout_q_out,
        query_start_loc,
        seq_lens,
        mrope_position,
        mrope_section,
        q_quant_mode,
        q_out_dtype,
    )
    return _make_meta_output_tensors(
        n_q, token_num, head_size, layout_q_out, q_quant_mode, q_out_dtype
    )


qkv_rms_norm_rope_cache_with_k_scale_op_builder = (
    QkvRmsNormRopeCacheWithKScaleOpBuilder()
)


@impl(AS_LIBRARY, INPLACE_OP_NAME, "PrivateUse1")
def qkv_rms_norm_rope_cache_with_k_scale_(
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
    layout_qkv: Optional[str] = QKV_LAYOUT_TND,
    layout_q_out: Optional[str] = QKV_LAYOUT_NTD,
    epsilon: float = 1e-6,
    mrope_position: Optional[torch.Tensor] = None,
    mrope_section: Optional[List[int]] = None,
    q_quant_mode: str = Q_QUANT_PER_TOKEN_PER_HEAD,
    q_out_dtype: int = GE_DTYPE_FLOAT8_E4M3FN,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Run Q/K/V RMSNorm, RoPE, scene-specific quantization, and in-place KV cache update.
    """
    op_module = qkv_rms_norm_rope_cache_with_k_scale_op_builder.load()
    return op_module.qkv_rms_norm_rope_cache_with_k_scale_(
        qkv,
        q_gamma,
        k_gamma,
        cos_sin,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale_cache,
        query_start_loc,
        seq_lens,
        head_nums,
        layout_qkv,
        layout_q_out,
        rotation,
        v_scale,
        epsilon,
        mrope_position,
        mrope_section,
        q_quant_mode,
        q_out_dtype,
    )


@impl(AS_LIBRARY, OP_NAME, "PrivateUse1")
def qkv_rms_norm_rope_cache_with_k_scale(
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
    layout_qkv: Optional[str] = QKV_LAYOUT_TND,
    layout_q_out: Optional[str] = QKV_LAYOUT_NTD,
    epsilon: float = 1e-6,
    mrope_position: Optional[torch.Tensor] = None,
    mrope_section: Optional[List[int]] = None,
    q_quant_mode: str = Q_QUANT_PER_TOKEN_PER_HEAD,
    q_out_dtype: int = GE_DTYPE_FLOAT8_E4M3FN,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Functional variant returning cloned cache outputs instead of mutating caller-visible caches.
    """
    op_module = qkv_rms_norm_rope_cache_with_k_scale_op_builder.load()
    return op_module.qkv_rms_norm_rope_cache_with_k_scale(
        qkv,
        q_gamma,
        k_gamma,
        cos_sin,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale_cache,
        query_start_loc,
        seq_lens,
        head_nums,
        layout_qkv,
        layout_q_out,
        rotation,
        v_scale,
        epsilon,
        mrope_position,
        mrope_section,
        q_quant_mode,
        q_out_dtype,
    )
