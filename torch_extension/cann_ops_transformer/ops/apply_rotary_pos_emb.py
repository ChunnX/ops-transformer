# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


OP_NAME = "apply_rotary_pos_emb"
ROTARY_MODE_HALF = 0
ROTARY_MODE_HALF_STR = "half"
LAYOUT_BSND = 1
LAYOUT_SBND = 2
LAYOUT_BNSD = 3
LAYOUT_TND = 4
SUPPORTED_LAYOUTS = (LAYOUT_BSND, LAYOUT_SBND, LAYOUT_BNSD, LAYOUT_TND)


def _check_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int,
    rotary_mode: int,
):
    if rotary_mode != ROTARY_MODE_HALF:
        raise ValueError(
            f"apply_rotary_pos_emb: rotary_mode only supports 0 (half), got {rotary_mode}."
        )
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(
            "apply_rotary_pos_emb: layout must be one of 1(BSND), "
            f"2(SBND), 3(BNSD), 4(TND), got {layout}."
        )
    for name, tensor in (
        ("query", query),
        ("key", key),
        ("cos", cos),
        ("sin", sin),
    ):
        if tensor.numel() == 0:
            raise ValueError(f"apply_rotary_pos_emb: {name} must not be empty.")


class ApplyRotaryPosEmbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, cos, sin, layout, rotary_mode):
        _check_inputs(query, key, cos, sin, layout, rotary_mode)
        op_module = apply_rotary_pos_emb_op_builder.load()
        query_emb, key_emb = op_module.apply_rotary_pos_emb(
            query, key, cos, sin, layout, rotary_mode
        )
        ctx.save_for_backward(query, key, cos, sin)
        ctx.layout = layout
        ctx.needs_query_grad = ctx.needs_input_grad[0]
        ctx.needs_key_grad = ctx.needs_input_grad[1]
        ctx.needs_cos_grad = ctx.needs_input_grad[2]
        ctx.needs_sin_grad = ctx.needs_input_grad[3]
        return query_emb, key_emb

    @staticmethod
    def backward(ctx, grad_query_embed, grad_key_embed):
        query, key, cos, sin = ctx.saved_tensors
        if grad_query_embed is None:
            grad_query_embed = torch.zeros_like(query)
        if grad_key_embed is None:
            grad_key_embed = torch.zeros_like(key)

        from cann_ops_transformer.ops.apply_rotary_pos_emb_grad import (
            apply_rotary_pos_emb_grad,
        )

        need_cos_sin_grad = ctx.needs_cos_grad or ctx.needs_sin_grad
        grad_query, grad_key, grad_cos, grad_sin = apply_rotary_pos_emb_grad(
            grad_query_embed,
            grad_key_embed,
            cos,
            sin,
            query=query if need_cos_sin_grad else None,
            key=key if need_cos_sin_grad else None,
            rotary_mode=ROTARY_MODE_HALF_STR,
            layout=ctx.layout,
        )

        return (
            grad_query if ctx.needs_query_grad else None,
            grad_key if ctx.needs_key_grad else None,
            grad_cos if ctx.needs_cos_grad else None,
            grad_sin if ctx.needs_sin_grad else None,
            None,
            None,
        )


class ApplyRotaryPosEmbOpBuilder(OpBuilder):
    def __init__(self):
        super(ApplyRotaryPosEmbOpBuilder, self).__init__(OP_NAME)

    def sources(self):
        return ["ops/csrc/apply_rotary_pos_emb.cpp"]

    def schema(self) -> str:
        return (
            "apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, "
            "int layout, int rotary_mode) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def apply_rotary_pos_emb_meta(query, key, cos, sin, layout, rotary_mode):
            _check_inputs(query, key, cos, sin, layout, rotary_mode)
            return torch.empty_like(query), torch.empty_like(key)


apply_rotary_pos_emb_op_builder = ApplyRotaryPosEmbOpBuilder()
apply_rotary_pos_emb_op_builder.load()


@impl(AS_LIBRARY, apply_rotary_pos_emb_op_builder.name, "PrivateUse1")
def _apply_rotary_pos_emb(query, key, cos, sin, layout, rotary_mode):
    _check_inputs(query, key, cos, sin, layout, rotary_mode)
    op_module = apply_rotary_pos_emb_op_builder.load()
    return op_module.apply_rotary_pos_emb(query, key, cos, sin, layout, rotary_mode)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = LAYOUT_BSND,
    rotary_mode: int = ROTARY_MODE_HALF,
) -> Tuple[torch.Tensor, torch.Tensor]:
    needs_grad = (
        torch.is_grad_enabled()
        and (
            query.requires_grad
            or key.requires_grad
            or cos.requires_grad
            or sin.requires_grad
        )
    )
    if needs_grad:
        return ApplyRotaryPosEmbFunction.apply(
            query, key, cos, sin, layout, rotary_mode
        )
    return torch.ops.cann_ops_transformer.apply_rotary_pos_emb(
        query, key, cos, sin, layout, rotary_mode
    )
