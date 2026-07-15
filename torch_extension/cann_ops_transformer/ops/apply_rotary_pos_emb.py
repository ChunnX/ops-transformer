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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class ApplyRotaryPosEmbFunction(torch.autograd.Function):
    """Autograd wrapper: forward -> apply_rotary_pos_emb,
    backward -> apply_rotary_pos_emb_backward.

    Backward computes gradients for query, key, cos, and sin.
    """

    @staticmethod
    def forward(ctx, query, key, cos, sin, layout, rotary_mode):
        op_module = apply_rotary_pos_emb_op_builder.load()
        query_emb, key_emb = op_module.apply_rotary_pos_emb(
            query, key, cos, sin, layout, rotary_mode
        )
        # Save original inputs needed for backward
        ctx.save_for_backward(query, key, cos, sin)
        ctx.layout = layout
        ctx.rotary_mode = rotary_mode
        return query_emb, key_emb

    @staticmethod
    def backward(ctx, grad_query_emb, grad_key_emb):
        query, key, cos, sin = ctx.saved_tensors
        layout = ctx.layout
        rotary_mode = ctx.rotary_mode

        from cann_ops_transformer.ops.apply_rotary_pos_emb_backward import (
            apply_rotary_pos_emb_backward,
        )

        grad_query, grad_key, grad_cos, grad_sin = apply_rotary_pos_emb_backward(
            grad_query_emb, grad_key_emb, cos, sin, query, key, rotary_mode, layout
        )

        # Return grads for all forward inputs: query, key, cos, sin, layout, rotary_mode
        return grad_query, grad_key, grad_cos, grad_sin, None, None


class ApplyRotaryPosEmbOpBuilder(OpBuilder):
    def __init__(self):
        super(ApplyRotaryPosEmbOpBuilder, self).__init__("apply_rotary_pos_emb")

    def sources(self):
        return ["ops/csrc/apply_rotary_pos_emb.cpp"]

    def schema(self) -> str:
        return (
            "apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, "
            "int layout, str rotary_mode) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def apply_rotary_pos_emb_meta(query, key, cos, sin, layout, rotary_mode):
            if rotary_mode != "half":
                raise ValueError(
                    f"rotary_mode only supports 'half', got '{rotary_mode}'."
                )
            query_out = torch.empty_like(query)
            key_out = torch.empty_like(key)
            return query_out, key_out


apply_rotary_pos_emb_op_builder = ApplyRotaryPosEmbOpBuilder()


@impl(AS_LIBRARY, apply_rotary_pos_emb_op_builder.name, "PrivateUse1")
def _apply_rotary_pos_emb(query, key, cos, sin, layout, rotary_mode):
    """Dispatcher implementation for NPU (PrivateUse1 backend)."""
    op_module = apply_rotary_pos_emb_op_builder.load()
    return op_module.apply_rotary_pos_emb(
        query, key, cos, sin, layout, rotary_mode
    )


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = 1,
    rotary_mode: str = "half",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embedding to query and key tensors on NPU.

    Wraps ``aclnnAiInfraApplyRotaryPosEmb``. Supports automatic differentiation:
    when any of ``query``, ``key``, ``cos``, or ``sin`` has ``requires_grad=True``,
    the backward pass will compute gradients for all four tensors via
    ``aclnnAiInfraApplyRotaryPosEmbGrad``.

    Args:
        query (Tensor): Query tensor. dtype supports float16/bfloat16.
        key (Tensor): Key tensor. Same dtype as query.
        cos (Tensor): Cosine component of rotary embedding.
        sin (Tensor): Sine component of rotary embedding.
        layout (int): Input layout mode.
            - ``1``: BSND_BSH
            - ``2``: SBND
            - ``4``: TND
            Defaults to ``1``.
        rotary_mode (str): Rotary mode. Only ``"half"`` is currently supported.
            Defaults to ``"half"``.

    Returns:
        Tuple[Tensor, Tensor]: (query_emb, key_emb) — the rotary-transformed
        query and key tensors, same shape and dtype as inputs.
    """
    needs_grad = (
        torch.is_grad_enabled()
        and (query.requires_grad or key.requires_grad
             or cos.requires_grad or sin.requires_grad)
    )

    if needs_grad:
        return ApplyRotaryPosEmbFunction.apply(
            query, key, cos, sin, layout, rotary_mode
        )

    # Non-autograd path: call backend directly
    return torch.ops.cann_ops_transformer.apply_rotary_pos_emb(
        query, key, cos, sin, layout, rotary_mode
    )
