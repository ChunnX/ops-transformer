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


class ApplyRotaryPosEmbBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super(ApplyRotaryPosEmbBackwardOpBuilder, self).__init__(
            "apply_rotary_pos_emb_backward"
        )

    def sources(self):
        return ["ops/csrc/apply_rotary_pos_emb_backward.cpp"]

    def schema(self) -> str:
        return (
            "apply_rotary_pos_emb_backward(Tensor grad_query_embed, Tensor grad_key_embed, "
            "Tensor cos, Tensor sin, Tensor query, Tensor key, "
            "str rotary_mode, int layout) -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def apply_rotary_pos_emb_backward_meta(
            grad_query_embed,
            grad_key_embed,
            cos,
            sin,
            query,
            key,
            rotary_mode,
            layout,
        ):
            if rotary_mode != "half":
                raise ValueError(
                    f"rotary_mode only supports 'half', got '{rotary_mode}'."
                )
            if layout not in (1, 2, 4):
                raise ValueError(
                    f"Layout only supports BSND_BSH:1 / SBND:2 / TND:4, got {layout}."
                )
            grad_query = torch.empty_like(grad_query_embed)
            grad_key = torch.empty_like(grad_key_embed)
            grad_cos = torch.empty_like(cos)
            grad_sin = torch.empty_like(sin)
            return grad_query, grad_key, grad_cos, grad_sin


apply_rotary_pos_emb_backward_op_builder = ApplyRotaryPosEmbBackwardOpBuilder()


@impl(AS_LIBRARY, apply_rotary_pos_emb_backward_op_builder.name, "PrivateUse1")
def apply_rotary_pos_emb_backward(
    grad_query_embed: torch.Tensor,
    grad_key_embed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_mode: str,
    layout: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward dispatcher for apply_rotary_pos_emb (PrivateUse1 backend).

    Computes gradients for query, key, cos, and sin from the gradients of the
    forward outputs. Wraps ``aclnnAiInfraApplyRotaryPosEmbGrad``.

    Args:
        grad_query_embed (Tensor): Gradient w.r.t. the forward query_emb output.
        grad_key_embed (Tensor): Gradient w.r.t. the forward key_emb output.
        cos (Tensor): Cosine component (saved from forward).
        sin (Tensor): Sine component (saved from forward).
        query (Tensor): Original query tensor (saved from forward).
        key (Tensor): Original key tensor (saved from forward).
        rotary_mode (str): Must be ``"half"``.
        layout (int): Layout mode (1: BSND_BSH, 2: SBND, 4: TND).

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            (grad_query, grad_key, grad_cos, grad_sin).
    """
    op_module = apply_rotary_pos_emb_backward_op_builder.load()
    return op_module.apply_rotary_pos_emb_backward(
        grad_query_embed, grad_key_embed, cos, sin, query, key, rotary_mode, layout
    )
