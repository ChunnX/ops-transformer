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


class MoeInitRoutingFn(torch.autograd.Function):
    """Autograd binding: forward -> aclnnMoeInitRoutingV2,
    backward -> aclnnMoeInitRoutingV2Grad.

    Only ``expanded_x`` is differentiable (it is a permuted copy of ``x``).
    ``expanded_row_idx`` and the two token-count outputs are integer tensors
    with no gradient.  ``expert_idx`` is an integer index tensor with no
    gradient.

    The grad op computes ``grad_x`` by scattering ``grad_expanded_x`` back to
    the original rows of ``x`` according to ``expanded_row_idx`` and ``topK``.
    """

    @staticmethod
    def forward(  # noqa: huawei-too-many-arguments
        ctx,
        x,
        expert_idx,
        active_num,
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_count_or_cumsum_flag,
        expert_tokens_before_capacity_flag,
    ):
        ctx.drop_pad_mode = drop_pad_mode
        ctx.active_num = active_num
        ctx.k = expert_idx.size(1)
        expanded_x, expanded_row_idx, tokens_count, tokens_before_capacity = (
            torch.ops.cann_ops_transformer.moe_init_routing(
                x,
                expert_idx,
                active_num,
                expert_capacity,
                expert_num,
                drop_pad_mode,
                expert_tokens_count_or_cumsum_flag,
                expert_tokens_before_capacity_flag,
            )
        )
        ctx.save_for_backward(expanded_row_idx)
        return expanded_x, expanded_row_idx, tokens_count, tokens_before_capacity

    @staticmethod
    def backward(  # noqa: too-many-return-values
        ctx,
        grad_expanded_x,
        grad_expanded_row_idx,
        grad_tokens_count,
        grad_tokens_before_capacity,
    ):
        (expanded_row_idx,) = ctx.saved_tensors
        needs = ctx.needs_input_grad  # [x, expert_idx, active_num, ...]
        if grad_expanded_x is None or not needs[0]:
            return (None, None, None, None, None, None, None, None)  # noqa: too-many-return-values
        grad_expanded_x = grad_expanded_x.contiguous()
        grad_x = torch.ops.cann_ops_transformer.moe_init_routing_grad(
            grad_expanded_x, expanded_row_idx, ctx.k, ctx.drop_pad_mode, ctx.active_num
        )
        return grad_x, None, None, None, None, None, None, None  # noqa: too-many-return-values


def moe_init_routing(  # noqa: huawei-too-many-arguments
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    active_num: int = 0,
    expert_capacity: int = 0,
    expert_num: int = 0,
    drop_pad_mode: int = 0,
    expert_tokens_count_or_cumsum_flag: int = 0,
    expert_tokens_before_capacity_flag: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MoE routing (forward), wraps aclnnMoeInitRoutingV2.

    Sorts ``expert_idx``, permutes ``x`` into ``expanded_x`` and produces the
    index mapping ``expanded_row_idx``.

    Supports automatic differentiation for ``x``: the backward pass calls
    :func:`moe_init_routing_grad` (wrapping ``aclnnMoeInitRoutingV2Grad``) to
    scatter ``grad_expanded_x`` back to ``grad_x``.  ``expert_idx`` and all
    integer outputs receive no gradient.

    Args:
        x (Tensor): token features, shape (R, H). dtype: float16/bfloat16/float32.
        expert_idx (Tensor): expert indices per token, shape (R, K). dtype: int32.
        active_num (int): max active rows in expanded_x (dropless mode).
        expert_capacity (int): per-expert capacity (drop mode). Required >0 when
            ``drop_pad_mode=1``.
        expert_num (int): number of experts. Required >0 when ``drop_pad_mode=1``
            or ``expert_tokens_count_or_cumsum_flag>0`` or
            ``expert_tokens_before_capacity_flag=True``.
        drop_pad_mode (int): 0 = dropless, 1 = drop/pad.
        expert_tokens_count_or_cumsum_flag (int): 0=none, 1=cumsum, 2=count
            (dropless mode only).
        expert_tokens_before_capacity_flag (bool): output pre-drop token counts
            (drop mode only).

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            (expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum,
             expert_tokens_before_capacity). The two token-count tensors are
            empty (0-element) when the corresponding flag is inactive.
    """
    return MoeInitRoutingFn.apply(
        x,
        expert_idx,
        active_num,
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_count_or_cumsum_flag,
        expert_tokens_before_capacity_flag,
    )
