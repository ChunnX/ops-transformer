# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not make this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch


def moe_init_routing_grad(
    grad_expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    top_k: int,
    drop_pad_mode: int = 0,
    active_num: int = 0,
) -> torch.Tensor:
    """Gradient of :func:`moe_init_routing`, wraps aclnnMoeInitRoutingV2Grad.

    Args:
        grad_expanded_x (Tensor): gradient w.r.t. ``expanded_x``, shape
            (active_num, H) in dropless mode or (expert_num, expert_capacity, H)
            in drop mode.
        expanded_row_idx (Tensor): the ``expanded_row_idx`` output from forward,
            shape (R*K,). dtype: int32.
        top_k (int): the K value (expert_idx.shape[1] from forward).
        drop_pad_mode (int): same as forward, 0 or 1.
        active_num (int): same as forward.

    Returns:
        Tensor: ``grad_x`` of shape (R, H), same dtype as ``grad_expanded_x``.
    """
    return torch.ops.custom.moe_init_routing_grad(
        grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num,
    )
