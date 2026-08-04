# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple

import torch
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class MoeFinalizeRoutingGradOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeFinalizeRoutingGradOpBuilder, self).__init__("moe_finalize_routing_grad")

    def sources(self):
        return ["ops/csrc/moe_finalize_routing_grad.cpp"]

    def schema(self) -> str:
        return (
            "moe_finalize_routing_grad(Tensor grad_y, Tensor expanded_row_idx, Tensor? expanded_x=None, "
            "Tensor? scales=None, Tensor? expert_idx=None, Tensor? bias=None, int drop_pad_mode=0, "
            "int active_num=0, int expert_num=0, int expert_capacity=0) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def moe_finalize_routing_grad_meta(
            grad_y: torch.Tensor,
            expanded_row_idx: torch.Tensor,
            expanded_x: Optional[torch.Tensor] = None,
            scales: Optional[torch.Tensor] = None,
            expert_idx: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            drop_pad_mode: int = 0,
            active_num: int = 0,
            expert_num: int = 0,
            expert_capacity: int = 0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            torch._check(
                grad_y.dim() == 2,
                lambda: f"grad_y must be a 2D tensor, but got {grad_y.dim()} dimensions."
                f"{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                expanded_row_idx.dim() == 1,
                lambda: f"expanded_row_idx must be a 1D tensor, but got {expanded_row_idx.dim()} dimensions."
                f"{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                0 <= drop_pad_mode <= 3,
                lambda: f"drop_pad_mode must be in [0, 3], but got {drop_pad_mode}."
                f"{ops_error(ErrCode.VALUE)}",
            )

            hidden = grad_y.size(1)
            if drop_pad_mode == 1:
                grad_expanded_x_shape = (expert_num, expert_capacity, hidden)
            else:
                dim0 = expanded_row_idx.numel()
                if drop_pad_mode == 0 and active_num > 0 and active_num < dim0:
                    dim0 = active_num
                grad_expanded_x_shape = (dim0, hidden)

            if scales is not None and scales.numel() > 0:
                grad_scales_shape = (grad_y.size(0), scales.size(1))
                grad_scales_dtype = scales.dtype
            else:
                grad_scales_shape = (grad_y.size(0), 1)
                grad_scales_dtype = grad_y.dtype

            grad_expanded_x = torch.empty(grad_expanded_x_shape, dtype=grad_y.dtype, device="meta")
            grad_scales = torch.empty(grad_scales_shape, dtype=grad_scales_dtype, device="meta")
            return grad_expanded_x, grad_scales


moe_finalize_routing_grad_op_builder = MoeFinalizeRoutingGradOpBuilder()


@impl(AS_LIBRARY, moe_finalize_routing_grad_op_builder.name, "PrivateUse1")
def _moe_finalize_routing_grad(
    grad_y: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    expanded_x: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
    active_num: int = 0,
    expert_num: int = 0,
    expert_capacity: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    op_module = moe_finalize_routing_grad_op_builder.load()
    return op_module.moe_finalize_routing_grad(
        grad_y, expanded_row_idx, expanded_x, scales, expert_idx, bias, drop_pad_mode, active_num,
        expert_num, expert_capacity,
    )


def moe_finalize_routing_grad(
    grad_y: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    expanded_x: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
    active_num: int = 0,
    expert_num: int = 0,
    expert_capacity: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gradient of :func:`moe_finalize_routing`, wraps aclnnMoeFinalizeRoutingV2Grad.

    Args:
        grad_y (Tensor): gradient w.r.t. the forward output, shape (R, H).
        expanded_row_idx (Tensor): row index, 1D. Note: the arrangement of
            ``expanded_row_idx`` follows the convention required by the grad op
            (flat index ``row * K + k``); it may differ from the forward layout
            (``row + k * R``). dtype: int32.
        expanded_x (Tensor, optional): forward expanded_x. Defaults to None.
        scales (Tensor, optional): forward scales, shape (R, K). Defaults to None.
        expert_idx (Tensor, optional): forward expert_idx. dtype: int32.
        bias (Tensor, optional): forward bias, shape (E, H). Defaults to None.
        drop_pad_mode (int): same as forward, range [0, 3]. Defaults to 0.
        active_num (int): max output rows of grad_expanded_x when drop_pad_mode
            is 0; 0 means use expanded_row_idx length. Defaults to 0.
        expert_num (int): expert count, required when drop_pad_mode is 1.
        expert_capacity (int): expert capacity, required when drop_pad_mode is 1.

    Returns:
        Tuple[Tensor, Tensor]: (grad_expanded_x, grad_scales). grad_expanded_x
        has the same dtype as ``grad_y``; grad_scales has the dtype of ``scales``
        when provided, otherwise the dtype of ``grad_y``.
    """
    return torch.ops.cann_ops_transformer.moe_finalize_routing_grad(
        grad_y, expanded_row_idx, expanded_x, scales, expert_idx, bias, drop_pad_mode, active_num,
        expert_num, expert_capacity,
    )
