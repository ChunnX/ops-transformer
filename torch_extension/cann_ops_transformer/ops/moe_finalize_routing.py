# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional

import torch
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class MoeFinalizeRoutingOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeFinalizeRoutingOpBuilder, self).__init__("moe_finalize_routing")

    def sources(self):
        return ["ops/csrc/moe_finalize_routing.cpp"]

    def schema(self) -> str:
        return (
            "moe_finalize_routing(Tensor expanded_x, Tensor expanded_row_idx, Tensor? x1=None, "
            "Tensor? x2=None, Tensor? bias=None, Tensor? scales=None, Tensor? expert_idx=None, "
            "int drop_pad_mode=0) -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def moe_finalize_routing_meta(
            expanded_x: torch.Tensor,
            expanded_row_idx: torch.Tensor,
            x1: Optional[torch.Tensor] = None,
            x2: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            scales: Optional[torch.Tensor] = None,
            expert_idx: Optional[torch.Tensor] = None,
            drop_pad_mode: int = 0,
        ) -> torch.Tensor:
            torch._check(
                expanded_x.dim() >= 1,
                lambda: f"expanded_x must be at least 1D, but got {expanded_x.dim()} dimensions."
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

            if scales is not None and scales.numel() > 0:
                num_rows = scales.size(0)
            else:
                num_rows = expanded_row_idx.numel()
            hidden = expanded_x.size(-1)
            return torch.empty((num_rows, hidden), dtype=expanded_x.dtype, device="meta")


moe_finalize_routing_op_builder = MoeFinalizeRoutingOpBuilder()


@impl(AS_LIBRARY, moe_finalize_routing_op_builder.name, "PrivateUse1")
def _moe_finalize_routing(
    expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    x1: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
) -> torch.Tensor:
    op_module = moe_finalize_routing_op_builder.load()
    return op_module.moe_finalize_routing(
        expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx, drop_pad_mode
    )


def _has_value(t: Optional[torch.Tensor]) -> bool:
    return t is not None and t.numel() > 0


def _compute_grad_bias(grad_out: torch.Tensor, scales: Optional[torch.Tensor],
                       expert_idx: Optional[torch.Tensor], bias: torch.Tensor) -> Optional[torch.Tensor]:
    """grad_bias[e, j] = sum_{(i,k): expert_idx[i,k]=e} scales[i,k] * grad_out[i, j].

    The aclnn grad op does not return grad_bias, so compute it via scatter-add.
    Requires expert_idx to be present; returns None otherwise.
    """
    if not _has_value(expert_idx):
        return None
    R, H = grad_out.shape[0], bias.shape[1]
    K = scales.shape[1] if _has_value(scales) else 1
    flat_idx = expert_idx.reshape(-1).to(torch.int64)
    row_idx = torch.arange(R, device=grad_out.device).repeat_interleave(K)
    if _has_value(scales):
        scales_flat = scales.reshape(-1, 1)
    else:
        scales_flat = torch.ones(R * K, 1, dtype=grad_out.dtype, device=grad_out.device)
    contrib = scales_flat * grad_out[row_idx]
    grad_bias = torch.zeros(bias.shape[0], H, dtype=bias.dtype, device=grad_out.device)
    grad_bias.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, H), contrib)
    return grad_bias


class MoeFinalizeRoutingFn(torch.autograd.Function):
    """Autograd binding: forward -> aclnnMoeFinalizeRoutingV2,
    backward -> aclnnMoeFinalizeRoutingV2Grad.

    Note: the forward op indexes ``expanded_row_idx`` with ``(K, R)`` layout
    (``idx[k*R + row]``), while the grad op expects ``(R, K)`` layout
    (``idx[row*K + k]``). Backward transposes the index layout accordingly.
    """

    @staticmethod
    def forward(ctx, expanded_x: torch.Tensor, expanded_row_idx: torch.Tensor,
                x1: Optional[torch.Tensor], x2: Optional[torch.Tensor],
                bias: Optional[torch.Tensor], scales: Optional[torch.Tensor],
                expert_idx: Optional[torch.Tensor], drop_pad_mode: int):
        ctx.drop_pad_mode = drop_pad_mode
        ctx.has_x1 = _has_value(x1)
        ctx.has_x2 = _has_value(x2)
        ctx.has_bias = _has_value(bias)
        ctx.has_scales = _has_value(scales)
        ctx.has_expert_idx = _has_value(expert_idx)
        # save_for_backward takes tensors only; use expanded_x/expanded_row_idx as
        # harmless placeholders for absent optionals so the saved list is fixed-length.
        ctx.save_for_backward(
            expanded_x, expanded_row_idx,
            scales if ctx.has_scales else expanded_x,
            bias if ctx.has_bias else expanded_x,
            expert_idx if ctx.has_expert_idx else expanded_row_idx,
        )
        return torch.ops.cann_ops_transformer.moe_finalize_routing(
            expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx, drop_pad_mode)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        expanded_x, expanded_row_idx, scales, bias, expert_idx = ctx.saved_tensors
        scales = scales if ctx.has_scales else None
        bias = bias if ctx.has_bias else None
        expert_idx = expert_idx if ctx.has_expert_idx else None
        drop_pad_mode = ctx.drop_pad_mode

        grad_out = grad_out.contiguous()
        R = grad_out.shape[0]
        K = scales.shape[1] if ctx.has_scales else 1

        # Convert expanded_row_idx from forward (K, R) layout to grad-op (R, K) layout.
        if K > 1:
            grad_row_idx = expanded_row_idx.view(K, R).t().contiguous().view(-1)
        else:
            grad_row_idx = expanded_row_idx

        grad_expanded_x, grad_scales = torch.ops.cann_ops_transformer.moe_finalize_routing_grad(
            grad_out, grad_row_idx, expanded_x, scales, expert_idx, bias,
            drop_pad_mode, 0, 0, 0)

        needs = ctx.needs_input_grad  # [expanded_x, row_idx, x1, x2, bias, scales, expert_idx, drop_pad_mode]
        grad_expanded_x = grad_expanded_x if needs[0] else None
        grad_x1 = grad_out if ctx.has_x1 and needs[2] else None
        grad_x2 = grad_out if ctx.has_x2 and needs[3] else None
        grad_bias = _compute_grad_bias(grad_out, scales, expert_idx, bias) if ctx.has_bias and needs[4] else None
        grad_scales = grad_scales if ctx.has_scales and needs[5] else None
        return grad_expanded_x, None, grad_x1, grad_x2, grad_bias, grad_scales, None, None


def moe_finalize_routing(
    expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    x1: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
) -> torch.Tensor:
    """Mixture-of-Experts finalize routing (forward), wraps aclnnMoeFinalizeRoutingV2.

    Computes::

        out(i, j) = x1(i, j) + x2(i, j)
                  + sum_k scales(i, k) * (expanded_x(expanded_row_idx[k*R + i], j)
                                          + bias(expert_idx(i, k), j))

    where ``R`` is the number of rows (``scales.shape[0]`` when ``scales`` is
    provided, else ``expanded_row_idx.numel()``) and ``K = scales.shape[1]``.

    Supports automatic differentiation: when any differentiable input
    (``expanded_x``/``x1``/``x2``/``bias``/``scales``) has
    ``requires_grad=True``, the backward pass calls
    :func:`moe_finalize_routing_grad` (wrapping ``aclnnMoeFinalizeRoutingV2Grad``)
    to compute gradients. ``grad_bias`` is computed via scatter-add since the
    aclnn grad op does not return it. ``expanded_row_idx`` and ``expert_idx``
    are integer tensors and receive no gradient.

    Args:
        expanded_x (Tensor): FFN expanded output, shape (R*K, H) when drop_pad_mode
            is 0/2 or (E, C, H) when drop_pad_mode is 1/3. dtype:
            float16/bfloat16/float32.
        expanded_row_idx (Tensor): row index, 1D of length R*K in (K, R) layout
            (``idx[k*R + row]``). dtype: int32.
        x1 (Tensor, optional): skip tensor of shape (R, H). Defaults to None.
        x2 (Tensor, optional): skip tensor of shape (R, H). Defaults to None.
        bias (Tensor, optional): expert bias of shape (E, H). Defaults to None.
        scales (Tensor, optional): per-expert scales of shape (R, K).
        expert_idx (Tensor, optional): expert indices of shape (R, K). dtype: int32.
        drop_pad_mode (int): expanded_row_idx arrangement, range [0, 3]. Defaults to 0.

    Returns:
        Tensor: output of shape (R, H), same dtype as ``expanded_x``.
    """
    return MoeFinalizeRoutingFn.apply(
        expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx, drop_pad_mode
    )
