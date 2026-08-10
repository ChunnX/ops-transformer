# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import warnings
from typing import Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class _MoeFinalizeRoutingOpBuilder(OpBuilder):
    def __init__(self):
        super(_MoeFinalizeRoutingOpBuilder, self).__init__("moe_finalize_routing")

    def sources(self):
        return ["ops/csrc/moe_finalize_routing.cpp"]

    def include_paths(self):
        paths = super().include_paths()
        paths.append(
            os.path.abspath(
                os.path.join(
                    self._package_path,
                    "..",
                    "..",
                    "moe",
                    "moe_finalize_routing_v2",
                    "op_host",
                    "op_api",
                )
            )
        )
        return paths

    def schema(self) -> str:
        return (
            "moe_finalize_routing(Tensor expanded_x, "
            "Tensor expanded_row_idx, "
            "Tensor? x1, Tensor? x2, Tensor? bias, Tensor? scales, "
            "Tensor? expert_idx, "
            "int? drop_pad_mode=0) -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def moe_finalize_routing_meta(
            expanded_x,
            expanded_row_idx,
            x1=None,
            x2=None,
            bias=None,
            scales=None,
            expert_idx=None,
            drop_pad_mode=0,
        ):
            mode = drop_pad_mode if drop_pad_mode is not None else 0
            dimm = expanded_row_idx.size(0)
            if scales is not None:
                dimm = scales.size(0)
            if mode == 1 or mode == 3:
                dimn = expanded_x.size(2)
            else:
                dimn = expanded_x.size(1)
            return expanded_x.new_empty((dimm, dimn))


_moe_finalize_routing_builder = _MoeFinalizeRoutingOpBuilder()


@impl(AS_LIBRARY, _moe_finalize_routing_builder.name, "PrivateUse1")
def _moe_finalize_routing(
    expanded_x,
    expanded_row_idx,
    x1=None,
    x2=None,
    bias=None,
    scales=None,
    expert_idx=None,
    drop_pad_mode=0,
):
    _op_module = _moe_finalize_routing_builder.load()
    return _op_module.moe_finalize_routing(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        drop_pad_mode,
    )


def _has_value(t):
    return t is not None and t.numel() > 0


_NUM_FORWARD_INPUTS = 8


def _check_backward_supported(ctx):
    """Raise NotImplementedError if an unsupported drop_pad_mode path is taken."""
    if ctx.drop_pad_mode not in (0, 1):
        raise NotImplementedError(
            "moe_finalize_routing autograd is only supported for drop_pad_mode 0 or 1 "
            "(column-major expanded_row_idx layout). The backward op "
            "(aclnnMoeFinalizeRoutingV2Grad) does not support drop_pad_mode "
            f"{ctx.drop_pad_mode}."
        )


def _transpose_row_idx(expanded_row_idx, top_k, row_num):
    """Convert expanded_row_idx from forward (K, R) layout to grad op (R, K) layout."""
    if top_k > 1:
        return expanded_row_idx.reshape(top_k, row_num).t().contiguous().reshape(-1)
    return expanded_row_idx


def _infer_grad_shape(expanded_x, drop_pad_mode):
    """Infer (active_num, expert_num, expert_capacity) for the grad op from expanded_x shape."""
    if drop_pad_mode == 1:
        return 0, expanded_x.shape[0], expanded_x.shape[1]
    return expanded_x.shape[0], 0, 0


def _build_grad_return_tuple(grad_expanded_x, grad_scales):
    """Build the 8-element backward return tuple (only positions 0 and 5 are non-None)."""
    grads = [None] * _NUM_FORWARD_INPUTS
    grads[0] = grad_expanded_x
    grads[5] = grad_scales
    return tuple(grads)


class MoeFinalizeRoutingFn(torch.autograd.Function):
    """Autograd binding: forward -> aclnnMoeFinalizeRoutingV2,
    backward -> aclnnMoeFinalizeRoutingV2Grad.

    The grad op only supports ``drop_pad_mode`` 0 or 1
    (column-major ``expanded_row_idx`` layout); ``drop_pad_mode`` 2 or 3
    (row-major layout) is not supported.

    The V2Grad op only outputs ``grad_expanded_x`` and ``grad_scales``.
    Gradients for ``x1``, ``x2``, and ``bias`` are not computed (returned as
    ``None``).

    Note: the forward op indexes ``expanded_row_idx`` with ``(K, R)`` layout
    (``idx[k*R + row]``) when ``drop_pad_mode`` is 0 or 1, while the grad op
    expects ``(R, K)`` layout (``idx[row*K + k]``).  Backward transposes the
    index layout accordingly.
    """

    @staticmethod
    def forward(
        ctx,
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        drop_pad_mode,
    ):
        ctx.drop_pad_mode = drop_pad_mode if drop_pad_mode is not None else 0
        ctx.has_scales = _has_value(scales)
        ctx.has_bias = _has_value(bias)
        ctx.has_expert_idx = _has_value(expert_idx)
        ctx.save_for_backward(
            expanded_x,
            expanded_row_idx,
            scales if ctx.has_scales else expanded_x,
            bias if ctx.has_bias else expanded_x,
            expert_idx if ctx.has_expert_idx else expanded_row_idx,
        )
        return torch.ops.cann_ops_transformer.moe_finalize_routing(
            expanded_x,
            expanded_row_idx,
            x1,
            x2,
            bias,
            scales,
            expert_idx,
            drop_pad_mode,
        )

    @staticmethod
    def backward(ctx, grad_out):
        _check_backward_supported(ctx)
        needs = ctx.needs_input_grad
        if needs[2] or needs[3]:
            warnings.warn(
                "Gradients for x1/x2 are not computed by aclnnMoeFinalizeRoutingV2Grad "
                "and will be None. If x1/x2 require gradients, use external residual "
                "addition instead of passing them as forward inputs.",
                stacklevel=1,
            )
        if not needs[0] and not (ctx.has_scales and needs[5]):
            return _build_grad_return_tuple(None, None)

        expanded_x, expanded_row_idx, scales, bias, expert_idx = ctx.saved_tensors
        scales = scales if ctx.has_scales else None
        bias = bias if ctx.has_bias else None
        expert_idx = expert_idx if ctx.has_expert_idx else None
        drop_pad_mode = ctx.drop_pad_mode

        grad_out = grad_out.contiguous()
        row_num = grad_out.shape[0]
        top_k = scales.shape[1] if ctx.has_scales else 1

        grad_row_idx = _transpose_row_idx(expanded_row_idx, top_k, row_num)
        grad_active_num, grad_expert_num, grad_expert_capacity = _infer_grad_shape(
            expanded_x, drop_pad_mode
        )

        grad_expanded_x, grad_scales = (
            torch.ops.cann_ops_transformer.moe_finalize_routing_grad(
                grad_out,
                grad_row_idx,
                expanded_x,
                scales,
                expert_idx,
                bias,
                drop_pad_mode,
                grad_active_num,
                grad_expert_num,
                grad_expert_capacity,
            )
        )

        grad_expanded_x = grad_expanded_x if needs[0] else None
        grad_scales = grad_scales if ctx.has_scales and needs[5] else None
        return _build_grad_return_tuple(grad_expanded_x, grad_scales)


def moe_finalize_routing(
    expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    x1: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    drop_pad_mode: Optional[int] = 0,
) -> torch.Tensor:
    needs_grad = expanded_x.requires_grad or (
        scales is not None and scales.requires_grad
    )
    if needs_grad:
        return MoeFinalizeRoutingFn.apply(
            expanded_x,
            expanded_row_idx,
            x1,
            x2,
            bias,
            scales,
            expert_idx,
            drop_pad_mode,
        )
    return torch.ops.cann_ops_transformer.moe_finalize_routing(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        drop_pad_mode,
    )
