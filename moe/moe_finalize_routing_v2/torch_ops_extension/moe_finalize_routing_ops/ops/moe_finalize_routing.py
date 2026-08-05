# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, List

import torch


def _has_value(t):
    return t is not None and t.numel() > 0


def _is_valid_expert_range(r):
    """A range like [start, end) is valid (non-empty) only when start >= 0 and end > start.
    The sentinel [-1, -1] means 'no such experts' and is treated as invalid."""
    if r is None or len(r) < 2:
        return False
    return r[0] >= 0 and r[1] > r[0]


_NUM_FORWARD_INPUTS = 16


def _check_backward_supported(ctx):
    """Raise NotImplementedError if the V4-specific or unsupported drop_pad_mode path is taken."""
    if ctx.has_v4_extras:
        raise NotImplementedError(
            "moe_finalize_routing autograd is not supported when V4-specific "
            "inputs (x, alpha1, alpha2, v) or zero_expert_range are provided. "
            "The aclnnMoeFinalizeRoutingV2Grad op only handles the regular-expert case."
        )
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
        return expanded_row_idx.view(top_k, row_num).t().contiguous().view(-1)
    return expanded_row_idx


def _synthesize_scales(scales, top_k, row_num, grad_out):
    """Return effective scales, synthesizing a ones tensor when scales is None and K > 1."""
    if scales is None and top_k > 1:
        return torch.ones(row_num, top_k, dtype=grad_out.dtype, device=grad_out.device)
    return scales


def _infer_grad_shape(expanded_x, drop_pad_mode):
    """Infer (active_num, expert_num, expert_capacity) for the grad op from expanded_x shape."""
    if drop_pad_mode == 1:
        return 0, expanded_x.shape[0], expanded_x.shape[1]
    return expanded_x.shape[0], 0, 0


def _build_grad_return_tuple(grad_expanded_x, grad_scales):
    """Build the 16-element backward return tuple (only positions 0 and 5 are non-None)."""
    grads = [None] * _NUM_FORWARD_INPUTS
    grads[0] = grad_expanded_x
    grads[5] = grad_scales
    return tuple(grads)


class MoeFinalizeRoutingFn(torch.autograd.Function):
    """Autograd binding: forward -> aclnnMoeFinalizeRoutingV4,
    backward -> aclnnMoeFinalizeRoutingV2Grad.

    The V4 forward generalizes the per-expert term: regular experts use
    ``expandedX[row, j]`` (V2 behavior), copy experts use ``x[i, j]``, constant
    experts use ``a1*e'*x + a2*e'*v``, and zero experts are skipped.

    The V2 grad op only handles the regular-expert case.  Autograd is therefore
    only supported when V4 reduces to V2 — i.e. none of the V4-specific inputs
    (``x``, ``alpha1``, ``alpha2``, ``v``) are provided and ``zero_expert_range``
    is empty.  When V4-specific features are used, backward raises
    ``NotImplementedError``.

    Additionally, the grad op only supports ``drop_pad_mode`` 0 or 1
    (column-major ``expanded_row_idx`` layout); ``drop_pad_mode`` 2 or 3
    (row-major layout, as produced by ``aclnnMoeInitRoutingV2``) is not
    supported and backward raises ``NotImplementedError``.

    The V2Grad op only outputs ``grad_expanded_x`` and ``grad_scales``.
    Gradients for ``x1``, ``x2``, and ``bias`` are not computed (returned as
    ``None``).

    Note: the forward op indexes ``expanded_row_idx`` with ``(K, R)`` layout
    (``idx[k*R + row]``) when ``drop_pad_mode`` is 0 or 1, while the grad op
    expects ``(R, K)`` layout (``idx[row*K + k]``).  Backward transposes the
    index layout accordingly.
    """

    @staticmethod
    def forward(  # noqa: huawei-too-many-arguments
        ctx,
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x,
        alpha1,
        alpha2,
        v,
        drop_pad_mode,
        zero_expert_range,
        copy_expert_range,
        constant_expert_range,
        k,
    ):
        ctx.drop_pad_mode = drop_pad_mode if drop_pad_mode is not None else 0
        ctx.k = k if k is not None else 1
        ctx.has_scales = _has_value(scales)
        ctx.has_bias = _has_value(bias)
        ctx.has_expert_idx = _has_value(expert_idx)
        # V4-specific inputs — when present, autograd is not supported.
        # zero_expert_range uses [-1, -1] as sentinel for "no zero experts";
        # _is_valid_expert_range treats that as inactive.
        ctx.has_v4_extras = (
            _has_value(x)
            or _has_value(alpha1)
            or _has_value(alpha2)
            or _has_value(v)
            or _is_valid_expert_range(zero_expert_range)
        )
        ctx.save_for_backward(
            expanded_x,
            expanded_row_idx,
            scales if ctx.has_scales else expanded_x,
            bias if ctx.has_bias else expanded_x,
            expert_idx if ctx.has_expert_idx else expanded_row_idx,
        )
        return torch.ops.custom.moe_finalize_routing(
            expanded_x,
            expanded_row_idx,
            x1,
            x2,
            bias,
            scales,
            expert_idx,
            x,
            alpha1,
            alpha2,
            v,
            drop_pad_mode,
            zero_expert_range,
            copy_expert_range,
            constant_expert_range,
            k,
        )

    @staticmethod
    def backward(ctx, grad_out):
        _check_backward_supported(ctx)
        expanded_x, expanded_row_idx, scales, bias, expert_idx = ctx.saved_tensors
        scales = scales if ctx.has_scales else None
        bias = bias if ctx.has_bias else None
        expert_idx = expert_idx if ctx.has_expert_idx else None
        drop_pad_mode = ctx.drop_pad_mode

        grad_out = grad_out.contiguous()
        row_num = grad_out.shape[0]
        top_k = scales.shape[1] if ctx.has_scales else ctx.k

        grad_row_idx = _transpose_row_idx(expanded_row_idx, top_k, row_num)
        eff_scales = _synthesize_scales(scales, top_k, row_num, grad_out)
        grad_active_num, grad_expert_num, grad_expert_capacity = _infer_grad_shape(
            expanded_x, drop_pad_mode
        )

        grad_expanded_x, grad_scales = torch.ops.custom.moe_finalize_routing_grad(
            grad_out,
            grad_row_idx,
            expanded_x,
            eff_scales,
            expert_idx,
            bias,
            drop_pad_mode,
            grad_active_num,
            grad_expert_num,
            grad_expert_capacity,
        )

        needs = ctx.needs_input_grad
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
    x: Optional[torch.Tensor] = None,
    alpha1: Optional[torch.Tensor] = None,
    alpha2: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    drop_pad_mode: Optional[int] = 0,
    zero_expert_range: Optional[List[int]] = None,
    copy_expert_range: Optional[List[int]] = None,
    constant_expert_range: Optional[List[int]] = None,
    k: Optional[int] = 1,
) -> torch.Tensor:
    """Mixture-of-Experts finalize routing (forward), wraps aclnnMoeFinalizeRoutingV4.

    Computes::

        out(i, j) = x1(i, j) + x2(i, j)
                  + sum_k scales(i, k) * (T(i, k, j) + bias(expert_idx(i, k), j))

    where ``T(i, k, j)`` depends on the expert type:

      - regular expert:    ``expanded_x(expanded_row_idx[k*R + i], j)``
      - copy expert:       ``x(i, j)``
      - constant expert:   ``alpha1(e', j) * x(i, j) + alpha2(e', j) * v(e', j)``
      - zero expert:       (slot skipped, no contribution)

    Expert type ranges are defined by ``zero_expert_range``, ``copy_expert_range``,
    and ``constant_expert_range``.  When these are all empty and ``x``/``alpha1``/
    ``alpha2``/``v`` are all ``None``, V4 reduces to V2.

    Supports automatic differentiation when V4 reduces to V2 (no V4-specific
    inputs) and ``drop_pad_mode`` is 0 or 1.  The backward pass calls
    :func:`moe_finalize_routing_grad` (wrapping ``aclnnMoeFinalizeRoutingV2Grad``).
    Only ``grad_expanded_x`` and ``grad_scales`` are computed; gradients for
    ``x1``, ``x2``, and ``bias`` are not provided (returned as ``None``).
    When V4-specific inputs are provided or ``drop_pad_mode`` is 2 or 3,
    backward raises ``NotImplementedError``.

    Args:
        expanded_x (Tensor): FFN expanded output, shape (R*K, H) or (E, C, H).
        expanded_row_idx (Tensor): row index, 1D of length R*K in (K, R) layout.
        x1 (Tensor, optional): skip tensor of shape (R, H).
        x2 (Tensor, optional): skip tensor of shape (R, H).
        bias (Tensor, optional): expert bias of shape (E, H).
        scales (Tensor, optional): per-expert scales of shape (R, K).
        expert_idx (Tensor, optional): expert indices of shape (R, K). dtype: int32.
        x (Tensor, optional): shared input for copy/constant experts, shape (R, H).
        alpha1 (Tensor, optional): alpha1 for constant experts, shape (E', H).
        alpha2 (Tensor, optional): alpha2 for constant experts, shape (E', H).
        v (Tensor, optional): v for constant experts, shape (E', H).
        drop_pad_mode (int, optional): expanded_row_idx arrangement, range [0, 3].
        zero_expert_range (List[int], optional): [start, end) of zero experts.
        copy_expert_range (List[int], optional): [start, end) of copy experts.
        constant_expert_range (List[int], optional): [start, end) of constant experts.
        k (int, optional): top-K expert count. Defaults to 1.

    Returns:
        Tensor: output of shape (R, H), same dtype as ``expanded_x``.
    """
    return MoeFinalizeRoutingFn.apply(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x,
        alpha1,
        alpha2,
        v,
        drop_pad_mode,
        zero_expert_range,
        copy_expert_range,
        constant_expert_range,
        k,
    )
