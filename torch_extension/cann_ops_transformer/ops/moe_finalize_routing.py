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
from typing import Optional, List

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
            "Tensor? expert_idx, Tensor? x, Tensor? alpha1, Tensor? alpha2, Tensor? v, "
            "int? drop_pad_mode=0, "
            "int[]? zero_expert_range=None, int[]? copy_expert_range=None, "
            "int[]? constant_expert_range=None, int? k=1) -> Tensor"
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
            x=None,
            alpha1=None,
            alpha2=None,
            v=None,
            drop_pad_mode=0,
            zero_expert_range=None,
            copy_expert_range=None,
            constant_expert_range=None,
            k=1,
        ):
            k_val = k if k is not None else 1
            mode = drop_pad_mode if drop_pad_mode is not None else 0
            dimm = expanded_row_idx.size(0)
            if scales is not None:
                dimm = scales.size(0)
            elif k_val > 0:
                dimm = dimm // k_val
            if mode == 1 or mode == 3:
                dimn = expanded_x.size(2)
            else:
                dimn = expanded_x.size(1)
            return expanded_x.new_empty((dimm, dimn))


_moe_finalize_routing_builder = _MoeFinalizeRoutingOpBuilder()
_op_module = _moe_finalize_routing_builder.load()


@impl(AS_LIBRARY, _moe_finalize_routing_builder.name, "PrivateUse1")
def _moe_finalize_routing(
    expanded_x,
    expanded_row_idx,
    x1=None,
    x2=None,
    bias=None,
    scales=None,
    expert_idx=None,
    x=None,
    alpha1=None,
    alpha2=None,
    v=None,
    drop_pad_mode=0,
    zero_expert_range=None,
    copy_expert_range=None,
    constant_expert_range=None,
    k=1,
):
    return _op_module.moe_finalize_routing(
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


def _has_value(t):
    return t is not None and t.numel() > 0


def _is_valid_expert_range(r):
    """A range like [start, end) is valid (non-empty) only when start >= 0 and end > start.
    The sentinel [-1, -1] means 'no such experts' and is treated as invalid."""
    if r is None or len(r) < 2:
        return False
    return r[0] >= 0 and r[1] > r[0]


def _compute_grad_bias(grad_out, scales, expert_idx, bias):
    """grad_bias[e, j] = sum_{(i,k): expert_idx[i,k]=e} scales[i,k] * grad_out[i, j]."""
    if not _has_value(expert_idx):
        return None
    R, H = grad_out.shape[0], bias.shape[1]
    K = scales.shape[1] if _has_value(scales) else 1
    flat_idx = expert_idx.reshape(-1).to(torch.int64)
    if _has_value(scales):
        # (R, K, 1) * (R, 1, H) -> (R, K, H) -> (R*K, H) via broadcasting
        contrib = (scales.reshape(R, K, 1) * grad_out.unsqueeze(1)).reshape(R * K, H)
    else:
        contrib = grad_out.repeat_interleave(K, dim=0)
    grad_bias = torch.zeros(bias.shape[0], H, dtype=contrib.dtype, device=grad_out.device)
    grad_bias.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, H), contrib)
    return grad_bias.to(bias.dtype)


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

    Note: the forward op indexes ``expanded_row_idx`` with ``(K, R)`` layout
    (``idx[k*R + row]``) when ``drop_pad_mode`` is 0 or 1, while the grad op
    expects ``(R, K)`` layout (``idx[row*K + k]``).  Backward transposes the
    index layout accordingly.
    """

    @staticmethod
    def forward(ctx, expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx,
                x, alpha1, alpha2, v, drop_pad_mode, zero_expert_range, copy_expert_range,
                constant_expert_range, k):
        ctx.drop_pad_mode = drop_pad_mode if drop_pad_mode is not None else 0
        ctx.k = k if k is not None else 1
        ctx.has_x1 = _has_value(x1)
        ctx.has_x2 = _has_value(x2)
        ctx.has_bias = _has_value(bias)
        ctx.has_scales = _has_value(scales)
        ctx.has_expert_idx = _has_value(expert_idx)
        # V4-specific inputs — when present, autograd is not supported.
        # zero_expert_range uses [-1, -1] as sentinel for "no zero experts";
        # _is_valid_expert_range treats that as inactive.
        ctx.has_v4_extras = (
            _has_value(x) or _has_value(alpha1) or _has_value(alpha2) or _has_value(v)
            or _is_valid_expert_range(zero_expert_range)
        )
        ctx.save_for_backward(
            expanded_x, expanded_row_idx,
            scales if ctx.has_scales else expanded_x,
            bias if ctx.has_bias else expanded_x,
            expert_idx if ctx.has_expert_idx else expanded_row_idx,
        )
        return torch.ops.cann_ops_transformer.moe_finalize_routing(
            expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx,
            x, alpha1, alpha2, v, drop_pad_mode, zero_expert_range, copy_expert_range,
            constant_expert_range, k)

    @staticmethod
    def backward(ctx, grad_out):
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

        expanded_x, expanded_row_idx, scales, bias, expert_idx = ctx.saved_tensors
        scales = scales if ctx.has_scales else None
        bias = bias if ctx.has_bias else None
        expert_idx = expert_idx if ctx.has_expert_idx else None
        drop_pad_mode = ctx.drop_pad_mode

        grad_out = grad_out.contiguous()
        R = grad_out.shape[0]
        K = scales.shape[1] if ctx.has_scales else ctx.k

        if K > 1:
            grad_row_idx = expanded_row_idx.view(K, R).t().contiguous().view(-1)
        else:
            grad_row_idx = expanded_row_idx

        # When scales was not provided in forward but K > 1, the grad op cannot
        # infer K from scales.  Synthesize a ones tensor so the grad op sees a
        # valid (R, K) scales — mathematically equivalent since the implicit
        # scale is 1.0.
        eff_scales = scales
        if scales is None and K > 1:
            eff_scales = torch.ones(R, K, dtype=grad_out.dtype, device=grad_out.device)

        # Infer grad_expanded_x shape from the saved forward expanded_x.
        # The grad op uses (expert_num, expert_capacity) when drop_pad_mode == 1
        # (expert packed) to allocate a 3D (E, C, H) output, and uses active_num
        # when drop_pad_mode == 0 (dropless) to optionally truncate the output
        # to fewer rows than expanded_row_idx.numel().
        if drop_pad_mode == 1:
            grad_expert_num = expanded_x.shape[0]
            grad_expert_capacity = expanded_x.shape[1]
            grad_active_num = 0
        else:
            grad_expert_num = 0
            grad_expert_capacity = 0
            grad_active_num = expanded_x.shape[0]

        grad_expanded_x, grad_scales = torch.ops.cann_ops_transformer.moe_finalize_routing_grad(
            grad_out, grad_row_idx, expanded_x, eff_scales, expert_idx, bias,
            drop_pad_mode, grad_active_num, grad_expert_num, grad_expert_capacity)

        # For grad_bias, use eff_scales (which is correct for both cases):
        # - scales provided: eff_scales == scales
        # - scales=None, K>1: eff_scales is ones(R,K), so grad_bias sums all K slots
        # - scales=None, K=1: eff_scales is None, _compute_grad_bias uses K=1
        needs = ctx.needs_input_grad
        # 16 inputs: expanded_x, row_idx, x1, x2, bias, scales, expert_idx,
        #            x, alpha1, alpha2, v, drop_pad_mode,
        #            zero_expert_range, copy_expert_range, constant_expert_range, k
        grad_expanded_x = grad_expanded_x if needs[0] else None
        grad_x1 = grad_out if ctx.has_x1 and needs[2] else None
        grad_x2 = grad_out if ctx.has_x2 and needs[3] else None
        grad_bias = _compute_grad_bias(grad_out, eff_scales, expert_idx, bias) if ctx.has_bias and needs[4] else None
        grad_scales = grad_scales if ctx.has_scales and needs[5] else None
        return (grad_expanded_x, None, grad_x1, grad_x2, grad_bias, grad_scales, None,
                None, None, None, None, None, None, None, None, None)


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
    ``grad_bias`` is computed via scatter-add.  When V4-specific inputs are
    provided or ``drop_pad_mode`` is 2 or 3, backward raises
    ``NotImplementedError``.

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
        expanded_x, expanded_row_idx, x1, x2, bias, scales, expert_idx,
        x, alpha1, alpha2, v, drop_pad_mode, zero_expert_range, copy_expert_range,
        constant_expert_range, k,
    )
