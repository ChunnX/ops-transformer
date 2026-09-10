#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Kernel, ACLNN and E2E TestSpecs for BlockAttnResUpdate."""

import numpy


_FP32_TINY = numpy.float32(numpy.finfo(numpy.float32).tiny)
_VREG_FP32_ELEMENTS = 64


__spec__ = {
    "block_attn_res_update": "BlockAttnResUpdateTestSpec",
    "aclnnBlockAttnResUpdate": "BlockAttnResUpdateAclnnTestSpec",
    "cann_ops_transformer.ops.block_attn_res_update": "BlockAttnResUpdateE2ETestSpec",
}


def _round_to_bfloat16_float32(value):
    """Round FP32 values to BF16 (RNE) in an FP32 NumPy container."""
    value = numpy.ascontiguousarray(value, dtype=numpy.float32)
    bits = value.view(numpy.uint32)
    nan_mask = numpy.isnan(value)
    rounding_bias = numpy.uint32(0x7FFF) + (
        (bits >> numpy.uint32(16)) & numpy.uint32(1)
    )
    rounded_bits = (bits + rounding_bias) & numpy.uint32(0xFFFF0000)
    # Keep NaNs as NaNs even when all payload bits are in the truncated half.
    rounded_bits = numpy.where(
        nan_mask,
        (bits & numpy.uint32(0xFFFF0000)) | numpy.uint32(0x00010000),
        rounded_bits,
    ).astype(numpy.uint32, copy=False)
    return rounded_bits.view(numpy.float32)


def _ftz_float32(value):
    """Flush FP32 subnormal values to signed zero."""
    value = numpy.asarray(value, dtype=numpy.float32)
    signed_zero = numpy.copysign(numpy.float32(0.0), value)
    return numpy.where(numpy.abs(value) < _FP32_TINY, signed_zero, value).astype(
        numpy.float32, copy=False
    )


def _div_ftz_float32(lhs, rhs):
    """Model Ascend 950's default FP32 Div behavior with ``--cce-ftz=true``."""
    lhs = _ftz_float32(lhs)
    rhs = _ftz_float32(rhs)
    with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
        quotient = numpy.divide(lhs, rhs)
    return _ftz_float32(quotient)


def _sqrt_ftz_float32(value):
    """Model Ascend 950's default FP32 Sqrt behavior with ``--cce-ftz=true``."""
    value = _ftz_float32(value)
    with numpy.errstate(invalid="ignore", under="ignore"):
        result = numpy.sqrt(value)
    return _ftz_float32(result)


def _exp_sub_ftz_float32(lhs, rhs):
    """Model the default-FTZ ``ExpSub(lhs, rhs)`` instruction boundary."""
    lhs = _ftz_float32(lhs)
    rhs = _ftz_float32(rhs)
    difference = _ftz_float32(lhs - rhs)
    with numpy.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = numpy.exp(difference).astype(numpy.float32, copy=False)
    return _ftz_float32(result)


def _fma_float32(dst, src0, src1):
    """Model ``Reg::MulDstAdd<float>`` with one FP32 rounding."""
    dst = numpy.asarray(dst, dtype=numpy.float32)
    src0 = numpy.asarray(src0, dtype=numpy.float32)
    src1 = numpy.asarray(src1, dtype=numpy.float32)
    with numpy.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = dst.astype(numpy.float64) * src0.astype(numpy.float64) + src1.astype(
            numpy.float64
        )
    return result.astype(numpy.float32, copy=False)


def _multiply_float32(lhs, rhs):
    """Multiply with an FP32 result, matching ``Reg::Mul<float>``."""
    with numpy.errstate(over="ignore", invalid="ignore", under="ignore"):
        return numpy.multiply(lhs, rhs, dtype=numpy.float32)


def _add_float32(lhs, rhs):
    """Add with an FP32 result, matching ``Reg::Add<float>``."""
    with numpy.errstate(over="ignore", invalid="ignore", under="ignore"):
        return numpy.add(lhs, rhs, dtype=numpy.float32)


def _binary_tree_sum_float32(value):
    """Match the adjacent-pair binary tree used by ``Reg::Reduce<SUM>``."""
    result = numpy.asarray(value, dtype=numpy.float32)
    while result.shape[-1] > 1:
        result = _add_float32(result[..., 0::2], result[..., 1::2])
    return result[..., 0]


def _phase1_reductions_float32(partial, query):
    """Match the D-dependent Phase 1 VREG accumulation and reduction order."""
    d_size = partial.shape[-1]
    vector_count = (d_size + _VREG_FP32_ELEMENTS - 1) // _VREG_FP32_ELEMENTS
    padded_d = vector_count * _VREG_FP32_ELEMENTS

    partial_padded = numpy.zeros(partial.shape[:-1] + (padded_d,), dtype=numpy.float32)
    partial_padded[..., :d_size] = partial
    query_padded = numpy.zeros((padded_d,), dtype=numpy.float32)
    query_padded[:d_size] = query
    partial_vectors = partial_padded.reshape(
        partial.shape[:-1] + (vector_count, _VREG_FP32_ELEMENTS)
    )
    query_vectors = query_padded.reshape(vector_count, _VREG_FP32_ELEMENTS)

    partial_reg0 = partial_vectors[..., 0, :]
    query_reg0 = query_vectors[0]
    if d_size <= _VREG_FP32_ELEMENTS:
        square_reg = _multiply_float32(partial_reg0, partial_reg0)
        dot_reg = _multiply_float32(partial_reg0, query_reg0)
    elif d_size <= 2 * _VREG_FP32_ELEMENTS:
        partial_reg1 = partial_vectors[..., 1, :]
        query_reg1 = query_vectors[1]
        square_reg = _multiply_float32(partial_reg0, partial_reg0)
        dot_reg = _multiply_float32(partial_reg0, query_reg0)
        # Reg::MulAddDst(dst, src0, src1) computes one fused dst + src0 * src1.
        square_reg = _fma_float32(partial_reg1, partial_reg1, square_reg)
        dot_reg = _fma_float32(partial_reg1, query_reg1, dot_reg)
    else:
        accumulator_shape = partial.shape[:-1] + (_VREG_FP32_ELEMENTS,)
        square_acc0 = numpy.zeros(accumulator_shape, dtype=numpy.float32)
        square_acc1 = numpy.zeros(accumulator_shape, dtype=numpy.float32)
        dot_acc0 = numpy.zeros(accumulator_shape, dtype=numpy.float32)
        dot_acc1 = numpy.zeros(accumulator_shape, dtype=numpy.float32)
        full_loops = d_size // _VREG_FP32_ELEMENTS
        pair_full_loops = full_loops // 2

        for loop in range(pair_full_loops):
            vector_index0 = 2 * loop
            vector_index1 = vector_index0 + 1
            partial_reg0 = partial_vectors[..., vector_index0, :]
            partial_reg1 = partial_vectors[..., vector_index1, :]
            square_acc0 = _fma_float32(partial_reg0, partial_reg0, square_acc0)
            square_acc1 = _fma_float32(partial_reg1, partial_reg1, square_acc1)
            dot_acc0 = _fma_float32(
                partial_reg0, query_vectors[vector_index0], dot_acc0
            )
            dot_acc1 = _fma_float32(
                partial_reg1, query_vectors[vector_index1], dot_acc1
            )

        if full_loops & 1:
            vector_index = 2 * pair_full_loops
            partial_reg0 = partial_vectors[..., vector_index, :]
            square_acc0 = _fma_float32(partial_reg0, partial_reg0, square_acc0)
            dot_acc0 = _fma_float32(partial_reg0, query_vectors[vector_index], dot_acc0)

        if d_size % _VREG_FP32_ELEMENTS:
            vector_index = full_loops
            partial_reg0 = partial_vectors[..., vector_index, :]
            if full_loops & 1:
                square_acc1 = _fma_float32(partial_reg0, partial_reg0, square_acc1)
                dot_acc1 = _fma_float32(
                    partial_reg0, query_vectors[vector_index], dot_acc1
                )
            else:
                square_acc0 = _fma_float32(partial_reg0, partial_reg0, square_acc0)
                dot_acc0 = _fma_float32(
                    partial_reg0, query_vectors[vector_index], dot_acc0
                )

        square_reg = _add_float32(square_acc0, square_acc1)
        dot_reg = _add_float32(dot_acc0, dot_acc1)

    return _binary_tree_sum_float32(square_reg), _binary_tree_sum_float32(dot_reg)


def _torch_ftz_float32(value):
    """Torch counterpart of :func:`_ftz_float32`."""
    import torch

    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny, dtype=value.dtype, device=value.device
    )
    signed_zero = torch.copysign(torch.zeros_like(value), value)
    return torch.where(torch.abs(value) < tiny, signed_zero, value)


def _torch_div_ftz_float32(lhs, rhs):
    """Torch counterpart of :func:`_div_ftz_float32`."""
    import torch

    quotient = torch.div(_torch_ftz_float32(lhs), _torch_ftz_float32(rhs))
    return _torch_ftz_float32(quotient)


def _torch_sqrt_ftz_float32(value):
    """Torch counterpart of :func:`_sqrt_ftz_float32`."""
    import torch

    return _torch_ftz_float32(torch.sqrt(_torch_ftz_float32(value)))


def _torch_exp_sub_ftz_float32(lhs, rhs):
    """Torch counterpart of :func:`_exp_sub_ftz_float32`."""
    import torch

    difference = _torch_ftz_float32(_torch_ftz_float32(lhs) - _torch_ftz_float32(rhs))
    return _torch_ftz_float32(torch.exp(difference))


def _torch_fma_float32(dst, src0, src1):
    """Torch counterpart of :func:`_fma_float32`."""
    import torch

    result = dst.to(dtype=torch.float64) * src0.to(dtype=torch.float64) + src1.to(
        dtype=torch.float64
    )
    return result.to(dtype=torch.float32)


def _torch_binary_tree_sum_float32(value):
    """Torch counterpart of :func:`_binary_tree_sum_float32`."""
    result = value
    while result.shape[-1] > 1:
        result = result[..., 0::2] + result[..., 1::2]
    return result[..., 0]


def _torch_phase1_reductions_float32(partial, query):
    """Torch counterpart of :func:`_phase1_reductions_float32`."""
    import torch

    d_size = partial.shape[-1]
    vector_count = (d_size + _VREG_FP32_ELEMENTS - 1) // _VREG_FP32_ELEMENTS
    padded_d = vector_count * _VREG_FP32_ELEMENTS
    padding = padded_d - d_size
    partial_padded = torch.cat(
        (partial, partial.new_zeros(partial.shape[:-1] + (padding,))), dim=-1
    )
    query_padded = torch.cat((query, query.new_zeros((padding,))), dim=-1)
    partial_vectors = partial_padded.reshape(
        partial.shape[:-1] + (vector_count, _VREG_FP32_ELEMENTS)
    )
    query_vectors = query_padded.reshape(vector_count, _VREG_FP32_ELEMENTS)

    partial_reg0 = partial_vectors[..., 0, :]
    query_reg0 = query_vectors[0]
    if d_size <= _VREG_FP32_ELEMENTS:
        square_reg = partial_reg0 * partial_reg0
        dot_reg = partial_reg0 * query_reg0
    elif d_size <= 2 * _VREG_FP32_ELEMENTS:
        partial_reg1 = partial_vectors[..., 1, :]
        query_reg1 = query_vectors[1]
        square_reg = partial_reg0 * partial_reg0
        dot_reg = partial_reg0 * query_reg0
        square_reg = _torch_fma_float32(partial_reg1, partial_reg1, square_reg)
        dot_reg = _torch_fma_float32(partial_reg1, query_reg1, dot_reg)
    else:
        accumulator_shape = partial.shape[:-1] + (_VREG_FP32_ELEMENTS,)
        square_acc0 = torch.zeros(
            accumulator_shape, dtype=torch.float32, device=partial.device
        )
        square_acc1 = torch.zeros_like(square_acc0)
        dot_acc0 = torch.zeros_like(square_acc0)
        dot_acc1 = torch.zeros_like(square_acc0)
        full_loops = d_size // _VREG_FP32_ELEMENTS
        pair_full_loops = full_loops // 2

        for loop in range(pair_full_loops):
            vector_index0 = 2 * loop
            vector_index1 = vector_index0 + 1
            partial_reg0 = partial_vectors[..., vector_index0, :]
            partial_reg1 = partial_vectors[..., vector_index1, :]
            square_acc0 = _torch_fma_float32(partial_reg0, partial_reg0, square_acc0)
            square_acc1 = _torch_fma_float32(partial_reg1, partial_reg1, square_acc1)
            dot_acc0 = _torch_fma_float32(
                partial_reg0, query_vectors[vector_index0], dot_acc0
            )
            dot_acc1 = _torch_fma_float32(
                partial_reg1, query_vectors[vector_index1], dot_acc1
            )

        if full_loops & 1:
            vector_index = 2 * pair_full_loops
            partial_reg0 = partial_vectors[..., vector_index, :]
            square_acc0 = _torch_fma_float32(partial_reg0, partial_reg0, square_acc0)
            dot_acc0 = _torch_fma_float32(
                partial_reg0, query_vectors[vector_index], dot_acc0
            )

        if d_size % _VREG_FP32_ELEMENTS:
            vector_index = full_loops
            partial_reg0 = partial_vectors[..., vector_index, :]
            if full_loops & 1:
                square_acc1 = _torch_fma_float32(
                    partial_reg0, partial_reg0, square_acc1
                )
                dot_acc1 = _torch_fma_float32(
                    partial_reg0, query_vectors[vector_index], dot_acc1
                )
            else:
                square_acc0 = _torch_fma_float32(
                    partial_reg0, partial_reg0, square_acc0
                )
                dot_acc0 = _torch_fma_float32(
                    partial_reg0, query_vectors[vector_index], dot_acc0
                )

        square_reg = square_acc0 + square_acc1
        dot_reg = dot_acc0 + dot_acc1

    return _torch_binary_tree_sum_float32(square_reg), _torch_binary_tree_sum_float32(
        dot_reg
    )


def _torch_golden(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    eps,
):
    """Return ``(updated_partial_block, h)`` as CPU torch tensors."""
    import torch

    partial = partial_block.to(dtype=torch.float32)
    delta_fp32 = delta.to(dtype=torch.float32)
    pseudo_query_fp32 = pseudo_query.to(dtype=torch.float32)
    numerator_fp32 = numerator.to(dtype=torch.float32)
    logit_max_fp32 = logit_max.to(dtype=torch.float32)
    exp_sum_fp32 = exp_sum.to(dtype=torch.float32)

    # Keep the golden functional: the real API updates partial_block in place,
    # but mutating TTK's CPU input here would affect later input reuse.
    partial_out = partial + delta_fp32
    if partial_out.numel() == 0:
        return partial_out, torch.empty_like(delta)

    square_sum, dot_sum = _torch_phase1_reductions_float32(
        partial_out, pseudo_query_fp32
    )
    inv_d = torch.tensor(
        float(numpy.float32(1.0 / partial_out.shape[-1])),
        dtype=torch.float32,
        device=partial_out.device,
    )
    rms = _torch_sqrt_ftz_float32(square_sum * inv_d + float(eps))
    score = _torch_div_ftz_float32(dot_sum, rms)

    current_max = torch.maximum(logit_max_fp32, score)
    alpha = _torch_exp_sub_ftz_float32(logit_max_fp32, current_max)
    beta = _torch_exp_sub_ftz_float32(score, current_max)
    denominator = _torch_fma_float32(exp_sum_fp32, alpha, beta)
    alpha = _torch_div_ftz_float32(alpha, denominator)
    beta = _torch_div_ftz_float32(beta, denominator)
    partial_scaled = partial_out * beta[:, None]
    h_fp32 = _torch_fma_float32(numerator_fp32, alpha[:, None], partial_scaled)
    return partial_out, h_fp32.to(dtype=torch.bfloat16)


class BlockAttnResUpdateTestSpec:
    """CPU reference for the kernel-only BlockAttnResUpdate test path."""

    @staticmethod
    def golden(
        partial_block,
        delta,
        pseudo_query,
        numerator,
        logit_max,
        exp_sum,
        eps=1e-6,
        **kwargs,
    ):
        del kwargs
        partial = numpy.asarray(partial_block, dtype=numpy.float32)
        delta_fp32 = numpy.asarray(delta, dtype=numpy.float32)
        pseudo_query = numpy.asarray(pseudo_query, dtype=numpy.float32)
        numerator = numpy.asarray(numerator, dtype=numpy.float32)
        logit_max = numpy.asarray(logit_max, dtype=numpy.float32)
        exp_sum = numpy.asarray(exp_sum, dtype=numpy.float32)

        # Do not mutate the input before TTK copies it to the device. Output 0
        # aliases input 0 only in the kernel launch buffers.
        partial_out = numpy.add(partial, delta_fp32).astype(numpy.float32, copy=False)

        square_sum, dot_sum = _phase1_reductions_float32(partial_out, pseudo_query)
        inv_d = numpy.float32(1.0 / partial_out.shape[-1])
        rms = _sqrt_ftz_float32(
            _add_float32(_multiply_float32(square_sum, inv_d), numpy.float32(eps))
        )
        score = _div_ftz_float32(dot_sum, rms)

        current_max = numpy.maximum(logit_max, score)
        alpha = _exp_sub_ftz_float32(logit_max, current_max)
        beta = _exp_sub_ftz_float32(score, current_max)
        denominator = _fma_float32(exp_sum, alpha, beta)
        alpha = _div_ftz_float32(alpha, denominator)
        beta = _div_ftz_float32(beta, denominator)

        partial_scaled = partial_out * beta[:, None]
        h_fp32 = _fma_float32(numerator, alpha[:, None], partial_scaled)
        h = _round_to_bfloat16_float32(h_fp32)
        return [partial_out, h]


class BlockAttnResUpdateAclnnTestSpec:
    """TestSpec registered by the exact ACLNN API name."""

    @staticmethod
    def golden(
        partialBlockRef,
        delta,
        pseudoQuery,
        numerator,
        logitMax,
        expSum,
        eps=1e-6,
        h=None,
        **kwargs,
    ):
        del h, kwargs
        partial_out, h_golden = _torch_golden(
            partialBlockRef,
            delta,
            pseudoQuery,
            numerator,
            logitMax,
            expSum,
            eps,
        )
        # ACLNN CSV output_tensor_indexes=(0, 6): inplace partialBlockRef, then h.
        return [partial_out, h_golden]


class BlockAttnResUpdateE2ETestSpec:
    """TestSpec registered by the explicit Python wrapper API name."""

    @staticmethod
    def golden(
        partial_block,
        delta,
        pseudo_query,
        numerator,
        logit_max,
        exp_sum,
        *,
        eps=1e-6,
        **kwargs,
    ):
        del kwargs
        partial_out, h_golden = _torch_golden(
            partial_block,
            delta,
            pseudo_query,
            numerator,
            logit_max,
            exp_sum,
            eps,
        )
        # E2E records the API return first, then inplace_input_indexes=(0,).
        return [h_golden, partial_out]
