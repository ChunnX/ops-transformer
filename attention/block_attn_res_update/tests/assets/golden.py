#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FP32 reference implementation for BlockAttnResUpdate.

Naming correspondence with the Phase-2 calculation document:

    partial_block   -> partial
    partial_delta   -> delta
    inter_numerator -> history_o
    inter_max       -> history_m
    inter_exp_sum   -> history_ell
    h               -> output
    partial_blocks  -> partial_out / updated partial block

The current operator contract uses FP32 ``partial_block`` and a scalar FP32
``epsilon`` attribute.  For compatibility with an older BF16-partial draft,
``_compute`` also models a low-precision add when ``partial_block`` itself is
FP16/BF16; the installed operator's normal path remains FP32 add.
"""

import numpy as np


__golden__ = {
    "e2e": {
        "block_attn_res_update_ttk_ops.block_attn_res_update_ttk": "block_attn_res_update_e2e_golden",
        "cann_ops_transformer.block_attn_res_update": "block_attn_res_update_e2e_golden",
    },
    "kernel": {
        "block_attn_res_update": "block_attn_res_update_golden",
    },
    "aclnn": {
        "aclnnBlockAttnResUpdate": "aclnn_block_attn_res_update_golden",
    },
}


def _to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    if hasattr(tensor, "numpy"):
        try:
            return tensor.numpy()
        except (TypeError, RuntimeError):
            # torch.bfloat16 has no direct NumPy representation in PyTorch.
            return tensor.float().numpy()
    return np.asarray(tensor)


def _dtype_name(value):
    # Inspect the framework dtype before conversion because PyTorch BF16 must
    # be promoted to FP32 in ``_to_numpy`` (NumPy has no portable BF16 dtype).
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        name = getattr(dtype, "name", None)
        if name:
            return name
        text = str(dtype)
        if text.startswith("torch."):
            return text[len("torch.") :]
    value = _to_numpy(value)
    return getattr(value.dtype, "name", str(value.dtype))


def _cast_like(value, template, use_torch=False):
    """Cast an output to its API output template/input dtype and framework."""
    value = np.asarray(value)
    if use_torch and hasattr(template, "device"):
        import torch

        result = torch.as_tensor(value, device=template.device)
        return result.to(dtype=template.dtype)
    if isinstance(template, np.ndarray):
        return value.astype(template.dtype, copy=False)
    return value


def _validate_shapes(
    partial_block,
    partial_delta,
    effective_query,
    inter_max,
    inter_exp_sum,
    inter_numerator,
):
    shape = partial_block.shape
    if partial_block.ndim != 2:
        raise ValueError(f"partial_block must have shape [T, D], got {shape}")
    token_num, hidden_size = shape
    expected = {
        "partial_delta": (partial_delta.shape, shape),
        "effective_query": (effective_query.shape, (hidden_size,)),
        "inter_max": (inter_max.shape, (token_num,)),
        "inter_exp_sum": (inter_exp_sum.shape, (token_num,)),
        "inter_numerator": (inter_numerator.shape, shape),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"{name} shape must be {wanted}, got {actual}")
    return token_num, hidden_size


def _compute(
    partial_block,
    partial_delta,
    effective_query,
    inter_max,
    inter_exp_sum,
    inter_numerator,
    epsilon,
):
    partial_block_np = _to_numpy(partial_block)
    partial_delta_np = _to_numpy(partial_delta)
    effective_query_np = _to_numpy(effective_query)
    inter_max_np = _to_numpy(inter_max)
    inter_exp_sum_np = _to_numpy(inter_exp_sum)
    inter_numerator_np = _to_numpy(inter_numerator)

    _, hidden_size = _validate_shapes(
        partial_block_np,
        partial_delta_np,
        effective_query_np,
        inter_max_np,
        inter_exp_sum_np,
        inter_numerator_np,
    )
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")
    if np.any(inter_exp_sum_np < 0.0):
        raise ValueError("inter_exp_sum must be non-negative")

    # Current interface: partial_block is FP32, so the accumulation is FP32.
    # Compatibility path: the linked calculation draft described BF16 add;
    # casting the sum back before promotion reproduces that rounding point.
    partial_dtype = _dtype_name(partial_block)
    partial_sum_f32 = (
        partial_block_np.astype(np.float32) + partial_delta_np.astype(np.float32)
    )
    if partial_dtype in ("float16", "bfloat16"):
        partial_sum = partial_sum_f32.astype(partial_block_np.dtype).astype(np.float32)
    else:
        partial_sum = partial_sum_f32.astype(np.float32, copy=False)

    # RMSNorm denominator, score, and online-softmax merge are all FP32.
    square_sum = np.sum(partial_sum * partial_sum, axis=-1, dtype=np.float32)
    rms = np.sqrt(square_sum / np.float32(hidden_size) + np.float32(epsilon)).astype(np.float32)
    score = np.sum(
        partial_sum * effective_query_np.astype(np.float32),
        axis=-1,
        dtype=np.float32,
    ) / rms

    history_max = inter_max_np.astype(np.float32)
    merged_max = np.maximum(history_max, score).astype(np.float32)
    alpha = np.exp(history_max - merged_max).astype(np.float32)
    beta = np.exp(score - merged_max).astype(np.float32)
    merged_denominator = (
        alpha * inter_exp_sum_np.astype(np.float32) + beta
    ).astype(np.float32)
    merged_numerator = (
        alpha[:, None] * inter_numerator_np.astype(np.float32)
        + beta[:, None] * partial_sum
    ).astype(np.float32)
    output = (merged_numerator / merged_denominator[:, None]).astype(np.float32)

    # Kernel output order is h first, updated partial block second.
    return output, partial_sum


def block_attn_res_update_golden(
    partial_block,
    delta,
    inter_numerator,
    inter_max,
    inter_exp_sum,
    effective_query,
    eps=1e-6,
    **kwargs,
):
    """Kernel golden; parameters follow block_attn_res_update_def.cpp."""
    del kwargs
    h, partial_blocks = _compute(
        partial_block,
        delta,
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
        eps,
    )
    # OpDef output order is partial_block (reference output), then h.
    return (
        _cast_like(partial_blocks, partial_block),
        _cast_like(h, delta),
    )


def block_attn_res_update_e2e_golden(
    partial_block,
    partial_delta,
    effective_query,
    inter_max,
    inter_exp_sum,
    inter_numerator,
    epsilon,
    **kwargs,
):
    """E2E golden; return order matches the public Torch API: (h, partial_blocks)."""
    output, updated_partial = _compute(
        partial_block,
        partial_delta,
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
        epsilon,
    )
    use_torch = kwargs.get("use_torch", False)
    return (
        _cast_like(output, partial_delta, use_torch),
        _cast_like(updated_partial, partial_block, use_torch),
    )


def aclnn_block_attn_res_update_golden(
    partialBlock,
    delta,
    interNumerator,
    interMax,
    interExpSum,
    effectiveQuery,
    eps,
    h,
    **kwargs,
):
    """ACLNN golden; parameters follow aclnnBlockAttnResUpdateGetWorkspaceSize."""
    output, updated_partial = _compute(
        partialBlock,
        delta,
        effectiveQuery,
        interMax,
        interExpSum,
        interNumerator,
        eps,
    )
    use_torch = kwargs.get("use_torch", False)
    return (
        _cast_like(updated_partial, partialBlock, use_torch),
        _cast_like(output, h, use_torch),
    )
