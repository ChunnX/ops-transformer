#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK golden references for MoeTokenPermute and MoeTokenPermuteV2."""

import numpy as np
import torch

from ttk.utilities import mx_quantize


__spec__ = {
    "aclnnMoeTokenPermute": "AclnnMoeTokenPermuteTestSpec",
    "aclnnMoeTokenPermuteV2": "AclnnMoeTokenPermuteV2TestSpec",
}

__golden__ = {
    "kernel": {
        "moe_token_permute": "moe_token_permute_kernel_golden",
    }
}


_EXACT_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
}


def _permute(tokens, indices, num_out_tokens, padded_mode):
    if padded_mode:
        raise ValueError("paddedMode=True is not supported by MoeTokenPermute")

    topk = indices.shape[1] if indices.ndim == 2 else 1
    flattened_indices = indices.reshape(-1)
    total = flattened_indices.numel() if isinstance(flattened_indices, torch.Tensor) else flattened_indices.size
    if num_out_tokens == 0:
        output_count = total
    elif num_out_tokens > 0:
        output_count = min(num_out_tokens, total)
    else:
        output_count = max(num_out_tokens + total, 0)

    if isinstance(flattened_indices, torch.Tensor):
        sorted_indices = torch.argsort(flattened_indices, stable=True)
        inverse_indices = torch.argsort(sorted_indices, stable=True).to(torch.int32)
        token_indices = torch.div(
            sorted_indices[:output_count], topk, rounding_mode="floor"
        ).to(torch.int64)
        permuted_tokens = torch.index_select(tokens, 0, token_indices)
    else:
        sorted_indices = np.argsort(flattened_indices, kind="stable")
        inverse_indices = np.empty(total, dtype=np.int32)
        inverse_indices[sorted_indices] = np.arange(total, dtype=np.int32)
        token_indices = sorted_indices[:output_count] // topk
        permuted_tokens = np.take(tokens, token_indices, axis=0)
    return permuted_tokens, inverse_indices


def moe_token_permute_kernel_golden(
    tokens,
    indices,
    *,
    num_out_tokens=0,
    padded_mode=False,
    **kwargs,
):
    return list(_permute(tokens, indices, num_out_tokens, padded_mode))


def _quantize(permuted_tokens, quant_mode):
    if isinstance(permuted_tokens, torch.Tensor):
        quant_input = permuted_tokens.to(torch.float32).cpu().numpy()
    else:
        quant_input = np.asarray(permuted_tokens)
        if quant_input.dtype.name not in ("float16", "float32", "bfloat16"):
            quant_input = quant_input.astype(np.float32)

    quant_dtype = {
        2: "float8_e5m2",
        3: "float8_e4m3fn",
        9: "float4_e2m1",
    }.get(int(quant_mode))
    if quant_dtype is None:
        raise ValueError(f"unsupported quantMode={quant_mode}")

    expanded_scale, quantized_tokens = mx_quantize(
        quant_input,
        mx_ele_dtype=quant_dtype,
        axis=-1,
        block_size=32,
        scale_alg=1 if int(quant_mode) in (2, 3) else 0,
    )
    if int(quant_mode) in (2, 3):
        expanded_scale = expanded_scale.reshape(expanded_scale.shape[0], -1)
    return quantized_tokens, expanded_scale


def _check_output_shape(name, actual, expected):
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"{name} shape mismatch: golden={tuple(actual.shape)}, configured={tuple(expected.shape)}"
        )


def aclnn_moe_token_permute_golden(
    tokens,
    indices,
    numOutTokens,
    paddedMode,
    permuteTokensOut,
    sortedIndicesOut,
    **kwargs,
):
    permuted_tokens, sorted_indices = _permute(tokens, indices, numOutTokens, paddedMode)
    _check_output_shape("permuteTokensOut", permuted_tokens, permuteTokensOut)
    _check_output_shape("sortedIndicesOut", sorted_indices, sortedIndicesOut)
    return [permuted_tokens, sorted_indices]


def aclnn_moe_token_permute_v2_golden(
    tokens,
    indices,
    numOutTokens,
    paddedMode,
    quantMode,
    permuteTokensOut,
    sortedIndicesOut,
    expandedScaleOut,
    **kwargs,
):
    permuted_tokens, sorted_indices = _permute(tokens, indices, numOutTokens, paddedMode)
    if int(quantMode) == -1:
        expanded_scale = torch.empty_like(expandedScaleOut)
    else:
        permuted_tokens, expanded_scale = _quantize(permuted_tokens, quantMode)
    _check_output_shape("permuteTokensOut", permuted_tokens, permuteTokensOut)
    _check_output_shape("sortedIndicesOut", sorted_indices, sortedIndicesOut)
    _check_output_shape("expandedScaleOut", expanded_scale, expandedScaleOut)
    return [permuted_tokens, sorted_indices, expanded_scale]


class AclnnMoeTokenPermuteTestSpec:
    golden = aclnn_moe_token_permute_golden
    tolerance = _EXACT_TOLERANCE


class AclnnMoeTokenPermuteV2TestSpec:
    golden = aclnn_moe_token_permute_v2_golden
    tolerance = _EXACT_TOLERANCE
