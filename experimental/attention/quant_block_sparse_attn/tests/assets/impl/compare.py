#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare helpers for quant_block_sparse_attn TestSpec adapters."""

import numpy as np
import torch

_DEFAULT_RTOL = 0.005
_DEFAULT_ATOL = 0.000025
_BF16_RTOL = 0.0078125
_BF16_ATOL = 0.0001
_FAIL_RATIO_LIMIT = 0.005
_MAX_RELATIVE_ERROR = 10.0
_EPSILON = 1e-10


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _tolerance(value):
    dtype_name = str(getattr(value, "dtype", ""))
    if "bfloat16" in dtype_name:
        return _BF16_RTOL, _BF16_ATOL
    return _DEFAULT_RTOL, _DEFAULT_ATOL


def _normalized_relative_error(npu, golden, mismatch):
    if not np.any(mismatch):
        return 0.0
    absolute_diff = np.abs(golden - npu)
    floor = (1.0 / (1 << 14)) / _DEFAULT_RTOL
    denominator = np.maximum(np.maximum(np.abs(npu), np.abs(golden)), floor)
    relative = absolute_diff / (denominator + _EPSILON)
    relative = np.where(np.isfinite(relative), relative, np.inf)
    return float(np.max(relative[mismatch]))


def attention_compare(npu_out, golden_out):
    """Compare attention output tensors with BF16-aware tolerance."""
    npu = as_numpy(npu_out)
    golden = as_numpy(golden_out)

    if npu.shape != golden.shape:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"shape mismatch: npu={npu.shape}, golden={golden.shape}",
        }

    if golden.size == 0:
        return {"pass": True, "precision": 100.0}

    npu_f32 = npu.astype(np.float32)
    golden_f32 = golden.astype(np.float32)

    rtol, atol = _tolerance(npu_out)
    abs_diff = np.abs(npu_f32 - golden_f32)
    denom = np.maximum(np.abs(golden_f32), 1e-10)
    rel_diff = abs_diff / denom

    mismatch = (rel_diff > rtol) & (abs_diff > atol)
    diff_count = int(np.sum(mismatch))
    precision = (golden.size - diff_count) / golden.size * 100
    fail_ratio = diff_count / golden.size
    max_relative_error = _normalized_relative_error(
        npu_f32, golden_f32, mismatch
    )

    passed = (
        fail_ratio <= _FAIL_RATIO_LIMIT
        and max_relative_error < _MAX_RELATIVE_ERROR
    )
    error_info = None
    if diff_count > 0:
        max_abs = float(np.max(abs_diff[mismatch])) if diff_count > 0 else 0.0
        max_rel = float(np.max(rel_diff[mismatch])) if diff_count > 0 else 0.0
        error_info = (
            f"attention compare mismatches={diff_count}, fail_ratio={fail_ratio:.6g}, "
            f"max_abs_diff={max_abs:.6g}, max_rel_diff={max_rel:.6g}, "
            f"max_normalized_rel_error={max_relative_error:.6g}"
        )

    return {
        "pass": passed,
        "precision": precision,
        "diff_indices": np.where(mismatch.reshape(-1))[0][:1000].tolist(),
        "error_info": error_info,
        "metrics": {
            "rtol": rtol,
            "atol": atol,
            "fail_ratio": fail_ratio,
            "fail_ratio_limit": _FAIL_RATIO_LIMIT,
            "max_normalized_rel_error": max_relative_error,
            "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0,
            "max_rel_diff": float(np.max(rel_diff)) if rel_diff.size > 0 else 0.0,
        },
    }


def lse_compare(npu_out, golden_out):
    """Compare softmax LSE tensors."""
    npu = as_numpy(npu_out)
    golden = as_numpy(golden_out)

    if npu.size == 0 and golden.size == 0:
        return {"pass": True, "precision": 100.0}

    if npu.shape != golden.shape:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"lse shape mismatch: npu={npu.shape}, golden={golden.shape}",
        }

    if golden.size == 0:
        return {"pass": True, "precision": 100.0}

    npu_f32 = npu.astype(np.float32)
    golden_f32 = golden.astype(np.float32)

    valid_mask = golden_f32 > -3e38
    if not np.any(valid_mask):
        return {"pass": True, "precision": 100.0}

    rtol, atol = _tolerance(npu_out)
    abs_diff = np.abs(npu_f32 - golden_f32)
    denom = np.maximum(np.abs(golden_f32), _EPSILON)
    rel_diff = abs_diff / denom
    mismatch = valid_mask & (rel_diff > rtol) & (abs_diff > atol)
    diff_count = int(np.sum(mismatch))
    valid_count = max(int(np.sum(valid_mask)), 1)
    fail_ratio = diff_count / valid_count
    max_relative_error = _normalized_relative_error(
        npu_f32, golden_f32, mismatch
    )
    max_abs_diff = float(np.max(abs_diff[valid_mask]))
    passed = (
        fail_ratio <= _FAIL_RATIO_LIMIT
        and max_relative_error < _MAX_RELATIVE_ERROR
    )

    return {
        "pass": passed,
        "precision": (valid_count - diff_count) / valid_count * 100.0,
        "error_info": (
            None
            if passed
            else f"lse max_abs_diff={max_abs_diff:.6g}, "
            f"fail_ratio={fail_ratio:.6g}, "
            f"max_normalized_rel_error={max_relative_error:.6g}"
        ),
        "metrics": {
            "rtol": rtol,
            "atol": atol,
            "max_abs_diff": max_abs_diff,
            "fail_ratio": fail_ratio,
            "max_normalized_rel_error": max_relative_error,
        },
    }


def compare(*outputs, **kwargs):
    """Compare NPU outputs against golden references.

    Expects [npu_attention_out, npu_softmax_lse, golden_attention_out, golden_softmax_lse]
    When return_softmax_lse=False, golden_softmax_lse is None because the NPU
    scalar-shaped placeholder does not contain a valid value.
    """
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": f"compare expects even number of outputs, got {len(outputs)}",
        }

    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]

    results = [attention_compare(npu_outputs[0], golden_outputs[0])]

    if half > 1 and golden_outputs[1] is not None:
        results.append(lse_compare(npu_outputs[1], golden_outputs[1]))

    return results[0] if len(results) == 1 else results
