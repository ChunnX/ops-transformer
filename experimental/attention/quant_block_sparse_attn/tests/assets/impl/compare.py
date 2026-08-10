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

"""PyTorch precision comparison for quant_block_sparse_attn TTK adapters."""

import torch

_FP32_RTOL = 0.005
_FP32_ATOL = 0.000025
_BF16_RTOL = 0.0078125
_BF16_ATOL = 0.0001
_FAIL_RATIO_LIMIT = 0.005
_MAX_NORMALIZED_ERROR = 10.0
_NORMALIZATION_EPSILON = 1e-10
_NORMALIZATION_FLOOR = (1.0 / (1 << 14)) / _FP32_RTOL


def _as_cpu_tensor(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _compare_tolerance(npu_out, golden_out):
    dtype_names = (
        str(getattr(npu_out, "dtype", "")),
        str(getattr(golden_out, "dtype", "")),
    )
    if any("bfloat16" in name for name in dtype_names):
        return _BF16_RTOL, _BF16_ATOL
    return _FP32_RTOL, _FP32_ATOL


def _torch_compare(npu_out, golden_out, output_name):
    """Compare one output with the unified PyTorch precision protocol."""
    npu = _as_cpu_tensor(npu_out)
    golden = _as_cpu_tensor(golden_out)

    if npu.numel() == 0 and golden.numel() == 0:
        return {"pass": True, "precision": 100.0}

    if npu.numel() != golden.numel():
        return {
            "pass": False,
            "precision": "size_mismatch",
            "error_info": (
                f"{output_name} size mismatch: npu={npu.numel()}, "
                f"golden={golden.numel()}, npu_shape={tuple(npu.shape)}, "
                f"golden_shape={tuple(golden.shape)}"
            ),
        }

    rtol, atol = _compare_tolerance(npu_out, golden_out)
    npu_f32 = npu.to(torch.float32).flatten()
    golden_f32 = golden.to(torch.float32).flatten()
    close_mask = torch.isclose(
        npu_f32,
        golden_f32,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    mismatch = ~close_mask
    diff_count = int(mismatch.sum().item())
    element_count = max(int(golden_f32.numel()), 1)
    fail_ratio = diff_count / element_count

    abs_diff = torch.abs(npu_f32 - golden_f32)
    floor = torch.tensor(
        _NORMALIZATION_FLOOR,
        dtype=torch.float32,
        device=npu_f32.device,
    )
    denominator = torch.maximum(
        torch.maximum(torch.abs(npu_f32), torch.abs(golden_f32)), floor
    ) + _NORMALIZATION_EPSILON
    normalized_error = (abs_diff / denominator)[mismatch]

    # Preserve NaN/Inf instead of coercing them. Equal NaN/NaN and same-sign
    # Inf/Inf were already accepted by isclose and do not enter this tensor.
    max_normalized_error = (
        float(torch.max(normalized_error).item())
        if normalized_error.numel() > 0
        else 0.0
    )
    max_abs_diff = (
        float(torch.max(abs_diff[mismatch]).item()) if diff_count > 0 else 0.0
    )

    passed = fail_ratio <= _FAIL_RATIO_LIMIT
    if max_normalized_error >= _MAX_NORMALIZED_ERROR:
        passed = False

    error_info = None
    if not passed:
        error_info = (
            f"{output_name} mismatches={diff_count}, "
            f"fail_ratio={fail_ratio:.6g}, "
            f"max_abs_diff={max_abs_diff:.6g}, "
            f"max_normalized_error={max_normalized_error:.6g}"
        )

    return {
        "pass": passed,
        "precision": (element_count - diff_count) / element_count * 100.0,
        "diff_indices": torch.nonzero(mismatch, as_tuple=False)
        .flatten()[:1000]
        .tolist(),
        "error_info": error_info,
        "metrics": {
            "rtol": rtol,
            "atol": atol,
            "fail_ratio": fail_ratio,
            "fail_ratio_limit": _FAIL_RATIO_LIMIT,
            "max_abs_diff": max_abs_diff,
            "max_normalized_error": max_normalized_error,
            "normalization_epsilon": _NORMALIZATION_EPSILON,
        },
    }


def attention_compare(npu_out, golden_out):
    return _torch_compare(npu_out, golden_out, "attention_out")


def lse_compare(npu_out, golden_out):
    return _torch_compare(npu_out, golden_out, "softmax_lse")


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


def _compare_outputs(outputs, *, return_softmax_lse=True):
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

    if (
        return_softmax_lse
        and half > 1
        and golden_outputs[1] is not None
    ):
        results.append(lse_compare(npu_outputs[1], golden_outputs[1]))

    return results[0] if len(results) == 1 else results


def compare(*outputs, **kwargs):
    """Compare FP8 and MXFP8 outputs with one PyTorch precision protocol."""
    return _compare_outputs(
        outputs,
        return_softmax_lse=_as_bool(kwargs.get("return_softmax_lse", True)),
    )
