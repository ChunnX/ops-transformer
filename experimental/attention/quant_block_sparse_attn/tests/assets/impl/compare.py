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

"""NumPy precision comparison for quant_block_sparse_attn TTK adapters."""

import numpy as np

_FP32_RTOL = 0.005
_FP32_ATOL = 0.000025
_BF16_RTOL = 0.0078125
_BF16_ATOL = 0.0001
_FAIL_RATIO_LIMIT = 0.005
_MAX_NORMALIZED_ERROR = 10.0
_NORMALIZATION_EPSILON = 1e-10
_NORMALIZATION_FLOOR = (1.0 / (1 << 14)) / _FP32_RTOL
_DISPLAY_ROW_LIMIT = 20


def _print_log(message):
    print(f"[precision_compare] {message}")


def _failure_result(output_name, precision, error_info):
    """Print and return a non-numeric comparison failure."""
    _print_log(f"Compare output: {output_name}")
    _print_log(f"Failed: {error_info}")
    return {
        "pass": False,
        "precision": precision,
        "error_info": error_info,
    }


def _display_indices(indices):
    """Limit large comparison tables to their first and last rows."""
    index_count = len(indices)
    if index_count <= _DISPLAY_ROW_LIMIT:
        return list(indices)
    half = _DISPLAY_ROW_LIMIT // 2
    return list(indices[:half]) + [None] + list(indices[-half:])


def _value_error(npu_value, golden_value):
    """Return display-friendly absolute and normalized errors for one pair."""
    npu_value = float(npu_value)
    golden_value = float(golden_value)
    if np.isfinite(npu_value) and np.isfinite(golden_value):
        abs_diff = abs(npu_value - golden_value)
        denominator = (
            max(abs(npu_value), abs(golden_value), _NORMALIZATION_FLOOR)
            + _NORMALIZATION_EPSILON
        )
        return abs_diff, abs_diff / denominator

    both_nan = np.isnan(npu_value) and np.isnan(golden_value)
    same_infinity = np.isinf(npu_value) and npu_value == golden_value
    if both_nan or same_infinity:
        return 0.0, 0.0
    return float("inf"), float("inf")


def _display_rows(title, npu_data, golden_data, indices):
    _print_log(title)
    _print_log(
        "Index     Golden          NPU             AbsDiff         "
        "NormalizedError"
    )
    _print_log("-" * 82)
    for index in _display_indices(indices):
        if index is None:
            _print_log("...       ...             ...             ...             ...")
            continue
        abs_diff, normalized_error = _value_error(
            npu_data[index], golden_data[index]
        )
        _print_log(
            f"{index:<9d} "
            f"{float(golden_data[index]):<15.7g} "
            f"{float(npu_data[index]):<15.7g} "
            f"{abs_diff:<15.7g} "
            f"{normalized_error:.7g}"
        )


def _display_comparison_result(
    output_name,
    npu_data,
    golden_data,
    mismatch,
    *,
    rtol,
    atol,
    precision,
    max_normalized_error,
    passed,
):
    """Print the comparison process and detailed mismatches on failure."""
    _print_log(f"Compare output: {output_name}")
    _print_log(
        f"element_count:{golden_data.size}; mismatch_count:{np.count_nonzero(mismatch)};"
    )
    _display_rows(
        "Comparison Data", npu_data, golden_data, range(golden_data.size)
    )
    _print_log("-" * 82)
    _print_log(
        "Rtol       Atol        PctThd       PctRlt       "
        "MaxNormalizedError   Result"
    )
    _print_log("-" * 82)
    result = "Pass" if passed else "Failed"
    precision_threshold = (1.0 - _FAIL_RATIO_LIMIT) * 100.0
    _print_log(
        f"{rtol:<10.6g} {atol:<11.6g} {precision_threshold:<12.6f}% "
        f"{precision:<12.6f}% {max_normalized_error:<20.7g} {result}"
    )

    if passed:
        return

    error_indices = np.flatnonzero(mismatch)
    _print_log("Error Line" + "-" * 72)
    _display_rows(
        "Failed Precision Comparison", npu_data, golden_data, error_indices
    )
    if error_indices.size:
        error_values = np.array(
            [
                _value_error(npu_data[index], golden_data[index])[1]
                for index in error_indices
            ]
        )
        max_error_index = int(error_indices[int(np.argmax(error_values))])
        _display_rows(
            "Max-Error Line", npu_data, golden_data, [max_error_index]
        )


def _as_numpy(value):
    """Normalize TTK output data to a host NumPy array."""
    if isinstance(value, np.ndarray):
        return value

    # Keep this adapter tolerant of direct unit-test calls with CPU tensors,
    # while using NumPy as the only comparison implementation. PyTorch cannot
    # expose BF16 tensors directly as NumPy, so convert that transport dtype.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        if "bfloat16" in str(getattr(tensor, "dtype", "")):
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def _compare_tolerance(npu_out, golden_out):
    dtype_names = (
        str(getattr(npu_out, "dtype", "")),
        str(getattr(golden_out, "dtype", "")),
    )
    if any("bfloat16" in name for name in dtype_names):
        return _BF16_RTOL, _BF16_ATOL
    return _FP32_RTOL, _FP32_ATOL


def _numpy_compare(npu_out, golden_out, output_name):
    """Compare one output with the unified NumPy precision protocol."""
    npu = _as_numpy(npu_out)
    golden = _as_numpy(golden_out)

    if npu.shape != golden.shape:
        error_info = (
            f"{output_name} shape mismatch: "
            f"npu_shape={tuple(npu.shape)}, "
            f"golden_shape={tuple(golden.shape)}"
        )
        return _failure_result(output_name, "shape_mismatch", error_info)

    rtol, atol = _compare_tolerance(npu_out, golden_out)
    if npu.size == 0:
        empty_data = npu.astype(np.float32, copy=False).reshape(-1)
        _display_comparison_result(
            output_name,
            empty_data,
            empty_data,
            np.zeros(0, dtype=bool),
            rtol=rtol,
            atol=atol,
            precision=100.0,
            max_normalized_error=0.0,
            passed=True,
        )
        return {"pass": True, "precision": 100.0}

    npu_f32 = npu.astype(np.float32, copy=False).reshape(-1)
    golden_f32 = golden.astype(np.float32, copy=False).reshape(-1)
    close_mask = np.isclose(
        npu_f32,
        golden_f32,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    mismatch = ~close_mask
    diff_count = int(np.count_nonzero(mismatch))
    element_count = max(int(golden_f32.size), 1)
    fail_ratio = diff_count / element_count

    nan_mismatch = np.isnan(npu_f32) ^ np.isnan(golden_f32)
    has_nan_mismatch = bool(np.any(nan_mismatch))
    finite_mismatch = mismatch & np.isfinite(npu_f32) & np.isfinite(golden_f32)
    nonfinite_mismatch = mismatch & ~(np.isfinite(npu_f32) & np.isfinite(golden_f32))
    has_nonfinite_mismatch = bool(np.any(nonfinite_mismatch))

    # Calculate error magnitudes only for finite mismatches. NaN/NaN and
    # same-sign Inf/Inf pairs are accepted by isclose. Other non-finite pairs
    # remain mismatches and are governed by the same failure-ratio threshold.
    finite_npu = npu_f32[finite_mismatch]
    finite_golden = golden_f32[finite_mismatch]
    finite_abs_diff = np.abs(finite_npu - finite_golden)
    denominator = (
        np.maximum(
            np.maximum(np.abs(finite_npu), np.abs(finite_golden)),
            _NORMALIZATION_FLOOR,
        )
        + _NORMALIZATION_EPSILON
    )
    normalized_error = finite_abs_diff / denominator
    max_normalized_error = (
        float(np.max(normalized_error)) if normalized_error.size else 0.0
    )
    if has_nan_mismatch:
        max_abs_diff = float("nan")
    elif has_nonfinite_mismatch:
        max_abs_diff = float("inf")
    else:
        max_abs_diff = float(np.max(finite_abs_diff)) if finite_abs_diff.size else 0.0

    passed = fail_ratio <= _FAIL_RATIO_LIMIT
    if max_normalized_error >= _MAX_NORMALIZED_ERROR:
        passed = False

    error_info = None
    if not passed:
        error_info = (
            f"{output_name} mismatches={diff_count}, "
            f"fail_ratio={fail_ratio:.6g}, "
            f"max_abs_diff={max_abs_diff:.6g}, "
            f"max_normalized_error={max_normalized_error:.6g}, "
            f"has_nan_mismatch={has_nan_mismatch}, "
            f"has_nonfinite_mismatch={has_nonfinite_mismatch}"
        )

    precision = (element_count - diff_count) / element_count * 100.0
    _display_comparison_result(
        output_name,
        npu_f32,
        golden_f32,
        mismatch,
        rtol=rtol,
        atol=atol,
        precision=precision,
        max_normalized_error=max_normalized_error,
        passed=passed,
    )

    return {
        "pass": passed,
        "precision": precision,
        "diff_indices": np.flatnonzero(mismatch)[:1000].tolist(),
        "error_info": error_info,
        "metrics": {
            "rtol": rtol,
            "atol": atol,
            "fail_ratio": fail_ratio,
            "fail_ratio_limit": _FAIL_RATIO_LIMIT,
            "max_abs_diff": max_abs_diff,
            "max_normalized_error": max_normalized_error,
            "normalization_epsilon": _NORMALIZATION_EPSILON,
            "has_nan_mismatch": has_nan_mismatch,
            "has_nonfinite_mismatch": has_nonfinite_mismatch,
        },
    }


def attention_compare(npu_out, golden_out):
    return _numpy_compare(npu_out, golden_out, "attention_out")


def lse_compare(npu_out, golden_out):
    return _numpy_compare(npu_out, golden_out, "softmax_lse")


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


def _is_absent_lse(value):
    """Treat None, an empty tensor, or the scalar ABI placeholder as no LSE."""
    if value is None:
        return True
    array = _as_numpy(value)
    return array.ndim == 0 or array.size == 0


def _compare_outputs(outputs, *, return_softmax_lse=None):
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return _failure_result(
            "output_contract",
            "invalid",
            f"compare expects even number of outputs, got {len(outputs)}",
        )

    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]
    results = [attention_compare(npu_outputs[0], golden_outputs[0])]

    npu_lse = npu_outputs[1] if half > 1 else None
    golden_lse = golden_outputs[1] if half > 1 else None

    # TTK does not guarantee that CSV operator attributes are forwarded to the
    # custom compare callback.  When the flag is absent, use the Golden output
    # contract as the source of truth: Golden emits None for LSE-off cases and
    # a non-empty tensor for LSE-on cases.  An explicitly forwarded flag still
    # takes precedence.
    if return_softmax_lse is None:
        return_softmax_lse = not _is_absent_lse(golden_lse)

    # The disabled-LSE contract permits only an absent output or the empty /
    # scalar ABI placeholder used by the fixed two-Tensor operator schema.
    # A real non-scalar LSE is an operator-contract violation, not an output to
    # silently ignore.
    if not return_softmax_lse:
        unexpected = []
        if not _is_absent_lse(npu_lse):
            unexpected.append(f"NPU shape={tuple(_as_numpy(npu_lse).shape)}")
        if not _is_absent_lse(golden_lse):
            unexpected.append(f"golden shape={tuple(_as_numpy(golden_lse).shape)}")
        if unexpected:
            results.append(
                _failure_result(
                    "softmax_lse",
                    "unexpected_lse",
                    "return_softmax_lse=False requires LSE to be absent "
                    f"or an empty/scalar placeholder; got {', '.join(unexpected)}",
                )
            )
            return results
        return results[0]

    # The enabled-LSE contract requires both sides to provide the second
    # output. Never silently turn a missing LSE into an attention-only pass.
    if npu_lse is None or golden_lse is None:
        missing = []
        if npu_lse is None:
            missing.append("NPU")
        if golden_lse is None:
            missing.append("golden")
        results.append(
            _failure_result(
                "softmax_lse",
                "missing_lse",
                "return_softmax_lse=True requires both NPU and golden "
                f"LSE outputs; missing {', '.join(missing)} LSE",
            )
        )
        return results

    npu_lse_array = _as_numpy(npu_lse)
    golden_lse_array = _as_numpy(golden_lse)
    invalid = []
    if npu_lse_array.ndim == 0 or npu_lse_array.size == 0:
        invalid.append(f"NPU shape={tuple(npu_lse_array.shape)}")
    if golden_lse_array.ndim == 0 or golden_lse_array.size == 0:
        invalid.append(f"golden shape={tuple(golden_lse_array.shape)}")
    if invalid:
        results.append(
            _failure_result(
                "softmax_lse",
                "invalid_lse",
                "return_softmax_lse=True requires non-scalar, non-empty "
                f"LSE outputs; invalid {', '.join(invalid)}",
            )
        )
        return results

    results.append(lse_compare(npu_lse, golden_lse))

    return results[0] if len(results) == 1 else results


def compare(*outputs, **kwargs):
    """Compare FP8 and MXFP8 outputs with one NumPy precision protocol."""
    return_softmax_lse = kwargs.get("return_softmax_lse")
    return _compare_outputs(
        outputs,
        return_softmax_lse=(
            None if return_softmax_lse is None else _as_bool(return_softmax_lse)
        ),
    )
