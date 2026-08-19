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

"""Two-way FP8 and method-12 three-way MXFP8 comparison for TTK."""

import importlib.util
import logging
import os
import sys
import threading

import numpy as np

_FP32_RTOL = 0.005
_FP32_ATOL = 0.000025
_BF16_RTOL = 0.0078125
_BF16_ATOL = 0.0001
_FAIL_RATIO_LIMIT = 0.005
_MAX_NORMALIZED_ERROR = 10.0
_NORMALIZATION_EPSILON = 1e-10
_NORMALIZATION_FLOOR = (1.0 / (1 << 14)) / _FP32_RTOL
_MXFP8_CV_STATE = threading.local()
_MXFP8_CV_COMPARE_MODULE = None


def set_mxfp8_cv_benchmark(outputs):
    """Pair the latest low-precision MXFP8 CPU result with TTK outputs."""
    _MXFP8_CV_STATE.benchmark_outputs = outputs


def _take_mxfp8_cv_benchmark():
    outputs = getattr(_MXFP8_CV_STATE, "benchmark_outputs", None)
    _MXFP8_CV_STATE.benchmark_outputs = None
    return outputs


def _load_mxfp8_cv_compare_module():
    """Load the FIA method-12 implementation used as the CV reference."""
    global _MXFP8_CV_COMPARE_MODULE
    if _MXFP8_CV_COMPARE_MODULE is not None:
        return _MXFP8_CV_COMPARE_MODULE

    relative_path = os.path.join(
        "attention",
        "fused_infer_attention_score",
        "tests",
        "pytest",
        "fia_fullquant_mxfp8_test",
        "common",
        "precision_compare_v2.py",
    )
    search_root = os.path.dirname(os.path.abspath(__file__))
    module_path = None
    while True:
        candidate = os.path.join(search_root, relative_path)
        if os.path.isfile(candidate):
            module_path = candidate
            break
        parent = os.path.dirname(search_root)
        if parent == search_root:
            break
        search_root = parent
    if module_path is None:
        raise ImportError(
            "cannot locate FIA precision_compare_v2.py for QBSA three-way compare"
        )

    module_name = "_qbsa_fia_precision_compare_v2"
    if module_name in sys.modules:
        _MXFP8_CV_COMPARE_MODULE = sys.modules[module_name]
        return _MXFP8_CV_COMPARE_MODULE
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CV compare module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    original_basic_config = logging.basicConfig
    logging.basicConfig = lambda *args, **kwargs: None
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        logging.basicConfig = original_basic_config
    _MXFP8_CV_COMPARE_MODULE = module
    return module


def _mxfp8_cv_report_metrics(report):
    return {
        "status": report.status,
        "failure_reasons": list(report.failure_reasons),
        "npu_diff_big_ratio_max": report.npu_diff_big_ratio_max,
        "npu_diff_big_ratio_avg": report.npu_diff_big_ratio_avg,
        "npu_diff_big_ratio_rmse": report.npu_diff_big_ratio_rmse,
        "benchmark_diff_big_ratio_max": report.bench_diff_big_ratio_max,
        "benchmark_diff_big_ratio_avg": report.bench_diff_big_ratio_avg,
        "benchmark_diff_big_ratio_rmse": report.bench_diff_big_ratio_rmse,
        "npu_err_h1_ratio": report.npu_err_h1_ratio,
        "benchmark_err_h1_ratio": report.bench_err_h1_ratio,
    }


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
    """Compare one FP8 output with the original NumPy precision protocol."""
    npu = _as_numpy(npu_out)
    golden = _as_numpy(golden_out)

    if npu.shape != golden.shape:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": (
                f"{output_name} shape mismatch: "
                f"npu_shape={tuple(npu.shape)}, "
                f"golden_shape={tuple(golden.shape)}"
            ),
        }

    if npu.size == 0:
        return {"pass": True, "precision": 100.0}

    rtol, atol = _compare_tolerance(npu_out, golden_out)
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

    return {
        "pass": passed,
        "precision": (element_count - diff_count) / element_count * 100.0,
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


def _mxfp8_cv_compare(
    npu_out, golden_out, benchmark_out, output_name, output_dtype
):
    """Run FIA method-12 for one MXFP8 output only."""
    direct_result = _numpy_compare(npu_out, golden_out, output_name)
    if direct_result.get("precision") in {
        "shape_mismatch",
        "invalid",
    }:
        return direct_result

    npu = _as_numpy(npu_out)
    golden = _as_numpy(golden_out)
    benchmark = _as_numpy(benchmark_out)
    if npu.size == 0:
        return direct_result
    if benchmark.shape != golden.shape:
        return {
            "pass": False,
            "precision": "benchmark_shape_mismatch",
            "error_info": (
                f"{output_name} benchmark shape mismatch: "
                f"benchmark_shape={tuple(benchmark.shape)}, "
                f"golden_shape={tuple(golden.shape)}"
            ),
        }

    try:
        cv_module = _load_mxfp8_cv_compare_module()
        report = cv_module.compute_cv_report(
            golden=golden,
            benchmark=benchmark,
            actual=npu,
            test_name=f"qbsa_mxfp8_{output_name}",
            output_dtype=output_dtype,
        )
        cv_module.display_cv_report(report)
    except Exception as error:  # pylint: disable=broad-except
        return {
            "pass": False,
            "precision": "cv_compare_error",
            "error_info": f"{output_name} three-way compare failed: {error}",
        }

    direct_tolerance_pass = direct_result["pass"]
    direct_result["pass"] = bool(report.passed)
    direct_result["metrics"]["direct_tolerance_pass"] = direct_tolerance_pass
    direct_result["metrics"]["cv"] = _mxfp8_cv_report_metrics(report)
    direct_result["error_info"] = (
        None
        if report.passed
        else f"{output_name} CV failed: {', '.join(report.failure_reasons)}"
    )
    return direct_result


def _mxfp8_attention_compare(npu_out, golden_out, benchmark_out):
    return _mxfp8_cv_compare(
        npu_out, golden_out, benchmark_out, "attention_out", "bf16"
    )


def _mxfp8_lse_compare(npu_out, golden_out, benchmark_out):
    return _mxfp8_cv_compare(
        npu_out, golden_out, benchmark_out, "softmax_lse", "fp32"
    )


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
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": f"compare expects even number of outputs, got {len(outputs)}",
        }

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
                {
                    "pass": False,
                    "precision": "unexpected_lse",
                    "error_info": (
                        "return_softmax_lse=False requires LSE to be absent "
                        f"or an empty/scalar placeholder; got {', '.join(unexpected)}"
                    ),
                }
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
            {
                "pass": False,
                "precision": "missing_lse",
                "error_info": (
                    "return_softmax_lse=True requires both NPU and golden "
                    f"LSE outputs; missing {', '.join(missing)} LSE"
                ),
            }
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
            {
                "pass": False,
                "precision": "invalid_lse",
                "error_info": (
                    "return_softmax_lse=True requires non-scalar, non-empty "
                    f"LSE outputs; invalid {', '.join(invalid)}"
                ),
            }
        )
        return results

    results.append(lse_compare(npu_lse, golden_lse))

    return results[0] if len(results) == 1 else results


def _compare_mxfp8_outputs(
    outputs, benchmark_outputs, *, return_softmax_lse=None
):
    """Compare MXFP8 high-CPU/low-CPU/NPU outputs without touching FP8."""
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": f"MXFP8 compare expects even outputs, got {len(outputs)}",
        }
    if not benchmark_outputs:
        return {
            "pass": False,
            "precision": "missing_mxfp8_benchmark",
            "error_info": "MXFP8 three-way compare requires low-precision CPU outputs",
        }

    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]
    results = [
        _mxfp8_attention_compare(
            npu_outputs[0], golden_outputs[0], benchmark_outputs[0]
        )
    ]

    npu_lse = npu_outputs[1] if half > 1 else None
    golden_lse = golden_outputs[1] if half > 1 else None
    benchmark_lse = benchmark_outputs[1] if len(benchmark_outputs) > 1 else None
    if return_softmax_lse is None:
        return_softmax_lse = not _is_absent_lse(golden_lse)

    if not return_softmax_lse:
        unexpected = []
        if not _is_absent_lse(npu_lse):
            unexpected.append(f"NPU shape={tuple(_as_numpy(npu_lse).shape)}")
        if not _is_absent_lse(golden_lse):
            unexpected.append(f"golden shape={tuple(_as_numpy(golden_lse).shape)}")
        if benchmark_lse is not None and not _is_absent_lse(benchmark_lse):
            unexpected.append(
                f"benchmark shape={tuple(_as_numpy(benchmark_lse).shape)}"
            )
        if unexpected:
            results.append(
                {
                    "pass": False,
                    "precision": "unexpected_lse",
                    "error_info": (
                        "return_softmax_lse=False requires MXFP8 LSE outputs "
                        f"to be absent; got {', '.join(unexpected)}"
                    ),
                }
            )
            return results
        return results[0]

    missing = []
    if npu_lse is None:
        missing.append("NPU")
    if golden_lse is None:
        missing.append("golden")
    if benchmark_lse is None:
        missing.append("benchmark")
    if missing:
        results.append(
            {
                "pass": False,
                "precision": "missing_lse",
                "error_info": (
                    "return_softmax_lse=True requires all three MXFP8 LSE "
                    f"outputs; missing {', '.join(missing)}"
                ),
            }
        )
        return results

    invalid = []
    for name, value in (
        ("NPU", npu_lse),
        ("golden", golden_lse),
        ("benchmark", benchmark_lse),
    ):
        array = _as_numpy(value)
        if array.ndim == 0 or array.size == 0:
            invalid.append(f"{name} shape={tuple(array.shape)}")
    if invalid:
        results.append(
            {
                "pass": False,
                "precision": "invalid_lse",
                "error_info": (
                    "return_softmax_lse=True requires non-scalar, non-empty "
                    f"MXFP8 LSE outputs; invalid {', '.join(invalid)}"
                ),
            }
        )
        return results

    results.append(_mxfp8_lse_compare(npu_lse, golden_lse, benchmark_lse))
    return results


def compare(*outputs, **kwargs):
    """Keep FP8 two-way comparison separate from MXFP8 method-12."""
    return_softmax_lse = kwargs.get("return_softmax_lse")
    return_softmax_lse = (
        None if return_softmax_lse is None else _as_bool(return_softmax_lse)
    )
    mxfp8_benchmark = _take_mxfp8_cv_benchmark()
    if mxfp8_benchmark is not None:
        return _compare_mxfp8_outputs(
            outputs,
            mxfp8_benchmark,
            return_softmax_lse=return_softmax_lse,
        )
    return _compare_outputs(outputs, return_softmax_lse=return_softmax_lse)
