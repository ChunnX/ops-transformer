#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Precision comparison aligned with the SparseFlashMla pytest policy."""

import numpy as np


_DEFAULT_RTOL = 0.005
_BFLOAT16_RTOL = 0.0078125
_DEFAULT_ATOL = 0.000025
_BFLOAT16_ATOL = 0.0001
_FAIL_RATIO = 0.005
_MAX_RELATIVE_ERROR = 10.0
_RELATIVE_FLOOR = (1.0 / (1 << 14)) / 0.005
_RELATIVE_EPSILON = 2e-9


def _is_bfloat16(value):
    return "bfloat16" in str(getattr(value, "dtype", ""))


def _as_float32(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if _is_bfloat16(value):
            value = value.float()
        value = value.numpy()
    return np.asarray(value).astype(np.float32)


def _compare_output(npu_out, golden_out):
    if golden_out is None:
        return {"pass": True, "precision": "SUPPRESSED"}
    if npu_out is None:
        return {
            "pass": False,
            "precision": "NO_OUTPUT",
            "error_info": "NPU output is None",
        }

    is_bfloat16 = _is_bfloat16(npu_out)
    rtol = _BFLOAT16_RTOL if is_bfloat16 else _DEFAULT_RTOL
    atol = _BFLOAT16_ATOL if is_bfloat16 else _DEFAULT_ATOL
    npu = _as_float32(npu_out)
    golden = _as_float32(golden_out)
    if npu.shape != golden.shape:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"output shape mismatch: npu={npu.shape}, golden={golden.shape}",
        }
    if golden.size == 0:
        return {"pass": True, "precision": 100.0}

    npu_flat = npu.reshape(-1)
    golden_flat = golden.reshape(-1)
    mismatch = ~np.isclose(
        npu_flat, golden_flat, rtol=rtol, atol=atol, equal_nan=True)
    diff_idx = np.where(mismatch)[0]
    fail_ratio = diff_idx.size / golden_flat.size
    max_relative_error = 0.0
    if diff_idx.size:
        diff_abs = np.abs(golden_flat - npu_flat)
        denominator = np.maximum(
            np.maximum(np.abs(npu_flat), np.abs(golden_flat)), _RELATIVE_FLOOR)
        relative_error = diff_abs / (denominator + _RELATIVE_EPSILON)
        max_relative_error = float(np.max(relative_error[diff_idx]))

    passed = fail_ratio <= _FAIL_RATIO and max_relative_error < _MAX_RELATIVE_ERROR
    precision = (golden_flat.size - diff_idx.size) / golden_flat.size * 100
    error_info = None
    if not passed:
        error_info = (
            f"SMLA precision failed: mismatches={diff_idx.size}, "
            f"fail_ratio={fail_ratio:.6g}, max_relative_error={max_relative_error:.6g}"
        )
    return {
        "pass": passed,
        "precision": precision,
        "diff_indices": diff_idx[:1000].tolist(),
        "error_info": error_info,
        "metrics": {
            "rtol": rtol,
            "atol": atol,
            "fail_ratio": fail_ratio,
            "fail_ratio_limit": _FAIL_RATIO,
            "max_relative_error": max_relative_error,
            "max_relative_error_limit": _MAX_RELATIVE_ERROR,
        },
    }


def compare(*outputs):
    """Compare NPU outputs followed by golden outputs."""
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }
    half = len(outputs) // 2
    return [
        _compare_output(npu_out, golden_out)
        for npu_out, golden_out in zip(outputs[:half], outputs[half:])
    ]
