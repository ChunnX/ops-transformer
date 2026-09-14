#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MC2 precision policies ported from mc2_test_0416."""

import numpy as np


def _as_float32(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "element_size") and str(value.dtype) in ("torch.bfloat16", "bfloat16"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _relative_error(output, benchmark, threshold):
    absolute_error = np.abs(output - benchmark)
    denominator = np.maximum(
        np.maximum(np.abs(output), np.abs(benchmark)),
        (1.0 / (1 << 14)) / threshold,
    )
    return np.where(
        absolute_error < threshold,
        absolute_error,
        absolute_error / (denominator + 10e-10),
    )


def data_compare_np(
    output, benchmark, threshold=0.005, ratio_threshold=0.005, max_relative_error=2.0
):
    """AllGatherMatmul/Moe policy: direct output-vs-benchmark comparison."""
    output = _as_float32(output)
    benchmark = _as_float32(benchmark)
    if output.size != benchmark.size:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"output size {output.size} != third_party size {benchmark.size}",
        }
    if output.size == 0:
        return {"pass": True, "precision": 100.0}

    absolute_error = np.abs(output - benchmark)
    relative_error = _relative_error(output, benchmark, threshold)
    error_count = int(np.count_nonzero(relative_error > threshold))
    pass_ratio = (output.size - error_count) / output.size
    max_error = float(relative_error.max()) if error_count else 0.0
    passed = pass_ratio > 1.0 - ratio_threshold and max_error <= max_relative_error
    return {
        "pass": passed,
        "precision": pass_ratio * 100.0,
        "error_info": None
        if passed
        else (
            f"MC2 ratio failed: error_count={error_count}, "
            f"pass_ratio={pass_ratio:.8f}, required={1.0 - ratio_threshold:.8f}, "
            f"max_relative_error={max_error:.8g}, allowed={max_relative_error:.8g}"
        ),
        "metrics": {
            "standard": "data_compare_np",
            "threshold": threshold,
            "ratio_threshold": ratio_threshold,
            "max_relative_error_allowed": max_relative_error,
            "max_relative_error": max_error,
        },
    }


def moe_data_compare_np(output, benchmark, diff_thd=0.001, pct_thd=0.001, max_diff_hd=0.0):
    """Port moe_A3/toolss.py::data_compare_np with precision_method=0."""
    output = _as_float32(output)
    benchmark = _as_float32(benchmark)
    if output.size == 0 and benchmark.size == 0:
        return {"pass": True, "precision": 100.0, "max_relative_error": 0.0}
    if output.size != benchmark.size:
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"output size {output.size} != benchmark size {benchmark.size}",
        }

    absolute_error = np.abs(output - benchmark)
    has_nan_inf = bool(
        np.isnan(output).any()
        or np.isinf(output).any()
        or np.isnan(benchmark).any()
        or np.isinf(benchmark).any()
    )
    if has_nan_inf:
        finite_index = np.where(
            ~np.isnan(output)
            & ~np.isinf(output)
            & ~np.isnan(benchmark)
            & ~np.isinf(benchmark)
        )[0]
        special_index = np.setdiff1d(np.arange(output.size), finite_index)

        output_copy = output.copy()
        benchmark_copy = benchmark.copy()
        inf_value = 65504.0
        for data in (output_copy, benchmark_copy):
            positive_inf = np.where(np.isinf(data) & (data > 0))[0]
            negative_inf = np.where(np.isinf(data) & (data < 0))[0]
            data[positive_inf] = inf_value
            data[negative_inf] = -inf_value

        finite_relative_error = _relative_error(
            output_copy[finite_index], benchmark_copy[finite_index], diff_thd
        )
        finite_error_mask = finite_relative_error > diff_thd
        finite_error_index = finite_index[finite_error_mask]
        finite_error_diff = finite_relative_error[finite_error_mask]

        output_strings = np.asarray([str(value) for value in output[special_index]])
        benchmark_strings = np.asarray([str(value) for value in benchmark[special_index]])
        special_error_mask = output_strings != benchmark_strings
        special_error_index = special_index[special_error_mask]
        special_error_diff = absolute_error[special_error_index]

        error_index = np.concatenate((finite_error_index, special_error_index))
        error_diff = np.concatenate((finite_error_diff, special_error_diff))
    else:
        nonzero_diff_index = np.where(absolute_error > 0)[0]
        relative_error = _relative_error(
            output[nonzero_diff_index], benchmark[nonzero_diff_index], diff_thd
        )
        error_mask = relative_error > diff_thd
        error_index = nonzero_diff_index[error_mask]
        error_diff = relative_error[error_mask]

    pass_ratio = (output.size - error_diff.size) / output.size * 100.0
    passed = pass_ratio >= (1.0 - pct_thd) * 100.0
    max_error = 0.0
    if error_diff.size > 0:
        max_error = float(error_diff.max())
        if max_error >= max_diff_hd:
            passed = False

    return {
        "pass": passed,
        "precision": pass_ratio,
        "error_info": None
        if passed
        else (
            f"moe_A3 data_compare_np failed: pass_ratio={pass_ratio:.6f}%, "
            f"required={(1.0 - pct_thd) * 100.0:.6f}%, "
            f"max_relative_error={max_error:.8g}, allowed={max_diff_hd:.8g}"
        ),
        "metrics": {
            "standard": "moe_A3/toolss.py::data_compare_np",
            "diff_thd": diff_thd,
            "pct_thd": pct_thd,
            "max_diff_hd": max_diff_hd,
            "max_relative_error": max_error,
            "error_count": int(error_diff.size),
        },
    }


def _statistics(output, golden):
    original_dtype = None
    if hasattr(output, "dtype"):
        original_dtype = str(output.dtype)
    output = _as_float32(output)
    golden = _as_float32(golden)

    golden_nan = np.isnan(golden)
    output_nan = np.isnan(output)
    golden_inf = np.isinf(golden) & (golden > 0)
    output_inf = np.isinf(output) & (output > 0)
    golden_ninf = np.isinf(golden) & (golden < 0)
    output_ninf = np.isinf(output) & (output < 0)

    statistics = {
        "err_total_nan": int(np.sum(golden_nan ^ output_nan)),
        "err_total_inf": int(np.sum(golden_inf ^ output_inf)),
        "err_total_ninf": int(np.sum(golden_ninf ^ output_ninf)),
    }

    golden = golden.copy()
    output = output.copy()
    golden[np.isinf(golden)] = 1
    output[np.isinf(output)] = 1
    golden[np.isnan(golden)] = 1
    output[np.isnan(output)] = 1

    if original_dtype in ("torch.bfloat16", "bfloat16"):
        small_value = 0.001
        small_value_atol = 0.00001
    elif original_dtype == "torch.float32":
        small_value = 0.000001
        small_value_atol = 0.000000001
    else:
        small_value = 0.000025
        small_value_atol = 0.000000001
    big = np.abs(golden) >= small_value
    total_big_count = int(np.sum(big))
    total_small_count = int(golden.size - total_big_count)

    output_big = output.copy()
    golden_big = golden.copy()
    output_big[~big] = 1
    golden_big[~big] = 1
    diff_big = np.abs(output_big - golden_big)

    output_small = output.copy()
    golden_small = golden.copy()
    output_small[big] = 1
    golden_small[big] = 1
    diff_small = np.abs(output_small - golden_small)

    statistics.update(
        {
            "diff_big_max": float(diff_big.max()) if total_big_count else 0.0,
            "diff_big_avg": float(diff_big.sum() / total_big_count) if total_big_count else 0.0,
            "diff_big_sum": float(diff_big.sum()),
            "err_small_num": int(np.sum(diff_small > small_value_atol)),
            "diff_rmse": float(np.sqrt(np.mean(np.square(output - golden)))),
            "total_big_count": total_big_count,
            "total_small_count": total_small_count,
        }
    )
    return statistics


def _check_result(target, benchmark, moe=False):
    limits = (
        ("diff_big_max", 10.0),
        ("diff_big_avg", 2.0),
        ("diff_big_sum", 2.0),
    )
    check_small_errors = target["total_small_count"] >= 100000
    if moe:
        check_small_errors = benchmark["err_small_num"] >= 100
    if check_small_errors:
        limits += (("err_small_num", 2.0),)
    limits += (("diff_rmse", 2.0),)
    reasons = []
    reasons.extend(
        f"{name} {target[name]:.8g} > benchmark {benchmark[name]:.8g} * {limit:g}"
        for name, limit in limits
        if target[name] > benchmark[name] * limit
    )
    return not reasons, "; ".join(reasons)


def tripartite_compare(output, golden, third_party, moe=False):
    """MatmulReduceScatter policy: NPU and HCCL cascade vs CPU golden."""
    target = _statistics(output, golden)
    benchmark = _statistics(third_party, golden)
    passed, reason = _check_result(target, benchmark, moe=moe)
    precision = f"diff_big_avg={target['diff_big_avg']:.12g}"
    return {
        "pass": passed,
        "precision": precision,
        "error_info": None if passed else reason,
        "metrics": {
            "standard": "mc2_check_result",
            "target": target,
            "third_party": benchmark,
            "reason": reason,
        },
    }


def eplb_data_compare_np(output, golden, diff_thd=0.001, pct_thd=0.001, max_diff_hd=0.0):
    """MoeUpdateExpert policy ported from test_eplb.py."""
    output = _as_float32(output)
    golden = _as_float32(golden)
    if output.size != golden.size:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"size {output.size} != {golden.size}",
        }
    if output.size == 0:
        return {"pass": True, "precision": 100.0}
    absolute_error = np.abs(output - golden)
    relative_error = _relative_error(output, golden, diff_thd)
    error_count = int(np.count_nonzero(relative_error > diff_thd))
    pass_ratio = (output.size - error_count) / output.size * 100.0
    max_error = float(relative_error.max()) if error_count else 0.0
    passed = pass_ratio >= (1.0 - pct_thd) * 100.0 and max_error <= max_diff_hd
    return {
        "pass": passed,
        "precision": pass_ratio,
        "error_info": None
        if passed
        else f"pass_ratio={pass_ratio:.6f}%, max_relative_error={max_error:.8g}",
        "metrics": {
            "standard": "eplb_data_compare_np",
            "diff_thd": diff_thd,
            "pct_thd": pct_thd,
            "max_diff_hd": max_diff_hd,
            "absolute_error_max": float(absolute_error.max()),
        },
    }
