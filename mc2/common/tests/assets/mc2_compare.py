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


def data_compare_np(output, benchmark, threshold=0.005, ratio_threshold=0.005):
    """AllGatherMatmul policy: NPU output vs HCCL cascade."""
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
    passed = pass_ratio > 1.0 - ratio_threshold
    return {
        "pass": passed,
        "precision": pass_ratio * 100.0,
        "error_info": None
        if passed
        else (
            f"MC2 ratio failed: error_count={error_count}, "
            f"pass_ratio={pass_ratio:.8f}, required={1.0 - ratio_threshold:.8f}"
        ),
    }


def _statistics(output, golden):
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

    small_value = 0.001
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
            "err_small_num": int(np.sum(diff_small > small_value)),
            "diff_rmse": float(np.sqrt(np.mean(np.square(output - golden)))),
            "total_big_count": total_big_count,
            "total_small_count": total_small_count,
        }
    )
    return statistics


def _check_result(target, benchmark):
    limits = (
        ("diff_big_max", 10.0),
        ("diff_big_avg", 2.0),
        ("diff_big_sum", 2.0),
        ("err_small_num", 2.0),
        ("diff_rmse", 2.0),
    )
    reasons = []
    reasons.extend(
        f"{name} {target[name]:.8g} > benchmark {benchmark[name]:.8g} * {limit:g}"
        for name, limit in limits
        if target[name] > max(2, benchmark[name] * limit)
    )
    for name in ("err_total_nan", "err_total_inf", "err_total_ninf"):
        if target[name] > max(2, benchmark[name]):
            reasons.append(
                f"{name} target={target[name]} benchmark={benchmark[name]} allowed={max(2, benchmark[name])}"
            )
    return not reasons, "; ".join(reasons)


def tripartite_compare(output, golden, third_party):
    """MatmulReduceScatter policy: NPU and HCCL cascade vs CPU golden."""
    target = _statistics(output, golden)
    benchmark = _statistics(third_party, golden)
    passed, reason = _check_result(target, benchmark)
    precision = target["diff_big_avg"]
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
