#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.

"""Contract-aware comparison for quantized Q/K and paged cache updates."""

from __future__ import annotations

import numpy as np

from .adapters import to_numpy


BF16_RTOL = 1e-3
BF16_ATOL = 1e-8
BF16_PTOL = 1e-3
FP8_RTOL = 1e-2
FP8_ATOL = 1e-8
FP8_PTOL = 1e-2
SCALE_FP32_RTOL = 1e-3
SCALE_FP32_ATOL = 1e-8
SCALE_FP32_PTOL = 1e-3
INT8_MAX_ABS_DIFF = 1


def _rate(mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    return 1.0 if mask.size == 0 else float(np.count_nonzero(mask) / mask.size)


def _exact_rate(actual, expected) -> float:
    actual = np.ascontiguousarray(actual)
    expected = np.ascontiguousarray(expected)
    if actual.shape == expected.shape and actual.dtype.itemsize == expected.dtype.itemsize:
        itemsize = actual.dtype.itemsize
        actual_bytes = actual.view(np.uint8).reshape(actual.shape + (itemsize,))
        expected_bytes = expected.view(np.uint8).reshape(expected.shape + (itemsize,))
        return _rate(np.all(actual_bytes == expected_bytes, axis=-1))
    return _rate(actual.astype(np.float32) == expected.astype(np.float32))


def _close_rate(actual, expected, rtol, atol) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return 0.0
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    matched = 0
    chunk_elements = 1 << 20
    for begin in range(0, actual_flat.size, chunk_elements):
        end = min(actual_flat.size, begin + chunk_elements)
        matched += int(np.count_nonzero(np.isclose(
            np.asarray(actual_flat[begin:end], dtype=np.float32),
            np.asarray(expected_flat[begin:end], dtype=np.float32),
            rtol=rtol, atol=atol, equal_nan=True,
        )))
    return 1.0 if actual_flat.size == 0 else matched / actual_flat.size


def _abs_le_rate(actual, expected, limit) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return 0.0
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    matched = 0
    chunk_elements = 1 << 20
    for begin in range(0, actual_flat.size, chunk_elements):
        end = min(actual_flat.size, begin + chunk_elements)
        difference = np.abs(
            actual_flat[begin:end].astype(np.int16, copy=False)
            - expected_flat[begin:end].astype(np.int16, copy=False)
        )
        matched += int(np.count_nonzero(difference <= limit))
    return 1.0 if actual_flat.size == 0 else matched / actual_flat.size


def _cache_rows(cache, slots):
    cache = np.asarray(cache)
    block_size = cache.shape[2]
    slots = np.asarray(slots, dtype=np.int64)
    return cache[slots // block_size, :, slots % block_size, ...]


def _untouched_exact(actual, expected, slots) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    mask = np.ones((actual.shape[0], actual.shape[2]), dtype=bool)
    block_size = actual.shape[2]
    for slot in slots:
        mask[int(slot) // block_size, int(slot) % block_size] = False
    actual_rows = np.transpose(actual, (0, 2, 1, 3))[mask]
    expected_rows = np.transpose(expected, (0, 2, 1, 3))[mask]
    return _exact_rate(actual_rows, expected_rows)


def compare_outputs(*values, compare_context):
    output_count = len(values) // 2
    # Kernel hooks receive NumPy arrays, while ACLNN hooks may receive CPU
    # Torch tensors.  Normalize here because Torch cannot expose BF16/FP8
    # tensors through Tensor.numpy() directly.
    outputs = [to_numpy(value) for value in values[:output_count]]
    goldens = [to_numpy(value) for value in values[output_count:]]
    attrs = dict(compare_context.attributes)
    api_name = str(compare_context.api_name)
    q_quant_mode = attrs.get("q_quant_mode", attrs.get("qQuantMode", "PerTokenPerHead"))
    is_mrope = str(q_quant_mode) == "NoQuant"
    is_kernel = api_name == "qkv_rms_norm_rope_cache_with_k_scale"
    is_functional = api_name.endswith("pta_functional")

    # Only slot_mapping is needed from the inputs.  Converting every input here
    # would duplicate multi-GiB QKV/cos tensors in the full-800 matrix.
    inputs = compare_context.input_tensors
    slots = np.asarray(to_numpy(inputs[4]), dtype=np.int64)

    if is_mrope and is_kernel:
        q_actual, _, k_actual, v_actual, ks_actual = outputs[:5]
        q_golden, _, k_golden, v_golden, ks_golden = goldens[:5]
    elif is_mrope:
        q_actual, k_actual, v_actual, ks_actual = outputs[:4]
        q_golden, k_golden, v_golden, ks_golden = goldens[:4]
    else:
        q_actual, qs_actual, k_actual, v_actual, ks_actual = outputs[:5]
        q_golden, qs_golden, k_golden, v_golden, ks_golden = goldens[:5]

    k_rows = _cache_rows(k_actual, slots)
    k_golden_rows = _cache_rows(k_golden, slots)
    v_rows = _cache_rows(v_actual, slots)
    v_golden_rows = _cache_rows(v_golden, slots)
    ks_rows = _cache_rows(ks_actual, slots)
    ks_golden_rows = _cache_rows(ks_golden, slots)

    metrics = {}
    if is_mrope:
        metrics.update({
            "q_bf16_close": _close_rate(
                q_actual, q_golden, BF16_RTOL, BF16_ATOL
            ),
            "k_int8_abs_le_1": _abs_le_rate(
                k_rows, k_golden_rows, INT8_MAX_ABS_DIFF
            ),
            "k_scale_fp32_close": _close_rate(
                ks_rows, ks_golden_rows, SCALE_FP32_RTOL, SCALE_FP32_ATOL
            ),
            "v_fp8_close": _close_rate(
                v_rows, v_golden_rows, FP8_RTOL, FP8_ATOL
            ),
        })
        thresholds = {
            "q_bf16_close": 1.0 - BF16_PTOL,
            "k_int8_abs_le_1": 1.0,
            "k_scale_fp32_close": 1.0 - SCALE_FP32_PTOL,
            "v_fp8_close": 1.0 - FP8_PTOL,
        }
    else:
        metrics.update({
            "q_fp8_close": _close_rate(
                q_actual, q_golden, FP8_RTOL, FP8_ATOL
            ),
            "q_scale_fp32_close": _close_rate(
                qs_actual, qs_golden, SCALE_FP32_RTOL, SCALE_FP32_ATOL
            ),
            "k_fp8_close": _close_rate(
                k_rows, k_golden_rows, FP8_RTOL, FP8_ATOL
            ),
            "k_scale_fp32_close": _close_rate(
                ks_rows, ks_golden_rows, SCALE_FP32_RTOL, SCALE_FP32_ATOL
            ),
            "v_fp8_close": _close_rate(
                v_rows, v_golden_rows, FP8_RTOL, FP8_ATOL
            ),
        })
        thresholds = {
            "q_fp8_close": 1.0 - FP8_PTOL,
            "q_scale_fp32_close": 1.0 - SCALE_FP32_PTOL,
            "k_fp8_close": 1.0 - FP8_PTOL,
            "k_scale_fp32_close": 1.0 - SCALE_FP32_PTOL,
            "v_fp8_close": 1.0 - FP8_PTOL,
        }

    metrics.update({
        "k_cache_untouched": _untouched_exact(k_actual, k_golden, slots),
        "v_cache_untouched": _untouched_exact(v_actual, v_golden, slots),
        "k_scale_cache_untouched": _untouched_exact(ks_actual, ks_golden, slots),
    })
    thresholds.update({
        "k_cache_untouched": 1.0,
        "v_cache_untouched": 1.0,
        "k_scale_cache_untouched": 1.0,
    })

    if is_functional:
        original_actual = outputs[-3:]
        original_golden = goldens[-3:]
        for name, actual, expected in zip(
            ("functional_k_input", "functional_v_input", "functional_k_scale_input"),
            original_actual, original_golden,
        ):
            metrics[name] = _exact_rate(actual, expected)
            thresholds[name] = 1.0

    failures = [
        f"{name}={metrics[name]:.6f} < {threshold:.6f}"
        for name, threshold in thresholds.items()
        if metrics[name] + 1e-12 < threshold
    ]
    precision = "; ".join(f"{name}={value * 100:.6f}%" for name, value in metrics.items())
    return {
        "pass": not failures,
        "precision": precision,
        "error_info": "; ".join(failures),
        "metrics": metrics,
    }
