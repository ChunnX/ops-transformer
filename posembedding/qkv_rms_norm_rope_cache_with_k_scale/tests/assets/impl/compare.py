#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.

"""Contract-aware comparison for quantized Q/K and paged cache updates."""

from __future__ import annotations

import numpy as np

from .adapters import to_numpy


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
    return _rate(np.isclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=rtol, atol=atol, equal_nan=True,
    ))


def _cache_rows(cache, slots):
    cache = np.asarray(cache)
    block_size = cache.shape[2]
    return np.stack([
        cache[int(slot) // block_size, :, int(slot) % block_size, ...]
        for slot in slots
    ])


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

    inputs = [to_numpy(value) for value in compare_context.input_tensors]
    slots = np.asarray(inputs[4], dtype=np.int64)
    initial_k, initial_v, initial_ks = inputs[5], inputs[6], inputs[7]

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
            "q_bf16_close": _close_rate(q_actual, q_golden, 1e-2, 1e-2),
            "k_int8_abs_le_1": _rate(
                np.abs(k_rows.astype(np.int16) - k_golden_rows.astype(np.int16)) <= 1
            ),
            "k_scale_close": _close_rate(ks_rows, ks_golden_rows, 1e-2, 1e-5),
            "v_fp8_exact": _exact_rate(v_rows, v_golden_rows),
        })
        thresholds = {
            "q_bf16_close": 0.99,
            "k_int8_abs_le_1": 1.0,
            "k_scale_close": 0.99,
            "v_fp8_exact": 1.0,
        }
    else:
        metrics.update({
            "q_fp8_exact": _exact_rate(q_actual, q_golden),
            "q_dequant_close": _close_rate(
                np.asarray(q_actual, dtype=np.float32) * np.asarray(qs_actual, dtype=np.float32)[..., None],
                np.asarray(q_golden, dtype=np.float32) * np.asarray(qs_golden, dtype=np.float32)[..., None],
                1e-1, 1e-1,
            ),
            "q_scale_close": _close_rate(qs_actual, qs_golden, 1e-2, 1e-5),
            "k_fp8_exact": _exact_rate(k_rows, k_golden_rows),
            "k_dequant_close": _close_rate(
                np.asarray(k_rows, dtype=np.float32) * np.asarray(ks_rows, dtype=np.float32),
                np.asarray(k_golden_rows, dtype=np.float32) * np.asarray(ks_golden_rows, dtype=np.float32),
                1e-1, 1e-1,
            ),
            "k_scale_close": _close_rate(ks_rows, ks_golden_rows, 1e-2, 1e-5),
            "v_fp8_exact": _exact_rate(v_rows, v_golden_rows),
        })
        thresholds = {
            "q_fp8_exact": 0.99,
            "q_dequant_close": 0.99,
            "q_scale_close": 0.99,
            "k_fp8_exact": 0.99,
            "k_dequant_close": 0.99,
            "k_scale_close": 0.99,
            "v_fp8_exact": 1.0,
        }

    metrics.update({
        "k_cache_untouched": _untouched_exact(k_actual, initial_k, slots),
        "v_cache_untouched": _untouched_exact(v_actual, initial_v, slots),
        "k_scale_cache_untouched": _untouched_exact(ks_actual, initial_ks, slots),
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
    precision = "; ".join(f"{name}={value * 100:.3f}%" for name, value in metrics.items())
    return {
        "pass": not failures,
        "precision": precision,
        "error_info": "; ".join(failures),
        "metrics": metrics,
    }
