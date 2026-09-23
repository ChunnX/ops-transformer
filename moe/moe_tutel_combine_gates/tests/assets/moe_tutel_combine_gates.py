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
"""TTK TestSpec (golden + input customization) for MoeTutelCombineGates.

Kernel flow: TTK hands the inputs over as ``numpy.ndarray`` in operator-definition
order (``x``, ``y_grad``, ``indices``, ``locations``) and expects the golden to
return a list of numpy arrays.

The golden is an independent CPU implementation of

    gates_grad[b, s] = sum_h x[s, h] * y_grad[e * C + p, h]
    e = indices[b, s], p = locations[b, s]
    t = b * S + s  (the flattened task id both this file and the kernel use)

with the out-of-range routing rule ``e < 0`` / ``e >= B`` / ``p < 0`` / ``p >= C``
evaluating to exactly 0. It mirrors the kernel's accumulation contract: the
hidden-dimension products are computed exactly (a float64 product of two
float32/bfloat16/float16 values is exact, matching the kernel's TwoProduct
channel on the fp32 path and its exact fp32 products on the 16b paths), summed
in float64, and the sum is rounded once to the output dtype.
"""

from __future__ import annotations

import zlib

import numpy as np

__spec__ = {"moe_tutel_combine_gates": "MoeTutelCombineGatesTestSpec"}


def _routing(task_num: int, batch: int, capacity: int, seed: int, illegal: bool):
    """Deterministic (indices, locations) pair in the operator's [B, S] layout.

    ``illegal`` injects only the ``e < 0`` and ``p >= C`` classes: those two are
    never rejected by the validation layer on any path, so the case stays valid
    while still pinning the "out-of-range routing evaluates to exactly 0" rule.
    """
    if task_num == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    rng = np.random.default_rng(seed)
    exp = (
        rng.integers(0, batch, size=task_num, dtype=np.int64)
        if batch > 0
        else np.zeros(task_num, np.int64)
    )
    pos = (
        rng.integers(0, capacity, size=task_num, dtype=np.int64)
        if capacity > 0
        else np.zeros(task_num, np.int64)
    )
    if illegal:
        lanes = np.arange(task_num, dtype=np.int64)
        exp = np.where(lanes % 4 == 0, -1, exp)
        pos = np.where(lanes % 4 == 2, capacity, pos)
    return exp.astype(np.int32), pos.astype(np.int32)


def _golden(x, y_grad, indices, locations):
    """Exact CPU reference; returns the gates_grad array of shape indices.shape."""
    x_arr = np.ascontiguousarray(x)
    y_arr = np.ascontiguousarray(y_grad)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    loc = np.asarray(locations, dtype=np.int64).reshape(-1)

    samples, hidden = int(x_arr.shape[0]), int(x_arr.shape[1])
    batch, capacity = int(y_arr.shape[0]), int(y_arr.shape[1])
    task_num = idx.size

    result = np.zeros(task_num, dtype=np.float64)
    if task_num > 0 and hidden > 0 and batch > 0 and capacity > 0:
        # float64 holds the exact product of two fp32/bf16/fp16 values, so the
        # products feeding the sum are the kernel's exact products (TwoProduct
        # channel on the fp32 path, exact fp32 products on the 16b paths).
        xf = x_arr.astype(np.float64).reshape(samples, hidden)
        yf = y_arr.astype(np.float64).reshape(batch * capacity, hidden)
        valid = np.nonzero((idx >= 0) & (idx < batch) & (loc >= 0) & (loc < capacity))[
            0
        ]
        if valid.size > 0:
            slot = idx[valid] * capacity + loc[valid]
            token = valid % samples  # samples > 0 holds whenever task_num > 0
            products = xf[token] * yf[slot]
            result[valid] = products.sum(axis=1, dtype=np.float64)

    return result.astype(x_arr.dtype).reshape(indices.shape)


def customize_inputs(x, y_grad, indices, locations, *, testcase_name=None, **_kwargs):
    """Replace TTK's independent random routing by a deterministic valid one.

    TTK draws ``indices`` / ``locations`` independently of the shapes they index,
    which is not a legal routing. This hook keeps ``x`` / ``y_grad`` untouched and
    rewrites the two index tensors in place (the caller consumes the returned
    arrays) so that every task addresses a real ``y_grad`` row.
    """
    samples, hidden = int(x.shape[0]), int(x.shape[1])
    batch, capacity = int(y_grad.shape[0]), int(y_grad.shape[1])
    task_num = samples * batch
    illegal = bool(testcase_name) and "illegal" in testcase_name

    seed = zlib.crc32(
        f"{testcase_name}:{samples}x{hidden}x{batch}x{capacity}".encode("utf-8")
    )
    exp, pos = _routing(task_num, batch, capacity, seed, illegal)

    indices[...] = exp.reshape(indices.shape)
    locations[...] = pos.reshape(locations.shape)
    return x, y_grad, indices, locations


class MoeTutelCombineGatesTestSpec:
    """TestSpec for the Ascend 950 (arch35) MoeTutelCombineGates implementation."""

    def golden(x, y_grad, indices, locations, **_kwargs):
        return [_golden(x, y_grad, indices, locations)]

    customize_inputs = staticmethod(customize_inputs)

    # The golden is an exact independent CPU implementation, so the ecosystem
    # default standard (mix_tolerance, per-dtype built-in thresholds) applies.
    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
    }
