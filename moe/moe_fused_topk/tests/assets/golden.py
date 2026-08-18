#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Shared ACLNN and direct-kernel Golden/Input helpers for MoeFusedTopk ST."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


__golden__ = {
    "aclnn": {"aclnnMoeFusedTopk": "aclnn_moe_fused_topk_golden"},
    "kernel": {"moe_fused_topk": "moe_fused_topk_golden"},
}

__input__ = {
    "aclnn": {"aclnnMoeFusedTopk": "aclnn_moe_fused_topk_input"},
    "kernel": {"moe_fused_topk": "moe_fused_topk_input"},
}


@lru_cache(maxsize=1)
def _special_inputs() -> dict[str, dict[str, str]]:
    """Load the NaN/Inf cases retained from the legacy ATK matrix."""
    source = (
        Path(__file__).parents[1]
        / "st"
        / "aclnnMoeFusedTopk"
        / "atk_aclnnMoeFusedTopk.json"
    )
    result: dict[str, dict[str, str]] = {}
    for case in json.loads(source.read_text(encoding="utf-8")):
        values = {item["name"]: item["range_values"] for item in case["inputs"]}
        special = {
            name: value[0]
            for name, value in values.items()
            if name in {"x", "add_num"}
            and isinstance(value, list)
            and len(value) == 1
            and value[0] in {"nan", "inf", "-inf"}
        }
        if special:
            result[f"moe_fused_topk_atk_{case['id']:03d}"] = special
            result[f"moe_fused_topk_kernel_atk_{case['id']:03d}"] = special
    return result


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except TypeError:
            # PyTorch does not expose a NumPy BF16 view on every platform.
            return value.float().numpy()
    return np.asarray(value)


def _write_tensor(target, value) -> None:
    """Write a NumPy value into either a NumPy or a torch test tensor."""
    if isinstance(target, np.ndarray):
        target[...] = np.asarray(value, dtype=target.dtype)
        return
    import torch

    target.copy_(torch.as_tensor(value, device=target.device, dtype=target.dtype))


def _fill_stable_inputs(x, add_num) -> None:
    """Avoid accidental TopK ties across FP16, BF16 and FP32 test runs."""
    x_shape = _as_numpy(x).shape
    expert_count = _as_numpy(add_num).shape[0]
    ranks = np.arange(expert_count, dtype=np.int32)
    values = -np.ldexp((1 + ranks % 127).astype(np.float32), -20 + ranks // 127).astype(
        np.float32
    )
    _write_tensor(x, np.zeros(x_shape, dtype=np.float32))
    _write_tensor(add_num, values)


def _customize_inputs(x, add_num, testcase_name) -> None:
    _fill_stable_inputs(x, add_num)
    for name, value in _special_inputs().get(testcase_name, {}).items():
        target = x if name == "x" else add_num
        _write_tensor(target, np.float32(value))


def aclnn_moe_fused_topk_input(
    x,
    addNum,
    mappingNum,
    mappingTable,
    groupNum,
    groupTopk,
    topN,
    topK,
    activateType,
    isNorm,
    scale,
    enableExpertMapping,
    y=None,
    indices=None,
    *,
    testcase_name=None,
    **_kwargs,
):
    del mappingNum, mappingTable, groupNum, groupTopk, topN, topK
    del activateType, isNorm, scale, enableExpertMapping, y, indices
    _customize_inputs(x, addNum, testcase_name)


def moe_fused_topk_input(
    x,
    add_num,
    mapping_num,
    mapping_table,
    *,
    testcase_name=None,
    **_kwargs,
):
    _customize_inputs(x, add_num, testcase_name)
    return x, add_num, mapping_num, mapping_table


def _calculate(
    x,
    add_num,
    mapping_num,
    mapping_table,
    group_num,
    group_topk,
    top_n,
    top_k,
    is_norm,
    scale,
    enable_expert_mapping,
):
    x_float = _as_numpy(x).astype(np.float32, copy=False)
    add_num_float = _as_numpy(add_num).astype(np.float32, copy=False)
    mapping_num_int = _as_numpy(mapping_num).astype(np.int64, copy=False)
    mapping_table_int = _as_numpy(mapping_table).astype(np.int32, copy=False)

    with np.errstate(over="ignore", invalid="ignore"):
        sigmoid = 1.0 / (1.0 + np.exp(-x_float))
        scores = sigmoid + add_num_float

    token_count, expert_count = scores.shape
    group_num = int(group_num)
    group_topk = int(group_topk)
    top_n = int(top_n)
    top_k = int(top_k)
    grouped = scores.reshape(token_count, group_num, expert_count // group_num)
    group_order = np.argsort(
        -np.take_along_axis(
            grouped,
            np.argsort(-grouped, axis=-1, kind="stable")[..., :top_n],
            axis=-1,
        ).sum(axis=-1),
        axis=-1,
        kind="stable",
    )

    masked = grouped.copy()
    if group_topk < group_num:
        deselected = group_order[:, group_topk:]
        rows = np.arange(token_count)[:, None]
        masked[rows, deselected, :] = 0.0

    selected_indices = np.argsort(
        -masked.reshape(token_count, expert_count), axis=-1, kind="stable"
    )[:, :top_k]
    result = np.take_along_axis(sigmoid, selected_indices, axis=-1)
    if bool(is_norm):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = result / result.sum(axis=-1, keepdims=True)
            result = result * np.float32(scale)

    output_indices = selected_indices.astype(np.int32, copy=True)
    if bool(enable_expert_mapping):
        for batch_index in range(token_count):
            for topk_index in range(top_k):
                expert_id = int(selected_indices[batch_index, topk_index])
                offset = batch_index % max(1, int(mapping_num_int[expert_id]))
                output_indices[batch_index, topk_index] = mapping_table_int[
                    expert_id, offset
                ]
    return result.astype(np.float32, copy=False), output_indices


def _to_output(value, template):
    if isinstance(template, np.ndarray):
        return np.asarray(value, dtype=template.dtype)
    import torch

    return torch.as_tensor(value, device=template.device, dtype=template.dtype)


def moe_fused_topk_golden(
    x,
    add_num,
    mapping_num,
    mapping_table,
    *,
    group_num,
    group_topk,
    top_n,
    top_k,
    activate_type,
    is_norm,
    scale,
    enable_expert_mapping,
    **_kwargs,
):
    del activate_type
    return _calculate(
        x,
        add_num,
        mapping_num,
        mapping_table,
        group_num,
        group_topk,
        top_n,
        top_k,
        is_norm,
        scale,
        enable_expert_mapping,
    )


def aclnn_moe_fused_topk_golden(
    x,
    addNum,
    mappingNum,
    mappingTable,
    groupNum,
    groupTopk,
    topN,
    topK,
    activateType,
    isNorm,
    scale,
    enableExpertMapping,
    y,
    indices,
    **_kwargs,
):
    del activateType
    result, output_indices = _calculate(
        x,
        addNum,
        mappingNum,
        mappingTable,
        groupNum,
        groupTopk,
        topN,
        topK,
        isNorm,
        scale,
        enableExpertMapping,
    )
    return _to_output(result, y), _to_output(output_indices, indices)
