#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""PyTorch Golden and input customization for MoeFusedTopk tests."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


__spec__ = {"moe_fused_topk": "MoeFusedTopkTestSpec"}

# Retain the legacy registrations while ATK cases migrate to TTK.
__golden__ = {
    "aclnn": {"aclnnMoeFusedTopk": "aclnn_moe_fused_topk_golden"},
    "kernel": {"moe_fused_topk": "moe_fused_topk_golden"},
}
__input__ = {
    "aclnn": {"aclnnMoeFusedTopk": "aclnn_moe_fused_topk_input"},
    "kernel": {"moe_fused_topk": "moe_fused_topk_input"},
}


def _check_activation(activate_type) -> None:
    if int(activate_type) != 0:
        raise ValueError(
            f"MoeFusedTopk only supports activate_type=0 (sigmoid), got {activate_type}"
        )


def _score_tensor(value: torch.Tensor) -> torch.Tensor:
    """Match kernel FP32 intermediates, while retaining TTK Promote FP64."""
    if value.dtype in (torch.float32, torch.float64):
        return value
    return value.float()


def _rankable(value: torch.Tensor) -> torch.Tensor:
    """Give non-finite scores the deterministic ordering used by the kernel."""
    limits = torch.finfo(value.dtype)
    return torch.nan_to_num(value, nan=limits.min, posinf=limits.max, neginf=limits.min)


def _torch_moe_fused_topk(
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
):
    """Implement MoeFusedTopk as a chain of small PyTorch operations."""
    _check_activation(activate_type)
    group_num = int(group_num)
    group_topk = int(group_topk)
    top_n = int(top_n)
    top_k = int(top_k)

    x_score = _score_tensor(x)
    add_score = _score_tensor(add_num)
    sigmoid = torch.sigmoid(x_score)
    scores = sigmoid + add_score

    token_num, expert_num = scores.shape
    grouped = scores.reshape(token_num, group_num, expert_num // group_num)

    # NaN cannot participate in routing. Normalize the ranking copy only so
    # the gathered sigmoid output preserves its IEEE value.
    group_top_values = torch.topk(_rankable(grouped), top_n, dim=-1).values
    group_scores = _rankable(torch.sum(group_top_values, dim=-1))
    group_order = torch.argsort(group_scores, dim=-1, descending=True, stable=True)

    selected_group = torch.zeros_like(group_scores, dtype=torch.bool)
    selected_group.scatter_(1, group_order[:, :group_topk], True)
    masked_scores = torch.where(selected_group.unsqueeze(-1), grouped, 0.0)
    flat_rank_scores = _rankable(masked_scores.reshape(token_num, expert_num))
    selected_indices = torch.argsort(
        flat_rank_scores, dim=-1, descending=True, stable=True
    )[:, :top_k]

    result = torch.gather(sigmoid, -1, selected_indices)
    if bool(is_norm):
        result = result / torch.sum(result, dim=-1, keepdim=True)
        result = result * float(scale)

    output_indices = selected_indices.to(torch.int32)
    if bool(enable_expert_mapping):
        if mapping_num is None or mapping_table is None:
            raise ValueError(
                "mapping_num and mapping_table are required when expert mapping is enabled"
            )
        raw_indices = selected_indices
        counts = mapping_num[raw_indices].to(torch.int64).clamp_min(1)
        token_offsets = torch.arange(
            token_num, device=raw_indices.device, dtype=torch.int64
        ).unsqueeze(1)
        redundant_offsets = token_offsets.remainder(counts)
        output_indices = mapping_table[raw_indices, redundant_offsets].to(torch.int32)

    return [result, output_indices]


def _numpy_to_torch(value):
    if value is None:
        return None
    array = np.ascontiguousarray(value)
    # NumPy/ML-dtypes BF16 has no torch.from_numpy bridge on some hosts.
    if str(array.dtype) not in {
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "bool",
    }:
        array = array.astype(np.float32)
    return torch.from_numpy(array)


@lru_cache(maxsize=1)
def _special_inputs() -> dict[str, dict[str, str]]:
    """Load only explicitly requested NaN/Inf inputs from the legacy matrix."""
    source = (
        Path(__file__).parents[1]
        / "st"
        / "aclnnMoeFusedTopk"
        / "atk_aclnnMoeFusedTopk.json"
    )
    result: dict[str, dict[str, str]] = {}
    for case in json.loads(source.read_text(encoding="utf-8")):
        special = {}
        for item in case["inputs"]:
            values = item.get("range_values")
            if (
                item["name"] in {"x", "add_num"}
                and isinstance(values, list)
                and len(values) == 1
                and values[0] in {"nan", "inf", "-inf"}
            ):
                special[item["name"]] = values[0]
        if special:
            result[f"moe_fused_topk_atk_{case['id']:03d}"] = special
    return result


def _write_tensor(target, value) -> None:
    if isinstance(target, np.ndarray):
        target[...] = np.asarray(value, dtype=target.dtype)
    else:
        target.copy_(torch.as_tensor(value, device=target.device, dtype=target.dtype))


def _customize_inputs(x, add_num, testcase_name):
    """Leave normal random inputs intact; inject only declared IEEE specials."""
    for name, value in _special_inputs().get(testcase_name, {}).items():
        _write_tensor(x if name == "x" else add_num, float(value))
    return x, add_num


class MoeFusedTopkTestSpec:
    """TTK kernel TestSpec for the Ascend 950 implementation."""

    def golden(
        x,
        add_num,
        mapping_num=None,
        mapping_table=None,
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
        inputs = [_numpy_to_torch(v) for v in (x, add_num, mapping_num, mapping_table)]
        return [
            output.numpy()
            for output in _torch_moe_fused_topk(
                *inputs,
                group_num=group_num,
                group_topk=group_topk,
                top_n=top_n,
                top_k=top_k,
                activate_type=activate_type,
                is_norm=is_norm,
                scale=scale,
                enable_expert_mapping=enable_expert_mapping,
            )
        ]

    class TorchImpl:
        def __init__(
            self,
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
            self.attrs = {
                "group_num": group_num,
                "group_topk": group_topk,
                "top_n": top_n,
                "top_k": top_k,
                "activate_type": activate_type,
                "is_norm": is_norm,
                "scale": scale,
                "enable_expert_mapping": enable_expert_mapping,
            }

        def __call__(self, x, add_num, mapping_num=None, mapping_table=None, **_kwargs):
            return _torch_moe_fused_topk(
                x, add_num, mapping_num, mapping_table, **self.attrs
            )

    third_party = {"torch": TorchImpl}
    tolerance = {
        "float32": {"standard": "cross_check"},
        "int32": {"standard": "binary_equal"},
    }

    def customize_inputs(
        x,
        add_num,
        mapping_num=None,
        mapping_table=None,
        *,
        testcase_name=None,
        **_kwargs,
    ):
        _customize_inputs(x, add_num, testcase_name)
        return x, add_num, mapping_num, mapping_table


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
    inputs = [_numpy_to_torch(v) for v in (x, add_num, mapping_num, mapping_table)]
    return tuple(
        output.numpy()
        for output in _torch_moe_fused_topk(
            *inputs,
            group_num=group_num,
            group_topk=group_topk,
            top_n=top_n,
            top_k=top_k,
            activate_type=activate_type,
            is_norm=is_norm,
            scale=scale,
            enable_expert_mapping=enable_expert_mapping,
        )
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
    del y, indices
    return tuple(
        _torch_moe_fused_topk(
            x,
            addNum,
            mappingNum,
            mappingTable,
            group_num=groupNum,
            group_topk=groupTopk,
            top_n=topN,
            top_k=topK,
            activate_type=activateType,
            is_norm=isNorm,
            scale=scale,
            enable_expert_mapping=enableExpertMapping,
        )
    )
