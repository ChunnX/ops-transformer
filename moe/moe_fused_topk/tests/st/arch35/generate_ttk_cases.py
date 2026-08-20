#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Convert the legacy MoeFusedTopk ATK matrix into static TTK kernel cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path


DTYPES = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}
ATTR_NAMES = (
    "group_num",
    "group_topk",
    "top_n",
    "top_k",
    "activate_type",
    "is_norm",
    "scale",
    "enable_expert_mapping",
)
HEADER = (
    "testcase_name",
    "network_name",
    "op_name",
    "input_shapes",
    "input_dtypes",
    "input_formats",
    "output_shapes",
    "output_dtypes",
    "output_formats",
    "input_ori_shapes",
    "input_ori_formats",
    "output_ori_shapes",
    "output_ori_formats",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "output_inplace_indexes",
    "output_shape_unknown_indexes",
    "is_enabled",
    "remark",
    "soc_series",
    "priority",
    "dump_file_prefix",
    "manual_input_binaries",
    "manual_golden_binaries",
)


def _select(value, case_id, *, integer=False):
    if not isinstance(value, list):
        return value
    if len(value) == 1:
        return value[0]
    low, high = value
    selector = case_id % 3
    if selector == 0:
        return low
    if selector == 1:
        return high
    midpoint = (low + high) / 2
    return int(midpoint) if integer else midpoint


def _finite_range(value):
    if (
        isinstance(value, list)
        and len(value) == 1
        and value[0] in {"nan", "inf", "-inf"}
    ):
        return (-1, 1)
    if isinstance(value, list) and len(value) == 2:
        return tuple(value)
    return (value, value)


def _case_row(case):
    items = {item["name"]: item for item in case["inputs"]}
    case_id = int(case["id"])
    attrs = {}
    for name in ATTR_NAMES:
        item = items[name]
        attrs[name] = _select(
            item["range_values"],
            case_id,
            integer=item["dtype"] == "int",
        )

    enabled_mapping = bool(attrs["enable_expert_mapping"])
    tensor_names = ("x", "add_num", "mapping_num", "mapping_table")
    shapes = []
    dtypes = []
    ranges = []
    for name in tensor_names:
        item = items[name]
        optional_absent = name.startswith("mapping_") and not enabled_mapping
        shapes.append(None if optional_absent else tuple(item["shape"]))
        dtypes.append(DTYPES.get(item["dtype"], item["dtype"]))
        ranges.append(
            (None, None) if optional_absent else _finite_range(item["range_values"])
        )

    token_num, _ = items["x"]["shape"]
    output_shape = (token_num, int(attrs["top_k"]))
    testcase_name = f"moe_fused_topk_atk_{case_id:03d}"
    return {
        "testcase_name": testcase_name,
        "network_name": "",
        "op_name": "moe_fused_topk",
        "input_shapes": repr(tuple(shapes)),
        "input_dtypes": repr(tuple(dtypes)),
        "input_formats": repr(("ND", "ND", "ND", "ND")),
        "output_shapes": repr((output_shape, output_shape)),
        "output_dtypes": repr(("float32", "int32")),
        "output_formats": repr(("ND", "ND")),
        "input_ori_shapes": "",
        "input_ori_formats": "",
        "output_ori_shapes": "",
        "output_ori_formats": "",
        "attributes": repr(attrs),
        "input_data_ranges": repr(tuple(ranges)),
        "precision_tolerances": "",
        "absolute_precision": "1e-8",
        "output_inplace_indexes": repr(()),
        "output_shape_unknown_indexes": repr(()),
        "is_enabled": "True",
        "remark": "legacy ATK case generalized to static TTK; low/high/mid attrs",
        "soc_series": "",
        "priority": "0",
        "dump_file_prefix": "",
        "manual_input_binaries": repr(()),
        "manual_golden_binaries": repr(()),
    }


def main():
    directory = Path(__file__).resolve().parent
    source = directory.parent / "aclnnMoeFusedTopk" / "atk_aclnnMoeFusedTopk.json"
    target = directory / "ttk_kernel_moe_fused_topk.csv"
    cases = json.loads(source.read_text(encoding="utf-8"))
    with target.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_case_row(case) for case in cases)
    print(f"generated {len(cases)} cases: {target}")


if __name__ == "__main__":
    main()
