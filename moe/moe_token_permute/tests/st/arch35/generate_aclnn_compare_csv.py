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
"""Generate ACLNN comparison cases from the direct-kernel case matrix."""

import argparse
import ast
import csv
from pathlib import Path


FIELDNAMES = (
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "tensor_storage_shapes",
    "tensor_view_offsets",
    "tensor_view_strides",
    "attributes",
    "output_tensor_indexes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
)


def parse(value):
    return ast.literal_eval(value)


def build_row(kernel_row, api_name):
    input_shapes = tuple(parse(kernel_row["input_shapes"]))
    output_shapes = tuple(parse(kernel_row["output_shapes"]))
    input_dtypes = tuple(parse(kernel_row["input_dtypes"]))
    output_dtypes = tuple(parse(kernel_row["output_dtypes"]))
    input_formats = tuple(parse(kernel_row["input_formats"]))
    output_formats = tuple(parse(kernel_row["output_formats"]))
    kernel_attrs = parse(kernel_row["attributes"])

    attributes = {
        "numOutTokens": kernel_attrs["num_out_tokens"],
        "paddedMode": kernel_attrs["padded_mode"],
    }
    tensor_shapes = input_shapes + output_shapes
    tensor_dtypes = input_dtypes + output_dtypes
    tensor_formats = input_formats + output_formats
    output_tensor_indexes = (2, 3)
    precision_tolerances = tuple(parse(kernel_row["precision_tolerances"]))

    if api_name == "aclnnMoeTokenPermuteV2":
        tensor_shapes += ((0,),)
        tensor_dtypes += ("float32",)
        tensor_formats += ("ND",)
        attributes["quantMode"] = -1
        output_tensor_indexes = (2, 3, 4)
        precision_tolerances += ((0, 0),)

    return {
        "testcase_name": kernel_row["testcase_name"],
        "api_name": api_name,
        "tensor_view_shapes": repr(tensor_shapes),
        "tensor_dtypes": repr(tensor_dtypes),
        "tensor_formats": repr(tensor_formats),
        "tensor_storage_shapes": "()",
        "tensor_view_offsets": "()",
        "tensor_view_strides": "()",
        "attributes": repr(attributes),
        "output_tensor_indexes": repr(output_tensor_indexes),
        "input_data_ranges": kernel_row["input_data_ranges"],
        "precision_tolerances": repr(precision_tolerances),
        "absolute_precision": kernel_row["absolute_precision"],
    }


def write_csv(source, destination, api_name):
    with source.open(newline="", encoding="utf-8") as src:
        rows = [build_row(row, api_name) for row in csv.DictReader(src)]
    if len(rows) != 100:
        raise ValueError(f"expected 100 kernel cases, found {len(rows)}")

    with destination.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("ttk_kernel_moe_token_permute.csv"),
    )
    args = parser.parse_args()
    output_dir = args.source.resolve().parent
    write_csv(
        args.source,
        output_dir / "ttk_aclnn_moe_token_permute_100.csv",
        "aclnnMoeTokenPermute",
    )
    write_csv(
        args.source,
        output_dir / "ttk_aclnn_moe_token_permute_v2_100.csv",
        "aclnnMoeTokenPermuteV2",
    )


if __name__ == "__main__":
    main()
