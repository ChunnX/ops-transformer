#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it
# under the terms and conditions of CANN Open Software License Agreement
# Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF
# ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of
# the License.
# ----------------------------------------------------------------------------

import csv
import os
import sys

OP_NAME = "mhc_pre_sinkhorn_backward"
N = 4
SK_ITER_COUNT = 20
HC_MIX = 2 * N + N * N  # 24

KERNEL_CSV_HEADER = [
    "testcase_name", "network_name", "op_name", "input_shapes", "input_dtypes",
    "input_formats", "output_shapes", "output_dtypes", "output_formats",
    "input_ori_shapes", "input_ori_formats", "output_ori_shapes",
    "output_ori_formats", "attributes", "input_data_ranges",
    "precision_tolerances", "absolute_precision", "output_inplace_indexes",
    "output_shape_unknown_indexes", "is_enabled", "remark", "soc_series",
    "priority", "dump_file_prefix", "manual_input_binaries",
    "manual_golden_binaries",
]

# B, S, C configurations (C must be divisible by 128, >= 128, < 100000)
BS_C_CONFIGS = [
    # small shapes for basic validation
    {"B": 1, "S": 1, "C": 128},
    {"B": 1, "S": 1, "C": 256},
    {"B": 1, "S": 2, "C": 256},
    {"B": 2, "S": 1, "C": 128},
    {"B": 2, "S": 1, "C": 256},
    {"B": 2, "S": 1, "C": 512},
    # medium shapes
    {"B": 1, "S": 128, "C": 256},
    {"B": 2, "S": 64, "C": 256},
    {"B": 2, "S": 128, "C": 256},
    {"B": 4, "S": 64, "C": 512},
    {"B": 4, "S": 128, "C": 256},
    {"B": 8, "S": 32, "C": 256},
    # large shapes
    {"B": 8, "S": 128, "C": 256},
    {"B": 8, "S": 128, "C": 512},
    {"B": 16, "S": 64, "C": 256},
]

HC_EPS_VALUES = [1e-6, 1e-5, 1e-4]

INPUT_DTYPES_OPTIONS = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
]


def build_input_shapes(B, S, C):
    gradHin = (B, S, C)
    gradHPost = (B, S, N)
    gradHRes = (B, S, N, N)
    x = (B, S, N, C)
    phi = (HC_MIX, N * C)
    alpha = (3,)
    bias = (HC_MIX,)
    hPre = (B, S, N)
    hcBeforeNorm = (B, S, HC_MIX)
    invRms = (B, S, 1)
    sumOut = (2 * SK_ITER_COUNT, B, S, N)
    normOut = (2 * SK_ITER_COUNT, B, S, N, N)
    return (
        gradHin, gradHPost, gradHRes, x, phi, alpha, bias,
        hPre, hcBeforeNorm, invRms, sumOut, normOut,
    )


def build_output_shapes(B, S, C):
    gradX = (B, S, N, C)
    gradPhi = (HC_MIX, N * C)
    gradAlpha = (3,)
    gradBias = (HC_MIX,)
    return (gradX, gradPhi, gradAlpha, gradBias)


def build_input_dtypes(x_dtype, other_dtype):
    return (
        x_dtype, other_dtype, other_dtype, x_dtype, other_dtype,
        other_dtype, other_dtype, other_dtype, other_dtype,
        other_dtype, other_dtype, other_dtype,
    )


def build_output_dtypes(x_dtype, other_dtype):
    return (x_dtype, other_dtype, other_dtype, other_dtype)


def build_input_data_ranges():
    return (
        (-1, 1), (-1, 1), (-0.1, 0.1), (-1, 1), (-0.01, 0.01),
        (-1, 1), (-0.1, 0.1), (0, 1), (-10, 10), (0, 2),
        (0, 1), (0, 1),
    )


def fmt_shapes(shapes_tuple):
    inner = ", ".join(f"({', '.join(str(d) for d in s)})" for s in shapes_tuple)
    return f"({inner},)"


def fmt_dtypes(dtypes_tuple):
    inner = ", ".join(f"'{d}'" for d in dtypes_tuple)
    return f"({inner},)"


def fmt_formats(count):
    return "('ND',)"


def fmt_attrs(hc_eps):
    return f"{{'hc_eps': {hc_eps}}}"


def fmt_ranges(ranges_tuple):
    inner = ", ".join(
        f"({lo}, {hi})" if lo is not None and hi is not None
        else f"({lo}, {hi})"
        for lo, hi in ranges_tuple
    )
    return f"({inner},)"


def fmt_precision():
    return "((0.001, 0.001),)"


def build_row(B, S, C, x_dtype, hc_eps, idx):
    input_shapes = build_input_shapes(B, S, C)
    output_shapes = build_output_shapes(B, S, C)
    input_dtypes = build_input_dtypes(x_dtype, "float32")
    output_dtypes = build_output_dtypes(x_dtype, "float32")
    input_ranges = build_input_data_ranges()

    short_x_dtype = "bf16" if x_dtype == "bfloat16" else "fp16"
    tc_name = f"mhc_pre_sinkhorn_backward_{short_x_dtype}_B{B}_S{S}_C{C}_eps{hc_eps}_{idx}"

    row = [
        tc_name,
        "UNKNOWN",
        OP_NAME,
        fmt_shapes(input_shapes),
        fmt_dtypes(input_dtypes),
        fmt_formats(len(input_shapes)),
        fmt_shapes(output_shapes),
        fmt_dtypes(output_dtypes),
        fmt_formats(len(output_shapes)),
        fmt_shapes(input_shapes),
        fmt_formats(len(input_shapes)),
        fmt_shapes(output_shapes),
        fmt_formats(len(output_shapes)),
        fmt_attrs(hc_eps),
        fmt_ranges(input_ranges),
        fmt_precision(),
        1e-03,
        (),
        (),
        True,
        "",
        "",
        0,
        "",
        (),
        (),
    ]
    return row


def generate_kernel_csv(output_path):
    rows = []
    idx = 0
    for config in BS_C_CONFIGS:
        B, S, C = config["B"], config["S"], config["C"]
        for x_dtype_pair in INPUT_DTYPES_OPTIONS:
            x_dtype = x_dtype_pair[0]
            for hc_eps in HC_EPS_VALUES:
                idx += 1
                row = build_row(B, S, C, x_dtype, hc_eps, idx)
                rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(KERNEL_CSV_HEADER)
        for row in rows:
            # fields with nested tuples/dicts need double-quoting in CSV
            quoted_row = []
            for i, val in enumerate(row):
                if i in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
                    quoted_row.append(str(val))
                else:
                    quoted_row.append(val)
            writer.writerow(quoted_row)

    print(f"Generated kernel mode CSV: {output_path}")
    print(f"Total test cases: {len(rows)}")


ACLNN_CSV_HEADER = [
    "testcase_name", "network_name", "api_name",
    "tensor_view_shapes", "tensor_dtypes", "tensor_formats",
    "tensor_storage_shapes", "tensor_view_offsets", "tensor_view_strides",
    "output_tensor_indexes", "output_inplace_indexes", "attributes",
    "scalar_dtypes", "input_data_ranges", "precision_tolerances",
    "absolute_precision", "scalar_data_ranges", "is_enabled", "remark",
    "soc_series", "priority", "dump_file_prefix",
    "manual_tensor_binaries", "manual_golden_binaries",
]


def build_aclnn_row(B, S, C, x_dtype, hc_eps, idx):
    input_shapes = build_input_shapes(B, S, C)
    output_shapes = build_output_shapes(B, S, C)

    all_shapes = list(input_shapes) + list(output_shapes)
    all_dtypes = list(build_input_dtypes(x_dtype, "float32")) + list(build_output_dtypes(x_dtype, "float32"))

    output_start = len(input_shapes)
    output_indexes = tuple(range(output_start, output_start + len(output_shapes)))

    short_x_dtype = "bf16" if x_dtype == "bfloat16" else "fp16"
    tc_name = f"aclnnMhcPreSinkhornBackward_{short_x_dtype}_B{B}_S{S}_C{C}_eps{hc_eps}_{idx}"

    input_ranges = build_input_data_ranges()
    all_ranges = list(input_ranges) + [(-1, 1), (-0.01, 0.01), (-1, 1), (-0.1, 0.1)]

    row = [
        tc_name,
        "UNKNOWN",
        "aclnnMhcPreSinkhornBackward",
        fmt_shapes(tuple(all_shapes)),
        fmt_dtypes(tuple(all_dtypes)),
        fmt_formats(len(all_shapes)),
        "",
        "",
        "",
        str(output_indexes),
        (),
        fmt_attrs(hc_eps),
        "('float64',)",
        fmt_ranges(tuple(all_ranges)),
        fmt_precision(),
        1e-03,
        "((None, None),)",
        True,
        "",
        "",
        0,
        "",
        (),
        (),
    ]
    return row


def generate_aclnn_csv(output_path):
    rows = []
    idx = 0
    for config in BS_C_CONFIGS:
        B, S, C = config["B"], config["S"], config["C"]
        for x_dtype_pair in INPUT_DTYPES_OPTIONS:
            x_dtype = x_dtype_pair[0]
            for hc_eps in HC_EPS_VALUES:
                idx += 1
                row = build_aclnn_row(B, S, C, x_dtype, hc_eps, idx)
                rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(ACLNN_CSV_HEADER)
        for row in rows:
            quoted_row = []
            for i, val in enumerate(row):
                if i in (3, 4, 5, 11, 12, 13, 14, 16):
                    quoted_row.append(str(val))
                else:
                    quoted_row.append(val)
            writer.writerow(quoted_row)

    print(f"Generated aclnn mode CSV: {output_path}")
    print(f"Total test cases: {len(rows)}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    kernel_dir = os.path.join(script_dir, "kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    generate_kernel_csv(os.path.join(kernel_dir, "mhc_pre_sinkhorn_backward.csv"))

    aclnn_dir = os.path.join(script_dir, "aclnn")
    os.makedirs(aclnn_dir, exist_ok=True)
    generate_aclnn_csv(os.path.join(aclnn_dir, "aclnnMhcPreSinkhornBackward.csv"))

    print("\nDone! CSV files generated successfully.")


if __name__ == "__main__":
    main()