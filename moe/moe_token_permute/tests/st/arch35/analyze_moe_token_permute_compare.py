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
"""Merge direct-kernel and ACLNN accuracy/performance results."""

import argparse
import ast
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as src:
        return list(csv.DictReader(src))


def index_rows(rows):
    return {row["testcase_name"]: row for row in rows}


def latest_file(directory, pattern):
    files = list(directory.glob(pattern))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def execution_status(result_row):
    precision = result_row.get("precision", "")
    failure_markers = ("NO_OUTPUT", "FAILED", "CRASH", "NOT_FOUND")
    return "FAIL" if any(marker in precision for marker in failure_markers) else "PASS"


def profile_case(root, case_name, api_name, executed):
    empty = {
        "device_us": None,
        "main_op_us": None,
        "aux_op_us": None,
        "host_execute_us": None,
        "host_workspace_us": None,
        "sample_count": None,
        "ops": "",
    }
    if not executed:
        return empty

    case_dir = root / case_name
    api_file = latest_file(case_dir, "api_statistic_*.csv")
    op_file = latest_file(case_dir, "op_statistic_*.csv")
    if api_file is None:
        return empty

    api_rows = read_rows(api_file)
    api_row = next(
        (
            row
            for row in api_rows
            if row.get("Level") == "acl" and row.get("API Name") == api_name
        ),
        None,
    )
    workspace_row = next(
        (
            row
            for row in api_rows
            if row.get("Level") == "acl"
            and row.get("API Name") == f"{api_name}GetWorkspaceSize"
        ),
        None,
    )
    sample_count = int(float(api_row["Count"])) if api_row else 2
    host_execute_us = float(api_row["Avg(us)"]) if api_row else None
    host_workspace_us = float(workspace_row["Avg(us)"]) if workspace_row else None

    if op_file is None:
        return {
            **empty,
            "device_us": 0.0,
            "main_op_us": 0.0,
            "aux_op_us": 0.0,
            "host_execute_us": host_execute_us,
            "host_workspace_us": host_workspace_us,
            "sample_count": sample_count,
        }

    grouped = defaultdict(float)
    for row in read_rows(op_file):
        grouped[row["OP Type"]] += float(row["Total Time(us)"]) / sample_count
    main_op_us = sum(
        value for name, value in grouped.items() if name.startswith("MoeInitRouting")
    )
    device_us = sum(grouped.values())
    ops = ";".join(f"{name}={value:.3f}" for name, value in grouped.items())
    return {
        "device_us": device_us,
        "main_op_us": main_op_us,
        "aux_op_us": device_us - main_op_us,
        "host_execute_us": host_execute_us,
        "host_workspace_us": host_workspace_us,
        "sample_count": sample_count,
        "ops": ops,
    }


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def fmt_number(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def geometric_mean(values):
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def build_comparison(source_rows, kernel_rows, v1_rows, v2_rows, v1_root, v2_root):
    kernel_by_name = index_rows(kernel_rows)
    v1_by_name = index_rows(v1_rows)
    v2_by_name = index_rows(v2_rows)
    merged = []

    for source in source_rows:
        name = source["testcase_name"]
        kernel = kernel_by_name[name]
        v1 = v1_by_name[name]
        v2 = v2_by_name[name]
        input_shapes = ast.literal_eval(source["input_shapes"])
        input_dtypes = ast.literal_eval(source["input_dtypes"])
        attrs = ast.literal_eval(source["attributes"])
        indices_shape = input_shapes[1]
        total_indices = math.prod(indices_shape)
        v1_exec = execution_status(v1)
        v2_exec = execution_status(v2)
        v1_prof = profile_case(v1_root, name, "aclnnMoeTokenPermute", v1_exec == "PASS")
        v2_prof = profile_case(
            v2_root, name, "aclnnMoeTokenPermuteV2", v2_exec == "PASS"
        )
        kernel_us = float(kernel["dyn_perf_us"])

        row = {
            "testcase_name": name,
            "tokens_dtype": input_dtypes[0],
            "indices_dtype": input_dtypes[1],
            "tokens_shape": repr(input_shapes[0]),
            "indices_shape": repr(indices_shape),
            "total_indices": total_indices,
            "num_out_tokens": attrs["num_out_tokens"],
            "kernel_precision_status": kernel["precision_status"],
            "kernel_precision": kernel["dyn_precision"],
            "kernel_device_us": fmt_number(kernel_us),
            "kernel_perf_status": kernel["perf_status"],
            "kernel_memory_status": kernel["memory_oob_status"],
            "v1_precision_status": v1["precision_status"],
            "v1_precision": v1["precision"],
            "v1_execution_status": v1_exec,
            "v1_device_us": fmt_number(v1_prof["device_us"]),
            "v1_main_op_us": fmt_number(v1_prof["main_op_us"]),
            "v1_aux_op_us": fmt_number(v1_prof["aux_op_us"]),
            "v1_host_execute_us": fmt_number(v1_prof["host_execute_us"]),
            "v1_host_workspace_us": fmt_number(v1_prof["host_workspace_us"]),
            "v1_profile_samples": v1_prof["sample_count"] or "",
            "v1_device_ops": v1_prof["ops"],
            "v1_vs_kernel": fmt_number(ratio(v1_prof["device_us"], kernel_us)),
            "v2_precision_status": v2["precision_status"],
            "v2_precision": v2["precision"],
            "v2_execution_status": v2_exec,
            "v2_device_us": fmt_number(v2_prof["device_us"]),
            "v2_main_op_us": fmt_number(v2_prof["main_op_us"]),
            "v2_aux_op_us": fmt_number(v2_prof["aux_op_us"]),
            "v2_host_execute_us": fmt_number(v2_prof["host_execute_us"]),
            "v2_host_workspace_us": fmt_number(v2_prof["host_workspace_us"]),
            "v2_profile_samples": v2_prof["sample_count"] or "",
            "v2_device_ops": v2_prof["ops"],
            "v2_vs_kernel": fmt_number(ratio(v2_prof["device_us"], kernel_us)),
            "v2_vs_v1": fmt_number(ratio(v2_prof["device_us"], v1_prof["device_us"])),
        }
        merged.append(row)
    return merged


def numeric(row, field):
    return float(row[field]) if row[field] != "" else None


def path_summary(rows, prefix):
    if prefix == "kernel":
        precision_pass = sum(row["kernel_precision_status"] == "PASS" for row in rows)
        exec_fail = 0
        precision_fail = len(rows) - precision_pass
        perf_values = [numeric(row, "kernel_device_us") for row in rows]
        ratios = []
    else:
        precision_pass = sum(
            row[f"{prefix}_precision_status"] == "PASS" for row in rows
        )
        exec_fail = sum(row[f"{prefix}_execution_status"] == "FAIL" for row in rows)
        precision_fail = len(rows) - precision_pass - exec_fail
        perf_values = [numeric(row, f"{prefix}_device_us") for row in rows]
        ratios = [
            numeric(row, f"{prefix}_vs_kernel")
            for row in rows
            if row[f"{prefix}_precision_status"] == "PASS"
            and numeric(row, f"{prefix}_vs_kernel") is not None
        ]
    perf_values = [value for value in perf_values if value is not None]
    return {
        "precision_pass": precision_pass,
        "exec_fail": exec_fail,
        "precision_fail": precision_fail,
        "perf_samples": len(perf_values),
        "median_us": statistics.median(perf_values),
        "median_ratio": statistics.median(ratios) if ratios else None,
        "p90_ratio": percentile(ratios, 0.9),
        "geomean_ratio": geometric_mean(ratios),
        "wins": sum(value < 1 for value in ratios),
        "ratio_samples": len(ratios),
    }


def markdown_table(headers, body):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in body)
    return "\n".join(lines)


def render_markdown(rows, output_csv_name):
    kernel = path_summary(rows, "kernel")
    v1 = path_summary(rows, "v1")
    v2 = path_summary(rows, "v2")
    summary_body = []
    for label, item in (("kernel", kernel), ("aclnn V1", v1), ("aclnn V2", v2)):
        summary_body.append(
            (
                label,
                f"{item['precision_pass']}/100",
                item["exec_fail"],
                item["precision_fail"],
                item["perf_samples"],
                fmt_number(item["median_us"]),
                fmt_number(item["median_ratio"]),
                fmt_number(item["p90_ratio"]),
                f"{item['wins']}/{item['ratio_samples']}"
                if item["ratio_samples"]
                else "-",
            )
        )

    dtype_body = []
    for dtype in ("float16", "bfloat16", "float32", "int8"):
        selected = [row for row in rows if row["tokens_dtype"] == dtype]
        v1_ratios = [
            numeric(row, "v1_vs_kernel")
            for row in selected
            if row["v1_precision_status"] == "PASS"
            and numeric(row, "v1_vs_kernel") is not None
        ]
        v2_ratios = [
            numeric(row, "v2_vs_kernel")
            for row in selected
            if row["v2_precision_status"] == "PASS"
            and numeric(row, "v2_vs_kernel") is not None
        ]
        dtype_body.append(
            (
                dtype,
                len(selected),
                sum(row["v1_precision_status"] == "PASS" for row in selected),
                fmt_number(statistics.median(v1_ratios) if v1_ratios else None),
                sum(row["v2_precision_status"] == "PASS" for row in selected),
                fmt_number(statistics.median(v2_ratios) if v2_ratios else None),
            )
        )

    failures = []
    for row in rows:
        if row["v1_precision_status"] != "PASS" or row["v2_precision_status"] != "PASS":
            failures.append(
                (
                    row["testcase_name"],
                    "PASS"
                    if row["v1_precision_status"] == "PASS"
                    else (
                        "EXEC_FAIL"
                        if row["v1_execution_status"] == "FAIL"
                        else row["v1_precision"]
                    ),
                    "PASS"
                    if row["v2_precision_status"] == "PASS"
                    else (
                        "EXEC_FAIL"
                        if row["v2_execution_status"] == "FAIL"
                        else row["v2_precision"]
                    ),
                )
            )

    perf_rows = [row for row in rows if row["testcase_name"].startswith("perf_")]
    perf_body = [
        (
            row["testcase_name"],
            row["kernel_device_us"],
            row["v1_device_us"] or "-",
            row["v1_vs_kernel"] or "-",
            row["v2_device_us"] or "-",
            row["v2_vs_kernel"] or "-",
        )
        for row in perf_rows
    ]

    v1_routes = Counter()
    v2_routes = Counter()
    for row in rows:
        for name in row["v1_device_ops"].split(";"):
            if name:
                v1_routes[name.split("=")[0]] += 1
        for name in row["v2_device_ops"].split(";"):
            if name:
                v2_routes[name.split("=")[0]] += 1

    sections = [
        "# MoeTokenPermute 100-case comparison",
        "",
        "All three paths use the same 100 logical cases, input ranges, seed 42, and run=3 on "
        "Ascend950DT_9584. Kernel time is TTK `dyn_perf_us`. ACLNN device time is the sum of every "
        "profiled device OP's total time divided by the two profiled API calls; the first call is excluded "
        "by TTK as warmup. Host API and workspace times are retained in the detailed CSV but are not used "
        "for kernel ratios.",
        "",
        "## Overall",
        "",
        markdown_table(
            (
                "path",
                "precision",
                "exec fail",
                "precision fail",
                "perf samples",
                "median us",
                "median vs kernel",
                "p90 vs kernel",
                "faster than kernel",
            ),
            summary_body,
        ),
        "",
        "Ratios use only precision-PASS cases. A ratio above 1 means the ACLNN device path is slower than "
        "the direct kernel. Aggregates span heterogeneous shapes, so the per-case CSV and performance matrix "
        "are the authoritative comparison.",
        "",
        "## By token dtype",
        "",
        markdown_table(
            (
                "dtype",
                "cases",
                "V1 pass",
                "V1 median ratio",
                "V2 pass",
                "V2 median ratio",
            ),
            dtype_body,
        ),
        "",
        "## ACLNN device routes",
        "",
        f"V1 profiled OP presence: `{dict(v1_routes)}`.",
        "",
        f"V2 profiled OP presence: `{dict(v2_routes)}`.",
        "",
        "## Accuracy failures",
        "",
        markdown_table(("case", "V1", "V2"), failures),
        "",
        "## Performance-focused cases",
        "",
        markdown_table(
            ("case", "kernel us", "V1 us", "V1/kernel", "V2 us", "V2/kernel"), perf_body
        ),
        "",
        "## Detailed data",
        "",
        f"See `{output_csv_name}` for every case, exact precision strings, execution status, device OP "
        "breakdown, host API time, workspace-query time, and all ratios.",
        "",
    ]
    return "\n".join(sections)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=here / "ttk_kernel_moe_token_permute.csv"
    )
    parser.add_argument("--kernel-result", type=Path, required=True)
    parser.add_argument("--v1-result", type=Path, required=True)
    parser.add_argument("--v2-result", type=Path, required=True)
    parser.add_argument("--v1-msprof", type=Path, required=True)
    parser.add_argument("--v2-msprof", type=Path, required=True)
    parser.add_argument(
        "--output-csv", type=Path, default=here / "moe_token_permute_100_comparison.csv"
    )
    parser.add_argument(
        "--output-md", type=Path, default=here / "moe_token_permute_100_comparison.md"
    )
    args = parser.parse_args()

    source_rows = read_rows(args.source)
    kernel_rows = read_rows(args.kernel_result)
    v1_rows = read_rows(args.v1_result)
    v2_rows = read_rows(args.v2_result)
    for label, rows in (
        ("source", source_rows),
        ("kernel", kernel_rows),
        ("V1", v1_rows),
        ("V2", v2_rows),
    ):
        if len(rows) != 100:
            raise ValueError(
                f"{label} result must contain 100 cases, found {len(rows)}"
            )

    merged = build_comparison(
        source_rows,
        kernel_rows,
        v1_rows,
        v2_rows,
        args.v1_msprof,
        args.v2_msprof,
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=merged[0].keys())
        writer.writeheader()
        writer.writerows(merged)
    args.output_md.write_text(
        render_markdown(merged, args.output_csv.name), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
