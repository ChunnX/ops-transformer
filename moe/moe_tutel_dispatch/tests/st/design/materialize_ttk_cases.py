#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Materialize executable Kernel/GEIR CSVs and the non-TTK harness manifest.

The ascendc-st-design generator emits an API-shaped table because its required
``aclnn_name`` field is also used for GEIR-only operators.  MoeTutelDispatch has
no ACLNN/E2E API, so the generated table is retained as a design source and is
deterministically converted to TestcaseOp/GeirTestcase schema here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


TTK_COLUMNS = [
    "testcase_name",
    "op_name",
    "input_shapes",
    "input_dtypes",
    "input_formats",
    "input_ori_shapes",
    "input_ori_formats",
    "output_shapes",
    "output_dtypes",
    "output_formats",
    "output_ori_shapes",
    "output_ori_formats",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "soc_series",
    "priority",
    "remark",
]

GEIR_COLUMNS = TTK_COLUMNS + ["dyn_input_shapes"]

MANIFEST_COLUMNS = [
    "case_id",
    "source",
    "layer",
    "harness",
    "target",
    "expected",
    "repeats",
    "parallelism",
    "dynamic",
    "notes",
]

P0_GEIR_CASES = {"MTD_REQ_031", "MTD_REQ_032", "MTD_REQ_033"}
CONCURRENT_CASES = {"MTD_REQ_029", "MTD_REQ_035"}


def fmt_tuple(items):
    def one(value):
        if isinstance(value, (tuple, list)):
            return fmt_tuple(value)
        if isinstance(value, float):
            if value != value:
                return "nan"
            if value == float("inf"):
                return "inf"
            if value == float("-inf"):
                return "-inf"
        return str(value)

    return f"({','.join(one(item) for item in items)},)"


def fmt_strings(items):
    return f"({','.join(repr(item) for item in items)},)"


def parse_ranges(value):
    return eval(  # noqa: S307 - frozen generator asset, no user-provided expression
        value,
        {"__builtins__": {}},
        {"nan": float("nan"), "inf": float("inf")},
    )


def write_rows(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def kernel_case(
    name,
    input_shapes,
    dtype,
    output_shape,
    capacity,
    ranges,
    remark,
    input_ori_shapes=None,
    output_ori_shape=None,
):
    input_dtypes = [dtype, dtype, "int32", "int32"]
    input_ori_shapes = input_shapes if input_ori_shapes is None else input_ori_shapes
    output_ori_shape = output_shape if output_ori_shape is None else output_ori_shape
    return {
        "testcase_name": name,
        "op_name": "moe_tutel_dispatch",
        "input_shapes": fmt_tuple(input_shapes),
        "input_dtypes": fmt_strings(input_dtypes),
        "input_formats": fmt_strings(["ND"] * 4),
        "input_ori_shapes": fmt_tuple(input_ori_shapes),
        "input_ori_formats": fmt_strings(["ND"] * 4),
        "output_shapes": fmt_tuple([output_shape]),
        "output_dtypes": fmt_strings([dtype]),
        "output_formats": fmt_strings(["ND"]),
        "output_ori_shapes": fmt_tuple([output_ori_shape]),
        "output_ori_formats": fmt_strings(["ND"]),
        "attributes": repr({"capacity": capacity}),
        "input_data_ranges": fmt_tuple(ranges),
        "precision_tolerances": "((0.0,0.0),)",
        "absolute_precision": "0",
        "soc_series": "",
        "priority": "1",
        "remark": remark,
    }


def convert_l0(source):
    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    result = []
    for row in rows:
        shapes = ast.literal_eval(row["tensor_view_shapes"])
        dtypes = ast.literal_eval(row["tensor_dtypes"])
        attrs = ast.literal_eval(row["attributes"])
        ranges = parse_ranges(row["input_data_ranges"])
        result.append(
            kernel_case(
                row["testcase_name"],
                list(shapes[:4]),
                dtypes[0],
                shapes[4],
                int(attrs["capacity"]),
                list(ranges[:4]),
                "L0 single-factor source converted to Kernel/GEIR schema; binary compare mandatory",
            )
        )
    if len(result) != 15:
        raise RuntimeError(f"expected 15 L0 rows, got {len(result)}")
    return result


def as_int(row, key):
    value = row[key]
    try:
        return int(value)
    except ValueError:
        return None


def required_ranges(row, expert_e, capacity):
    x_range = (-1, 1)
    gates_range = (-1, 1)
    idx_range = (0, max(expert_e - 1, 0))
    loc_range = (0, max(capacity - 1, 0))
    route = row["route_pattern"]
    gate = row["gate_pattern"]
    if route == "idx_negative":
        idx_range = (-1, -1)
    elif route == "idx_equal_E":
        idx_range = (expert_e, expert_e)
    elif route == "loc_negative":
        loc_range = (-1, -1)
    elif route == "loc_equal_capacity":
        loc_range = (capacity, capacity)
    if gate == "all_one":
        gates_range = (1, 1)
    elif gate == "all_positive_zero":
        gates_range = (0.0, 0.0)
    elif gate == "all_negative_zero":
        gates_range = (-0.0, -0.0)
    return [x_range, gates_range, idx_range, loc_range]


def required_ttk_rows(source):
    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    kernel_rows = []
    geir_rows = []
    manifest = []
    for row in rows:
        dims = {name: as_int(row, name) for name in ("S", "H", "K", "E", "capacity")}
        layer = row["layer"]
        is_dynamic_p0 = row["case_id"] == "MTD_REQ_033"
        if is_dynamic_p0:
            # Storage and ori tensors stay concrete so TestcaseOp can validate
            # and generate data.  GEIR's dedicated dyn_input_shapes field keeps
            # K known while deferring S/H in the graph descriptor.
            dims.update({"S": 4, "H": 8, "K": 3, "E": 2, "capacity": 5})
        runnable = (
            layer in {"kernel", "GEIR"} or row["case_id"] in P0_GEIR_CASES
        ) and all(value is not None for value in dims.values())
        if runnable:
            shapes = [
                (dims["S"], dims["H"]),
                (dims["K"], dims["S"]),
                (dims["K"], dims["S"]),
                (dims["K"], dims["S"]),
            ]
            output_shape = (dims["E"], dims["capacity"], dims["H"])
            case = kernel_case(
                row["case_id"],
                shapes,
                row["dtype"],
                output_shape,
                dims["capacity"],
                required_ranges(row, dims["E"], dims["capacity"]),
                "; ".join(
                    [
                        f"route={row['route_pattern']}",
                        f"gate={row['gate_pattern']}",
                        f"x={row['x_pattern']}",
                        f"assert={row['assertion']}",
                    ]
                ),
            )
            if is_dynamic_p0:
                case["dyn_input_shapes"] = fmt_tuple(
                    [(-1, -1), (dims["K"], -1), (dims["K"], -1), (dims["K"], -1)]
                )
            is_geir = layer == "GEIR" or row["case_id"] in P0_GEIR_CASES
            output = geir_rows if is_geir else kernel_rows
            output.append(case)
            csv_name = (
                "MoeTutelDispatch_required_geir.csv"
                if is_geir
                else "MoeTutelDispatch_required_kernel.csv"
            )
            harness = "ttk_geir" if is_geir else "ttk_kernel"
            manifest.append(
                {
                    "case_id": row["case_id"],
                    "source": source.name,
                    "layer": layer,
                    "harness": harness,
                    "target": f"testcases/{csv_name}#{row['case_id']}",
                    "expected": row["assertion"],
                    "repeats": row["repeats"],
                    "parallelism": "2" if row["case_id"] in CONCURRENT_CASES else "1",
                    "dynamic": "1" if is_dynamic_p0 else "0",
                    "notes": row["notes"],
                }
            )
            continue

        harness, target = non_ttk_required_target(row)
        manifest.append(
            {
                "case_id": row["case_id"],
                "source": source.name,
                "layer": layer,
                "harness": harness,
                "target": target,
                "expected": row["assertion"],
                "repeats": row["repeats"],
                "parallelism": "1",
                "dynamic": "0",
                "notes": row["notes"],
            }
        )
    return kernel_rows, geir_rows, manifest


def pascal(value):
    return "".join(
        part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", value) if part
    )


def non_ttk_required_target(row):
    layer = row["layer"]
    case_id = row["case_id"]
    if layer == "build":
        return "build_scan", f"build_isolation:{case_id}"
    if layer == "contract":
        return "static_contract", f"spec_readme_alias:{case_id}"
    if layer == "test-infra":
        return "test_infra_ut", "tests/ut/test_output_shape_spec_kwargs.py"
    raise RuntimeError(f"required case {case_id} has no harness for layer {layer}")


def l2_manifest(source):
    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    result = []
    for row in rows:
        if row["layer"] == "Host-UT" or row["mutation"] in {
            "missing_caller_y_desc",
            "dynamic_rank",
        }:
            harness = "host_ut"
            target = f"MoeTutelDispatchTiling.{pascal(row['mutation'])}"
        else:
            raise RuntimeError(f"unknown L2 layer: {row['layer']}")
        result.append(
            {
                "case_id": row["case_id"],
                "source": source.name,
                "layer": row["layer"],
                "harness": harness,
                "target": target,
                "expected": row["expected_error"],
                "repeats": "1",
                "parallelism": "1",
                "dynamic": "0",
                "notes": row["notes"],
            }
        )
    return result


def validate_manifest(rows):
    allowed = {
        "ttk_kernel",
        "ttk_geir",
        "host_ut",
        "build_scan",
        "static_contract",
        "test_infra_ut",
    }
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case_id in harness manifest")
    bad = [
        row
        for row in rows
        if row["harness"] not in allowed
        or not row["target"]
        or int(row["repeats"]) < 1
        or int(row["parallelism"]) < 1
        or row["dynamic"] not in {"0", "1"}
    ]
    if bad:
        raise RuntimeError(f"unroutable harness rows: {bad}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate sources/outputs after materialization",
    )
    args = parser.parse_args()

    design_dir = Path(__file__).resolve().parent
    st_dir = design_dir.parent
    case_dir = st_dir / "testcases"
    l0_source = design_dir / "MoeTutelDispatch_L0_design_source.csv"
    required_source = case_dir / "MoeTutelDispatch_required_test_cases.csv"
    l2_source = case_dir / "MoeTutelDispatch_L2_test_cases.csv"

    l0_rows = convert_l0(l0_source)
    kernel_rows, geir_rows, manifest = required_ttk_rows(required_source)
    manifest.extend(l2_manifest(l2_source))
    validate_manifest(manifest)

    write_rows(case_dir / "MoeTutelDispatch_L0_test_cases.csv", TTK_COLUMNS, l0_rows)
    write_rows(
        case_dir / "MoeTutelDispatch_required_kernel.csv", TTK_COLUMNS, kernel_rows
    )
    write_rows(case_dir / "MoeTutelDispatch_required_geir.csv", GEIR_COLUMNS, geir_rows)
    write_rows(
        case_dir / "MoeTutelDispatch_contract_harness_manifest.csv",
        MANIFEST_COLUMNS,
        manifest,
    )

    if args.check:
        print(
            f"PASS: L0={len(l0_rows)}, required_kernel={len(kernel_rows)}, "
            f"required_geir={len(geir_rows)}, manifest={len(manifest)}"
        )


if __name__ == "__main__":
    main()
