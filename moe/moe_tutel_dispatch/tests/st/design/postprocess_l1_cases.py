#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

"""Build 500 distinct, constraint-valid Kernel/GEIR L1 rows.

The generic pairwise selector sees x/gates/y dtype as three independent factors, although
the solver correctly ties them. Its theoretical pair count therefore contains impossible
combinations and its target-count padding repeats three rows. This deterministic post-step
keeps the generated constraint-valid pool, removes duplicates and reports contract-specific
E/K, empty-axis and H-alignment coverage.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from collections import Counter
from pathlib import Path

import yaml


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


def parse_cell(value: str):
    return ast.literal_eval(value)


def fmt_number(value):
    if isinstance(value, str) and value in {"inf", "-inf", "nan", "+0", "-0"}:
        return value
    return str(value)


def fmt_tuple(items):
    if not items:
        return "()"
    values = []
    for item in items:
        if isinstance(item, (list, tuple)):
            values.append(fmt_tuple(item))
        else:
            values.append(fmt_number(item))
    return f"({','.join(values)},)"


def fmt_dtype_tuple(items):
    return f"({','.join(repr(item) for item in items)},)"


def relation(row):
    expert_e = int(row["_shape_symbols.E"])
    route_k = int(row["_shape_symbols.K"])
    if expert_e < route_k:
        return "E<K"
    if expert_e > route_k:
        return "E>K"
    return "E=K"


def empty_class(row):
    for name in ("E", "K", "S", "H"):
        if int(row[f"_shape_symbols.{name}"]) == 0:
            return f"{name}=0"
    if int(row["capacity.value"]) == 0:
        return "C=0"
    return "nonempty"


def hidden_class(row):
    hidden = int(row["_shape_symbols.H"])
    if hidden == 0:
        return "zero"
    if hidden == 1:
        return "one"
    if hidden % 32 == 0:
        return "aligned32"
    return "unaligned32"


def signature(row):
    keys = (
        "x.shape",
        "gates.shape",
        "y.shape",
        "x.dtype",
        "x.value_range",
        "gates.value_range",
        "indices.value_range",
        "locations.value_range",
        "capacity.value",
    )
    return tuple(row[key] for key in keys)


def select_rows(rows, target, seed):
    unique = {}
    for row in rows:
        unique.setdefault(signature(row), row)
    pool = list(unique.values())
    random.Random(seed).shuffle(pool)

    selected = []
    selected_ids = set()
    # Seed every reachable contract stratum before deterministic random fill.
    strata = {}
    for row in pool:
        key = (row["x.dtype"], relation(row), empty_class(row), hidden_class(row))
        strata.setdefault(key, row)
    for row in strata.values():
        sig = signature(row)
        if sig not in selected_ids:
            selected.append(row)
            selected_ids.add(sig)
            if len(selected) == target:
                return selected
    for row in pool:
        sig = signature(row)
        if sig in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(sig)
        if len(selected) == target:
            break
    if len(selected) < target:
        raise RuntimeError(
            f"only {len(selected)} distinct rows available, need {target}"
        )
    return selected


def to_ttk_case(row, index):
    input_shapes = [
        parse_cell(row[name])
        for name in ("x.shape", "gates.shape", "indices.shape", "locations.shape")
    ]
    output_shapes = [parse_cell(row["y.shape"])]
    input_dtypes = [
        row[name]
        for name in ("x.dtype", "gates.dtype", "indices.dtype", "locations.dtype")
    ]
    output_dtypes = [row["y.dtype"]]
    ranges = [
        parse_cell(row[name])
        for name in (
            "x.value_range",
            "gates.value_range",
            "indices.value_range",
            "locations.value_range",
        )
    ]
    return {
        "testcase_name": f"MoeTutelDispatch_L1_{index:03d}",
        "op_name": "moe_tutel_dispatch",
        "input_shapes": fmt_tuple(input_shapes),
        "input_dtypes": fmt_dtype_tuple(input_dtypes),
        "input_formats": fmt_dtype_tuple(["ND"] * 4),
        "input_ori_shapes": fmt_tuple(input_shapes),
        "input_ori_formats": fmt_dtype_tuple(["ND"] * 4),
        "output_shapes": fmt_tuple(output_shapes),
        "output_dtypes": fmt_dtype_tuple(output_dtypes),
        "output_formats": fmt_dtype_tuple(["ND"]),
        "output_ori_shapes": fmt_tuple(output_shapes),
        "output_ori_formats": fmt_dtype_tuple(["ND"]),
        "attributes": repr({"capacity": int(row["capacity.value"])}),
        "input_data_ranges": fmt_tuple(ranges),
        "precision_tolerances": "((0.0,0.0),)",
        "absolute_precision": "0",
        "soc_series": "",
        "priority": "1",
        "remark": "constraint-valid factor-pool case; binary compare is mandatory",
    }


def build_report(selected):
    return {
        "operator": "MoeTutelDispatch",
        "level": "L1",
        "case_count": len(selected),
        "unique_case_count": len({signature(row) for row in selected}),
        "generator": "ascendc-st-design factor solver + contract-aware deterministic postprocess",
        "seed": 42,
        "coverage": {
            "dtype": dict(sorted(Counter(row["x.dtype"] for row in selected).items())),
            "e_k_relation": dict(
                sorted(Counter(relation(row) for row in selected).items())
            ),
            "empty_axis": dict(
                sorted(Counter(empty_class(row) for row in selected).items())
            ),
            "hidden_alignment": dict(
                sorted(Counter(hidden_class(row) for row in selected).items())
            ),
        },
        "notes": [
            "x/gates/y dtype are one constrained combination, not three independent pairwise factors",
            "bitwise, long-H SPLIT_H, guarded duplicate routes and P0 cases are mandatory curated rows outside this random L1 pool",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("factor_values")
    parser.add_argument("case_output")
    parser.add_argument("report_output")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.factor_values, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    selected = select_rows(rows, args.target, args.seed)

    case_path = Path(args.case_output)
    case_path.parent.mkdir(parents=True, exist_ok=True)
    with case_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TTK_COLUMNS)
        writer.writeheader()
        for index, row in enumerate(selected, 1):
            writer.writerow(to_ttk_case(row, index))

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            build_report(selected), file, allow_unicode=True, sort_keys=False
        )


if __name__ == "__main__":
    main()
