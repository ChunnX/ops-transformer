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

"""Convert the legacy QBSA MXFP8 pytest parameter sets to a TTK E2E CSV."""

import argparse
import csv
import io
import json
import math
import runpy
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
LEGACY_DIR = CURRENT_DIR.parents[1] / "pytest" / "bsa_fullquant_mxfp8_test"
PYTHON_CASES = LEGACY_DIR / "qbsa_mxfp8_test_cases.py"
CSV_CASES = LEGACY_DIR / "qbsa_mxfp8_test_cases.csv"
DEFAULT_OUTPUT = CURRENT_DIR / "qbsa_mxfp8_cases.csv"

FIELDNAMES = (
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "input_data_ranges",
    "attributes",
    "is_enabled",
    "remark",
    "priority",
)

BOOL_FIELDS = {
    "enable",
    "is_contiguous",
    "return_softmax_lse",
    "empty_actual_seq",
}
INT_FIELDS = {
    "B",
    "N1",
    "N2",
    "D",
    "S1",
    "S2",
    "s2_base_size",
    "block_size",
    "mask_mode",
    "quant_group_size",
    "seed",
    "device_id",
    "sparse_q_block_size",
    "sparse_kv_block_size",
    "sparse_count",
    "quant_mode",
}
FLOAT_FIELDS = {
    "p_scale_value",
    "data_range_q",
    "data_range_k",
    "data_range_v",
    "softmax_scale",
}
LIST_FIELDS = {"actual_seq_q", "actual_seq_kv"}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_value(field, value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if field in BOOL_FIELDS:
        return parse_bool(value)
    if field in INT_FIELDS:
        return int(value)
    if field in FLOAT_FIELDS:
        return float(value)
    if field in LIST_FIELDS:
        parsed = value if isinstance(value, (list, tuple)) else json.loads(value)
        return [int(item) for item in parsed]
    return value


def load_python_cases():
    namespace = runpy.run_path(str(PYTHON_CASES))
    return [(dict(case), "python") for case in namespace["TestCases"]]


def load_csv_cases():
    cases = []
    with CSV_CASES.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not any(value and value.strip() for value in row.values()):
                continue
            cases.append(
                (
                    {field: parse_value(field, value) for field, value in row.items()},
                    "csv",
                )
            )
    return cases


def normalize_lengths(case, field, max_field):
    batch = int(case["B"])
    values = [int(item) for item in case[field]]
    if len(values) == batch:
        return values
    if len(values) == 1 and batch > 1:
        total = values[0]
        if total == int(case[max_field]) and total % batch == 0:
            return [total // batch] * batch
    raise ValueError(
        f"{case['name']}: {field} must contain B={batch} values, got {values}"
    )


def validate_case(case, q_lengths, kv_lengths):
    name = case["name"]
    if int(case["quant_mode"]) != 2:
        raise ValueError(f"{name}: only quant_mode=2 belongs in the MXFP8 CSV")
    if case["layout_q"] != "TND":
        raise ValueError(f"{name}: MXFP8 currently requires layout_q='TND'")
    if case["layout_kv"] != "PA_BNSD":
        raise ValueError(f"{name}: MXFP8 currently requires layout_kv='PA_BNSD'")
    if int(case["D"]) % 64 != 0:
        raise ValueError(f"{name}: D must be divisible by 64 for paired MXFP8 scales")
    if int(case["N1"]) % int(case["N2"]) != 0:
        raise ValueError(f"{name}: N1 must be divisible by N2")
    page_size = int(case["block_size"])
    if page_size <= 0 or page_size % int(case["sparse_kv_block_size"]) != 0:
        raise ValueError(
            f"{name}: block_size must be a positive multiple of sparse_kv_block_size"
        )
    if sum(q_lengths) <= 0 or sum(kv_lengths) <= 0:
        raise ValueError(f"{name}: total Q/KV sequence length must be positive")


def case_to_row(case, source):
    q_lengths = normalize_lengths(case, "actual_seq_q", "S1")
    kv_lengths = normalize_lengths(case, "actual_seq_kv", "S2")
    validate_case(case, q_lengths, kv_lengths)

    batch = int(case["B"])
    n1 = int(case["N1"])
    n2 = int(case["N2"])
    head_dim = int(case["D"])
    page_size = int(case["block_size"])
    sparse_q = int(case["sparse_q_block_size"])
    sparse_kv = int(case["sparse_kv_block_size"])
    total_q = sum(q_lengths)
    page_count = sum(math.ceil(length / page_size) for length in kv_lengths)
    q_block_count = math.ceil(max(q_lengths) / sparse_q)
    kv_block_count = math.ceil(max(kv_lengths) / sparse_kv)

    tensor_shapes = (
        (total_q, n1, head_dim),
        (page_count, n2, page_size, head_dim),
        (page_count, n2, page_size, head_dim),
        (total_q, n1, head_dim // 64, 2),
        (page_count, n2, page_size, head_dim // 64, 2),
        (page_count, n2, math.ceil(page_size / 64), head_dim, 2),
        (1,),
        (batch, n1, q_block_count, kv_block_count),
        (batch, n1, q_block_count),
        (2048, 2048),
    )
    tensor_dtypes = (
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e8m0",
        "float8_e8m0",
        "float8_e8m0",
        "float8_e8m0",
        "int32",
        "int32",
        "uint8",
    )
    softmax_scale = case.get("softmax_scale")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    attributes = {
        "softmax_scale": float(softmax_scale),
        "sparse_q_block_size": sparse_q,
        "sparse_kv_block_size": sparse_kv,
        "layout_kv": case["layout_kv"],
        "layout_q": case["layout_q"],
        "layout_sparse_indices": case["layout_sparse_indices"],
        "layout_out": case["layout_out"],
        "quant_mode": 2,
        "mask_mode": int(case["mask_mode"]),
        "return_softmax_lse": bool(case["return_softmax_lse"]),
        "actual_seq_q": q_lengths,
        "actual_seq_kv": kv_lengths,
        "empty_actual_seq": bool(case.get("empty_actual_seq", False)),
        "block_table_pattern": case["block_table_pattern"],
        "seed": int(case["seed"]),
        "sparse_count": int(case["sparse_count"]),
        "sparse_pattern": case["sparse_pattern"],
        "p_scale_value": float(case["p_scale_value"]),
        "quant_group_size": int(case["quant_group_size"]),
        "s2_base_size": int(case["s2_base_size"]),
        "data_range_q": float(case["data_range_q"]),
        "data_range_k": float(case["data_range_k"]),
        "data_range_v": float(case["data_range_v"]),
    }
    group = case.get("case_group") or "ungrouped"
    return {
        "testcase_name": case["name"],
        "api_name": "bsa_ttk_ops.quant_block_sparse_attn",
        "tensor_view_shapes": repr(tensor_shapes),
        "tensor_dtypes": repr(tensor_dtypes),
        "input_data_ranges": "",
        "attributes": repr(attributes),
        "is_enabled": str(bool(case.get("enable", True))),
        "remark": f"source={source}; case_group={group}",
        "priority": "0",
    }


def render_csv():
    rows = []
    names = set()
    for case, source in load_python_cases() + load_csv_cases():
        row = case_to_row(case, source)
        name = row["testcase_name"]
        if name in names:
            raise ValueError(f"duplicate testcase_name: {name}")
        names.add(name)
        rows.append(row)

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue(), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the checked-in CSV differs from regenerated content",
    )
    args = parser.parse_args()

    content, count = render_csv()
    if args.check:
        current = args.output.read_text(encoding="utf-8")
        if current != content:
            raise SystemExit(f"{args.output} is stale; regenerate it")
        print(f"validated {count} TTK cases in {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="")
    print(f"generated {count} TTK cases at {args.output}")


if __name__ == "__main__":
    main()
