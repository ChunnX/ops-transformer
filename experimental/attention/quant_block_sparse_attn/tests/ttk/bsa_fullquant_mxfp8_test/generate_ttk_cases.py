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

"""Convert enabled QBSA MXFP8 Python pytest cases to a TTK E2E CSV."""

import argparse
import csv
import io
import math
import runpy
import warnings
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
LEGACY_DIR = CURRENT_DIR.parents[1] / "pytest" / "bsa_fullquant_mxfp8_test"
PYTHON_CASES = LEGACY_DIR / "qbsa_mxfp8_test_cases.py"
DEFAULT_OUTPUT = CURRENT_DIR / "qbsa_mxfp8_cases.csv"

FIELDNAMES = (
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "attributes",
    "output_tensor_indexes",
    "golden_api",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "uncontiguous_dim",
)

PRECISION_TOLERANCES = ((0.0078125, 0.0001), (0.005, 0.000025))
ABSOLUTE_PRECISION = 1e-8


def prefix_sum(lengths):
    total = 0
    result = [0]
    for length in lengths:
        total += int(length)
        result.append(total)
    return result


def validate_sparse_count(name, sparse_count, batch):
    if isinstance(sparse_count, (list, tuple)):
        if len(sparse_count) != batch:
            raise ValueError(
                f"{name}: sparse_count list length must equal B={batch}"
            )
        if any(int(value) < 0 for value in sparse_count):
            raise ValueError(f"{name}: sparse_count list values must be >= 0")
        return
    if isinstance(sparse_count, str):
        value = sparse_count.strip().lower()
        if value in {"random", "empty_random", "empty"}:
            return
        if value.endswith("%"):
            percent = float(value[:-1])
            if 0 < percent <= 100:
                return
            raise ValueError(
                f"{name}: sparse_count percentage must be in (0, 100]"
            )
    if int(sparse_count) < 0:
        raise ValueError(f"{name}: sparse_count integer must be >= 0")

def load_python_cases():
    namespace = runpy.run_path(str(PYTHON_CASES))
    return [
        (dict(case), "python")
        for case in namespace["TestCases"]
        if case.get("enable") is True
    ]


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


def normalize_data_range(case, field):
    value = case[field]
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"{case['name']}: {field} must contain exactly [min, max]"
            )
        low, high = float(value[0]), float(value[1])
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(
                f"{case['name']}: {field} bounds must be finite, got {value}"
            )
        if low > high:
            raise ValueError(
                f"{case['name']}: {field} min must not exceed max, got {value}"
            )
        return [low, high]

    radius = float(value)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError(
            f"{case['name']}: scalar {field} must be finite and non-negative, "
            f"got {value}"
        )
    return radius


def ttk_data_range(value):
    if isinstance(value, list):
        return tuple(value)
    radius = float(value)
    return (-radius, radius)


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
    validate_sparse_count(name, case["sparse_count"], int(case["B"]))
    blocknum = case.get("blocknum")
    if blocknum is not None and int(blocknum) <= 0:
        warnings.warn(
            f"{name}: blocknum is expected to be >= 0; got {blocknum} (<= 0), "
            "so it will be derived from actual_seq_kv",
            UserWarning,
            stacklevel=2,
        )


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
    logical_page_count = sum(math.ceil(length / page_size) for length in kv_lengths)
    blocknum = case.get("blocknum")
    physical_page_count = (
        int(blocknum)
        if blocknum is not None and int(blocknum) > 0
        else logical_page_count
    )
    q_block_count = math.ceil(max(q_lengths) / sparse_q)
    kv_block_count = math.ceil(max(kv_lengths) / sparse_kv)
    p_scale_value = case.get("p_scale_value")

    # TTK's E2E param planner drops None entries while matching a Python
    # signature.  Keep the optional p_scale slot last so a missing value cannot
    # shift the required sparse-index and mask tensors into the wrong positions.
    tensor_shapes = (
        (total_q, n1, head_dim),
        (physical_page_count, n2, page_size, head_dim),
        (physical_page_count, n2, page_size, head_dim),
        (total_q, n1, head_dim // 64, 2),
        (physical_page_count, n2, page_size, head_dim // 64, 2),
        (
            physical_page_count,
            n2,
            math.ceil(page_size / 64),
            head_dim,
            2,
        ),
        (batch, n1, q_block_count, kv_block_count),
        (batch, n1, q_block_count),
        (2048, 2048),
        None if p_scale_value is None else (1,),
    )
    tensor_dtypes = (
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e8m0",
        "float8_e8m0",
        "float8_e8m0",
        "int32",
        "int32",
        "uint8",
        None if p_scale_value is None else "float8_e8m0",
    )
    softmax_scale = case.get("softmax_scale")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Keep CSV attributes in three stable groups:
    # 1) native operator attributes, in API order;
    # 2) values materialized as optional input tensors;
    # 3) test-data/case construction controls.
    attributes = {
        # Native operator attributes.
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
        # Optional input-tensor values, in API input order.
        "p_scale_value": (
            None if p_scale_value is None else float(p_scale_value)
        ),
        "cu_seqlens_q_value": prefix_sum(q_lengths),
        "cu_seqlens_kv_value": [],
        "seqused_q_value": [],
        "seqused_kv_value": kv_lengths,
        # Test-data and case-construction controls.
        "is_contiguous": bool(case.get("is_contiguous", True)),
        "block_table_pattern": case["block_table_pattern"],
        "sparse_count": case["sparse_count"],
        "sparse_pattern": case["sparse_pattern"],
        "seed": int(case["seed"]),
        "quant_group_size": int(case["quant_group_size"]),
        "s2_base_size": int(case["s2_base_size"]),
        "data_range_q": normalize_data_range(case, "data_range_q"),
        "data_range_k": normalize_data_range(case, "data_range_k"),
        "data_range_v": normalize_data_range(case, "data_range_v"),
    }
    # blocknum is a construction control; insert it before the data ranges while
    # preserving the three-group layout above.
    if blocknum is not None:
        range_items = {
            key: attributes.pop(key)
            for key in ("data_range_q", "data_range_k", "data_range_v")
        }
        attributes["blocknum"] = int(blocknum)
        attributes.update(range_items)
    tensor_formats = tuple(
        None if shape is None else "ND" for shape in tensor_shapes
    )
    input_data_ranges = (
        ttk_data_range(attributes["data_range_q"]),
        ttk_data_range(attributes["data_range_k"]),
        ttk_data_range(attributes["data_range_v"]),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (
            (None, None)
            if p_scale_value is None
            else (float(p_scale_value), float(p_scale_value))
        ),
    )
    return {
        "testcase_name": case["name"],
        "api_name": "bsa_ttk_ops.quant_block_sparse_attn",
        "tensor_view_shapes": repr(tensor_shapes),
        "tensor_dtypes": repr(tensor_dtypes),
        "tensor_formats": repr(tensor_formats),
        "attributes": repr(attributes),
        "output_tensor_indexes": "()",
        "golden_api": "",
        "input_data_ranges": repr(input_data_ranges),
        "precision_tolerances": repr(PRECISION_TOLERANCES),
        "absolute_precision": repr(ABSOLUTE_PRECISION),
        "uncontiguous_dim": (
            "-1" if bool(case.get("is_contiguous", True)) else "1"
        ),
    }


def render_csv():
    rows = []
    names = set()
    for case, source in load_python_cases():
        row = case_to_row(case, source)
        name = row["testcase_name"]
        if name in names:
            raise ValueError(f"duplicate enabled testcase_name: {name}")
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
