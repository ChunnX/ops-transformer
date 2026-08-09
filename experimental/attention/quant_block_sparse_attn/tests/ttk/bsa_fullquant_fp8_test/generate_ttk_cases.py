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

"""Convert enabled QBSA FP8 pytest STC cases to a TTK E2E CSV."""

import argparse
import ast
import csv
import io
import math
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = CURRENT_DIR.parents[1] / "pytest" / "batch" / "cases_stc.csv"
DEFAULT_OUTPUT = CURRENT_DIR / "qbsa_fp8_stc_cases.csv"

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


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no", ""):
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_range(value, case_name, field):
    try:
        parsed = ast.literal_eval(str(value).strip())
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            f"{case_name}: {field} must be a two-element numeric range"
        ) from error
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        raise ValueError(
            f"{case_name}: {field} must be a two-element numeric range"
        )
    low, high = float(parsed[0]), float(parsed[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{case_name}: invalid {field} range {parsed!r}")
    return (low, high)


def load_enabled_cases(source):
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"empty FP8 STC source: {source}")
        rows = []
        for source_row, case in enumerate(reader, start=2):
            if not parse_bool(case.get("enable", False)):
                continue
            case = dict(case)
            case["_source_row"] = source_row
            rows.append(case)
    return rows


def query_shapes(layout_q, batch, seq_q, num_heads_q, head_dim):
    total_q = batch * seq_q
    if layout_q == "TND":
        query_shape = (total_q, num_heads_q, head_dim)
        q_descale_shape = (total_q, num_heads_q)
    elif layout_q == "NTD":
        query_shape = (num_heads_q, total_q, head_dim)
        q_descale_shape = (num_heads_q, total_q)
    elif layout_q == "BSND":
        query_shape = (batch, seq_q, num_heads_q, head_dim)
        q_descale_shape = (batch, seq_q, num_heads_q)
    else:
        raise ValueError(f"unsupported layout_q: {layout_q!r}")
    return query_shape, q_descale_shape


def validate_case(case):
    name = case["case_name"]
    positive_fields = (
        "B",
        "N1",
        "N2",
        "S1",
        "S2",
        "D",
        "sparse_q_block_size",
        "sparse_kv_block_size",
        "block_num",
    )
    for field in positive_fields:
        if int(case[field]) <= 0:
            raise ValueError(f"{name}: {field} must be positive")
    if int(case["quant_mode"]) != 1:
        raise ValueError(f"{name}: FP8 STC requires quant_mode=1")
    if int(case["N1"]) % int(case["N2"]) != 0:
        raise ValueError(f"{name}: N1 must be divisible by N2")
    if int(case["N1"]) // int(case["N2"]) > 16:
        raise ValueError(f"{name}: N1/N2 must not exceed 16")
    if case["layout_kv"] != "PA_BNSD":
        raise ValueError(f"{name}: layout_kv must be PA_BNSD")
    if case["layout_sparse_indices"] != "B_N_Qb_Kb":
        raise ValueError(
            f"{name}: layout_sparse_indices must be B_N_Qb_Kb"
        )
    if case["actlen_mode"] not in ("full", "random"):
        raise ValueError(
            f"{name}: unsupported actlen_mode {case['actlen_mode']!r}"
        )
    if case["sparse_pattern"] not in (
        "sequential",
        "reverse",
        "random",
        "causal",
        "dense",
        "tail",
        "empty",
    ):
        raise ValueError(
            f"{name}: unsupported sparse_pattern {case['sparse_pattern']!r}"
        )
    if case["block_table_pattern"] not in (
        "sequential",
        "reverse",
        "random",
    ):
        raise ValueError(
            f"{name}: unsupported block_table_pattern "
            f"{case['block_table_pattern']!r}"
        )


def case_to_row(case):
    validate_case(case)
    name = case["case_name"]
    batch = int(case["B"])
    n1 = int(case["N1"])
    n2 = int(case["N2"])
    seq_q = int(case["S1"])
    seq_kv = int(case["S2"])
    head_dim = int(case["D"])
    sparse_q = int(case["sparse_q_block_size"])
    sparse_kv = int(case["sparse_kv_block_size"])
    block_num = int(case["block_num"])
    page_block_size = sparse_kv
    query_shape, q_descale_shape = query_shapes(
        case["layout_q"], batch, seq_q, n1, head_dim
    )

    # The public TTK wrapper keeps all required tensors first and the optional
    # p_scale last.  This differs from the native operator's argument order.
    tensor_shapes = (
        query_shape,
        (block_num, n2, page_block_size, head_dim),
        (block_num, n2, page_block_size, head_dim),
        q_descale_shape,
        (block_num, n2, page_block_size, 1),
        (n2,),
        (batch, n1, math.ceil(seq_q / sparse_q), math.ceil(seq_kv / sparse_kv)),
        (batch, n1, math.ceil(seq_q / sparse_q)),
        (2048, 2048),
        (1,),
    )
    tensor_dtypes = (
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float32",
        "float32",
        "float32",
        "int32",
        "int32",
        "uint8",
        "float32",
    )
    p_scale_value = float(case["p_scale_value"])
    input_ranges = (
        parse_range(case["q_datarange"], name, "q_datarange"),
        parse_range(case["k_datarange"], name, "k_datarange"),
        parse_range(case["v_datarange"], name, "v_datarange"),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (p_scale_value, p_scale_value),
    )
    attributes = {
        "softmax_scale": float(case["softmax_scale"]),
        "sparse_q_block_size": sparse_q,
        "sparse_kv_block_size": sparse_kv,
        "max_seqlen_q": seq_q,
        "max_seqlen_kv": seq_kv,
        "layout_kv": case["layout_kv"],
        "layout_q": case["layout_q"],
        "layout_sparse_indices": case["layout_sparse_indices"],
        "layout_out": case["layout_out"],
        "quant_mode": 1,
        "mask_mode": int(case["mask_mode"]),
        "return_softmax_lse": parse_bool(case["return_softmax_lse"]),
        "sparse_count": int(case["sparse_count"]),
        "sparse_pattern": case["sparse_pattern"],
        "block_table_pattern": case["block_table_pattern"],
        "actlen_mode": case["actlen_mode"],
        "p_scale_value": p_scale_value,
        "blocknum": block_num,
        "seed": int(case["seed"]),
    }
    remark = (
        f"source=pytest/batch/cases_stc.csv:{case['_source_row']}; "
        f"{case.get('remark', '').strip()}"
    )
    return {
        "testcase_name": name,
        "api_name": "bsa_ttk_ops.quant_block_sparse_attn",
        "tensor_view_shapes": repr(tensor_shapes),
        "tensor_dtypes": repr(tensor_dtypes),
        "input_data_ranges": repr(input_ranges),
        "attributes": repr(attributes),
        "is_enabled": "True",
        "remark": remark,
        "priority": "0",
    }


def render_csv(source):
    rows = []
    names = set()
    for case in load_enabled_cases(source):
        row = case_to_row(case)
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
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the checked-in CSV differs from regenerated content",
    )
    args = parser.parse_args()

    content, count = render_csv(args.source)
    if args.check:
        current = args.output.read_text(encoding="utf-8")
        if current != content:
            raise SystemExit(f"{args.output} is stale; regenerate it")
        print(f"validated {count} FP8 STC TTK cases in {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="")
    print(f"generated {count} FP8 STC TTK cases at {args.output}")


if __name__ == "__main__":
    main()
