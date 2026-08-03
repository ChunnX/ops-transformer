#!/usr/bin/python
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

import argparse
import ast
import operator
import os

import pandas as pd
import torch

from quant_lightning_indexer_v2_golden import generate_qliv2_test_data


PARAM_COLUMNS = (
    "batch_size",
    "q_seq",
    "k_seq",
    "q_t_size",
    "k_t_size",
    "q_head_num",
    "k_head_num",
    "head_dim",
    "block_size",
    "block_num",
    "qk_dtype",
    "weight_dtype",
    "dequant_dtype",
    "actual_seq_dtype",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_q",
    "seqused_k",
    "cmp_residual_k",
    "max_seqlen_q",
    "quant_mode",
    "layout_query",
    "layout_key",
    "sparse_count",
    "sparse_mode",
    "query_datarange",
    "key_datarange",
    "weights_datarange",
    "q_scale_datarange",
    "k_scale_datarange",
    "cmp_ratio",
    "return_value",
    "output_idx_offset",
)

PARAMSET_REQUIRED_COLUMNS = {"Testcase_Name", *PARAM_COLUMNS} - {"weight_dtype"}

# QLI_LI_redline.xlsx style columns. Shape/dtype columns which are output-only are
# intentionally not required because PT generation derives outputs from op inputs.
REDLINE_REQUIRED_COLUMNS = {
    "Testcase_Name",
    "B",
    "Q_S",
    "KV_S",
    "Q_N",
    "KV_N",
    "D",
    "block_size",
    "topk",
    "max_seqlen_q",
    "mask_mode",
    "cmp_ratio",
    "quant_mode",
    "return_value",
    "q_layout",
    "k_layout",
    "q_shape",
    "q_dtype",
    "q_datarange",
    "k_shape",
    "k_dtype",
    "k_datarange",
    "weight_dtype",
    "weight_datarange",
    "q_descale_dtype",
    "q_descale_datarange",
    "k_descale_dtype",
    "k_descale_datarange",
    "cu_seqlens_q_dtype",
    "cu_seqlens_q_datarange",
    "cu_seqlens_k_dtype",
    "cu_seqlens_k_datarange",
    "seqused_q_dtype",
    "seqused_q_datarange",
    "seqused_k_dtype",
    "seqused_k_datarange",
    "cmp_residual_k_dtype",
    "cmp_residual_k_datarange",
    "output_idx_offset_dtype",
    "output_idx_offset",
}

DTYPE_ALIASES = {
    "FP8_E4M3FN": "FLOAT8_E4M3FN",
    "FLOAT8_E4M3FN": "FLOAT8_E4M3FN",
    "FP8_E8M0": "FLOAT8_E8M0",
    "FP8_E8M0FNU": "FLOAT8_E8M0FNU",
    "FLOAT8_E8M0": "FLOAT8_E8M0",
    "FLOAT8_E8M0FNU": "FLOAT8_E8M0FNU",
    "FP4_E2M1": "FLOAT4_E2M1",
    "FLOAT4_E2M1": "FLOAT4_E2M1",
    "FLOAT4_E2M1FN_X2": "FLOAT4_E2M1FN_X2",
}

_BINARY_NUMERIC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_NUMERIC_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_int(value, field_name, allow_none=False):
    if _is_missing(value):
        if allow_none:
            return None
        raise ValueError(f"{field_name} must not be empty")
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{field_name} must be an integer, but got {value!r}")
    return int(number)


def _eval_numeric_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_NUMERIC_OPS:
        return _UNARY_NUMERIC_OPS[type(node.op)](_eval_numeric_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_NUMERIC_OPS:
        return _BINARY_NUMERIC_OPS[type(node.op)](
            _eval_numeric_node(node.left), _eval_numeric_node(node.right)
        )
    raise ValueError(f"unsupported numeric expression: {ast.unparse(node)}")


def _parse_numeric_list(value, field_name):
    if _is_missing(value):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (int, float)):
        return [value]

    text = str(value).strip()
    if not text:
        return None
    try:
        node = ast.parse(text, mode="eval").body
        nodes = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
        return [_eval_numeric_node(item) for item in nodes]
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(f"invalid {field_name}: {value!r}") from error


def _parse_int_list(value, field_name):
    values = _parse_numeric_list(value, field_name)
    if values is None:
        return None
    return [_as_int(item, field_name) for item in values]


def _shape_first(value, field_name):
    shape = _parse_int_list(value, field_name)
    if not shape:
        raise ValueError(f"{field_name} must contain at least one dimension")
    return shape[0]


def _normalize_dtype(value, field_name):
    if _is_missing(value):
        raise ValueError(f"{field_name} must not be empty")
    if not isinstance(value, str):
        return value
    dtype = value.strip().upper()
    if dtype.startswith("TORCH."):
        dtype = dtype[6:]
    return DTYPE_ALIASES.get(dtype, dtype)


def _first_sequence_dtype(row):
    fields = (
        "cu_seqlens_q_dtype",
        "cu_seqlens_k_dtype",
        "seqused_q_dtype",
        "seqused_k_dtype",
        "cmp_residual_k_dtype",
        "output_idx_offset_dtype",
    )
    dtypes = {
        _normalize_dtype(row[field], field)
        for field in fields
        if not _is_missing(row[field])
    }
    if len(dtypes) > 1:
        raise ValueError(f"sequence tensors must use one dtype, but got {sorted(dtypes)}")
    return next(iter(dtypes), "INT32")


def _offset_range_expression(value):
    values = _parse_int_list(value, "output_idx_offset")
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("output_idx_offset must contain [min, max]")
    return repr(values)


def _validate_tnd_size(layout, total_size, sequence, field_name):
    if layout != "TND" or sequence is None:
        return
    if sequence[-1] != total_size:
        raise ValueError(
            f"{field_name} last value ({sequence[-1]}) does not match tensor total size ({total_size})"
        )


def _normalize_paramset_row(row):
    weight_dtype = row.get("weight_dtype")
    if _is_missing(weight_dtype):
        weight_dtype = row["dequant_dtype"]
    params = {name: row.get(name) for name in PARAM_COLUMNS}
    params["weight_dtype"] = weight_dtype
    return (str(row["Testcase_Name"]), *(params[name] for name in PARAM_COLUMNS))


def _normalize_redline_row(row):
    q_layout = str(row["q_layout"]).strip().upper()
    k_layout = str(row["k_layout"]).strip().upper()
    q_total_size = _shape_first(row["q_shape"], "q_shape")
    k_total_size = _shape_first(row["k_shape"], "k_shape")
    cu_seqlens_q = _parse_int_list(row["cu_seqlens_q_datarange"], "cu_seqlens_q_datarange")
    cu_seqlens_k = _parse_int_list(row["cu_seqlens_k_datarange"], "cu_seqlens_k_datarange")

    _validate_tnd_size(q_layout, q_total_size, cu_seqlens_q, "cu_seqlens_q")
    _validate_tnd_size(k_layout, k_total_size, cu_seqlens_k, "cu_seqlens_k")

    q_dtype = _normalize_dtype(row["q_dtype"], "q_dtype")
    k_dtype = _normalize_dtype(row["k_dtype"], "k_dtype")
    if q_dtype != k_dtype:
        raise ValueError(f"q_dtype and k_dtype must match, but got {q_dtype} and {k_dtype}")

    q_descale_dtype = _normalize_dtype(row["q_descale_dtype"], "q_descale_dtype")
    k_descale_dtype = _normalize_dtype(row["k_descale_dtype"], "k_descale_dtype")
    if q_descale_dtype != k_descale_dtype:
        raise ValueError(
            "q_descale_dtype and k_descale_dtype must match, "
            f"but got {q_descale_dtype} and {k_descale_dtype}"
        )

    params = {
        "batch_size": _as_int(row["B"], "B"),
        "q_seq": _as_int(row["Q_S"], "Q_S"),
        "k_seq": _as_int(row["KV_S"], "KV_S"),
        "q_t_size": q_total_size if q_layout == "TND" else None,
        "k_t_size": k_total_size if k_layout == "TND" else None,
        "q_head_num": _as_int(row["Q_N"], "Q_N"),
        "k_head_num": _as_int(row["KV_N"], "KV_N"),
        "head_dim": _as_int(row["D"], "D"),
        "block_size": _as_int(row["block_size"], "block_size", allow_none=True),
        "block_num": k_total_size if k_layout.startswith("PA_") else None,
        "qk_dtype": q_dtype,
        "weight_dtype": _normalize_dtype(row["weight_dtype"], "weight_dtype"),
        "dequant_dtype": q_descale_dtype,
        "actual_seq_dtype": _first_sequence_dtype(row),
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "seqused_q": _parse_int_list(row["seqused_q_datarange"], "seqused_q_datarange"),
        "seqused_k": _parse_int_list(row["seqused_k_datarange"], "seqused_k_datarange"),
        "cmp_residual_k": _parse_int_list(
            row["cmp_residual_k_datarange"], "cmp_residual_k_datarange"
        ),
        "max_seqlen_q": _as_int(row["max_seqlen_q"], "max_seqlen_q"),
        "quant_mode": _as_int(row["quant_mode"], "quant_mode"),
        "layout_query": q_layout,
        "layout_key": k_layout,
        "sparse_count": _as_int(row["topk"], "topk"),
        "sparse_mode": _as_int(row["mask_mode"], "mask_mode"),
        "query_datarange": _parse_numeric_list(row["q_datarange"], "q_datarange"),
        "key_datarange": _parse_numeric_list(row["k_datarange"], "k_datarange"),
        "weights_datarange": _parse_numeric_list(row["weight_datarange"], "weight_datarange"),
        "q_scale_datarange": _parse_numeric_list(
            row["q_descale_datarange"], "q_descale_datarange"
        ),
        "k_scale_datarange": _parse_numeric_list(
            row["k_descale_datarange"], "k_descale_datarange"
        ),
        "cmp_ratio": _as_int(row["cmp_ratio"], "cmp_ratio"),
        "return_value": _as_int(row["return_value"], "return_value"),
        # Golden treats a string as a [min, max] range and expands it to one
        # random offset per query row.
        "output_idx_offset": _offset_range_expression(row["output_idx_offset"]),
    }
    return (str(row["Testcase_Name"]), *(params[name] for name in PARAM_COLUMNS))


def load_excel_test_cases(excel_file_path: str, sheetname: str = "Sheet1"):
    """Load either paramset-style or QLI_LI_redline-style Excel cases."""
    if not os.path.exists(excel_file_path):
        raise FileNotFoundError(f"Excel file not found: {excel_file_path}")

    df = pd.read_excel(excel_file_path, sheet_name=sheetname or "Sheet1")
    df = df.rename(columns=lambda column: str(column).strip())
    df = df.where(pd.notna(df), None)
    columns = set(df.columns)

    if PARAMSET_REQUIRED_COLUMNS.issubset(columns):
        normalizer = _normalize_paramset_row
        schema_name = "paramset"
    elif REDLINE_REQUIRED_COLUMNS.issubset(columns):
        normalizer = _normalize_redline_row
        schema_name = "QLI_LI_redline"
    else:
        paramset_missing = sorted(PARAMSET_REQUIRED_COLUMNS - columns)
        redline_missing = sorted(REDLINE_REQUIRED_COLUMNS - columns)
        raise ValueError(
            "Unsupported QLIV2 Excel schema. "
            f"Missing paramset columns: {paramset_missing}; "
            f"missing QLI_LI_redline columns: {redline_missing}"
        )

    test_cases = []
    for row_index, row in df.iterrows():
        if _is_missing(row.get("Testcase_Name")):
            continue
        try:
            test_cases.append(normalizer(row))
        except (KeyError, TypeError, ValueError) as error:
            excel_row = row_index + 2
            raise ValueError(
                f"Invalid {schema_name} case at Excel row {excel_row} "
                f"({row.get('Testcase_Name')}): {error}"
            ) from error
    return test_cases


def save_test_case(test_cases, file_path):
    print("正在保存 PT 文件...")
    os.makedirs(file_path, exist_ok=True)
    failures = []

    for index, case in enumerate(test_cases):
        try:
            case_name = case[0]
            output_tensors = generate_qliv2_test_data(case[1:])
            input_filepath = os.path.join(file_path, f"{case_name}.pt")
            torch.save(output_tensors, input_filepath)
            print(f"测试用例已保存到: {input_filepath}")
        except Exception as error:
            failures.append((case[0], index, error))
            print(f"[失败] 生成 PT 文件失败: {case[0]} (索引: {index})")
            print(f"错误详情: {error}")

    if failures:
        names = ", ".join(name for name, _, _ in failures)
        raise RuntimeError(f"Failed to generate {len(failures)} PT case(s): {names}")


def main():
    parser = argparse.ArgumentParser(description="Generate QLIV2 PT cases from Excel")
    parser.add_argument("path1", type=str, help="Excel file path")
    parser.add_argument("path2", type=str, help="PT output directory")
    parser.add_argument(
        "--sheet", "-S", type=str, default="Sheet1", help="Sheet name (default: Sheet1)"
    )
    args = parser.parse_args()
    test_cases = load_excel_test_cases(args.path1, args.sheet)
    save_test_case(test_cases, args.path2)


if __name__ == "__main__":
    main()
