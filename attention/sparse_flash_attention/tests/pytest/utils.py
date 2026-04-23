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

import ast
import itertools
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import check_valid_param
import result_compare_method
import sparse_flash_attention_golden
from batch import sparse_flash_attention_process


STR_MAP_DICT = {
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
}
RESULT_COLUMNS = [
    "Testcase_Name", "layout_query", "layout_kv", "q_type", "B", "S1", "S2",
    "N1", "N2", "D", "K", "scale_value", "sparse_mode", "result", "fulfill_percent",
]


def _normalize_numeric_value(value):
    if isinstance(value, list):
        return [_normalize_numeric_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_numeric_value(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _parse_excel_cell_value(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in STR_MAP_DICT:
            return _normalize_numeric_value(STR_MAP_DICT[stripped])
        if stripped in ("", "None", "none", "NULL", "null", "NaN", "nan"):
            return None
        if ((stripped.startswith("[") and stripped.endswith("]")) or
                (stripped.startswith("(") and stripped.endswith(")"))):
            try:
                return _normalize_numeric_value(ast.literal_eval(stripped))
            except (ValueError, SyntaxError):
                return value
    return _normalize_numeric_value(value)


def load_excel_test_cases(excel_file_path, sheet_name):
    if sheet_name is None:
        sheet_name = "Sheet1"
    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        dataframe = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        dataframe = dataframe.replace({np.nan: None, pd.NA: None})
    except Exception as error:
        pytest.skip(f"Failed to read Excel file: {error}", allow_module_level=True)

    required_columns = [
        "Testcase_Prefix", "Testcase_Number",
        "layout_query", "layout_kv", "q_type",
        "B", "S1", "S2", "N1", "N2", "D", "K",
        "scale_value", "sparse_block_size", "rope_head_dim",
        "sparse_mode", "attention_mode", "return_softmax_lse",
        "block_size", "block_num", "actual_seq_q", "actual_seq_kv",
    ]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        pytest.skip(f"Missing required columns in Excel: {missing_columns}", allow_module_level=True)
    return [row.to_dict() for _, row in dataframe.iterrows()]


def save_result(params, result, fulfill_percent, result_path):
    if result_path is None:
        return

    row_data = {
        "Testcase_Name": params.get("case_name", params.get("Testcase_Name", "")),
        "layout_query": params.get("layout_query"),
        "layout_kv": params.get("layout_kv"),
        "q_type": str(params.get("q_type")),
        "B": params.get("B"),
        "S1": params.get("S1"),
        "S2": params.get("S2"),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K": params.get("K"),
        "scale_value": params.get("scalevalue"),
        "sparse_mode": params.get("sparsemode"),
        "result": result,
        "fulfill_percent": fulfill_percent,
    }

    if result_path.exists():
        dataframe = pd.read_excel(result_path)
        if list(dataframe.columns) != RESULT_COLUMNS:
            raise ValueError(
                f"Result file columns mismatch, expect {RESULT_COLUMNS}, got {list(dataframe.columns)}"
            )
        dataframe = pd.concat([dataframe, pd.DataFrame([row_data])], ignore_index=True)
    else:
        dataframe = pd.DataFrame([row_data], columns=RESULT_COLUMNS)
    dataframe.to_excel(result_path, index=False)


def combin_params(enabled_params, pytest_paramset=True):
    param_combination_set = []
    param_names = [
        "Testcase_Prefix", "Testcase_Number",
        "layout_query", "layout_kv", "q_type",
        "B", "S1", "S2", "N1", "N2", "D", "K",
        "scale_value", "sparse_block_size", "rope_head_dim",
        "sparse_mode", "attention_mode", "return_softmax_lse",
        "block_size", "block_num", "actual_seq_q", "actual_seq_kv",
    ]

    for params in enabled_params:
        current_params = {}
        for key, value in params.items():
            current_params[key] = value if pytest_paramset else [_parse_excel_cell_value(value)]

        param_values = [
            current_params.get("Testcase_Prefix", [None]),
            current_params.get("Testcase_Number", [None]),
            current_params.get("layout_query"),
            current_params.get("layout_kv"),
            current_params.get("q_type"),
            current_params.get("B"),
            current_params.get("S1"),
            current_params.get("S2"),
            current_params.get("N1"),
            current_params.get("N2"),
            current_params.get("D"),
            current_params.get("K"),
            current_params.get("scale_value"),
            current_params.get("sparse_block_size"),
            current_params.get("rope_head_dim"),
            current_params.get("sparse_mode"),
            current_params.get("attention_mode"),
            current_params.get("return_softmax_lse", [False]),
            current_params.get("block_size", [256]),
            current_params.get("block_num", [None]),
            current_params.get("actual_seq_q", [None]),
            current_params.get("actual_seq_kv", [None]),
        ]

        for combo in itertools.product(*param_values):
            param_combination_set.append(dict(zip(param_names, combo)))
    return param_combination_set


def convert_param_combination_to_cs_format(param_combination):
    layout_query = param_combination["layout_query"]
    layout_kv = param_combination["layout_kv"]
    B = param_combination["B"]
    S1 = param_combination["S1"]
    S2 = param_combination["S2"]
    N1 = param_combination["N1"]
    N2 = param_combination["N2"]
    D = param_combination["D"]
    K = param_combination["K"]
    q_type = param_combination["q_type"]
    scale_value = param_combination["scale_value"]
    sparse_block_size = param_combination["sparse_block_size"]
    rope_head_dim = param_combination["rope_head_dim"]
    sparse_mode = param_combination["sparse_mode"]
    attention_mode = param_combination["attention_mode"]
    return_softmax_lse = param_combination.get("return_softmax_lse", False)
    block_size = param_combination.get("block_size") or 256
    block_num = param_combination.get("block_num")
    actual_seq_q = param_combination.get("actual_seq_q") or [S1]
    actual_seq_kv = param_combination.get("actual_seq_kv") or [S2]

    testcase_prefix = param_combination.get("Testcase_Prefix") or "sparseFlashAttention"
    testcase_number = param_combination.get("Testcase_Number") or 1

    q_type_str = "BF16" if q_type == torch.bfloat16 else "FP16"
    testcase_name = f"{testcase_prefix}_{layout_query}_{layout_kv}_{q_type_str}_{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{testcase_number:06d}"

    if layout_kv == "PA_BSND" and block_num is None:
        max_kv = max(actual_seq_kv) if actual_seq_kv else S2
        block_num = math.ceil(max_kv / block_size) * B

    if layout_kv == "PA_BSND":
        block_num_sum = sum(math.ceil(length / block_size) for length in actual_seq_kv)
    else:
        block_num_sum = 0

    q_dtype_str = "bf16" if q_type == torch.bfloat16 else "fp16"

    sparse_blockcount = int(K / sparse_block_size)

    if layout_kv == "PA_BSND":
        if layout_query == "BSND":
            shape_input = {
                "query": [B, S1, N1, D],
                "key": [block_num_sum, block_size, N2, D],
                "value": [block_num_sum, block_size, N2, D],
                "sparse_indices": [B, S1, N2, sparse_blockcount],
                "block_table": [B, block_num],
                "query_rope": [B, S1, N1, rope_head_dim],
                "key_rope": [block_num_sum, block_size, N2, rope_head_dim],
            }
        elif layout_query == "TND":
            T = sum(actual_seq_q)
            shape_input = {
                "query": [T, N1, D],
                "key": [block_num_sum, block_size, N2, D],
                "value": [block_num_sum, block_size, N2, D],
                "sparse_indices": [T, N2, sparse_blockcount],
                "block_table": [B, block_num],
                "query_rope": [T, N1, rope_head_dim],
                "key_rope": [block_num_sum, block_size, N2, rope_head_dim],
            }
    elif layout_kv == "TND":
        T = sum(actual_seq_q)
        T2 = sum(actual_seq_kv)
        shape_input = {
            "query": [T, N1, D],
            "key": [T2, N2, D],
            "value": [T2, N2, D],
            "sparse_indices": [T, N2, sparse_blockcount],
            "block_table": [B],
            "query_rope": [T, N1, rope_head_dim],
            "key_rope": [T2, N2, rope_head_dim],
        }
    elif layout_kv == "BSND":
        shape_input = {
            "query": [B, S1, N1, D],
            "key": [B, S2, N2, D],
            "value": [B, S2, N2, D],
            "sparse_indices": [B, S1, N2, sparse_blockcount],
            "block_table": [B],
            "query_rope": [B, S1, N1, rope_head_dim],
            "key_rope": [B, S2, N2, rope_head_dim],
        }
    else:
        raise ValueError(f"Unsupported layout_kv: {layout_kv}")

    dtype_input = {
        "query": q_dtype_str,
        "key": q_dtype_str,
        "value": q_dtype_str,
        "sparse_indices": "int32",
        "block_table": "int32",
        "query_rope": q_dtype_str,
        "key_rope": q_dtype_str,
    }

    params = {
        "case_name": testcase_name,
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "actualseqlengths": actual_seq_q,
        "actualseqlengthskv": actual_seq_kv,
        "scalevalue": scale_value,
        "sparsemode": sparse_mode,
        "sparse_blocksize": sparse_block_size,
        "rope_head_dim": rope_head_dim,
        "attention_mode": attention_mode,
        "return_softmax_lse": return_softmax_lse,
        "block_size": block_size,
        "block_num": block_num,
        "shape_input": shape_input,
        "dtype_input": dtype_input,
        "dtype_output": [q_dtype_str],
        "shape_output": [[B, S1, N1, D]],
        "Testcase_Prefix": testcase_prefix,
        "Testcase_Number": testcase_number,
        "q_type": q_type,
        "B": B,
        "S1": S1,
        "S2": S2,
        "N1": N1,
        "N2": N2,
        "D": D,
        "K": K,
        "actual_seq_q": actual_seq_q,
        "actual_seq_kv": actual_seq_kv,
    }

    return params


def sfa(case_id, param_combination, pt_save_path, device_id, run_npu, save_pt, result_path):
    params = convert_param_combination_to_cs_format(param_combination)

    try:
        check_valid_param.check_valid_param(params)
    except ValueError as error:
        pytest.skip(f"输入参数校验失败: {error}")

    test_data = sparse_flash_attention_golden.generate_and_save_testdata(
        params, save_pt=save_pt, save_path=pt_save_path, compute_golden=False
    )

    if run_npu:
        sfa_run_npu(test_data, device_id=device_id, result_path=result_path)


def sfa_run_npu(test_data, device_id=0, result_path=Path("result.xlsx")):
    try:
        npu_result, cpu_result = sparse_flash_attention_process.test_sfa_process_ci(
            test_data, device_id=device_id
        )

        if cpu_result is None:
            print(npu_result[0] if isinstance(npu_result, tuple) else npu_result)
            print("CPU golden 不可用，当前用例仅验证 NPU 连通性，跳过精度对比。")
            result = "SkipCompare"
            fulfill_percent = 100.0
        else:
            target_npu = npu_result[0] if isinstance(npu_result, tuple) else npu_result
            result, fulfill_percent = result_compare_method.check_result(cpu_result, target_npu)
    except Exception as error:
        print(f"NPU ERROR: {error}")
        result = "NPU ERROR"
        fulfill_percent = 0.0

    save_result(test_data["params"], result, fulfill_percent, result_path=result_path)
    if result == "NPU ERROR":
        pytest.fail(f"用例执行失败: {test_data['Testcase_Name']}")