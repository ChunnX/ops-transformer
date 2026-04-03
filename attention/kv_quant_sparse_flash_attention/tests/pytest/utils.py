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
import kv_quant_sparse_flash_attention_golden
import result_compare_method
from batch import kv_quant_sparse_flash_attention_process


STR_MAP_DICT = {
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
}


def _parse_excel_cell_value(value):
    # 将 Excel 中的字符串单元格转成 pytest 框架需要的 Python 类型。
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in STR_MAP_DICT:
            return STR_MAP_DICT[stripped]
        if stripped in ("", "None", "none", "NULL", "null", "NaN", "nan"):
            return None
        if ((stripped.startswith("[") and stripped.endswith("]")) or
                (stripped.startswith("(") and stripped.endswith(")"))):
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return value
    return value


def load_excel_test_cases(excel_file_path, sheet_name):
    # batch 模式从 Excel 读取原始参数行，并校验字段完整性。
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
        "layout_query", "layout_kv", "q_type", "kv_dtype",
        "B", "S1", "S2", "N1", "N2", "D", "K",
        "scale_value", "key_quant_mode", "value_quant_mode",
        "sparse_block_size", "tile_size", "rope_head_dim",
        "sparse_mode", "attention_mode", "quant_scale_repo_mode",
        "block_size", "block_num", "actual_seq_q", "actual_seq_kv",
    ]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        pytest.skip(f"Missing required columns in Excel: {missing_columns}", allow_module_level=True)
    return [row.to_dict() for _, row in dataframe.iterrows()]


def save_result(params, result, fulfill_percent, result_path):
    # 把每个用例的执行结果统一追加到 result.xlsx。
    row_data = {
        "Testcase_Name": params[0],
        "layout_query": params[1],
        "layout_kv": params[2],
        "q_type": str(params[3]),
        "kv_dtype": params[4],
        "B": params[5],
        "S1": params[6],
        "S2": params[7],
        "N1": params[8],
        "N2": params[9],
        "D": params[10],
        "K": params[11],
        "scale_value": params[12],
        "sparse_mode": params[18],
        "result": result,
        "fulfill_percent": fulfill_percent,
    }

    if result_path.exists():
        dataframe = pd.read_excel(result_path)
        if set(dataframe.columns) != set(row_data.keys()):
            raise ValueError(
                f"Result file columns mismatch, expect {list(row_data.keys())}, got {list(dataframe.columns)}"
            )
        dataframe = pd.concat([dataframe, pd.DataFrame([row_data])], ignore_index=True)
    else:
        dataframe = pd.DataFrame([row_data])
    dataframe.to_excel(result_path, index=False)


def combin_params(enabled_params, pytest_paramset=True):
    # 将参数表展开成 pytest.parametrize 可直接消费的字典列表。
    param_combination_set = []
    param_names = [
        "Testcase_Prefix", "Testcase_Number",
        "layout_query", "layout_kv", "q_type", "kv_dtype",
        "B", "S1", "S2", "N1", "N2", "D", "K",
        "scale_value", "key_quant_mode", "value_quant_mode",
        "sparse_block_size", "tile_size", "rope_head_dim",
        "sparse_mode", "attention_mode", "quant_scale_repo_mode",
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
            current_params.get("kv_dtype"),
            current_params.get("B"),
            current_params.get("S1"),
            current_params.get("S2"),
            current_params.get("N1"),
            current_params.get("N2"),
            current_params.get("D"),
            current_params.get("K"),
            current_params.get("scale_value"),
            current_params.get("key_quant_mode"),
            current_params.get("value_quant_mode"),
            current_params.get("sparse_block_size"),
            current_params.get("tile_size"),
            current_params.get("rope_head_dim"),
            current_params.get("sparse_mode"),
            current_params.get("attention_mode"),
            current_params.get("quant_scale_repo_mode"),
            current_params.get("block_size", [256]),
            current_params.get("block_num", [None]),
            current_params.get("actual_seq_q", [None]),
            current_params.get("actual_seq_kv", [None]),
        ]

        print("param_values", param_values)
        for combo in itertools.product(*param_values):
            param_combination_set.append(dict(zip(param_names, combo)))
        print(param_combination_set)
    return param_combination_set


def qsfa(case_id, param_combination, pt_save_path, device_id, run_npu, save_pt, result_path):
    # single/batch 共用入口：负责参数整理、校验、构造输入和可选执行。
    testcase_prefix = param_combination.get("Testcase_Prefix") or "kvQuantSparseFlashAttn"
    testcase_number = param_combination.get("Testcase_Number") or case_id
    layout_query = param_combination["layout_query"]
    layout_kv = param_combination["layout_kv"]
    q_type = param_combination["q_type"]
    kv_dtype = param_combination["kv_dtype"]
    batch_size = param_combination["B"]
    seq_q = param_combination["S1"]
    seq_kv = param_combination["S2"]
    head_num_q = param_combination["N1"]
    head_num_kv = param_combination["N2"]
    head_dim = param_combination["D"]
    sparse_count = param_combination["K"]
    scale_value = param_combination["scale_value"]
    key_quant_mode = param_combination["key_quant_mode"]
    value_quant_mode = param_combination["value_quant_mode"]
    sparse_block_size = param_combination["sparse_block_size"]
    tile_size = param_combination["tile_size"]
    rope_head_dim = param_combination["rope_head_dim"]
    sparse_mode = param_combination["sparse_mode"]
    attention_mode = param_combination["attention_mode"]
    quant_scale_repo_mode = param_combination["quant_scale_repo_mode"]
    block_size = param_combination.get("block_size") or 256
    block_num = param_combination.get("block_num")
    actual_seq_q = param_combination.get("actual_seq_q")
    actual_seq_kv = param_combination.get("actual_seq_kv")

    q_type_str = "BF16" if q_type == torch.bfloat16 else "FP16"
    testcase_name = (
        f"{testcase_prefix}_{layout_query}_{layout_kv}_{q_type_str}_"
        f"{batch_size}_{head_num_q}_{head_num_kv}_{seq_q}_{seq_kv}_{head_dim}_{sparse_count}_{testcase_number:06d}"
    )

    if layout_kv == "PA_BSND" and block_num is None:
        # 未显式给出 block_num 时，按当前实际 kv 长度自动推导。
        max_kv = max(actual_seq_kv) if actual_seq_kv else seq_kv
        block_num = math.ceil(max_kv / block_size) * batch_size

    params = (
        testcase_name, layout_query, layout_kv, q_type, kv_dtype,
        batch_size, seq_q, seq_kv, head_num_q, head_num_kv, head_dim, sparse_count,
        scale_value, key_quant_mode, value_quant_mode, sparse_block_size,
        tile_size, rope_head_dim, sparse_mode, attention_mode, quant_scale_repo_mode,
        block_size, block_num, actual_seq_q, actual_seq_kv,
    )

    try:
        check_valid_param.check_valid_param(params)
    except ValueError as error:
        pytest.skip(f"输入参数校验失败: {error}")

    test_data = kv_quant_sparse_flash_attention_golden.generate_and_save_testdata(
        params, save_pt=save_pt, save_path=pt_save_path, compute_golden=False
    )
    if run_npu:
        qsfa_run_npu(test_data, device_id=device_id, result_path=result_path)


def qsfa_run_npu(test_data, device_id=0, result_path=Path("result.xlsx")):
    # 回放 NPU 执行，并根据 cpu_output 决定是否做精度对比。
    try:
        npu_result, cpu_result = kv_quant_sparse_flash_attention_process.test_qsfa_process_ci(
            test_data, device_id=device_id
        )

        if cpu_result is None:
            print(npu_result)
            print("CPU golden 不可用，当前用例仅验证 NPU 连通性，跳过精度对比。")
            result = "SkipCompare"
            fulfill_percent = 100.0
        else:
            result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result)
    except Exception as error:
        print(f"NPU ERROR: {error}")
        result = "NPU ERROR"
        fulfill_percent = 0.0

    save_result(test_data["params"], result, fulfill_percent, result_path=result_path)
    if result == "NPU ERROR":
        pytest.fail(f"用例执行失败: {test_data['Testcase_Name']}")
