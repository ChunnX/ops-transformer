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
import generate_tensor_data
import tools
import traceback
import json


STR_MAP_DICT = {
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
}


def _normalize_numeric_value(value):
    # Excel 混合空值/整数列时，pandas 容易把整列抬升成 float，这里统一回收成 int。
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
    # 将 Excel 中的字符串单元格转成 pytest 框架需要的 Python 类型。
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
        "B", "T", "T2", "S1", "S2", "N1", "N2", "D", "K",
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
            current_params.get("T", [None]),
            current_params.get("T2", [None]),
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
    return param_combination_set


def generate_input_tensors(params, json_data):
    input_tensor_dict = {}
    input_tensor_list_fp32 = list()

    shape_keys = ['query', 'key', 'value', 'sparse_indices', 'block_table', 'query_cache', 'key_cache', 'value_cache', 'query_rope', 'key_rope', 'dequant_scale', 'v_dequant_scale']
    for i, key in enumerate(shape_keys):
        input_data_i = generate_tensor_data.gen_tensor_data(params, key, json_data)
        tools.convert_tensor_data(input_data_i, f"input{i}", params['dtype_input'][key], params)
        input_tensor_dict[key] = input_data_i
        if params['dtype_input'][key] == 'fp16':
            input_tensor_list_fp32.append(input_data_i.astype(np.float32))

    return input_tensor_dict, input_tensor_list_fp32


def sub_process_cpu(params, json_data):
    case_name = params["case_name"]
    result, error_percent, max_error = "Pass", 1.0, 0.0
    input_tensor_dict, input_tensor_list_fp32 = generate_input_tensors(params, json_data)

    try:
        out_data = route(input_tensor_dict, params, "cpu", json_data)
        return out_data, input_tensor_dict, params
    except Exception as e:
        print(f"[ERROR] sub_process_cpu failed: {e}")
        return None, input_tensor_dict, params


def route(input_tensor_dict, params, device, json_data):
    for key in input_tensor_dict:
        if input_tensor_dict[key].dtype == np.float16 or str(input_tensor_dict[key].dtype) == "bfloat16":
            input_tensor_dict[key] = input_tensor_dict[key].astype(np.float32)
    return kv_quant_sparse_flash_attention_golden.aclnn_op_func(input_tensor_dict, params, json_data)


def convert_param_combination_to_cs_format(param_combination):
    layout_query = param_combination["layout_query"]
    layout_kv = param_combination["layout_kv"]
    if (layout_query == "TND"):
        T = param_combination["T"]
    B = param_combination["B"]
    S1 = param_combination["S1"]
    if (layout_kv == "TND"):
        T2 = param_combination["T2"]
    S2 = param_combination["S2"]
    N1 = param_combination["N1"]
    N2 = param_combination["N2"]
    D = param_combination["D"]
    K = param_combination["K"]
    q_type = param_combination["q_type"]
    kv_dtype = param_combination["kv_dtype"]
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
    actual_seq_q = param_combination.get("actual_seq_q") or [S1]
    actual_seq_kv = param_combination.get("actual_seq_kv") or [S2]
    sparse_blockcount = int(K / sparse_block_size)
    testcase_prefix = param_combination.get("Testcase_Prefix") or "kvQuantSparseFlashAttn"
    testcase_number = param_combination.get("Testcase_Number") or 1
    
    q_type_str = "BF16" if q_type == torch.bfloat16 else "FP16"
    testcase_name = f"{testcase_prefix}_{layout_query}_{layout_kv}_{q_type_str}_{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{testcase_number:06d}"
    
    if layout_kv == "PA_BSND" and block_num is None:
        max_kv = max(actual_seq_kv) if actual_seq_kv else S2
        block_num = math.ceil(max_kv / block_size) * B
    if layout_kv == "PA_BSND":
        block_num_sum = 0
        for length in actual_seq_kv:
            block_num_sum = block_num_sum + math.ceil(length / block_size)
    
    if q_type == torch.bfloat16:
        q_dtype_str = "bf16"
    elif q_type == torch.float16:
        q_dtype_str = "fp16"
    else:
        q_dtype_str = "fp32"
    
    if kv_dtype is None or str(kv_dtype) == "float8_e4m3fn":
        kv_dtype_str = "float8_e4m3fn"
    else:
        kv_dtype_str = str(kv_dtype)
    if (layout_kv == 'PA_BSND'):
        if (layout_query == 'BSND'):
            shape_input = {
                'query': [B, S1, N1, D],
                'key': [B, S2, N2, D],
                'value': [B, S2, N2, D],
                'sparse_indices': [B, S1, N2, sparse_blockcount],
                'block_table': [B, block_num],
                'query_cache': [B, S1, N1, D + rope_head_dim],
                'key_cache': [block_num_sum, block_size, N2, D + rope_head_dim * 2 + D // tile_size * 4],
                'value_cache': [block_num_sum, block_size, N2, D + rope_head_dim * 2 + D // tile_size * 4],
                'query_rope': [B, S1, N1, rope_head_dim],
                'key_rope': [B, S2, N2, rope_head_dim],
                'dequant_scale': [B, S2, N2, D // tile_size],
                'v_dequant_scale': [B, S2, N2, D // tile_size],
            }
        elif (layout_query == 'TND'):
            shape_input = {
                'query': [T, N1, D],
                'key': [B, S2, N2, D],
                'value': [B, S2, N2, D],
                'sparse_indices': [T, N2, sparse_blockcount],
                'block_table': [B, block_num],
                'query_cache': [T, N1, D + rope_head_dim],
                'key_cache': [block_num_sum, block_size, N2, D + rope_head_dim * 2 + D // tile_size * 4],
                'value_cache': [block_num_sum, block_size, N2, D + rope_head_dim * 2 + D // tile_size * 4],
                'query_rope': [T, N1, rope_head_dim],
                'key_rope': [B, S2, N2, rope_head_dim],
                'dequant_scale': [B, S2, N2, D // tile_size],
                'v_dequant_scale': [B, S2, N2, D // tile_size],
            }
        else:
            print("Unsupported layout_query: ", layout_query)
    elif (layout_kv == 'TND'):
        shape_input = {
            'query': [T, N1, D],
            'key': [T2, N2, D],
            'value': [T2, N2, D],
            'sparse_indices': [T, N2, sparse_blockcount],
            'block_table': [B],
            'query_cache': [T, N1, D + rope_head_dim],
            'key_cache': [T2, N2, D + rope_head_dim * 2 + D // tile_size * 4],
            'value_cache': [T2, N2, D + rope_head_dim * 2 + D // tile_size * 4],
            'query_rope': [T, N1, rope_head_dim],
            'key_rope': [T2, N2, rope_head_dim],
            'dequant_scale': [T2, N2, D // tile_size],
            'v_dequant_scale': [T2, N2, D // tile_size],
        }
    elif (layout_kv == 'BSND'):
        shape_input = {
            'query': [B, S1, N1, D],
            'key': [B, S2, N2, D],
            'value': [B, S2, N2, D],
            'sparse_indices': [B, S1, N2, sparse_blockcount],
            'block_table': [B],
            'query_cache': [B, S1, N1, D + rope_head_dim],
            'key_cache': [B, S2, N2, D + rope_head_dim * 2 + D // tile_size * 4],
            'value_cache': [B, S2, N2, D + rope_head_dim * 2 + D // tile_size * 4],
            'query_rope': [B, S1, N1, rope_head_dim],
            'key_rope': [B, S2, N2, rope_head_dim],
            'dequant_scale': [B, S2, N2, D // tile_size],
            'v_dequant_scale': [B, S2, N2, D // tile_size],
        }
    else:
        print("Unsupported layout_kv: ", layout_kv)
    dtype_input = {
        'query': q_dtype_str,
        'key': kv_dtype_str,
        'value': kv_dtype_str,
        'sparse_indices': "int32",
        'block_table': "int32",
        'query_cache': q_dtype_str,
        'key_cache': kv_dtype_str,
        'value_cache': kv_dtype_str,
        'query_rope': q_dtype_str,
        'key_rope': q_dtype_str,
        'dequant_scale': "fp32",
        'v_dequant_scale': "fp32",
    }
    
    range_input = {
        'query': [2, 10],
        'key': [-100, 100.0],
        'value': [-100.0, 100.0],
        'sparse_indices': [-10, 10],
        'block_table': [0, 1],
        'query_cache': [-10, 10],
        'key_cache': [-10.0, 10.0],
        'value_cache': [-10.0, 10.0],
        'query_rope': [-1, 1],
        'key_rope': [-10.0, -2],
        'dequant_scale': [0, 1],
        'v_dequant_scale': [0, 1],
    }
    
    params = {
        "action_type": "bm",
        "case_name": testcase_name,
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "actualseqlengths": actual_seq_q,
        "actualseqlengthskv": actual_seq_kv,
        "scalevalue": scale_value,
        "sparsemode": sparse_mode,
        "sparse_blocksize": sparse_block_size,
        "shape_input": shape_input,
        "dtype_input": dtype_input,
        "range_input": range_input,
        "dtype_output": [q_dtype_str],
        "shape_output": [[B, S1, N1, D]],
        "tile_size": tile_size,
        "rope_head_dim": rope_head_dim,
        "key_quant_mode": key_quant_mode,
        "value_quant_mode": value_quant_mode,
        "attention_mode": attention_mode,
        "quant_scale_repo_mode": quant_scale_repo_mode,
        "block_size": block_size,
        "k_antiquant_mode": key_quant_mode,
        "v_antiquant_mode": value_quant_mode,
        "antiquant_scale": 1,
        "antiquant_offset": 0,
    }
    
    json_data = {
        "op_name": "aclnnKvQuantSparseFlashAttention",
        "input": []
    }
    shape_keys = ['query', 'key', 'value', 'sparse_indices', 'block_table', 'query_cache', 'key_cache', 'value_cache', 'query_rope', 'key_rope', 'dequant_scale', 'v_dequant_scale']
    for key in shape_keys:
        json_data["input"].append({
            "name": key,
            "type": "tensor",
            "shape": shape_input[key],
            "dtype": dtype_input[key],
            "acl_format": "nd",
            "data_distribution": "uniform",
            "data_range": range_input[key],
        })
    json_data["output"] = [{
        "name": "output0",
        "type": "tensor",
        "shape": [B, S1, N1, D],
        "dtype": q_dtype_str,
        "acl_format": "nd",
    }]
    
    return params, json_data



def qsfa(case_id, param_combination, pt_save_path, device_id, run_npu, save_pt, result_path):
    # single/batch 共用入口：负责参数整理、校验、构造输入和可选执行。
    testcase_prefix = param_combination.get("Testcase_Prefix") or "kvQuantSparseFlashAttn"
    testcase_number = param_combination.get("Testcase_Number") or case_id
    layout_query = param_combination["layout_query"]
    layout_kv = param_combination["layout_kv"]
    q_type = param_combination["q_type"]
    kv_dtype = param_combination["kv_dtype"]
    B = param_combination["B"]
    S1 = param_combination["S1"]
    S2 = param_combination["S2"]
    N1 = param_combination["N1"]
    N2 = param_combination["N2"]
    D = param_combination["D"]
    K = param_combination["K"]
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
        f"{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{testcase_number:06d}"
    )

    if layout_kv == "PA_BSND" and block_num is None:
        # 未显式给出 block_num 时，按当前实际 kv 长度自动推导。
        max_kv = max(actual_seq_kv) if actual_seq_kv else S2
        block_num = math.ceil(max_kv / block_size) * B

    params = (
        testcase_name, layout_query, layout_kv, q_type, kv_dtype,
        B, S1, S2, N1, N2, D, K,
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
