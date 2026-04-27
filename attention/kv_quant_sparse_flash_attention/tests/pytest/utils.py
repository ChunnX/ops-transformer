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
import traceback
import tensorflow as tf
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


def load_paramset(paramset_file):
    module = __import__(paramset_file)
    return module.ENABLED_PARAMS


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
        "layout_query", "layout_kv", "q_type", "kv_type",
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
    """保存测试结果，列与 example.xlsx 一致，结果文件可直接用于批量生成 pt。"""
    row_data = {
        "Testcase_Prefix": params.get("Testcase_Prefix", "kvQuantSparseFlashAttn"),
        "layout_query": params.get("layout_query"),
        "layout_kv": params.get("layout_kv"),
        "q_type": str(params.get("q_type")),
        "kv_type": str(params.get("kv_type")) if params.get("kv_type") is not None else None,
        "B": params.get("B"),
        "T": params.get("T"),
        "T2": params.get("T2"),
        "S1": params.get("S1"),
        "S2": params.get("S2"),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K": params.get("K"),
        "scale_value": params.get("scalevalue"),
        "key_quant_mode": params.get("key_quant_mode"),
        "value_quant_mode": params.get("value_quant_mode"),
        "sparse_block_size": params.get("sparse_blocksize"),
        "tile_size": params.get("tile_size"),
        "rope_head_dim": params.get("rope_head_dim"),
        "sparse_mode": params.get("sparsemode"),
        "attention_mode": params.get("attention_mode"),
        "quant_scale_repo_mode": params.get("quant_scale_repo_mode"),
        "block_size": params.get("block_size"),
        "block_num": params.get("block_num"),
        "actual_seq_q": str(params.get("actual_seq_q")) if params.get("actual_seq_q") is not None else None,
        "actual_seq_kv": str(params.get("actual_seq_kv")) if params.get("actual_seq_kv") is not None else None,
        "result": result,
        "fulfill_percent": fulfill_percent,
    }

    if result_path.exists():
        dataframe = pd.read_excel(result_path)
        dataframe = pd.concat([dataframe, pd.DataFrame([row_data])], ignore_index=True)
    else:
        dataframe = pd.DataFrame([row_data])
    dataframe.to_excel(result_path, index=False)


def combin_params(enabled_params, pytest_paramset=True):
    param_combination_set = []
    param_names = [
        "Testcase_Prefix", "Testcase_Number",
        "layout_query", "layout_kv", "q_type", "kv_type",
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
            current_params.get("kv_type"),
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
    kv_type = param_combination["kv_type"]
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
    
    if layout_kv == "PA_BSND":
        block_num_per_batch = math.ceil(S2 / block_size)
    if layout_kv == "PA_BSND" and block_num is None:
        block_num = 0
        for length in actual_seq_kv:
            block_num = block_num + math.ceil(length / block_size)
    
    if q_type == torch.bfloat16:
        q_dtype_str = "bf16"
    elif q_type == torch.float16:
        q_dtype_str = "fp16"
    else:
        q_dtype_str = "fp32"
    
    if kv_type is None or str(kv_type) == "float8_e4m3fn":
        kv_dtype_str = "float8_e4m3fn"
    else:
        kv_dtype_str = str(kv_type)
    if (layout_kv == 'PA_BSND'):
        if (layout_query == 'BSND'):
            shape_input = {
                'query': [B, S1, N1, D],
                'key': [B, S2, N2, D],
                'value': [B, S2, N2, D],
                'sparse_indices': [B, S1, N2, sparse_blockcount],
                'block_table': [B, block_num_per_batch],
                'query_cache': [B, S1, N1, D + rope_head_dim],
                'key_cache': [block_num, block_size, N2, D + rope_head_dim * 2 + D // tile_size * 4],
                'value_cache': [block_num, block_size, N2, D + rope_head_dim * 2 + D // tile_size * 4],
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
                'block_table': [B, block_num_per_batch],
                'query_cache': [T, N1, D + rope_head_dim],
                'key_cache': [block_num, block_size, N2, D + rope_head_dim * 2 + D // tile_size * 4],
                'value_cache': [block_num, block_size, N2, D + rope_head_dim * 2 + D // tile_size * 4],
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
        # 原始参数，用于结果保存（与 example.xlsx 列对齐）
        "Testcase_Prefix": testcase_prefix,
        "q_type": q_type,
        "kv_type": kv_type,
        "B": B,
        "T": param_combination.get("T"),
        "T2": param_combination.get("T2"),
        "S1": S1,
        "S2": S2,
        "N1": N1,
        "N2": N2,
        "D": D,
        "K": K,
        "block_num": block_num,
        "actual_seq_q": actual_seq_q,
        "actual_seq_kv": actual_seq_kv,
    }
    
    return params


def get_np_dtype(type_str):
    type_dict = {
        'fp32': np.float32, 'fp16': np.float16,
        'int32': np.int32, 'int8': np.int8,
        'uint8': np.uint8,
        'bf16': tf.bfloat16.as_numpy_dtype,
        'bfloat16': tf.bfloat16.as_numpy_dtype,
        'float32': np.float32,
        'float16': np.float16,
        'hifloat8': np.uint8,
    }
    if type_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    else:
        return type_dict[type_str]


def convert_tensor_data(data_pool, data_path, type_str, params=None):
    KNOW_NP_DTYPES = [
        'fp32', 'fp16', 'int32', 'int8', 'uint8',
        'bf16', 'bfloat16',
        'float32', 'float16',
    ]

    np_type = data_pool.dtype
    dump_np_data = None
    if len(np_type) > 1:
        dump_np_data = data_pool
    elif type_str.lower() in KNOW_NP_DTYPES:
        np_dtype = get_np_dtype(type_str.lower())
        dump_np_data = np.array(data_pool).astype(np_dtype)
    elif type_str.lower() in ['float8_e4m3fn', 'hifloat8']:
        dump_np_data = np.array(data_pool)
    else:
        dump_np_data = np.array(data_pool)
    if dump_np_data is not None:
        dump_shape = list(dump_np_data.shape)
        file_name = os.path.basename(data_path)
        if "cpu_output" in file_name and isinstance(params, type({})):
            if "shape_output_tmp" not in params.keys():
                params["shape_output_tmp"] = []
            params["shape_output_tmp"].append(dump_shape)
            shape_info = {"shape_output": params["shape_output_tmp"]}
            modify_cs_info(shape_info, params)
    return dump_np_data


def get_str_dtype(type_str):
    type_dict = {
        torch.float32: 'fp32', torch.float16: 'fp16',
        torch.int8: 'int8', torch.int32: 'int32',
        torch.uint8: 'uint8', torch.bfloat16: 'bf16'
    }
    return type_dict[type_str]


def get_torch_dtype(type_str):
    type_dict = {
        'fp32': torch.float32, 'float32': torch.float32,
        'fp16': torch.float16, 'float16': torch.float16,
        'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16,
        'int8': torch.int8,
        'int32': torch.int32,
        'uint8': torch.uint8,
        'float8_e4m3fn': torch.float8_e4m3fn,
        'hifloat8': None,
    }
    return type_dict.get(type_str)


def get_tensor_dtype(tensor, input_dtype):
    if input_dtype:
        tensor_dtype = input_dtype
    else:
        if torch.is_tensor(tensor):
            tensor_dtype = get_str_dtype(tensor.dtype)
        else:
            tensor_dtype = str(tensor.dtype)
    return tensor_dtype


def modify_alcnn_input_file(name: str,
                            tensors,
                            params: dict,
                            data_range=None,
                            data_dtype=None):
    out_tensor = tensors
    if tensors is not None:
        data_range = data_range or [-10.0, 10.0]
        tensor_dtype = get_tensor_dtype(tensors, data_dtype)
        if torch.is_tensor(tensors):
            if tensor_dtype == "bf16":
                out_tensor = np.array(tensors.float().detach().cpu().numpy().astype(tf.bfloat16.as_numpy_dtype))
            else:
                out_tensor = tensors.detach().cpu().numpy()
        else:
            out_tensor = convert_tensor_data(tensors, name, tensor_dtype, params)

        params["shape_input"][name] = list(tensors.shape)
        params["dtype_input"][name] = tensor_dtype
        params["range_input"][name] = data_range

    return out_tensor


def qsfa_run_npu(test_data, testcase_name=None, device_id=0, result_path=None):
    from batch import kv_quant_sparse_flash_attention_process
    params = test_data.get("params")
    if testcase_name:
        params["Testcase_Prefix"] = testcase_name
    cpu_result = test_data["cpu_output"]
    npu_result = kv_quant_sparse_flash_attention_process.call_npu(test_data["input"], params)
    result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result)
    if result_path:
        save_result(params, result, fulfill_percent, result_path)
    return result
