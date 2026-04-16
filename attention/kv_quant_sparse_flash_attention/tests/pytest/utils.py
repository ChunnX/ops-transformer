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
import kv_quant_sparse_flash_attention_cal

STR_MAP_DICT = {
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
}

def pa_to_bsnd(pa_in, block_table, actual_seq_lengths):
    block_num, block_size, n, d = pa_in.shape
    b = len(actual_seq_lengths)
    out = torch.zeros((b, block_num * block_size // b, 1, d)).to(pa_in.dtype)
    for i in range(b):
        for j in range(math.ceil(actual_seq_lengths[i] / block_size)):
            out[i, j * block_size: (j + 1) * block_size, 0, :] = \
                pa_in[block_table[i][j], :, 0, :].reshape(block_size, d)
    return out

def npu_params_to_golden(npu_params):
    """
    将 NPU 参数转换为 kv_quant_sparse_flash_attention_cal.py 需要的格式

    NPU 参数 (来自 torch_npu.npu_kv_quant_sparse_flash_attention):
        query, key, value, sparse_indices
        scale_value, key_quant_mode, value_quant_mode
        key_dequant_scale, value_dequant_scale
        block_table
        actual_seq_lengths_query, actual_seq_lengths_kv
        sparse_block_size, layout_query, layout_kv
        sparse_mode, attention_mode, quant_scale_repo_mode
        tile_size, rope_head_dim
        key_dtype, value_dtype
        pre_tokens, next_tokens

    返回:
        torch_tensor_list: 10个tensor的列表
        params: 配置字典
    """
    query = npu_params["query"]
    key = npu_params["key"]
    value = npu_params["value"]
    sparse_indices = npu_params["sparse_indices"]
    scale_value = npu_params["scale_value"]
    key_quant_mode = npu_params.get("key_quant_mode", "dynamic")
    value_quant_mode = npu_params.get("value_quant_mode", "dynamic")
    key_dequant_scale = npu_params.get("key_dequant_scale")
    value_dequant_scale = npu_params.get("value_dequant_scale")
    block_table = npu_params.get("block_table")
    actual_seq_lengths_query = npu_params.get("actual_seq_lengths_query")
    actual_seq_lengths_kv = npu_params.get("actual_seq_lengths_kv")
    sparse_block_size = npu_params.get("sparse_block_size", 1)
    layout_query = npu_params.get("layout_query", "BSND")
    layout_kv = npu_params.get("layout_kv", "PA_BSND")
    sparse_mode = npu_params.get("sparse_mode", 3)
    attention_mode = npu_params.get("attention_mode", 0)
    quant_scale_repo_mode = npu_params.get("quant_scale_repo_mode", 1)
    tile_size = npu_params.get("tile_size", 128)
    rope_head_dim = npu_params.get("rope_head_dim", 64)

    def _to_numpy(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            try:
                return tensor.detach().cpu().numpy()
            except TypeError:
                if tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                    return tensor.detach().cpu().to(torch.float32).numpy()
                raise
        return tensor

    def _get_shape(tensor):
        if tensor is None:
            return []
        if isinstance(tensor, (torch.Tensor, np.ndarray)):
            return list(tensor.shape)
        return []

    def _get_dtype(tensor):
        if tensor is None:
            return "float32"
        if isinstance(tensor, torch.Tensor):
            dtype_map = {
                torch.float32: "fp32",
                torch.float16: "fp16",
                torch.bfloat16: "bf16",
                torch.int32: "int32",
                torch.int64: "int64",
                torch.int8: "int8",
            }
            return dtype_map.get(tensor.dtype, str(tensor.dtype))
        if isinstance(tensor, np.ndarray):
            dtype_map = {
                np.float32: "fp32",
                np.float16: "fp16",
                np.int32: "int32",
                np.int64: "int64",
                np.int8: "int8",
            }
            return dtype_map.get(tensor.dtype, str(tensor.dtype))
        return "float32"

    q_tensor = _to_numpy(query)
    k_tensor = _to_numpy(key)
    v_tensor = _to_numpy(value)
    sparse_indices_tensor = _to_numpy(sparse_indices)
    block_table_tensor = _to_numpy(block_table)
    k_dequant_scale_tensor = _to_numpy(key_dequant_scale)
    v_dequant_scale_tensor = _to_numpy(value_dequant_scale)

    q_shape = _get_shape(q_tensor)
    k_shape = _get_shape(k_tensor)
    v_shape = _get_shape(v_tensor)
    sparse_indices_shape = _get_shape(sparse_indices_tensor)
    bt_shape = _get_shape(block_table_tensor)

    if layout_kv == "PA_BSND" and block_table_tensor is not None:
        block_size = block_table_tensor.shape[1] if len(block_table_tensor.shape) > 1 else 256
        k_cache_shape = [block_table_tensor.shape[0] * block_size, 1, k_shape[-2], k_shape[-1]]
        v_cache_shape = k_cache_shape
    else:
        max_seq = actual_seq_lengths_kv[0] if actual_seq_lengths_kv else k_shape[1]
        k_cache_shape = [max_seq, 1, k_shape[-2], k_shape[-1]]
        v_cache_shape = k_cache_shape

    torch_tensor_list = [
        q_tensor,
        k_tensor,
        v_tensor,
        sparse_indices_tensor,
        block_table_tensor,
        np.zeros(k_cache_shape, dtype=np.float16),
        np.zeros(v_cache_shape, dtype=np.float16),
    ]

    q_dtype = _get_dtype(q_tensor)
    k_dtype = _get_dtype(k_tensor)
    v_dtype = _get_dtype(v_tensor)
    sparse_indices_dtype = _get_dtype(sparse_indices_tensor)
    bt_dtype = _get_dtype(block_table_tensor)
    k_cache_dtype = k_dtype
    v_cache_dtype = v_dtype

    shape_input = [
        q_shape,
        k_shape,
        v_shape,
        sparse_indices_shape,
        bt_shape,
        k_cache_shape,
        v_cache_shape
    ]

    dtype_input = [
        q_dtype,
        k_dtype,
        v_dtype,
        sparse_indices_dtype,
        bt_dtype,
        k_cache_dtype,
        v_cache_dtype
    ]

    out_shape = q_shape
    out_dtype = q_dtype

    B = q_shape[0] if q_shape else 1
    dtype_output = [out_dtype]
    shape_output = [out_shape, out_shape, out_shape] if out_shape else [[], [], []]

    if sparse_indices is not None and block_table is not None:
        flaglist = "1111"
    elif sparse_indices is not None:
        flaglist = "1110"
    else:
        flaglist = "1110"

    actual_seq_q = actual_seq_lengths_query.tolist() if isinstance(actual_seq_lengths_query, (np.ndarray, torch.Tensor)) else list(actual_seq_lengths_query) if actual_seq_lengths_query is not None else [16]
    actual_seq_kv = actual_seq_lengths_kv.tolist() if isinstance(actual_seq_lengths_kv, (np.ndarray, torch.Tensor)) else list(actual_seq_lengths_kv) if actual_seq_lengths_kv is not None else [32]

    if isinstance(actual_seq_q, torch.Tensor):
        actual_seq_q = actual_seq_q.cpu().numpy().tolist()
    if isinstance(actual_seq_kv, torch.Tensor):
        actual_seq_kv = actual_seq_kv.cpu().numpy().tolist()

    params = {
        'actualseqlengths': actual_seq_q,
        'actualseqlengthskv': actual_seq_kv,
        'scalevalue': scale_value,
        'layout_query': layout_query,
        'layout_kv': layout_kv,
        'sparsemode': sparse_mode,
        'sparse_blocksize': sparse_block_size,
        'shape_input': shape_input,
        'dtype_input': dtype_input,
        'shape_output': shape_output,
        'dtype_output': dtype_output,
        'flaglist': flaglist,
        'return_softmax': False,
        'case_name': "npu_to_golden",
        'action_type': "bm_output_gold",
    }

    return torch_tensor_list, params


def golden_testdata_to_golden_input(test_data):
    """
    将 kv_quant_sparse_flash_attention_golden.py 生成的 test_data 转换为 kv_quant_sparse_flash_attention_cal.py 格式

    test_data 结构:
        {
            "params": params,
            "input": {
                "query": torch.Tensor,
                "key": torch.Tensor,
                "value": torch.Tensor,
                "sparse_indices": torch.Tensor,
                "key_dequant_scale": torch.Tensor,
                "value_dequant_scale": torch.Tensor,
                "block_table": torch.Tensor,
                "actual_seq_lengths_query": torch.Tensor,
                "actual_seq_lengths_kv": torch.Tensor,
                "scale_value": float,
                "key_quant_mode": str,
                "value_quant_mode": str,
                "sparse_block_size": int,
                "layout_query": str,
                "layout_kv": str,
                "sparse_mode": int,
                ...
            },
            "cpu_output": None,
        }

    返回:
        torch_tensor_list: 10个tensor的列表
        params: 配置字典
    """
    params = test_data["params"]
    input_dict = test_data["input"]

    testcase_name = test_data["Testcase_Name"]
    layout_query = input_dict["layout_query"]
    layout_kv = input_dict["layout_kv"]
    scale_value = input_dict["scale_value"]
    sparse_block_size = input_dict["sparse_block_size"]
    sparse_mode = input_dict["sparse_mode"]

    def _to_numpy(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            try:
                return tensor.detach().cpu().numpy()
            except TypeError:
                if tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                    return tensor.detach().cpu().to(torch.float32).numpy()
                raise
        return tensor

    def _get_shape(tensor):
        if tensor is None:
            return []
        if isinstance(tensor, (torch.Tensor, np.ndarray)):
            return list(tensor.shape)
        return []

    def _get_dtype(tensor):
        if tensor is None:
            return "float32"
        dtype_map_torch = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.int8: "int8",
            torch.uint8: "uint8",
        }
        dtype_map_np = {
            np.float32: "fp32",
            np.float16: "fp16",
            np.int32: "int32",
            np.int64: "int64",
            np.int8: "int8",
            np.uint8: "uint8",
        }
        if isinstance(tensor, torch.Tensor):
            return dtype_map_torch.get(tensor.dtype, str(tensor.dtype))
        if isinstance(tensor, np.ndarray):
            return dtype_map_np.get(tensor.dtype, str(tensor.dtype))
        return "float32"

    q_tensor = input_dict["query"]
    sparse_indices_tensor = input_dict["sparse_indices"]
    block_table_tensor = input_dict.get("block_table")
    key_dequant_scale_tensor = input_dict.get("key_dequant_scale")
    value_dequant_scale_tensor = input_dict.get("value_dequant_scale")
    actual_seq_q = input_dict.get("actual_seq_lengths_query")
    actual_seq_kv = input_dict.get("actual_seq_lengths_kv")
    tile_size = input_dict.get("tile_size", 128)
    key = pa_to_bsnd(input_dict["key"], block_table_tensor, actual_seq_kv)
    key_rope = key[..., 512: 512 + 64 * 2].view(q_tensor.dtype)
    key_quant_scale = key[..., 512 + 64 * 2:].view(torch.float32)
    key = key[..., :512].to(torch.float32)

    key_quant_scale = np.repeat(key_quant_scale, repeats=tile_size, axis=-1)
    key = (key * key_quant_scale).to(q_tensor.dtype)
    v_tensor = key
    k_tensor = torch.cat((key, key_rope), axis=-1)
    q_shape = _get_shape(q_tensor)
    k_shape = _get_shape(k_tensor)
    v_shape = _get_shape(v_tensor)
    sparse_indices_shape = _get_shape(sparse_indices_tensor)
    bt_shape = _get_shape(block_table_tensor)

    rope_head_dim = input_dict.get("rope_head_dim", 64)

    if layout_kv == "PA_BSND" and block_table_tensor is not None:
        block_size = params[21] if len(params) > 20 else 256
        k_cache_shape = [block_table_tensor.shape[0] * block_size, 1, k_shape[-2], k_shape[-1]]
        v_cache_shape = k_cache_shape
    else:
        max_seq = k_shape[1] if k_shape else 16
        k_cache_shape = [max_seq, 1, k_shape[-2], k_shape[-1]]
        v_cache_shape = k_cache_shape

    
    torch_tensor_list = [
        q_tensor,
        k_tensor,
        v_tensor,
        sparse_indices_tensor,
        block_table_tensor,
        np.zeros(k_cache_shape, dtype=np.float16) if k_cache_shape else None,
        np.zeros(v_cache_shape, dtype=np.float16) if v_cache_shape else None
    ]

    q_dtype = _get_dtype(q_tensor)
    k_dtype = _get_dtype(k_tensor)
    v_dtype = _get_dtype(v_tensor)
    sparse_indices_dtype = _get_dtype(sparse_indices_tensor)
    bt_dtype = _get_dtype(block_table_tensor)
    k_cache_dtype = k_dtype
    v_cache_dtype = v_dtype

    shape_input = [
        q_shape,
        k_shape,
        v_shape,
        sparse_indices_shape,
        bt_shape,
        k_cache_shape,
        v_cache_shape,
    ]

    dtype_input = [
        q_dtype,
        k_dtype,
        v_dtype,
        sparse_indices_dtype,
        bt_dtype,
        k_cache_dtype,
        v_cache_dtype,
    ]

    out_shape = q_shape
    out_dtype = q_dtype
    dtype_output = [out_dtype]
    shape_output = [out_shape, out_shape, out_shape] if out_shape else [[], [], []]

    if sparse_indices_tensor is not None and block_table_tensor is not None:
        flaglist = "1111"
    elif sparse_indices_tensor is not None:
        flaglist = "1110"
    else:
        flaglist = "1110"


    if actual_seq_q is not None:
        if isinstance(actual_seq_q, torch.Tensor):
            actual_seq_q = actual_seq_q.detach().cpu().numpy()
        if isinstance(actual_seq_q, np.ndarray):
            actual_seq_q = actual_seq_q.tolist()
        if isinstance(actual_seq_q, (list, tuple)):
            actual_seq_q = list(actual_seq_q)
    else:
        actual_seq_q = [16]

    if actual_seq_kv is not None:
        if isinstance(actual_seq_kv, torch.Tensor):
            actual_seq_kv = actual_seq_kv.detach().cpu().numpy()
        if isinstance(actual_seq_kv, np.ndarray):
            actual_seq_kv = actual_seq_kv.tolist()
        if isinstance(actual_seq_kv, (list, tuple)):
            actual_seq_kv = list(actual_seq_kv)
    else:
        actual_seq_kv = [32]

    golden_params = {
        'actualseqlengths': actual_seq_q,
        'actualseqlengthskv': actual_seq_kv,
        'scalevalue': scale_value,
        'layout_query': layout_query,
        'layout_kv': layout_kv,
        'sparsemode': sparse_mode,
        'sparse_blocksize': sparse_block_size,
        'shape_input': shape_input,
        'dtype_input': dtype_input,
        'shape_output': shape_output,
        'dtype_output': dtype_output,
        'flaglist': flaglist,
        'return_softmax': False,
        'case_name': testcase_name,
        'action_type': "bm_output_gold",
        "rope_head_dim": rope_head_dim,
    }

    return torch_tensor_list, golden_params

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
    return param_combination_set


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
    cpu_result = kv_quant_sparse_flash_attention_cal.aclnn_op_func_cpu(torch_tensor_list, golden_params)
    test_data['cpu_output'] = cpu_result
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
