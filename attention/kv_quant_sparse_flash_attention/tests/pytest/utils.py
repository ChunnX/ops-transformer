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
import datetime
import hashlib
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import torch
from ml_dtypes import float8_e4m3fn
import check_valid_param
import result_compare_method
import traceback


STR_MAP_DICT = {
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
}

INPUT_TENSOR_NAMES = [
    "query",
    "key",
    "value",
    "sparse_indices",
    "block_table",
    "actual_seq_lengths_query",
    "actual_seq_lengths_kv",
    "key_dequant_scale",
    "value_dequant_scale",
]

INPUT_ATTR_NAMES = [
    "scale_value",
    "layout_query",
    "layout_kv",
    "sparse_mode",
    "key_quant_mode",
    "value_quant_mode",
    "sparse_block_size",
    "attention_mode",
    "quant_scale_repo_mode",
    "tile_size",
    "rope_head_dim",
    "kv_dtype",
    "pre_tokens",
    "next_tokens",
]


def numpy_to_torch(np_array, device="cpu"):
    dtype_str = str(np_array.dtype)
    if dtype_str == "uint64":
        torch_tensor = torch.from_numpy(np_array.astype(np.int64)).to(torch.int64)
    elif dtype_str == "bfloat16":
        torch_tensor = torch.from_numpy(np_array.astype(np.float32)).to(torch.bfloat16)
    elif dtype_str == "float8_e4m3fn":
        torch_tensor = torch.from_numpy(np_array.astype(np.float32)).to(torch.float8_e4m3fn)
    elif dtype_str == "float8_e5m2":
        torch_tensor = torch.from_numpy(np_array.astype(np.float32)).to(torch.float8_e5m2)
    elif dtype_str == "float16" or dtype_str == "np.float16":
        torch_tensor = torch.from_numpy(np_array.astype(np.float32)).to(torch.float16)
    else:
        torch_tensor = torch.from_numpy(np_array)
    
    if device != "cpu":
        torch_tensor = torch_tensor.to(device)
    return torch_tensor


def get_torch_dtype(type_str):
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
    }
    return dtype_map.get(type_str, torch.float32)


def print_log(data=None, level='INFO'):
    print("[%s] [%s]-%s:%s - %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                    str(sys._getframe().f_back.f_lineno).zfill(4), data))


def get_np_dtype(type_str):
    type_dict = {
        'fp64': np.float64, 'fp32': np.float32, 'fp16': np.float16,
        'int64': np.int64, 'int32': np.int32, 'int16': np.int16, 'int8': np.int8,
        'uint64': np.uint64, 'uint32': np.uint32, 'uint16': np.uint16, 'uint8': np.uint8,
        'bool': np.bool_, 'complex64': np.complex64, 'complex128': np.complex128,
        'complex32': np.float16,
        'bf16': tf.bfloat16.as_numpy_dtype,
        'bfloat16': tf.bfloat16.as_numpy_dtype,
        'float4_e2m1': np.uint8,
        'float4_e1m2': np.uint8,
        'float8_e8m0': np.uint8,
        'hifloat8': np.uint8,
        'fp4_e1m2': np.uint8,
        "fp4_e2m1": np.uint8,
        "fp8_e8m0": np.uint8,
        "float32": np.float32,
        "float16": np.float16,
        "qint8": np.int8,
        "qint32": np.int32,
        "quint8": np.uint8,
        "qint16": np.int16,
        "uint1": np.uint8,
        "quint16": np.uint16,
        "fp4_e2m1_as_fp32": np.float32
    }
    if type_str == "int4":
        from ml_dtypes import int4
        return int4
    elif type_str == "float8_e5m2":
        from ml_dtypes import float8_e5m2
        return float8_e5m2
    elif type_str == "float8_e4m3fn":
        return float8_e4m3fn
    else:
        return type_dict[type_str]


def get_str_dtype(type_str):
    type_dict = {
        torch.float32: 'fp32', torch.float16: 'fp16', torch.float64: 'fp64',
        torch.int8: 'int8', torch.int16: 'int16', torch.int32: 'int32', torch.int64: 'int64',
        torch.uint8: 'uint8', torch.bool: 'bool', torch.complex64: 'complex64',
        torch.complex128: 'complex128', torch.bfloat16: 'bf16'
    }
    return type_dict[type_str]


def get_tensor_dtype(tensor, input_dtype):
    if input_dtype:
        tensor_dtype = input_dtype
    else:
        if torch.is_tensor(tensor):
            tensor_dtype = get_str_dtype(tensor.dtype)
        else:
            tensor_dtype = str(tensor.dtype)
    return tensor_dtype


def dump_file(data_pool, data_path, type_str, params=None):
    KNOW_NP_DTYPES = [
        'fp64', 'fp32', 'fp16', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'bool',
        'complex64', 'complex128', 'complex32', 'bf16', 'bfloat16',
        'fp8_e8m0', 'float32', 'float16', 'qint8', 'qint32', 'quint8', 'qint16',
        "uint1", "quint16", 'fp4_e2m1_as_fp32',
    ]
    dump_shape = None
    is_tensor = False

    np_type = data_pool.dtype
    dump_np_data = None
    if len(np_type) > 1:
        print_log(f"[dump_file] input dtype {np_type} is stuct ndarray, will directly save to file {data_path}")
        dump_np_data = data_pool
    elif type_str.lower() in KNOW_NP_DTYPES:
        np_dtype = get_np_dtype(type_str.lower())
        dump_np_data = np.array(data_pool).astype(np_dtype)
    elif type_str.lower() in ['int4', 'float4_e2m1', 'float4_e1m2', 'fp4_e1m2', 'fp4_e2m1']:
        data_pool_np = np.array(data_pool)
        if not is_tensor:
            print_log(f"[dump_file]numpy for {type_str} packed to uint8")
            data_pool_np = pack_int4(data_pool_np)
        dump_np_data = np.array(data_pool_np)
    elif type_str.lower() in ['float8_e5m2', 'float8_e4m3fn', 'float8_e8m0', 'hifloat8']:
        print_log(f"[dump_file]for {type_str} will directly save to file")
        dump_np_data = np.array(data_pool)
    else:
        print_log(f"[dump_file]WARN UNKNOW TYPE {type_str},will directly save to file")
        dump_np_data = np.array(data_pool)
    if dump_np_data is not None:
        dump_shape = list(dump_np_data.shape)
        dump_np_data.tofile(data_path)
        file_name = os.path.basename(data_path)
        if "cpu_output" in file_name and isinstance(params, type({})):
            if "shape_output_tmp" not in params.keys():
                params["shape_output_tmp"] = []
            params["shape_output_tmp"].append(dump_shape)
            shape_info = {"shape_output": params["shape_output_tmp"]}
            modify_cs_info(shape_info, params)
        print_log(f"[dump_file]save data to {data_path} with shape:{dump_shape} dtype:{type_str}")


def pack_int4(data):
    return data


def safe_write_fp8(tensor, file_path):
    tensor.detach().cpu().numpy().tofile(file_path)


def modify_cs_info(shape_info, params):
    pass


def modify_alcnn_input_file(ids: int,
                            origin_index: list,
                            type: str,
                            mode: str,
                            tensors,
                            params: dict,
                            keepcs=None,
                            data_range=None,
                            data_dtype=None):
    json_path = os.path.join(params['case_path'], params['case_name'] + ".json")
    with open(json_path, "r") as f:
        json_data = json.load(f)

    cs_path = os.path.join(params['case_path'], params['case_name'] + ".cs")
    with open(cs_path, "r") as f:
        cs_data = json.load(f)

    input_dic = dict()
    input_dic["name"] = "input" + str(ids)
    input_dic["type"] = type
    input_dic["acl_format"] = "nd"
    input_dic["data_distribution"] = "uniform"

    if tensors is not None:
        for i in json_data["input"]:
            if i["name"] == "input" + str(origin_index[0]):
                input_dic["bin_path"] = i["bin_path"]
                input_dic["acl_format"] = i["acl_format"]
                json_data["input"].remove(i)

        input_dic["data_range"] = data_range or [-10.0, 10.0]
        input_dic["shape"] = tensors.shape
        input_dic["dtype"] = get_tensor_dtype(tensors, data_dtype)
        input_file_i = os.path.join(params['case_path'], input_dic["bin_path"])
        if torch.is_tensor(tensors):
            if input_dic["dtype"] == "bf16":
                np.array(tensors.float().detach().cpu().numpy().astype(tf.bfloat16.as_numpy_dtype)).tofile(
                    input_file_i)
            elif input_dic["dtype"] == "float8_e4m3fn":
                safe_write_fp8(tensors, input_file_i)
            else:
                tensors.detach().cpu().numpy().tofile(input_file_i)
        else:
            dump_file(tensors, input_file_i, input_dic["dtype"], params)

        json_data["input"].insert(ids, input_dic)
    else:
        for i in json_data["input"]:
            if i["name"] == "input" + str(origin_index[0]):
                i["name"] == "input" + str(ids)

    cs_dtype_input = []
    cs_format_input = []
    cs_range_input = []
    cs_shape_input = []
    cs_type_input = []
    cs_bin_path_input = []

    for i in json_data["input"]:
        if re.search(r'input\d+', i['name']):
            exist = True
            if i["type"] == 'tensor_list':
                for j in range(len(i["shapes"])):
                    cs_dtype_input.append(i["dtype"])
                    cs_format_input.append(i["acl_format"])
                    cs_range_input.append(i["data_range"][j])
                    cs_shape_input.append(i["shapes"][j])
                    cs_type_input.append('tensor')
                    cs_bin_path_input.append(i['bin_paths'][j])
            elif i["type"] == 'tensor':
                cs_dtype_input.append(i["dtype"])
                cs_format_input.append(i["acl_format"])
                cs_range_input.append(i["data_range"])
                cs_shape_input.append(i["shape"])
                cs_type_input.append(i["type"])
                cs_bin_path_input.append(i['bin_path'])

    cs_data["dtype_input"] = cs_dtype_input
    cs_data["format_input"] = cs_format_input
    cs_data["range_input"] = cs_range_input
    cs_data["shape_input"] = cs_shape_input
    cs_data["type_input"] = cs_type_input
    cs_data["bin_path_input"] = cs_bin_path_input
    return tensors


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
    row_data = {
        "Testcase_Name": params.get("case_name", params.get("Testcase_Name", "")),
        "layout_query": params.get("layout_query", ""),
        "layout_kv": params.get("layout_kv", ""),
        "q_type": str(params.get("q_type", "")),
        "kv_dtype": params.get("kv_dtype", ""),
        "B": params.get("B", 0),
        "S1": params.get("S1", 0),
        "S2": params.get("S2", 0),
        "N1": params.get("N1", 0),
        "N2": params.get("N2", 0),
        "D": params.get("D", 0),
        "K": params.get("K", 0),
        "scale_value": params.get("scale_value", params.get("scalevalue", 0)),
        "sparse_mode": params.get("sparse_mode", params.get("sparsemode", 0)),
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


def get_case_output_bin_path(params):
    # 统一管理测试用例输出 bin 路径，避免各流程各自拼接文件名。
    return os.path.join(params["case_path"], params["case_name"] + "_output_0.bin")


def calculate_file_md5(file_path):
    # 对 bin 文件做 md5，供重构前后黑盒回归复用。
    md5 = hashlib.md5()
    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compare_md5(expected_md5, actual_md5):
    # 返回 md5 比较结果以及便于日志输出的摘要信息。
    return {
        "match": expected_md5 == actual_md5,
        "expected_md5": expected_md5,
        "actual_md5": actual_md5,
    }


def compare_output_bin_md5(expected_file_path, actual_file_path):
    # 比较两个输出 bin 的 md5，适合做重构前后基线回归。
    expected_md5 = calculate_file_md5(expected_file_path)
    actual_md5 = calculate_file_md5(actual_file_path)
    result = compare_md5(expected_md5, actual_md5)
    result["expected_file_path"] = expected_file_path
    result["actual_file_path"] = actual_file_path
    return result


BASELINE_MD5_FILE = "baseline_md5.json"


def is_baseline_update_enabled():
    # baseline 默认只读，只有显式设置环境变量才允许创建或覆盖。
    return os.environ.get("UPDATE_BASELINE", "").strip().lower() in ("1", "true", "yes", "on")


def load_baseline_md5():
    # 加载所有用例的 baseline MD5，不存在时返回空字典。
    if os.path.exists(BASELINE_MD5_FILE):
        with open(BASELINE_MD5_FILE, "r") as f:
            return json.load(f)
    return {}


def save_baseline_md5(testcase_name, md5_value):
    # 将单个用例的 MD5 写入 baseline 文件。
    baseline_data = load_baseline_md5()
    baseline_data[testcase_name] = md5_value
    with open(BASELINE_MD5_FILE, "w") as f:
        json.dump(baseline_data, f, indent=4)
    print_log(f"baseline MD5 saved: {testcase_name} -> {md5_value}")


def verify_md5_with_baseline(testcase_name, actual_md5):
    # 对比实际 MD5 与 baseline，返回对比结果字典。
    baseline_data = load_baseline_md5()
    expected_md5 = baseline_data.get(testcase_name)
    if expected_md5 is None:
        return {
            "match": True,
            "status": "no_baseline",
            "message": f"No baseline MD5 for {testcase_name}, skipping verification",
        }
    match = expected_md5 == actual_md5
    return {
        "match": match,
        "status": "verified",
        "expected_md5": expected_md5,
        "actual_md5": actual_md5,
        "message": f"MD5 {'MATCH' if match else 'MISMATCH'}: expected={expected_md5}, actual={actual_md5}",
    }


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
    
    testcase_prefix = param_combination.get("Testcase_Prefix") or "kvQuantSparseFlashAttn"
    testcase_number = param_combination.get("Testcase_Number") or 1
    
    q_type_str = "BF16" if q_type == torch.bfloat16 else "FP16"
    testcase_name = f"{testcase_prefix}_{layout_query}_{layout_kv}_{q_type_str}_{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{testcase_number:06d}"
    
    if layout_kv == "PA_BSND" and block_num is None:
        max_kv = max(actual_seq_kv) if actual_seq_kv else S2
        block_num = math.ceil(max_kv / block_size) * B
    
    sparse_blockcount = 16
    
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
    
    shape_input = [
        [B, S1, N1, D],
        [block_num, block_size, N2, D],
        [block_num, block_size, N2, D],
        [B, S1, N2, sparse_blockcount],
        [B, block_num],
        [B, S1, N1, D + rope_head_dim],
        [block_num, block_size, N2, D + rope_head_dim + D // tile_size],
        [block_num, block_size, N2, D + rope_head_dim + D // tile_size],
        [B, S1, N1, rope_head_dim],
        [block_num, block_size, N2, rope_head_dim],
        [block_num, block_size, N2, D // tile_size],
        [block_num, block_size, N2, D // tile_size],
    ]
    
    dtype_input = [
        q_dtype_str,
        kv_dtype_str,
        kv_dtype_str,
        "int32",
        "int32",
        q_dtype_str,
        kv_dtype_str,
        kv_dtype_str,
        q_dtype_str,
        q_dtype_str,
        "fp32",
        "fp32",
    ]
    
    range_input = [
        [-10.0, 100.0],
        [-5.0, 10.0],
        [-5.0, 10.0],
        [-1, max(actual_seq_kv) if actual_seq_kv else S2],
        [-1, block_num - 1 if block_num else 0],
        [-10.0, 100.0],
        [-10.0, 10.0],
        [-5.0, 10.0],
        [-10.0, 10.0],
        [-10.0, 10.0],
        [-100.0, 100.0],
        [-100.0, 100.0],
    ]
    
    type_input = ["tensor"] * 12
    
    params = {
        "action_type": "bm",
        "case_path": "./",
        "case_name": testcase_name,
        "bin_dir": "./",
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "actualseqlengths": actual_seq_q,
        "actualseqlengthskv": actual_seq_kv,
        "scalevalue": scale_value,
        "sparsemode": sparse_mode,
        "sparse_blocksize": sparse_block_size,
        "flaglist": [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        "shape_input": shape_input,
        "dtype_input": dtype_input,
        "range_input": range_input,
        "type_input": type_input,
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
    for i in range(12):
        json_data["input"].append({
            "name": f"input{i}",
            "type": "tensor",
            "shape": shape_input[i],
            "dtype": dtype_input[i],
            "acl_format": "nd",
            "data_distribution": "uniform",
            "data_range": range_input[i],
            "bin_path": f"{testcase_name}_input_{i}.bin"
        })
    json_data["output"] = [{
        "name": "output0",
        "type": "tensor",
        "shape": [B, S1, N1, D],
        "dtype": q_dtype_str,
        "acl_format": "nd",
        "bin_path": f"{testcase_name}_output_0.bin"
    }]
    
    json_path = os.path.join(params["case_path"], testcase_name + ".json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    
    cs_path = os.path.join(params["case_path"], testcase_name + ".cs")
    with open(cs_path, "w") as f:
        json.dump(params, f, indent=4)
    
    return params









