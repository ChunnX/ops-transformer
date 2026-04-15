#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Adapter: 将 NPU 参数格式转换为 golden.py 可用的入参格式
"""

import copy
import time
import math
import os
import random
import numpy as np
import torch


def npu_params_to_golden(npu_params):
    """
    将 NPU 参数转换为 golden.py 需要的格式

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

    q_rope_shape = [q_shape[0], q_shape[1], rope_head_dim] if q_shape else []
    k_rope_shape = [k_shape[0], k_shape[1], rope_head_dim] if k_shape else []
    k_rope_cache_shape = [k_cache_shape[0], 1, k_shape[-2], rope_head_dim] if k_cache_shape else []

    q_rope_tensor = np.zeros(q_rope_shape, dtype=np.float16) if q_rope_shape else np.array([])
    k_rope_tensor = np.zeros(k_rope_shape, dtype=np.float16) if k_rope_shape else np.array([])
    k_rope_cache = np.zeros(k_rope_cache_shape, dtype=np.float16) if k_rope_cache_shape else np.array([])

    torch_tensor_list = [
        q_tensor,
        k_tensor,
        v_tensor,
        sparse_indices_tensor,
        block_table_tensor,
        np.zeros(k_cache_shape, dtype=np.float16),
        np.zeros(v_cache_shape, dtype=np.float16),
        q_rope_tensor,
        k_rope_tensor,
        k_rope_cache,
    ]

    q_dtype = _get_dtype(q_tensor)
    k_dtype = _get_dtype(k_tensor)
    v_dtype = _get_dtype(v_tensor)
    sparse_indices_dtype = _get_dtype(sparse_indices_tensor)
    bt_dtype = _get_dtype(block_table_tensor)
    k_cache_dtype = k_dtype
    v_cache_dtype = v_dtype
    q_rope_dtype = _get_dtype(q_rope_tensor)
    k_rope_dtype = _get_dtype(k_rope_tensor)
    k_rope_cache_dtype = k_dtype

    shape_input = [
        q_shape,
        k_shape,
        v_shape,
        sparse_indices_shape,
        bt_shape,
        k_cache_shape,
        v_cache_shape,
        q_rope_shape,
        k_rope_shape,
        k_rope_cache_shape,
    ]

    dtype_input = [
        q_dtype,
        k_dtype,
        v_dtype,
        sparse_indices_dtype,
        bt_dtype,
        k_cache_dtype,
        v_cache_dtype,
        q_rope_dtype,
        k_rope_dtype,
        k_rope_cache_dtype,
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
    将 kv_quant_sparse_flash_attention_golden.py 生成的 test_data 转换为 golden.py 格式

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

    q_tensor = _to_numpy(input_dict["query"])
    k_tensor = _to_numpy(input_dict["key"])
    v_tensor = _to_numpy(input_dict["value"])
    sparse_indices_tensor = _to_numpy(input_dict["sparse_indices"])
    block_table_tensor = _to_numpy(input_dict.get("block_table"))
    key_dequant_scale_tensor = _to_numpy(input_dict.get("key_dequant_scale"))
    value_dequant_scale_tensor = _to_numpy(input_dict.get("value_dequant_scale"))

    q_shape = _get_shape(q_tensor)
    k_shape = _get_shape(k_tensor)
    v_shape = _get_shape(v_tensor)
    sparse_indices_shape = _get_shape(sparse_indices_tensor)
    bt_shape = _get_shape(block_table_tensor)

    tile_size = input_dict.get("tile_size", 128)
    rope_head_dim = input_dict.get("rope_head_dim", 64)

    if layout_kv == "PA_BSND" and block_table_tensor is not None:
        block_size = params[20] if len(params) > 20 else 256
        k_cache_shape = [block_table_tensor.shape[0] * block_size, 1, k_shape[-2], k_shape[-1]]
        v_cache_shape = k_cache_shape
    else:
        max_seq = k_shape[1] if k_shape else 16
        k_cache_shape = [max_seq, 1, k_shape[-2], k_shape[-1]]
        v_cache_shape = k_cache_shape

    q_rope_shape = [q_shape[0], q_shape[1], rope_head_dim] if q_shape else []
    k_rope_shape = [k_shape[0], k_shape[1], rope_head_dim] if k_shape else []
    k_rope_cache_shape = [k_cache_shape[0], 1, k_shape[-2], rope_head_dim] if k_cache_shape else []

    q_rope_tensor = np.zeros(q_rope_shape, dtype=np.float16) if q_rope_shape else None
    k_rope_tensor = np.zeros(k_rope_shape, dtype=np.float16) if k_rope_shape else None
    k_rope_cache = np.zeros(k_rope_cache_shape, dtype=np.float16) if k_rope_cache_shape else None

    torch_tensor_list = [
        q_tensor,
        k_tensor,
        v_tensor,
        sparse_indices_tensor,
        block_table_tensor,
        np.zeros(k_cache_shape, dtype=np.float16) if k_cache_shape else None,
        np.zeros(v_cache_shape, dtype=np.float16) if v_cache_shape else None,
        q_rope_tensor,
        k_rope_tensor,
        k_rope_cache,
    ]

    q_dtype = _get_dtype(q_tensor)
    k_dtype = _get_dtype(k_tensor)
    v_dtype = _get_dtype(v_tensor)
    sparse_indices_dtype = _get_dtype(sparse_indices_tensor)
    bt_dtype = _get_dtype(block_table_tensor)
    k_cache_dtype = k_dtype
    v_cache_dtype = v_dtype
    q_rope_dtype = _get_dtype(q_rope_tensor)
    k_rope_dtype = _get_dtype(k_rope_tensor)
    k_rope_cache_dtype = k_dtype

    shape_input = [
        q_shape,
        k_shape,
        v_shape,
        sparse_indices_shape,
        bt_shape,
        k_cache_shape,
        v_cache_shape,
        q_rope_shape,
        k_rope_shape,
        k_rope_cache_shape,
    ]

    dtype_input = [
        q_dtype,
        k_dtype,
        v_dtype,
        sparse_indices_dtype,
        bt_dtype,
        k_cache_dtype,
        v_cache_dtype,
        q_rope_dtype,
        k_rope_dtype,
        k_rope_cache_dtype,
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

    actual_seq_q = input_dict.get("actual_seq_lengths_query")
    actual_seq_kv = input_dict.get("actual_seq_lengths_kv")

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
    }

    return torch_tensor_list, golden_params


def golden_output_to_npu(golden_output):
    """
    将 golden.py 的输���转换为 NPU 格式

    golden.py 返回: torch.Tensor (或 list [y, softmax_max, softmax_sum])
    NPU 期望: attention output tensor
    """
    if isinstance(golden_output, list):
        return golden_output[0]
    return golden_output


import importlib.util
import sys

def load_golden_module():
    golden_source = """
import copy
import time
import math
import os
import random
import numpy as np
import torch
"""
    spec = importlib.util.spec_from_file_location("golden", "/home/panjie/workspace/codex/golden.py")
    golden = importlib.util.module_from_spec(spec)
    sys.modules["golden"] = golden
    exec(golden_source, golden.__dict__)
    spec.loader.exec_module(golden)
    return golden


def test_with_golden_testdata():
    """手动构造 test_data 测试 adapter"""
    test_data = {
        "Testcase_Name": "test_BSND_fp16",
        "params": (
            "test_BSND_fp16",
            "BSND", "BSND", torch.float16, "float8_e4m3fn",
            1, 8, 16, 8, 8, 64, 4,
            0.125, "dynamic", "dynamic",
            16, 128, 64, 3, 0, 1,
            256, 8, [8], [16],
        ),
        "input": {
            "query": torch.rand(1, 8, 16, 80),
            "key": torch.rand(1, 16, 8, 80),
            "value": torch.rand(1, 16, 8, 80),
            "sparse_indices": torch.randint(0, 10, (1, 16, 8, 4)),
            "key_dequant_scale": torch.rand(1, 16, 8, 1),
            "value_dequant_scale": torch.rand(1, 16, 8, 1),
            "block_table": None,
            "actual_seq_lengths_query": torch.tensor([8]),
            "actual_seq_lengths_kv": torch.tensor([16]),
            "scale_value": 0.125,
            "key_quant_mode": "dynamic",
            "value_quant_mode": "dynamic",
            "sparse_block_size": 16,
            "layout_query": "BSND",
            "layout_kv": "BSND",
            "sparse_mode": 3,
            "pre_tokens": (1 << 63) - 1,
            "next_tokens": (1 << 63) - 1,
            "attention_mode": 0,
            "quant_scale_repo_mode": 1,
            "tile_size": 128,
            "rope_head_dim": 64,
            "kv_dtype": "float8_e4m3fn",
        },
        "cpu_output": None,
    }

    print("=== 原始 test_data input ===")
    for k, v in test_data["input"].items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}, {v.dtype}")
        else:
            print(f"  {k}: {v}")

    torch_tensor_list, golden_params = golden_testdata_to_golden_input(test_data)

    print("\n=== 转换后 torch_tensor_list ===")
    for i, t in enumerate(torch_tensor_list):
        if t is not None:
            print(f"  [{i}] shape: {t.shape}, dtype: {t.dtype}")
        else:
            print(f"  [{i}] None")

    print("\n=== 转换后 params ===")
    for k, v in golden_params.items():
        print(f"  {k}: {v}")

    print("\n=== 测试 golden.py ===")
    golden = load_golden_module()
    golden.copy = __import__('copy')
    golden.time = __import__('time')
    golden.math = __import__('math')
    golden.os = __import__('os')
    golden.random = __import__('random')

    fa_param = golden.get_param_fus(torch_tensor_list, golden_params)
    print(f"normal_flag: {fa_param.get('normal_flag', True)}")
    print("✓ 转换成功!")


if __name__ == "__main__":
    test_with_golden_testdata()