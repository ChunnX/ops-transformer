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

"""Pytest-backed CPU golden adapter for KvQuantSparseFlashAttention."""

import importlib.util
import sys
from pathlib import Path

import torch


PYTEST_GOLDEN_MODULE = None


class CaseDataStore:
    """Keep one pytest-generated result for each TTK testcase."""

    def __init__(self):
        self._cases = {}

    def put(self, testcase_name, data):
        if testcase_name is None:
            raise RuntimeError("KvQuantSparseFlashAttention requires testcase_name for pytest-backed golden")
        self._cases[str(testcase_name)] = data

    def get(self, testcase_name):
        data = self._cases.get(str(testcase_name)) if testcase_name is not None else None
        if data is None:
            raise RuntimeError(
                "KvQuantSparseFlashAttention golden requires customize_inputs to run before golden generation"
            )
        return data


CASE_DATA = CaseDataStore()


def load_pytest_golden_module():
    """Load the pytest generator in an operator-local module namespace."""
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE

    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "kv_quant_sparse_flash_attention_golden.py"
    saved_generator = sys.modules.pop("generate_tensor_data", None)
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location("qsfa_ttk_pytest_golden", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("qsfa_ttk_pytest_golden", None)
        raise
    finally:
        sys.path.remove(str(pytest_dir))
        sys.modules.pop("generate_tensor_data", None)
        if saved_generator is not None:
            sys.modules["generate_tensor_data"] = saved_generator
    PYTEST_GOLDEN_MODULE = module
    return module


def to_int_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(to_int_list(item))
        return result
    return [int(value)]


def dtype_name(tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return "fp16"
    mapping = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
        torch.int32: "int32",
        torch.int8: "int8",
    }
    try:
        return mapping[tensor.dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported KvQuantSparseFlashAttention pytest dtype: {tensor.dtype}") from exc


def _range_at(input_ranges, index, default):
    if input_ranges and index < len(input_ranges) and input_ranges[index] is not None:
        value = input_ranges[index]
        if isinstance(value, (list, tuple)) and len(value) in (1, 2):
            return list(value)
    return list(default)


def _prefix_lengths(values, total, field):
    if not values:
        return [int(total)]
    values = [int(item) for item in values]
    if values[-1] == int(total) and all(values[idx] >= values[idx - 1] for idx in range(1, len(values))):
        return values
    if sum(values) == int(total) and all(item >= 0 for item in values):
        result = []
        current = 0
        for item in values:
            current += item
            result.append(current)
        return result
    raise ValueError(f"{field} must be cumulative or sum to T={total}, got {values}")


def _sequence_values(value, values_override, layout, total, batch, field):
    values = to_int_list(values_override) if values_override is not None else to_int_list(value)
    if layout == "TND":
        return _prefix_lengths(values, total, field)
    if not values:
        return [int(total)] * int(batch)
    if len(values) == 1 and int(batch) > 1:
        return values * int(batch)
    if len(values) != int(batch):
        raise ValueError(f"{field} length must be 1 or batch={batch}, got {values}")
    return values


def build_pytest_params(query, key, value, sparse_indices, scale_value,
                        key_quant_mode, value_quant_mode, *, key_dequant_scale=None,
                        value_dequant_scale=None, block_table=None,
                        actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                        sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
                        sparse_mode=3, attention_mode=0, quant_scale_repo_mode=1,
                        tile_size=128, rope_head_dim=64, input_ranges=None,
                        testcase_name=None, **kwargs):
    """Translate the packed TTK API tensors to the pytest raw-input contract."""
    if query is None or key is None or sparse_indices is None:
        raise ValueError("KvQuantSparseFlashAttention requires query, key, and sparse_indices")

    tile_size = int(tile_size)
    rope_head_dim = int(rope_head_dim)
    d_nope = int(query.shape[-1]) - rope_head_dim
    if d_nope <= 0:
        raise ValueError(
            f"query last dimension must be larger than rope_head_dim: {query.shape[-1]} <= {rope_head_dim}"
        )
    packed_width = d_nope + rope_head_dim * 2 + d_nope // tile_size * 4
    if int(key.shape[-1]) != packed_width:
        raise ValueError(
            f"key packed width mismatch: TTK={key.shape[-1]}, expected={packed_width} "
            f"for D={d_nope}, rope_head_dim={rope_head_dim}, tile_size={tile_size}"
        )
    if value is not None and tuple(value.shape) != tuple(key.shape):
        raise ValueError("KvQuantSparseFlashAttention requires key and value cache shapes to match")

    if layout_query == "TND":
        batch = len(to_int_list(kwargs.get("actual_seq_lengths_query_values")))
        batch = batch or len(to_int_list(actual_seq_lengths_query)) or 1
        q_total = int(query.shape[0])
        q_heads = int(query.shape[1])
        q_raw_shape = [q_total, q_heads, d_nope]
    else:
        batch = int(query.shape[0])
        q_total = batch * int(query.shape[1])
        q_heads = int(query.shape[2])
        q_raw_shape = [batch, int(query.shape[1]), q_heads, d_nope]

    actual_q = _sequence_values(
        actual_seq_lengths_query,
        kwargs.get("actual_seq_lengths_query_values"),
        layout_query,
        int(query.shape[0]) if layout_query == "TND" else int(query.shape[1]),
        batch,
        "actual_seq_lengths_query",
    )

    if layout_kv == "PA_BSND":
        block_size = int(kwargs.get("block_size", key.shape[1]))
        block_num = int(kwargs.get("block_num", key.shape[0]))
        kv_heads = int(key.shape[2])
        kv_total = max(to_int_list(kwargs.get("actual_seq_lengths_kv_values")) or
                       to_int_list(actual_seq_lengths_kv) or [block_size])
        k_raw_shape = [batch, kv_total, kv_heads, d_nope]
    elif layout_kv == "TND":
        block_size = int(kwargs.get("block_size", 1))
        block_num = int(kwargs.get("block_num", 0))
        kv_heads = int(key.shape[1])
        kv_total = int(key.shape[0])
        k_raw_shape = [kv_total, kv_heads, d_nope]
    else:
        block_size = int(kwargs.get("block_size", 1))
        block_num = int(kwargs.get("block_num", 0))
        kv_heads = int(key.shape[2])
        kv_total = int(key.shape[1])
        k_raw_shape = [batch, kv_total, kv_heads, d_nope]

    actual_kv = _sequence_values(
        actual_seq_lengths_kv,
        kwargs.get("actual_seq_lengths_kv_values"),
        layout_kv,
        kv_total,
        batch,
        "actual_seq_lengths_kv",
    )
    if layout_kv == "PA_BSND":
        k_raw_shape[1] = max(actual_kv)

    q_rope_shape = list(q_raw_shape)
    q_rope_shape[-1] = rope_head_dim
    k_rope_shape = list(k_raw_shape)
    k_rope_shape[-1] = rope_head_dim
    scale_shape = list(k_raw_shape)
    scale_shape[-1] = d_nope // tile_size

    params = {
        "case_name": testcase_name or "ttk_kv_quant_sparse_flash_attention",
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "actualseqlengths": actual_q,
        "actualseqlengthskv": actual_kv,
        "scalevalue": float(scale_value),
        "key_quant_mode": int(key_quant_mode),
        "value_quant_mode": int(value_quant_mode),
        "sparsemode": int(sparse_mode),
        "sparse_blocksize": int(sparse_block_size),
        "shape_input": {
            "query": q_raw_shape,
            "key": k_raw_shape,
            "value": k_raw_shape,
            "sparse_indices": list(sparse_indices.shape),
            "block_table": list(block_table.shape) if block_table is not None else [batch],
            "query_cache": list(query.shape),
            "key_cache": list(key.shape),
            "value_cache": list(value.shape) if value is not None else list(key.shape),
            "query_rope": q_rope_shape,
            "key_rope": k_rope_shape,
            "dequant_scale": scale_shape,
            "v_dequant_scale": scale_shape,
        },
        "dtype_input": {
            "query": dtype_name(query),
            "key": dtype_name(key),
            "value": dtype_name(value if value is not None else key),
            "sparse_indices": "int32",
            "block_table": "int32",
            "query_cache": dtype_name(query),
            "key_cache": dtype_name(key),
            "value_cache": dtype_name(value if value is not None else key),
            "query_rope": dtype_name(query),
            "key_rope": dtype_name(query),
            "dequant_scale": "fp32",
            "v_dequant_scale": "fp32",
        },
        "range_input": {
            "query": _range_at(input_ranges, 0, (-1.0, 1.0)),
            "key": _range_at(input_ranges, 1, (-5, 5)),
            "sparse_indices": _range_at(input_ranges, 3, (0, max(int(sparse_indices.shape[-1]) - 1, 0))),
            "block_table": _range_at(input_ranges, 6, (0, max(block_num - 1, 0))),
            "query_rope": _range_at(input_ranges, 0, (-1.0, 1.0)),
            "key_rope": _range_at(input_ranges, 1, (-1.0, 1.0)),
            "dequant_scale": _range_at(input_ranges, 4, (1.0, 1.0)),
            "v_dequant_scale": _range_at(input_ranges, 5, (1.0, 1.0)),
        },
        "dtype_output": [dtype_name(query)],
        "shape_output": [list(q_raw_shape)],
        "attention_mode": int(attention_mode),
        "quant_scale_repo_mode": int(quant_scale_repo_mode),
        "tile_size": tile_size,
        "rope_head_dim": rope_head_dim,
        "block_size": block_size,
        "block_num": block_num,
    }
    return params


def cpu_kv_quant_sparse_flash_attention(query, key, value, sparse_indices, scale_value,
                                         key_quant_mode, value_quant_mode, *,
                                         testcase_name=None, **kwargs):
    """Return the pytest CPU golden that was generated for this exact case."""
    del query, key, value, sparse_indices, scale_value, key_quant_mode, value_quant_mode, kwargs
    return CASE_DATA.get(testcase_name)["golden"]


__golden__ = {
    "e2e": {
        "torch_npu.npu_kv_quant_sparse_flash_attention": "cpu_kv_quant_sparse_flash_attention",
    }
}
