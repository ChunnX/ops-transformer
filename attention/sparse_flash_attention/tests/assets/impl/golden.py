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

"""Pytest-backed CPU golden adapter for SparseFlashAttention."""

import importlib.util
import sys
from importlib import metadata
from pathlib import Path

import torch


PYTEST_GOLDEN_MODULE = None


class CaseDataStore:
    """Keep the pytest-generated result paired with its TTK testcase."""

    def __init__(self):
        self._cases = {}

    def put(self, testcase_name, data):
        if testcase_name is None:
            raise RuntimeError("SparseFlashAttention requires testcase_name for pytest-backed golden")
        self._cases = {str(testcase_name): data}

    def get(self, testcase_name):
        data = self._cases.get(str(testcase_name)) if testcase_name is not None else None
        if data is None:
            raise RuntimeError(
                "SparseFlashAttention golden requires customize_inputs to run before golden generation"
            )
        return data


CASE_DATA = CaseDataStore()


class SparseFlashAttentionPytestAdapter:
    """Hold implementation details used to translate and load pytest Golden."""

    @staticmethod
    def package_version(*distribution_names):
        for name in distribution_names:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
        return "not-installed"

    @classmethod
    def load_error(cls, module_path, dependency_path, exc):
        versions = (
            f"python={sys.version.split()[0]}, "
            f"tensorflow={cls.package_version('tensorflow', 'tensorflow-cpu')}, "
            f"numpy={cls.package_version('numpy')}"
        )
        return RuntimeError(
            "Failed to load SparseFlashAttention pytest golden during module execution; "
            f"module={module_path}; dependency={dependency_path}; {versions}; "
            "generate_tensor_data imports TensorFlow at module import time. Verify that "
            "TensorFlow and generate_tensor_data can be imported in this Python/CANN "
            f"environment. Original error: {type(exc).__name__}: {exc}"
        )

    @staticmethod
    def range_at(input_ranges, index, default):
        if input_ranges and index < len(input_ranges) and input_ranges[index] is not None:
            value = input_ranges[index]
            if isinstance(value, (list, tuple)) and len(value) in (1, 2):
                return list(value)
        return list(default)

    @staticmethod
    def prefix_lengths(values, total, field):
        if not values:
            return [int(total)]
        values = [int(item) for item in values]
        if values[-1] == int(total) and all(
                values[index] >= values[index - 1] for index in range(1, len(values))):
            return values
        if sum(values) == int(total) and all(item >= 0 for item in values):
            total_value = 0
            prefix = []
            for item in values:
                total_value += item
                prefix.append(total_value)
            return prefix
        raise ValueError(f"{field} must be cumulative or sum to T={total}, got {values}")

    @classmethod
    def sequence_values(cls, value, values_override, layout, total, batch, field):
        values = to_int_list(values_override) if values_override is not None else to_int_list(value)
        if layout == "TND":
            return cls.prefix_lengths(values, total, field)
        if not values:
            return [int(total)] * int(batch)
        if len(values) == 1 and int(batch) > 1:
            return values * int(batch)
        if len(values) != int(batch):
            raise ValueError(f"{field} length must be 1 or batch={batch}, got {values}")
        return values


def load_pytest_golden_module():
    """Load the pytest implementation without polluting another operator's imports."""
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE

    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "sparse_flash_attention_golden.py"
    dependency_path = pytest_dir / "generate_tensor_data.py"
    module_name = "sfa_ttk_pytest_golden"
    saved_generator = sys.modules.pop("generate_tensor_data", None)
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise SparseFlashAttentionPytestAdapter.load_error(
            module_path, dependency_path, exc
        ) from exc
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
        raise ValueError(f"Unsupported SparseFlashAttention pytest dtype: {tensor.dtype}") from exc


def build_pytest_params(query, key, value, sparse_indices, scale_value, *, block_table=None,
                        actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                        query_rope=None, key_rope=None, sparse_block_size=1,
                        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
                        attention_mode=0, return_softmax_lse=False, input_ranges=None,
                        testcase_name=None, **kwargs):
    """Translate the TTK API tensors to the parameter dictionary used by pytest."""
    if query is None or key is None or sparse_indices is None:
        raise ValueError("SparseFlashAttention requires query, key, and sparse_indices")

    rope_dim = int(kwargs.get("rope_head_dim", query_rope.shape[-1] if query_rope is not None else 64))
    if layout_query == "TND":
        batch = len(to_int_list(kwargs.get("actual_seq_lengths_query_values")))
        batch = batch or len(to_int_list(actual_seq_lengths_query)) or 1
        q_total = int(query.shape[0])
        q_heads = int(query.shape[1])
        q_seq = q_total
    else:
        batch = int(query.shape[0])
        q_total = batch * int(query.shape[1])
        q_heads = int(query.shape[2])
        q_seq = int(query.shape[1])

    actual_q = SparseFlashAttentionPytestAdapter.sequence_values(
        actual_seq_lengths_query,
        kwargs.get("actual_seq_lengths_query_values"),
        layout_query,
        int(query.shape[0]) if layout_query == "TND" else q_seq,
        batch,
        "actual_seq_lengths_query",
    )

    if layout_kv == "PA_BSND":
        block_size = int(kwargs.get("block_size", key.shape[1]))
        block_num = int(kwargs.get("block_num", key.shape[0]))
        kv_heads = int(key.shape[2])
        kv_total = max(to_int_list(kwargs.get("actual_seq_lengths_kv_values")) or
                       to_int_list(actual_seq_lengths_kv) or [block_size])
        key_shape = [batch, kv_total, kv_heads, int(key.shape[3])]
    elif layout_kv == "TND":
        block_size = int(kwargs.get("block_size", 1))
        block_num = int(kwargs.get("block_num", 0))
        kv_heads = int(key.shape[1])
        kv_total = int(key.shape[0])
        key_shape = list(key.shape)
    else:
        block_size = int(kwargs.get("block_size", 1))
        block_num = int(kwargs.get("block_num", 0))
        kv_heads = int(key.shape[2])
        kv_total = int(key.shape[1])
        key_shape = list(key.shape)

    actual_kv = SparseFlashAttentionPytestAdapter.sequence_values(
        actual_seq_lengths_kv,
        kwargs.get("actual_seq_lengths_kv_values"),
        layout_kv,
        kv_total,
        batch,
        "actual_seq_lengths_kv",
    )
    if layout_kv == "PA_BSND":
        key_shape[1] = max(actual_kv)

    if query_rope is None:
        query_rope_shape = list(query.shape)
        query_rope_shape[-1] = rope_dim
    else:
        query_rope_shape = list(query_rope.shape)
    if key_rope is None or layout_kv == "PA_BSND":
        key_rope_shape = list(key_shape)
        key_rope_shape[-1] = rope_dim
    else:
        key_rope_shape = list(key_rope.shape)

    group = q_heads // kv_heads
    if layout_query == "TND":
        softmax_shape = [kv_heads, int(query.shape[0]), group]
    else:
        softmax_shape = [batch, kv_heads, q_seq, group]

    params = {
        "case_name": testcase_name or "ttk_sparse_flash_attention",
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "actualseqlengths": actual_q,
        "actualseqlengthskv": actual_kv,
        "scalevalue": float(scale_value),
        "sparsemode": int(sparse_mode),
        "sparse_blocksize": int(sparse_block_size),
        "shape_input": {
            "query": list(query.shape),
            "key": key_shape,
            "value": key_shape,
            "sparse_indices": list(sparse_indices.shape),
            "block_table": list(block_table.shape) if block_table is not None else [batch],
            "query_rope": query_rope_shape,
            "key_rope": key_rope_shape,
            "sinks": [q_heads],
        },
        "dtype_input": {
            "query": dtype_name(query),
            "key": dtype_name(key),
            "value": dtype_name(value if value is not None else key),
            "sparse_indices": "int32",
            "block_table": "int32",
            "query_rope": dtype_name(query_rope if query_rope is not None else query),
            "key_rope": dtype_name(key_rope if key_rope is not None else key),
            "sinks": "fp32",
        },
        "range_input": {
            "query": SparseFlashAttentionPytestAdapter.range_at(
                input_ranges, 0, (-10.0, 10.0)
            ),
            "key": SparseFlashAttentionPytestAdapter.range_at(
                input_ranges, 1, (-10.0, 10.0)
            ),
            "sparse_indices": SparseFlashAttentionPytestAdapter.range_at(
                input_ranges, 3, (0, max(int(sparse_indices.shape[-1]) - 1, 0))
            ),
            "block_table": SparseFlashAttentionPytestAdapter.range_at(
                input_ranges, 4, (0, max(block_num - 1, 0))
            ),
            "query_rope": SparseFlashAttentionPytestAdapter.range_at(
                input_ranges, 7, (-10.0, 10.0)
            ),
            "key_rope": SparseFlashAttentionPytestAdapter.range_at(
                input_ranges, 8, (-10.0, 10.0)
            ),
            "sinks": (-0.2, 0.2),
        },
        "dtype_output": [dtype_name(query)],
        "shape_output": {
            "attn_out": list(query.shape),
            "softmax_max": softmax_shape,
            "softmax_sum": softmax_shape,
        },
        "rope_head_dim": rope_dim,
        "attention_mode": int(attention_mode),
        "return_softmax_lse": bool(return_softmax_lse),
        "use_sinks": False,
        "block_size": block_size,
        "block_num": block_num,
    }
    return params


def normalize_pytest_outputs(outputs, query, return_softmax_lse):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise RuntimeError("SparseFlashAttention pytest compute_cpu returned an invalid output tuple")
    attn_out = outputs[0].detach().cpu().to(dtype=query.dtype)
    if not bool(return_softmax_lse):
        empty = torch.zeros(0, dtype=torch.float32)
        return attn_out, empty, empty
    return tuple(output.detach().cpu() for output in outputs)


def cpu_sparse_flash_attention(query, key, value, sparse_indices, scale_value, *,
                               testcase_name=None, **kwargs):
    """Return the result generated by pytest for this exact TTK testcase."""
    del query, key, value, sparse_indices, scale_value, kwargs
    return CASE_DATA.get(testcase_name)["golden"]


__golden__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "cpu_sparse_flash_attention",
    }
}
