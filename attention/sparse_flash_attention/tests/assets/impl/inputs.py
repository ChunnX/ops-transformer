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

"""Input adapter that delegates SparseFlashAttention data construction to pytest."""

import importlib.util
import sys
from pathlib import Path

import torch


def _golden_module():
    name = "sfa_ttk_golden"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("golden.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _copy_tensor(destination, source, name):
    if destination is None:
        return
    if source is None:
        raise ValueError(f"{name} is declared by CSV but pytest did not produce it")
    source_cpu = source.detach().cpu() if torch.is_tensor(source) else torch.as_tensor(source)
    if tuple(destination.shape) != tuple(source_cpu.shape):
        raise ValueError(
            f"{name} shape mismatch: TTK={tuple(destination.shape)}, pytest={tuple(source_cpu.shape)}"
        )
    destination.copy_(source_cpu.to(dtype=destination.dtype, device=destination.device))


def _copy_sequence(destination, values, name):
    if destination is None:
        return
    source = torch.tensor(values, dtype=destination.dtype, device=destination.device)
    if source.numel() != destination.numel():
        raise ValueError(
            f"{name} size mismatch: TTK={destination.numel()}, pytest={source.numel()}"
        )
    destination.copy_(source.reshape(destination.shape))


def generate_sfa_inputs(query, key, value, sparse_indices, scale_value, *, block_table=None,
                        actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                        query_rope=None, key_rope=None, sparse_block_size=1,
                        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
                        attention_mode=0, return_softmax_lse=False, testcase_name=None,
                        input_ranges=None, **kwargs):
    """Generate the final API tensors and CPU golden through the pytest implementation."""
    golden_module = _golden_module()
    params = golden_module.build_pytest_params(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
        input_ranges=input_ranges,
        testcase_name=testcase_name,
        **kwargs,
    )
    pytest_golden = golden_module.load_pytest_golden_module()
    generated = pytest_golden.generate_input_tensors(params)
    outputs, generated, _ = pytest_golden.compute_cpu(generated, params)
    if outputs is None:
        raise RuntimeError("SparseFlashAttention pytest compute_cpu failed")
    if hasattr(torch, "npu"):
        torch.npu.synchronize()

    _copy_tensor(query, generated.get("query"), "query")
    _copy_tensor(key, generated.get("key_cache", generated.get("key")), "key")
    _copy_tensor(value, generated.get("value_cache", generated.get("value")), "value")
    _copy_tensor(sparse_indices, generated.get("sparse_indices"), "sparse_indices")
    _copy_tensor(block_table, generated.get("block_table"), "block_table")
    _copy_sequence(actual_seq_lengths_query, params["actualseqlengths"], "actual_seq_lengths_query")
    _copy_sequence(actual_seq_lengths_kv, params["actualseqlengthskv"], "actual_seq_lengths_kv")
    _copy_tensor(query_rope, generated.get("query_rope"), "query_rope")
    _copy_tensor(key_rope, generated.get("key_rope_cache", generated.get("key_rope")), "key_rope")

    golden_module.CASE_DATA.put(
        testcase_name,
        {"golden": golden_module.normalize_pytest_outputs(outputs, query, return_softmax_lse)},
    )


__input__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "generate_sfa_inputs",
    }
}
