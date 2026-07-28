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

"""Input adapter that delegates KvQuantSparseFlashAttention construction to pytest."""

import importlib.util
import sys
from pathlib import Path

import torch


def _module_load_error(path, exc):
    return RuntimeError(
        "Failed to load KvQuantSparseFlashAttention module; "
        f"stage=assets Golden store; module={path.resolve()}; "
        f"original error: {type(exc).__name__}: {exc}"
    )


def _golden_module():
    name = "qsfa_ttk_golden"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("golden.py")
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise _module_load_error(path, exc) from exc
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


def _packed_scale(key_cache, destination, d_nope, rope_head_dim):
    if destination is None:
        return
    if key_cache is None:
        raise ValueError("pytest did not produce key_cache for dequant scale extraction")
    raw_bytes = key_cache.detach().cpu().contiguous().view(torch.uint8)
    scale_bytes = raw_bytes[..., int(d_nope) + int(rope_head_dim) * 2:]
    scale = scale_bytes.contiguous().view(torch.float32)
    _copy_tensor(destination, scale, "dequant_scale")


def generate_qsfa_inputs(query, key, value, sparse_indices, scale_value,
                         key_quant_mode, value_quant_mode, *, key_dequant_scale=None,
                         value_dequant_scale=None, block_table=None,
                         actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                         sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
                         sparse_mode=3, attention_mode=0, quant_scale_repo_mode=1,
                         tile_size=128, rope_head_dim=64, testcase_name=None,
                         input_ranges=None, **kwargs):
    """Generate final packed API tensors and the CPU golden through pytest."""
    golden_module = _golden_module()
    params = golden_module.build_pytest_params(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        key_dequant_scale=key_dequant_scale,
        value_dequant_scale=value_dequant_scale,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        attention_mode=attention_mode,
        quant_scale_repo_mode=quant_scale_repo_mode,
        tile_size=tile_size,
        rope_head_dim=rope_head_dim,
        input_ranges=input_ranges,
        testcase_name=testcase_name,
        **kwargs,
    )
    pytest_golden = golden_module.load_pytest_golden_module()
    generated = pytest_golden.generate_input_tensors(params)
    output, generated, _ = pytest_golden.compute_cpu(generated, params)
    if output is None:
        raise RuntimeError("KvQuantSparseFlashAttention pytest compute_cpu failed")
    if hasattr(torch, "npu"):
        torch.npu.synchronize()

    _copy_tensor(query, generated.get("query_cache"), "query")
    _copy_tensor(key, generated.get("key_cache"), "key")
    _copy_tensor(value, generated.get("value_cache"), "value")
    _copy_tensor(sparse_indices, generated.get("sparse_indices"), "sparse_indices")
    _copy_tensor(block_table, generated.get("block_table"), "block_table")
    _copy_sequence(actual_seq_lengths_query, params["actualseqlengths"], "actual_seq_lengths_query")
    _copy_sequence(actual_seq_lengths_kv, params["actualseqlengthskv"], "actual_seq_lengths_kv")
    d_nope = int(query.shape[-1]) - int(rope_head_dim)
    _packed_scale(generated.get("key_cache"), key_dequant_scale, d_nope, rope_head_dim)
    _packed_scale(generated.get("value_cache"), value_dequant_scale, d_nope, rope_head_dim)

    golden_module.CASE_DATA.put(
        testcase_name,
        {"golden": output.detach().cpu().to(dtype=query.dtype)},
    )


__input__ = {
    "e2e": {
        "torch_npu.npu_kv_quant_sparse_flash_attention": "generate_qsfa_inputs",
    }
}
