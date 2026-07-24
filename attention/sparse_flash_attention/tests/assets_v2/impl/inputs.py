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

"""Input adapter for the SparseFlashAttention standard-import control."""

import torch

from . import golden as _GOLDEN_MODULE


def _golden_module():
    return _GOLDEN_MODULE


def _tensor_summary(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "stride": tuple(value.stride()),
        }
    return {"type": type(value).__name__}


def _adapter_log(golden_module, event, force=False, **fields):
    logger = getattr(golden_module, "_diagnostic_log", None)
    if callable(logger):
        logger(event, force=force, **fields)


def _adapter_diagnostics_enabled(golden_module):
    enabled = getattr(golden_module, "_diagnostic_enabled", None)
    return bool(enabled()) if callable(enabled) else False


def _copy_tensor(destination, source, name, golden_module=None):
    diagnostics_enabled = _adapter_diagnostics_enabled(golden_module)
    if diagnostics_enabled:
        _adapter_log(
            golden_module,
            "copy_tensor_start",
            field=name,
            destination=_tensor_summary(destination),
            source=_tensor_summary(source),
        )
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
    if diagnostics_enabled:
        _adapter_log(
            golden_module,
            "copy_tensor_complete",
            field=name,
            destination=_tensor_summary(destination),
        )


def _copy_sequence(destination, values, name, golden_module=None):
    diagnostics_enabled = _adapter_diagnostics_enabled(golden_module)
    if diagnostics_enabled:
        _adapter_log(
            golden_module,
            "copy_sequence_start",
            field=name,
            destination=_tensor_summary(destination),
            source_type=type(values).__name__,
        )
    if destination is None:
        return
    source = torch.tensor(values, dtype=destination.dtype, device=destination.device)
    if source.numel() != destination.numel():
        raise ValueError(
            f"{name} size mismatch: TTK={destination.numel()}, pytest={source.numel()}"
        )
    destination.copy_(source.reshape(destination.shape))
    if diagnostics_enabled:
        _adapter_log(
            golden_module,
            "copy_sequence_complete",
            field=name,
            destination=_tensor_summary(destination),
        )


def generate_sfa_inputs(query, key, value, sparse_indices, scale_value, *, block_table=None,
                        actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                        query_rope=None, key_rope=None, sparse_block_size=1,
                        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
                        attention_mode=0, return_softmax_lse=False, testcase_name=None,
                        input_ranges=None, **kwargs):
    """Generate the final API tensors and CPU golden through the pytest implementation."""
    golden_module = _golden_module()
    diagnostics_enabled = _adapter_diagnostics_enabled(golden_module)
    if diagnostics_enabled:
        _adapter_log(
            golden_module,
            "input_generation_start",
            testcase_name=testcase_name,
            layout_query=layout_query,
            layout_kv=layout_kv,
            sparse_mode=sparse_mode,
            attention_mode=attention_mode,
            tensors={
                "query": _tensor_summary(query),
                "key": _tensor_summary(key),
                "value": _tensor_summary(value),
                "sparse_indices": _tensor_summary(sparse_indices),
                "block_table": _tensor_summary(block_table),
                "query_rope": _tensor_summary(query_rope),
                "key_rope": _tensor_summary(key_rope),
            },
            extra_attribute_names=sorted(kwargs),
        )
    try:
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
        if diagnostics_enabled:
            _adapter_log(
                golden_module,
                "pytest_params_ready",
                testcase_name=testcase_name,
                actual_seq_lengths_query=params.get("actualseqlengths"),
                actual_seq_lengths_kv=params.get("actualseqlengthskv"),
                shape_input=params.get("shape_input"),
                dtype_input=params.get("dtype_input"),
            )
        pytest_golden = golden_module.load_pytest_golden_module()
        if diagnostics_enabled:
            _adapter_log(
                golden_module,
                "pytest_generate_inputs_start",
                pytest_module_file=getattr(pytest_golden, "__file__", None),
            )
        generated = pytest_golden.generate_input_tensors(params)
        if diagnostics_enabled:
            _adapter_log(
                golden_module,
                "pytest_generate_inputs_complete",
                generated={key: _tensor_summary(item) for key, item in generated.items()},
            )
            _adapter_log(golden_module, "pytest_compute_cpu_start")
        outputs, generated, _ = pytest_golden.compute_cpu(generated, params)
        if diagnostics_enabled:
            _adapter_log(
                golden_module,
                "pytest_compute_cpu_complete",
                outputs=[_tensor_summary(output) for output in outputs] if outputs is not None else None,
            )
        if outputs is None:
            raise RuntimeError("SparseFlashAttention pytest compute_cpu failed")
        if hasattr(torch, "npu"):
            if diagnostics_enabled:
                _adapter_log(golden_module, "torch_npu_synchronize_start")
            torch.npu.synchronize()
            if diagnostics_enabled:
                _adapter_log(golden_module, "torch_npu_synchronize_complete")

        _copy_tensor(query, generated.get("query"), "query", golden_module)
        _copy_tensor(key, generated.get("key_cache", generated.get("key")), "key", golden_module)
        _copy_tensor(value, generated.get("value_cache", generated.get("value")), "value", golden_module)
        _copy_tensor(sparse_indices, generated.get("sparse_indices"), "sparse_indices", golden_module)
        _copy_tensor(block_table, generated.get("block_table"), "block_table", golden_module)
        _copy_sequence(
            actual_seq_lengths_query,
            params["actualseqlengths"],
            "actual_seq_lengths_query",
            golden_module,
        )
        _copy_sequence(
            actual_seq_lengths_kv,
            params["actualseqlengthskv"],
            "actual_seq_lengths_kv",
            golden_module,
        )
        _copy_tensor(query_rope, generated.get("query_rope"), "query_rope", golden_module)
        _copy_tensor(key_rope, generated.get("key_rope_cache", generated.get("key_rope")), "key_rope", golden_module)

        golden_module.CASE_DATA.put(
            testcase_name,
            {"golden": golden_module.normalize_pytest_outputs(outputs, query, return_softmax_lse)},
        )
        if diagnostics_enabled:
            _adapter_log(golden_module, "input_generation_complete", testcase_name=testcase_name)
    except Exception as exc:
        exception_logger = getattr(golden_module, "_log_exception", None)
        if callable(exception_logger):
            exception_logger("input_generation_failure", exc)
        raise


__input__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "generate_sfa_inputs",
    }
}
