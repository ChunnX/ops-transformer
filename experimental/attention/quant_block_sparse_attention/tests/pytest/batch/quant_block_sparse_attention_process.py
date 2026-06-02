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

import pytest
import torch

try:
    import torch_npu
except ImportError as import_error:  # pragma: no cover - depends on NPU environment
    torch_npu = None
    TORCH_NPU_IMPORT_ERROR = import_error
else:
    TORCH_NPU_IMPORT_ERROR = None

import custom_ops


def _to_npu(tensor):
    if tensor is None:
        return None
    return tensor.npu()


def _to_cpu(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _check_runtime_available():
    if torch_npu is None:
        pytest.skip(f"torch_npu is not available: {TORCH_NPU_IMPORT_ERROR}")
    if not custom_ops.has_quant_block_sparse_attention_op():
        detail = "; ".join(custom_ops.load_errors()[:3])
        suffix = f" Load errors: {detail}" if detail else ""
        pytest.skip(f"torch.ops.custom.npu_quant_block_sparse_attention is not registered.{suffix}")


def _normalize_npu_output(output, return_softmax_lse):
    if isinstance(output, dict):
        attention_out = output.get("attention_out")
        softmax_lse = output.get("softmax_lse")
    elif isinstance(output, (tuple, list)):
        attention_out = output[0]
        softmax_lse = output[1] if len(output) > 1 else None
    else:
        attention_out = output
        softmax_lse = None

    result = {"attention_out": _to_cpu(attention_out)}
    if return_softmax_lse and softmax_lse is not None:
        result["softmax_lse"] = _to_cpu(softmax_lse)
    return result


def _golden_output(test_data):
    inputs = test_data["input"]
    if "golden" in test_data:
        golden = {"attention_out": test_data["golden"]["attention_out"]}
        if inputs["return_softmax_lse"]:
            golden["softmax_lse"] = test_data["golden"]["softmax_lse"]
        return golden

    golden = {"attention_out": test_data["cpu_output"]}
    if inputs["return_softmax_lse"]:
        golden["softmax_lse"] = test_data["cpu_softmax_lse"]
    return golden


def _call_npu_quant_block_sparse_attention(test_data, device_id):
    _check_runtime_available()
    torch_npu.npu.set_device(device_id)
    inputs = test_data["input"]
    output = torch.ops.custom.npu_quant_block_sparse_attention(
        _to_npu(inputs["query"]),
        _to_npu(inputs["key"]),
        _to_npu(inputs["value"]),
        _to_npu(inputs["q_scale"]),
        _to_npu(inputs["k_scale"]),
        _to_npu(inputs["v_scale"]),
        _to_npu(inputs["p_scale"]),
        _to_npu(inputs["sparse_indices"]),
        _to_npu(inputs["sparse_seq_len"]),
        _to_npu(inputs["atten_mask"]),
        inputs["softmax_scale"],
        inputs["q_block_size"],
        inputs["kv_block_size"],
        cu_seqlens_q=_to_npu(inputs["cu_seqlens_q"]),
        cu_seqlens_kv=_to_npu(inputs["cu_seqlens_kv"]),
        seqused_q=_to_npu(inputs["seqused_q"]),
        seqused_kv=_to_npu(inputs["seqused_kv"]),
        block_table=_to_npu(inputs["block_table"]),
        metadata=_to_npu(inputs["metadata"]),
        max_seqlen_q=inputs["max_seqlen_q"],
        max_seqlen_kv=inputs["max_seqlen_kv"],
        kv_layout=inputs["kv_layout"],
        sparse_indices_layout=inputs["sparse_indices_layout"],
        quant_mode=inputs["quant_mode"],
        mask_mode=inputs["mask_mode"],
        return_softmax_lse=inputs["return_softmax_lse"],
    )
    torch_npu.npu.synchronize()
    return _normalize_npu_output(output, inputs["return_softmax_lse"])


def test_quant_block_sparse_attention_process_graph(test_data, device_id=0):
    npu_result = _call_npu_quant_block_sparse_attention(test_data, device_id)
    return npu_result, _golden_output(test_data)


def test_quant_block_sparse_attention_process_ci(test_data, device_id=0):
    npu_result = _call_npu_quant_block_sparse_attention(test_data, device_id)
    return npu_result, _golden_output(test_data)
