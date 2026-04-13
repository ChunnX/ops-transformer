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

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig


def _load_inputs_to_npu(input_dict):
    # 将输入字典中的 tensor 逐个搬到 NPU。
    def to_npu(value):
        if isinstance(value, torch.Tensor):
            return value.npu()
        return value

    return {key: to_npu(value) for key, value in input_dict.items()}


def _call_npu_op(npu_inputs, input_dict):
    # single/batch 默认走该路径，直接执行 aclnn 单算子。
    return torch_npu.npu_sparse_flash_attention(
        query=npu_inputs["query"],
        key=npu_inputs["key"],
        value=npu_inputs["value"],
        sparse_indices=npu_inputs["sparse_indices"],
        scale_value=input_dict["scale_value"],
        block_table=npu_inputs.get("block_table"),
        actual_seq_lengths_query=npu_inputs.get("actual_seq_lengths_query"),
        actual_seq_lengths_kv=npu_inputs.get("actual_seq_lengths_kv"),
        query_rope=npu_inputs.get("query_rope"),
        key_rope=npu_inputs.get("key_rope"),
        sparse_block_size=input_dict.get("sparse_block_size", 1),
        layout_query=input_dict.get("layout_query", "BSND"),
        layout_kv=input_dict.get("layout_kv", "BSND"),
        sparse_mode=input_dict.get("sparse_mode", 3),
        pre_tokens=input_dict.get("pre_tokens", (1 << 63) - 1),
        next_tokens=input_dict.get("next_tokens", (1 << 63) - 1),
        attention_mode=input_dict.get("attention_mode", 2),
        return_softmax_lse=input_dict.get("return_softmax_lse", False),
    )


class Network(torch.nn.Module):
    # 图模式下的最小封装，便于 torch.compile 捕获整个调用。
    def forward(self, query, key, value, sparse_indices, scale_value,
                block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
                query_rope, key_rope, sparse_block_size, layout_query, layout_kv,
                sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse):
        return torch_npu.npu_sparse_flash_attention(
            query=query,
            key=key,
            value=value,
            sparse_indices=sparse_indices,
            scale_value=scale_value,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope,
            key_rope=key_rope,
            sparse_block_size=sparse_block_size,
            layout_query=layout_query,
            layout_kv=layout_kv,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            attention_mode=attention_mode,
            return_softmax_lse=return_softmax_lse,
        )


def test_sfa_process_ci(test_data, device_id=0):
    params = test_data["params"]
    input_dict = test_data["input"]
    torch_npu.npu.set_device(device_id)
    npu_inputs = _load_inputs_to_npu(input_dict)

    print("test_data:", params)
    print("npu_sparse_flash_attention (CI mode) ...")
    npu_result = _call_npu_op(npu_inputs, input_dict)
    torch.npu.synchronize()
    return npu_result, test_data["cpu_output"]


def test_sfa_process_graph(test_data, device_id=0):
    # graph 模式下先编译，再用同一组输入做回放。
    params = test_data["params"]
    input_dict = test_data["input"]
    torch_npu.npu.set_device(device_id)

    torch._dynamo.reset()
    npu_model = Network().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend, dynamic=False)

    npu_inputs = _load_inputs_to_npu(input_dict)
    print("test_data:", params)
    print("npu_sparse_flash_attention (graph mode) ...")
    npu_result = npu_model(
        query=npu_inputs["query"],
        key=npu_inputs["key"],
        value=npu_inputs["value"],
        sparse_indices=npu_inputs["sparse_indices"],
        scale_value=input_dict["scale_value"],
        block_table=npu_inputs.get("block_table"),
        actual_seq_lengths_query=npu_inputs.get("actual_seq_lengths_query"),
        actual_seq_lengths_kv=npu_inputs.get("actual_seq_lengths_kv"),
        query_rope=npu_inputs.get("query_rope"),
        key_rope=npu_inputs.get("key_rope"),
        sparse_block_size=input_dict.get("sparse_block_size", 1),
        layout_query=input_dict.get("layout_query", "BSND"),
        layout_kv=input_dict.get("layout_kv", "BSND"),
        sparse_mode=input_dict.get("sparse_mode", 3),
        pre_tokens=input_dict.get("pre_tokens", (1 << 63) - 1),
        next_tokens=input_dict.get("next_tokens", (1 << 63) - 1),
        attention_mode=input_dict.get("attention_mode", 2),
        return_softmax_lse=input_dict.get("return_softmax_lse", False),
    )
    torch.npu.synchronize()
    return npu_result, test_data["cpu_output"]
