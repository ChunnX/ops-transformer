#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig


TENSOR_INPUT_NAMES = {
    "query", "key", "value", "sparse_indices", "block_table",
    "actual_seq_lengths_query", "actual_seq_lengths_kv", "query_rope", "key_rope",
}


def _load_inputs_to_npu(input_dict):
    def to_npu(value):
        if isinstance(value, torch.Tensor):
            return value.npu()
        return value
    return {key: to_npu(value) for key, value in input_dict.items()}


def _call_npu_op(npu_inputs, input_dict):
    return torch_npu.npu_sparse_flash_attention(
        npu_inputs["query"],
        npu_inputs["key"],
        npu_inputs["value"],
        npu_inputs["sparse_indices"],
        input_dict["scale_value"],
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
    def forward(self, query, key, value, sparse_indices, scale_value,
                block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
                query_rope, key_rope, sparse_block_size, layout_query, layout_kv,
                sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse):
        return torch_npu.npu_sparse_flash_attention(
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
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            attention_mode=attention_mode,
            return_softmax_lse=return_softmax_lse,
        )


def test_sfa_process_ci(test_data, device_id=0):
    torch_npu.npu.set_device(device_id)
    input_dict = test_data["input"]
    npu_inputs = _load_inputs_to_npu(input_dict)
    print("test_data:", test_data["params"])
    print("npu_sparse_flash_attention (CI mode) ...")
    npu_result = _call_npu_op(npu_inputs, input_dict)
    torch.npu.synchronize()
    return npu_result, test_data["cpu_output"]


def test_sfa_process_graph(test_data, device_id=0):
    torch_npu.npu.set_device(device_id)
    input_dict = test_data["input"]
    npu_inputs = _load_inputs_to_npu(input_dict)

    torch._dynamo.reset()
    model = Network().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=backend, dynamic=False)

    print("test_data:", test_data["params"])
    print("npu_sparse_flash_attention (graph mode) ...")
    npu_result = model(
        npu_inputs["query"],
        npu_inputs["key"],
        npu_inputs["value"],
        npu_inputs["sparse_indices"],
        input_dict["scale_value"],
        npu_inputs.get("block_table"),
        npu_inputs.get("actual_seq_lengths_query"),
        npu_inputs.get("actual_seq_lengths_kv"),
        npu_inputs.get("query_rope"),
        npu_inputs.get("key_rope"),
        input_dict.get("sparse_block_size", 1),
        input_dict.get("layout_query", "BSND"),
        input_dict.get("layout_kv", "BSND"),
        input_dict.get("sparse_mode", 3),
        input_dict.get("pre_tokens", (1 << 63) - 1),
        input_dict.get("next_tokens", (1 << 63) - 1),
        input_dict.get("attention_mode", 2),
        input_dict.get("return_softmax_lse", False),
    )
    torch.npu.synchronize()
    return npu_result, test_data["cpu_output"]
