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
# import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

import os
import numpy as np
import time
import math
import random
import tensorflow as tf
from ml_dtypes import float8_e4m3fn
import utils


def _load_inputs_to_npu(input_dict):
    # 将输入字典中的 tensor 逐个搬到 NPU。
    def to_npu(value):
        if isinstance(value, torch.Tensor):
            return value.npu()
        return value
    return {key: to_npu(value) for key, value in input_dict.items()}


def _get_kv_torch_dtype(kv_dtype_str):
    # 把配置里的 kv 类型字符串映射成接口实际需要的 dtype。
    if kv_dtype_str == "hifloat8":
        return torch_npu.hifloat8
    if kv_dtype_str == "float8_e4m3fn":
        return torch.float8_e4m3fn
    return None


def aclnn_op_graph_func_sfa(torch_tensor_list, params):
    flagList = params['flaglist']
    print(f"flagList:{flagList}")

    # attr
    scale_value = params.get('scalevalue', 1)
    layout_query = params.get('layout_query', "BSND")
    layout_kv = params.get('layout_kv', "PA_BSND")
    sparse_mode = params.get('sparsemode', 0)
    sparse_blocksize = params.get('sparse_blocksize', 0)
    key_quant_mode = params.get('key_quant_mode', 2)
    value_quant_mode = params.get('value_quant_mode', 2)
    attention_mode = params.get('attention_mode', 2)
    quant_scale_repo_mode = params.get('quant_scale_repo_mode', 1)
    tile_size = params.get('tile_size', 128)
    rope_head_dim = params.get('rope_head_dim', 64)
    pre_tokens = params.get('pre_tokens', (1 << 63) - 1)
    next_tokens = params.get('next_tokens', (1 << 63) - 1)

    # q info
    q_tensor = torch_tensor_list[5]
    k_tensor = torch_tensor_list[6]  # .unsqueeze(-2)
    v_tensor = torch_tensor_list[7]  # .unsqueeze(-2)

    key_dtype_name = params['dtype_input'][1]
    value_dtype_name = params['dtype_input'][2]
    key_dtype = torch_npu.hifloat8 if key_dtype_name == "hifloat8" else None 
    value_dtype = torch_npu.hifloat8 if value_dtype_name == "hifloat8" else None

    sparse_indices = torch_tensor_list[3] if flagList[3] != 0 else None

    block_table = torch_tensor_list[4] if flagList[6] != 0 else None

    # act_seq info
    actualSeqLengthqs = torch.tensor(params["actualseqlengths"], dtype=torch.int32).to('npu') if flagList[
                                                                                                      4] != 0 else None
    actualSeqLengthkvs = torch.tensor(params["actualseqlengthskv"], dtype=torch.int32).to('npu') if flagList[
                                                                                                         5] != 0 else None
    actualseqtensor = False
    aclgraph = False

    # rope info
    # q_rope_tensor = torch_tensor_list[7] if flagList[8] != 0 else None
    # k_rope_tensor = torch_tensor_list[8] if params['layout_kv'] != "PA_BSND" else torch_tensor_list[9]#.unsqueeze(-2)
    q_rope_tensor = None
    k_rope_tensor = None

    key_dequant_scale = None
    value_dequant_scale = None
    if key_dtype_name == "hifloat8":
        output = torch_npu.npu_kv_quant_sparse_flash_attention(q_tensor,
                                                            k_tensor,
                                                            v_tensor,
                                                            sparse_indices=sparse_indices,
                                                            scale_value=scale_value,
                                                            sparse_block_size=sparse_blocksize,
                                                            key_quant_mode=key_quant_mode,
                                                            value_quant_mode=value_quant_mode,
                                                            key_dequant_scale=key_dequant_scale,
                                                            value_dequant_scale=value_dequant_scale,
                                                            block_table=block_table,
                                                            actual_seq_lengths_query=actualSeqLengthqs,
                                                            actual_seq_lengths_kv=actualSeqLengthkvs,
                                                            layout_query=layout_query,
                                                            layout_kv=layout_kv,
                                                            sparse_mode=sparse_mode,
                                                            attention_mode=attention_mode,
                                                            quant_scale_repo_mode=quant_scale_repo_mode,
                                                            tile_size=tile_size,
                                                            rope_head_dim=rope_head_dim,
                                                            key_dtype=key_dtype,
                                                            value_dtype=value_dtype,
                                                            pre_tokens=pre_tokens,
                                                            next_tokens=next_tokens)           
    else:
        output = torch_npu.npu_kv_quant_sparse_flash_attention(q_tensor,
                                                            k_tensor,
                                                            v_tensor,
                                                            sparse_indices=sparse_indices,
                                                            scale_value=scale_value,
                                                            sparse_block_size=sparse_blocksize,
                                                            key_quant_mode=key_quant_mode,
                                                            value_quant_mode=value_quant_mode,
                                                            key_dequant_scale=key_dequant_scale,
                                                            value_dequant_scale=value_dequant_scale,
                                                            block_table=block_table,
                                                            actual_seq_lengths_query=actualSeqLengthqs,
                                                            actual_seq_lengths_kv=actualSeqLengthkvs,
                                                            layout_query=layout_query,
                                                            layout_kv=layout_kv,
                                                            sparse_mode=sparse_mode,
                                                            attention_mode=attention_mode,
                                                            quant_scale_repo_mode=quant_scale_repo_mode,
                                                            tile_size=tile_size,
                                                            rope_head_dim=rope_head_dim,
                                                            pre_tokens=pre_tokens,
                                                            next_tokens=next_tokens)
    print("out shape", output.shape)
    return output


def find_output_value(input_list, output_value_list):
    temp_list = []
    is_list_or_tuple = False
    for item in input_list:
        if not isinstance(item, list) and not isinstance(item, tuple):
            output_value_list.append(item)
        else:
            for elment in item:
                temp_list.append(elment)
    if len(temp_list):
        is_list_or_tuple = True
    return is_list_or_tuple, temp_list


def get_output_list(is_list_or_tuple, input_list, output_value_list):
    while True:
        if not is_list_or_tuple:
            return output_value_list
        else:
            is_list_or_tuple, input_list = find_output_value(input_list, output_value_list)


def call_npu(input_tensor_list_cpu, params):
    overflows_count = 0
    input_tensor_list = []
    for i in range(len(params['shape_input'])):
        np_dtype = get_np_dtype(params['dtype_input'][i])
        type_input = params['type_input'][i]
        if type_input == 'tensor':
            if "bin_path_input" in params:
                if params["bin_dir"]:
                    input_file_i = os.path.join(params['bin_dir'], params['bin_path_input'][i])
                else:
                    input_file_i = os.path.join(params['case_path'], params['bin_path_input'][i])
            else:
                if params["bin_dir"]:
                    input_file_i = os.path.join(params['bin_dir'],
                                                params['case_name'] + '_input_' + str(i) + '.bin')
                else:
                    input_file_i = os.path.join(params['case_path'],
                                                params['case_name'] + '_input_' + str(i) + '.bin')
            try:
                if params['range_input'][i][0] == "null":
                    input_data_i = np.fromfile(input_file_i, np_dtype)
                else:
                    if params['dtype_input'][i] in ['fp4_e1m2', 'fp4_e2m1', 'float4_e1m2', 'float4_e2m1']:
                        params['shape_input'][i][-1] = params['shape_input'][i][-1] >> 1  # 最后一维长度缩短一半
                        input_data_i = np.fromfile(input_file_i, np_dtype).reshape(
                            params['shape_input'][i])
                    else:
                        if params['dtype_input'][i] not in ['complex32'] or "Cast" not in params['op_name']:
                            input_data_i = np.fromfile(input_file_i, np_dtype).reshape(
                                    params['shape_input'][i])
                        else:
                            input_data_i = np.fromfile(input_file_i, np_dtype)
                overflows_count += input_data_i[np.isinf(
                    input_data_i)].size + input_data_i[np.isnan(input_data_i)].size
                input_tensor_list.append(input_data_i)
            except FileNotFoundError:
                print(traceback.format_exc())
                return ["Failed", 0, 1]
                return False


    torch_tensor_list = list()
    for i in range(len(input_tensor_list)):
        input_tensor = input_tensor_list[i]
        input_tensor = input_tensor_list[i]

        if any("Tensor" in _key for _key in params.keys()):
            torch_tensor_list.append(input_tensor.to('npu'))
        else:
            if str(input_tensor.dtype) == "uint64":
                print(type(input_tensor), "uint64转int64")
                # print(input_tensor)
                input_tensor_i = input_tensor.astype(np.int64)
                torch_tensor_list.append(input_tensor_i)

            elif str(input_tensor.dtype) == "bfloat16":
                input_tensor_i = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
                torch_tensor_list.append(input_tensor_i.to('npu'))

            elif input_tensor.dtype == "float8_e5m2":
                print("^^^^^^^^^^^^^^^float8_e5m2^^^^^^^^^^^^^")
                input_tensor_i = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.float8_e5m2)
                torch_tensor_list.append(input_tensor_i.to('npu'))

            elif input_tensor.dtype == "float8_e4m3fn":
                print("^^^^^^^^^^^^^^float8_e4m3fn^^^^^^^^^^^^^^^^^")
                input_tensor_i = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.float8_e4m3fn)
                torch_tensor_list.append(input_tensor_i.to('npu'))
            else:
                input_tensor_i = torch.from_numpy(input_tensor)
                torch_tensor_list.append(input_tensor_i.to('npu'))
    output_data = aclnn_op_graph_func_sfa(torch_tensor_list, params)

    # if isinstance(output_data, list) or isinstance(output_data, tuple):
    #     output_value = []
    #     get_output_list(True, output_data, output_value)
    #     output_list = []
    #     for i in range(len(output_value)):
    #         tensor_tmp = output_value[i].to("cpu")
    #         if params["dtype_output"][i] == "bf16":
    #             tensor_tmp = tensor_tmp.float()
    #         tensor_tmp = tensor_tmp.detach().numpy()
    #         output_list.append(tensor_tmp)
    #     return output_list
    # elif isinstance(output_data, bool):
    #     output_data = output_data
    # else:
    #     output_data = output_data.to("cpu")
    #     if params["dtype_output"][0] == "bf16":
    #         output_data = output_data.float()
    #     output_data = output_data.detach().numpy()
    return output_data


def _call_npu_op(npu_inputs, input_dict):
    # 单算子直调路径，参数组织与 torch_npu 接口保持一致。
    kv_torch_dtype = _get_kv_torch_dtype(input_dict.get("kv_dtype", "hifloat8"))
    return torch_npu.npu_kv_quant_sparse_flash_attention(
        query=npu_inputs["query"],
        key=npu_inputs["key"],
        value=npu_inputs["value"],
        sparse_indices=npu_inputs["sparse_indices"],
        scale_value=input_dict["scale_value"],
        key_quant_mode=input_dict["key_quant_mode"],
        value_quant_mode=input_dict["value_quant_mode"],
        key_dequant_scale=npu_inputs.get("key_dequant_scale"),
        value_dequant_scale=npu_inputs.get("value_dequant_scale"),
        block_table=npu_inputs.get("block_table"),
        actual_seq_lengths_query=npu_inputs.get("actual_seq_lengths_query"),
        actual_seq_lengths_kv=npu_inputs.get("actual_seq_lengths_kv"),
        sparse_block_size=input_dict.get("sparse_block_size", 1),
        layout_query=input_dict["layout_query"],
        layout_kv=input_dict["layout_kv"],
        sparse_mode=input_dict.get("sparse_mode", 3),
        attention_mode=input_dict.get("attention_mode", 0),
        quant_scale_repo_mode=input_dict.get("quant_scale_repo_mode", 1),
        tile_size=input_dict.get("tile_size", 128),
        rope_head_dim=input_dict.get("rope_head_dim", 64),
        key_dtype=kv_torch_dtype,
        value_dtype=kv_torch_dtype,
        pre_tokens=input_dict.get("pre_tokens", (1 << 63) - 1),
        next_tokens=input_dict.get("next_tokens", (1 << 63) - 1),
    )


class Network(torch.nn.Module):
    # 图模式下的最小封装，便于 torch.compile 捕获整个调用。
    def forward(self, query, key, value, sparse_indices,
                scale_value, key_quant_mode, value_quant_mode,
                key_dequant_scale, value_dequant_scale, block_table,
                actual_seq_lengths_query, actual_seq_lengths_kv,
                sparse_block_size, layout_query, layout_kv,
                sparse_mode, attention_mode, quant_scale_repo_mode,
                tile_size, rope_head_dim, key_dtype, value_dtype,
                pre_tokens, next_tokens):
        return torch_npu.npu_kv_quant_sparse_flash_attention(
            query=query,
            key=key,
            value=value,
            sparse_indices=sparse_indices,
            scale_value=scale_value,
            key_quant_mode=key_quant_mode,
            value_quant_mode=value_quant_mode,
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
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
        )


def test_qsfa_process_ci(test_data, device_id=0):
    # single/batch 默认走该路径，直接执行 aclnn 单算子。
    params = test_data['params']
    input_dict = test_data["input"]
    torch_npu.npu.set_device(device_id)
    npu_inputs = _load_inputs_to_npu(input_dict)
    print("test_data:", params)
    print("npu_kv_quant_sparse_flash_attention (CI mode) ...")
    npu_result = _call_npu_op(npu_inputs, input_dict)
    torch.npu.synchronize()
    return npu_result, test_data["cpu_output"]


def test_qsfa_process_graph(test_data, device_id=0):
    # graph 模式下先编译，再用同一组输入做回放。
    params = test_data['params']
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
    kv_torch_dtype = _get_kv_torch_dtype(input_dict.get("kv_dtype", "hifloat8"))

    print("test_data:", params)
    print("npu_kv_quant_sparse_flash_attention (graph mode) ...")
    npu_result = npu_model(
        query=npu_inputs["query"],
        key=npu_inputs["key"],
        value=npu_inputs["value"],
        sparse_indices=npu_inputs["sparse_indices"],
        scale_value=input_dict["scale_value"],
        key_quant_mode=input_dict["key_quant_mode"],
        value_quant_mode=input_dict["value_quant_mode"],
        key_dequant_scale=npu_inputs.get("key_dequant_scale"),
        value_dequant_scale=npu_inputs.get("value_dequant_scale"),
        block_table=npu_inputs.get("block_table"),
        actual_seq_lengths_query=npu_inputs.get("actual_seq_lengths_query"),
        actual_seq_lengths_kv=npu_inputs.get("actual_seq_lengths_kv"),
        sparse_block_size=input_dict.get("sparse_block_size", 1),
        layout_query=input_dict["layout_query"],
        layout_kv=input_dict["layout_kv"],
        sparse_mode=input_dict.get("sparse_mode", 3),
        attention_mode=input_dict.get("attention_mode", 0),
        quant_scale_repo_mode=input_dict.get("quant_scale_repo_mode", 1),
        tile_size=input_dict.get("tile_size", 128),
        rope_head_dim=input_dict.get("rope_head_dim", 64),
        key_dtype=kv_torch_dtype,
        value_dtype=kv_torch_dtype,
        pre_tokens=input_dict.get("pre_tokens", (1 << 63) - 1),
        next_tokens=input_dict.get("next_tokens", (1 << 63) - 1),
    )
    torch.npu.synchronize()
    return npu_result, test_data["cpu_output"]
