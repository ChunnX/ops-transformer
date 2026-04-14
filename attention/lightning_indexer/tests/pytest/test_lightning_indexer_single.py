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

import itertools
import torch
import torch_npu
from test_lightning_indexer_paramset import ENABLED_PARAMS
import result_compare_method
import lightning_indexer_golden
import pytest

input_params = {}
param_combination_set = []
for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        input_params[f"param_{key}"] = value
    param_names = [
        "batch_size", "q_seq", "k_seq", "q_t_size", "k_t_size", "q_head_num", "k_head_num","head_dim", 
        "block_size", "block_num", "qk_dtype", "weight_dtype", "actual_seq_dtype", "act_seq_q","act_seq_k",
        "layout_query","layout_key", "sparse_count", "sparse_mode", 
        "query_datarange","key_datarange","weights_datarange","q_scale_datarange","k_scale_datarange","cmp_ratio"
    ]
    param_values = [
        input_params["param_batch_size"],
        input_params["param_q_seq"],
        input_params["param_k_seq"],
        input_params["param_q_t_size"],
        input_params["param_k_t_size"],
        input_params["param_q_head_num"], 
        input_params["param_k_head_num"],
        input_params["param_head_dim"],
        input_params["param_block_size"],
        input_params["param_block_num"],
        input_params["param_qk_dtype"],
        input_params["param_weight_dtype"],
        input_params["param_actual_seq_dtype"],
        input_params["param_act_seq_q"],
        input_params["param_act_seq_k"],
        input_params["param_layout_query"],
        input_params["param_layout_key"],
        input_params["param_sparse_count"],
        input_params["param_sparse_mode"],
        input_params["param_query_datarange"],
        input_params["param_key_datarange"],
        input_params["param_weights_datarange"],
        input_params["param_q_scale_datarange"],
        input_params["param_k_scale_datarange"],
        input_params["param_cmp_ratio"]
    ]

    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        param_combination_set.append(param_dict)

    print(f"共生成 {len(param_combination_set)} 种参数组合。")

@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combination_set)
def test_qli(param_combinations):   # 初始化参数和tensor
    batch_size = param_combinations['batch_size']
    q_seq = param_combinations['q_seq']
    k_seq = param_combinations['k_seq']
    q_t_size = param_combinations['q_t_size']
    k_t_size = param_combinations['k_t_size']
    q_head_num = param_combinations['q_head_num']
    k_head_num = param_combinations['k_head_num']
    head_dim = param_combinations['head_dim']
    block_size = param_combinations['block_size']
    block_num = param_combinations['block_num']
    qk_dtype= param_combinations['qk_dtype']
    weight_dtype= param_combinations['weight_dtype']
    actual_seq_dtype = param_combinations['actual_seq_dtype']
    act_seq_q = param_combinations['act_seq_q']
    act_seq_k = param_combinations['act_seq_k']
    layout_query = param_combinations['layout_query']
    layout_key = param_combinations['layout_key']
    sparse_count = param_combinations['sparse_count']
    sparse_mode = param_combinations['sparse_mode']
    query_datarange = param_combinations['query_datarange']
    key_datarange = param_combinations['key_datarange']
    weights_datarange = param_combinations['weights_datarange']
    q_scale_datarange = param_combinations['q_scale_datarange']
    k_scale_datarange = param_combinations['k_scale_datarange']
    cmp_ratio = param_combinations['cmp_ratio']
    torch_npu.npu.set_device(0)
    test_data = batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num,\
                qk_dtype, weight_dtype, actual_seq_dtype, act_seq_q, act_seq_k, layout_query,\
                layout_key, sparse_count, sparse_mode, query_datarange, key_datarange, weights_datarange, q_scale_datarange,\
                k_scale_datarange, cmp_ratio

    # 获得cpu结果(真值)和算子结果（测试值）
    cpu_result, npu_result, topk_value = lightning_indexer_golden.qli_output_single(test_data)
    # 结果精度对比
    result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result, topk_value, test_data)
    print("result", result)
    print("result", fulfill_percent)
    if result != "Pass":
        pytest.fail(f"用例执行失败")