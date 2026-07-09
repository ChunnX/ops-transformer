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
import random

import numpy as np
import torch
import torch_npu
from test_quant_lightning_indexer_v2_paramset import ENABLED_PARAMS
import result_compare_method
import quant_lightning_indexer_v2_golden
import pytest

TEST_SEED = 3

param_names = [
    "batch_size", "q_seq", "k_seq", "q_t_size", "k_t_size", "q_head_num", "k_head_num","head_dim",
    "block_size", "block_num", "qk_dtype", "weight_dtype", "dequant_dtype", "actual_seq_dtype", "cu_seqlens_q",
    "cu_seqlens_k", "seqused_q", "seqused_k", "cmp_residual_k", "max_seqlen_q", "quant_mode",
    "layout_query", "layout_key", "sparse_count", "sparse_mode",  "query_datarange", "key_datarange",
    "weights_datarange", "q_scale_datarange", "k_scale_datarange", "cmp_ratio", "return_value",
    "output_idx_offset", "run_mode"
]

param_combinations = []
for params in ENABLED_PARAMS:
    param_values = []
    for name in param_names:
        if name == "weight_dtype":
            param_values.append(params.get(name, params["dequant_dtype"]))
        else:
            param_values.append(params.get(name, ["eager"]))
    for combo in itertools.product(*param_values):
        param_combinations.append(dict(zip(param_names, combo)))


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_qliv2(param_combinations):   # Init params and tensors
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
    weight_dtype = param_combinations['weight_dtype']
    dequant_dtype = param_combinations['dequant_dtype']
    actual_seq_dtype = param_combinations['actual_seq_dtype']
    cu_seqlens_q = param_combinations['cu_seqlens_q']
    cu_seqlens_k = param_combinations['cu_seqlens_k']
    seqused_q = param_combinations['seqused_q']
    seqused_k = param_combinations['seqused_k']
    cmp_residual_k = param_combinations['cmp_residual_k']
    max_seqlen_q = param_combinations['max_seqlen_q']
    quant_mode = param_combinations['quant_mode']
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
    return_value = param_combinations['return_value']
    output_idx_offset = param_combinations['output_idx_offset']
    run_mode = param_combinations['run_mode']
    torch_npu.npu.set_device(0)
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    torch.npu.manual_seed_all(TEST_SEED)
    print(f"[single stage] seed_ready seed={TEST_SEED}", flush=True)
    test_data = batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size,\
                block_num, qk_dtype, weight_dtype, dequant_dtype, actual_seq_dtype, cu_seqlens_q, cu_seqlens_k, seqused_q,\
                seqused_k, cmp_residual_k, max_seqlen_q, quant_mode, layout_query, layout_key, sparse_count,\
                sparse_mode, query_datarange, key_datarange, weights_datarange, q_scale_datarange,\
                k_scale_datarange, cmp_ratio, return_value, output_idx_offset

    # Get CPU golden and NPU result
    print(f"[single stage] qliv2_output_start run_mode={run_mode}", flush=True)
    if run_mode == "eager":
        cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value = quant_lightning_indexer_v2_golden.qliv2_output_single(test_data)
    elif run_mode == "graph":
        import quant_lightning_indexer_v2_acl_graph
        cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value = quant_lightning_indexer_v2_acl_graph.qliv2_output_acl_graph(test_data)
    else:
        raise ValueError(f"unsupported run_mode: {run_mode}")
    print("[single stage] qliv2_output_done", flush=True)
    # print("npu_result", npu_result)
    # print("cpu_result:", cpu_result)
    # Compare result accuracy
    print("[single stage] index_compare_start", flush=True)
    result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result, topk_value, output_idx_offset, test_data)
    print("[single stage] index_compare_done", flush=True)
    print("result", result)
    print("result", fulfill_percent)
    if return_value:
        print("[single stage] return_value_compare_start", flush=True)
        result_return_value, fulfill_precent_return_value = result_compare_method.check_result_return_value(
            cpu_topk_value, npu_topk_value, test_data, cpu_result, npu_result,
            topk_value, output_idx_offset)
        print("[single stage] return_value_compare_done", flush=True)
        print(f"result_return_value: {result_return_value}")
        print(f"result_return_value: {fulfill_precent_return_value}")
        assert result_return_value == "Pass", (
            f"return value mismatch, fulfill_percent={fulfill_precent_return_value}"
        )
    assert result == "Pass", f"index result mismatch, fulfill_percent={fulfill_percent}"
