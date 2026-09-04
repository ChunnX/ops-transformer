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

import pytest
import torch_npu

import quant_lightning_indexer_golden
import result_compare_method
from test_quant_lightning_indexer_paramset import ENABLED_PARAMS


PARAM_NAMES = [
    "batch_size",
    "q_seq",
    "k_seq",
    "q_t_size",
    "k_t_size",
    "q_head_num",
    "k_head_num",
    "head_dim",
    "block_size",
    "block_num",
    "qk_dtype",
    "dequant_dtype",
    "actual_seq_dtype",
    "act_seq_q",
    "act_seq_k",
    "query_quant_mode",
    "key_quant_mode",
    "layout_query",
    "layout_key",
    "sparse_count",
    "sparse_mode",
    "query_datarange",
    "key_datarange",
    "weights_datarange",
    "q_scale_datarange",
    "k_scale_datarange",
    "cmp_ratio",
]


def build_param_combinations():
    """Flatten every enabled parameter group into cases visible to pytest."""
    combinations = []
    for params in ENABLED_PARAMS:
        param_values = [params[name] for name in PARAM_NAMES]
        combinations.extend(
            dict(zip(PARAM_NAMES, combo)) for combo in itertools.product(*param_values)
        )
    return combinations


def case_id(params):
    dtype_name = str(params["qk_dtype"]).removeprefix("torch.")
    return (
        f"{params['layout_query']}-b{params['batch_size']}-q{params['q_seq']}-"
        f"qt{params['q_t_size']}-k{params['k_seq']}-top{params['sparse_count']}-{dtype_name}"
    )


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", build_param_combinations(), ids=case_id)
def test_qli(param_combinations):
    batch_size = param_combinations["batch_size"]
    q_seq = param_combinations["q_seq"]
    k_seq = param_combinations["k_seq"]
    q_t_size = param_combinations["q_t_size"]
    k_t_size = param_combinations["k_t_size"]
    q_head_num = param_combinations["q_head_num"]
    k_head_num = param_combinations["k_head_num"]
    head_dim = param_combinations["head_dim"]
    block_size = param_combinations["block_size"]
    block_num = param_combinations["block_num"]
    qk_dtype = param_combinations["qk_dtype"]
    dequant_dtype = param_combinations["dequant_dtype"]
    actual_seq_dtype = param_combinations["actual_seq_dtype"]
    act_seq_q = param_combinations["act_seq_q"]
    act_seq_k = param_combinations["act_seq_k"]
    query_quant_mode = param_combinations["query_quant_mode"]
    key_quant_mode = param_combinations["key_quant_mode"]
    layout_query = param_combinations["layout_query"]
    layout_key = param_combinations["layout_key"]
    sparse_count = param_combinations["sparse_count"]
    sparse_mode = param_combinations["sparse_mode"]
    query_datarange = param_combinations["query_datarange"]
    key_datarange = param_combinations["key_datarange"]
    weights_datarange = param_combinations["weights_datarange"]
    q_scale_datarange = param_combinations["q_scale_datarange"]
    k_scale_datarange = param_combinations["k_scale_datarange"]
    cmp_ratio = param_combinations["cmp_ratio"]

    torch_npu.npu.set_device(0)
    test_data = (
        batch_size,
        q_seq,
        k_seq,
        q_t_size,
        k_t_size,
        q_head_num,
        k_head_num,
        head_dim,
        block_size,
        block_num,
        qk_dtype,
        dequant_dtype,
        actual_seq_dtype,
        act_seq_q,
        act_seq_k,
        query_quant_mode,
        key_quant_mode,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        query_datarange,
        key_datarange,
        weights_datarange,
        q_scale_datarange,
        k_scale_datarange,
        cmp_ratio,
    )

    cpu_result, npu_result, topk_value = (
        quant_lightning_indexer_golden.qli_output_single(test_data)
    )
    result, fulfill_percent = result_compare_method.check_result(
        cpu_result, npu_result, topk_value, test_data
    )

    assert result == "Pass", (
        f"QLI accuracy check failed, fulfill_percent={fulfill_percent:.6f}%"
    )
