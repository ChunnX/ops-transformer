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

import ast
import math
import result_compare_method
import utils
import itertools
import torch
import torch_npu
from sparse_attn_sharedkv_paramset import ENABLED_PARAMS
import sparse_attn_sharedkv_golden
import os
import pytest
import time
import random
from batch import sparse_attn_sharedkv_process

save_pt = False
pt_save_path = "data"

# 处理所有参数组合
result_path = os.getenv("SAS_RESULT_SAVE_PATH", './result/sas_result.xlsx')
param_combinations = utils.generate_param_combinations(ENABLED_PARAMS)

def fill_random_len(T, SMax, B, has_prefix=True, qk_equal_len=False):
    seqlens = [0] if has_prefix else []
    S = min(SMax, T // B)
    length = S if qk_equal_len else random.randint(0, S)
    seqlens.append(random.randint(0, S))
    for i in range(B - 1):
        length = S if qk_equal_len else random.randint(0, S)
        seqlens.append(seqlens[-1] + random.randint(0, S))
    seqlens[-1] = T
    return seqlens

def generate_case(param_combinations):
    case_param = param_combinations.copy()
    case_param.setdefault('testcase_name', "case_" + str(int(time.time() * 1000000)))
    case_param.update({'q_datarange': ast.literal_eval(param_combinations.get('q_datarange')) if param_combinations.get('q_datarange') is not None else [-10, 10]})
    case_param.update({'ori_kv_datarange': ast.literal_eval(param_combinations.get('ori_kv_datarange')) if param_combinations.get('ori_kv_datarange') is not None else [-10, 10]})
    case_param.update({'cmp_kv_datarange': ast.literal_eval(param_combinations.get('cmp_kv_datarange')) if param_combinations.get('cmp_kv_datarange') is not None else [-10, 10]})

    # 测试用，数据预填充，TND场景下cu_seqlens如果没有传入，根据T和S自动填充随机值
    # 常用参数提取
    layout_q = case_param['layout_q']
    layout_kv = case_param['layout_kv']
    T1 = case_param['T1']
    T2 = case_param['T2']
    B = case_param['B']
    S1 = case_param['S1']
    S2 = case_param['S2']
    cmp_ratio = case_param['cmp_ratio']
    # 数据预填充
    if case_param['cu_seqlens_q'] is None:
        if layout_q == "TND":
            case_param['cu_seqlens_q'] = fill_random_len(T1, S1, B, True, param_combinations['qk_equal_len'])
            print("cu_seqlens_q auto set to: ", case_param['cu_seqlens_q'])
        elif layout_q == "BSND":
            case_param['cu_seqlens_q'] = fill_random_len(B * S1, S1, B, True, True)
            print("cu_seqlens_q auto set to: ", case_param['cu_seqlens_q'])

    if case_param['cu_seqlens_ori_kv'] is None and layout_kv == "TND":
        case_param['cu_seqlens_ori_kv'] = fill_random_len(T2, S2, B, True, param_combinations['qk_equal_len'])
        print("cu_seqlens_ori_kv auto set to: ", case_param['cu_seqlens_ori_kv'])

    if case_param['cu_seqlens_cmp_kv'] is None and layout_kv == "TND" and cmp_ratio is not None:
        case_param['cu_seqlens_cmp_kv'] = [x // cmp_ratio for x in case_param['cu_seqlens_ori_kv']]
        print("cu_seqlens_cmp_kv auto set to: ", case_param['cu_seqlens_cmp_kv'])
        case_param['T3'] = case_param['cu_seqlens_cmp_kv'][-1]

    # maxSeqLen / block_size向上取整
    ori_block_num_per_batch = []
    ori_block_num_sum = 0
    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0

    if layout_kv == "PA_BNBD":
        for cur_ori_act_kv in case_param['seqused_kv']:
            cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / case_param['block_size1'])
            ori_block_num_per_batch.append(cur_ori_kv_block_num)
            ori_block_num_sum += cur_ori_kv_block_num
            if cmp_ratio is not None and case_param['block_size2'] is not None:
                cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
                cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / case_param['block_size2'])
                cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
                cmp_block_num_sum += cur_cmp_kv_block_num
    case_param['block_num1'] = ori_block_num_sum
    case_param['block_num2'] = cmp_block_num_sum
    return case_param

@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_example(param_combinations):
    test_data = generate_case(param_combinations)
    torch_npu.npu.set_device(0)

    print("test_data:", test_data)
    # 获得cpu结果(真值)和算子结果（测试值）
    input_data = sparse_attn_sharedkv_golden.gen_data(test_data)
    if save_pt:
        sparse_attn_sharedkv_golden.save_test_case(input_data, pt_save_path)
    npu_result, softmax_lse = sparse_attn_sharedkv_process.call_npu(input_data)
    print("npu_result.size():", npu_result.size())

    # 结果精度对比
    result, fulfill_percent = result_compare_method.check_result(input_data['cpu_output'], npu_result)
    # 记录结果
    utils.save_result(result, fulfill_percent, test_data, result_path)