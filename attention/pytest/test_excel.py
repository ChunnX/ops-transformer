#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
from functools import partial
import pandas as pd 
import torch
import torch_npu
import check_valid_param
import gqa_no_quant_bnsd_bsnd
import gqa_no_quant_bnsd_bsnd_ge
import pytest

DEBUG_ON = 0


def load_excel_test_cases():
    # 加载测试用例
    filename = os.getenv('EXCEL_FILE')
    sheetname = os.getenv('SHEET_NAME', 'IFA_FIA_Case') # 优先使用环境变量，如果没有则使用默认值

    if not filename:
        pytest.skip("\nset EXCEL_FILE=xxx.xlsx(file path), \
        eg: EXCLE_FILE=file_path python3 -m pytest -rA -s test_excel.py", allow_module_level=True)

    if not os.path.exists(filename):
        pytest.skip(f"Excel file: {filename} not exist!", allow_module_level=True)
    
    try:
        df = pd.read_excel(filename, sheet_name=sheetname)

        test_cases = []
        for _, row in df.iterrows():
            test_cases.append((row['Testcase_Name'], row['inputLayout'], row['q_shape'], row['q_dtype'], \
                row['k_shape'], row['k_cache_shape'], row['actual_seq_lengths_kv'], row['blockSize'], \
                row['scaleValue'], row['kn_pre'], row['kn_nxt']))

        return test_cases
    
    except Exception as e:
        pytest.skip(f"failed to read excel file: {e}", allow_module_level=True)
        return None


def convert_excel_pytest(test_data):
    testcase_name, input_layout, q_shape, q_dtype, k_shape, \
    k_cache_shape, actual_seq_lengths_kv, blocksize, scale_value, kn_pre, kn_nxt = test_data

    # 解析形状
    q_shape_list = [int(x.strip()) for x in q_shape.split(',')]
    kv_shape_list = [int(x.strip()) for x in k_shape.split(',')]
    kv_cache_shape_list = [int(x.strip()) for x in k_cache_shape.split(',')]
    kv_actual_seq_list = [int(x.strip()) for x in actual_seq_lengths_kv.split(',')]

    if input_layout == "BNSD":
        batch_size, q_head_num, q_seq, head_dim = q_shape_list
        _, kv_head_num, kv_seq, _ = kv_shape_list
    elif input_layout == "BSND":
        batch_size, q_seq, q_head_num, head_dim = q_shape_list
        _, kv_seq, kv_head_num, _ = kv_shape_list
    else:
        raise ValueError("Right now only support: BNSD, BSND")
    
    if q_dtype == "BF16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    scaled_value = float(scale_value)

    if len(kv_cache_shape_list) == 3:
        cache_layout = "BBH"
        _, block_size, _ = kv_cache_shape_list
    else:
        cache_layout = "BNBD"
        _, _, block_size, _ = kv_cache_shape_list

    act_seq_len = [q_seq] * batch_size
    convert_params = (
        batch_size, q_head_num, kv_head_num, q_seq, kv_seq, head_dim, dtype,
        act_seq_len, kv_actual_seq_list, block_size, cache_layout, scaled_value
    )
    return convert_params


def _base_flash_attention_test(test_data, test_func, marker="ci"):
    # 基础测试逻辑
    testcase_name, input_layout, q_shape, q_dtype, k_shape, \
    k_cache_shape, actual_seq_lengths_kv, blocksize, scale_value, kn_pre, kn_nxt = test_data

    convert_params = convert_excel_pytest(test_data)
    
    batch_size, q_head_num, kv_head_num, q_seq, kv_seq, head_dim, dtype, \
    act_seq_len, act_seq_len_kv, block_size, cache_layout, scaled_value = convert_params

    torch_npu.npu.set_device(0)

    try:
        params = (
            batch_size, q_head_num, kv_head_num, q_seq, kv_seq, head_dim, dtype, input_layout
            )
        check_valid_param.validate_config(params)
    except ValueError as e:
        pytest.skip(f"参数校验失败: {e}")
    if input_layout == "BNSD" or input_layout == "BSND":
        params = (
            batch_size, q_head_num, kv_head_num, q_seq, kv_seq, head_dim, dtype, input_layout, 
            act_seq_len, act_seq_len_kv, scaled_value, block_size, cache_layout
            )
        expect, result = test_func(params)
    check_valid_param.check_result(expect, result)


# 加载测试数据
TEST_CASES = load_excel_test_cases()


@pytest.mark.ci
@pytest.mark.parametrize(
    "test_data", 
    TEST_CASES, ids=[f"{case[0]}" for case in TEST_CASES]
    )
def test_flash_attention(test_data):
    _base_flash_attention_test(
        test_data,
        test_func=gqa_no_quant_bnsd_bsnd.test_gqa_no_quant,
        marker="ci"
    )


@pytest.mark.graph
@pytest.mark.parametrize(
    "test_data", 
    TEST_CASES, ids=[f"{case[0]}" for case in TEST_CASES]
    )
def test_flash_attention_ge(test_data):
    _base_flash_attention_test(
        test_data,
        test_func=gqa_no_quant_bnsd_bsnd.test_gqa_no_quant,
        marker="graph"
    )