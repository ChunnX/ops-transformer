#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
from functools import partial
from recurrent_gated_delta_rule_golden import cpu_recurrent_gated_delta_rule
import pandas as pd
import numpy as np
import torch
import pytest
import random
import math
import ast
import argparse

def load_excel_test_cases(excel_file_path: str, sheetname: str):

    """
    从 Excel 文件加载测试用例。

    参数:
        excel_file_path (str): Excel 文件的路径。
        sheetname (str, optional): 工作表名称。若未提供，则默认 'Sheet1'。

    返回:
        list[tuple]: 测试用例元组列表，每个元组包含 20+ 个字段。
                       若失败或跳过，则返回空列表。
    """
    # 优先使用传入的 sheetname，否则尝试从环境变量获取
    if sheetname is None:
        sheetname = 'Sheet1'

    # 检查文件是否存在
    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        # 读取 Excel 文件的指定 sheet
        df = pd.read_excel(excel_file_path, sheet_name=sheetname)
        df = df.replace({np.nan: None, pd.NA: None})

        # 定义必需的列名
        required_columns = [
            'Testcase_Name', "batch_size", "mtp", "nk", "nv", "dk", "dv", "actual_seq_lengths", "ssm_state_indices", "has_gamma",
            "has_gamma_k", "has_num_accepted_tokens", "scale_value", "num_accepted_tokens", "block_num", "data_type",
            "query_datarange","key_datarange","value_datarange","gamma_datarange","gamma_k_datarange",
            "beta_datarange", "state_datarange"
        ]

        # 检查是否缺少必要列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            pytest.skip(f"Missing required columns in Excel: {missing_cols}", allow_module_level=True)

        # 构建测试用例列表
        test_cases = []
        for _, row in df.iterrows():
            test_cases.append((
                row['Testcase_Name'],
                row['batch_size'], 
                row['mtp'], 
                row['nk'], 
                row['nv'], 
                row['dk'], 
                row['dv'], 
                row['actual_seq_lengths'], 
                row['ssm_state_indices'], 
                row['has_gamma'], 
                row['has_gamma_k'], 
                row['has_num_accepted_tokens'], 
                row['scale_value'], 
                row['num_accepted_tokens'], 
                row['block_num'], 
                row['data_type'], 
                row['query_datarange'], 
                row['key_datarange'], 
                row['value_datarange'], 
                row['gamma_datarange'], 
                row['gamma_k_datarange'], 
                row['beta_datarange'], 
                row['state_datarange']
            ))

        return test_cases

    except Exception as e:
        pytest.skip(f"Failed to read Excel file: {e}", allow_module_level=True)
        return None

class Generalized_operator():
    def forward(self, q, k, v, state, beta, scale_value, act_seq_len, ssm_state_indices,
                num_accepted_tokens=None, g=None, gk=None):

        return cpu_recurrent_gated_delta_rule(
            q, k, v, state, beta, scale_value, act_seq_len, ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens, g=g, gk=gk)

def rand_range(shape, data_range=[-10, 10], dtype=torch.bfloat16, device=None):
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(shape, dtype=dtype, device=device)

def recurrent_gated_delta_rule_output_single(data_case):
    casename = data_case[0]
    params = data_case[1:]

    B, mtp, nk, nv, dk, dv, actual_seq_lengths, ssm_state_indices, has_gamma, \
    has_gamma_k, has_num_accepted_tokens, scale_value, num_accepted_tokens, block_num, data_type, \
    query_datarange, key_datarange, value_datarange, gamma_datarange, gamma_k_datarange, \
    beta_datarange, state_datarange = params

    if data_type == "float16" or data_type == "FP16" or data_type == "fp16":
        data_type = torch.float16
    elif data_type == "bfloat16" or data_type == "BF16" or data_type == "bf16":
        data_type = torch.bfloat16

    # 处理B+1
    # print(f"===== {actual_seq_lengths} =====")
    if isinstance(actual_seq_lengths, int):
        actual_seq_lengths = [actual_seq_lengths]
    elif isinstance(actual_seq_lengths, list):
        actual_seq_lengths = actual_seq_lengths
    elif actual_seq_lengths != None:
        actual_seq_lengths = [int(x.strip()) for x in actual_seq_lengths.split(',')]

    # print(f"===== {ssm_state_indices} =====")
    if isinstance(ssm_state_indices, int):
        ssm_state_indices = [ssm_state_indices]
    elif isinstance(ssm_state_indices, list):
        ssm_state_indices = ssm_state_indices
    elif ssm_state_indices != None and B !=1:
        ssm_state_indices = [int(x.strip()) for x in ssm_state_indices.split(',')]
    elif ssm_state_indices != None:
        ssm_state_indices = [int(ssm_state_indices)]

    # print(f"===== {num_accepted_tokens} =====")
    if isinstance(num_accepted_tokens, int):
        num_accepted_tokens = [num_accepted_tokens]
    elif isinstance(num_accepted_tokens, list):
        num_accepted_tokens = num_accepted_tokens
    elif num_accepted_tokens != None and B !=1:
        num_accepted_tokens = [int(x.strip()) for x in num_accepted_tokens.split(',')]
    elif num_accepted_tokens != None:
        num_accepted_tokens = [int(num_accepted_tokens)]

    query_datarange = [float(x.strip()) for x in query_datarange.split(',')]
    key_datarange = [float(x.strip()) for x in key_datarange.split(',')]
    value_datarange = [float(x.strip()) for x in value_datarange.split(',')]
    gamma_datarange = [float(x.strip()) for x in gamma_datarange.split(',')]
    gamma_k_datarange = [float(x.strip()) for x in gamma_k_datarange.split(',')]
    beta_datarange = [float(x.strip()) for x in beta_datarange.split(',')]
    state_datarange = [float(x.strip()) for x in state_datarange.split(',')]

    block_num = B * mtp if block_num == None else block_num
    if scale_value == None:
        scale_value = dk ** -0.5
    if actual_seq_lengths == None:
        actual_seq_lengths = [mtp] * B
    if has_num_accepted_tokens == True and num_accepted_tokens == None:
        num_accepted_tokens = torch.tensor([torch.randint(0, h, (1,)) for h in actual_seq_lengths]) + 1
    T = int(sum(actual_seq_lengths))
    if ssm_state_indices == None:
        ssm_state_indices = torch.arange(T, dtype=torch.int32)
    # ======================== set input params finish ========================
    # ======================== check input params start ========================
    if len(actual_seq_lengths) != B:
        print(f"Error: the len of seqused is {len(actual_seq_lengths)}, it should be B({B})")
        return
    if has_num_accepted_tokens == True and len(num_accepted_tokens) != B:
        print(f"Error: the len of num_accepted_tokens is {len(num_accepted_tokens)}, it should be B({B})")
        return
    for i in range(B):
        act_seq = actual_seq_lengths[i]
        if act_seq < 0 or act_seq > mtp:
            print(f"Error: actual_seq_lengths[{i}] is {act_seq}, it should >= 0 and <= mtp({mtp})")
            return
        if has_num_accepted_tokens == True:
            accepted_token = num_accepted_tokens[i]
            if accepted_token < 1 or accepted_token > act_seq:
                print(f"Error: num_accepted_tokens[{i}] is {accepted_token}, it should >= 1 and <= actual_seq_lengths[{i}]({act_seq})")
                return
    if len(ssm_state_indices) != T:
        print(f"Error: the len of ssm_state_indices is {len(ssm_state_indices)}, it should be T({T})")
        return
    for i in range(T):
        idx = ssm_state_indices[i]
        if idx < 0 or idx > block_num:
            print(f"Error: ssm_state_indices[{i}] is {idx}, it should >= 0 and < block_num({block_num})")
            return
    # ======================== check input params finish ========================
    # ======================== gen input data start =============================
    query = rand_range((T, nk, dk), query_datarange, data_type)
    key = rand_range((T, nk, dk), key_datarange, data_type)
    value = rand_range((T, nv, dv), value_datarange, data_type)
    g = rand_range((T, nv), gamma_datarange, dtype=torch.float32) if has_gamma == True else None
    gk = rand_range((T, nv, dk), gamma_k_datarange, dtype=torch.float32) if has_gamma_k == True else None
    beta = rand_range((T, nv), beta_datarange, data_type)
    num_accepted_tokens = torch.tensor(num_accepted_tokens, dtype=torch.int32) if has_num_accepted_tokens == True else None
    state = rand_range((block_num, nv, dv, dk), state_datarange, data_type)
    act_seq_len = torch.tensor(actual_seq_lengths, dtype=torch.int32)
    ssm_state_indices = torch.tensor(ssm_state_indices, dtype=torch.int32)

    ### ======================== gen input data finish =============================
    ### ======================== execute cpu start =================================
    cpu_state_result = state.clone()

    test_operator = Generalized_operator()
    cpu_result, cpu_state_result = test_operator.forward(
        query, key, value, state, beta, scale_value, act_seq_len, ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens, g=g, gk=gk)

    output_tensors = {
        "params":params,
        "cpu_result": cpu_result,
        "cpu_state_result": cpu_state_result,
        "query":query,
        "key":key,
        "value":value,
        "state":state,
        "beta":beta,
        "scale_value":scale_value,
        "act_seq_len":act_seq_len, 
        "ssm_state_indices":ssm_state_indices,
        "num_accepted_tokens":num_accepted_tokens,
        "g":g,
        "gk":gk
    }
    return casename, output_tensors

def save_test_case(test_cases, file_path):
    print("正在保存pt文件...")
    # 创建输出目录
    os.makedirs(file_path, exist_ok=True)

    for idx, case in enumerate(test_cases):
        try:
            case_name, output_tensors = recurrent_gated_delta_rule_output_single(case)
            # 生成文件名
            input_filename = f"{case_name}.pt"
            input_filepath = os.path.join(file_path, input_filename)

            # 保存数据
            torch.save(output_tensors, input_filepath)
            print(f"测试用例已保存到: {input_filepath}")

        except Exception as e:
            print(f"[失败] 生成 pt 文件失败: {case[0]} (索引: {idx})")
            print(f"错误详情: {e}")

def main():
    parser = argparse.ArgumentParser(description='recurrent_gated_delta_rule_pt_save.py 接收路径参数')
    parser.add_argument('path1', type=str, help='第一个路径')
    parser.add_argument('path2', type=str, help='第二个路径')
    args = parser.parse_args()
    path1 = args.path1
    path2 = args.path2
    testcase =  load_excel_test_cases(path1, "Sheet1")
    save_test_case(testcase, path2)

if __name__ == "__main__":
    main()

