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
import torch
import torch_npu
import pytest
import random
import math
import ast

def test_recurrent_gated_delta_rule_process(filepath, device_id=0):
    # 加载测试数据
    test_data = torch.load(filepath, map_location="cpu",weights_only=False)

    params = test_data['params']
    cpu_result = test_data['cpu_result']
    cpu_state_result = test_data['cpu_state_result']
    print("执行用例：", filepath)
    torch_npu.npu.set_device(device_id)

    query =test_data['query']
    key = test_data['key']
    value = test_data['value']
    state = test_data['state']
    beta = test_data['beta']
    scale_value = test_data['scale_value']
    act_seq_len = test_data['act_seq_len']
    ssm_state_indices = test_data['ssm_state_indices']
    num_accepted_tokens = test_data['num_accepted_tokens']
    g = test_data['g']
    gk = test_data['gk']

    if num_accepted_tokens is not None: 
        num_accepted_tokens = num_accepted_tokens.npu()
    if g is not None: 
        g = torch.tensor(g).npu()
    if gk is not None: 
        gk = torch.tensor(gk).npu()
    initial_state = state.npu().clone()
    #调用recurrent_gated_delta_rule算子
    npu_result = torch_npu.npu_recurrent_gated_delta_rule(
                query.npu(), key.npu(), value.npu(), initial_state, beta=beta.npu(), 
                scale=scale_value, actual_seq_lengths=act_seq_len.npu(), 
                ssm_state_indices=ssm_state_indices.npu(), num_accepted_tokens=num_accepted_tokens, 
                g=g, gk=gk)
    npu_state_result = initial_state.to(torch.float32)
    npu_result = npu_result.to(torch.float32)
    return cpu_result, npu_result, cpu_state_result, npu_state_result, params
