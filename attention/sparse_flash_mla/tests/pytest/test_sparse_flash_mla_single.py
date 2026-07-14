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

import result_compare_method
import utils
import check_valid_param
import itertools
import torch
import torch_npu
from sparse_flash_mla_paramset import ENABLED_PARAMS
import sparse_flash_mla_golden
import os
import pytest
from batch import sparse_flash_mla_process

save_pt = False
pt_save_path = "smla_testcase"

# 处理所有参数组合
result_path = os.getenv("SMLA_RESULT_SAVE_PATH", './result/smla_result.xlsx')
param_combinations = utils.generate_param_combinations(ENABLED_PARAMS)

@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_example(param_combinations):
    test_data = utils.fill_none_params(param_combinations)
    torch_npu.npu.set_device(0)

    try:
        check_valid_param.check_valid_param(test_data)
    except ValueError as e:
        pytest.skip(f"输入参数校验失败:{e}")

    print("test_data:", test_data)
    # 获得cpu结果(真值)和算子结果（测试值）
    input_data = sparse_flash_mla_golden.generate_and_save_testdata(test_data, save_pt=save_pt, save_path=pt_save_path)

    testcase_name = test_data.get('testcase_name', 'unknown')
    npu_error_msg = None
    try:
        npu_result, softmax_lse = sparse_flash_mla_process.call_npu(input_data)
    except Exception as e:
        npu_error_msg = str(e)
        print("NPU ERROR：", npu_error_msg)
        utils.save_result('NPU ERROR', 0, test_data, result_path)
        pytest.fail(f"用例执行失败:{testcase_name} NPU ERROR: {npu_error_msg}")

    print("npu_result.size():", npu_result.size())
    # 结果精度对比
    result, fulfill_percent = result_compare_method.check_result(input_data['cpu_output'], npu_result)
    if test_data.get("return_softmax_lse", False):
        result, fulfill_percent = result_compare_method.check_result(input_data['softmax_lse'], softmax_lse)
    # 记录结果
    utils.save_result(result, fulfill_percent, test_data, result_path)

    if result == "Failed":
        pytest.fail(f"用例精度失败:{testcase_name} 精度:{fulfill_percent:.2f}%")
