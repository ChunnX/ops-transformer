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

import concurrent.futures
from pathlib import Path

import pytest
import torch

import utils


TESTCASE_PATH = Path("./pt_files/")
RESULT_PATH = Path("result.xlsx")
DEVICE_ID = 0


def _collect_testcase_files(testcase_path):
    if not testcase_path.is_dir():
        print(f"错误: 输出目录不存在: {testcase_path}")
        return []

    testcase_files = sorted(testcase_path.glob("*.pt"))
    if not testcase_files:
        print(f"错误: 目录中没有找到.pt文件: {testcase_path}")
        return []

    print(f"找到 {len(testcase_files)} 个测试用例文件")
    return [str(testcase_file) for testcase_file in testcase_files]


def execute_sfa(testcase_file):
    # 从 pt 文件加载已生成好的输入，并执行一次 NPU 回放。
    test_data = torch.load(testcase_file, map_location="cpu")
    utils.sfa_run_npu(test_data, device_id=DEVICE_ID, result_path=RESULT_PATH)


TESTCASE_FILES = _collect_testcase_files(TESTCASE_PATH)


@pytest.mark.ci
@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_sparse_flash_attention_batch(testcase_file):
    # batch 模式逐个消费 pt 用例文件。
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(execute_sfa, testcase_file)
        for completed_future in concurrent.futures.as_completed([future]):
            try:
                completed_future.result()
            except Exception as error:
                pytest.fail(f"当前用例线程执行失败: {error}")
