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

import os
from pathlib import Path
import concurrent.futures
import pytest
import torch
import utils


TESTCASE_PATH = "./pt_files/"
RESULT_PATH = Path("result.xlsx")
DEVICE_ID = 0

locals()["testcase_files"] = []
if os.path.isdir(TESTCASE_PATH):
    pt_files = [f for f in os.listdir(TESTCASE_PATH) if f.endswith('.pt')]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {TESTCASE_PATH}")
    else:
        print(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:
            filepath = os.path.join(TESTCASE_PATH, pt_file)
            locals()["testcase_files"].append(filepath)
else:
    print(f"错误: 输出目录不存在: {TESTCASE_PATH}")


def execute_qsfa(testcase_files):
    # 从 pt 文件加载已生成好的输入，并执行一次 NPU 回放。
    test_data = torch.load(testcase_files, map_location="cpu")
    utils.qsfa_run_npu(test_data, DEVICE_ID, RESULT_PATH)


@pytest.mark.ci
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"])
def test_kv_quant_sparse_flash_attention(testcase_files):
    # batch 模式逐个消费 pt 用例文件。
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(execute_qsfa, testcase_files)
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"当前用例线程执行失败")
