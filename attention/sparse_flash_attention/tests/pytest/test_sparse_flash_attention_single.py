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

from sparse_flash_attention_paramset import ENABLED_PARAMS
import utils


PT_SAVE_PATH = "./pt_files/"
DEVICE_ID = 0
RUN_NPU = True
SAVE_PT = False
RESULT_PATH = Path("result.xlsx")
PARAM_COMBINATION_SET = utils.combin_params(ENABLED_PARAMS)
case_id = 0


def execute_sfa(param_combination):
    # 单用例线程入口：把参数组合交给统一执行函数。
    global case_id
    utils.sfa(case_id, param_combination, PT_SAVE_PATH, DEVICE_ID, RUN_NPU, SAVE_PT, RESULT_PATH)
    case_id += 1


@pytest.mark.ci
@pytest.mark.parametrize("param_combination", PARAM_COMBINATION_SET)
def test_sparse_flash_attention_single(param_combination):
    # single 模式直接基于参数表构造输入并执行回放。
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(execute_sfa, param_combination)
        for completed_future in concurrent.futures.as_completed([future]):
            try:
                completed_future.result()
            except Exception as error:
                pytest.fail(f"当前用例线程执行失败: {error}")
