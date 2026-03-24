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
import torch
import check_valid_param
import kv_quant_sparse_attn_sharedkv_golden
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import utils
import argparse
import concurrent.futures

pt_save_path = "qsas_testcase"
device_id = 0
run_npu = False
save_pt = True
result_path = Path('result.xlsx')

# 读取表格
ENABLED_PARAMS = utils.load_excel_test_cases("./excel/example.xlsx", "decode")
param_combination_set = utils.combin_params(ENABLED_PARAMS, pytest_paramset=False)

case_id = 0
def execute_sas(param_combination):
    global case_id
    utils.sas(case_id, param_combination, pt_save_path, device_id, run_npu, save_pt, result_path)
    case_id += 1

@pytest.mark.ci
@pytest.mark.parametrize("param_combination", param_combination_set)
def test_sparse_attn_sharedkv(param_combination):
    # 线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(execute_sas, param_combination)
        # 等待并获取结果
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"当前用例线程执行失败")