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
import utils
import itertools
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pytest
import random
import sparse_flash_mla_process
import sparse_flash_mla_golden
import torch

excel_path = os.getenv("SAS_EXCEL_PATH", "./excel/example.xlsx")
excel_sheet = os.getenv("SAS_EXCEL_SHEET", "SCFA")
ENABLED_PARAMS_FROM_FILE = utils.load_excel_test_cases(excel_path, excel_sheet)
save_path = os.getenv("SAS_PT_SAVE_PATH", "./data")

param_combinations = utils.generate_param_combinations(ENABLED_PARAMS_FROM_FILE, is_save_pt=True)

@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_sparse_flash_mla(param_combinations):   # 初始化参数和tensor
    test_data = utils.generate_case_with_default_param(param_combinations)
    print("data parsed.", test_data)
    print("strat to generate data")
    # 生成测试数据
    input_data = sparse_flash_mla_golden.gen_data(test_data, test_data.get("template_mode"))
    print("strat to save data")
    sparse_flash_mla_golden.save_test_case(input_data, save_path)
