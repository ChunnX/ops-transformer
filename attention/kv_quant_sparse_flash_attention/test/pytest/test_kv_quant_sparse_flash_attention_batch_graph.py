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

import pytest
import torch

import utils


TESTCASE_PATH = "../../pt_files/"
RESULT_PATH = Path("result.xlsx")
DEVICE_ID = 0

if os.path.isdir(TESTCASE_PATH):
    TESTCASE_FILES = sorted(
        os.path.join(TESTCASE_PATH, filename)
        for filename in os.listdir(TESTCASE_PATH)
        if filename.endswith(".pt")
    )
else:
    TESTCASE_FILES = []

if not TESTCASE_FILES:
    pytest.skip("No batch testcase `.pt` files found, please generate them first.", allow_module_level=True)


@pytest.mark.graph
@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_kv_quant_sparse_flash_attention_graph(testcase_file):
    test_data = torch.load(testcase_file, map_location="cpu")
    utils.qsfa_run_npu(test_data, device_id=DEVICE_ID, result_path=RESULT_PATH, graph_mode=True)
