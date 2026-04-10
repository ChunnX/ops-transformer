#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import concurrent.futures
from pathlib import Path

import pytest

from sparse_flash_attention_paramset import ENABLED_PARAMS
import utils


PT_SAVE_PATH = "./pt_files/"
RESULT_PATH = Path("result.xlsx")
DEVICE_ID = 0
RUN_NPU = True
SAVE_PT = False

PARAM_COMBINATION_SET = utils.combin_params(ENABLED_PARAMS)
case_id = 0


def execute_sfa(param_combination):
    global case_id
    utils.sfa(case_id, param_combination, PT_SAVE_PATH, DEVICE_ID, RUN_NPU, SAVE_PT, RESULT_PATH)
    case_id += 1


@pytest.mark.ci
@pytest.mark.parametrize("param_combination", PARAM_COMBINATION_SET)
def test_sparse_flash_attention_single(param_combination):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(execute_sfa, param_combination)
        for done in concurrent.futures.as_completed([future]):
            try:
                done.result()
            except Exception as error:
                pytest.fail(f"当前用例线程执行失败: {error}")
