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

from pathlib import Path
import concurrent.futures
import pytest
from kv_quant_sparse_flash_attention_paramset import ENABLED_PARAMS
import utils
# from batch import kv_quant_sparse_flash_attention_process
import result_compare_method

PT_SAVE_PATH = "./pt_files/"
DEVICE_ID = 0
RUN_NPU = True
SAVE_PT = False
RESULT_PATH = Path("result.xlsx")
PARAM_COMBINATION_SET = utils.combin_params(ENABLED_PARAMS)
case_id = 0


def execute_qsfa(param_combination):
    # 单用例线程入口：把参数组合交给统一执行函数。
    global case_id
    result = utils.qsfa_run_cpu_and_save_md5(param_combination)

    testcase_name = result["params"]["case_name"]
    actual_md5 = result["cpu_output_md5"]

    if actual_md5 is None:
        pytest.fail(f"CPU output MD5 not calculated for {testcase_name}")

    md5_result = utils.verify_md5_with_baseline(testcase_name, actual_md5)
    allow_update = utils.is_baseline_update_enabled()

    if md5_result["status"] == "no_baseline":
        if not allow_update:
            pytest.fail(
                f"Missing baseline for {testcase_name}. "
                "Set UPDATE_BASELINE=1 to create or refresh baseline explicitly."
            )
        utils.save_baseline_md5(testcase_name, actual_md5)
        print(f"[INFO] New baseline created for {testcase_name}: {actual_md5}")
    elif not md5_result["match"]:
        if not allow_update:
            pytest.fail(
                f"MD5 mismatch for {testcase_name}: {md5_result['message']}. "
                "Set UPDATE_BASELINE=1 to refresh baseline explicitly."
            )
        utils.save_baseline_md5(testcase_name, actual_md5)
        print(f"[INFO] Baseline updated for {testcase_name}: {actual_md5}")
    else:
        print(f"[PASS] {testcase_name}: MD5 verified successfully")

    # npu_result = kv_quant_sparse_flash_attention_process.call_npu(input_tensor_list, params)
    # result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result)
    case_id += 1


@pytest.mark.ci
@pytest.mark.parametrize("param_combination", PARAM_COMBINATION_SET)
def test_kv_quant_sparse_flash_attention(param_combination):
    # single 模式直接基于参数表构造输入并执行回放。
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(execute_qsfa, param_combination)
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"当前用例线程执行失败")
