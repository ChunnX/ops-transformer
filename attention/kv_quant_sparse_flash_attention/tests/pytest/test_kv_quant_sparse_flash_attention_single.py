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
import result_compare_method
import kv_quant_sparse_flash_attention_golden

PT_SAVE_PATH = "./pt_files/"
DEVICE_ID = 0
RUN_NPU = True
SAVE_PT = True
RESULT_PATH = Path("result.xlsx")
PARAM_COMBINATION_SET = utils.combin_params(ENABLED_PARAMS)
case_id = 0


def execute_qsfa(param_combination):
    global case_id
    params = utils.convert_param_combination_to_cs_format(param_combination)
    input_tensor_dict = utils.generate_input_tensors(params)
    cpu_result, _, _ = utils.compute_cpu(input_tensor_dict, params)
    if SAVE_PT:
        testcase_name = params["case_name"]
        input_data = {
            "Testcase_Name": testcase_name,
            "params": params,
            "input": input_tensor_dict,
            "cpu_output": cpu_result,
        }
        kv_quant_sparse_flash_attention_golden._save_test_case(input_data, PT_SAVE_PATH)
    if RUN_NPU:
        from batch import kv_quant_sparse_flash_attention_process
        npu_result = kv_quant_sparse_flash_attention_process.call_npu(input_tensor_dict, params)
        result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result)
    case_id += 1


@pytest.mark.ci
@pytest.mark.parametrize("param_combination", PARAM_COMBINATION_SET)
def test_kv_quant_sparse_flash_attention(param_combination):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(execute_qsfa, param_combination)
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"当前用例线程执行失败")
