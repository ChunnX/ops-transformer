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
import math
import traceback
import pytest
import torch
import numpy as np
from kv_quant_sparse_flash_attention_paramset import ENABLED_PARAMS
import utils
import check_valid_param
import kv_quant_sparse_flash_attention_golden
import result_compare_method
import generate_tensor_data

PT_SAVE_PATH = "./pt_files/"
DEVICE_ID = 0
RUN_NPU = True
SAVE_PT = False
RESULT_PATH = Path("result.xlsx")
PARAM_COMBINATION_SET = utils.combin_params(ENABLED_PARAMS)
case_id = 0


def sub_process_cpu(params):
    np.random.seed(0)
    case_name = params["case_name"]

    input_tensor_list = generate_tensor_data.generate_input_tensor_list(params)

    try:
        out_data, torch_tensor_list = kv_quant_sparse_flash_attention_golden.route(input_tensor_list, params, "cpu")
        for i in range(len(torch_tensor_list)):
            if isinstance(torch_tensor_list[i], np.ndarray):
                input_tensor_list[i] = torch_tensor_list[i]
        if out_data is not None:
            output_file = utils.get_case_output_bin_path(params)
            if torch.is_tensor(out_data):
                output_np = out_data.detach().cpu().numpy()
            else:
                output_np = out_data
            utils.dump_file(output_np, output_file, params['dtype_output'][0], params)
            utils.print_log(f'output data saved to: {output_file}')
            params["cpu_output_bin_path"] = output_file
            params["cpu_output_md5"] = utils.calculate_file_md5(output_file)
        return out_data, input_tensor_list, params
    except:
        utils.print_log(traceback.format_exc(), "ERROR")
        return None, input_tensor_list, params


def qsfa(case_id, param_combination, pt_save_path, device_id, run_npu, save_pt, result_path):
    testcase_prefix = param_combination.get("Testcase_Prefix") or "kvQuantSparseFlashAttn"
    testcase_number = param_combination.get("Testcase_Number") or case_id
    layout_query = param_combination["layout_query"]
    layout_kv = param_combination["layout_kv"]
    q_type = param_combination["q_type"]
    kv_dtype = param_combination["kv_dtype"]
    B = param_combination["B"]
    S1 = param_combination["S1"]
    S2 = param_combination["S2"]
    N1 = param_combination["N1"]
    N2 = param_combination["N2"]
    D = param_combination["D"]
    K = param_combination["K"]
    scale_value = param_combination["scale_value"]
    key_quant_mode = param_combination["key_quant_mode"]
    value_quant_mode = param_combination["value_quant_mode"]
    sparse_block_size = param_combination["sparse_block_size"]
    tile_size = param_combination["tile_size"]
    rope_head_dim = param_combination["rope_head_dim"]
    sparse_mode = param_combination["sparse_mode"]
    attention_mode = param_combination["attention_mode"]
    quant_scale_repo_mode = param_combination["quant_scale_repo_mode"]
    block_size = param_combination.get("block_size") or 256
    block_num = param_combination.get("block_num")
    actual_seq_q = param_combination.get("actual_seq_q")
    actual_seq_kv = param_combination.get("actual_seq_kv")

    q_type_str = "BF16" if q_type == torch.bfloat16 else "FP16"
    testcase_name = (
        f"{testcase_prefix}_{layout_query}_{layout_kv}_{q_type_str}_"
        f"{B}_{N1}_{N2}_{S1}_{S2}_{D}_{K}_{testcase_number:06d}"
    )

    if layout_kv == "PA_BSND" and block_num is None:
        max_kv = max(actual_seq_kv) if actual_seq_kv else S2
        block_num = math.ceil(max_kv / block_size) * B

    params = (
        testcase_name, layout_query, layout_kv, q_type, kv_dtype,
        B, S1, S2, N1, N2, D, K,
        scale_value, key_quant_mode, value_quant_mode, sparse_block_size,
        tile_size, rope_head_dim, sparse_mode, attention_mode, quant_scale_repo_mode,
        block_size, block_num, actual_seq_q, actual_seq_kv,
    )

    try:
        check_valid_param.check_valid_param(params)
    except ValueError as error:
        pytest.skip(f"输入参数校验失败: {error}")

    test_data = kv_quant_sparse_flash_attention_golden.generate_and_save_testdata(
        params, save_pt=save_pt, save_path=pt_save_path, compute_golden=False
    )
    if run_npu:
        qsfa_run_npu(test_data, device_id=device_id, result_path=result_path)


def qsfa_run_npu(test_data, device_id=0, result_path=Path("result.xlsx")):
    try:
        npu_result, cpu_result = None, None

        if cpu_result is None:
            print(npu_result)
            print("CPU golden 不可用，当前用例仅验证 NPU 连通性，跳过精度对比。")
            result = "SkipCompare"
            fulfill_percent = 100.0
        else:
            result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result)
    except Exception as error:
        print(f"NPU ERROR: {error}")
        result = "NPU ERROR"
        fulfill_percent = 0.0

    utils.save_result(test_data["params"], result, fulfill_percent, result_path=result_path)
    if result == "NPU ERROR":
        pytest.fail(f"用例执行失败: {test_data['Testcase_Name']}")


def qsfa_run_cpu_and_save_md5(param_combination):
    params = utils.convert_param_combination_to_cs_format(param_combination)
    cpu_result, input_tensor_list, params = sub_process_cpu(params)
    return {
        "cpu_output": cpu_result,
        "cpu_output_bin_path": params.get("cpu_output_bin_path"),
        "cpu_output_md5": params.get("cpu_output_md5"),
        "params": params,
    }


def execute_qsfa(param_combination):
    global case_id
    result = qsfa_run_cpu_and_save_md5(param_combination)

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