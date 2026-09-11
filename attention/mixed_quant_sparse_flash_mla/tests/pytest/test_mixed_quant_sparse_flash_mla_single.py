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
import itertools
import os
from pathlib import Path

import pytest
import torch

import check_valid_param
import mixed_quant_sparse_flash_mla_golden
import utils
from mixed_quant_sparse_flash_mla_paramset import ENABLED_PARAMS
from common.fd_performance_runtime import expand_params, run_single

pt_save_path = os.environ.get("MQSMLA_PT_DIR", "mqsmla_testcase")
inputs_only = os.environ.get("MQSMLA_INPUTS_ONLY", "1") == "1"
device_id = int(os.environ.get("MQSMLA_DEVICE_ID", "0"))
result_path = Path(os.environ.get("MQSMLA_RESULT_SAVE_PATH", "result.xlsx"))

param_combinations = []
for params in ENABLED_PARAMS:
    if os.environ.get("FD_PERF_MODE") == "1":
        param_combinations.extend(expand_params([params]))
        continue
    param_values = {
        "Testcase_Name": params.get("Testcase_Name", [None]),
        "layout_q": params.get("layout_q"),
        "layout_kv": params.get("layout_kv"),
        "q_type": params.get("q_type"),
        "ori_kv_type": params.get("ori_kv_type"),
        "cmp_kv_type": params.get("cmp_kv_type", [None]),
        "B": params.get("B"),
        "S1": params.get("S1"),
        "S2": params.get("S2", [None]),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K1": params.get("K1", [None]),
        "K": params.get("K", [None]),
        "block_num1": params.get("block_num1", [None]),
        "block_num2": params.get("block_num2", [None]),
        "block_size1": params.get("block_size1"),
        "block_size2": params.get("block_size2", [None]),
        "seqused_q": params.get("seqused_q", [None]),
        "cu_seqlens_q": params.get("cu_seqlens_q", [None]),
        "seqused_ori_kv": params.get("seqused_ori_kv", [None]),
        "seqused_cmp_kv": params.get("seqused_cmp_kv", [None]),
        "cu_seqlens_ori_kv": params.get("cu_seqlens_ori_kv", [None]),
        "cu_seqlens_cmp_kv": params.get("cu_seqlens_cmp_kv", [None]),
        "cmp_residual_kv": params.get("cmp_residual_kv", [None]),
        "softmax_scale": params.get("softmax_scale"),
        "cmp_ratio": params.get("cmp_ratio", [None]),
        "ori_mask_mode": params.get("ori_mask_mode"),
        "cmp_mask_mode": params.get("cmp_mask_mode", [None]),
        "ori_win_left": params.get("ori_win_left"),
        "ori_win_right": params.get("ori_win_right"),
        "quant_mode": params.get("quant_mode"),
        "tile_size": params.get("tile_size"),
        "rope_head_dim": params.get("rope_head_dim"),
        "template_run_mode": params.get("template_run_mode"),
        "actlen_mode": params.get("actlen_mode"),
        "S1EQS2": params.get("S1EQS2", [False]),
        "return_softmax_lse": params.get("return_softmax_lse", [False]),
        "ori_kv_topk_mode": params.get("ori_kv_topk_mode", [None]),
        "cmp_kv_topk_mode": params.get("cmp_kv_topk_mode", [None]),
        "ori_sparse_indices_mode": params.get("ori_sparse_indices_mode", ["full"]),
        "cmp_sparse_indices_mode": params.get("cmp_sparse_indices_mode", ["full"]),
        "ori_topk_length": params.get("ori_topk_length", [None]),
        "cmp_topk_length": params.get("cmp_topk_length", [None]),
        "batch_consistency": params.get("batch_consistency", [None]),
        "batch_consistency_seed": params.get("batch_consistency_seed", [None]),
        "batch_consistency_order": params.get("batch_consistency_order", [None]),
        "batch_consistency_batch_split": params.get(
            "batch_consistency_batch_split", [None]
        ),
        "batch_consistency_mode_batch": params.get(
            "batch_consistency_mode_batch", [None]
        ),
        "batch_consistency_token_split": params.get(
            "batch_consistency_token_split", [None]
        ),
        "batch_consistency_shape_change": params.get(
            "batch_consistency_shape_change", [None]
        ),
    }

    param_names = list(param_values.keys())
    values_lists = [param_values[name] for name in param_names]

    for combo in itertools.product(*values_lists):
        combination = dict(zip(param_names, combo))
        param_combinations.append(combination)
case_id = 0


def mqsmla(param_combinations):
    global case_id
    if os.environ.get("FD_PERF_MODE") == "1":
        from batch import mixed_quant_sparse_flash_mla_process
        run_single("mqsmla", param_combinations, mixed_quant_sparse_flash_mla_golden.gen_data,
                   lambda data: mixed_quant_sparse_flash_mla_process.test_mqsmla_quant_process_ci(
                       data, device_id=int(os.environ.get("FD_DEVICE_ID", "0"))))
        return

    # 填充None参数的默认值
    params = utils.fill_none_params(param_combinations)

    # 生成测试用例名称
    Testcase_Name = params["Testcase_Name"]
    if Testcase_Name is None:
        ops_mode = "prefill" if params["S1"] > 4 else "decode"
        q_type_str = "BF16" if params["q_type"] == torch.bfloat16 else "FP16"
        kv_type_str = (
            "FP8_AS_UINT8" if params["ori_kv_type"] == torch.uint8 else "FP8_E4M3FN"
        )
        prefix_part = (
            f"{param_combinations['tc_prefix']}_"
            if param_combinations.get("tc_prefix", "")
            else ""
        )
        Testcase_Name = f"mixedQuantSparseFlashMla_{params['template_run_mode']}_{ops_mode}_{params['layout_q']}_{q_type_str}_{params['layout_kv']}_{kv_type_str}_{params['B']}_{params['N1']}_{params['N2']}_{params['S1']}_{params['S2']}_{params['D']}_{params['K']}_{params['rope_head_dim']}_{case_id:06d}"
        params["Testcase_Name"] = Testcase_Name

    # 输入参数的合法性校验
    try:
        check_valid_param.check_valid_param(params)
    except ValueError as e:
        pytest.skip(f"输入参数校验失败:{e}")

    # Only operator inputs are persisted; CPU golden data is not generated.
    test_data = mixed_quant_sparse_flash_mla_golden.gen_data(
        params, generate_golden=False
    )
    for key in ("golden_state", "cpu_output", "cpu_lse"):
        test_data.pop(key, None)
    mixed_quant_sparse_flash_mla_golden.save_test_case(test_data, pt_save_path)
    case_id += 1

    if inputs_only:
        return

    # Import NPU dependencies only when execution is requested (e.g. profiling).
    from batch import mixed_quant_sparse_flash_mla_process

    utils.run_without_accuracy(
        test_data,
        mixed_quant_sparse_flash_mla_process.test_mqsmla_quant_process_ci,
        device_id,
        result_path,
    )


def _gen_testcase_id(params, idx):
    name = params.get("Testcase_Name")
    if name is not None:
        return name
    ops_mode = "prefill" if params["S1"] > 4 else "decode"
    q_type_str = "BF16" if params["q_type"] == torch.bfloat16 else "FP16"
    kv_type_str = (
        "FP8_AS_UINT8" if params["ori_kv_type"] == torch.uint8 else "FP8_E4M3FN"
    )
    return f"{params['template_run_mode']}_{ops_mode}_{params['layout_q']}_{q_type_str}_{params['layout_kv']}_{kv_type_str}_B{params['B']}_S1{params['S1']}_S2{params['S2']}_D{params['D']}_K{params['K']}_{idx:06d}"


testcase_ids = [_gen_testcase_id(p, i) for i, p in enumerate(param_combinations)]


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations, ids=testcase_ids)
def test_mixed_quant_sparse_flash_mla(param_combinations):
    # 线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(mqsmla, param_combinations)
        # 等待并获取结果
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail(f"当前用例线程执行失败: {e}")
