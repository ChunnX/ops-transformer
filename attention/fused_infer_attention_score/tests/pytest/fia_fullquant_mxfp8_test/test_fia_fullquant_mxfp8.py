#!/usr/bin/python3
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

import pytest
import torch
import torch_npu

import fia_fullquant_mxfp8_golden as golden
import fia_fullquant_mxfp8_paramset as paramset
from precision_compare_v2 import check_result, compute_cv_report, display_cv_report

CASE_IDS = [case["name"] for case in paramset.CASES]

param_combinations = []
for params in paramset.ENABLED_PARAMS:
    expanded = dict(params)
    for key, default_vals in paramset.TEST_PARAMS_DEFAULTS.items():
        if key not in expanded:
            expanded[key] = default_vals
    param_values = [expanded[name] for name in paramset.PARAM_NAMES]
    for combo in itertools.product(*param_values):
        param_combinations.append(dict(zip(paramset.PARAM_NAMES, combo)))


def _apply_params(params):
    golden.B = params["B"]
    golden.N_q = params["N_q"]
    golden.N_kv = params["N_kv"]
    golden.D = params["D"]
    golden.ACTUAL_SEQ_Q = params["actual_seq_q"]
    golden.ACTUAL_SEQ_KV = params["actual_seq_kv"]
    golden.ENABLE_PA = params["enable_pa"]
    golden.KV_CACHE_LAYOUT = params["kv_cache_layout"]
    golden.BLOCK_SIZE = params["block_size"]
    golden.SPARSE_MODE = params["sparse_mode"]
    golden.Q_SCALE_LAYOUT = params["q_scale_layout"]
    golden.P_SCALE = params["p_scale"]
    if "data_range_q" in params:
        golden.DATA_RANGE_Q = params["data_range_q"]
    if "data_range_k" in params:
        golden.DATA_RANGE_K = params["data_range_k"]
    if "data_range_v" in params:
        golden.DATA_RANGE_V = params["data_range_v"]
    golden.USE_FP64_GOLDEN = params.get("use_fp64_golden", golden.USE_FP64_GOLDEN)
    golden.USE_FP64_COMPARE = params.get("use_fp64_compare", golden.USE_FP64_COMPARE)


def _execute_test(params):
    _apply_params(params)

    data = golden.generate_data()
    q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table_torch = data

    compare_layout = "TND" if golden.ENABLE_PA else golden.INPUT_LAYOUT

    if golden.USE_FP64_GOLDEN:
        cpu_out_fp64, cpu_lse_fp64 = golden.cpu_mxfp8_golden_fp64(
            q_fp8, k_fp8, v_fp8,
            deq_q, deq_k, deq_v, p_scale,
            golden.ACTUAL_SEQ_Q, golden.ACTUAL_SEQ_KV,
            qr_bf16, kr_bf16,
        )

        cpu_out_fp32, cpu_lse_fp32 = golden.cpu_mxfp8_golden(
            q_fp8, k_fp8, v_fp8,
            deq_q, deq_k, deq_v, p_scale,
            golden.ACTUAL_SEQ_Q, golden.ACTUAL_SEQ_KV,
            qr_bf16, kr_bf16,
        )

        npu_out, lse_out = golden.npu_mxfp8_fa(
            q_fp8, k_fp8, v_fp8,
            deq_q, deq_k, deq_v, p_scale,
            golden.ACTUAL_SEQ_Q, golden.ACTUAL_SEQ_KV,
            block_table_torch, qr_bf16, kr_bf16,
        )

        golden_tnd = golden.convert_q_bnsd_to_layout(
            cpu_out_fp64 if golden.USE_FP64_COMPARE else cpu_out_fp32,
            golden.ACTUAL_SEQ_Q, compare_layout)
        benchmark_tnd = golden.convert_q_bnsd_to_layout(cpu_out_fp32, golden.ACTUAL_SEQ_Q, compare_layout)

        cv_report = compute_cv_report(
            golden=golden_tnd,
            benchmark=benchmark_tnd,
            actual=npu_out,
            test_name=params.get("name", "fia_mxfp8"),
        )

        atten_passed = cv_report.passed

        lse_passed = True
        if golden.ENABLE_LSE:
            golden_lse = cpu_lse_fp64 if golden.USE_FP64_COMPARE else cpu_lse_fp32
            cv_lse_report = compute_cv_report(
                golden=golden_lse,
                benchmark=cpu_lse_fp32,
                actual=lse_out,
                test_name=params.get("name", "fia_mxfp8") + "_lse",
            )
            lse_passed = cv_lse_report.passed
    else:
        cpu_out, cpu_lse = golden.cpu_mxfp8_golden(
            q_fp8, k_fp8, v_fp8,
            deq_q, deq_k, deq_v, p_scale,
            golden.ACTUAL_SEQ_Q, golden.ACTUAL_SEQ_KV,
            qr_bf16, kr_bf16,
        )

        npu_out, lse_out = golden.npu_mxfp8_fa(
            q_fp8, k_fp8, v_fp8,
            deq_q, deq_k, deq_v, p_scale,
            golden.ACTUAL_SEQ_Q, golden.ACTUAL_SEQ_KV,
            block_table_torch, qr_bf16, kr_bf16,
        )

        cpu_cmp = golden.convert_q_bnsd_to_layout(cpu_out, golden.ACTUAL_SEQ_Q, compare_layout)
        atten_result = check_result(cpu_cmp, npu_out)
        atten_passed = atten_result[0] == "Pass" if isinstance(atten_result, tuple) else atten_result == "Pass"

        lse_passed = True
        if golden.ENABLE_LSE:
            lse_result = check_result(cpu_lse, lse_out)
            lse_passed = lse_result[0] == "Pass" if isinstance(lse_result, tuple) else lse_result == "Pass"

    return atten_passed, lse_passed


@pytest.mark.ci
@pytest.mark.parametrize("params", paramset.CASES, ids=CASE_IDS)
def test_fia_fullquant_mxfp8(params):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute_test, params)
        atten_passed, lse_passed = future.result()

    if not atten_passed:
        pytest.fail(f"Attention output compare failed")
    if not lse_passed:
        pytest.fail(f"LSE compare failed")
