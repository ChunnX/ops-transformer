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

import pytest

from common import fia_fullquant_mxfp8_golden as golden
from common import golden_cache
from common import result_compare_method
import fia_fullquant_mxfp8_paramset_perf_rdv as paramset

CASES = paramset.CASES
SKIP_CASES = getattr(paramset, "SKIP_CASES", set())
CASE_IDS = [case["name"] for case in CASES]


def _build_parametrize_args():
    args = []
    for case in CASES:
        if case["name"] in SKIP_CASES:
            args.append(pytest.param(case, marks=pytest.mark.skip(reason="execution time too long")))
        else:
            args.append(case)
    return args


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
    golden.DATA_RANGE_Q = params["data_range_q"]
    golden.DATA_RANGE_K = params["data_range_k"]
    golden.DATA_RANGE_V = params["data_range_v"]
    golden.ENABLE_LSE = params["enable_lse"]
    golden.ENABLE_ROPE = params["enable_rope"]
    golden.D_rope = params["D_rope"]
    golden.GRAPH_PATH = params["graph_path"]
    golden.DEVICE_ID = params["device_id"]
    golden.IS_CONTIGUOUS = params["is_contiguous"]


def _execute_test(params, mode, cdir=None):
    _apply_params(params)
    case_name = params["name"]

    if "gen" in mode:
        data = golden.generate_data()
        q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table_torch = data
        golden_cache.save_input(case_name, golden_cache.build_input_dict(
            q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table_torch
        ), cache_dir=cdir)
    else:
        q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table_torch = \
            golden_cache.load_input(case_name, cache_dir=cdir)

    if "gen" in mode and "cpu" not in mode and "npu" not in mode and "compare" not in mode:
        return None, None

    if "cpu" in mode:
        cpu_out, cpu_lse = golden.cpu_mxfp8_golden(
            q_fp8, k_fp8, v_fp8,
            deq_q, deq_k, deq_v, p_scale,
            golden.ACTUAL_SEQ_Q, golden.ACTUAL_SEQ_KV,
            qr_bf16, kr_bf16,
        )
        golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)
    else:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cdir)

    if "cpu" in mode and "npu" not in mode and "compare" not in mode:
        return None, None

    if "npu" in mode:
        npu_out, lse_out = golden.npu_mxfp8_fa(
            q_fp8, k_fp8, v_fp8,
            deq_q, deq_k, deq_v, p_scale,
            golden.ACTUAL_SEQ_Q, golden.ACTUAL_SEQ_KV,
            block_table_torch, qr_bf16, kr_bf16,
        )
        golden_cache.save_npu_output(case_name, npu_out, lse_out, cache_dir=cdir)
    else:
        npu_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        return None, None

    compare_layout = "TND" if golden.ENABLE_PA else golden.INPUT_LAYOUT
    cpu_cmp = golden.convert_q_bnsd_to_layout(cpu_out, golden.ACTUAL_SEQ_Q, compare_layout)

    atten_result = result_compare_method.check_result(cpu_cmp, npu_out)

    lse_result = None
    if golden.ENABLE_LSE:
        lse_cmp = golden.convert_q_bnsd_to_layout(cpu_lse, golden.ACTUAL_SEQ_Q, compare_layout)
        lse_result = result_compare_method.check_result(lse_cmp, lse_out)

    return atten_result, lse_result


@pytest.mark.perf_rdv
@pytest.mark.parametrize("params", _build_parametrize_args(), ids=CASE_IDS)
def test_fia_fullquant_mxfp8(params, golden_mode, cache_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute_test, params, golden_mode, cache_dir)
        atten_result, lse_result = future.result()

    if atten_result is None:
        return

    atten_status = atten_result[0] if isinstance(atten_result, tuple) else atten_result
    if atten_status != "Pass":
        pct = atten_result[1] if isinstance(atten_result, tuple) and len(atten_result) > 1 else "N/A"
        pytest.fail(f"Attention output compare failed: result={atten_status}, PctRlt={pct}")

    if lse_result is not None:
        lse_status = lse_result[0] if isinstance(lse_result, tuple) else lse_result
        if lse_status != "Pass":
            pct = lse_result[1] if isinstance(lse_result, tuple) and len(lse_result) > 1 else "N/A"
            pytest.fail(f"LSE compare failed: result={lse_status}, PctRlt={pct}")
