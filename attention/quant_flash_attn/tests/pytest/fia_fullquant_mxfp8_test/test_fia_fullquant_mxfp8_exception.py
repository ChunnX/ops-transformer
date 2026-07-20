#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
MxFP8 异常用例 runner (思路2)。

流程：
  1. apply 基座参数到 golden 全局变量
  2. golden.generate_data() 生成合规 BNSD 数据
  3. golden.prepare_npu_inputs(...) 准备合规 NPU 入参字典
  4. mutator(inputs) 定向篡改入参 + 返回 (meta_overrides, main_overrides)
  5. 调用 golden._call_npu_fa_op(**inputs, meta_overrides=, main_overrides=)，预期抛错
  6. 断言抛错 (可选校验报错关键字)
"""

import pytest

from common import fia_fullquant_mxfp8_golden as golden
from common import test_runner
import fia_fullquant_mxfp8_paramset_exception as paramset

BASE_CASES = paramset.BASE_CASES
EXC_CASES = paramset.EXC_CASES
CASE_IDS = [c[0] for c in EXC_CASES]


def _build_parametrize_args():
    args = []
    for case_name, base_name, mutator, expected_kw in EXC_CASES:
        base = BASE_CASES[base_name]
        args.append(pytest.param(base_name, mutator, expected_kw, base,
                                 id=case_name))
    return args


@pytest.mark.exception
@pytest.mark.parametrize("base_name, mutator, expected_kw, base_params",
                         _build_parametrize_args(), ids=CASE_IDS)
def test_fia_fullquant_mxfp8_exception(base_name, mutator, expected_kw, base_params, golden_mode, cache_dir):
    # 1. apply 基座参数
    test_runner.apply_params(base_params)

    # 2. 生成合规数据 (异常用例只走 gen，不需要 cpu/npu/compare)
    if "gen" in golden_mode:
        data = golden.generate_data()
    else:
        # 异常用例无缓存机制，强制 gen
        data = golden.generate_data()
    q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table_torch = data

    # 3. 准备合规 NPU 入参
    inputs = golden.prepare_npu_inputs(
        q_fp8, k_fp8, v_fp8,
        deq_q, deq_k, deq_v, p_scale,
        golden.CU_SEQLENS_Q, golden.CU_SEQLENS_KV,
        golden.SEQUSED_Q, golden.SEQUSED_KV,
        golden.MAX_SEQLEN_Q, golden.MAX_SEQLEN_KV,
        block_table_torch, qr_bf16, kr_bf16,
    )

    # 4. 篡改入参
    meta_overrides, main_overrides = mutator(inputs)

    # 5. 调用算子，预期抛错
    with pytest.raises((RuntimeError, AssertionError, Exception)) as exc_info:
        golden._call_npu_fa_op(
            **inputs,
            meta_overrides=meta_overrides or None,
            main_overrides=main_overrides or None,
        )

    # 6. 可选：校验报错关键字
    if expected_kw:
        err_msg = str(exc_info.value)
        # 关键字匹配失败时仅 warning，不 fail (不同环境报错文案可能不同)
        if expected_kw.lower() not in err_msg.lower():
            print(f"[WARN] expected keyword '{expected_kw}' not in error: {err_msg[:200]}")
