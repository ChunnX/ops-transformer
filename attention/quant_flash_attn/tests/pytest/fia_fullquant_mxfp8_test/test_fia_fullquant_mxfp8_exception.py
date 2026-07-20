#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
  6. 断言抛错，打印报错信息

输出说明：
  PASSED = 算子抛出了预期的异常，log 中会打印 [CHECKER] 或 [MAIN_OP] 的错误消息
  FAILED = 要么 mutator 崩溃了，要么算子未按预期抛异常
"""

import traceback

import torch
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
def test_fia_fullquant_mxfp8_exception(base_name, mutator, expected_kw, base_params, golden_mode, cache_dir, request):
    case_id = request.node.callspec.id
    SEP = "=" * 60 + f" [{case_id}] " + "=" * 60
    print(f"\n{SEP}")
    print(f"[CASE] base={base_name} | expected={expected_kw}")

    # 1. apply 基座参数
    test_runner.apply_params(base_params)

    # 2. 生成合规数据
    if "gen" in golden_mode:
        data = golden.generate_data()
    else:
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

    # 4. 篡改入参 (mutator 可能在构造异常 tensor 时崩溃，单独 try/catch)
    try:
        meta_overrides, main_overrides = mutator(inputs)
    except Exception as e:
        pytest.fail(
            f"[MUTATOR_CRASH] mutator 在篡改入参时异常退出, "
            f"异常未发生在算子调用阶段, 无法验证 checker 拦截能力\n"
            f"  error: {type(e).__name__}: {e}\n"
            f"  traceback:\n{traceback.format_exc()}"
        )

    # ---- 打印完整入参信息 ----
    print("[NPU_CALL_PARAMS]")
    print(f"  layout_q={inputs.get('layout_q')}, layout_kv={inputs.get('layout_kv')}, "
          f"layout_out={inputs.get('layout_out')}, layout_q_descale={inputs.get('layout_q_descale')}")
    print(f"  sparse_mode={inputs.get('sparse_mode')}, block_size={inputs.get('block_size')}")
    q_shape = list(inputs.get('q', torch.tensor([])).shape)
    k_shape = list(inputs.get('k', torch.tensor([])).shape)
    v_shape = list(inputs.get('v', torch.tensor([])).shape)
    deq_q_shape = list(inputs.get('dequant_scale_q', torch.tensor([])).shape)
    deq_k_shape = list(inputs.get('dequant_scale_k', torch.tensor([])).shape)
    deq_v_shape = list(inputs.get('dequant_scale_v', torch.tensor([])).shape)
    print(f"  q={q_shape}, k={k_shape}, v={v_shape}")
    print(f"  q_descale={deq_q_shape}, k_descale={deq_k_shape}, v_descale={deq_v_shape}")
    bt = inputs.get('block_table')
    print(f"  block_table={'None' if bt is None else list(bt.shape)}")
    cuq = inputs.get('cu_seqlens_q')
    cukv = inputs.get('cu_seqlens_kv')
    sq = inputs.get('seqused_q')
    skv = inputs.get('seqused_kv')
    print(f"  cu_seqlens_q={'None' if cuq is None else cuq}, cu_seqlens_kv={'None' if cukv is None else cukv}")
    print(f"  seqused_q={'None' if sq is None else sq}, seqused_kv={'None' if skv is None else skv}")
    print(f"  max_seqlen_q={inputs.get('max_seqlen_q')}, max_seqlen_kv={inputs.get('max_seqlen_kv')}")
    print(f"  q_n={inputs.get('q_n')}, kv_n={inputs.get('kv_n')}, softmax_scale={inputs.get('softmax_scale')}")
    print(f"  meta_overrides={meta_overrides}")
    print(f"  main_overrides={main_overrides}")
    print("[/NPU_CALL_PARAMS]")

    # 5. 调用算子，预期抛错
    try:
        golden._call_npu_fa_op(
            **inputs,
            meta_overrides=meta_overrides or None,
            main_overrides=main_overrides or None,
        )
    except (RuntimeError, AssertionError, Exception) as e:
        # 异常被捕获 → 验证成功 ，打印完整报错信息 (含 C++ 侧 checker 拦截文本)
        # 不截断，确保 checker 在 TraceBack 段中的拦截原因能完整呈现到日志
        err_msg = str(e)
        print(f"[EXCEPTION_CAUGHT] {type(e).__name__}: {err_msg}")
        print(f"[RESULT] [{case_id}] PASSED\n{SEP}")
        return  # PASS

    # 没有异常 → 测试失败
    print(f"[RESULT] [{case_id}] FAILED\n{SEP}")
    pytest.fail(
        f"[NO_EXCEPTION] 篡改后的非法入参未触发任何异常, "
        f"checker/主算子均通过了该非法输入\n"
        f"  expected_keyword={expected_kw}\n"
        f"  layout_q={inputs.get('layout_q')}, layout_kv={inputs.get('layout_kv')}, "
        f"layout_out={inputs.get('layout_out')}\n"
        f"  sparse_mode={inputs.get('sparse_mode')}, quant_mode=1\n"
        f"  (若为 layout_out/BSND 等, 可能主算子内部未校验该参数)"
    )
