#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import result_compare_method
import sparse_flash_mla_golden
from sparse_flash_mla_paramset import TEST_PARAMS
import utils


def run_golden(params, sinks_mode):
    sparse_flash_mla_golden.SINKS_HANDLE_MODE = sinks_mode
    np.random.seed(42)
    torch.manual_seed(42)
    input_data = sparse_flash_mla_golden.gen_data(params)
    return input_data['cpu_output'], input_data.get('softmax_lse')


def compare_pair(expect, real, label):
    print(f"\n==== {label} ====")
    result, fulfill_percent = result_compare_method.check_result(expect, real)
    print(f"{label}: result={result}, fulfill_percent={fulfill_percent}")
    return result, fulfill_percent


def main():
    case_name = sys.argv[1] if len(sys.argv) > 1 else "csa_decode"
    if case_name not in TEST_PARAMS:
        raise ValueError(f"case '{case_name}' not in TEST_PARAMS, available: {list(TEST_PARAMS.keys())}")

    params = utils.generate_param_combinations([TEST_PARAMS[case_name]])[0]
    params = utils.generate_case_with_default_param(params)

    print(f"==== compare sinks handle modes on case: {case_name} ====")
    print("mode 0 = FD 分段 (sinks 重复计入 N_splits 次, bug)")
    print("mode 1 = 单循环 (sinks 只计入1次, 正确)")
    print("mode 2 = FD 分段 + sinks 只计入首段 (sinks 只计入1次, 正确)")
    print("mode 3 = FD 分段 + FA阶段不含sinks, 规约时作为虚拟split纳入 (sinks 只计入1次, 正确)")
    print("mode 4 = FD 分段(每段含sinks) + 规约时再纳入一次 (sinks 计入 N_splits+1 次, bug)")

    out0, lse0 = run_golden(params, 0)
    out1, lse1 = run_golden(params, 1)
    out2, lse2 = run_golden(params, 2)
    out3, lse3 = run_golden(params, 3)
    out4, lse4 = run_golden(params, 4)

    compare_pair(out1, out2, "attn_out: expect=mode1(单循环), real=mode2(FD首段)")
    compare_pair(out1, out3, "attn_out: expect=mode1(单循环), real=mode3(FD规约纳入)")
    compare_pair(out1, out0, "attn_out: expect=mode1(单循环), real=mode0(FD bug)")
    compare_pair(out1, out4, "attn_out: expect=mode1(单循环), real=mode4(FD+规约 bug)")
    compare_pair(out4, out0, "attn_out: expect=mode4(FD+规约 bug), real=mode0(FD bug)")

    if lse1 is not None and lse2 is not None and lse0 is not None and lse3 is not None and lse4 is not None:
        compare_pair(lse1, lse2, "softmax_lse: expect=mode1, real=mode2")
        compare_pair(lse1, lse3, "softmax_lse: expect=mode1, real=mode3")
        compare_pair(lse1, lse0, "softmax_lse: expect=mode1, real=mode0")
        compare_pair(lse1, lse4, "softmax_lse: expect=mode1, real=mode4")
        compare_pair(lse4, lse0, "softmax_lse: expect=mode4, real=mode0")


if __name__ == "__main__":
    main()
