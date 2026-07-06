#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch_npu

# 通过已安装的 cann_ops_transformer 注册 extension
from cann_ops_transformer.ops import sparse_flash_mla_metadata as _metadata_registration  # noqa: F401

import result_compare_method
import sparse_flash_mla_golden
from sparse_flash_mla_paramset import TEST_PARAMS
import utils


def run_golden(params, run_mode):
    sparse_flash_mla_golden.RUN_MODE = run_mode
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
    print("RUN_MODE=1 = 单循环FA (sinks 只计入1次, 正确)")
    print("RUN_MODE=2 = FD分段 + sinks只计入首段 (sinks 只计入1次, 正确)")

    out1, lse1 = run_golden(params, 1)
    out2, lse2 = run_golden(params, 2)

    compare_pair(out1, out2, "attn_out: expect=RUN_MODE=1(单循环), real=RUN_MODE=2(FD首段)")

    if lse1 is not None and lse2 is not None:
        compare_pair(lse1, lse2, "softmax_lse: expect=RUN_MODE=1, real=RUN_MODE=2")


if __name__ == "__main__":
    main()
