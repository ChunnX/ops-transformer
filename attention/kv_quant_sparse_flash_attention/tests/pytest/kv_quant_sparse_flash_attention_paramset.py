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

import torch


def _wrap_value(key, value):
    if key in ("actual_seq_q", "actual_seq_kv"):
        return [value]
    return [value]


def _make_case(testcase_prefix, testcase_number, **overrides):
    case = {
        "Testcase_Prefix": [testcase_prefix],
        "Testcase_Number": [testcase_number],
        "layout_query": ["BSND"],
        "layout_kv": ["PA_BSND"],
        "q_type": [torch.bfloat16],
        "kv_dtype": [None],
        "B": [1],
        "S1": [1],
        "S2": [128],
        "N1": [1],
        "N2": [1],
        "D": [512],
        "K": [512],
        "scale_value": [0.041666666666666664],
        "key_quant_mode": [2],
        "value_quant_mode": [2],
        "sparse_block_size": [1],
        "tile_size": [128],
        "rope_head_dim": [64],
        "sparse_mode": [3],
        "attention_mode": [2],
        "quant_scale_repo_mode": [1],
        "block_size": [128],
        "actual_seq_q": [[1]],
        "actual_seq_kv": [[128]],
    }

    for key, value in overrides.items():
        case[key] = _wrap_value(key, value)
    return case


# MD5 baseline 只放当前 CPU baseline 链路已经验证稳定的 BSND + PA_BSND case。
# 其余布局和边界值覆盖下沉到 snapshot/contract 测试，避免把当前链路本身不支持的场景误塞进 baseline。
TEST_PARAMS = {
    "dtype_bf16_f8": _make_case("baseline_dtype_bf16_f8", 1),
    "dtype_fp16_f8": _make_case("baseline_dtype_fp16_f8", 2, q_type=torch.float16),
    "dtype_bf16_hif8": _make_case("baseline_dtype_bf16_hif8", 3, kv_dtype="hifloat8"),
    "b_2": _make_case("baseline_b_2", 60, B=2, S1=2, actual_seq_q=[1, 2], actual_seq_kv=[128, 128]),
    "b_3": _make_case("baseline_b_3", 61, B=3, S1=3, actual_seq_q=[1, 2, 3], actual_seq_kv=[128, 128, 128]),
    "b_4": _make_case("baseline_b_4", 62, B=4, S1=4, actual_seq_q=[1, 2, 3, 4], actual_seq_kv=[128, 128, 128, 128]),
    "n1_2": _make_case("baseline_n1_2", 10, N1=2),
    "n1_4": _make_case("baseline_n1_4", 11, N1=4),
    "n1_8": _make_case("baseline_n1_8", 12, N1=8),
    "n1_16": _make_case("baseline_n1_16", 13, N1=16),
    "n1_32": _make_case("baseline_n1_32", 14, N1=32),
    "n1_48": _make_case("baseline_n1_48", 15, N1=48),
    "n1_64": _make_case("baseline_n1_64", 16, N1=64),
    "s1_4": _make_case("baseline_s1_4", 30, S1=4, actual_seq_q=[4]),
    "s1_5": _make_case("baseline_s1_5", 31, S1=5, actual_seq_q=[5]),
    "s1_128": _make_case("baseline_s1_128", 32, S1=128, actual_seq_q=[128]),
    "s2_1": _make_case("baseline_s2_1", 40, S2=1, block_size=1, actual_seq_kv=[1]),
    "k_2048": _make_case("baseline_k_2048", 50, K=2048),
}

ENABLED_PARAMS = list(TEST_PARAMS.values())
