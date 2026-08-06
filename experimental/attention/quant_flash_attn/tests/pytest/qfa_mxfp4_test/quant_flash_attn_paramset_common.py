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
"""参数展开公共逻辑 + 默认值

case 字段说明（与 run_fia_eager 签名对齐）：
  b, n2, g, s1, s2, qk_d, v_d               - shape 参数
  input_layout, kv_storage_mode             - layout
  q_dtype, kv_dtype, out_dtype              - dtype
  q_quant_mode                              - 量化模式（mxfp4=5）
  act_seq_lens_q, act_seq_lens_kv           - 实际序列长度

n1 = n2 * g（num_heads_q = num_heads_kv * gqa_group）
run_fia_eager 接收 n2 与 g，内部计算 num_heads = n2 * g。
"""

import itertools

# run_fia_eager 实际接收的参数白名单
PARAM_NAMES = [
    "b", "n2", "g", "s1", "s2", "qk_d", "v_d",
    "input_layout", "kv_storage_mode",
    "q_dtype", "kv_dtype", "out_dtype",
    "q_quant_mode",
    "act_seq_lens_q", "act_seq_lens_kv",
]

# 默认值：未显式指定时自动补齐
TEST_PARAMS_DEFAULTS = {
    "g": [1],
    "v_d": [None],  # None 表示与 qk_d 相同
    "kv_dtype": [None],  # None 表示与 q_dtype 相同
    "out_dtype": [None],  # None 表示与 q_dtype 相同
    "kv_storage_mode": ["continue"],
    "q_quant_mode": [5],
}


def expand_paramset_to_cases(test_params):
    """将 paramset 展开为 case 列表，每个 case 为单值 dict。

    对每个配置中的所有维度做笛卡尔积展开，并补齐默认值。
    """
    cases = []
    for name, params in test_params.items():
        expanded = dict(params)
        for key, default_vals in TEST_PARAMS_DEFAULTS.items():
            if key not in expanded:
                expanded[key] = default_vals
        # 仅对 PARAM_NAMES 中的字段做笛卡尔积
        param_values = [expanded[n] if n in expanded else [None] for n in PARAM_NAMES]
        for combo in itertools.product(*param_values):
            case = {"name": name}
            for key, val in zip(PARAM_NAMES, combo):
                case[key] = val
            cases.append(case)
    return cases
