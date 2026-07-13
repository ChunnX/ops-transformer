#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import torch
import utils

TEST_PARAMS = {
    "decode_first":{
        "Testcase_Name": [None],
        "layout_q": ["TND"],
        "layout_kv": ["PA_BBND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.float8_e4m3fn],
        "cmp_kv_type": [torch.float8_e4m3fn],
        "B": [1],
        "S1": [1],
        "S2": [8193],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [512],
        "block_num1": [None],
        "block_num2": [None],
        "block_size1": [128],
        "block_size2": [128],
        "seqused_q": [None],
        "cu_seqlens_q": [None],
        "seqused_ori_kv": [None],
        "seqused_cmp_kv": [None],
        "cu_seqlens_ori_kv": [None],
        "cu_seqlens_cmp_kv": [None],
        "cmp_residual_kv": [None],
        "softmax_scale": [0.04419417],
        "cmp_ratio": [1],
        "ori_mask_mode": [4],
        "cmp_mask_mode": [3],
        "ori_win_left": [127],
        "ori_win_right": [0],
        "quant_mode": [1],
        "tile_size": [64],
        "rope_head_dim": [64],
        "template_run_mode": ["SWA"],
        "actlen_mode":["full"],
        "S1EQS2":[False],
        "q_datarange": [[-10, 10]],
        "ori_kv_datarange": [[-10, 10]],
        "cmp_kv_datarange": [[-10, 10]],
    },

    "prefill_first":{
        "Testcase_Name": [None],
        "layout_q": ["TND"],
        "layout_kv": ["PA_BBND"],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.float8_e4m3fn],
        "cmp_kv_type": [torch.float8_e4m3fn],
        "B": [1],
        "S1": [8192],
        "S2": [8192],
        "N1": [64],
        "N2": [1],
        "D": [512],
        "K": [512],
        "block_num1": [None],
        "block_num2": [None],
        "block_size1": [128],
        "block_size2": [128],
        "seqused_q": [None],
        "cu_seqlens_q": [None],
        "seqused_ori_kv": [None],
        "seqused_cmp_kv": [None],
        "cu_seqlens_ori_kv": [None],
        "cu_seqlens_cmp_kv": [None],
        "cmp_residual_kv": [None],
        "softmax_scale": [0.04419417],
        "cmp_ratio": [1],
        "ori_mask_mode": [4],
        "cmp_mask_mode": [3],
        "ori_win_left": [127],
        "ori_win_right": [0],
        "quant_mode": [1],
        "tile_size": [64],
        "rope_head_dim": [64],
        "template_run_mode": ["SWA"],
        "actlen_mode":["full"],
        "S1EQS2":[False],
        "q_datarange": [[-10, 10]],
        "ori_kv_datarange": [[-10, 10]],
        "cmp_kv_datarange": [[-10, 10]],
    }
}

_LIST_PARAM_KEYS = [
    "seqused_q", "cu_seqlens_q", "seqused_ori_kv", "seqused_cmp_kv",
    "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "cmp_residual_kv",
]
_DATARANGE_PARAM_KEYS = [
    "q_datarange", "ori_kv_datarange", "cmp_kv_datarange",
]
_DEFAULT_VALUES = {
    "topk_value_mode": 1,
    "return_softmax_lse": False,
    "actlen_mode": "full",
    "S1EQS2": False,
    "cmp_ratio": 1,
    "cmp_mask_mode": 3,
    "block_num1": None,
    "block_num2": None,
    "block_size2": None,
    "K": None,
    "cmp_kv_type": None,
    "q_datarange": None,
    "ori_kv_datarange": None,
    "cmp_kv_datarange": None,
}

def _convert_excel_row_to_paramset_format(row):
    """将Excel行转换为paramset格式（每个value包裹在list中，供itertools.product使用）。"""
    normalized = {}
    for key, value in row.items():
        if value == 'torch.bfloat16':
            value = torch.bfloat16
        elif value == 'torch.float8_e4m3fn':
            value = torch.float8_e4m3fn
        elif value == "FALSE":
            value = False
        elif value == "TRUE":
            value = True
        normalized[key] = value

    for key in _LIST_PARAM_KEYS:
        normalized[key] = utils.parse_list_param(normalized.get(key))
    for key in _DATARANGE_PARAM_KEYS:
        normalized[key] = utils.parse_datarange_param(normalized.get(key))

    for key, default_val in _DEFAULT_VALUES.items():
        if key not in normalized or normalized[key] is None:
            normalized[key] = default_val

    if isinstance(normalized.get("template_run_mode"), list):
        normalized["template_run_mode"] = normalized["template_run_mode"][0]

    return {key: [value] for key, value in normalized.items()}

def _load_params_from_excel():
    """若设置了QSMLA_EXCEL和QSMLA_SHEET环境变量，则从Excel加载测试参数。"""
    excel_file = os.environ.get("QSMLA_EXCEL")
    sheet_name = os.environ.get("QSMLA_SHEET")
    if not excel_file or not sheet_name:
        return None
    if not os.path.exists(excel_file):
        print(f"[paramset] QSMLA_EXCEL文件不存在: {excel_file}，使用硬编码TEST_PARAMS")
        return None

    test_cases = utils.load_excel_test_cases(excel_file, sheet_name)
    if not test_cases:
        print("[paramset] Excel未加载到测试用例，使用硬编码TEST_PARAMS")
        return None

    enabled = []
    for row in test_cases:
        paramset_format = _convert_excel_row_to_paramset_format(row)
        enabled.append(paramset_format)
    print(f"[paramset] 从Excel加载了 {len(enabled)} 条测试用例 (excel={excel_file}, sheet={sheet_name})")
    return enabled

_excel_params = _load_params_from_excel()
if _excel_params is not None:
    ENABLED_PARAMS = _excel_params
else:
    ENABLED_PARAMS = [TEST_PARAMS["decode_first"]]
    # ENABLED_PARAMS = [TEST_PARAMS[key] for key in TEST_PARAMS.keys()]