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

# s2候选值: 覆盖 <128(奇/偶) / >128(奇/偶) / <16384
S2_CANDIDATES = [100, 127, 5000, 8001]

# layout组合: (layout_q, layout_kv)
LAYOUT_COMBOS = [
    ("BSND", "PA_BBND"),
    ("TND",  "PA_BBND"),
    ("BSND", "BSND"),
    ("TND",  "TND"),
]

BSIZE = 2
N1 = 64
N2 = 1
D = 512
BLOCK_SIZE = 128
SOFTMAX_SCALE = 0.04419417
CMP_RATIO = 4
ORI_MASK_MODE = 0
CMP_MASK_MODE = 0


def build_case(template_mode, s2, layout_q, layout_kv):
    """构造一组参数。cu_seqlens/seqused均由utils自动生成(random_seq=True, 每batch不等长)。

    K1/K在gen_data中根据maxOriS2/maxCmpS2计算, 但paramset需先给定。
    这里取maxOriS2//2 / maxCmpS2//2作为固定值, 保证 >=1 且 < maxS2。
    """
    name = f"{template_mode.lower()}_s2{s2}_{layout_q.lower()}_{layout_kv.lower()}"
    is_ori_cmp = (template_mode == "ORI_CMP_SPARSE")
    K1_val = max(1, s2 // 2)
    K_val = max(1, (s2 // CMP_RATIO) // 2) if is_ori_cmp else None

    case = {
        "testcase_name": [name],
        "layout_q": [layout_q],
        "layout_kv": [layout_kv],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.bfloat16],
        "B": [BSIZE],
        "S1": [1],
        "S2": [s2],
        "N1": [N1],
        "N2": [N2],
        "D": [D],
        "K1": [K1_val],
        "block_size1": [BLOCK_SIZE],
        "block_num1": [None],
        "softmax_scale": [SOFTMAX_SCALE],
        "return_softmax_lse": [False],
        "ori_mask_mode": [ORI_MASK_MODE],
        "template_mode": [template_mode],
        "cmp_ratio": [None],
        "cmp_mask_mode": [None],
        "ori_win_left": [None],
        "ori_win_right": [None],
        "ori_kv_topk_mode": ["random"],
        "cmp_kv_topk_mode": ["random"],
        "ori_sparse_indices_mode": ["full"],
        "cmp_sparse_indices_mode": ["full"],
        "cu_seqlens_q": [None],
        "cu_seqlens_ori_kv": [None],
        "cu_seqlens_cmp_kv": [None],
        "seqused_q": [None],
        "seqused_ori_kv": [None],
        "seqused_cmp_kv": [None],
        "cmp_residual_kv": [None],
        "random_seq": [True],
    }

    # TND_q需要T1; TND_kv需要T2/T3
    if layout_q == "TND":
        case["T1"] = [BSIZE * 1]
    else:
        case["T1"] = [None]
    if layout_kv == "TND":
        case["T2"] = [BSIZE * s2]
        if is_ori_cmp:
            case["T3"] = [BSIZE * (s2 // CMP_RATIO)]
        else:
            case["T3"] = [None]
    else:
        case["T2"] = [None]
        case["T3"] = [None]

    if is_ori_cmp:
        case["cmp_kv_type"] = [torch.bfloat16]
        case["cmp_ratio"] = [CMP_RATIO]
        case["cmp_mask_mode"] = [CMP_MASK_MODE]
        case["K"] = [K_val]
        case["block_size2"] = [BLOCK_SIZE]
        case["block_num2"] = [None]
    else:
        case["cmp_kv_type"] = [None]
        case["K"] = [None]

    return case


TEST_PARAMS = {}

# ORI_SPARSE: ori/cmp mask_mode 仅 0, 无 cmp
for s2 in S2_CANDIDATES:
    for lq, lkv in LAYOUT_COMBOS:
        key = f"ori_sparse_s2{s2}_{lq}_{lkv}"
        TEST_PARAMS[key] = build_case("ORI_SPARSE", s2, lq, lkv)

# ORI_CMP_SPARSE: ori/cmp mask_mode 仅 0
for s2 in S2_CANDIDATES:
    for lq, lkv in LAYOUT_COMBOS:
        key = f"ori_cmp_sparse_s2{s2}_{lq}_{lkv}"
        TEST_PARAMS[key] = build_case("ORI_CMP_SPARSE", s2, lq, lkv)

ENABLED_PARAMS = [{**TEST_PARAMS[key], "testcase_name": [key]} for key in TEST_PARAMS.keys()]
