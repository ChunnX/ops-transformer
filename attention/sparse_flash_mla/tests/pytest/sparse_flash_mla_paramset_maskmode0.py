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

# s2候选值: 覆盖 <128(奇/偶) / >128(奇/偶) / <16384(大值)
S2_CANDIDATES = [100, 127, 5000, 8001]

# 每种layout组合对应的参数模板
# (layout_q, layout_kv)
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


def make_seqused(s2, bsize=BSIZE, cmp_ratio=CMP_RATIO):
    """构造满足 seqused_ori_kv[b] = seqused_cmp_kv[b]*cmp_ratio + residual[b] 的三组长度。

    让 batch0 整除、batch1 带余数，覆盖两种情形。
    """
    kv0 = s2 - (s2 % cmp_ratio)
    kv1 = s2
    seqused_ori = [kv0, kv1]
    seqused_cmp = [v // cmp_ratio for v in seqused_ori]
    residual = [v - c * cmp_ratio for v, c in zip(seqused_ori, seqused_cmp)]
    return seqused_ori, seqused_cmp, residual


def make_cu_seqlens(seqused):
    cu = [0]
    for v in seqused:
        cu.append(cu[-1] + v)
    return cu


def make_q_lens(s1, bsize=BSIZE):
    """q 长度均设为 S1, 便于对齐。"""
    return [s1] * bsize


def build_case(template_mode, ori_mask_mode, cmp_mask_mode, s2, layout_q, layout_kv):
    """根据组合构造一组参数字典(单元素列表形式, 兼容 generate_param_combinations)。"""
    s1 = 1
    seqused_ori, seqused_cmp, residual = make_seqused(s2, BSIZE, CMP_RATIO)
    seqused_q = make_q_lens(s1, BSIZE)

    name = f"{template_mode.lower()}_ori{ori_mask_mode}"
    if cmp_mask_mode is not None:
        name += f"_cmp{cmp_mask_mode}"
    name += f"_s2{s2}_{layout_q.lower()}_{layout_kv.lower()}"

    case = {
        "testcase_name": [name],
        "layout_q": [layout_q],
        "layout_kv": [layout_kv],
        "q_type": [torch.bfloat16],
        "ori_kv_type": [torch.bfloat16],
        "B": [BSIZE],
        "S1": [s1],
        "S2": [s2],
        "N1": [N1],
        "N2": [N2],
        "D": [D],
        "block_size1": [BLOCK_SIZE],
        "block_num1": [None],
        "softmax_scale": [SOFTMAX_SCALE],
        "return_softmax_lse": [False],
        "ori_mask_mode": [ori_mask_mode],
        "template_mode": [template_mode],
        "cmp_ratio": [None],
        "cmp_mask_mode": [None],
        "ori_win_left": [None],
        "ori_win_right": [None],
        "ori_kv_topk_mode": ["no"],
        "cmp_kv_topk_mode": ["no"],
        "ori_sparse_indices_mode": ["full"],
        "cmp_sparse_indices_mode": ["full"],
        # 默认手动传入 q/kv 的 cu_seqlens 及 seqused
        "cu_seqlens_q": [None],
        "cu_seqlens_ori_kv": [None],
        "cu_seqlens_cmp_kv": [None],
        "seqused_q": [None],
        "seqused_ori_kv": [seqused_ori],
        "seqused_cmp_kv": [None],
        "cmp_residual_kv": [None],
    }

    # ori mask_mode 3/4 需要显式 winLeft/winRight; mask_mode=0 传 None
    if ori_mask_mode in (3, 4):
        case["ori_win_left"] = [127]
        case["ori_win_right"] = [0]

    # cmp 相关参数: HCA/CSA 需要 cmp_kv
    if template_mode in ("HCA", "CSA"):
        case["cmp_kv_type"] = [torch.bfloat16]
        case["cmp_ratio"] = [CMP_RATIO]
        case["cmp_mask_mode"] = [cmp_mask_mode]
        case["block_size2"] = [BLOCK_SIZE]
        case["block_num2"] = [None]
        case["seqused_cmp_kv"] = [seqused_cmp]
        case["cmp_residual_kv"] = [residual]
        if template_mode == "CSA":
            case["K"] = [512]
            # CSA: cmp_mask_mode=0 时 NPU要求提供 cmp_topk_length, 需启用 fullK 生成
            if cmp_mask_mode == 0:
                case["cmp_kv_topk_mode"] = ["fullK"]
        else:
            case["K"] = [None]
    else:
        case["cmp_kv_type"] = [None]
        case["K"] = [None]

    # layout 相关: 手动传入 cu_seqlens / seqused
    if layout_q == "TND":
        cu_q = make_cu_seqlens(seqused_q)
        case["cu_seqlens_q"] = [cu_q]
        case["T1"] = [cu_q[-1]]
    else:  # BSND
        case["T1"] = [None]
        case["seqused_q"] = [seqused_q]

    if layout_kv == "TND":
        cu_ori = make_cu_seqlens(seqused_ori)
        case["cu_seqlens_ori_kv"] = [cu_ori]
        case["T2"] = [cu_ori[-1]]
        if template_mode in ("HCA", "CSA"):
            cu_cmp = make_cu_seqlens(seqused_cmp)
            case["cu_seqlens_cmp_kv"] = [cu_cmp]
            case["T3"] = [cu_cmp[-1]]
        else:
            case["T3"] = [None]
    else:  # PA_BBND / BSND
        case["T2"] = [None]
        case["T3"] = [None]

    return case


TEST_PARAMS = {}

# SWA: ori_mask_mode 0/3/4, 无 cmp
for ori_mm in [0, 3, 4]:
    for s2 in S2_CANDIDATES:
        for lq, lkv in LAYOUT_COMBOS:
            key = f"swa_ori{ori_mm}_s2{s2}_{lq}_{lkv}"
            TEST_PARAMS[key] = build_case("SWA", ori_mm, None, s2, lq, lkv)

# HCA: ori_mask_mode 0/3/4 × cmp_mask_mode 0/3
for ori_mm in [0, 3, 4]:
    for cmp_mm in [0, 3]:
        for s2 in S2_CANDIDATES:
            for lq, lkv in LAYOUT_COMBOS:
                key = f"hca_ori{ori_mm}_cmp{cmp_mm}_s2{s2}_{lq}_{lkv}"
                TEST_PARAMS[key] = build_case("HCA", ori_mm, cmp_mm, s2, lq, lkv)

# CSA: ori_mask_mode 0/3/4 × cmp_mask_mode 0/3
for ori_mm in [0, 3, 4]:
    for cmp_mm in [0, 3]:
        for s2 in S2_CANDIDATES:
            for lq, lkv in LAYOUT_COMBOS:
                key = f"csa_ori{ori_mm}_cmp{cmp_mm}_s2{s2}_{lq}_{lkv}"
                TEST_PARAMS[key] = build_case("CSA", ori_mm, cmp_mm, s2, lq, lkv)

ENABLED_PARAMS = [{**TEST_PARAMS[key], "testcase_name": [key]} for key in TEST_PARAMS.keys()]
