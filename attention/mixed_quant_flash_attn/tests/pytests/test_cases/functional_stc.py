#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

####### 参数说明（按 func_signature.csv）########
# 必选 INPUT:  q, k, v, k_descale, v_descale, block_table
# 可选 INPUT:  cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask, metadata
# 必选 ATTR:   quant_mode
# 可选 ATTR:   softmax_scale, mask_mode, win_left, win_right,
#              max_seqlen_q, max_seqlen_kv, layout_q, layout_kv, layout_out, return_softmax_lse
#
# quant_mode=1: q不量化, kv为fp4_e2m1 (uint8伪装)
# layout_kv: PA_BBND 或 PA_BNBD (PA模式默认开启)
# mask_mode: 0=defaultMask, 3=rightDownCausal, 4=band

BASE = {"B": [2],
        "N1": [8],
        "N2": [1],
        "S1": [64],
        "S2": [1024],
        "D": [128],
        "Dtype": ["fp16"],
        "layout_q": ["BNSD"],
        "layout_kv": ["PA_BNBD"],
        "layout_out": ["BNSD"],
        "block_size": [128],
        "quant_mode": [1],
        "mask_mode": [0],
        "win_left": [-1],
        "win_right": [-1],
        "return_softmax_lse": [0],
        "cu_seqlens_q": [[None]],
        "cu_seqlens_kv": [[None]],
        "seqused_q": [[None]],
        "seqused_kv": [[None]],
        "q_range": [(-5.0, 5.0)]}

TestCases = {
    # ── dtype ──
    "FP16_01": {**BASE, "Dtype": ["fp16"]},
    "BF16_02": {**BASE, "Dtype": ["bf16"]},

    # ── Q layout ──
    "Q_BSND_03": {**BASE, "layout_q": ["BSND"], "layout_out": ["BSND"]},
    "Q_TND_04": {**BASE,
        "layout_q": ["TND"], "layout_out": ["TND"],
        "cu_seqlens_q": [[0, 64, 128]],
        "cu_seqlens_kv": [[0, 1024, 2048]]},

    # ── KV PA layout ──
    "KV_BBND_05": {**BASE, "layout_kv": ["PA_BBND"]},
    "KV_BNBD_06": {**BASE, "layout_kv": ["PA_BNBD"]},

    # ── GQA / MHA ──
    "GQA_07": {**BASE, "N1": [8], "N2": [2]},
    "MHA_08": {**BASE, "N1": [8], "N2": [8]},

    # ── D 维度 ──
    "D64_09": {**BASE, "D": [64], "S2": [512]},
    "D256_10": {**BASE, "D": [256]},

    # ── block_size ──
    "BS64_11": {**BASE, "block_size": [64], "S2": [512]},
    "BS128_12": {**BASE, "block_size": [128]},

    # ── mask ──
    "MASK_CAUSAL_13": {**BASE, "mask_mode": [3]},
    "MASK_BAND_14": {**BASE, "mask_mode": [4], "win_left": [32], "win_right": [32]},

    # ── LSE ──
    "LSE_15": {**BASE, "return_softmax_lse": [1]},

    # ── TND + PA ──
    "TND_PA_16": {**BASE,
        "layout_q": ["TND"], "layout_kv": ["PA_BNBD"],
        "cu_seqlens_q": [[0, 64, 128]],
        "cu_seqlens_kv": [[0, 1024, 2048]]},

    # ── output layout ──
    "OUT_BSND_17": {**BASE, "layout_out": ["BSND"]},
    "OUT_TND_18": {**BASE,
        "layout_q": ["TND"], "layout_out": ["TND"],
        "cu_seqlens_q": [[0, 64, 128]],
        "cu_seqlens_kv": [[0, 1024, 2048]]},

    # ── decode 场景 (S1=1) ──
    "DECODE_19": {**BASE, "S1": [1], "mask_mode": [3]},

    # ── 单 batch ──
    "BATCH1_20": {**BASE, "B": [1]},

    # ── 变长序列 (TND, 各 batch 长度不同) ──
    "VARLEN_21": {**BASE,
        "layout_q": ["TND"], "layout_kv": ["PA_BNBD"],
        "cu_seqlens_q": [[0, 32, 96]],
        "cu_seqlens_kv": [[0, 512, 1536]]},

    # ── 大序列 ──
    "LARGE_22": {**BASE, "B": [1], "S1": [128], "S2": [4096]},
}
