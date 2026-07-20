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

####### STC 用例设计原则 ########
# 仅测试伪量化 fp4 场景（quant_mode=1）
# 除了 shape 相关参数外，其他参数均使用 func_signature.csv 默认值：
#   Dtype=fp16, mask_mode=0, win_left/right=默认, return_softmax_lse=0, softmax_scale=自动

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
        "return_softmax_lse": [0],
        "cu_seqlens_q": [[None]],
        "cu_seqlens_kv": [[None]],
        "seqused_q": [[None]],
        "seqused_kv": [[None]],
        "q_range": [(-5.0, 5.0)]}

TestCases = {
    # ── 基础 ──
    "BASE_01": {**BASE},

    # ── Q layout ──
    "Q_BSND_02": {**BASE, "layout_q": ["BSND"], "layout_out": ["BSND"]},
    "Q_TND_03": {**BASE, "layout_q": ["TND"], "layout_out": ["TND"],
                 "cu_seqlens_q": [[0, 64, 128]], "cu_seqlens_kv": [[0, 1024, 2048]]},

    # ── KV PA layout ──
    "KV_BBND_04": {**BASE, "layout_kv": ["PA_BBND"]},
    "KV_BNBD_05": {**BASE, "layout_kv": ["PA_BNBD"]},

    # ── GQA / MHA / MQA ──
    "GQA_06": {**BASE, "N1": [8], "N2": [2]},
    "MHA_07": {**BASE, "N1": [8], "N2": [8]},
    "MQA_08": {**BASE, "N1": [8], "N2": [1]},

    # ── D 维度（32 对齐）──
    "D64_09": {**BASE, "D": [64], "S2": [512]},
    "D256_10": {**BASE, "D": [256]},

    # ── block_size ──
    "BS64_11": {**BASE, "block_size": [64], "S2": [512]},
    "BS128_12": {**BASE, "block_size": [128]},

    # ── prefill / decode ──
    "PREFILL_13": {**BASE, "S1": [1024], "S2": [1024]},
    "DECODE_14": {**BASE, "S1": [1], "S2": [1024]},

    # ── batch ──
    "BATCH1_15": {**BASE, "B": [1]},
    "BATCH4_16": {**BASE, "B": [4]},

    # ── TND 变长序列 ──
    "VARLEN_17": {**BASE, "layout_q": ["TND"], "layout_out": ["TND"],
                  "cu_seqlens_q": [[0, 32, 96]], "cu_seqlens_kv": [[0, 512, 1536]]},

    # ── 大序列 ──
    "LARGE_18": {**BASE, "B": [1], "S1": [128], "S2": [4096]},

    # ── output layout ──
    "OUT_BSND_19": {**BASE, "layout_q": ["BNSD"], "layout_out": ["BSND"]},
}
