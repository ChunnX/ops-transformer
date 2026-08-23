#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

####### MLA 参数说明（m0070 / R1）########
# B: batch size（TND 下由 cu_seqlens_q 推导，可省略）
# N1: query head 数（kv head num 恒 1）
# S1: 每 batch query 序列长度（decode 场景 = 1）
# S2: 每 batch KV cache 长度（与 cache_seqlens 对应；TND 下 cache_seqlens 显式给出）
# D: 固定 512 = head_dim_v（qk nope 段），q/k_cache 最后维恒 576（nope512 + rope64）
# layout_q: ["TND"]；layout_kv: ["PA_NZ"]（PA_NZ，block_table 必传）；layout_out: ["TND"]（恒等于 q，T11）
# mask_mode: 0=NO_MASK, 3=CAUSAL/RIGHT_DOWN
# Dtype: ["fp16", "bf16"]
# return_softmax_lse: LSE 输出开关
# expect_multi_section: True 时断言 metadata sectionNum >= 2（>1-section）

BASE = {
    "N1": [4],
    "S1": [64],
    "S2": [1024],
    "D": [512],
    "DV": [512],
    "Dtype": ["fp16"],
    "layout_q": ["TND"],
    "layout_kv": ["PA_NZ"],
    "layout_out": ["TND"],
    "block_size": [128],
    "return_softmax_lse": [False],
    "cu_seqlens_q": [[None]],
    "cache_seqlens": [[None]],
    "seqused_q": [[None]],
    "mask_mode": [0],
    "q_range": [(-5.0, 5.0)],
    "k_range": [(-5.0, 5.0)],
}

TestCases = {
    # (a) prefill（多 token q，FD=false）+ (c) PA 分页形状（每 batch 8 个 page，首/末页非满）
    "PREFILL_NO_MASK": {**BASE, "B": [2]},
    "PREFILL_CAUSAL": {**BASE, "B": [2], "S1": [64], "S2": [64], "mask_mode": [3]},
    # bf16 prefill
    "PREFILL_BF16": {**BASE, "B": [2], "Dtype": ["bf16"]},
    # (d) LSE 开关
    "PREFILL_LSE": {**BASE, "B": [2], "return_softmax_lse": [True]},
    "PREFILL_CAUSAL_LSE": {
        **BASE,
        "B": [2],
        "S1": [128],
        "S2": [128],
        "mask_mode": [3],
        "return_softmax_lse": [True],
    },
    # (b) decode（每 batch q 长 1，FD path；kv 2048 = 16 pages/batch）
    "DECODE_FD": {**BASE, "B": [8], "S1": [1], "S2": [2048]},
    "DECODE_CAUSAL_BF16": {
        **BASE,
        "B": [4],
        "S1": [1],
        "S2": [2048],
        "mask_mode": [3],
        "Dtype": ["bf16"],
    },
    "DECODE_LSE": {
        **BASE,
        "B": [4],
        "S1": [1],
        "S2": [2048],
        "return_softmax_lse": [True],
    },
    # 变长（batch 间 q/kv 长度不等，mask 0）
    "VARLEN_NO_MASK": {
        **BASE,
        "N1": [8],
        "cu_seqlens_q": [[0, 64, 192, 256, 512]],
        "cache_seqlens": [[1024, 512, 2048, 1024]],
    },
    # 变长 + 因果（S1 != S2，右下对齐 RIGHT_DOWN 语义）
    "VARLEN_CAUSAL": {
        **BASE,
        "N1": [2],
        "cu_seqlens_q": [[0, 32, 96, 224, 288]],
        "cache_seqlens": [[32, 96, 224, 288]],
        "mask_mode": [3],
    },
    # (f) >1-section：64 batch × 1024 token → AICPU 按 96MB L2 预算切多 section（断言 sectionNum>=2）
    "SECTION_64B": {
        **BASE,
        "N1": [1],
        "B": [64],
        "S1": [1024],
        "S2": [1024],
        "expect_multi_section": [True],
    },
    # (f) 替代几何：16 batch × 2048 token（单 head 成本 8.4MB，预算内 12 batch/section）
    "SECTION_16B_2K": {
        **BASE,
        "N1": [1],
        "B": [16],
        "S1": [2048],
        "S2": [2048],
        "expect_multi_section": [True],
    },
}
