#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 调试用例集（S2 基本块 512 几何核心覆盖，LSE 各一代表）
CASES = [
    # 单基本块 128×512：C1V1 两轮 + C2 四次 mmad + V2 整块
    {
        "name": "DBG_S2_512",
        "B": 1,
        "N_q": 1,
        "N_kv": 1,
        "D": 128,
        "cu_seqlens_q": None,
        "cu_seqlens_kv": None,
        "seqused_q": [128],
        "seqused_kv": [512],
        "enable_pa": False,
        "kv_cache_layout": None,
        "block_size": None,
        "mask_mode": 0,
        "q_scale_layout": "BNSD",
        "p_scale": 1.0,
        "enable_lse": False,
    },
    # 两基本块（1024）：跨块 rescale 链
    {
        "name": "DBG_S2_1024",
        "B": 1,
        "N_q": 1,
        "N_kv": 1,
        "D": 128,
        "cu_seqlens_q": None,
        "cu_seqlens_kv": None,
        "seqused_q": [128],
        "seqused_kv": [1024],
        "enable_pa": False,
        "kv_cache_layout": None,
        "block_size": None,
        "mask_mode": 0,
        "q_scale_layout": "BNSD",
        "p_scale": 1.0,
        "enable_lse": False,
    },
    # 基本块边界 +1（513）：第二块尾块 1 行
    {
        "name": "DBG_S2_513",
        "B": 1,
        "N_q": 1,
        "N_kv": 1,
        "D": 128,
        "cu_seqlens_q": None,
        "cu_seqlens_kv": None,
        "seqused_q": [128],
        "seqused_kv": [513],
        "enable_pa": False,
        "kv_cache_layout": None,
        "block_size": None,
        "mask_mode": 0,
        "q_scale_layout": "BNSD",
        "p_scale": 1.0,
        "enable_lse": False,
    },
    # subloop 边界（257）：第二轮 subloop 尾行 1
    {
        "name": "DBG_S2_257",
        "B": 1,
        "N_q": 1,
        "N_kv": 1,
        "D": 128,
        "cu_seqlens_q": None,
        "cu_seqlens_kv": None,
        "seqused_q": [128],
        "seqused_kv": [257],
        "enable_pa": False,
        "kv_cache_layout": None,
        "block_size": None,
        "mask_mode": 0,
        "q_scale_layout": "BNSD",
        "p_scale": 1.0,
        "enable_lse": False,
    },
    # GQA + LSE
    {
        "name": "DBG_GQA4_S2_1024_LSE",
        "B": 1,
        "N_q": 4,
        "N_kv": 1,
        "D": 128,
        "cu_seqlens_q": None,
        "cu_seqlens_kv": None,
        "seqused_q": [128],
        "seqused_kv": [1024],
        "enable_pa": False,
        "kv_cache_layout": None,
        "block_size": None,
        "mask_mode": 0,
        "q_scale_layout": "BNSD",
        "p_scale": 1.0,
        "enable_lse": True,
    },
    # per-batch 变长 + 空 kv 批
    {
        "name": "DBG_B4_VAR_ZF",
        "B": 4,
        "N_q": 1,
        "N_kv": 1,
        "D": 128,
        "cu_seqlens_q": None,
        "cu_seqlens_kv": None,
        "seqused_q": [128, 128, 128, 128],
        "seqused_kv": [0, 512, 1024, 513],
        "enable_pa": False,
        "kv_cache_layout": None,
        "block_size": None,
        "mask_mode": 0,
        "q_scale_layout": "BNSD",
        "p_scale": 1.0,
        "enable_lse": False,
    },
]
