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

from quant_flash_attn_paramset_common import expand_paramset_to_cases

# 第一个 smoke 用例：BNSD / 非 PA / mask_mode=0 / quant_mode=3 / D=128 / S1=S2=256 / 不输出 lse
TEST_PARAMS = {
    "BNSD_B1_S256_Nq1_Nkv1_D128_qm3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[256]],
        "seqused_kv": [[256]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)