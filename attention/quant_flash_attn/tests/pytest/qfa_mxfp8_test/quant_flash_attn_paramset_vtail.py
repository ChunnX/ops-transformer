#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# V尾块高精窗口(BF16直存)专属用例集, 对应 issue#3 需求:
#   - 一期约束: quant_mode=1(MXFP8), PA场景, layout_kv=BnNBsD(PA_BNBD),
#     q_scale_layout=N2TGD(decode), tail∈[0,64), (seqused_kv-tail)%64==0
#   - 覆盖点: 多batch不同尾长(含尾长极小值)/GQA/纯尾块等边界
from quant_flash_attn_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {
    # B=2 双batch混合尾长(61/2) + GQA(Nq4:Nkv2) decode:
    #   batch0: kv=1085 -> main=1024(2x512 block), tail=61
    #   batch1: kv=642  -> main=640,               tail=2 (尾长极小边界)
    "PA_BnNBsD_VTAIL_B2_QS1_KVS1085_642_T61_T2_Nq4_Nkv2_D128_SP0": {
        "B": [2],
        "N_q": [4],
        "N_kv": [2],
        "D": [128],
        "cu_seqlens_q": [[0, 1, 2]],
        "cu_seqlens_kv": [None],
        "seqused_q": [[1, 1]],
        "seqused_kv": [[1085, 642]],
        "max_seqlen_q": [1],
        "max_seqlen_kv": [1085],
        "enable_pa": [True],
        "kv_cache_layout": ["BnNBsD"],
        "block_size": [512],
        "mask_mode": [0],
        "q_scale_layout": ["N2TGD"],
        "p_scale": [1.0],
        "enable_lse": [False],
        "enable_v_tail": [True],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)
