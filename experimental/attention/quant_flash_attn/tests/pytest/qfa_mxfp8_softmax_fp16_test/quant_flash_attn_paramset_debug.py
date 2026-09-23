#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, and you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# You should have received a copy of the License along with this program.
# If not, see <https://www.hiascend.com/legal/cannlicense>.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# -----------------------------------------------------------------------------------------------------------

from quant_flash_attn_paramset_common import expand_paramset_to_cases

# 调试用例（2026-09-23 泛化验证 FAIL，待修复：S1 非对齐·V1 部分列路径）
# 失败模式：actMSize ∈ (128, 256) 时 V1 subBlock 输出全错（V0 全对；V1 满宽/全空均正常）
# —— QS192 通过率 67.2% ≈ 128/192、QS444 通过率 86.7% ≈ 384/444，失配区域精确对齐 V1 列
# 通过的 12 个泛化用例及 GQA 泛化系列（G=1/2/3/4/8/16）已迁至 quant_flash_attn_paramset_func_rdv.py
TEST_PARAMS = {
    # S1 非对齐·部分列 V1=64：actMSize=192（V0=128, V1=64）——V1 输出全错
    "BNSD_B1_QS192_KVS256_Nq1_Nkv1_D128_qm3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[192]],
        "seqused_kv": [[256]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # S1 非对齐·非 64 倍数尾：256+188（actMSizeAlign32=192，V1=60 非整组尾）——尾行 V1 输出全错
    "BNSD_B1_QS444_KVS256_Nq1_Nkv1_D128_qm3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[444]],
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
