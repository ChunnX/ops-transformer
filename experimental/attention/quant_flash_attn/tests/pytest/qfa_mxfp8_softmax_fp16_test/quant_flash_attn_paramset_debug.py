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

# 调度覆盖验证用例（kernel 层核间流水，32 AIC + 64 AIV，S1/S2 base=256）
# 命名沿用主线 QS<seqused_q>_KVS<seqused_kv> 惯例
TEST_PARAMS = {
    # smoke：单行单块，S1/S2 各 256（恰好一个 base 块）
    "BNSD_B1_QS256_KVS256_Nq1_Nkv1_D128_qm3": {
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
    # 多 S2 分块：单行 3 个 S2 块（256+256+192 尾块），验证 isFirstS2Loop/isLastS2Loop 翻转、
    # softmaxStateSlot 跨块推进、actSingleLoopS2Size=192 尾块、C2/V2 逐块触发
    # 注：S2 需为 QUANT_GROUP_SIZE(64) 的倍数（golden 生成 reshape 约束）
    "BNSD_B1_QS256_KVS704_Nq1_Nkv1_D128_qm3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[256]],
        "seqused_kv": [[704]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # 多 gS1：3 行（256+256+188），验证 actMSize=188 尾块、mloop 行推进、逐行任务创建
    "BNSD_B1_QS700_KVS256_Nq1_Nkv1_D128_qm3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[700]],
        "seqused_kv": [[256]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # 多 bN2：B=2 × Nq=2(g=2) → 4 个 bN2，每 bN2 2 个 gS1 → 共 8 行 16 任务，
    # 验证 bN2 维度推进、n2Idx/realN2Idx 计算与跨行调度
    "BNSD_B2_QS512_KVS512_Nq2_Nkv1_D128_qm3": {
        "B": [2],
        "N_q": [2],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[512, 512]],
        "seqused_kv": [[512, 512]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # DEAL_ZERO：batch0 actSeqLen=0 走 DEAL_ZERO/SKIP_ZERO 分支，batch1 正常，
    # 验证空 batch 快进跳过与正常路径共存
    "BNSD_B2_QS0_256_KVS0_256_Nq1_Nkv1_D128_qm3": {
        "B": [2],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[0, 256]],
        "seqused_kv": [[0, 256]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # 行切分多核：B=8 × Nq=8(g=8) → 64 行 > 32 核，每核分到整数行（bN2/gS1 端点切分），
    # 验证多核 metadata 分发、ShouldDispatchTask 边界
    "BNSD_B8_QS256_KVS256_Nq8_Nkv1_D128_qm3": {
        "B": [8],
        "N_q": [8],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[256] * 8],
        "seqused_kv": [[256] * 8],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # 单核多 S2 块顺序执行：单行 8 个 S2 块（2048/256）。
    # 注：当前 S2 切分策略按 bN2 粒度分配，同一 bN2 的所有 S2 块绑定同一核（不跨核），
    # 本用例验证单核内 8 块顺序执行、slot 跨 8 块不变（同行 mloop 相同）、C2/V2 逐块触发
    "BNSD_B1_QS256_KVS2048_Nq1_Nkv1_D128_qm3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[256]],
        "seqused_kv": [[2048]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # 大 shape：B=4 × Nq=8 → 32 行，每行 8 gS1 × 16 S2 块 = 4096 任务，
    # 验证大任务量下流水稳定性与调度总量推演
    "BNSD_B4_QS2048_KVS4096_Nq8_Nkv1_D128_qm3": {
        "B": [4],
        "N_q": [8],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[2048] * 4],
        "seqused_kv": [[4096] * 4],
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
