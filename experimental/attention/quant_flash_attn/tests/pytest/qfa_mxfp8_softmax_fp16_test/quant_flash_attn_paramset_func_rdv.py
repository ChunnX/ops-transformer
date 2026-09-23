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

# 功能正确性 RDV 用例（2026-09-23 泛化验证通过：全部 100.0000% PctRlt）
# 维度矩阵：
#   S2（KV 轴）：256 对齐（整块）/ 128 对齐（单 subLoop 尾块）/ 非对齐（64 粒度尾 subLoop）
#   S1（Q 轴）：256 对齐（整行）/ 128 对齐（V1 空列）
#   bN2（行 = B×N_q×ceil(S1/256)，跨 32 核切分）：单核 / 分不满（≤32 行）/ 大于 32（核内多行）
#   GQA（G = N_q/N_kv）：2 / 3（非 2 幂）/ 4 / 8 / 16（G=1 MHA 基线即基础冒烟用例）
# 注：S2 需为 QUANT_GROUP_SIZE(64) 的倍数（golden 生成 V-scale reshape 约束）；S1 无此约束
TEST_PARAMS = {
    # ===== 基础冒烟（兼 bN2 单核：1 行 → 1 核，S1/S2 各一个 base 块）=====
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
    # ===== S2 系列（B1/Nq1/QS256 固定，单行隔离 S2 维度）=====
    # S2 256 对齐·2 块：跨块 accMax/pscale/rowsum 链 + 单核多块顺序执行（同 bN2 绑同核）
    "BNSD_B1_QS256_KVS512_Nq1_Nkv1_D128_qm3": {
        "B": [1], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[512]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # S2 128 对齐·尾块单 subLoop：256+128，第二块 c1v1Loop=1（首 subLoop 即末 subLoop，pscale 恒 2^0）
    "BNSD_B1_QS256_KVS384_Nq1_Nkv1_D128_qm3": {
        "B": [1], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[384]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # S2 128 对齐·单 subLoop：整块 128（c1v1Loop=1 满宽，actSubKSize=128）
    "BNSD_B1_QS256_KVS128_Nq1_Nkv1_D128_qm3": {
        "B": [1], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[128]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # S2 非对齐·尾 subLoop 64：128+64（c1v1Loop=2，尾节 validRows=64，C2 actSubKSizeAlign64=64）
    "BNSD_B1_QS256_KVS192_Nq1_Nkv1_D128_qm3": {
        "B": [1], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[192]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # S2 非对齐·极小：单 subLoop 64（V-scale 单组，Mmad m=64，pscale 网格 yStep=1）
    "BNSD_B1_QS256_KVS64_Nq1_Nkv1_D128_qm3": {
        "B": [1], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[64]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # ===== S1 系列（B1/Nq1/KVS256 固定，隔离 S1 维度）=====
    # S1 256 对齐·2 gS1：跨行任务推进（mloop 行推进、softmaxStateSlot 跨行复位）
    "BNSD_B1_QS512_KVS256_Nq1_Nkv1_D128_qm3": {
        "B": [1], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[512]], "seqused_kv": [[256]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # S1 128 对齐·尾行 V1 空：256+128，第二行 actMSize=128（vCore=1 无有效列跳过路径）
    "BNSD_B1_QS384_KVS256_Nq1_Nkv1_D128_qm3": {
        "B": [1], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[384]], "seqused_kv": [[256]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # S1 128 对齐·单 V 半：actMSize=128（V0 满宽、V1 全空）
    "BNSD_B1_QS128_KVS256_Nq1_Nkv1_D128_qm3": {
        "B": [1], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[128]], "seqused_kv": [[256]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # ===== GQA 系列（B1/Nkv1/QS256/KVS256 固定，G = N_q/N_kv 隔离头组维度）=====
    # 折叠头维寻址（realN2 = n2×G, realG = 1）：Q/Qscale 按 realN2Idx 扁平头索引搬入，
    # K/V 组内共享（n2Idx = realN2Idx/G）；G=3 为非 2 幂泛化，G=1 基线即上方基础冒烟用例
    "BNSD_B1_QS256_KVS256_Nq2_Nkv1_D128_qm3": {
        "B": [1], "N_q": [2], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[256]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    "BNSD_B1_QS256_KVS256_Nq3_Nkv1_D128_qm3": {
        "B": [1], "N_q": [3], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[256]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    "BNSD_B1_QS256_KVS256_Nq4_Nkv1_D128_qm3": {
        "B": [1], "N_q": [4], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[256]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    "BNSD_B1_QS256_KVS256_Nq8_Nkv1_D128_qm3": {
        "B": [1], "N_q": [8], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[256]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    "BNSD_B1_QS256_KVS256_Nq16_Nkv1_D128_qm3": {
        "B": [1], "N_q": [16], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[256]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # ===== bN2 系列（QS256/KVS256 固定 → 行 = B×N_q，隔离行数/核数维度）=====
    # bN2 分不满·8 行：B2×Nq4（多 head 维度推进，8/32 核有任务）
    "BNSD_B2_QS256_KVS256_Nq4_Nkv4_D128_qm3": {
        "B": [2], "N_q": [4], "N_kv": [4], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256, 256]], "seqused_kv": [[256, 256]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # bN2 分不满上界·32 行：B32×Nq1（恰好每核 1 行，全部核有任务）
    "BNSD_B32_QS256_KVS256_Nq1_Nkv1_D128_qm3": {
        "B": [32], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256] * 32], "seqused_kv": [[256] * 32],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
    # bN2 大于 32·48 行：B48×Nq1（不均匀切分：16 核 2 行 + 16 核 1 行，核内多行顺序）
    "BNSD_B48_QS256_KVS256_Nq1_Nkv1_D128_qm3": {
        "B": [48], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256] * 48], "seqused_kv": [[256] * 48],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)
