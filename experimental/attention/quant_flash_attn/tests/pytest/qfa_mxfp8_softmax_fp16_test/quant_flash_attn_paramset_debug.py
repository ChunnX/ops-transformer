#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from quant_flash_attn_paramset_common import expand_paramset_to_cases

# 调试用例集（seq_used=0 空批探索，2026-09-27 生成、待上板验证）
# =====================================================================
# 修复存档（本文件历史上承载的 FAIL 用例及根因，均已修复并迁至 func_rdv）
# =====================================================================
# ① 2026-09-24（首批）S1 非对齐·V1 部分列（QS192/QS444，自 func_rdv 回流）
#    失败模式：actMSize ∈ (128,256) 时 V1 subBlock 输出全错（QS192 通过率 67.2%≈128/192）
#    根因：LoadPToL0B 固定满宽 pitch（kStep=4/dstStride=8/xStep=8）与 MatmulVP 动态 n
#    不匹配——转置加载下 Mmad 按自身 n 紧凑分形距读 L0B，呈现宽度须整体 = n
#    修复：LoadPToL0B 三参数随 actVecS1SizeAlign32 动态化（详见 cube block 头注释）
#
# ② 2026-09-24 S1 极小 × 多块（QS1_KVS832）：C2_V2 释放竞态
#    失败模式：非末块 V2 释放 mm2Res 挂 PIPE_MTE3 但无 V→MTE3 屏障，flag 提前触发，
#    AIC 下一次 FixpipeMm2 覆写在途 VfFlashUpdateAcc 读（窄 n 时窗口放大，时变指纹）
#    修复：释放管道按任务类型区分——末块 PIPE_MTE3（corrections #8 原语义）、非末块 PIPE_V
#
# ③ 2026-09-24 S2 任意长度（KVS100/65/388/556，设备崩溃 / 丢 key）
#    根因三处：MatmulVP k 未对齐 64（MmadMx K∈64Z 硬约束 → 崩溃）；P 槽尾行陈旧数据
#    混入 k 对齐扩展行；VfSoftmaxFp16ToFp8 P 循环缺尾行处理（validRows%4 余量被丢弃）
#    修复：k=align64 + V1 尾 subLoop 整槽清零 + P 尾行配对链（含奇数余量零补位）
#    配套：golden _get_v_block_scale 改 ceil 组 + 尾组 0 补齐
#
# ④ 2026-09-24 per-batch seqused（B2/B3 变长组合）
#    kernel 零修改即正确（actSeqLens 按 batch 解析为设计内能力）；
#    golden cpu_mxfp8_golden 改逐 batch 循环（原为满 shape 全量循环），
#    test_runner 对无效 query 行双侧清零（kernel 不写该区域，DEAL_ZERO 清零为 TODO）
#
# ⑤ 2026-09-27 S2=1 边界系列（GQA G=4/16、MHA Nq2Nkv2、B2 双批、per-batch 混合、B32 满核）
#    上板 6/6 全过（100% PctRlt），已迁至 func_rdv——kernel 零修改即正确
#    （k=align64(1)=64 + P 零填充 + softmax 尾行单行链均为 ③ 修复的既有路径）
# =====================================================================
TEST_PARAMS = {
    # ===== seq_used=0 空批系列（2026-09-27 生成，探索零长 batch 的全链路行为）=====
    # 语义背景：metadata CheckNeedInitOutput 对 q=0 / kv=0 批均置 needInitOutput=1（输出应清零），
    # 但 kernel DEAL_ZERO 的 AIV 清零仍是 TODO、needInitOutput 读入后未消费、torch 扩展输出
    # 用 torch.empty 分配——零批输出当前无清零保障。分三组隔离验证：
    #   A 组 seqused_q=0（KV 有效）：golden 无 q 循环输出 0，test_runner 掩码将零批双侧
    #       清零后对比 → 预期通过（验证调度器 DEAL_ZERO/SKIP_ZERO 不崩不挂）
    #   B 组 seqused_kv=0（Q 有效）：掩码不遮挡（seqused_q=满）→ 严格对比；golden 分母
    #       零保护输出 0，但 lse 行 m_acc=None.float() 会崩（golden 已知缺陷）；kernel
    #       不写零批输出=未初始化内存 → 预期 ERROR/FAIL，暴露 DEAL_ZERO TODO 空白
    #   C 组 双轴混合：同用例内两种路径交错
    # 注：全批均为 0 不构造（max(seqused)=0 → 空 tensor，host 侧 GetEmptyTensorFlag 拒绝）
    # ===== A 组：seqused_q 含 0（B2/B3，KV 满宽 256）=====
    # 首批空 Q：batch0 q=0（DEAL_ZERO）× batch1 正常
    "BNSD_B2_QS0x256_KVS256x256_Nq1_Nkv1_D128_qm3": {
        "B": [2],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[0, 256]],
        "seqused_kv": [[256, 256]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # 尾批空 Q：batch0 正常 × batch1 q=0（零批在末位，验证行推进收尾）
    "BNSD_B2_QS256x0_KVS256x256_Nq1_Nkv1_D128_qm3": {
        "B": [2],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[256, 0]],
        "seqused_kv": [[256, 256]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # 空批夹正常批：q=[0, 256, 0]（首尾零、中间正常，验证零批间任务推进）
    "BNSD_B3_QS0x256x0_KVS256x256x256_Nq1_Nkv1_D128_qm3": {
        "B": [3],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[0, 256, 0]],
        "seqused_kv": [[256, 256, 256]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # ===== B 组：seqused_kv 含 0（Q 满宽 256，严格对比路径）=====
    # 首批空 KV：batch0 kv=0（q 有效，输出应全 0）× batch1 正常
    "BNSD_B2_QS256x256_KVS0x256_Nq1_Nkv1_D128_qm3": {
        "B": [2],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[256, 256]],
        "seqused_kv": [[0, 256]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # 尾批空 KV：batch0 正常 × batch1 kv=0
    "BNSD_B2_QS256x256_KVS256x0_Nq1_Nkv1_D128_qm3": {
        "B": [2],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[256, 256]],
        "seqused_kv": [[256, 0]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # 空批夹正常批：kv=[0, 256, 0]（首尾零 KV、中间正常）
    "BNSD_B3_QS256x256x256_KVS0x256x0_Nq1_Nkv1_D128_qm3": {
        "B": [3],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[256, 256, 256]],
        "seqused_kv": [[0, 256, 0]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # ===== C 组：双轴混合 0 =====
    # batch0 空 Q × batch1 空 KV：同用例覆盖 A/B 两条路径
    "BNSD_B2_QS0x256_KVS256x0_Nq1_Nkv1_D128_qm3": {
        "B": [2],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[0, 256]],
        "seqused_kv": [[256, 0]],
        "enable_pa": [False],
        "kv_cache_layout": [None],
        "block_size": [None],
        "mask_mode": [0],
        "q_scale_layout": ["BNSD"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    # 双轴交错变长：q=[0,256,0,128] × kv=[256,0,65,832]——空Q批/空KV批/双零批/正常非对齐批
    # （batch3 叠加 S1=128 半宽部分行 + S2=832 任意长度，零批与泛化路径同场）
    "BNSD_B4_QS0x256x0x128_KVS256x0x65x832_Nq1_Nkv1_D128_qm3": {
        "B": [4],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [None],
        "cu_seqlens_kv": [None],
        "seqused_q": [[0, 256, 0, 128]],
        "seqused_kv": [[256, 0, 65, 832]],
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
