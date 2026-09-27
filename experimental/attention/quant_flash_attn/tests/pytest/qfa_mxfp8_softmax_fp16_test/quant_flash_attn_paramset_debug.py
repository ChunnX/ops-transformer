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

# 调试用例集（当前为空）
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
TEST_PARAMS = {}

CASES = expand_paramset_to_cases(TEST_PARAMS)
