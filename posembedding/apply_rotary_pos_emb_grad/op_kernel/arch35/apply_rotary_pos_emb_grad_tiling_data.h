/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_grad_tiling_data.h
 * \brief apply_rotary_pos_emb_grad kernel 侧 tiling 数据结构
 * 相比单路算子的 RopeGradTilingData，新增 nQ/nK/layout 字段支持双路
 */

#ifndef _APPLY_ROTARY_POS_EMB_GRAD_TILING_DATA_H_
#define _APPLY_ROTARY_POS_EMB_GRAD_TILING_DATA_H_

#include "atvoss/reduce/reduce_tiling_data.h"

// ============================================================================
// 通用 tiling 参数 — BAB/AB/A 共用, BNSD(Layout=2) 参数预留
// 同时处理 Q 和 K 双路
// ============================================================================
struct ApplyRopeGradRegbaseParams {
    int64_t b;                 // batch size
    int64_t s;                 // seq_len
    int64_t d;                 // head_dim
    int64_t nQ;                // Q num_heads
    int64_t nK;                // K num_heads (GQA: nK <= nQ)
    int64_t blockNumB;         // B 轴分核数
    int64_t blockFactorB;      // 每核处理的 B 数
    int64_t blockNumS;         // S 轴分核数
    int64_t blockFactorS;      // 每核处理的 S 数
    int64_t ubFactorS;         // UB 单次处理的 S 数
    int64_t ubFactorB;         // UB 单次处理的 B 数
    int64_t ubLoopNumN;        // N 轴 UB 循环次数（用 max(nQ,nK) 计算）
    int64_t ubFactorN;         // UB 单次处理的 N 数
    int64_t ubTailFactorN;     // UB N 轴尾块大小
    int64_t usedCoreNum;       // 实际使用的核数
    int64_t rotaryMode;        // HALF = 0
    int64_t layout;            // internal Layout: 0=BSND, 1=SBND, 2=BNSD(reserved), 3=NO_BROADCAST
};

// ============================================================================
// SBND (Layout=1) AB split tiling 参数
// ============================================================================
struct ApplyRopeGradRegbaseABParams {
    int64_t b;
    int64_t s;
    int64_t d;
    int64_t nQ;
    int64_t nK;
    int64_t dAlign;
    int64_t dSplitCoef;        // = 2 (half mode)
    int64_t blockNumBS;        // BS 合并轴分核数
    int64_t blockFactorBS;     // 每核处理的 BS 数
    int64_t blockTailBS;       // 尾核 BS 数
    int64_t blockNumN;         // N 轴分核数
    int64_t blockFactorN;      // 每核处理的 N 数
    int64_t blockTailN;        // 尾核 N 数
    int64_t ubFactorBS;        // UB 单次处理的 BS 数
    int64_t ubFactorN;         // UB 单次处理的 N 数
    int64_t usedCoreNum;
    int64_t rotaryMode;
};

// ============================================================================
// grad_cos/grad_sin 计算阶段 (RotaryX + Reduce) 的 tiling 参数
// 对 query 和 key 分别做 rotate_half，存入 workspace
// ============================================================================
struct ApplyRotaryXParams {
    int64_t b;
    int64_t s;
    int64_t d;
    int64_t nQ;                // Q 路的 N 维度大小
    int64_t nK;                // K 路的 N 维度大小
    int64_t usedCoreNum;
    int64_t blockFactorN;      // 每核处理的总 N 数（= nQ + nK 或其他 split 方式）
    int64_t blockTailFactorN;  // 尾核 N 数
    int64_t ubFactorN;         // UB 单次处理的 N 数
    int64_t rotaryMode;        // HALF = 0
};

// ============================================================================
// 顶层 tiling data，包含所有 split 模式需要的参数
// ============================================================================
struct ApplyRopeGradTilingData {
    Ops::Base::ReduceOpTilingData reduceTiling;
    ApplyRopeGradRegbaseParams ropeGradParams;        // BAB/A 共用
    ApplyRopeGradRegbaseABParams ropeGradABParams;    // AB 专用
    ApplyRotaryXParams rotaryXParams;                 // RotaryX + Reduce 阶段
    uint32_t dCosFlag;         // 0=不计算 grad_cos/grad_sin, 1=需要计算
    uint32_t layout;           // 0=BSND, 1=SBND, 2=BNSD(reserved), 3=NO_BROADCAST
};

#endif // _APPLY_ROTARY_POS_EMB_GRAD_TILING_DATA_H_
