/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

// ========================================================================
// mhc_sandwich_norm_post_preonly_common.h — 常量、类型萃取、workspace layout 定义
// ========================================================================
#ifndef MHC_SANDWICH_NORM_POST_PREONLY_COMMON_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_COMMON_H_

#include "kernel_operator.h"
using namespace AscendC;

// ---------------------------------------------------------------------------
// 全局常量
// ---------------------------------------------------------------------------
constexpr uint32_t N_MAX  = 4;
constexpr uint32_t ALIGN8 = 8;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t BUFFER_NUM_TWO = 2;
constexpr uint32_t BUFFER_NUM_ONE = 1;
constexpr uint32_t DUAL_CORE_COUNT   = 2;     // 双核模式核数
constexpr uint32_t D_MAX             = 5120;   // 最大 D 维度
constexpr uint32_t D_TILE_MAX        = 2560;   // 单 tile 最大 D
constexpr uint32_t D_TILE_ALIGN_ELEM = 128;    // D轴 tiling 对齐元素数 (256B / 2B)
constexpr uint32_t SCALAR_BUF_SIZE   = 256;    // 标量计算 UB buffer 大小 (bytes)
constexpr uint32_t SYNC_UB_SIZE      = 32;     // 同步 UB buffer 大小 (bytes)
constexpr uint32_t EXP_DIV_ELEM      = 8;      // sigmoid ExpDiv 计算元素数
constexpr uint32_t POST_COMB_SIZE    = 8 + 16; // h_post[8] + h_res[16] buffer 大小

// ---------------------------------------------------------------------------
// Workspace layout per token-pair (2 cores sharing 1 token):
//
// DMA-based sync: all reads/writes through MTE2/MTE3 pipes for L2 coherency.
// Each sync block is 8 floats (32 bytes, aligned for efficient DMA).
//
//   [0..7]:                 sync1 block core0: [sumSqPre, xHat[0..3], flag, pad, pad]
//   [8..15]:                sync1 block core1: [sumSqPre, xHat[0..3], flag, pad, pad]
//   [16..16+myN*D-1]:       x2 core0 (myN heads x D floats)
//   [16+myN*D..16+N*D-1]:   x2 core1 (myN heads x D floats)
// Total per pair: 16 + N*D floats
// ---------------------------------------------------------------------------
constexpr uint32_t WS_SYNC1_CORE0 = 0;
constexpr uint32_t WS_SYNC1_CORE1 = 8;
constexpr uint32_t WS_HIN_BASE    = 16;
constexpr uint32_t WS_SYNC_BLOCK  = 8;    // int32 elements per core (32B DMA aligned)
constexpr uint32_t WS_NUM_EVENTS  = 1;    // sense-reversing barrier: single flag per core

// ---------------------------------------------------------------------------
// Type trait: fp32->half uses CAST_NONE, fp32->bfloat16_t uses CAST_RINT
// (per CANN RmsNorm reference)
// ---------------------------------------------------------------------------
template<typename A, typename B> struct IsSameT { static constexpr bool value = false; };
template<typename A> struct IsSameT<A, A> { static constexpr bool value = true; };

#endif // MHC_SANDWICH_NORM_POST_PREONLY_COMMON_H_

