/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_NORM_POST_PREONLY_COMMON_H_
#define MHC_NORM_POST_PREONLY_COMMON_H_

// ========================================================================
// mhc_norm_post_preonly_kernel.h — 类定义和共享计算模块
//
// 文件结构:
//   1. 类声明 + Init / Process
//   2. 共享计算成员实现
//   3. 成员变量
//   4. 引入各执行路径的实现头文件
// ========================================================================



// ========================================================================
// mhc_norm_post_preonly_common.h — 常量、类型萃取、workspace layout 定义
// ========================================================================

#include "kernel_operator.h"
using namespace AscendC;

// ---------------------------------------------------------------------------
// 全局常量
// ---------------------------------------------------------------------------
constexpr uint32_t N_MAX = 4;
constexpr uint32_t ALIGN8 = 8;
constexpr uint32_t BUFFER_NUM_ONE = 1;
constexpr uint32_t BUFFER_NUM_FOUR = 4;       // phi ping-pong：4槽让相邻o的搬运与计算重叠
constexpr uint32_t GAMMA1_CACHE_MAX_D = 5120; // 目标模型维度以内缓存完整gamma1
constexpr uint32_t SCALAR_BUF_SIZE = 256;     // 标量计算 UB buffer 大小 (bytes)
constexpr uint32_t EXP_DIV_ELEM = 8;          // sigmoid ExpDiv 计算元素数
constexpr uint32_t POST_COMB_SIZE = 8 + 16;   // h_post[8] + h_res[16] buffer 大小
constexpr uint32_t PARTIAL_SLOT_ELEMS = 8;    // 每个lane的partial固定占32B
constexpr uint16_t MID_READY_FLAG = 0;        // R3/RMS_mid平方和交换完成标志
constexpr uint16_t PAIR_READY_FLAG = 1;       // R4/R5 partial交换完成标志

// ---------------------------------------------------------------------------
// Type trait: fp32->half uses CAST_NONE, fp32->bfloat16_t uses CAST_RINT
// (per CANN RmsNorm reference)
// ---------------------------------------------------------------------------
template <typename A, typename B>
struct IsSameT {
    static constexpr bool value = false;
};
template <typename A>
struct IsSameT<A, A> {
    static constexpr bool value = true;
};







// clang-format on







// 引入各执行路径的方法实现

#endif // MHC_NORM_POST_PREONLY_COMMON_H_
