/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_NORM_POST_PREONLY_KERNEL_SUPPORT_H_
#define MHC_NORM_POST_PREONLY_KERNEL_SUPPORT_H_

// ========================================================================
// mhc_norm_post_preonly_kernel.h — 类定义和共享计算模块
//
// 文件结构:
//   1. 类声明 + Init / Process
//   2. 共享计算模块
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
constexpr uint32_t N_MAX  = 4;
constexpr uint32_t ALIGN8 = 8;
constexpr uint32_t BUFFER_NUM_TWO = 2;
constexpr uint32_t BUFFER_NUM_ONE = 1;
constexpr uint32_t SCALAR_BUF_SIZE   = 256;    // 标量计算 UB buffer 大小 (bytes)
constexpr uint32_t EXP_DIV_ELEM      = 8;      // sigmoid ExpDiv 计算元素数
constexpr uint32_t POST_COMB_SIZE    = 8 + 16; // h_post[8] + h_res[16] buffer 大小
constexpr uint32_t PARTIAL_SLOT_ELEMS = 8;     // 每个token的partial1固定占32B
constexpr uint16_t PAIR_READY_FLAG = 0;        // 每个token仅使用一次mode1握手

// ---------------------------------------------------------------------------
// Type trait: fp32->half uses CAST_NONE, fp32->bfloat16_t uses CAST_RINT
// (per CANN RmsNorm reference)
// ---------------------------------------------------------------------------
template<typename A, typename B> struct IsSameT { static constexpr bool value = false; };
template<typename A> struct IsSameT<A, A> { static constexpr bool value = true; };







// 引入各执行路径的方法实现



// 构造一个低精度x2_post分块。R1使用normEps_，R2保持h_res的[j,n]索引顺序，
// 并显式保留设计规定的两处低精度往返边界。


// 将完整x2以输出dtype常驻UB，同时完成设计要求的低精度往返；写GM仅用于保留x2_out输出语义。


// 双sibling路径仍在每个lane的私有UB中保留完整x2，便于两个lane按同一顺序执行R1/R2/R3；
// 写回x2_out时每个lane只写自己负责的两个head，避免任何输出竞争。




// R6读取完当前x2分块后，首个head的对应缓存区已经不会再被使用，
// 将其复用为低精度h_in缓存，保留量化往返边界并避免GM临时写回。




// 固定的R4/R5归约树将与双sibling AIV路径共用。half0固定覆盖展平区间[0,2D)，
// half1固定覆盖[2D,4D)，调用方只执行一次顶层合并partial0+partial1。









// mode1只在同一物理AI Core的两个sibling AIV之间握手。
// 调用点保持无条件控制流，保证每个lane、每个token各执行且只执行一次Set/Wait。
__aicore__ inline void MhcPairReadySync()
{
    CrossCoreSetFlag<0x1, PIPE_MTE3>(PAIR_READY_FLAG);
    CrossCoreWaitFlag<0x1, PIPE_MTE2>(PAIR_READY_FLAG);
}

#endif // MHC_NORM_POST_PREONLY_KERNEL_SUPPORT_H_
