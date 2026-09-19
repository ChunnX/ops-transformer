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
constexpr uint32_t BUFFER_NUM_TWO = 2;
constexpr uint32_t BUFFER_NUM_ONE = 1;
constexpr uint32_t SCALAR_BUF_SIZE   = 256;    // 标量计算 UB buffer 大小 (bytes)
constexpr uint32_t EXP_DIV_ELEM      = 8;      // sigmoid ExpDiv 计算元素数
constexpr uint32_t POST_COMB_SIZE    = 8 + 16; // h_post[8] + h_res[16] buffer 大小

// 门控只有8个对齐元素，在一个寄存器函数内完成整条sigmoid链，避免高层短向量逐条同步。
__simd_vf__ inline void ComputeGateRegVf(__ubuf__ float *gateAddr, float hcEps)
{
    AscendC::Reg::RegTensor<float> inputReg;
    AscendC::Reg::RegTensor<float> expReg;
    AscendC::Reg::RegTensor<float> denomReg;
    AscendC::Reg::RegTensor<float> oneReg;
    AscendC::Reg::RegTensor<float> gateReg;
    uint32_t elementCount = EXP_DIV_ELEM;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(elementCount);

    AscendC::Reg::LoadAlign(inputReg, gateAddr);
    AscendC::Reg::Exp(expReg, inputReg, mask);
    AscendC::Reg::Adds(denomReg, expReg, 1.0f, mask);
    AscendC::Reg::Duplicate(oneReg, 1.0f);
    AscendC::Reg::Div(gateReg, oneReg, denomReg, mask);
    AscendC::Reg::Adds(gateReg, gateReg, hcEps, mask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(gateAddr, gateReg, mask);
}

// ---------------------------------------------------------------------------
// Type trait: fp32->half uses CAST_NONE, fp32->bfloat16_t uses CAST_RINT
// (per CANN RmsNorm reference)
// ---------------------------------------------------------------------------
template<typename A, typename B> struct IsSameT { static constexpr bool value = false; };
template<typename A> struct IsSameT<A, A> { static constexpr bool value = true; };

#endif // MHC_SANDWICH_NORM_POST_PREONLY_COMMON_H_
