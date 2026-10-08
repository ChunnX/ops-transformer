// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Adapted from DeepEP; original copyright and license notices follow.

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef MEGA_MOE_A8W4_K_PLAN_H
#define MEGA_MOE_A8W4_K_PLAN_H
#include <cstdint>

namespace MegaMoeImpl::GmmKernel {
// Shared by AIC and AIV0: B must use exactly the same K window on both cores.
struct A8W4KPlan {
    uint32_t ka;
    uint32_t kb;
};
__aicore__ constexpr inline A8W4KPlan MakeA8W4KPlan(uint32_t m, uint32_t n, bool allowDynamicB)
{
    const uint32_t kb = allowDynamicB && m <= 256U && n <= 128U ? 512U : 256U;
    const uint32_t alignedM = (m + 15U) / 16U * 16U;
    if (alignedM == 0U || m >= n) {
        return {kb, kb};
    }
    const uint32_t balancedDepth = (n + 2U * alignedM - 1U) / (2U * alignedM);
    const uint32_t capacityDepth = (128U * 1024U) / (alignedM * kb);
    const uint32_t depth = balancedDepth < capacityDepth ? balancedDepth : capacityDepth;
    return {depth * kb, kb};
}
} // namespace MegaMoeImpl::GmmKernel
#endif
