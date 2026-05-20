/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file a2av_gmm_utils.h
 * \brief
 */
#ifndef A2AV_GMM_UTILS_H
#define A2AV_GMM_UTILS_H

namespace MC2KernelTemplate {
__aicore__ inline uint64_t Align(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

// MX_QUANT_MODE: 决定是否启用 MX quant scale 通信处理
// 正确逻辑：只有当 DTYPE_GMM_X_SCALE 是 MX 类型（如 fp8_e8m0）时才为 true
// 非量化算子不定义 DTYPE_GMM_X_SCALE，或 scaleType 为 float，此时应为 false
#if defined(DTYPE_GMM_X_SCALE) && (DTYPE_GMM_X_SCALE == DT_FLOAT8_E8M0)
#define MX_QUANT_MODE true
#else
#define MX_QUANT_MODE false
#endif

#if ((ORIG_DTYPE_GMM_X == ORIG_DTYPE_GMM_WEIGHT) && (ORIG_DTYPE_GMM_X == DT_FLOAT4_E2M1))
#define PACK_FACTOR 2U
#else
#define PACK_FACTOR 1U
#endif

}
#endif // A2AV_GMM_UTILS_H