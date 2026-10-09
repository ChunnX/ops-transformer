/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_MID_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_MID_H

#include "rms_norm_common.h"

namespace MegaMhc::Compute {

/** Normalize rounded Post with its fused square sum: x2[k]=Cast<T>(FP32(post[k])*rsqrt(sum/K+eps)*gamma2[k]). */
template <typename T>
__aicore__ inline void ComputeRmsNormMidFromSquares(__ubuf__ T *postRow, __ubuf__ float *gamma2, uint32_t k,
                                                    float epsilon, __ubuf__ float *sumScratch,
                                                    __ubuf__ float *invRmsScratch, __ubuf__ T *x2)
{
    ApplyGammaAndCastRow<T, true>(postRow, gamma2, k, sumScratch, x2, epsilon);
}

/**
 * What: compute RMSNorm_mid for a complete flattened four-head row.
 * Formula: x2[k] = Cast<T>(FP32(post[k]) * rsqrt(Σ_i FP32(post[i])²/(4D)+eps) * gamma2[k]).
 * Inputs: postRow and gamma2 each contain the full K=4D elements; scratch rows hold FP32 scalars.
 * Output: x2 is a complete T row. The Pre square sum is deliberately accumulated from this rounded x2.
 */
template <typename T>
__aicore__ inline void ComputeRmsNormMidRow(__ubuf__ T *postRow, __ubuf__ float *gamma2, uint32_t k, float epsilon,
                                            __ubuf__ float *sumScratch, __ubuf__ float *invRmsScratch, __ubuf__ T *x2)
{
    AccumulateRowSquares<T>(postRow, k, sumScratch);
    ComputeRmsNormMidFromSquares<T>(postRow, gamma2, k, epsilon, sumScratch, invRmsScratch, x2);
}

} // namespace MegaMhc::Compute

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_MID_H
