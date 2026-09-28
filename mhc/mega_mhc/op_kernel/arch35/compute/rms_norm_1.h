/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_1_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_1_H

#include "rms_norm_common.h"

namespace MegaMhc::Compute {

/** Normalize rounded h_in using its fused square sum: y[d]=Cast<T>(FP32(hIn[d])*rsqrt(sum/D+eps)*gamma1[d]). */
template <typename T>
__aicore__ inline void ComputeRmsNorm1FromSquares(__ubuf__ T *hIn, __ubuf__ float *gamma1, uint32_t d, float epsilon,
                                                  __ubuf__ float *sumScratch, __ubuf__ float *invRmsScratch,
                                                  __ubuf__ T *hInPrime)
{
    ApplyGammaAndCastRow<T, true>(hIn, gamma1, d, sumScratch, hInPrime, epsilon);
}

/**
 * What: normalize the already rounded h_in row and produce the final output.
 * Formula: hInPrime[d] = Cast<T>(FP32(hIn[d]) * rsqrt(Σ_i FP32(hIn[i])²/D+eps) * gamma1[d]).
 * Inputs: hIn is the prior Cast<T> result, not the FP32 head accumulator; gamma1 is a D-element row.
 * Output: hInPrime is a complete T row. This preserves the required h_in rounding point before RMS1.
 */
template <typename T>
__aicore__ inline void ComputeRmsNorm1Row(__ubuf__ T *hIn, __ubuf__ float *gamma1, uint32_t d, float epsilon,
                                          __ubuf__ float *sumScratch, __ubuf__ float *invRmsScratch,
                                          __ubuf__ T *hInPrime)
{
    AccumulateRowSquares<T>(hIn, d, sumScratch);
    ComputeRmsNorm1FromSquares<T>(hIn, gamma1, d, epsilon, sumScratch, invRmsScratch, hInPrime);
}

} // namespace MegaMhc::Compute

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_1_H
