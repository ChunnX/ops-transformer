/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms and conditions of the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_0_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_0_H

#include "rms_norm_common.h"

namespace AiInfraMegaMhc::Compute {

/**
 * What: compute the RMSNorm_0 statistic before any dependent Post work begins.
 * Formula: invRms = rsqrt(Σ_i FP32(hOut[i])² / D + epsilon).
 * Inputs: hOut is a complete D-element UB row; sumScratch and invRmsScratch each hold one FP32 scalar.
 * Output: invRmsScratch contains the statistic shared by the later RMS apply and Post pipeline.
 */
template <typename T>
__aicore__ inline void ComputeRmsNorm0Statistics(__ubuf__ T *hOut, uint32_t d, float epsilon,
                                                 __ubuf__ float *sumScratch,
                                                 __ubuf__ float *invRmsScratch)
{
    AccumulateRowSquares<T>(hOut, d, sumScratch);
    FinalizeInverseRms(sumScratch, d, epsilon, invRmsScratch);
}

/**
 * What: apply a completed RMSNorm_0 statistic and preserve its Cast<T> boundary.
 * Formula: x1[d] = Cast<T>(FP32(hOut[d]) * invRms * gamma0[d]).
 * Inputs: hOut and gamma0 are D-element UB rows; invRms is the precomputed FP32 scalar.
 * Output: x1 is a complete low-precision row consumed by MHC_Post.
 */
template <typename T>
__aicore__ inline void ApplyRmsNorm0Gamma(__ubuf__ T *hOut, __ubuf__ float *gamma0,
                                         uint32_t d, __ubuf__ float *invRms,
                                         __ubuf__ T *x1)
{
    ApplyGammaAndCastRow<T>(hOut, gamma0, d, invRms, x1);
}

/**
 * What: compute one RMSNorm_0 row and its low-precision x1 output.
 * Formula: x1[d] = Cast<T>(FP32(hOut[d]) * rsqrt(Σ_i FP32(hOut[i])²/D + eps) * gamma0[d]).
 * Inputs: hOut and gamma0 are complete D-element UB rows; scratch holds one FP32 scalar.
 * Output: x1 is a complete T row. The statistic is complete before ApplyGammaAndCastRow starts.
 */
template <typename T>
__aicore__ inline void ComputeRmsNorm0Row(__ubuf__ T *hOut, __ubuf__ float *gamma0,
                                          uint32_t d, float epsilon, __ubuf__ float *sumScratch,
                                          __ubuf__ float *invRmsScratch, __ubuf__ T *x1)
{
    ComputeRmsNorm0Statistics<T>(hOut, d, epsilon, sumScratch, invRmsScratch);
    ApplyRmsNorm0Gamma<T>(hOut, gamma0, d, invRmsScratch, x1);
}

}  // namespace AiInfraMegaMhc::Compute

#endif  // OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_0_H
