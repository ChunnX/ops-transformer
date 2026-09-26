/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms and conditions of the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_MID_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_MID_H

#include "rms_norm_common.h"

namespace AiInfraMegaMhc::Compute {

/** Normalize rounded Post with its fused square sum: x2[k]=Cast<T>(FP32(post[k])*rsqrt(sum/K+eps)*gamma2[k]). */
template <typename T>
__aicore__ inline void ComputeRmsNormMidFromSquares(__ubuf__ T *postRow, __ubuf__ float *gamma2,
    uint32_t k, float epsilon, __ubuf__ float *sumScratch, __ubuf__ float *invRmsScratch,
    __ubuf__ T *x2)
{
    FinalizeInverseRms(sumScratch, k, epsilon, invRmsScratch);
    ApplyGammaAndCastRow<T>(postRow, gamma2, k, invRmsScratch, x2);
}

/**
 * What: compute RMSNorm_mid for a complete flattened four-head row.
 * Formula: x2[k] = Cast<T>(FP32(post[k]) * rsqrt(Σ_i FP32(post[i])²/(4D)+eps) * gamma2[k]).
 * Inputs: postRow and gamma2 each contain the full K=4D elements; scratch rows hold FP32 scalars.
 * Output: x2 is a complete T row. The Pre square sum is deliberately accumulated from this rounded x2.
 */
template <typename T>
__aicore__ inline void ComputeRmsNormMidRow(__ubuf__ T *postRow, __ubuf__ float *gamma2,
                                            uint32_t k, float epsilon, __ubuf__ float *sumScratch,
                                            __ubuf__ float *invRmsScratch, __ubuf__ T *x2)
{
    AccumulateRowSquares<T>(postRow, k, sumScratch);
    ComputeRmsNormMidFromSquares<T>(postRow, gamma2, k, epsilon, sumScratch, invRmsScratch, x2);
}

}  // namespace AiInfraMegaMhc::Compute

#endif  // OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_MID_H
