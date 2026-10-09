/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_COMMON_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_COMMON_H

#include "kernel_operator.h"

namespace MegaMhc::Compute {
using namespace AscendC;

constexpr uint32_t kRegFloatLanes = 64;
constexpr uint32_t kRegCastVectorLanes = 64;

/** Build the low count lanes without changing the caller's loop increment (UpdateMask decrements its argument). */
__simd_callee__ inline Reg::MaskReg MaskForCount(uint32_t count)
{
    return Reg::UpdateMask<float>(count);
}

constexpr Reg::CastTrait kCastInputToFloat = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::UNKNOWN};
constexpr Reg::CastTrait kCastFloatToInput = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::CAST_RINT};

/**
 * What: add the squared FP32 values of one aligned UB range to an FP32 scalar.
 * Formula: sumOut = (INITIALIZE ? 0 : sumOut) + Σ_i FP32(x[i])² over [0, rowLength).
 * Inputs: inputRow is input type T; rowLength is the valid element count; INITIALIZE selects first range at compile
 * time. Output: one FP32 scalar at sumOut. Tail lanes are masked and excluded. Precision: both input types use two
 * independent FP32 chains (even/odd 64-element blocks), then sum the chains. Formula: sumOut = sum_even(x*x) +
 * sum_odd(x*x); only input/output casts depend on T.
 */
template <typename T, bool INITIALIZE>
__aicore__ inline void AccumulateValuesSquares(__ubuf__ T *inputRow, uint32_t rowLength, __ubuf__ float *sumOut)
{
    const uint16_t repeatTimes =
        static_cast<uint16_t>((rowLength + Compute::kRegCastVectorLanes - 1U) / Compute::kRegCastVectorLanes);
    const uint16_t pairedRepeats = static_cast<uint16_t>(repeatTimes / 2U);
    const uint16_t tailStart = static_cast<uint16_t>(pairedRepeats * 2U);
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<T> input;
        Reg::RegTensor<float> value;
        Reg::RegTensor<float> square;
        Reg::RegTensor<float> partial;
        Reg::RegTensor<float> sum;
        Reg::MaskReg scalarMask = Compute::MaskForCount(1);
        Reg::MaskReg fullMask = Compute::MaskForCount(kRegFloatLanes);
        Reg::Duplicate(sum, 0.0f, fullMask);
        uint32_t remaining = rowLength;
        Reg::RegTensor<float> sumOdd;
        Reg::Duplicate(sumOdd, 0.0f, fullMask);
        for (uint16_t repeat = 0; repeat < pairedRepeats; ++repeat) {
            {
                const uint32_t offset = (repeat * 2U + 0U) * Compute::kRegCastVectorLanes;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, inputRow + offset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                // Cast zeroes inactive tail lanes, so all lane accumulators stay defined.
                Reg::Mul(square, value, value, fullMask);
                Reg::Add(sum, sum, square, fullMask);
            }
            {
                const uint32_t offset = (repeat * 2U + 1U) * Compute::kRegCastVectorLanes;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, inputRow + offset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                // Cast zeroes inactive tail lanes, so all lane accumulators stay defined.
                Reg::Mul(square, value, value, fullMask);
                Reg::Add(sumOdd, sumOdd, square, fullMask);
            }
        }
        for (uint16_t repeat = tailStart; repeat < repeatTimes; ++repeat) {
            const uint32_t offset = repeat * Compute::kRegCastVectorLanes;
            mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, inputRow + offset);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            // Cast zeroes inactive tail lanes, so all lane accumulators stay defined.
            Reg::Mul(square, value, value, fullMask);
            Reg::Add(sum, sum, square, fullMask);
        }
        Reg::Add(sum, sum, sumOdd, fullMask);

        Reg::Reduce<Reg::ReduceType::SUM>(partial, sum, fullMask);
        if constexpr (!INITIALIZE) {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(sum, sumOut);
            Reg::Add(partial, sum, partial, scalarMask);
        }
        Reg::Store<float>(sumOut, partial, 1U);
    }
}

/**
 * What: accumulate all squared values in a complete RMS row.
 * Formula: sumOut = Σ_i FP32(inputRow[i])² over [0, rowLength).
 * Inputs: inputRow is an aligned UB row in input type T; rowLength is the valid element count.
 * Output: one FP32 scalar at sumOut, initialized to zero before accumulation.
 */
template <typename T>
__aicore__ inline void AccumulateRowSquares(__ubuf__ T *inputRow, uint32_t rowLength, __ubuf__ float *sumOut)
{
    AccumulateValuesSquares<T, true>(inputRow, rowLength, sumOut);
}

/**
 * What: add one FP32 scalar partial to an ordered FP32 reduction accumulator.
 * Formula: accumulator = accumulator + partial.
 * Inputs: accumulator and partial point to one UB float each; caller fixes their reduction order.
 * Output: the updated scalar is stored at accumulator.
 */
__aicore__ inline void AddFp32Scalar(__ubuf__ float *accumulator, __ubuf__ float *partial)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> left;
        Reg::RegTensor<float> right;
        Reg::MaskReg mask = Compute::MaskForCount(1U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(left, accumulator);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(right, partial);
        Reg::Add(left, left, right, mask);
        Reg::Store<float>(accumulator, left, 1U);
    }
}

/**
 * What: finish RMS normalization statistics for one row.
 * Formula: invRms = rsqrt(sumSquares * meanScale + epsilon), meanScale = 1 / rowLength.
 * Refinement: y <- y + 0.5*y*(1 - x*y*y), using fused residual evaluation for x=meanSquare+epsilon.
 * Inputs: sumSquares is the FP32 square sum; meanScale is computed outside VF; epsilon is FP32.
 * Output: lane zero holds inverse RMS. The caller broadcasts it or stores it for chunk reuse.
 */
__simd_callee__ inline void InverseRmsRegister(__ubuf__ float *sumSquares, float meanScale, float epsilon,
                                               Reg::RegTensor<float> &rms)
{
    Reg::RegTensor<float> sum;
    Reg::RegTensor<float> scale;
    Reg::RegTensor<float> eps;
    Reg::RegTensor<float> one;
    Reg::RegTensor<float> mean;
    Reg::RegTensor<float> negative;
    Reg::RegTensor<float> product;
    Reg::RegTensor<float> error;
    Reg::RegTensor<float> halfInverse;
    Reg::MaskReg mask = Compute::MaskForCount(1);
    Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(sum, sumSquares);
    Reg::Duplicate(scale, meanScale, mask);
    Reg::Duplicate(eps, epsilon, mask);
    Reg::Duplicate(one, 1.0f, mask);
    Reg::Mul(rms, sum, scale, mask);
    Reg::Add(rms, rms, eps, mask);
    Reg::Adds(mean, rms, 0.0f, mask);
    Reg::Sqrt(rms, mean, mask);
    Reg::Div(rms, one, rms, mask);
    Reg::Muls(negative, rms, -1.0f, mask);
    Reg::Mul(product, rms, mean, mask);
    Reg::Duplicate(error, 1.0f, mask);
    Reg::MulAddDst(error, product, negative, mask);
    Reg::Muls(halfInverse, rms, 0.5f, mask);
    Reg::MulAddDst(rms, error, halfInverse, mask);
}

/** Finish one RMS statistic in UB: invRms=rsqrt(sumSquares/rowLength+epsilon), with Newton refinement. */
__aicore__ inline void FinalizeInverseRms(__ubuf__ float *sumSquares, uint32_t rowLength, float epsilon,
                                          __ubuf__ float *invRmsOut)
{
    const float meanScale = 1.0f / static_cast<float>(rowLength);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> inverse;
        InverseRmsRegister(sumSquares, meanScale, epsilon, inverse);
        Reg::Store<float>(invRmsOut, inverse, 1U);
    }
}

/**
 * What: apply inverse RMS and gamma to a row, then round to the input type.
 * Formula: y[i] = Cast<T>(FP32(x[i]) * invRms * gamma[i]).
 * Inputs: inputRow and gammaRow are UB views; valid range is [0, rowLength).
 * FROM_SQUARES consumes a square sum and computes invRms in registers; otherwise invRms is precomputed.
 * Output: outputRow is T. This is the explicit stage rounding point for RMSNorm.
 */
template <typename T, bool FROM_SQUARES = false>
__aicore__ inline void ApplyGammaAndCastRow(__ubuf__ T *inputRow, __ubuf__ float *gammaRow, uint32_t rowLength,
                                            __ubuf__ float *invRms, __ubuf__ T *outputRow, float epsilon = 0.0f)
{
    const float meanScale = 1.0f / static_cast<float>(rowLength);
    const uint16_t repeatTimes =
        static_cast<uint16_t>((rowLength + Compute::kRegCastVectorLanes - 1U) / Compute::kRegCastVectorLanes);
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<T> input;
        Reg::RegTensor<float> value;
        Reg::RegTensor<float> gamma;
        Reg::RegTensor<float> inverse;
        Reg::RegTensor<float> scaled;
        Reg::RegTensor<T> output;
        if constexpr (FROM_SQUARES) {
            Reg::RegTensor<float> scalarInverse;
            InverseRmsRegister(invRms, meanScale, epsilon, scalarInverse);
            Reg::MaskReg fullMask = MaskForCount(kRegFloatLanes);
            Reg::Duplicate(inverse, scalarInverse, fullMask);
        } else {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(inverse, invRms);
        }
        uint32_t remaining = rowLength;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            const uint32_t offset = repeat * Compute::kRegCastVectorLanes;
            mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, inputRow + offset);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            Reg::LoadAlign<float>(gamma, gammaRow + offset);
            Reg::Mul(scaled, value, inverse, mask);
            Reg::Mul(scaled, scaled, gamma, mask);
            Reg::Cast<T, float, kCastFloatToInput>(output, scaled, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputRow + offset, output, mask);
        }
    }
}

} // namespace MegaMhc::Compute

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_COMMON_H
