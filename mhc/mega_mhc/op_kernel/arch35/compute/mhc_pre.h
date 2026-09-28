/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_PRE_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_PRE_H

#include "rms_norm_common.h"

namespace MegaMhc::Compute {
/** Convert an entire AIV tile and accumulate Pre statistics in the same VF.
 * Formula: stage[r,c] = FP32(x2[r,c]); squares[r] += sum_c stage[r,c]^2.
 * Input is packed by valid K; stage uses the host-planned L1 row stride.
 * Paired rows share the column loop and mask while retaining independent, ordered reductions.
 * Each lane adds at most
 * 16 squares for host K<=1024, then reduces in fixed order. COMPENSATE retains the compensated variant for
 * numerical/performance comparisons.
 */
template <typename T, bool COMPENSATE = false>
__aicore__ inline void PrepareProjectionTile(__ubuf__ T *input, uint32_t rows, uint32_t kCount, uint32_t stageStride,
                                             __ubuf__ float *stage, __ubuf__ float *squares)
{
    const uint16_t rowCount = static_cast<uint16_t>(rows);
    const uint16_t repeats = static_cast<uint16_t>((kCount + 63U) / 64U);
    const uint16_t pairedRows = rowCount / 2U;
    const uint16_t tailRow = pairedRows * 2U;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> sum;
        Reg::RegTensor<float> compensation;
        Reg::RegTensor<float> value;
        Reg::RegTensor<float> square;
        Reg::RegTensor<float> corrected;
        Reg::RegTensor<float> next;
        Reg::RegTensor<float> difference;
        Reg::RegTensor<float> partial;
        Reg::RegTensor<float> previous;
        Reg::RegTensor<T> raw;
        Reg::MaskReg mask;
        Reg::MaskReg fullMask = MaskForCount(64U);
        Reg::MaskReg scalarMask = MaskForCount(1U);
        Reg::RegTensor<float> sumNext;
        Reg::RegTensor<float> compensationNext;
        Reg::RegTensor<float> valueNext;
        Reg::RegTensor<float> squareNext;
        Reg::RegTensor<float> correctedNext;
        Reg::RegTensor<float> updatedNext;
        Reg::RegTensor<float> differenceNext;
        Reg::RegTensor<float> partialNext;
        Reg::RegTensor<float> previousNext;
        Reg::RegTensor<T> rawNext;
        // Interleave two rows while preserving each row's ordered FP32 reduction.
        for (uint16_t pair = 0; pair < pairedRows; ++pair) {
            const uint32_t row = pair * 2U;
            Reg::Duplicate(sum, 0.0f, fullMask);
            Reg::Duplicate(compensation, 0.0f, fullMask);
            auto inputRow = input + row * kCount;
            auto stageRow = stage + row * stageStride;
            uint32_t remaining = kCount;
            Reg::Duplicate(sumNext, 0.0f, fullMask);
            Reg::Duplicate(compensationNext, 0.0f, fullMask);
            auto inputRowNext = input + (row + 1U) * kCount;
            auto stageRowNext = stage + (row + 1U) * stageStride;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t column = repeat * 64U;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(raw, inputRow + column);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(rawNext, inputRowNext + column);
                Reg::Cast<float, T, kCastInputToFloat>(value, raw, mask);
                Reg::StoreAlign<float>(stageRow + column, value, mask);
                Reg::Mul(square, value, value, fullMask);
                if constexpr (COMPENSATE) {
                    Reg::Sub(corrected, square, compensation, fullMask);
                    Reg::Add(next, sum, corrected, fullMask);
                    Reg::Sub(difference, next, sum, fullMask);
                    Reg::Sub(compensation, difference, corrected, fullMask);
                    Reg::Adds(sum, next, 0.0f, fullMask);
                } else {
                    Reg::Add(sum, sum, square, fullMask);
                }
                Reg::Cast<float, T, kCastInputToFloat>(valueNext, rawNext, mask);
                Reg::StoreAlign<float>(stageRowNext + column, valueNext, mask);
                Reg::Mul(squareNext, valueNext, valueNext, fullMask);
                if constexpr (COMPENSATE) {
                    Reg::Sub(correctedNext, squareNext, compensationNext, fullMask);
                    Reg::Add(updatedNext, sumNext, correctedNext, fullMask);
                    Reg::Sub(differenceNext, updatedNext, sumNext, fullMask);
                    Reg::Sub(compensationNext, differenceNext, correctedNext, fullMask);
                    Reg::Adds(sumNext, updatedNext, 0.0f, fullMask);
                } else {
                    Reg::Add(sumNext, sumNext, squareNext, fullMask);
                }
            }
            Reg::Reduce<Reg::ReduceType::SUM>(partial, sum, fullMask);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(previous, squares + row);
            Reg::Add(partial, previous, partial, scalarMask);
            Reg::Store<float>(squares + row, partial, 1U);
            Reg::Reduce<Reg::ReduceType::SUM>(partialNext, sumNext, fullMask);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(previousNext, squares + row + 1U);
            Reg::Add(partialNext, previousNext, partialNext, scalarMask);
            Reg::Store<float>(squares + row + 1U, partialNext, 1U);
        }
        for (uint16_t row = tailRow; row < rowCount; ++row) {
            Reg::Duplicate(sum, 0.0f, fullMask);
            Reg::Duplicate(compensation, 0.0f, fullMask);
            auto inputRow = input + row * kCount;
            auto stageRow = stage + row * stageStride;
            uint32_t remaining = kCount;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t column = repeat * 64U;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(raw, inputRow + column);
                Reg::Cast<float, T, kCastInputToFloat>(value, raw, mask);
                Reg::StoreAlign<float>(stageRow + column, value, mask);
                Reg::Mul(square, value, value, fullMask);
                if constexpr (COMPENSATE) {
                    Reg::Sub(corrected, square, compensation, fullMask);
                    Reg::Add(next, sum, corrected, fullMask);
                    Reg::Sub(difference, next, sum, fullMask);
                    Reg::Sub(compensation, difference, corrected, fullMask);
                    Reg::Adds(sum, next, 0.0f, fullMask);
                } else {
                    Reg::Add(sum, sum, square, fullMask);
                }
            }
            Reg::Reduce<Reg::ReduceType::SUM>(partial, sum, fullMask);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(previous, squares + row);
            Reg::Add(partial, previous, partial, scalarMask);
            Reg::Store<float>(squares + row, partial, 1U);
        }
    }
}

using namespace AscendC;

/** Normalize a row batch in place in group zero.
 * Formula: z[r,q] = ordered_sum_g(P[g,r,q]) * rsqrt(ordered_sum_k(S[k,r])/K + eps).
 * RMS refinement and partial addition order match the scalar-row path exactly.
 */
__aicore__ inline void NormalizeProjectionRows(__ubuf__ float *partial, __ubuf__ float *rms, uint32_t rows,
                                               uint32_t rowStride, uint32_t groupStride, uint32_t groups,
                                               uint32_t rmsStride, uint32_t kGrid, uint32_t k, float eps)
{
    const uint16_t rowCount = static_cast<uint16_t>(rows);
    const uint16_t groupCount = static_cast<uint16_t>(groups);
    const uint16_t shardCount = static_cast<uint16_t>(kGrid);
    const float meanScale = 1.0f / static_cast<float>(k);
    __VEC_SCOPE__
    {
        Reg::MaskReg mask = MaskForCount(24U);
        Reg::RegTensor<float> sum, value, squares, mean, inverse, one, negative, product, error, half;
        Reg::Duplicate(one, 1.0f, mask);
        for (uint16_t row = 0; row < rowCount; ++row) {
            auto partialRow = partial + row * rowStride;
            auto rmsRow = rms + row;
            Reg::Duplicate(sum, 0.0f, mask);
            for (uint16_t group = 0; group < groupCount; ++group) {
                Reg::LoadAlign(value, partialRow + group * groupStride);
                Reg::Add(sum, sum, value, mask);
            }
            Reg::Duplicate(squares, 0.0f, mask);
            for (uint16_t shard = 0; shard < shardCount; ++shard) {
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(value, rmsRow + shard * rmsStride);
                Reg::Add(squares, squares, value, mask);
            }
            Reg::Muls(mean, squares, meanScale, mask);
            Reg::Adds(mean, mean, eps, mask);
            Reg::Sqrt(inverse, mean, mask);
            Reg::Div(inverse, one, inverse, mask);
            Reg::Muls(negative, inverse, -1.0f, mask);
            Reg::Mul(product, inverse, mean, mask);
            Reg::Duplicate(error, 1.0f, mask);
            Reg::MulAddDst(error, product, negative, mask);
            Reg::Muls(half, inverse, 0.5f, mask);
            Reg::MulAddDst(inverse, error, half, mask);
            Reg::Mul(sum, sum, inverse, mask);
            Reg::StoreAlign(partialRow, sum, mask);
        }
    }
}

/**
 * What: compute the four Pre branch gates after projection RMS scaling.
 * Formula: hPre[n] = sigmoid(alpha0*normalized[n]+bias[n]) + hcEps.
 * Inputs: normalizedProjection[0:4], alpha[0], bias[0:4].
 * Output: four FP32 hPre coefficients used by the later weighted input mix.
 */
__simd_callee__ inline void ComputePreGates(__ubuf__ float *normalizedProjection, __ubuf__ float *alpha,
                                            __ubuf__ float *bias, float hcEps, __ubuf__ float *hPre)
{
    Reg::RegTensor<float> projected;
    Reg::RegTensor<float> scale;
    Reg::RegTensor<float> offset;
    Reg::RegTensor<float> logit;
    Reg::RegTensor<float> negative;
    Reg::RegTensor<float> expValue;
    Reg::RegTensor<float> one;
    Reg::RegTensor<float> gate;
    Reg::RegTensor<float> epsilon;
    Reg::MaskReg mask = Compute::MaskForCount(4);
    Reg::Duplicate(one, 1.0f, mask);
    Reg::Duplicate(epsilon, hcEps, mask);
    Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(scale, alpha);
    Reg::RegTensor<uint32_t> indices;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(indices), 0);
    Reg::Gather(projected, normalizedProjection + 0U, indices, mask);
    Reg::Gather(offset, bias + 0U, indices, mask);
    Reg::Mul(logit, projected, scale, mask);
    Reg::Add(logit, logit, offset, mask);
    Reg::Muls(negative, logit, -1.0f, mask);
    Reg::Exp(expValue, negative, mask);
    Reg::Add(expValue, expValue, one, mask);
    Reg::Div(gate, one, expValue, mask);
    Reg::Add(gate, gate, epsilon, mask);
    Reg::Store<float>(hPre, gate, 4U);
}

/**
 * What: compute the four post coefficients after projection RMS scaling.
 * Formula: hPostNew[n] = 2 * sigmoid(alpha1*normalized[n+4]+bias[n+4]).
 * Inputs: normalizedProjection[4:8], alpha[1], bias[4:8].
 * Output: four FP32 hPostNew coefficients.
 */
__simd_callee__ inline void ComputePostGates(__ubuf__ float *normalizedProjection, __ubuf__ float *alpha,
                                             __ubuf__ float *bias, __ubuf__ float *hPostNew)
{
    Reg::RegTensor<float> projected;
    Reg::RegTensor<float> scale;
    Reg::RegTensor<float> offset;
    Reg::RegTensor<float> logit;
    Reg::RegTensor<float> negative;
    Reg::RegTensor<float> expValue;
    Reg::RegTensor<float> one;
    Reg::RegTensor<float> gate;
    Reg::MaskReg mask = Compute::MaskForCount(4);
    Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(scale, alpha + 1);
    Reg::Duplicate(one, 1.0f, mask);
    Reg::RegTensor<uint32_t> indices;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(indices), 0);
    Reg::Gather(projected, normalizedProjection + 4U, indices, mask);
    Reg::Gather(offset, bias + 4U, indices, mask);
    Reg::Mul(logit, projected, scale, mask);
    Reg::Add(logit, logit, offset, mask);
    Reg::Muls(negative, logit, -1.0f, mask);
    Reg::Exp(expValue, negative, mask);
    Reg::Add(expValue, expValue, one, mask);
    Reg::Div(gate, one, expValue, mask);
    Reg::Muls(gate, gate, 2.0f, mask);
    Reg::Store<float>(hPostNew, gate, 4U);
}

/**
 * What: compute the 4x4 residual coefficients after projection RMS scaling.
 * Formula: hResNew[i,j] = alpha2*normalized[8+i*4+j]+bias[8+i*4+j].
 * Inputs: normalizedProjection[8:24], alpha[2], bias[8:24].
 * Output: sixteen FP32 values in row-major [4,4] order; Sinkhorn is not applied.
 */
__simd_callee__ inline void ComputeResidualGates(__ubuf__ float *normalizedProjection, __ubuf__ float *alpha,
                                                 __ubuf__ float *bias, __ubuf__ float *hResNew)
{
    Reg::RegTensor<float> projected;
    Reg::RegTensor<float> scale;
    Reg::RegTensor<float> offset;
    Reg::RegTensor<float> result;
    Reg::MaskReg mask = Compute::MaskForCount(16);
    Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(scale, alpha + 2);
    Reg::RegTensor<uint32_t> indices;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(indices), 0);
    Reg::Gather(projected, normalizedProjection + 8U, indices, mask);
    Reg::Gather(offset, bias + 8U, indices, mask);
    Reg::Mul(result, projected, scale, mask);
    Reg::Add(result, result, offset, mask);
    Reg::Store<float>(hResNew, result, 16U);
}

/** Compute all row gates in one VF, with packed planes for two batched output DMAs.
 * Formula: pre=sigmoid(a0*z[0:4]+b)+eps; post=2*sigmoid(a1*z[4:8]+b); res=a2*z[8:24]+b.
 * Output planes are [pre:4*alignedRows, post:4*alignedRows, res:16*alignedRows].
 */
__aicore__ inline void ComputeGateRows(__ubuf__ float *normalized, __ubuf__ float *alpha, __ubuf__ float *bias,
                                       float eps, uint32_t rows, uint32_t rowStride, uint32_t alignedRows,
                                       __ubuf__ float *coefficients)
{
    const uint16_t rowCount = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        for (uint16_t row = 0; row < rowCount; ++row) {
            auto normalizedRow = normalized + row * rowStride;
            auto preRow = coefficients + row * 4U;
            ComputePreGates(normalizedRow, alpha, bias, eps, preRow);
            ComputePostGates(normalizedRow, alpha, bias, preRow + alignedRows * 4U);
            ComputeResidualGates(normalizedRow, alpha, bias, coefficients + alignedRows * 8U + row * 16U);
        }
    }
}

/**
 * What: mix the four rounded x2 heads into one h_in D tile.
 * Formula: hIn[d] = Cast<T>(Σ_n hPre[n] * FP32(x2[n,d])).
 * Inputs: hPre is FP32[4]; x2 is low-precision [4,D]; only [dStart,dStart+dCount) is read.
 * Both input types sum rounded hIn squares in independent even/odd 64-column FP32 chains.
 * Output: hIn is T[dCount]; sumSquares = Σ_d FP32(hIn[d])² after the Cast<T> rounding point.
 */
template <typename T>
__aicore__ inline void ComputeHInTile(__ubuf__ float *hPre, __ubuf__ T *x2, uint32_t d, uint32_t dStart,
                                      uint32_t dCount, __ubuf__ T *hIn, __ubuf__ float *sumSquares)
{
    const uint16_t repeatTimes = static_cast<uint16_t>((dCount + kRegCastVectorLanes - 1U) / kRegCastVectorLanes);
    const uint16_t pairedRepeats = static_cast<uint16_t>(repeatTimes / 2U);
    const uint16_t tailStart = static_cast<uint16_t>(pairedRepeats * 2U);
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<T> input;
        Reg::RegTensor<float> value;
        Reg::RegTensor<float> weight0, weight1, weight2, weight3;
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight0, hPre);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight1, hPre + 1);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight2, hPre + 2);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight3, hPre + 3);
        Reg::RegTensor<float> product;
        Reg::RegTensor<float> sum;
        Reg::RegTensor<T> result;
        Reg::RegTensor<float> squareSum, square;
        Reg::MaskReg fullMask = Compute::MaskForCount(kRegFloatLanes);
        Reg::Duplicate(squareSum, 0.0f, fullMask);
        uint32_t remaining = dCount;
        Reg::RegTensor<float> oddSquares;
        Reg::Duplicate(oddSquares, 0.0f, fullMask);
        auto x2Head0 = x2 + 0U * d + dStart;
        auto x2Head1 = x2 + 1U * d + dStart;
        auto x2Head2 = x2 + 2U * d + dStart;
        auto x2Head3 = x2 + 3U * d + dStart;
        for (uint16_t repeat = 0; repeat < pairedRepeats; ++repeat) {
            {
                const uint32_t tileOffset = (repeat * 2U + 0U) * kRegCastVectorLanes;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head0 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(sum, value, weight0, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head1 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(product, value, weight1, mask);
                Reg::Add(sum, sum, product, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head2 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(product, value, weight2, mask);
                Reg::Add(sum, sum, product, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head3 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(product, value, weight3, mask);
                Reg::Add(sum, sum, product, mask);
                Reg::Cast<T, float, kCastFloatToInput>(result, sum, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(hIn + tileOffset, result, mask);
                Reg::Cast<float, T, kCastInputToFloat>(value, result, mask);
                Reg::Mul(square, value, value, fullMask);
                Reg::Add(squareSum, squareSum, square, fullMask);
            }
            {
                const uint32_t tileOffset = (repeat * 2U + 1U) * kRegCastVectorLanes;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head0 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(sum, value, weight0, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head1 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(product, value, weight1, mask);
                Reg::Add(sum, sum, product, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head2 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(product, value, weight2, mask);
                Reg::Add(sum, sum, product, mask);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head3 + tileOffset);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(product, value, weight3, mask);
                Reg::Add(sum, sum, product, mask);
                Reg::Cast<T, float, kCastFloatToInput>(result, sum, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(hIn + tileOffset, result, mask);
                Reg::Cast<float, T, kCastInputToFloat>(value, result, mask);
                Reg::Mul(square, value, value, fullMask);
                Reg::Add(oddSquares, oddSquares, square, fullMask);
            }
        }
        for (uint16_t repeat = tailStart; repeat < repeatTimes; ++repeat) {
            const uint32_t tileOffset = repeat * kRegCastVectorLanes;
            mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head0 + tileOffset);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            Reg::Mul(sum, value, weight0, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head1 + tileOffset);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            Reg::Mul(product, value, weight1, mask);
            Reg::Add(sum, sum, product, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head2 + tileOffset);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            Reg::Mul(product, value, weight2, mask);
            Reg::Add(sum, sum, product, mask);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x2Head3 + tileOffset);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            Reg::Mul(product, value, weight3, mask);
            Reg::Add(sum, sum, product, mask);
            Reg::Cast<T, float, kCastFloatToInput>(result, sum, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(hIn + tileOffset, result, mask);
            Reg::Cast<float, T, kCastInputToFloat>(value, result, mask);
            Reg::Mul(square, value, value, fullMask);
            Reg::Add(squareSum, squareSum, square, fullMask);
        }
        Reg::Add(squareSum, squareSum, oddSquares, fullMask);

        Reg::Reduce<Reg::ReduceType::SUM>(square, squareSum, fullMask);
        Reg::Store<float>(sumSquares, square, 1U);
    }
}

} // namespace MegaMhc::Compute

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_PRE_H
