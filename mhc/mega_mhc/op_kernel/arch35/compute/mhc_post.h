/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_POST_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_POST_H

#include "kernel_operator.h"
#include "rms_norm_common.h"

namespace MegaMhc::Compute {
using namespace AscendC;

/** Add one source head to four output heads: sum[n] += FP32(source[d]) * weight[n].
 * Keep four independent Mul/Add chains so the Post stage retains its original rounding order.
 */
template <typename T>
__simd_callee__ inline void AccumulatePostSource(__ubuf__ T *source, Reg::RegTensor<float> &w0,
                                                 Reg::RegTensor<float> &w1, Reg::RegTensor<float> &w2,
                                                 Reg::RegTensor<float> &w3, Reg::RegTensor<float> &sum0,
                                                 Reg::RegTensor<float> &sum1, Reg::RegTensor<float> &sum2,
                                                 Reg::RegTensor<float> &sum3, Reg::MaskReg &mask)
{
    Reg::RegTensor<T> input;
    Reg::RegTensor<float> value;
    Reg::RegTensor<float> product;
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, source);
    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
    Reg::Mul(product, value, w0, mask);
    Reg::Add(sum0, sum0, product, mask);
    Reg::Mul(product, value, w1, mask);
    Reg::Add(sum1, sum1, product, mask);
    Reg::Mul(product, value, w2, mask);
    Reg::Add(sum2, sum2, product, mask);
    Reg::Mul(product, value, w3, mask);
    Reg::Add(sum3, sum3, product, mask);
}

/** Store one rounded Post head: output[d] = Cast<T>(sum[d]) for the active lanes. */
template <typename T>
__simd_callee__ inline void StorePostHead(__ubuf__ T *output, Reg::RegTensor<float> &sum, Reg::MaskReg &mask)
{
    Reg::RegTensor<T> result;
    Reg::Cast<T, float, kCastFloatToInput>(result, sum, mask);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output, result, mask);
}

/** Store Cast<T>(sum) and reuse its consumed register for FP32(Cast<T>(sum)), the RMSMid input. */
template <typename T>
__simd_callee__ inline void StorePostHeadRounded(__ubuf__ T *output, Reg::RegTensor<float> &sum, Reg::MaskReg &mask)
{
    Reg::RegTensor<T> result;
    Reg::Cast<T, float, kCastFloatToInput>(result, sum, mask);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output, result, mask);
    Reg::Cast<float, T, kCastInputToFloat>(sum, result, mask);
}

/** Compute all four heads together: load/cast x1 and each residual slice once per 64 columns.
 * Formula: output[row,n,d] = Cast<T>(hPost[row,n]*x1[row,d] + sum_j hRes[row,j,n]*residual[row,j,d]).
 * Weights are broadcast once per row; source heads accumulate strictly in order 0,1,2,3.
 * RMSMid first forms the FP32 pairs x0*x0+x1*x1 and x2*x2+x3*x3 using FMA, then accumulates pairs.
 * Consumed Post sum registers hold each pair; one persistent square-sum register is sufficient.
 * FP16/BF16 share this order and retain Cast<T> before squaring.
 */
template <typename T, bool ACCUMULATE_SQUARES = false>
__aicore__ inline void ComputePostFourHeads(__ubuf__ T *x1, __ubuf__ T *residual, __ubuf__ float *hPost,
                                            __ubuf__ float *hRes, __ubuf__ T *output, uint32_t rows, uint32_t d,
                                            uint32_t dStart, uint32_t dCount, __ubuf__ float *sumSquares = nullptr)
{
    const uint16_t rowCount = static_cast<uint16_t>(rows);
    const uint16_t repeatTimes = static_cast<uint16_t>((dCount + 63U) / 64U);
    __VEC_SCOPE__
    {
        Reg::MaskReg fullMask = Compute::MaskForCount(kRegFloatLanes);
        Reg::RegTensor<float> squares;
        Reg::RegTensor<float> post0;
        Reg::RegTensor<float> post1;
        Reg::RegTensor<float> post2;
        Reg::RegTensor<float> post3;
        Reg::RegTensor<float> w00;
        Reg::RegTensor<float> w01;
        Reg::RegTensor<float> w02;
        Reg::RegTensor<float> w03;
        Reg::RegTensor<float> w10;
        Reg::RegTensor<float> w11;
        Reg::RegTensor<float> w12;
        Reg::RegTensor<float> w13;
        Reg::RegTensor<float> w20;
        Reg::RegTensor<float> w21;
        Reg::RegTensor<float> w22;
        Reg::RegTensor<float> w23;
        Reg::RegTensor<float> w30;
        Reg::RegTensor<float> w31;
        Reg::RegTensor<float> w32;
        Reg::RegTensor<float> w33;
        Reg::RegTensor<T> input;
        Reg::RegTensor<float> value;
        Reg::RegTensor<float> sum0;
        Reg::RegTensor<float> sum1;
        Reg::RegTensor<float> sum2;
        Reg::RegTensor<float> sum3;
        Reg::RegTensor<float> reduced;
        Reg::MaskReg mask;
        for (uint16_t row = 0; row < rowCount; ++row) {
            auto rowPost = hPost + row * 4U;
            auto rowRes = hRes + row * 16U;
            auto rowX1 = x1 + row * d + dStart;
            const uint32_t rowOffset = row * 4U * d + dStart;
            auto residual0 = residual + rowOffset;
            auto residual1 = residual0 + d;
            auto residual2 = residual1 + d;
            auto residual3 = residual2 + d;
            auto output0 = output + rowOffset;
            auto output1 = output0 + d;
            auto output2 = output1 + d;
            auto output3 = output2 + d;
            if constexpr (ACCUMULATE_SQUARES) {
                Reg::Duplicate(squares, 0.0f, fullMask);
            }
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post0, rowPost + 0U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post1, rowPost + 1U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post2, rowPost + 2U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post3, rowPost + 3U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w00, rowRes + 0U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w01, rowRes + 1U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w02, rowRes + 2U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w03, rowRes + 3U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w10, rowRes + 4U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w11, rowRes + 5U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w12, rowRes + 6U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w13, rowRes + 7U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w20, rowRes + 8U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w21, rowRes + 9U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w22, rowRes + 10U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w23, rowRes + 11U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w30, rowRes + 12U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w31, rowRes + 13U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w32, rowRes + 14U);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w33, rowRes + 15U);
            uint32_t remaining = dCount;
            for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
                const uint32_t column = repeat * kRegCastVectorLanes;
                mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, rowX1 + column);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::Mul(sum0, value, post0, mask);
                Reg::Mul(sum1, value, post1, mask);
                Reg::Mul(sum2, value, post2, mask);
                Reg::Mul(sum3, value, post3, mask);
                AccumulatePostSource<T>(residual0 + column, w00, w01, w02, w03, sum0, sum1, sum2, sum3, mask);
                AccumulatePostSource<T>(residual1 + column, w10, w11, w12, w13, sum0, sum1, sum2, sum3, mask);
                AccumulatePostSource<T>(residual2 + column, w20, w21, w22, w23, sum0, sum1, sum2, sum3, mask);
                AccumulatePostSource<T>(residual3 + column, w30, w31, w32, w33, sum0, sum1, sum2, sum3, mask);
                if constexpr (ACCUMULATE_SQUARES) {
                    StorePostHeadRounded<T>(output0 + column, sum0, mask);
                    Reg::Mul(sum0, sum0, sum0, fullMask);
                    StorePostHeadRounded<T>(output1 + column, sum1, mask);
                    Reg::MulAddDst(sum0, sum1, sum1, fullMask);
                    Reg::Add(squares, squares, sum0, fullMask);
                    StorePostHeadRounded<T>(output2 + column, sum2, mask);
                    Reg::Mul(sum2, sum2, sum2, fullMask);
                    StorePostHeadRounded<T>(output3 + column, sum3, mask);
                    Reg::MulAddDst(sum2, sum3, sum3, fullMask);
                    Reg::Add(squares, squares, sum2, fullMask);
                } else {
                    StorePostHead<T>(output0 + column, sum0, mask);
                    StorePostHead<T>(output1 + column, sum1, mask);
                    StorePostHead<T>(output2 + column, sum2, mask);
                    StorePostHead<T>(output3 + column, sum3, mask);
                }
            }
            if constexpr (ACCUMULATE_SQUARES) {
                Reg::Reduce<Reg::ReduceType::SUM>(reduced, squares, fullMask);
                Reg::Store<float>(sumSquares + row, reduced, 1U);
            }
        }
    }
}

/**
 * What: compute a tile of the four-head MHC_Post output with RegBase arithmetic.
 * Formula: z[t,n,d] = Cast<T>(h_post[t,n] * FP32(x1[t,d])
 *                    + Σ_j h_res[t,j,n] * FP32(residual[t,j,d])).
 * Inputs: x1 is [rows,D]; residual is [rows,4,D]; hPost is [rows,4]; hRes is [rows,4,4].
 * Output: output is [rows,4,D]; only heads [headStart,headStart+headCount) and
 *         columns [dStart,dStart+dCount) are written. Every output head reads all four input heads.
 * Precision: multiply/accumulate in FP32; Cast<T> is the Post-to-next-stage rounding point.
 * Ownership: caller owns buffers, block mapping, and publication to x2 output/Cube staging.
 */
template <typename T>
__aicore__ inline void ComputePostTile(__ubuf__ T *x1, __ubuf__ T *residual, __ubuf__ float *hPost,
                                       __ubuf__ float *hRes, __ubuf__ T *output, uint32_t rows, uint32_t d,
                                       uint32_t headStart, uint32_t headCount, uint32_t dStart, uint32_t dCount)
{
    if (headStart == 0U && headCount == 4U) {
        ComputePostFourHeads<T>(x1, residual, hPost, hRes, output, rows, d, dStart, dCount);
        return;
    }
    const uint16_t rowCount = static_cast<uint16_t>(rows);
    const uint16_t outputHeads = static_cast<uint16_t>(headCount);
    const uint16_t repeatTimes = static_cast<uint16_t>((dCount + 63U) / 64U);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> postWeight;
        Reg::RegTensor<float> weight0;
        Reg::RegTensor<float> weight1;
        Reg::RegTensor<float> weight2;
        Reg::RegTensor<float> weight3;
        Reg::RegTensor<T> input;
        Reg::RegTensor<float> value;
        Reg::RegTensor<float> product;
        Reg::RegTensor<float> sum;
        Reg::MaskReg mask;
        for (uint16_t row = 0; row < rowCount; ++row) {
            auto rowPost = hPost + row * 4U;
            auto rowRes = hRes + row * 16U;
            auto rowX1 = x1 + row * d + dStart;
            const uint32_t rowOffset = row * 4U * d + dStart;
            auto residual0 = residual + rowOffset;
            auto residual1 = residual0 + d;
            auto residual2 = residual1 + d;
            auto residual3 = residual2 + d;
            auto output0 = output + rowOffset;
            for (uint16_t headIndex = 0; headIndex < outputHeads; ++headIndex) {
                const uint32_t head = headStart + headIndex;
                auto outputHead = output0 + head * d;
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(postWeight, rowPost + head);
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight0, rowRes + 0U + head);
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight1, rowRes + 4U + head);
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight2, rowRes + 8U + head);
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight3, rowRes + 12U + head);
                uint32_t remaining = dCount;
                for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
                    const uint32_t column = repeat * kRegCastVectorLanes;
                    mask = Reg::UpdateMask<float>(remaining);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, rowX1 + column);
                    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                    Reg::Mul(sum, value, postWeight, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, residual0 + column);
                    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                    Reg::Mul(product, value, weight0, mask);
                    Reg::Add(sum, sum, product, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, residual1 + column);
                    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                    Reg::Mul(product, value, weight1, mask);
                    Reg::Add(sum, sum, product, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, residual2 + column);
                    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                    Reg::Mul(product, value, weight2, mask);
                    Reg::Add(sum, sum, product, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, residual3 + column);
                    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                    Reg::Mul(product, value, weight3, mask);
                    Reg::Add(sum, sum, product, mask);
                    StorePostHead<T>(outputHead + column, sum, mask);
                }
            }
        }
    }
}

} // namespace MegaMhc::Compute

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_POST_H
