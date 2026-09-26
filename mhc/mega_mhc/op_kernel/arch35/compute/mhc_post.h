/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms and conditions of the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_POST_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_POST_H

#include "kernel_operator.h"
#include "rms_norm_common.h"

namespace AiInfraMegaMhc::Compute {
using namespace AscendC;

/** Add one source head to four output heads: sum[n] += FP32(source[d]) * weight[n].
 * Keep four independent Mul/Add chains so the Post stage retains its original rounding order.
 */
template <typename T>
__simd_callee__ inline void AccumulatePostSource(__ubuf__ T *source,
    Reg::RegTensor<float> &w0, Reg::RegTensor<float> &w1,
    Reg::RegTensor<float> &w2, Reg::RegTensor<float> &w3,
    Reg::RegTensor<float> &sum0, Reg::RegTensor<float> &sum1,
    Reg::RegTensor<float> &sum2, Reg::RegTensor<float> &sum3, Reg::MaskReg &mask)
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
__simd_callee__ inline void StorePostHead(__ubuf__ T *output,
    Reg::RegTensor<float> &sum, Reg::MaskReg &mask)
{
    Reg::RegTensor<T> result;
    Reg::Cast<T, float, kCastFloatToInput>(result, sum, mask);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output, result, mask);
}

/** Store Cast<T>(sum) and reuse its consumed register for FP32(Cast<T>(sum)), the RMSMid input. */
template <typename T>
__simd_callee__ inline void StorePostHeadRounded(__ubuf__ T *output,
    Reg::RegTensor<float> &sum, Reg::MaskReg &mask)
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
__aicore__ inline void ComputePostFourHeads(__ubuf__ T *x1, __ubuf__ T *residual,
    __ubuf__ float *hPost, __ubuf__ float *hRes, __ubuf__ T *output,
    uint32_t rows, uint32_t d, uint32_t dStart, uint32_t dCount, __ubuf__ float *sumSquares = nullptr)
{
    const uint16_t rowCount = static_cast<uint16_t>(rows);
    const uint16_t repeatTimes = static_cast<uint16_t>((dCount + 63U) / 64U);
    __VEC_SCOPE__
    {
        for (uint16_t row = 0; row < rowCount; ++row) {
            Reg::RegTensor<float> squares;
            Reg::MaskReg fullMask = Compute::MaskForCount(kRegFloatLanes);
            if constexpr (ACCUMULATE_SQUARES) {
                Reg::Duplicate(squares, 0.0f, fullMask);
            }
            Reg::RegTensor<float> post0;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post0, hPost + row * 4U + 0U);
            Reg::RegTensor<float> post1;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post1, hPost + row * 4U + 1U);
            Reg::RegTensor<float> post2;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post2, hPost + row * 4U + 2U);
            Reg::RegTensor<float> post3;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post3, hPost + row * 4U + 3U);
            Reg::RegTensor<float> w00;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w00, hRes + row * 16U + 0U);
            Reg::RegTensor<float> w01;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w01, hRes + row * 16U + 1U);
            Reg::RegTensor<float> w02;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w02, hRes + row * 16U + 2U);
            Reg::RegTensor<float> w03;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w03, hRes + row * 16U + 3U);
            Reg::RegTensor<float> w10;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w10, hRes + row * 16U + 4U);
            Reg::RegTensor<float> w11;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w11, hRes + row * 16U + 5U);
            Reg::RegTensor<float> w12;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w12, hRes + row * 16U + 6U);
            Reg::RegTensor<float> w13;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w13, hRes + row * 16U + 7U);
            Reg::RegTensor<float> w20;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w20, hRes + row * 16U + 8U);
            Reg::RegTensor<float> w21;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w21, hRes + row * 16U + 9U);
            Reg::RegTensor<float> w22;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w22, hRes + row * 16U + 10U);
            Reg::RegTensor<float> w23;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w23, hRes + row * 16U + 11U);
            Reg::RegTensor<float> w30;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w30, hRes + row * 16U + 12U);
            Reg::RegTensor<float> w31;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w31, hRes + row * 16U + 13U);
            Reg::RegTensor<float> w32;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w32, hRes + row * 16U + 14U);
            Reg::RegTensor<float> w33;
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w33, hRes + row * 16U + 15U);
            uint32_t remaining = dCount;
            for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
                const uint32_t column = dStart + repeat * kRegCastVectorLanes;
                Reg::MaskReg mask = Reg::UpdateMask<float>(remaining);
                Reg::RegTensor<T> input;
                Reg::RegTensor<float> value;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x1 + row * d + column);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                Reg::RegTensor<float> sum0;
                Reg::Mul(sum0, value, post0, mask);
                Reg::RegTensor<float> sum1;
                Reg::Mul(sum1, value, post1, mask);
                Reg::RegTensor<float> sum2;
                Reg::Mul(sum2, value, post2, mask);
                Reg::RegTensor<float> sum3;
                Reg::Mul(sum3, value, post3, mask);
                AccumulatePostSource<T>(residual + (row * 4U + 0U) * d + column,
                    w00, w01, w02, w03, sum0, sum1, sum2, sum3, mask);
                AccumulatePostSource<T>(residual + (row * 4U + 1U) * d + column,
                    w10, w11, w12, w13, sum0, sum1, sum2, sum3, mask);
                AccumulatePostSource<T>(residual + (row * 4U + 2U) * d + column,
                    w20, w21, w22, w23, sum0, sum1, sum2, sum3, mask);
                AccumulatePostSource<T>(residual + (row * 4U + 3U) * d + column,
                    w30, w31, w32, w33, sum0, sum1, sum2, sum3, mask);
                if constexpr (ACCUMULATE_SQUARES) {
                    StorePostHeadRounded<T>(output + (row * 4U + 0U) * d + column, sum0, mask);
                    Reg::Mul(sum0, sum0, sum0, fullMask);
                    StorePostHeadRounded<T>(output + (row * 4U + 1U) * d + column, sum1, mask);
                    Reg::MulAddDst(sum0, sum1, sum1, fullMask);
                    Reg::Add(squares, squares, sum0, fullMask);
                    StorePostHeadRounded<T>(output + (row * 4U + 2U) * d + column, sum2, mask);
                    Reg::Mul(sum2, sum2, sum2, fullMask);
                    StorePostHeadRounded<T>(output + (row * 4U + 3U) * d + column, sum3, mask);
                    Reg::MulAddDst(sum2, sum3, sum3, fullMask);
                    Reg::Add(squares, squares, sum2, fullMask);
                } else {
                    StorePostHead<T>(output + (row * 4U + 0U) * d + column, sum0, mask);
                    StorePostHead<T>(output + (row * 4U + 1U) * d + column, sum1, mask);
                    StorePostHead<T>(output + (row * 4U + 2U) * d + column, sum2, mask);
                    StorePostHead<T>(output + (row * 4U + 3U) * d + column, sum3, mask);
                }
            }
            if constexpr (ACCUMULATE_SQUARES) {
                Reg::RegTensor<float> reduced;
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
__aicore__ inline void ComputePostTile(__ubuf__ T *x1, __ubuf__ T *residual,
                                      __ubuf__ float *hPost, __ubuf__ float *hRes,
                                      __ubuf__ T *output, uint32_t rows, uint32_t d,
                                      uint32_t headStart, uint32_t headCount,
                                      uint32_t dStart, uint32_t dCount)
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
        for (uint16_t row = 0; row < rowCount; ++row) {
            for (uint16_t headIndex = 0; headIndex < outputHeads; ++headIndex) {
                const uint32_t head = headStart + headIndex;
                Reg::RegTensor<float> postWeight;
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(postWeight, hPost + row * 4U + head);
                Reg::RegTensor<float> weight0;
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight0, hRes + row * 16U + 0U + head);
                Reg::RegTensor<float> weight1;
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight1, hRes + row * 16U + 4U + head);
                Reg::RegTensor<float> weight2;
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight2, hRes + row * 16U + 8U + head);
                Reg::RegTensor<float> weight3;
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(weight3, hRes + row * 16U + 12U + head);
                uint32_t remaining = dCount;
                for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
                    const uint32_t column = dStart + repeat * kRegCastVectorLanes;
                    Reg::MaskReg mask = Reg::UpdateMask<float>(remaining);
                    Reg::RegTensor<T> input;
                    Reg::RegTensor<float> value;
                    Reg::RegTensor<float> product;
                    Reg::RegTensor<float> sum;
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, x1 + row * d + column);
                    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                    Reg::Mul(sum, value, postWeight, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input,
                        residual + (row * 4U + 0U) * d + column);
                    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                    Reg::Mul(product, value, weight0, mask);
                    Reg::Add(sum, sum, product, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input,
                        residual + (row * 4U + 1U) * d + column);
                    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                    Reg::Mul(product, value, weight1, mask);
                    Reg::Add(sum, sum, product, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input,
                        residual + (row * 4U + 2U) * d + column);
                    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                    Reg::Mul(product, value, weight2, mask);
                    Reg::Add(sum, sum, product, mask);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input,
                        residual + (row * 4U + 3U) * d + column);
                    Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
                    Reg::Mul(product, value, weight3, mask);
                    Reg::Add(sum, sum, product, mask);
                    StorePostHead<T>(output + (row * 4U + head) * d + column, sum, mask);
                }
            }
        }
    }
}

}  // namespace AiInfraMegaMhc::Compute

#endif  // OPS_TRANSFORMER_MHC_MEGA_MHC_POST_H
