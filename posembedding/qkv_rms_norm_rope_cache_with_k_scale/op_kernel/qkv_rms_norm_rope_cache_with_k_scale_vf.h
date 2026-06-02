/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VF_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VF_H_

#include "kernel_operator.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_common.h"

namespace QkvRmsNormRopeCacheWithKScale {
namespace MicroAPI = AscendC::MicroAPI;

constexpr uint32_t QKV_RRC_D128_FLOAT_REPEAT_SIZE = 64U;
constexpr uint32_t QKV_RRC_D128_HALF_SIZE = 64U;
constexpr float QKV_RRC_FP8_E4M3FN_MAX = 448.0F;

constexpr MicroAPI::CastTrait QKV_RRC_CAST_F32_TO_BF16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr MicroAPI::CastTrait QKV_RRC_CAST_BF16_TO_F32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait QKV_RRC_CAST_F32_TO_FP8_E4M3FN = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

__simd_callee__ inline void ScatterRopeBf16HalfToNz(MicroAPI::RegTensor<bfloat16_t> &outBf16,
    __ubuf__ bfloat16_t *rowOutNz, __ubuf__ uint16_t *nzScatterIndex, uint32_t indexOffset)
{
    MicroAPI::RegTensor<bfloat16_t> packedBf16;
    MicroAPI::RegTensor<uint16_t> index;
    uint32_t halfMaskValue = QKV_RRC_D128_HALF_SIZE;
    MicroAPI::MaskReg mask64 = MicroAPI::UpdateMask<bfloat16_t>(halfMaskValue);

    MicroAPI::Pack((MicroAPI::RegTensor<uint16_t> &)packedBf16, (MicroAPI::RegTensor<uint32_t> &)outBf16);
    MicroAPI::LoadAlign<uint16_t>(index, nzScatterIndex + indexOffset);
    MicroAPI::Scatter<bfloat16_t, uint16_t>(rowOutNz, packedBf16, index, mask64);
}

__simd_callee__ inline void StoreRopeBf16RowToNzByScatter(MicroAPI::RegTensor<bfloat16_t> &outLowBf16,
    MicroAPI::RegTensor<bfloat16_t> &outHighBf16, __ubuf__ bfloat16_t *outBf16Nz,
    __ubuf__ uint16_t *nzScatterIndex, uint32_t localRow)
{
    __ubuf__ bfloat16_t *rowOutNz = outBf16Nz + localRow * QKV_RRC_NZ_C0;
    ScatterRopeBf16HalfToNz(outLowBf16, rowOutNz, nzScatterIndex, 0U);
    ScatterRopeBf16HalfToNz(outHighBf16, rowOutNz, nzScatterIndex, QKV_RRC_D128_HALF_SIZE);
}

__simd_callee__ inline void QkRmsNormRopeD128RowNzVf(__ubuf__ bfloat16_t *rowBf16, __ubuf__ float *gamma,
    __ubuf__ float *cosSin, __ubuf__ bfloat16_t *outBf16Nz, __ubuf__ uint16_t *nzScatterIndex,
    uint32_t localRow, float reciprocal, float epsilon)
{
    MicroAPI::RegTensor<bfloat16_t> xLowBf16;
    MicroAPI::RegTensor<bfloat16_t> xHighBf16;
    MicroAPI::RegTensor<float> xLow;
    MicroAPI::RegTensor<float> xHigh;
    MicroAPI::RegTensor<float> gammaLow;
    MicroAPI::RegTensor<float> gammaHigh;
    MicroAPI::RegTensor<float> squareLow;
    MicroAPI::RegTensor<float> squareHigh;
    MicroAPI::RegTensor<float> squareSum;
    MicroAPI::RegTensor<float> reduceSum;
    MicroAPI::RegTensor<float> divisor;
    MicroAPI::RegTensor<float> rms;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xLowBf16, rowBf16);
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        xHighBf16, rowBf16 + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
    MicroAPI::Cast<float, bfloat16_t, QKV_RRC_CAST_BF16_TO_F32>(xLow, xLowBf16, mask64);
    MicroAPI::Cast<float, bfloat16_t, QKV_RRC_CAST_BF16_TO_F32>(xHigh, xHighBf16, mask64);

    MicroAPI::Mul(squareLow, xLow, xLow, mask64);
    MicroAPI::Mul(squareHigh, xHigh, xHigh, mask64);
    MicroAPI::Add(squareSum, squareLow, squareHigh, mask64);

    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        reduceSum, squareSum, mask64);
    MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(reduceSum, reduceSum, reciprocal, maskFirst);
    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(reduceSum, reduceSum, epsilon, maskFirst);
    MicroAPI::Sqrt(rms, reduceSum, maskFirst);
    MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(divisor, rms, mask64);

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(gammaLow, gamma);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(gammaHigh, gamma + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
    MicroAPI::Div(xLow, xLow, divisor, mask64);
    MicroAPI::Div(xHigh, xHigh, divisor, mask64);
    MicroAPI::Mul(xLow, xLow, gammaLow, mask64);
    MicroAPI::Mul(xHigh, xHigh, gammaHigh, mask64);

    MicroAPI::RegTensor<float> cosValue;
    MicroAPI::RegTensor<float> sinValue;
    MicroAPI::RegTensor<float> outLow;
    MicroAPI::RegTensor<float> outHigh;
    MicroAPI::RegTensor<float> tmpLow;
    MicroAPI::RegTensor<float> tmpHigh;
    MicroAPI::RegTensor<bfloat16_t> outLowBf16;
    MicroAPI::RegTensor<bfloat16_t> outHighBf16;

    uint32_t halfMaskValue = QKV_RRC_D128_HALF_SIZE;
    mask64 = MicroAPI::UpdateMask<float>(halfMaskValue);

    MicroAPI::DataCopy(cosValue, cosSin);
    MicroAPI::DataCopy(sinValue, cosSin + QKV_RRC_D128_HALF_SIZE);

    MicroAPI::Mul(outLow, xLow, cosValue, mask64);
    MicroAPI::Mul(tmpLow, xHigh, sinValue, mask64);
    MicroAPI::Sub(outLow, outLow, tmpLow, mask64);

    MicroAPI::Mul(outHigh, xHigh, cosValue, mask64);
    MicroAPI::Mul(tmpHigh, xLow, sinValue, mask64);
    MicroAPI::Add(outHigh, outHigh, tmpHigh, mask64);

    MicroAPI::Cast<bfloat16_t, float, QKV_RRC_CAST_F32_TO_BF16>(outLowBf16, outLow, mask64);
    MicroAPI::Cast<bfloat16_t, float, QKV_RRC_CAST_F32_TO_BF16>(outHighBf16, outHigh, mask64);
    StoreRopeBf16RowToNzByScatter(outLowBf16, outHighBf16, outBf16Nz, nzScatterIndex, localRow);
}

__simd_vf__ inline void QkRmsNormRopeD128NzVf(__ubuf__ bfloat16_t *inputBf16, __ubuf__ float *gamma,
    __ubuf__ float *cosSin, __ubuf__ bfloat16_t *outBf16Nz, __ubuf__ uint16_t *nzScatterIndex,
    uint32_t tokenSize, uint32_t qkHeadOffset, uint32_t qkHeadSize, uint32_t qkvStrideT,
    uint32_t numQHeads, float reciprocal, float epsilon)
{
    for (uint32_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        for (uint32_t sliceIdx = 0U; sliceIdx < qkHeadSize; ++sliceIdx) {
            const uint32_t qkHead = qkHeadOffset + sliceIdx;
            const uint32_t srcOffset = tokenIdx * qkvStrideT + qkHead * QKV_RRC_D128_FLOAT_REPEAT_SIZE * 2U;
            const uint32_t localRow = tokenIdx * qkHeadSize + sliceIdx;
            const uint32_t gammaOffset =
                qkHead < numQHeads ? 0U : QKV_RRC_D128_FLOAT_REPEAT_SIZE * 2U;
            const uint32_t cosSinOffset = tokenIdx * QKV_RRC_D128_FLOAT_REPEAT_SIZE * 2U;
            QkRmsNormRopeD128RowNzVf(inputBf16 + srcOffset, gamma + gammaOffset, cosSin + cosSinOffset,
                outBf16Nz, nzScatterIndex, localRow, reciprocal, epsilon);
        }
    }
}

__simd_callee__ inline void VScaleFp8D128RowVf(__ubuf__ bfloat16_t *vBf16, __ubuf__ float *vScale,
    __ubuf__ fp8_e4m3fn_t *vOutFp8)
{
    MicroAPI::RegTensor<bfloat16_t> vLowBf16;
    MicroAPI::RegTensor<bfloat16_t> vHighBf16;
    MicroAPI::RegTensor<float> vLow;
    MicroAPI::RegTensor<float> vHigh;
    MicroAPI::RegTensor<float> scale;
    MicroAPI::RegTensor<fp8_e4m3fn_t> vLowFp8;
    MicroAPI::RegTensor<fp8_e4m3fn_t> vHighFp8;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(vLowBf16, vBf16);
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        vHighBf16, vBf16 + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
    MicroAPI::Cast<float, bfloat16_t, QKV_RRC_CAST_BF16_TO_F32>(vLow, vLowBf16, mask64);
    MicroAPI::Cast<float, bfloat16_t, QKV_RRC_CAST_BF16_TO_F32>(vHigh, vHighBf16, mask64);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(scale, vScale);
    MicroAPI::Mul(vLow, vLow, scale, mask64);
    MicroAPI::Mul(vHigh, vHigh, scale, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(vLowFp8, vLow, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(vHighFp8, vHigh, mask64);
    MicroAPI::DataCopy<fp8_e4m3fn_t, MicroAPI::StoreDist::DIST_PACK4_B32>(vOutFp8, vLowFp8, mask64);
    MicroAPI::DataCopy<fp8_e4m3fn_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
        vOutFp8 + QKV_RRC_D128_FLOAT_REPEAT_SIZE, vHighFp8, mask64);
}

__simd_vf__ inline void VScaleFp8D128Vf(__ubuf__ bfloat16_t *inputBf16, __ubuf__ float *vScale,
    __ubuf__ fp8_e4m3fn_t *vOutFp8, uint32_t tokenSize, uint32_t vHeadOffset, uint32_t vHeadSize,
    uint32_t qkvStrideT, uint32_t vInputHeadBase)
{
    for (uint32_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        for (uint32_t headIdx = 0U; headIdx < vHeadSize; ++headIdx) {
            const uint32_t vHead = vHeadOffset + headIdx;
            const uint32_t srcOffset =
                tokenIdx * qkvStrideT + (vInputHeadBase + vHead) * QKV_RRC_D128_FLOAT_REPEAT_SIZE * 2U;
            const uint32_t dstOffset =
                (tokenIdx * vHeadSize + headIdx) * QKV_RRC_D128_FLOAT_REPEAT_SIZE * 2U;
            VScaleFp8D128RowVf(inputBf16 + srcOffset, vScale + vHead, vOutFp8 + dstOffset);
        }
    }
}

__simd_callee__ inline void QkDynamicQuantD128RowVf(__ubuf__ float *qkFp32, __ubuf__ fp8_e4m3fn_t *qkFp8,
    __ubuf__ float *scaleOut)
{
    MicroAPI::RegTensor<float> qkLow;
    MicroAPI::RegTensor<float> qkHigh;
    MicroAPI::RegTensor<float> absLow;
    MicroAPI::RegTensor<float> absHigh;
    MicroAPI::RegTensor<float> maxAbs;
    MicroAPI::RegTensor<float> fp8Max;
    MicroAPI::RegTensor<float> scale;
    MicroAPI::RegTensor<float> scaleBroadcast;
    MicroAPI::RegTensor<fp8_e4m3fn_t> qkLowFp8;
    MicroAPI::RegTensor<fp8_e4m3fn_t> qkHighFp8;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(qkLow, qkFp32);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
        qkHigh, qkFp32 + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
    MicroAPI::Abs(absLow, qkLow, mask64);
    MicroAPI::Abs(absHigh, qkHigh, mask64);
    MicroAPI::Max(maxAbs, absLow, absHigh, mask64);
    MicroAPI::ReduceMax(maxAbs, maxAbs, mask64);
    MicroAPI::Duplicate(fp8Max, QKV_RRC_FP8_E4M3FN_MAX);
    MicroAPI::Div(scale, maxAbs, fp8Max, maskFirst);
    MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
        scaleBroadcast, scale, mask64);
    MicroAPI::Div(qkLow, qkLow, scaleBroadcast, mask64);
    MicroAPI::Div(qkHigh, qkHigh, scaleBroadcast, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(qkLowFp8, qkLow, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(qkHighFp8, qkHigh, mask64);
    MicroAPI::DataCopy<fp8_e4m3fn_t, MicroAPI::StoreDist::DIST_PACK4_B32>(qkFp8, qkLowFp8, mask64);
    MicroAPI::DataCopy<fp8_e4m3fn_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
        qkFp8 + QKV_RRC_D128_FLOAT_REPEAT_SIZE, qkHighFp8, mask64);
    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleOut, scale, maskFirst);
}

__simd_vf__ inline void QkDynamicQuantD128Vf(__ubuf__ float *qkFp32, __ubuf__ fp8_e4m3fn_t *qkFp8,
    __ubuf__ float *scaleOut, uint32_t rowSize)
{
    for (uint32_t rowIdx = 0U; rowIdx < rowSize; ++rowIdx) {
        const uint32_t rowOffset = rowIdx * QKV_RRC_D128_FLOAT_REPEAT_SIZE * 2U;
        QkDynamicQuantD128RowVf(qkFp32 + rowOffset, qkFp8 + rowOffset, scaleOut + rowIdx);
    }
}

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VF_H_
