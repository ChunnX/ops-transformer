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
constexpr uint32_t QKV_RRC_D128_FULL_SIZE = QKV_RRC_D128_FLOAT_REPEAT_SIZE * 2U;
constexpr float QKV_RRC_D128_RECIP = 1.0F / static_cast<float>(QKV_RRC_HEAD_DIM_D128);
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

template <bool IS_TND>
__simd_vf__ inline void QkRmsNormRopeD128SegmentNzVfImpl(__ubuf__ bfloat16_t *inputBf16, __ubuf__ float *gamma,
                                                        __ubuf__ float *cosSin, __ubuf__ bfloat16_t *outBf16Nz,
                                                        __ubuf__ uint16_t *nzScatterIndex, uint32_t tokenSize,
                                                        uint32_t qkHeadBase, uint32_t headSize, uint32_t qkvHeadNum,
                                                        uint32_t tokenCapacity, float epsilon)
{
    MicroAPI::RegTensor<float> gammaLow;
    MicroAPI::RegTensor<float> gammaHigh;
    MicroAPI::RegTensor<uint16_t> lowIndex;
    MicroAPI::RegTensor<uint16_t> highIndex;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    uint32_t fp32HalfMaskValue = QKV_RRC_D128_HALF_SIZE;
    uint32_t bf16HalfMaskValue = QKV_RRC_D128_HALF_SIZE;
    MicroAPI::MaskReg maskHalf = MicroAPI::UpdateMask<float>(fp32HalfMaskValue);
    MicroAPI::MaskReg bf16MaskHalf = MicroAPI::UpdateMask<bfloat16_t>(bf16HalfMaskValue);

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(gammaLow, gamma);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(gammaHigh, gamma + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
    MicroAPI::LoadAlign<uint16_t>(lowIndex, nzScatterIndex);
    MicroAPI::LoadAlign<uint16_t>(highIndex, nzScatterIndex + QKV_RRC_D128_HALF_SIZE);

    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        const uint32_t tokenCosSinOffset = static_cast<uint32_t>(tokenIdx) * QKV_RRC_D128_FULL_SIZE;
        MicroAPI::RegTensor<float> cosValue;
        MicroAPI::RegTensor<float> sinValue;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(cosValue, cosSin + tokenCosSinOffset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(sinValue,
                                                                  cosSin + tokenCosSinOffset + QKV_RRC_D128_HALF_SIZE);

        for (uint16_t headIdx = 0U; headIdx < headSize; ++headIdx) {
            const uint32_t qkHead = qkHeadBase + static_cast<uint32_t>(headIdx);
            MicroAPI::RegTensor<bfloat16_t> xLowBf16;
            MicroAPI::RegTensor<bfloat16_t> xHighBf16;
            MicroAPI::RegTensor<float> xLow;
            MicroAPI::RegTensor<float> xHigh;
            MicroAPI::RegTensor<float> squareLow;
            MicroAPI::RegTensor<float> squareHigh;
            MicroAPI::RegTensor<float> squareSum;
            MicroAPI::RegTensor<float> reduceSum;
            MicroAPI::RegTensor<float> divisor;
            MicroAPI::RegTensor<float> rms;

            if constexpr (IS_TND) {
                const uint32_t inputOffset = static_cast<uint32_t>(tokenIdx) * qkvHeadNum * QKV_RRC_D128_FULL_SIZE +
                                             qkHead * QKV_RRC_D128_FULL_SIZE;
                MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xLowBf16,
                                                                                     inputBf16 + inputOffset);
                MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    xHighBf16, inputBf16 + inputOffset + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
            } else {
                const uint32_t inputOffset = qkHead * tokenSize * QKV_RRC_D128_FULL_SIZE +
                                             static_cast<uint32_t>(tokenIdx) * QKV_RRC_D128_FULL_SIZE;
                MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xLowBf16,
                                                                                     inputBf16 + inputOffset);
                MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    xHighBf16, inputBf16 + inputOffset + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
            }
            MicroAPI::Cast<float, bfloat16_t, QKV_RRC_CAST_BF16_TO_F32>(xLow, xLowBf16, mask64);
            MicroAPI::Cast<float, bfloat16_t, QKV_RRC_CAST_BF16_TO_F32>(xHigh, xHighBf16, mask64);

            MicroAPI::Mul(squareLow, xLow, xLow, mask64);
            MicroAPI::Mul(squareHigh, xHigh, xHigh, mask64);
            MicroAPI::Add(squareSum, squareLow, squareHigh, mask64);

            MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                reduceSum, squareSum, mask64);
            MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(
                reduceSum, reduceSum, QKV_RRC_D128_RECIP, maskFirst);
            MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(reduceSum, reduceSum, epsilon, maskFirst);
            MicroAPI::Sqrt(rms, reduceSum, maskFirst);
            MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(divisor, rms,
                                                                                                        mask64);

            MicroAPI::Div(xLow, xLow, divisor, mask64);
            MicroAPI::Div(xHigh, xHigh, divisor, mask64);
            MicroAPI::Mul(xLow, xLow, gammaLow, mask64);
            MicroAPI::Mul(xHigh, xHigh, gammaHigh, mask64);

            MicroAPI::RegTensor<float> outLow;
            MicroAPI::RegTensor<float> outHigh;
            MicroAPI::RegTensor<float> tmpLow;
            MicroAPI::RegTensor<float> tmpHigh;
            MicroAPI::RegTensor<bfloat16_t> outLowBf16;
            MicroAPI::RegTensor<bfloat16_t> outHighBf16;

            MicroAPI::Mul(outLow, xLow, cosValue, maskHalf);
            MicroAPI::Mul(tmpLow, xHigh, sinValue, maskHalf);
            MicroAPI::Sub(outLow, outLow, tmpLow, maskHalf);

            MicroAPI::Mul(outHigh, xHigh, cosValue, maskHalf);
            MicroAPI::Mul(tmpHigh, xLow, sinValue, maskHalf);
            MicroAPI::Add(outHigh, outHigh, tmpHigh, maskHalf);

            MicroAPI::Cast<bfloat16_t, float, QKV_RRC_CAST_F32_TO_BF16>(outLowBf16, outLow, maskHalf);
            MicroAPI::Cast<bfloat16_t, float, QKV_RRC_CAST_F32_TO_BF16>(outHighBf16, outHigh, maskHalf);

            MicroAPI::RegTensor<bfloat16_t> packedLowBf16;
            MicroAPI::Pack((MicroAPI::RegTensor<uint16_t> &)packedLowBf16, (MicroAPI::RegTensor<uint32_t> &)outLowBf16);
            MicroAPI::RegTensor<bfloat16_t> packedHighBf16;
            MicroAPI::Pack((MicroAPI::RegTensor<uint16_t> &)packedHighBf16,
                           (MicroAPI::RegTensor<uint32_t> &)outHighBf16);
            if constexpr (IS_TND) {
                const uint32_t rowOutOffset = (static_cast<uint32_t>(tokenIdx) * headSize +
                                               static_cast<uint32_t>(headIdx)) *
                                              QKV_RRC_NZ_C0;
                MicroAPI::Scatter<bfloat16_t, uint16_t>(outBf16Nz + rowOutOffset, packedLowBf16, lowIndex,
                                                        bf16MaskHalf);
                MicroAPI::Scatter<bfloat16_t, uint16_t>(outBf16Nz + rowOutOffset, packedHighBf16, highIndex,
                                                        bf16MaskHalf);
            } else {
                const uint32_t rowOutOffset = (static_cast<uint32_t>(headIdx) * tokenCapacity +
                                               static_cast<uint32_t>(tokenIdx)) *
                                              QKV_RRC_NZ_C0;
                MicroAPI::Scatter<bfloat16_t, uint16_t>(outBf16Nz + rowOutOffset, packedLowBf16, lowIndex,
                                                        bf16MaskHalf);
                MicroAPI::Scatter<bfloat16_t, uint16_t>(outBf16Nz + rowOutOffset, packedHighBf16, highIndex,
                                                        bf16MaskHalf);
            }
        }
    }
}

template <bool IS_TND>
__simd_vf__ inline void VScaleFp8D128ToNtdVfImpl(__ubuf__ bfloat16_t *inputBf16, __ubuf__ float *vScale,
                                                __ubuf__ fp8_e4m3fn_t *vOutFp8Ntd, uint32_t tokenSize,
                                                uint32_t vHeadOffset, uint32_t vHeadSize, uint32_t qkvHeadNum,
                                                uint32_t vInputHeadBase)
{
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    const uint32_t outputHeadStride = tokenSize * QKV_RRC_D128_FULL_SIZE;
    const uint32_t outputTokenStride = QKV_RRC_D128_FULL_SIZE;
    for (uint16_t headIdx = 0U; headIdx < vHeadSize; ++headIdx) {
        const uint32_t vHead = vHeadOffset + static_cast<uint32_t>(headIdx);
        MicroAPI::RegTensor<float> scale;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(scale, vScale + vHead);

        for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
            MicroAPI::RegTensor<bfloat16_t> vLowBf16;
            MicroAPI::RegTensor<bfloat16_t> vHighBf16;
            MicroAPI::RegTensor<float> vLow;
            MicroAPI::RegTensor<float> vHigh;
            MicroAPI::RegTensor<fp8_e4m3fn_t> vLowFp8;
            MicroAPI::RegTensor<fp8_e4m3fn_t> vHighFp8;

            if constexpr (IS_TND) {
                const uint32_t srcOffset = static_cast<uint32_t>(tokenIdx) * qkvHeadNum * QKV_RRC_D128_FULL_SIZE +
                                           (vInputHeadBase + vHead) * QKV_RRC_D128_FULL_SIZE;
                MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(vLowBf16,
                                                                                     inputBf16 + srcOffset);
                MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vHighBf16, inputBf16 + srcOffset + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
            } else {
                const uint32_t srcOffset = (vInputHeadBase + vHead) * tokenSize * QKV_RRC_D128_FULL_SIZE +
                                           static_cast<uint32_t>(tokenIdx) * QKV_RRC_D128_FULL_SIZE;
                MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(vLowBf16,
                                                                                     inputBf16 + srcOffset);
                MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vHighBf16, inputBf16 + srcOffset + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
            }
            MicroAPI::Cast<float, bfloat16_t, QKV_RRC_CAST_BF16_TO_F32>(vLow, vLowBf16, mask64);
            MicroAPI::Cast<float, bfloat16_t, QKV_RRC_CAST_BF16_TO_F32>(vHigh, vHighBf16, mask64);
            MicroAPI::Mul(vLow, vLow, scale, mask64);
            MicroAPI::Mul(vHigh, vHigh, scale, mask64);
            MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(vLowFp8, vLow, mask64);
            MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(vHighFp8, vHigh, mask64);
            MicroAPI::AddrReg outputLowAddrReg =
                MicroAPI::CreateAddrReg<uint8_t>(headIdx, outputHeadStride, tokenIdx, outputTokenStride);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *&)vOutFp8Ntd, (MicroAPI::RegTensor<uint8_t> &)vLowFp8, outputLowAddrReg, mask64);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *&)vOutFp8Ntd + QKV_RRC_D128_FLOAT_REPEAT_SIZE,
                (MicroAPI::RegTensor<uint8_t> &)vHighFp8, outputLowAddrReg, mask64);
        }
    }
}

template <bool IS_TND>
__simd_vf__ inline void QDynamicQuantD128VfImpl(__ubuf__ float *qFp32, __ubuf__ fp8_e4m3fn_t *qFp8,
                                               __ubuf__ float *qScale, uint32_t tokenSize, uint32_t qHeadSize,
                                               uint32_t qkUbTokenCapacity)
{
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<float> fp8Max;
    MicroAPI::Duplicate(fp8Max, QKV_RRC_FP8_E4M3FN_MAX);
    (void)qkUbTokenCapacity;

    if constexpr (IS_TND) {
        const uint32_t tokenStride = qHeadSize * QKV_RRC_D128_FULL_SIZE;
        for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
            for (uint16_t headIdx = 0U; headIdx < qHeadSize; ++headIdx) {
                const uint32_t srcOffset = static_cast<uint32_t>(tokenIdx) * tokenStride +
                                           static_cast<uint32_t>(headIdx) * QKV_RRC_D128_FULL_SIZE;
                MicroAPI::RegTensor<float> qkLow;
                MicroAPI::RegTensor<float> qkHigh;
                MicroAPI::RegTensor<float> absLow;
                MicroAPI::RegTensor<float> absHigh;
                MicroAPI::RegTensor<float> maxAbs;
                MicroAPI::RegTensor<float> scale;
                MicroAPI::RegTensor<float> scaleBroadcast;
                MicroAPI::RegTensor<fp8_e4m3fn_t> qkLowFp8;
                MicroAPI::RegTensor<fp8_e4m3fn_t> qkHighFp8;

                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(qkLow, qFp32 + srcOffset);
                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
                    qkHigh, qFp32 + srcOffset + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
                MicroAPI::Abs(absLow, qkLow, mask64);
                MicroAPI::Abs(absHigh, qkHigh, mask64);
                MicroAPI::Max(maxAbs, absLow, absHigh, mask64);
                MicroAPI::ReduceMax(maxAbs, maxAbs, mask64);
                MicroAPI::Div(scale, maxAbs, fp8Max, maskFirst);
                MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
                    scaleBroadcast, scale, mask64);
                MicroAPI::Div(qkLow, qkLow, scaleBroadcast, mask64);
                MicroAPI::Div(qkHigh, qkHigh, scaleBroadcast, mask64);
                MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(qkLowFp8, qkLow, mask64);
                MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(qkHighFp8, qkHigh, mask64);
                MicroAPI::AddrReg qFp8LowAddrReg =
                    MicroAPI::CreateAddrReg<uint8_t>(tokenIdx, tokenStride, headIdx, QKV_RRC_D128_FULL_SIZE);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    (__ubuf__ uint8_t *&)qFp8, (MicroAPI::RegTensor<uint8_t> &)qkLowFp8, qFp8LowAddrReg, mask64);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    (__ubuf__ uint8_t *&)qFp8 + QKV_RRC_D128_FLOAT_REPEAT_SIZE,
                    (MicroAPI::RegTensor<uint8_t> &)qkHighFp8, qFp8LowAddrReg, mask64);
                MicroAPI::AddrReg qScaleAddrReg = MicroAPI::CreateAddrReg<float>(tokenIdx, qHeadSize, headIdx, 1U);
                MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    qScale, scale, qScaleAddrReg, maskFirst);
            }
        }
    } else {
        const uint32_t inputHeadStride = qkUbTokenCapacity * QKV_RRC_D128_FULL_SIZE;
        const uint32_t outputHeadStride = tokenSize * QKV_RRC_D128_FULL_SIZE;
        const uint32_t scaleHeadStride =
            ((tokenSize + QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS - 1U) / QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS) *
            QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS;
        for (uint16_t headIdx = 0U; headIdx < qHeadSize; ++headIdx) {
            for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
                const uint32_t srcOffset = static_cast<uint32_t>(headIdx) * inputHeadStride +
                                           static_cast<uint32_t>(tokenIdx) * QKV_RRC_D128_FULL_SIZE;
                MicroAPI::RegTensor<float> qkLow;
                MicroAPI::RegTensor<float> qkHigh;
                MicroAPI::RegTensor<float> absLow;
                MicroAPI::RegTensor<float> absHigh;
                MicroAPI::RegTensor<float> maxAbs;
                MicroAPI::RegTensor<float> scale;
                MicroAPI::RegTensor<float> scaleBroadcast;
                MicroAPI::RegTensor<fp8_e4m3fn_t> qkLowFp8;
                MicroAPI::RegTensor<fp8_e4m3fn_t> qkHighFp8;

                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(qkLow, qFp32 + srcOffset);
                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
                    qkHigh, qFp32 + srcOffset + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
                MicroAPI::Abs(absLow, qkLow, mask64);
                MicroAPI::Abs(absHigh, qkHigh, mask64);
                MicroAPI::Max(maxAbs, absLow, absHigh, mask64);
                MicroAPI::ReduceMax(maxAbs, maxAbs, mask64);
                MicroAPI::Div(scale, maxAbs, fp8Max, maskFirst);
                MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
                    scaleBroadcast, scale, mask64);
                MicroAPI::Div(qkLow, qkLow, scaleBroadcast, mask64);
                MicroAPI::Div(qkHigh, qkHigh, scaleBroadcast, mask64);
                MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(qkLowFp8, qkLow, mask64);
                MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(qkHighFp8, qkHigh, mask64);
                MicroAPI::AddrReg qFp8LowAddrReg =
                    MicroAPI::CreateAddrReg<uint8_t>(headIdx, outputHeadStride, tokenIdx,
                                                     QKV_RRC_D128_FULL_SIZE);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    (__ubuf__ uint8_t *&)qFp8, (MicroAPI::RegTensor<uint8_t> &)qkLowFp8, qFp8LowAddrReg, mask64);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    (__ubuf__ uint8_t *&)qFp8 + QKV_RRC_D128_FLOAT_REPEAT_SIZE,
                    (MicroAPI::RegTensor<uint8_t> &)qkHighFp8, qFp8LowAddrReg, mask64);
                MicroAPI::AddrReg qScaleAddrReg =
                    MicroAPI::CreateAddrReg<float>(headIdx, scaleHeadStride, tokenIdx, 1U);
                MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    qScale, scale, qScaleAddrReg, maskFirst);
            }
        }
    }
}

template <bool IS_TND>
__simd_vf__ inline void KDynamicQuantD128VfImpl(__ubuf__ float *kFp32, __ubuf__ fp8_e4m3fn_t *kFp8,
                                               __ubuf__ float *kScaleStaging, uint32_t tokenSize,
                                               uint32_t kHeadSize, uint32_t qkUbTokenCapacity)
{
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<float> fp8Max;
    MicroAPI::Duplicate(fp8Max, QKV_RRC_FP8_E4M3FN_MAX);
    (void)qkUbTokenCapacity;

    if constexpr (IS_TND) {
        const uint32_t tokenStride = kHeadSize * QKV_RRC_D128_FULL_SIZE;
        const uint32_t scaleTokenStride = kHeadSize * QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS;
        for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
            for (uint16_t headIdx = 0U; headIdx < kHeadSize; ++headIdx) {
                const uint32_t srcOffset = static_cast<uint32_t>(tokenIdx) * tokenStride +
                                           static_cast<uint32_t>(headIdx) * QKV_RRC_D128_FULL_SIZE;
                MicroAPI::RegTensor<float> qkLow;
                MicroAPI::RegTensor<float> qkHigh;
                MicroAPI::RegTensor<float> absLow;
                MicroAPI::RegTensor<float> absHigh;
                MicroAPI::RegTensor<float> maxAbs;
                MicroAPI::RegTensor<float> scale;
                MicroAPI::RegTensor<float> scaleBroadcast;
                MicroAPI::RegTensor<fp8_e4m3fn_t> qkLowFp8;
                MicroAPI::RegTensor<fp8_e4m3fn_t> qkHighFp8;

                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(qkLow, kFp32 + srcOffset);
                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
                    qkHigh, kFp32 + srcOffset + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
                MicroAPI::Abs(absLow, qkLow, mask64);
                MicroAPI::Abs(absHigh, qkHigh, mask64);
                MicroAPI::Max(maxAbs, absLow, absHigh, mask64);
                MicroAPI::ReduceMax(maxAbs, maxAbs, mask64);
                MicroAPI::Div(scale, maxAbs, fp8Max, maskFirst);
                MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
                    scaleBroadcast, scale, mask64);
                MicroAPI::Div(qkLow, qkLow, scaleBroadcast, mask64);
                MicroAPI::Div(qkHigh, qkHigh, scaleBroadcast, mask64);
                MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(qkLowFp8, qkLow, mask64);
                MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(qkHighFp8, qkHigh, mask64);
                MicroAPI::AddrReg kFp8LowAddrReg =
                    MicroAPI::CreateAddrReg<uint8_t>(tokenIdx, tokenStride, headIdx, QKV_RRC_D128_FULL_SIZE);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    (__ubuf__ uint8_t *&)kFp8, (MicroAPI::RegTensor<uint8_t> &)qkLowFp8, kFp8LowAddrReg, mask64);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    (__ubuf__ uint8_t *&)kFp8 + QKV_RRC_D128_FLOAT_REPEAT_SIZE,
                    (MicroAPI::RegTensor<uint8_t> &)qkHighFp8, kFp8LowAddrReg, mask64);
                MicroAPI::AddrReg kScaleAddrReg = MicroAPI::CreateAddrReg<float>(
                    tokenIdx, scaleTokenStride, headIdx, QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS);
                MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    kScaleStaging, scale, kScaleAddrReg, maskFirst);
            }
        }
    } else {
        const uint32_t inputHeadStride = qkUbTokenCapacity * QKV_RRC_D128_FULL_SIZE;
        const uint32_t outputHeadStride = tokenSize * QKV_RRC_D128_FULL_SIZE;
        const uint32_t scaleTokenStride = kHeadSize * QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS;
        for (uint16_t headIdx = 0U; headIdx < kHeadSize; ++headIdx) {
            for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
                const uint32_t srcOffset = static_cast<uint32_t>(headIdx) * inputHeadStride +
                                           static_cast<uint32_t>(tokenIdx) * QKV_RRC_D128_FULL_SIZE;
                MicroAPI::RegTensor<float> qkLow;
                MicroAPI::RegTensor<float> qkHigh;
                MicroAPI::RegTensor<float> absLow;
                MicroAPI::RegTensor<float> absHigh;
                MicroAPI::RegTensor<float> maxAbs;
                MicroAPI::RegTensor<float> scale;
                MicroAPI::RegTensor<float> scaleBroadcast;
                MicroAPI::RegTensor<fp8_e4m3fn_t> qkLowFp8;
                MicroAPI::RegTensor<fp8_e4m3fn_t> qkHighFp8;

                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(qkLow, kFp32 + srcOffset);
                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
                    qkHigh, kFp32 + srcOffset + QKV_RRC_D128_FLOAT_REPEAT_SIZE);
                MicroAPI::Abs(absLow, qkLow, mask64);
                MicroAPI::Abs(absHigh, qkHigh, mask64);
                MicroAPI::Max(maxAbs, absLow, absHigh, mask64);
                MicroAPI::ReduceMax(maxAbs, maxAbs, mask64);
                MicroAPI::Div(scale, maxAbs, fp8Max, maskFirst);
                MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
                    scaleBroadcast, scale, mask64);
                MicroAPI::Div(qkLow, qkLow, scaleBroadcast, mask64);
                MicroAPI::Div(qkHigh, qkHigh, scaleBroadcast, mask64);
                MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(qkLowFp8, qkLow, mask64);
                MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_RRC_CAST_F32_TO_FP8_E4M3FN>(qkHighFp8, qkHigh, mask64);
                MicroAPI::AddrReg kFp8LowAddrReg =
                    MicroAPI::CreateAddrReg<uint8_t>(headIdx, outputHeadStride, tokenIdx,
                                                     QKV_RRC_D128_FULL_SIZE);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    (__ubuf__ uint8_t *&)kFp8, (MicroAPI::RegTensor<uint8_t> &)qkLowFp8, kFp8LowAddrReg, mask64);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    (__ubuf__ uint8_t *&)kFp8 + QKV_RRC_D128_FLOAT_REPEAT_SIZE,
                    (MicroAPI::RegTensor<uint8_t> &)qkHighFp8, kFp8LowAddrReg, mask64);
                MicroAPI::AddrReg kScaleAddrReg =
                    MicroAPI::CreateAddrReg<float>(headIdx, QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS, tokenIdx,
                                                   scaleTokenStride);
                MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    kScaleStaging, scale, kScaleAddrReg, maskFirst);
            }
        }
    }
}

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VF_H_
