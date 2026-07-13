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

constexpr uint32_t QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE = 64U;
constexpr uint32_t QKV_K_SCALE_D128_HALF_SIZE = 64U;
constexpr uint32_t QKV_K_SCALE_D128_FULL_SIZE = QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE * 2U;
constexpr float QKV_K_SCALE_D128_RECIP = 1.0F / static_cast<float>(QKV_K_SCALE_HEAD_DIM_D128);
constexpr float QKV_K_SCALE_FP8_E4M3FN_MAX = 448.0F;

constexpr MicroAPI::CastTrait QKV_K_SCALE_CAST_F32_TO_BF16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr MicroAPI::CastTrait QKV_K_SCALE_CAST_BF16_TO_F32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait QKV_K_SCALE_CAST_F32_TO_FP8_E4M3FN = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

__simd_callee__ inline void
RmsNormBf16ToFp32D128(MicroAPI::RegTensor<float> &outLow, MicroAPI::RegTensor<float> &outHigh,
                      MicroAPI::RegTensor<bfloat16_t> &inLowBf16, MicroAPI::RegTensor<bfloat16_t> &inHighBf16,
                      MicroAPI::RegTensor<float> &gammaLow, MicroAPI::RegTensor<float> &gammaHigh, float epsilon,
                      MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst)
{
    MicroAPI::RegTensor<float> squareLow;
    MicroAPI::RegTensor<float> squareHigh;
    MicroAPI::RegTensor<float> squareSum;
    MicroAPI::RegTensor<float> reduceSum;
    MicroAPI::RegTensor<float> divisor;
    MicroAPI::RegTensor<float> rms;

    MicroAPI::Cast<float, bfloat16_t, QKV_K_SCALE_CAST_BF16_TO_F32>(outLow, inLowBf16, mask64);
    MicroAPI::Cast<float, bfloat16_t, QKV_K_SCALE_CAST_BF16_TO_F32>(outHigh, inHighBf16, mask64);
    MicroAPI::Mul(squareLow, outLow, outLow, mask64);
    MicroAPI::Mul(squareHigh, outHigh, outHigh, mask64);
    MicroAPI::Add(squareSum, squareLow, squareHigh, mask64);
    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(reduceSum, squareSum,
                                                                                                mask64);
    MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(reduceSum, reduceSum, QKV_K_SCALE_D128_RECIP,
                                                                   maskFirst);
    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(reduceSum, reduceSum, epsilon, maskFirst);
    MicroAPI::Sqrt(rms, reduceSum, maskFirst);
    MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(divisor, rms, mask64);

    MicroAPI::Div(outLow, outLow, divisor, mask64);
    MicroAPI::Div(outHigh, outHigh, divisor, mask64);
    MicroAPI::Mul(outLow, outLow, gammaLow, mask64);
    MicroAPI::Mul(outHigh, outHigh, gammaHigh, mask64);
}

__simd_callee__ inline void RopeCastBf16D128(MicroAPI::RegTensor<bfloat16_t> &outLowBf16,
                                             MicroAPI::RegTensor<bfloat16_t> &outHighBf16,
                                             MicroAPI::RegTensor<float> &xLow, MicroAPI::RegTensor<float> &xHigh,
                                             MicroAPI::RegTensor<float> &cosValue, MicroAPI::RegTensor<float> &sinValue,
                                             MicroAPI::MaskReg maskHalf)
{
    MicroAPI::RegTensor<float> outLow;
    MicroAPI::RegTensor<float> outHigh;
    MicroAPI::RegTensor<float> tmpLow;
    MicroAPI::RegTensor<float> tmpHigh;

    MicroAPI::Mul(outLow, xLow, cosValue, maskHalf);
    MicroAPI::Mul(tmpLow, xHigh, sinValue, maskHalf);
    MicroAPI::Sub(outLow, outLow, tmpLow, maskHalf);

    MicroAPI::Mul(outHigh, xHigh, cosValue, maskHalf);
    MicroAPI::Mul(tmpHigh, xLow, sinValue, maskHalf);
    MicroAPI::Add(outHigh, outHigh, tmpHigh, maskHalf);

    MicroAPI::Cast<bfloat16_t, float, QKV_K_SCALE_CAST_F32_TO_BF16>(outLowBf16, outLow, maskHalf);
    MicroAPI::Cast<bfloat16_t, float, QKV_K_SCALE_CAST_F32_TO_BF16>(outHighBf16, outHigh, maskHalf);
}

__simd_callee__ inline void ScatterNzBf16D128(__ubuf__ bfloat16_t *outBf16Nz,
                                              MicroAPI::RegTensor<bfloat16_t> &outLowBf16,
                                              MicroAPI::RegTensor<bfloat16_t> &outHighBf16,
                                              MicroAPI::RegTensor<uint16_t> &nzIndex, uint16_t tokenIdx,
                                              uint16_t headIdx, uint32_t outputTokenStride, uint32_t outputHeadStride,
                                              uint32_t halfDOffset, MicroAPI::MaskReg bf16HighBitMask)
{
    const uint32_t rowOutOffset =
        (static_cast<uint32_t>(tokenIdx) * outputTokenStride + static_cast<uint32_t>(headIdx) * outputHeadStride) *
        QKV_K_SCALE_NZ_C0;
    MicroAPI::Scatter<bfloat16_t, uint16_t>(outBf16Nz + rowOutOffset, outLowBf16, nzIndex, bf16HighBitMask);
    MicroAPI::Scatter<bfloat16_t, uint16_t>(outBf16Nz + rowOutOffset + halfDOffset, outHighBf16, nzIndex,
                                            bf16HighBitMask);
}

__simd_callee__ inline void ScaleBf16ToFp8D128(MicroAPI::RegTensor<fp8_e4m3fn_t> &outLowFp8,
                                               MicroAPI::RegTensor<fp8_e4m3fn_t> &outHighFp8,
                                               MicroAPI::RegTensor<bfloat16_t> &inLowBf16,
                                               MicroAPI::RegTensor<bfloat16_t> &inHighBf16,
                                               MicroAPI::RegTensor<float> &scale, MicroAPI::MaskReg mask64)
{
    MicroAPI::RegTensor<float> inLow;
    MicroAPI::RegTensor<float> inHigh;

    MicroAPI::Cast<float, bfloat16_t, QKV_K_SCALE_CAST_BF16_TO_F32>(inLow, inLowBf16, mask64);
    MicroAPI::Cast<float, bfloat16_t, QKV_K_SCALE_CAST_BF16_TO_F32>(inHigh, inHighBf16, mask64);
    MicroAPI::Mul(inLow, inLow, scale, mask64);
    MicroAPI::Mul(inHigh, inHigh, scale, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_K_SCALE_CAST_F32_TO_FP8_E4M3FN>(outLowFp8, inLow, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_K_SCALE_CAST_F32_TO_FP8_E4M3FN>(outHighFp8, inHigh, mask64);
}

__simd_callee__ inline void DynamicQuantFp8D128(MicroAPI::RegTensor<fp8_e4m3fn_t> &qkLowFp8,
                                                MicroAPI::RegTensor<fp8_e4m3fn_t> &qkHighFp8,
                                                MicroAPI::RegTensor<float> &scale, MicroAPI::RegTensor<float> &qkLow,
                                                MicroAPI::RegTensor<float> &qkHigh, MicroAPI::RegTensor<float> &fp8Max,
                                                MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst)
{
    MicroAPI::RegTensor<float> absLow;
    MicroAPI::RegTensor<float> absHigh;
    MicroAPI::RegTensor<float> maxAbs;
    MicroAPI::RegTensor<float> scaleBroadcast;

    MicroAPI::Abs(absLow, qkLow, mask64);
    MicroAPI::Abs(absHigh, qkHigh, mask64);
    MicroAPI::Max(maxAbs, absLow, absHigh, mask64);
    MicroAPI::ReduceMax(maxAbs, maxAbs, mask64);
    MicroAPI::Div(scale, maxAbs, fp8Max, maskFirst);
    MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(scaleBroadcast, scale,
                                                                                                mask64);
    MicroAPI::Div(qkLow, qkLow, scaleBroadcast, mask64);
    MicroAPI::Div(qkHigh, qkHigh, scaleBroadcast, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_K_SCALE_CAST_F32_TO_FP8_E4M3FN>(qkLowFp8, qkLow, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, QKV_K_SCALE_CAST_F32_TO_FP8_E4M3FN>(qkHighFp8, qkHigh, mask64);
}

__simd_vf__ inline void QkRmsNormRopeD128SegmentNzVfImpl(__ubuf__ bfloat16_t *inputBf16, __ubuf__ float *gamma,
                                                         __ubuf__ float *cosSin, __ubuf__ bfloat16_t *outBf16Nz,
                                                         __ubuf__ uint16_t *nzScatterIndex, uint16_t tokenSize,
                                                         uint16_t headSize, uint32_t inputTokenStride,
                                                         uint32_t inputHeadStride, uint32_t outputTokenStride,
                                                         uint32_t outputHeadStride, uint32_t outputRowStride,
                                                         float epsilon)
{
    MicroAPI::RegTensor<float> gammaLow;
    MicroAPI::RegTensor<float> gammaHigh;
    MicroAPI::RegTensor<uint16_t> nzIndex;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    uint32_t fp32HalfMaskValue = QKV_K_SCALE_D128_HALF_SIZE;
    MicroAPI::MaskReg maskHalf = MicroAPI::UpdateMask<float>(fp32HalfMaskValue);
    MicroAPI::MaskReg bf16HighBitMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(gammaLow, gamma);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(gammaHigh, gamma + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE);
    MicroAPI::LoadAlign<uint16_t>(nzIndex, nzScatterIndex);
    const uint32_t halfDOffset = (QKV_K_SCALE_D128_HALF_SIZE / QKV_K_SCALE_NZ_C0) * outputRowStride * QKV_K_SCALE_NZ_C0;

    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        MicroAPI::RegTensor<float> cosValue;
        MicroAPI::RegTensor<float> sinValue;
        MicroAPI::AddrReg cosSinAddrReg = MicroAPI::CreateAddrReg<float>(tokenIdx, QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(cosValue, cosSin, cosSinAddrReg);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(sinValue, cosSin + QKV_K_SCALE_D128_HALF_SIZE,
                                                                  cosSinAddrReg);

        for (uint16_t headIdx = 0U; headIdx < headSize; ++headIdx) {
            MicroAPI::RegTensor<bfloat16_t> xLowBf16;
            MicroAPI::RegTensor<bfloat16_t> xHighBf16;
            MicroAPI::RegTensor<float> xLow;
            MicroAPI::RegTensor<float> xHigh;

            MicroAPI::AddrReg inputAddrReg =
                MicroAPI::CreateAddrReg<bfloat16_t>(tokenIdx, inputTokenStride, headIdx, inputHeadStride);
            MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xLowBf16, inputBf16, inputAddrReg);
            MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                xHighBf16, inputBf16 + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE, inputAddrReg);
            RmsNormBf16ToFp32D128(xLow, xHigh, xLowBf16, xHighBf16, gammaLow, gammaHigh, epsilon, mask64, maskFirst);

            MicroAPI::RegTensor<bfloat16_t> outLowBf16;
            MicroAPI::RegTensor<bfloat16_t> outHighBf16;
            RopeCastBf16D128(outLowBf16, outHighBf16, xLow, xHigh, cosValue, sinValue, maskHalf);
            ScatterNzBf16D128(outBf16Nz, outLowBf16, outHighBf16, nzIndex, tokenIdx, headIdx, outputTokenStride,
                              outputHeadStride, halfDOffset, bf16HighBitMask);
        }
    }
}

__simd_vf__ inline void VScaleFp8D128ToNtdVfImpl(__ubuf__ bfloat16_t *inputBf16, __ubuf__ float *vScale,
                                                 __ubuf__ fp8_e4m3fn_t *vOutFp8Ntd, uint16_t tokenSize,
                                                 uint16_t vHeadSize, uint32_t inputTokenStride,
                                                 uint32_t inputHeadStride)
{
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    const uint32_t outputHeadStride = tokenSize * QKV_K_SCALE_D128_FULL_SIZE;
    const uint32_t outputTokenStride = QKV_K_SCALE_D128_FULL_SIZE;
    for (uint16_t headIdx = 0U; headIdx < vHeadSize; ++headIdx) {
        MicroAPI::RegTensor<float> scale;
        MicroAPI::AddrReg vScaleAddrReg = MicroAPI::CreateAddrReg<float>(headIdx, 1U);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(scale, vScale, vScaleAddrReg);

        for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
            MicroAPI::RegTensor<bfloat16_t> vLowBf16;
            MicroAPI::RegTensor<bfloat16_t> vHighBf16;
            MicroAPI::RegTensor<fp8_e4m3fn_t> vLowFp8;
            MicroAPI::RegTensor<fp8_e4m3fn_t> vHighFp8;

            MicroAPI::AddrReg inputAddrReg =
                MicroAPI::CreateAddrReg<bfloat16_t>(headIdx, inputHeadStride, tokenIdx, inputTokenStride);
            MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(vLowBf16, inputBf16, inputAddrReg);
            MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vHighBf16, inputBf16 + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE, inputAddrReg);
            ScaleBf16ToFp8D128(vLowFp8, vHighFp8, vLowBf16, vHighBf16, scale, mask64);
            MicroAPI::AddrReg outputLowAddrReg =
                MicroAPI::CreateAddrReg<uint8_t>(headIdx, outputHeadStride, tokenIdx, outputTokenStride);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *&)vOutFp8Ntd, (MicroAPI::RegTensor<uint8_t> &)vLowFp8, outputLowAddrReg, mask64);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *&)vOutFp8Ntd + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE,
                (MicroAPI::RegTensor<uint8_t> &)vHighFp8, outputLowAddrReg, mask64);
        }
    }
}

__simd_vf__ inline void QDynamicQuantD128NtdVfImpl(__ubuf__ float *qFp32, __ubuf__ fp8_e4m3fn_t *qFp8,
                                                   __ubuf__ float *qScale, uint16_t tokenSize, uint16_t qHeadSize,
                                                   uint32_t qkUbTokenCapacity)
{
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<float> fp8Max;
    MicroAPI::Duplicate(fp8Max, QKV_K_SCALE_FP8_E4M3FN_MAX);
    const uint32_t scaleHeadStride =
        ((tokenSize + QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS - 1U) / QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS) *
        QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS;
    for (uint16_t headIdx = 0U; headIdx < qHeadSize; ++headIdx) {
        for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
            MicroAPI::RegTensor<float> qkLow;
            MicroAPI::RegTensor<float> qkHigh;
            MicroAPI::RegTensor<float> scale;
            MicroAPI::RegTensor<fp8_e4m3fn_t> qkLowFp8;
            MicroAPI::RegTensor<fp8_e4m3fn_t> qkHighFp8;

            MicroAPI::AddrReg qkAddrReg = MicroAPI::CreateAddrReg<float>(
                headIdx, qkUbTokenCapacity * QKV_K_SCALE_D128_FULL_SIZE, tokenIdx, QKV_K_SCALE_D128_FULL_SIZE);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(qkLow, qFp32, qkAddrReg);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
                qkHigh, qFp32 + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE, qkAddrReg);
            DynamicQuantFp8D128(qkLowFp8, qkHighFp8, scale, qkLow, qkHigh, fp8Max, mask64, maskFirst);

            const uint32_t qFp8Offset = (static_cast<uint32_t>(headIdx) * tokenSize + static_cast<uint32_t>(tokenIdx)) *
                                        QKV_K_SCALE_D128_FULL_SIZE;
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *&)qFp8 + qFp8Offset, (MicroAPI::RegTensor<uint8_t> &)qkLowFp8, mask64);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *&)qFp8 + qFp8Offset + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE,
                (MicroAPI::RegTensor<uint8_t> &)qkHighFp8, mask64);
            MicroAPI::AddrReg qScaleAddrReg = MicroAPI::CreateAddrReg<float>(headIdx, scaleHeadStride, tokenIdx, 1U);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(qScale, scale, qScaleAddrReg,
                                                                                     maskFirst);
        }
    }
}

__simd_vf__ inline void QDynamicQuantD128TndVfImpl(__ubuf__ float *qFp32, __ubuf__ fp8_e4m3fn_t *qFp8,
                                                   __ubuf__ float *qScale, uint16_t tokenSize, uint16_t qHeadSize)
{
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<float> fp8Max;
    MicroAPI::Duplicate(fp8Max, QKV_K_SCALE_FP8_E4M3FN_MAX);
    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        for (uint16_t headIdx = 0U; headIdx < qHeadSize; ++headIdx) {
            MicroAPI::RegTensor<float> qkLow;
            MicroAPI::RegTensor<float> qkHigh;
            MicroAPI::RegTensor<float> scale;
            MicroAPI::RegTensor<fp8_e4m3fn_t> qkLowFp8;
            MicroAPI::RegTensor<fp8_e4m3fn_t> qkHighFp8;

            MicroAPI::AddrReg qkAddrReg = MicroAPI::CreateAddrReg<float>(
                tokenIdx, qHeadSize * QKV_K_SCALE_D128_FULL_SIZE, headIdx, QKV_K_SCALE_D128_FULL_SIZE);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(qkLow, qFp32, qkAddrReg);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
                qkHigh, qFp32 + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE, qkAddrReg);
            DynamicQuantFp8D128(qkLowFp8, qkHighFp8, scale, qkLow, qkHigh, fp8Max, mask64, maskFirst);

            const uint32_t qFp8Offset = (static_cast<uint32_t>(tokenIdx) * qHeadSize + static_cast<uint32_t>(headIdx)) *
                                        QKV_K_SCALE_D128_FULL_SIZE;
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *&)qFp8 + qFp8Offset, (MicroAPI::RegTensor<uint8_t> &)qkLowFp8, mask64);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *&)qFp8 + qFp8Offset + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE,
                (MicroAPI::RegTensor<uint8_t> &)qkHighFp8, mask64);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                qScale + static_cast<uint32_t>(tokenIdx) * qHeadSize + static_cast<uint32_t>(headIdx), scale,
                maskFirst);
        }
    }
}

__simd_vf__ inline void KDynamicQuantD128VfImpl(__ubuf__ float *kFp32, __ubuf__ fp8_e4m3fn_t *kFp8,
                                                __ubuf__ float *kScaleStaging, uint16_t tokenSize, uint16_t kHeadSize,
                                                uint32_t inputHeadStride, uint32_t inputTokenStride,
                                                uint32_t outputHeadStrideBytes, uint32_t outputTokenStrideBytes)
{
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<float> fp8Max;
    MicroAPI::Duplicate(fp8Max, QKV_K_SCALE_FP8_E4M3FN_MAX);
    const uint32_t scaleTokenStride = kHeadSize * QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS;

    for (uint16_t headIdx = 0U; headIdx < kHeadSize; ++headIdx) {
        for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
            MicroAPI::RegTensor<float> qkLow;
            MicroAPI::RegTensor<float> qkHigh;
            MicroAPI::RegTensor<float> scale;
            MicroAPI::RegTensor<fp8_e4m3fn_t> qkLowFp8;
            MicroAPI::RegTensor<fp8_e4m3fn_t> qkHighFp8;

            MicroAPI::AddrReg qkAddrReg =
                MicroAPI::CreateAddrReg<float>(headIdx, inputHeadStride, tokenIdx, inputTokenStride);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(qkLow, kFp32, qkAddrReg);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
                qkHigh, kFp32 + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE, qkAddrReg);
            DynamicQuantFp8D128(qkLowFp8, qkHighFp8, scale, qkLow, qkHigh, fp8Max, mask64, maskFirst);

            MicroAPI::AddrReg kFp8LowAddrReg =
                MicroAPI::CreateAddrReg<uint8_t>(headIdx, outputHeadStrideBytes, tokenIdx, outputTokenStrideBytes);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *&)kFp8, (MicroAPI::RegTensor<uint8_t> &)qkLowFp8, kFp8LowAddrReg, mask64);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                (__ubuf__ uint8_t *&)kFp8 + QKV_K_SCALE_D128_FLOAT_REPEAT_SIZE,
                (MicroAPI::RegTensor<uint8_t> &)qkHighFp8, kFp8LowAddrReg, mask64);
            MicroAPI::AddrReg kScaleAddrReg = MicroAPI::CreateAddrReg<float>(
                headIdx, QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS, tokenIdx, scaleTokenStride);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(kScaleStaging, scale,
                                                                                     kScaleAddrReg, maskFirst);
        }
    }
}

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VF_H_
