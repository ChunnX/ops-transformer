/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_VF_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_VF_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_layout.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_vf.h"

namespace QkvRmsNormRopeCacheWithKScale {

constexpr uint32_t MROPE_MX_FP32_ABS_MASK = 0x7fffffffU;
constexpr uint32_t MROPE_MX_FP32_INF_BITS = 0x7f800000U;
constexpr uint32_t MROPE_MX_FP32_CUBLAS_FOLDED_ROUND_OFFSET = 0xfc1fffffU;
constexpr uint32_t MROPE_MX_FP32_CUBLAS_SUBNORMAL_EDGE = 0x04600001U;
constexpr uint32_t MROPE_MX_FP32_EXP_BIAS_CUBLAS = 0x7f000000U;
constexpr uint32_t MROPE_MX_FP32_NEG_EXP_UNIT = 0xff800000U;
constexpr uint32_t MROPE_MX_FP32_NAN_BITS = 0x7f810000U;
constexpr uint32_t MROPE_MX_E8M0_NAN = 0x000000ffU;
constexpr int16_t MROPE_MX_FP32_EXP_SHIFT = 23;
constexpr MicroAPI::CastTrait MROPE_MX_CAST_FP32_TO_FP8 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait MROPE_MX_CAST_FP32_TO_FP8_ONE = {
    MicroAPI::RegLayout::ONE, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait MROPE_MX_CAST_FP32_TO_FP8_TWO = {
    MicroAPI::RegLayout::TWO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait MROPE_MX_CAST_FP32_TO_FP8_THREE = {
    MicroAPI::RegLayout::THREE, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

__simd_callee__ inline void RmsNormMropeFp32D128(
    MicroAPI::RegTensor<float> &outLow, MicroAPI::RegTensor<float> &outHigh,
    MicroAPI::RegTensor<bfloat16_t> &inLowBf16, MicroAPI::RegTensor<bfloat16_t> &inHighBf16,
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh,
    float epsilon, MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst)
{
    MicroAPI::RegTensor<float> normLow;
    MicroAPI::RegTensor<float> normHigh;
    MicroAPI::RegTensor<float> rotateTmp;
    RmsNormBf16ToFp32D128<false>(normLow, normHigh, inLowBf16, inHighBf16, gammaCosLow, gammaCosHigh, epsilon,
                                 mask64, maskFirst);
    MicroAPI::Mul(outLow, normLow, gammaCosLow, mask64);
    MicroAPI::Mul(rotateTmp, normHigh, gammaSinHigh, mask64);
    MicroAPI::Sub(outLow, outLow, rotateTmp, mask64);
    MicroAPI::Mul(outHigh, normHigh, gammaCosHigh, mask64);
    MicroAPI::Mul(rotateTmp, normLow, gammaSinLow, mask64);
    MicroAPI::Add(outHigh, outHigh, rotateTmp, mask64);
}

__simd_callee__ inline void MxReduceMaxD32x2(MicroAPI::RegTensor<uint32_t> &maxValue,
                                             MicroAPI::RegTensor<float> &input,
                                             MicroAPI::MaskReg mask64,
                                             MicroAPI::MaskReg maskLow32,
                                             MicroAPI::MaskReg maskHigh32)
{
    MicroAPI::RegTensor<uint32_t> absMask;
    MicroAPI::RegTensor<uint32_t> absValue;
    MicroAPI::RegTensor<uint32_t> lowMax;
    MicroAPI::RegTensor<uint32_t> highMax;
    MicroAPI::RegTensor<uint32_t> interleaveScratch;

    MicroAPI::Duplicate(absMask, MROPE_MX_FP32_ABS_MASK);
    MicroAPI::And(absValue, (MicroAPI::RegTensor<uint32_t> &)input, absMask, mask64);
    MicroAPI::ReduceMax(lowMax, absValue, maskLow32);
    MicroAPI::ReduceMax(highMax, absValue, maskHigh32);
    MicroAPI::Interleave(maxValue, interleaveScratch, lowMax, highMax);
}

__simd_callee__ inline void MxQuantCublasScaleD32x4(MicroAPI::RegTensor<float> &reciprocal,
                                                    MicroAPI::RegTensor<uint32_t> &scaleExp,
                                                    MicroAPI::RegTensor<uint32_t> &maxValue,
                                                    MicroAPI::MaskReg mask4)
{
    MicroAPI::MaskReg normalPredicate;
    MicroAPI::MaskReg invalidPredicate;
    MicroAPI::RegTensor<uint32_t> adjustedBits;

    // For max > 0x04600001:
    //   ((max + 0x001fffff) >> 23) - 8
    // == ((max + 0xfc1fffff) mod 2^32) >> 23.
    // ZEROING keeps every lower value, including the locked edge, at scale 0.
    MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(
        normalPredicate, maxValue, MROPE_MX_FP32_CUBLAS_SUBNORMAL_EDGE, mask4);
    MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GE>(
        invalidPredicate, maxValue, MROPE_MX_FP32_INF_BITS, mask4);
    MicroAPI::Adds(adjustedBits, maxValue, MROPE_MX_FP32_CUBLAS_FOLDED_ROUND_OFFSET, mask4);
    MicroAPI::ShiftRights<uint32_t, int16_t, MicroAPI::MaskMergeMode::ZEROING>(
        scaleExp, adjustedBits, MROPE_MX_FP32_EXP_SHIFT, normalPredicate);

    MicroAPI::Muls<uint32_t>((MicroAPI::RegTensor<uint32_t> &)reciprocal, scaleExp,
                             MROPE_MX_FP32_NEG_EXP_UNIT, mask4);
    MicroAPI::Adds<uint32_t>((MicroAPI::RegTensor<uint32_t> &)reciprocal,
                             (MicroAPI::RegTensor<uint32_t> &)reciprocal,
                             MROPE_MX_FP32_EXP_BIAS_CUBLAS, mask4);
    MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(
        scaleExp, MROPE_MX_E8M0_NAN, invalidPredicate);
    MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(
        (MicroAPI::RegTensor<uint32_t> &)reciprocal, MROPE_MX_FP32_NAN_BITS, invalidPredicate);
}

__simd_callee__ inline void MxQuantCublasD128(__ubuf__ uint8_t *outBytes,
                                              MicroAPI::AddrReg outAddr,
                                              __ubuf__ uint8_t *scaleBytes,
                                              MicroAPI::RegTensor<float> &inLow,
                                              MicroAPI::RegTensor<float> &inHigh,
                                              MicroAPI::RegTensor<uint32_t> &scaleGatherLowIndex,
                                              MicroAPI::RegTensor<uint32_t> &scaleGatherHighIndex,
                                              MicroAPI::MaskReg mask64,
                                              MicroAPI::MaskReg maskLow32,
                                              MicroAPI::MaskReg maskHigh32,
                                              MicroAPI::MaskReg mask4)
{
    MicroAPI::RegTensor<uint32_t> maxLow;
    MicroAPI::RegTensor<uint32_t> maxHigh;
    MicroAPI::RegTensor<uint32_t> maxCombined;
    MicroAPI::RegTensor<uint32_t> interleaveScratch;
    MxReduceMaxD32x2(maxLow, inLow, mask64, maskLow32, maskHigh32);
    MxReduceMaxD32x2(maxHigh, inHigh, mask64, maskLow32, maskHigh32);
    MicroAPI::Interleave(maxCombined, interleaveScratch, maxLow, maxHigh);

    MicroAPI::RegTensor<float> reciprocalCombined;
    MicroAPI::RegTensor<uint32_t> scaleExpCombined;
    MxQuantCublasScaleD32x4(reciprocalCombined, scaleExpCombined, maxCombined, mask4);

    MicroAPI::RegTensor<uint32_t> scaleExpLow;
    MicroAPI::RegTensor<uint32_t> scaleExpHigh;
    MicroAPI::DeInterleave(scaleExpLow, scaleExpHigh, scaleExpCombined, scaleExpCombined);

    MicroAPI::RegTensor<fp8_e4m3fn_t> output;
    MicroAPI::RegTensor<uint16_t> scaleB16;
    MicroAPI::RegTensor<uint8_t> scale;
    MicroAPI::RegTensor<float> expandedScale;
    MicroAPI::UnalignReg scaleStore;

    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scaleB16, scaleExpLow);
    MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(scale, scaleB16);
    MicroAPI::Gather(expandedScale, reciprocalCombined, scaleGatherLowIndex);
    MicroAPI::Mul(inLow, inLow, expandedScale, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8>(output, inLow, mask64);
    MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
        outBytes, (MicroAPI::RegTensor<uint8_t> &)output, outAddr, mask64);
    MicroAPI::StoreUnAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        scaleBytes, scale, scaleStore, 2U);

    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scaleB16, scaleExpHigh);
    MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(scale, scaleB16);
    MicroAPI::Gather(expandedScale, reciprocalCombined, scaleGatherHighIndex);
    MicroAPI::Mul(inHigh, inHigh, expandedScale, mask64);
    MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8>(output, inHigh, mask64);
    MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
        outBytes + QKV_K_SCALE_D128_HALF_SIZE, (MicroAPI::RegTensor<uint8_t> &)output, outAddr, mask64);
    MicroAPI::StoreUnAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        scaleBytes, scale, scaleStore, 2U);
    MicroAPI::StoreUnAlignPost(scaleBytes, scaleStore, 0U);
}

__simd_callee__ inline void PrepareMropeGammaD128(
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh,
    MicroAPI::RegTensor<float> &gammaLow, MicroAPI::RegTensor<float> &gammaHigh,
    MicroAPI::RegTensor<uint32_t> &mropeIndex, __ubuf__ float *rawToken, MicroAPI::MaskReg mask64)
{
    MicroAPI::RegTensor<float> cosValue;
    MicroAPI::RegTensor<float> sinValue;
    MicroAPI::Gather(cosValue, rawToken, mropeIndex, mask64);
    MicroAPI::Gather(sinValue, rawToken + QKV_K_SCALE_D128_HALF_SIZE, mropeIndex, mask64);
    MicroAPI::Mul(gammaSinLow, gammaLow, sinValue, mask64);
    MicroAPI::Mul(gammaSinHigh, gammaHigh, sinValue, mask64);
    MicroAPI::Mul(gammaCosLow, gammaLow, cosValue, mask64);
    MicroAPI::Mul(gammaCosHigh, gammaHigh, cosValue, mask64);
}

__simd_callee__ inline void LoadRmsNormMropeFp32D128(
    MicroAPI::RegTensor<float> &outLow, MicroAPI::RegTensor<float> &outHigh, __ubuf__ bfloat16_t *inputBf16,
    MicroAPI::AddrReg inputAddr,
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh, float epsilon,
    MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst)
{
    MicroAPI::RegTensor<bfloat16_t> inLowBf16;
    MicroAPI::RegTensor<bfloat16_t> inHighBf16;
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(inLowBf16, inputBf16, inputAddr);
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        inHighBf16, inputBf16 + QKV_K_SCALE_D128_HALF_SIZE, inputAddr);
    RmsNormMropeFp32D128(outLow, outHigh, inLowBf16, inHighBf16, gammaCosLow, gammaCosHigh, gammaSinLow,
                         gammaSinHigh, epsilon, mask64, maskFirst);
}

// MX quantization only observes the normalized value after another per-D32
// power-of-two scaling.  Keep the row data unnormalized here and defer the
// common RMS divisor to the 16-row scale phase, where one vector Sqrt/Div pair
// serves all rows in the batch.
__simd_callee__ inline void LoadRmsSquaredMropeUnnormalizedFp32D128(
    MicroAPI::RegTensor<float> &outLow, MicroAPI::RegTensor<float> &outHigh,
    MicroAPI::RegTensor<float> &rmsSquared, __ubuf__ bfloat16_t *inputBf16,
    MicroAPI::AddrReg inputAddr,
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh,
    float epsilon, MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst)
{
    MicroAPI::RegTensor<bfloat16_t> inLowBf16;
    MicroAPI::RegTensor<bfloat16_t> inHighBf16;
    MicroAPI::RegTensor<float> inLow;
    MicroAPI::RegTensor<float> inHigh;
    MicroAPI::RegTensor<float> squareLow;
    MicroAPI::RegTensor<float> squareHigh;
    MicroAPI::RegTensor<float> squareSum;
    MicroAPI::RegTensor<float> rotateTmp;

    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(inLowBf16, inputBf16, inputAddr);
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        inHighBf16, inputBf16 + QKV_K_SCALE_D128_HALF_SIZE, inputAddr);
    MicroAPI::Cast<float, bfloat16_t, QKV_K_SCALE_CAST_BF16_TO_F32>(inLow, inLowBf16, mask64);
    MicroAPI::Cast<float, bfloat16_t, QKV_K_SCALE_CAST_BF16_TO_F32>(inHigh, inHighBf16, mask64);
    MicroAPI::Mul(squareLow, inLow, inLow, mask64);
    MicroAPI::Mul(squareHigh, inHigh, inHigh, mask64);
    MicroAPI::Add(squareSum, squareLow, squareHigh, mask64);
    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        rmsSquared, squareSum, mask64);
    MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(
        rmsSquared, rmsSquared, QKV_K_SCALE_D128_RECIP, maskFirst);
    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(
        rmsSquared, rmsSquared, epsilon, maskFirst);

    MicroAPI::Mul(outLow, inLow, gammaCosLow, mask64);
    MicroAPI::Mul(rotateTmp, inHigh, gammaSinHigh, mask64);
    MicroAPI::Sub(outLow, outLow, rotateTmp, mask64);
    MicroAPI::Mul(outHigh, inHigh, gammaCosHigh, mask64);
    MicroAPI::Mul(rotateTmp, inLow, gammaSinLow, mask64);
    MicroAPI::Add(outHigh, outHigh, rotateTmp, mask64);
}

template <bool REDUCE_MAX = true>
__simd_callee__ inline void StoreAndReduceMropeMxD128x2(
    __ubuf__ float *row0LowScratch, __ubuf__ float *row0HighScratch,
    __ubuf__ float *row1LowScratch, __ubuf__ float *row1HighScratch,
    __ubuf__ uint32_t *maxScratch, MicroAPI::AddrReg dataAddr, MicroAPI::AddrReg maxAddr,
    MicroAPI::RegTensor<float> &row0Low, MicroAPI::RegTensor<float> &row0High,
    MicroAPI::RegTensor<float> &row1Low, MicroAPI::RegTensor<float> &row1High,
    MicroAPI::RegTensor<uint32_t> &absMask, MicroAPI::MaskReg mask64, MicroAPI::MaskReg mask8)
{
    MicroAPI::StoreAlign<float>(row0LowScratch, row0Low, dataAddr, mask64);
    MicroAPI::StoreAlign<float>(row0HighScratch, row0High, dataAddr, mask64);
    MicroAPI::StoreAlign<float>(row1LowScratch, row1Low, dataAddr, mask64);
    MicroAPI::StoreAlign<float>(row1HighScratch, row1High, dataAddr, mask64);
    if constexpr (REDUCE_MAX) {
        MicroAPI::DeInterleave(row0Low, row0High, row0Low, row0High);
        MicroAPI::DeInterleave(row1Low, row1High, row1Low, row1High);
        MicroAPI::DeInterleave(row0Low, row1Low, row0Low, row1Low);
        MicroAPI::DeInterleave(row0High, row1High, row0High, row1High);
        MicroAPI::And((MicroAPI::RegTensor<uint32_t> &)row0Low, (MicroAPI::RegTensor<uint32_t> &)row0Low,
                      absMask, mask64);
        MicroAPI::And((MicroAPI::RegTensor<uint32_t> &)row0High, (MicroAPI::RegTensor<uint32_t> &)row0High,
                      absMask, mask64);
        MicroAPI::And((MicroAPI::RegTensor<uint32_t> &)row1Low, (MicroAPI::RegTensor<uint32_t> &)row1Low,
                      absMask, mask64);
        MicroAPI::And((MicroAPI::RegTensor<uint32_t> &)row1High, (MicroAPI::RegTensor<uint32_t> &)row1High,
                      absMask, mask64);
        MicroAPI::Max((MicroAPI::RegTensor<uint32_t> &)row0Low, (MicroAPI::RegTensor<uint32_t> &)row0Low,
                      (MicroAPI::RegTensor<uint32_t> &)row1Low, mask64);
        MicroAPI::Max((MicroAPI::RegTensor<uint32_t> &)row0High, (MicroAPI::RegTensor<uint32_t> &)row0High,
                      (MicroAPI::RegTensor<uint32_t> &)row1High, mask64);
        MicroAPI::Max((MicroAPI::RegTensor<uint32_t> &)row0Low, (MicroAPI::RegTensor<uint32_t> &)row0Low,
                      (MicroAPI::RegTensor<uint32_t> &)row0High, mask64);
        MicroAPI::ReduceMaxWithDataBlock((MicroAPI::RegTensor<uint32_t> &)row0Low,
                                         (MicroAPI::RegTensor<uint32_t> &)row0Low, mask64);
        MicroAPI::StoreAlign<uint32_t>(
            maxScratch, (MicroAPI::RegTensor<uint32_t> &)row0Low, maxAddr, mask8);
    }
}

template <bool REDUCE_MAX = true>
__simd_callee__ inline void StageMropeMxHeadPairD128(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ float *row0LowScratch,
    __ubuf__ float *row0HighScratch, __ubuf__ float *row1LowScratch,
    __ubuf__ float *row1HighScratch, __ubuf__ uint32_t *maxScratch,
    MicroAPI::AddrReg inputAddr, MicroAPI::AddrReg dataAddr, MicroAPI::AddrReg maxAddr,
    uint32_t inputHeadStride,
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh,
    MicroAPI::RegTensor<uint32_t> &absMask, float epsilon, MicroAPI::MaskReg mask64,
    MicroAPI::MaskReg maskFirst, MicroAPI::MaskReg mask8)
{
    MicroAPI::RegTensor<float> row0Low;
    MicroAPI::RegTensor<float> row0High;
    MicroAPI::RegTensor<float> row1Low;
    MicroAPI::RegTensor<float> row1High;
    LoadRmsNormMropeFp32D128(row0Low, row0High, inputBf16, inputAddr, gammaCosLow, gammaCosHigh, gammaSinLow,
                             gammaSinHigh, epsilon, mask64, maskFirst);
    LoadRmsNormMropeFp32D128(row1Low, row1High, inputBf16 + inputHeadStride, inputAddr, gammaCosLow, gammaCosHigh,
                             gammaSinLow, gammaSinHigh, epsilon, mask64, maskFirst);
    StoreAndReduceMropeMxD128x2<REDUCE_MAX>(
        row0LowScratch, row0HighScratch, row1LowScratch, row1HighScratch, maxScratch, dataAddr, maxAddr,
        row0Low, row0High, row1Low, row1High, absMask, mask64, mask8);
}

__simd_callee__ inline void StageMropeMxHeadPairDeferredRmsD128(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ float *row0LowScratch,
    __ubuf__ float *row0HighScratch, __ubuf__ float *row1LowScratch,
    __ubuf__ float *row1HighScratch, __ubuf__ uint32_t *maxScratch,
    __ubuf__ float *rmsSquaredScratch, MicroAPI::AddrReg inputAddr,
    MicroAPI::AddrReg dataAddr, MicroAPI::AddrReg maxAddr, MicroAPI::AddrReg rmsAddr,
    uint32_t inputHeadStride,
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh,
    MicroAPI::RegTensor<uint32_t> &absMask, float epsilon, MicroAPI::MaskReg mask64,
    MicroAPI::MaskReg maskFirst, MicroAPI::MaskReg mask8)
{
    // row0RmsSquared = rms²_head[2×pairIdx]
    // row1RmsSquared = rms²_head[2×pairIdx+1]
    MicroAPI::RegTensor<float> row0Low;
    MicroAPI::RegTensor<float> row0High;
    MicroAPI::RegTensor<float> row1Low;
    MicroAPI::RegTensor<float> row1High;
    MicroAPI::RegTensor<float> row0RmsSquared;
    MicroAPI::RegTensor<float> row1RmsSquared;
    MicroAPI::RegTensor<float> rmsPair;
    MicroAPI::RegTensor<float> rmsPairScratch;
    LoadRmsSquaredMropeUnnormalizedFp32D128(
        row0Low, row0High, row0RmsSquared, inputBf16, inputAddr, gammaCosLow, gammaCosHigh,
        gammaSinLow, gammaSinHigh, epsilon, mask64, maskFirst);
    LoadRmsSquaredMropeUnnormalizedFp32D128(
        row1Low, row1High, row1RmsSquared, inputBf16 + inputHeadStride, inputAddr,
        gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, epsilon, mask64, maskFirst);
    StoreAndReduceMropeMxD128x2(
        row0LowScratch, row0HighScratch, row1LowScratch, row1HighScratch, maxScratch,
        dataAddr, maxAddr, row0Low, row0High, row1Low, row1High, absMask, mask64, mask8);
    MicroAPI::Interleave(rmsPair, rmsPairScratch, row0RmsSquared, row1RmsSquared);
    // [rms²_head0, rms²_head1, 0,0,0,0,0,0]
    MicroAPI::StoreAlign<float>(rmsSquaredScratch, rmsPair, rmsAddr, mask8);
}

__simd_callee__ inline void MxQuantCublasScaleRowBatch16DeferredRms(
    __ubuf__ uint32_t *maxScratch, __ubuf__ float *rmsSquaredScratch,
    __ubuf__ float *reciprocalScratch, __ubuf__ uint8_t *scaleBytes,
    MicroAPI::RegTensor<uint32_t> &rmsGatherIndex,
    MicroAPI::MaskReg maskScale, MicroAPI::MaskReg maskScaleB16)
{
    // scale = 根据 normalizedMax 生成的2的幂
    // combinedReciprocal = 1 / (rms × scale)
    MicroAPI::RegTensor<uint32_t> maxValue;
    MicroAPI::RegTensor<float> rmsSquaredSparse;
    MicroAPI::RegTensor<float> rmsSquared;
    MicroAPI::RegTensor<float> rms;
    MicroAPI::RegTensor<float> normalizedMax;
    MicroAPI::RegTensor<float> reciprocal;
    MicroAPI::RegTensor<uint32_t> scaleExp;
    MicroAPI::RegTensor<uint16_t> scaleB16;

    // 读取64个 D32 最大值
    MicroAPI::LoadAlign<uint32_t>(maxValue, maxScratch);
    // 读取并广播16个 rms²
    // head0, head1, 空×6,
    // head2, head3, 空×6,
    // ...
    MicroAPI::LoadAlign<float>(rmsSquaredSparse, rmsSquaredScratch);
    MicroAPI::Gather(rmsSquared, rmsSquaredSparse, rmsGatherIndex);
    // rms = sqrt(rms²)
    MicroAPI::Sqrt(rms, rmsSquared, maskScale);
    // normalizedMax = maxZ / rms
    MicroAPI::Div(normalizedMax, (MicroAPI::RegTensor<float> &)maxValue, rms, maskScale);
    // reciprocal = 1 / scale
    MxQuantCublasScaleD32x4(reciprocal, scaleExp,
                            (MicroAPI::RegTensor<uint32_t> &)normalizedMax, maskScale);
    // 1 / scale / rms
    MicroAPI::Div(reciprocal, reciprocal, rms, maskScale);

    // 64 × uint32 exponent
    //         ↓
    // 64 × E8M0 byte
    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scaleB16, scaleExp);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_PACK_B16>(
        reinterpret_cast<__ubuf__ uint16_t *>(scaleBytes), scaleB16, maskScaleB16);
    MicroAPI::StoreAlign<float>(reciprocalScratch, reciprocal, maskScale);
}

// The two 8-row halves belong to two different tokens. Keep the accepted
// normalized-row scale formula, execute it once for 16 rows, then split the
// 64 row-major E8M0 bytes into the two original token destinations.
__simd_callee__ inline void MxQuantCublasScaleRowBatch16Split(
    __ubuf__ uint32_t *maxScratch, __ubuf__ float *reciprocalScratch,
    __ubuf__ uint8_t *scaleBytes0, __ubuf__ uint8_t *scaleBytes1,
    MicroAPI::RegTensor<uint32_t> &scaleHighGatherIndex,
    MicroAPI::MaskReg maskScale, MicroAPI::MaskReg maskScaleB16,
    MicroAPI::MaskReg maskScaleHalfB16)
{
    MicroAPI::RegTensor<uint32_t> maxValue;
    MicroAPI::RegTensor<float> reciprocal;
    MicroAPI::RegTensor<uint32_t> scaleExp;
    MicroAPI::RegTensor<uint32_t> scaleExpHigh;
    MicroAPI::RegTensor<uint16_t> scaleB16;
    MicroAPI::RegTensor<uint16_t> scaleHighB16;

    MicroAPI::LoadAlign<uint32_t>(maxValue, maxScratch);
    MxQuantCublasScaleD32x4(reciprocal, scaleExp, maxValue, maskScale);

    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scaleB16, scaleExp);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_PACK_B16>(
        reinterpret_cast<__ubuf__ uint16_t *>(scaleBytes0), scaleB16, maskScaleHalfB16);
    MicroAPI::Gather(scaleExpHigh, scaleExp, scaleHighGatherIndex);
    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scaleHighB16, scaleExpHigh);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_PACK_B16>(
        reinterpret_cast<__ubuf__ uint16_t *>(scaleBytes1), scaleHighB16, maskScaleHalfB16);

    MicroAPI::StoreAlign<float>(reciprocalScratch, reciprocal, maskScale);
}

template <uint32_t TOKEN_SCALE_COUNT>
__simd_callee__ inline void StoreMropeMxScaleRowBatch16ToPadded(
    __ubuf__ uint32_t *scaleScratch, __ubuf__ uint8_t *paddedScaleBytes, uint16_t tokenCount)
{
    MicroAPI::RegTensor<uint32_t> tokenScaleExp;
    MicroAPI::RegTensor<uint16_t> tokenScaleB16;
    uint32_t tokenScaleCount = TOKEN_SCALE_COUNT;
    MicroAPI::MaskReg tokenScaleMaskB16 = MicroAPI::UpdateMask<uint16_t>(tokenScaleCount);
    for (uint16_t localToken = 0U; localToken < tokenCount; ++localToken) {
        MicroAPI::AddrReg scaleLoadAddr =
            MicroAPI::CreateAddrReg<uint32_t>(localToken, TOKEN_SCALE_COUNT);
        MicroAPI::LoadAlign<uint32_t>(tokenScaleExp, scaleScratch, scaleLoadAddr);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(tokenScaleB16, tokenScaleExp);
        MicroAPI::AddrReg scaleStoreAddr = MicroAPI::CreateAddrReg<uint16_t>(localToken, 16U);
        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_PACK_B16>(
            reinterpret_cast<__ubuf__ uint16_t *>(paddedScaleBytes), tokenScaleB16,
            scaleStoreAddr, tokenScaleMaskB16);
    }
}

__simd_callee__ inline void MxQuantCublasScaleRowBatch16DeferredRmsToPadded(
    __ubuf__ uint32_t *maxScratch, __ubuf__ float *rmsSquaredScratch,
    __ubuf__ float *reciprocalScratch, __ubuf__ uint8_t *paddedScaleBytes,
    uint16_t tokenCount, uint16_t headSize,
    MicroAPI::RegTensor<uint32_t> &rmsGatherIndex,
    MicroAPI::MaskReg maskScale, MicroAPI::MaskReg maskScaleB16)
{
    MicroAPI::RegTensor<uint32_t> maxValue;
    MicroAPI::RegTensor<float> rmsSquaredSparse;
    MicroAPI::RegTensor<float> rmsSquared;
    MicroAPI::RegTensor<float> rms;
    MicroAPI::RegTensor<float> normalizedMax;
    MicroAPI::RegTensor<float> reciprocal;
    MicroAPI::RegTensor<uint32_t> scaleExp;

    MicroAPI::LoadAlign<uint32_t>(maxValue, maxScratch);
    MicroAPI::LoadAlign<float>(rmsSquaredSparse, rmsSquaredScratch);
    MicroAPI::Gather(rmsSquared, rmsSquaredSparse, rmsGatherIndex);
    MicroAPI::Sqrt(rms, rmsSquared, maskScale);
    MicroAPI::Div(normalizedMax, (MicroAPI::RegTensor<float> &)maxValue, rms, maskScale);
    MxQuantCublasScaleD32x4(reciprocal, scaleExp,
                            (MicroAPI::RegTensor<uint32_t> &)normalizedMax, maskScale);
    MicroAPI::Div(reciprocal, reciprocal, rms, maskScale);

    MicroAPI::StoreAlign<uint32_t>(maxScratch, scaleExp, maskScale);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    if (headSize == 2U) {
        StoreMropeMxScaleRowBatch16ToPadded<8U>(maxScratch, paddedScaleBytes, tokenCount);
    } else {
        StoreMropeMxScaleRowBatch16ToPadded<16U>(maxScratch, paddedScaleBytes, tokenCount);
    }

    MicroAPI::StoreAlign<float>(reciprocalScratch, reciprocal, maskScale);
}

__simd_callee__ inline void MxQuantCublasDataRowBatch16(
    __ubuf__ float *dataScratch, __ubuf__ float *reciprocalScratch, __ubuf__ uint8_t *outBytes,
    uint16_t pairCount, MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFp8)
{
    for (uint16_t pairIdx = 0U; pairIdx < pairCount; ++pairIdx) {
        MicroAPI::RegTensor<float> reciprocalFp32;
        MicroAPI::RegTensor<float> x0Zero;
        MicroAPI::RegTensor<float> x1Zero;
        MicroAPI::RegTensor<float> x0One;
        MicroAPI::RegTensor<float> x1One;
        MicroAPI::RegTensor<fp8_e4m3fn_t> x0ZeroFp8;
        MicroAPI::RegTensor<fp8_e4m3fn_t> x1ZeroFp8;
        MicroAPI::RegTensor<fp8_e4m3fn_t> x0OneFp8;
        MicroAPI::RegTensor<fp8_e4m3fn_t> x1OneFp8;
        MicroAPI::AddrReg reciprocalAddr = MicroAPI::CreateAddrReg<float>(
            pairIdx, 2U * MROPE_MX_SCALE_COUNT_D128);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_E2B_B32>(
            reciprocalFp32, reciprocalScratch, reciprocalAddr);
        MicroAPI::AddrReg dataAddr = MicroAPI::CreateAddrReg<float>(
            pairIdx, 2U * QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            x0Zero, x1Zero, dataScratch, dataAddr);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            x0One, x1One, dataScratch + QKV_K_SCALE_D128_FULL_SIZE, dataAddr);
        MicroAPI::DeInterleave(x0Zero, x0One, x0Zero, x0One);
        MicroAPI::DeInterleave(x1Zero, x1One, x1Zero, x1One);
        MicroAPI::Mul(x0Zero, x0Zero, reciprocalFp32, mask64);
        MicroAPI::Mul(x1Zero, x1Zero, reciprocalFp32, mask64);
        MicroAPI::Mul(x0One, x0One, reciprocalFp32, mask64);
        MicroAPI::Mul(x1One, x1One, reciprocalFp32, mask64);
        MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8>(x0ZeroFp8, x0Zero, mask64);
        MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8_ONE>(x1ZeroFp8, x1Zero, mask64);
        MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8_TWO>(x0OneFp8, x0One, mask64);
        MicroAPI::Cast<fp8_e4m3fn_t, float, MROPE_MX_CAST_FP32_TO_FP8_THREE>(x1OneFp8, x1One, mask64);
        MicroAPI::Add((MicroAPI::RegTensor<uint8_t> &)x0ZeroFp8, (MicroAPI::RegTensor<uint8_t> &)x0ZeroFp8,
                      (MicroAPI::RegTensor<uint8_t> &)x0OneFp8, maskFp8);
        MicroAPI::Add((MicroAPI::RegTensor<uint8_t> &)x1ZeroFp8, (MicroAPI::RegTensor<uint8_t> &)x1ZeroFp8,
                      (MicroAPI::RegTensor<uint8_t> &)x1OneFp8, maskFp8);
        MicroAPI::Add((MicroAPI::RegTensor<uint8_t> &)x0ZeroFp8, (MicroAPI::RegTensor<uint8_t> &)x0ZeroFp8,
                      (MicroAPI::RegTensor<uint8_t> &)x1ZeroFp8, maskFp8);
        MicroAPI::AddrReg outAddr = MicroAPI::CreateAddrReg<uint8_t>(
            pairIdx, 2U * QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
            outBytes, (MicroAPI::RegTensor<uint8_t> &)x0ZeroFp8, outAddr, maskFp8);
    }
}

__simd_callee__ inline void QuantizeMropeMxRowBatch8AfterReady(
    __ubuf__ float *dataScratch, __ubuf__ uint8_t *outBytes,
    __ubuf__ uint8_t *scaleBytes, MicroAPI::MaskReg mask64)
{
    MicroAPI::RegTensor<uint32_t> scaleGatherLowIndex;
    MicroAPI::RegTensor<uint32_t> scaleGatherHighIndex;
    MicroAPI::MaskReg maskLow32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg maskHigh32;
    MicroAPI::Not(maskHigh32, maskLow32, mask64);
    uint32_t scaleCount = MROPE_MX_SCALE_COUNT_D128;
    MicroAPI::MaskReg mask4 = MicroAPI::UpdateMask<float>(scaleCount);
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)scaleGatherLowIndex, 0);
    MicroAPI::ShiftRights(scaleGatherLowIndex, scaleGatherLowIndex, static_cast<int16_t>(5), mask64);
    MicroAPI::ShiftLefts(scaleGatherLowIndex, scaleGatherLowIndex, static_cast<int16_t>(1), mask64);
    MicroAPI::Adds(scaleGatherHighIndex, scaleGatherLowIndex, 1U, mask64);
    for (uint16_t row = 0U; row < 8U; ++row) {
        MicroAPI::RegTensor<float> rowLow;
        MicroAPI::RegTensor<float> rowHigh;
        MicroAPI::AddrReg dataAddr =
            MicroAPI::CreateAddrReg<float>(row, QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::AddrReg outAddr =
            MicroAPI::CreateAddrReg<uint8_t>(row, QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::LoadAlign<float>(rowLow, dataScratch, dataAddr);
        MicroAPI::LoadAlign<float>(rowHigh, dataScratch + QKV_K_SCALE_D128_HALF_SIZE, dataAddr);
        MxQuantCublasD128(outBytes, outAddr,
                          scaleBytes + static_cast<uint32_t>(row) * MROPE_MX_SCALE_COUNT_D128,
                          rowLow, rowHigh, scaleGatherLowIndex, scaleGatherHighIndex,
                          mask64, maskLow32, maskHigh32, mask4);
    }
}

template <bool REDUCE_MAX>
__simd_callee__ inline void StageQTailToken8D128(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ uint8_t *waveScratch,
    uint16_t rowOffset, uint32_t inputHeadStride,
    MicroAPI::RegTensor<float> &gammaCosLow, MicroAPI::RegTensor<float> &gammaCosHigh,
    MicroAPI::RegTensor<float> &gammaSinLow, MicroAPI::RegTensor<float> &gammaSinHigh,
    MicroAPI::RegTensor<uint32_t> &absMask, float epsilon,
    MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst, MicroAPI::MaskReg mask8)
{
    __ubuf__ float *dataScratch = reinterpret_cast<__ubuf__ float *>(waveScratch) +
        static_cast<uint32_t>(rowOffset) * QKV_K_SCALE_D128_FULL_SIZE;
    __ubuf__ float *row0LowScratch = dataScratch;
    __ubuf__ float *row0HighScratch = dataScratch + QKV_K_SCALE_D128_HALF_SIZE;
    __ubuf__ float *row1LowScratch = dataScratch + QKV_K_SCALE_D128_FULL_SIZE;
    __ubuf__ float *row1HighScratch =
        dataScratch + QKV_K_SCALE_D128_FULL_SIZE + QKV_K_SCALE_D128_HALF_SIZE;
    __ubuf__ uint32_t *maxScratch = reinterpret_cast<__ubuf__ uint32_t *>(
        waveScratch + MROPE_MX_ROW_BATCH_DATA_BYTES) +
        static_cast<uint32_t>(rowOffset) * MROPE_MX_SCALE_COUNT_D128;
    for (uint16_t pairIdx = 0U; pairIdx < 4U; ++pairIdx) {
        MicroAPI::AddrReg inputAddr =
            MicroAPI::CreateAddrReg<bfloat16_t>(pairIdx, 2U * inputHeadStride);
        MicroAPI::AddrReg dataAddr =
            MicroAPI::CreateAddrReg<float>(pairIdx, 2U * QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::AddrReg maxAddr =
            MicroAPI::CreateAddrReg<uint32_t>(pairIdx, 2U * MROPE_MX_SCALE_COUNT_D128);
        StageMropeMxHeadPairD128<REDUCE_MAX>(
            inputBf16, row0LowScratch, row0HighScratch, row1LowScratch, row1HighScratch,
            maxScratch, inputAddr, dataAddr, maxAddr, inputHeadStride,
            gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, absMask,
            epsilon, mask64, maskFirst, mask8);
    }
}

// Flatten the complete Q prefix batches and the cross-token Q-tail waves into
// one tile-wide Stage -> Scale -> Quant pipeline. Every scratch slot is an
// independent RowBatch16 AoS record, so all rows in the tile share exactly one
// producer/consumer barrier triplet. A final odd token uses the low eight rows
// of its tail slot and keeps the existing single-token exact quantization path.
template <bool HAS_Q_TAIL>
__simd_vf__ inline void QRmsNormMropeMxD128GlobalTileWaveVfImpl(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ float *gamma, __ubuf__ float *rawCosSin,
    __ubuf__ uint32_t *gatherIndex, __ubuf__ fp8_e4m3fn_t *out, __ubuf__ fp8_e8m0_t *outScale,
    __ubuf__ uint8_t *scratch, uint16_t tokenSize, uint32_t inputTokenStride,
    uint16_t headSize, uint32_t inputHeadStride, uint32_t scaleTokenStrideWords, float epsilon)
{
    MicroAPI::RegTensor<uint32_t> mropeIndex;
    MicroAPI::RegTensor<uint32_t> absMask;
    MicroAPI::RegTensor<float> gammaLow;
    MicroAPI::RegTensor<float> gammaHigh;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    uint32_t mask8Count = 8U;
    MicroAPI::MaskReg mask8 = MicroAPI::UpdateMask<uint32_t>(mask8Count);
    MicroAPI::MaskReg maskFp8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskScale = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    uint32_t scaleB16Count = MROPE_MX_ROW_BATCH_ROWS * MROPE_MX_SCALE_COUNT_D128;
    MicroAPI::MaskReg maskScaleB16 = MicroAPI::UpdateMask<uint16_t>(scaleB16Count);
    uint32_t scaleHalfB16Count = 8U * MROPE_MX_SCALE_COUNT_D128;
    MicroAPI::MaskReg maskScaleHalfB16 = MicroAPI::UpdateMask<uint16_t>(scaleHalfB16Count);
    MicroAPI::LoadAlign<uint32_t>(mropeIndex, gatherIndex);
    MicroAPI::LoadAlign<float>(gammaLow, gamma);
    MicroAPI::LoadAlign<float>(gammaHigh, gamma + QKV_K_SCALE_D128_HALF_SIZE);
    MicroAPI::Duplicate(absMask, MROPE_MX_FP32_ABS_MASK);

    MicroAPI::RegTensor<uint32_t> rmsGatherIndex;
    MicroAPI::RegTensor<uint32_t> rmsGatherBase;
    MicroAPI::RegTensor<uint32_t> rmsGatherOne;
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)rmsGatherIndex, 0);
    MicroAPI::ShiftRights(rmsGatherBase, rmsGatherIndex, static_cast<int16_t>(3), maskScale);
    MicroAPI::ShiftLefts(rmsGatherBase, rmsGatherBase, static_cast<int16_t>(3), maskScale);
    MicroAPI::ShiftRights(rmsGatherIndex, rmsGatherIndex, static_cast<int16_t>(2), maskScale);
    MicroAPI::Duplicate(rmsGatherOne, 1U);
    MicroAPI::And(rmsGatherIndex, rmsGatherIndex, rmsGatherOne, maskScale);
    MicroAPI::Add(rmsGatherIndex, rmsGatherBase, rmsGatherIndex, maskScale);

    MicroAPI::RegTensor<uint32_t> scaleHighGatherIndex;
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)scaleHighGatherIndex, 0);
    MicroAPI::ShiftLefts(scaleHighGatherIndex, scaleHighGatherIndex, static_cast<int16_t>(27), maskScale);
    MicroAPI::ShiftRights(scaleHighGatherIndex, scaleHighGatherIndex, static_cast<int16_t>(27), maskScale);
    MicroAPI::Adds(scaleHighGatherIndex, scaleHighGatherIndex, 32U, maskScale);

    const uint16_t fullBatchCount = static_cast<uint16_t>(headSize / MROPE_MX_ROW_BATCH_ROWS);
    const uint16_t fullHeadCount = static_cast<uint16_t>(fullBatchCount * MROPE_MX_ROW_BATCH_ROWS);
    const uint16_t prefixScratchBatchCount = static_cast<uint16_t>(tokenSize * fullBatchCount);
    const uint16_t tailPairCount = HAS_Q_TAIL ? static_cast<uint16_t>(tokenSize / 2U) : 0U;
    const bool hasOddTailToken = HAS_Q_TAIL && (tokenSize & 1U) != 0U;
    constexpr uint32_t scaleBatchStride = MROPE_MX_ROW_BATCH_ROWS * MROPE_MX_SCALE_COUNT_D128;

    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        MicroAPI::RegTensor<float> gammaCosLow;
        MicroAPI::RegTensor<float> gammaCosHigh;
        MicroAPI::RegTensor<float> gammaSinLow;
        MicroAPI::RegTensor<float> gammaSinHigh;
        PrepareMropeGammaD128(
            gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, gammaLow, gammaHigh,
            mropeIndex, rawCosSin + static_cast<uint32_t>(tokenIdx) * 3U * QKV_K_SCALE_D128_FULL_SIZE,
            mask64);
        __ubuf__ bfloat16_t *tokenInput =
            inputBf16 + static_cast<uint32_t>(tokenIdx) * inputTokenStride;

        for (uint16_t batchIdx = 0U; batchIdx < fullBatchCount; ++batchIdx) {
            const uint16_t scratchIdx = static_cast<uint16_t>(tokenIdx * fullBatchCount + batchIdx);
            __ubuf__ uint8_t *batchScratch = scratch +
                static_cast<uint32_t>(scratchIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
            __ubuf__ float *dataScratch = reinterpret_cast<__ubuf__ float *>(batchScratch);
            __ubuf__ float *row0LowScratch = dataScratch;
            __ubuf__ float *row0HighScratch = dataScratch + QKV_K_SCALE_D128_HALF_SIZE;
            __ubuf__ float *row1LowScratch = dataScratch + QKV_K_SCALE_D128_FULL_SIZE;
            __ubuf__ float *row1HighScratch =
                dataScratch + QKV_K_SCALE_D128_FULL_SIZE + QKV_K_SCALE_D128_HALF_SIZE;
            __ubuf__ uint32_t *maxScratch = reinterpret_cast<__ubuf__ uint32_t *>(
                batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES);
            __ubuf__ float *rmsSquaredScratch = reinterpret_cast<__ubuf__ float *>(
                batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES +
                MROPE_MX_ROW_BATCH_RECIP_BYTES);
            __ubuf__ bfloat16_t *batchInput = tokenInput +
                static_cast<uint32_t>(batchIdx) * MROPE_MX_ROW_BATCH_ROWS * inputHeadStride;
            for (uint16_t pairIdx = 0U; pairIdx < 8U; ++pairIdx) {
                MicroAPI::AddrReg inputAddr = MicroAPI::CreateAddrReg<bfloat16_t>(
                    pairIdx, 2U * inputHeadStride);
                MicroAPI::AddrReg dataAddr = MicroAPI::CreateAddrReg<float>(
                    pairIdx, 2U * QKV_K_SCALE_D128_FULL_SIZE);
                MicroAPI::AddrReg maxAddr = MicroAPI::CreateAddrReg<uint32_t>(
                    pairIdx, 2U * MROPE_MX_SCALE_COUNT_D128);
                MicroAPI::AddrReg rmsAddr = MicroAPI::CreateAddrReg<float>(pairIdx, 8U);
                StageMropeMxHeadPairDeferredRmsD128(
                    batchInput, row0LowScratch, row0HighScratch, row1LowScratch, row1HighScratch,
                    maxScratch, rmsSquaredScratch, inputAddr, dataAddr, maxAddr, rmsAddr,
                    inputHeadStride, gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh,
                    absMask, epsilon, mask64, maskFirst, mask8);
            }
        }

        if constexpr (HAS_Q_TAIL) {
            const uint16_t tailScratchIdx = static_cast<uint16_t>(prefixScratchBatchCount + tokenIdx / 2U);
            __ubuf__ uint8_t *tailScratch = scratch +
                static_cast<uint32_t>(tailScratchIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
            const uint16_t tailRowOffset = static_cast<uint16_t>((tokenIdx & 1U) * 8U);
            __ubuf__ bfloat16_t *tailInput = tokenInput +
                static_cast<uint32_t>(fullHeadCount) * inputHeadStride;
            const bool isFinalOddToken = hasOddTailToken && tokenIdx + 1U == tokenSize;
            if (isFinalOddToken) {
                StageQTailToken8D128<false>(
                    tailInput, tailScratch, tailRowOffset, inputHeadStride, gammaCosLow, gammaCosHigh,
                    gammaSinLow, gammaSinHigh, absMask, epsilon, mask64, maskFirst, mask8);
            } else {
                StageQTailToken8D128<true>(
                    tailInput, tailScratch, tailRowOffset, inputHeadStride, gammaCosLow, gammaCosHigh,
                    gammaSinLow, gammaSinHigh, absMask, epsilon, mask64, maskFirst, mask8);
            }
        }
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        __ubuf__ uint8_t *tokenScaleBytes = reinterpret_cast<__ubuf__ uint8_t *>(outScale) +
            static_cast<uint32_t>(tokenIdx) * scaleTokenStrideWords * sizeof(uint32_t);
        for (uint16_t batchIdx = 0U; batchIdx < fullBatchCount; ++batchIdx) {
            const uint16_t scratchIdx = static_cast<uint16_t>(tokenIdx * fullBatchCount + batchIdx);
            __ubuf__ uint8_t *batchScratch = scratch +
                static_cast<uint32_t>(scratchIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
            __ubuf__ uint32_t *maxScratch = reinterpret_cast<__ubuf__ uint32_t *>(
                batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES);
            __ubuf__ float *reciprocalScratch = reinterpret_cast<__ubuf__ float *>(
                batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES);
            __ubuf__ float *rmsSquaredScratch = reinterpret_cast<__ubuf__ float *>(
                batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES +
                MROPE_MX_ROW_BATCH_RECIP_BYTES);
            MxQuantCublasScaleRowBatch16DeferredRms(
                maxScratch, rmsSquaredScratch, reciprocalScratch,
                tokenScaleBytes + static_cast<uint32_t>(batchIdx) * scaleBatchStride,
                rmsGatherIndex, maskScale, maskScaleB16);
        }
    }
    for (uint16_t pairIdx = 0U; pairIdx < tailPairCount; ++pairIdx) {
        const uint16_t token0 = static_cast<uint16_t>(pairIdx * 2U);
        const uint16_t token1 = static_cast<uint16_t>(token0 + 1U);
        __ubuf__ uint8_t *tailScratch = scratch +
            static_cast<uint32_t>(prefixScratchBatchCount + pairIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
        __ubuf__ uint32_t *maxScratch = reinterpret_cast<__ubuf__ uint32_t *>(
            tailScratch + MROPE_MX_ROW_BATCH_DATA_BYTES);
        __ubuf__ float *reciprocalScratch = reinterpret_cast<__ubuf__ float *>(
            tailScratch + MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES);
        __ubuf__ uint8_t *scaleBytes0 = reinterpret_cast<__ubuf__ uint8_t *>(outScale) +
            (static_cast<uint32_t>(token0) * scaleTokenStrideWords + fullHeadCount) * sizeof(uint32_t);
        __ubuf__ uint8_t *scaleBytes1 = reinterpret_cast<__ubuf__ uint8_t *>(outScale) +
            (static_cast<uint32_t>(token1) * scaleTokenStrideWords + fullHeadCount) * sizeof(uint32_t);
        MxQuantCublasScaleRowBatch16Split(
            maxScratch, reciprocalScratch, scaleBytes0, scaleBytes1, scaleHighGatherIndex,
            maskScale, maskScaleB16, maskScaleHalfB16);
    }

    if (prefixScratchBatchCount != 0U || tailPairCount != 0U) {
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        __ubuf__ uint8_t *tokenOutBytes = reinterpret_cast<__ubuf__ uint8_t *>(out) +
            static_cast<uint32_t>(tokenIdx) * headSize * QKV_K_SCALE_D128_FULL_SIZE;
        for (uint16_t batchIdx = 0U; batchIdx < fullBatchCount; ++batchIdx) {
            const uint16_t scratchIdx = static_cast<uint16_t>(tokenIdx * fullBatchCount + batchIdx);
            __ubuf__ uint8_t *batchScratch = scratch +
                static_cast<uint32_t>(scratchIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
            __ubuf__ float *dataScratch = reinterpret_cast<__ubuf__ float *>(batchScratch);
            __ubuf__ float *reciprocalScratch = reinterpret_cast<__ubuf__ float *>(
                batchScratch + MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES);
            MxQuantCublasDataRowBatch16(
                dataScratch, reciprocalScratch,
                tokenOutBytes + static_cast<uint32_t>(batchIdx) * MROPE_MX_ROW_BATCH_ROWS *
                    QKV_K_SCALE_D128_FULL_SIZE,
                8U, mask64, maskFp8);
        }
    }
    for (uint16_t pairIdx = 0U; pairIdx < tailPairCount; ++pairIdx) {
        const uint16_t token0 = static_cast<uint16_t>(pairIdx * 2U);
        const uint16_t token1 = static_cast<uint16_t>(token0 + 1U);
        __ubuf__ uint8_t *tailScratch = scratch +
            static_cast<uint32_t>(prefixScratchBatchCount + pairIdx) * MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
        __ubuf__ float *dataScratch = reinterpret_cast<__ubuf__ float *>(tailScratch);
        __ubuf__ float *reciprocalScratch = reinterpret_cast<__ubuf__ float *>(
            tailScratch + MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES);
        __ubuf__ uint8_t *outBytes0 = reinterpret_cast<__ubuf__ uint8_t *>(out) +
            (static_cast<uint32_t>(token0) * headSize + fullHeadCount) * QKV_K_SCALE_D128_FULL_SIZE;
        __ubuf__ uint8_t *outBytes1 = reinterpret_cast<__ubuf__ uint8_t *>(out) +
            (static_cast<uint32_t>(token1) * headSize + fullHeadCount) * QKV_K_SCALE_D128_FULL_SIZE;
        MxQuantCublasDataRowBatch16(
            dataScratch, reciprocalScratch, outBytes0, 4U, mask64, maskFp8);
        MxQuantCublasDataRowBatch16(
            dataScratch + 8U * QKV_K_SCALE_D128_FULL_SIZE,
            reciprocalScratch + 8U * MROPE_MX_SCALE_COUNT_D128,
            outBytes1, 4U, mask64, maskFp8);
    }
    if (hasOddTailToken) {
        const uint16_t oddToken = static_cast<uint16_t>(tokenSize - 1U);
        __ubuf__ uint8_t *oddScratch = scratch +
            static_cast<uint32_t>(prefixScratchBatchCount + tailPairCount) *
                MROPE_MX_ROW_BATCH_SCRATCH_BYTES;
        __ubuf__ uint8_t *oddOutBytes = reinterpret_cast<__ubuf__ uint8_t *>(out) +
            (static_cast<uint32_t>(oddToken) * headSize + fullHeadCount) * QKV_K_SCALE_D128_FULL_SIZE;
        __ubuf__ uint8_t *oddScaleBytes = reinterpret_cast<__ubuf__ uint8_t *>(outScale) +
            (static_cast<uint32_t>(oddToken) * scaleTokenStrideWords + fullHeadCount) * sizeof(uint32_t);
        QuantizeMropeMxRowBatch8AfterReady(
            reinterpret_cast<__ubuf__ float *>(oddScratch), oddOutBytes, oddScaleBytes, mask64);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

// Process every complete two-token Q-tail wave in groups that share the same
// three scratch lifecycle barriers.  A final unpaired token occupies the last
// half-wave and shares the group's first/final barrier without crossing a tile
// or an AIV range.
__simd_callee__ inline void StageKMropeMxEvenTokenD128(
    __ubuf__ bfloat16_t *tokenInputBf16, __ubuf__ float *rawToken,
    __ubuf__ float *tokenDataScratch, __ubuf__ uint32_t *tokenMaxScratch,
    __ubuf__ float *tokenRmsSquaredScratch,
    uint16_t headSize, uint32_t inputHeadStride, float epsilon,
    MicroAPI::RegTensor<uint32_t> &mropeIndex, MicroAPI::RegTensor<uint32_t> &absMask,
    MicroAPI::RegTensor<float> &gammaLow, MicroAPI::RegTensor<float> &gammaHigh,
    MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst, MicroAPI::MaskReg mask8)
{
    MicroAPI::RegTensor<float> gammaCosLow;
    MicroAPI::RegTensor<float> gammaCosHigh;
    MicroAPI::RegTensor<float> gammaSinLow;
    MicroAPI::RegTensor<float> gammaSinHigh;
    PrepareMropeGammaD128(gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, gammaLow, gammaHigh,
                          mropeIndex, rawToken, mask64);

    const uint16_t pairCount = static_cast<uint16_t>(headSize / 2U);
    for (uint16_t pairIdx = 0U; pairIdx < pairCount; ++pairIdx) {
        MicroAPI::AddrReg inputAddr =
            MicroAPI::CreateAddrReg<bfloat16_t>(pairIdx, 2U * inputHeadStride);
        MicroAPI::AddrReg dataAddr =
            MicroAPI::CreateAddrReg<float>(pairIdx, 2U * QKV_K_SCALE_D128_FULL_SIZE);
        MicroAPI::AddrReg maxAddr =
            MicroAPI::CreateAddrReg<uint32_t>(pairIdx, 2U * MROPE_MX_SCALE_COUNT_D128);
        MicroAPI::AddrReg rmsAddr = MicroAPI::CreateAddrReg<float>(pairIdx, 8U);
        StageMropeMxHeadPairDeferredRmsD128(
            tokenInputBf16, tokenDataScratch,
            tokenDataScratch + QKV_K_SCALE_D128_HALF_SIZE,
            tokenDataScratch + QKV_K_SCALE_D128_FULL_SIZE,
            tokenDataScratch + QKV_K_SCALE_D128_FULL_SIZE + QKV_K_SCALE_D128_HALF_SIZE,
            tokenMaxScratch, tokenRmsSquaredScratch, inputAddr, dataAddr, maxAddr, rmsAddr,
            inputHeadStride,
            gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, absMask, epsilon,
            mask64, maskFirst, mask8);
    }
}

__simd_callee__ inline void FinishKMropeMxD128EvenRows(
    __ubuf__ float *dataScratch, __ubuf__ uint32_t *maxScratch, __ubuf__ float *rmsSquaredScratch,
    __ubuf__ float *reciprocalScratch, __ubuf__ uint8_t *batchOutBytes,
    __ubuf__ uint8_t *batchPaddedScaleBytes, uint16_t tokenCount, uint16_t headSize,
    MicroAPI::RegTensor<uint32_t> &rmsGatherIndex,
    MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFp8)
{
    MicroAPI::MaskReg maskScale = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    uint32_t batchScaleCount = MROPE_MX_ROW_BATCH_ROWS * MROPE_MX_SCALE_COUNT_D128;
    MicroAPI::MaskReg maskScaleB16 = MicroAPI::UpdateMask<uint16_t>(batchScaleCount);
    if (headSize == 8U) {
        MxQuantCublasScaleRowBatch16DeferredRms(
            maxScratch, rmsSquaredScratch, reciprocalScratch, batchPaddedScaleBytes,
            rmsGatherIndex, maskScale, maskScaleB16);
    } else {
        MxQuantCublasScaleRowBatch16DeferredRmsToPadded(
            maxScratch, rmsSquaredScratch, reciprocalScratch, batchPaddedScaleBytes,
            tokenCount, headSize, rmsGatherIndex, maskScale, maskScaleB16);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    MxQuantCublasDataRowBatch16(dataScratch, reciprocalScratch, batchOutBytes, 8U, mask64, maskFp8);
}

__simd_callee__ inline void ProcessKMropeMxD128RowBatch16(
    __ubuf__ bfloat16_t *batchInputBf16, __ubuf__ float *batchRawCosSin,
    __ubuf__ uint8_t *batchOutBytes, __ubuf__ uint8_t *batchPaddedScaleBytes,
    __ubuf__ float *dataScratch, __ubuf__ uint32_t *maxScratch,
    __ubuf__ float *reciprocalScratch, __ubuf__ float *rmsSquaredScratch,
    uint16_t batchTokenCount, uint16_t headSize,
    uint32_t inputTokenStride, uint32_t inputHeadStride, uint32_t scaleTokenStrideWords,
    float epsilon, MicroAPI::RegTensor<uint32_t> &mropeIndex,
    MicroAPI::RegTensor<uint32_t> &absMask, MicroAPI::RegTensor<float> &gammaLow,
    MicroAPI::RegTensor<float> &gammaHigh, MicroAPI::RegTensor<uint32_t> &rmsGatherIndex,
    MicroAPI::MaskReg mask64, MicroAPI::MaskReg maskFirst, MicroAPI::MaskReg mask8,
    MicroAPI::MaskReg maskFp8)
{
    for (uint16_t localToken = 0U; localToken < batchTokenCount; ++localToken) {
        const uint16_t tokenRowBegin = static_cast<uint16_t>(localToken * headSize);
        __ubuf__ bfloat16_t *tokenInputBf16 =
            batchInputBf16 + static_cast<uint32_t>(localToken) * inputTokenStride;
        __ubuf__ float *tokenDataScratch =
            dataScratch + static_cast<uint32_t>(tokenRowBegin) * QKV_K_SCALE_D128_FULL_SIZE;
        __ubuf__ uint32_t *tokenMaxScratch =
            maxScratch + static_cast<uint32_t>(tokenRowBegin) * MROPE_MX_SCALE_COUNT_D128;
        __ubuf__ float *tokenRmsSquaredScratch =
            rmsSquaredScratch + static_cast<uint32_t>(tokenRowBegin) * 4U;
        __ubuf__ float *rawToken =
            batchRawCosSin + static_cast<uint32_t>(localToken) * 3U * QKV_K_SCALE_D128_FULL_SIZE;
        StageKMropeMxEvenTokenD128(
            tokenInputBf16, rawToken, tokenDataScratch, tokenMaxScratch, tokenRmsSquaredScratch,
            headSize, inputHeadStride, epsilon,
            mropeIndex, absMask, gammaLow, gammaHigh,
            mask64, maskFirst, mask8);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    FinishKMropeMxD128EvenRows(
        dataScratch, maxScratch, rmsSquaredScratch, reciprocalScratch,
        batchOutBytes, batchPaddedScaleBytes, batchTokenCount, headSize,
        rmsGatherIndex, mask64, maskFp8);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

__simd_vf__ inline void KRmsNormMropeMxD128RowBatch16EvenVfImpl(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ float *gamma, __ubuf__ float *rawCosSin,
    __ubuf__ uint32_t *gatherIndex, __ubuf__ fp8_e4m3fn_t *out, __ubuf__ fp8_e8m0_t *outScale,
    __ubuf__ uint8_t *scratch, uint16_t tokenSize, uint16_t headSize,
    uint32_t inputTokenStride, uint32_t inputHeadStride, uint32_t scaleTokenStrideWords, float epsilon)
{
    MicroAPI::RegTensor<uint32_t> mropeIndex;
    MicroAPI::RegTensor<uint32_t> absMask;
    MicroAPI::RegTensor<float> gammaLow;
    MicroAPI::RegTensor<float> gammaHigh;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    uint32_t mask8Count = 8U;
    MicroAPI::MaskReg mask8 = MicroAPI::UpdateMask<uint32_t>(mask8Count);
    MicroAPI::MaskReg maskFp8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::LoadAlign<uint32_t>(mropeIndex, gatherIndex);
    MicroAPI::LoadAlign<float>(gammaLow, gamma);
    MicroAPI::LoadAlign<float>(gammaHigh, gamma + QKV_K_SCALE_D128_HALF_SIZE);
    MicroAPI::Duplicate(absMask, MROPE_MX_FP32_ABS_MASK);
    MicroAPI::RegTensor<uint32_t> rmsGatherIndex;
    MicroAPI::RegTensor<uint32_t> rmsGatherBase;
    MicroAPI::RegTensor<uint32_t> rmsGatherOne;
    MicroAPI::MaskReg maskScale = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)rmsGatherIndex, 0);
    MicroAPI::ShiftRights(rmsGatherBase, rmsGatherIndex, static_cast<int16_t>(3), maskScale);
    MicroAPI::ShiftLefts(rmsGatherBase, rmsGatherBase, static_cast<int16_t>(3), maskScale);
    MicroAPI::ShiftRights(rmsGatherIndex, rmsGatherIndex, static_cast<int16_t>(2), maskScale);
    MicroAPI::Duplicate(rmsGatherOne, 1U);
    MicroAPI::And(rmsGatherIndex, rmsGatherIndex, rmsGatherOne, maskScale);
    MicroAPI::Add(rmsGatherIndex, rmsGatherBase, rmsGatherIndex, maskScale);

    __ubuf__ float *dataScratch = reinterpret_cast<__ubuf__ float *>(scratch);
    __ubuf__ uint32_t *maxScratch = reinterpret_cast<__ubuf__ uint32_t *>(
        scratch + static_cast<uint32_t>(MROPE_MX_ROW_BATCH_DATA_BYTES));
    __ubuf__ float *reciprocalScratch = reinterpret_cast<__ubuf__ float *>(
        scratch + static_cast<uint32_t>(MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES));
    __ubuf__ float *rmsSquaredScratch = reinterpret_cast<__ubuf__ float *>(
        scratch + static_cast<uint32_t>(MROPE_MX_ROW_BATCH_DATA_BYTES + MROPE_MX_ROW_BATCH_MAX_BYTES +
                                        MROPE_MX_ROW_BATCH_RECIP_BYTES));
    __ubuf__ uint8_t *outBytes = reinterpret_cast<__ubuf__ uint8_t *>(out);
    __ubuf__ uint8_t *paddedScaleBytes = reinterpret_cast<__ubuf__ uint8_t *>(outScale);
    const uint16_t tokensPerBatch = static_cast<uint16_t>(MROPE_MX_ROW_BATCH_ROWS / headSize);

    uint16_t batchTokenBegin = 0U;
    while (static_cast<uint16_t>(tokenSize - batchTokenBegin) >= tokensPerBatch) {
        const uint16_t batchTokenCount = tokensPerBatch;
        __ubuf__ bfloat16_t *batchInputBf16 =
            inputBf16 + static_cast<uint32_t>(batchTokenBegin) * inputTokenStride;
        __ubuf__ float *batchRawCosSin = rawCosSin +
            static_cast<uint32_t>(batchTokenBegin) * 3U * QKV_K_SCALE_D128_FULL_SIZE;
        __ubuf__ uint8_t *batchOutBytes = outBytes +
            static_cast<uint32_t>(batchTokenBegin) * headSize * QKV_K_SCALE_D128_FULL_SIZE;
        __ubuf__ uint8_t *batchPaddedScaleBytes = paddedScaleBytes +
            static_cast<uint32_t>(batchTokenBegin) * scaleTokenStrideWords * sizeof(uint32_t);
        ProcessKMropeMxD128RowBatch16(
            batchInputBf16, batchRawCosSin, batchOutBytes, batchPaddedScaleBytes,
            dataScratch, maxScratch, reciprocalScratch, rmsSquaredScratch, batchTokenCount, headSize,
            inputTokenStride, inputHeadStride, scaleTokenStrideWords, epsilon,
            mropeIndex, absMask, gammaLow, gammaHigh, rmsGatherIndex,
            mask64, maskFirst, mask8, maskFp8);
        batchTokenBegin = static_cast<uint16_t>(batchTokenBegin + batchTokenCount);
    }
}
__simd_vf__ inline void KRmsNormMropeMxD128VfImpl(
    __ubuf__ bfloat16_t *inputBf16, __ubuf__ float *gamma, __ubuf__ float *rawCosSin,
    __ubuf__ uint32_t *gatherIndex, __ubuf__ fp8_e4m3fn_t *out, __ubuf__ fp8_e8m0_t *outScale,
    __ubuf__ uint8_t *scratch, uint16_t tokenSize, uint16_t headSize, uint32_t inputTokenStride,
    uint32_t inputHeadStride, uint32_t scaleTokenStrideWords, float epsilon)
{
    MicroAPI::RegTensor<uint32_t> mropeIndex;
    MicroAPI::RegTensor<float> gammaLow;
    MicroAPI::RegTensor<float> gammaHigh;
    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::LoadAlign<uint32_t>(mropeIndex, gatherIndex);
    MicroAPI::LoadAlign<float>(gammaLow, gamma);
    MicroAPI::LoadAlign<float>(gammaHigh, gamma + QKV_K_SCALE_D128_HALF_SIZE);

    (void)scratch;

    MicroAPI::RegTensor<uint32_t> scaleGatherLowIndex;
    MicroAPI::RegTensor<uint32_t> scaleGatherHighIndex;
    MicroAPI::MaskReg maskLow32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg maskHigh32;
    MicroAPI::Not(maskHigh32, maskLow32, mask64);
    uint32_t scaleCountPerHead = 4U;
    MicroAPI::MaskReg mask4 = MicroAPI::UpdateMask<float>(scaleCountPerHead);
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)scaleGatherLowIndex, 0);
    MicroAPI::ShiftRights(scaleGatherLowIndex, scaleGatherLowIndex, static_cast<int16_t>(5), mask64);
    MicroAPI::ShiftLefts(scaleGatherLowIndex, scaleGatherLowIndex, static_cast<int16_t>(1), mask64);
    MicroAPI::Adds(scaleGatherHighIndex, scaleGatherLowIndex, 1U, mask64);
    for (uint16_t tokenIdx = 0U; tokenIdx < tokenSize; ++tokenIdx) {
        MicroAPI::RegTensor<float> gammaCosLow;
        MicroAPI::RegTensor<float> gammaCosHigh;
        MicroAPI::RegTensor<float> gammaSinLow;
        MicroAPI::RegTensor<float> gammaSinHigh;
        __ubuf__ float *rawToken =
            rawCosSin + static_cast<uint32_t>(tokenIdx) * 3U * QKV_K_SCALE_D128_FULL_SIZE;
        PrepareMropeGammaD128(gammaCosLow, gammaCosHigh, gammaSinLow, gammaSinHigh, gammaLow, gammaHigh,
                              mropeIndex, rawToken, mask64);
        for (uint16_t headIdx = 0U; headIdx < headSize; ++headIdx) {
            MicroAPI::RegTensor<float> outLowFp32;
            MicroAPI::RegTensor<float> outHighFp32;
            MicroAPI::AddrReg inputAddr =
                MicroAPI::CreateAddrReg<bfloat16_t>(tokenIdx, inputTokenStride, headIdx, inputHeadStride);
            LoadRmsNormMropeFp32D128(outLowFp32, outHighFp32, inputBf16, inputAddr, gammaCosLow, gammaCosHigh,
                                     gammaSinLow, gammaSinHigh, epsilon, mask64, maskFirst);
            MicroAPI::AddrReg outAddr = MicroAPI::CreateAddrReg<uint8_t>(
                tokenIdx, static_cast<uint32_t>(headSize) * QKV_K_SCALE_D128_FULL_SIZE,
                headIdx, QKV_K_SCALE_D128_FULL_SIZE);
            __ubuf__ uint8_t *scaleBytes = reinterpret_cast<__ubuf__ uint8_t *>(outScale) +
                (static_cast<uint32_t>(tokenIdx) * scaleTokenStrideWords + headIdx) * sizeof(uint32_t);
            MxQuantCublasD128(reinterpret_cast<__ubuf__ uint8_t *>(out), outAddr, scaleBytes,
                              outLowFp32, outHighFp32, scaleGatherLowIndex,
                              scaleGatherHighIndex, mask64, maskLow32, maskHigh32, mask4);
        }
    }
}

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_VF_H_
