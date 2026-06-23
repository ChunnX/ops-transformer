/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_basic_block_unaligned512_no_update.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED512_NO_UPDATE_H
#define VF_BASIC_BLOCK_UNALIGNED512_NO_UPDATE_H

#include "vf_basic_block_utils.h"
#include "vf_basic_block_unaligned512_common.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ void ProcessVec1NoUpdateGeneralImpl512VF(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ T2 * expUb3, __ubuf__ T2 * expUb4, __ubuf__ OUTPUT_T * pseUb,
    __ubuf__ T * expSumUb, __ubuf__ T * maxUb, __ubuf__ T * maxUbStart, __ubuf__ T * srcUb, __ubuf__ uint32_t * maskUb1, __ubuf__ uint32_t * maskUb2,
    __ubuf__ uint32_t * maskUb3, __ubuf__ uint32_t * maskUb4, __ubuf__ uint32_t * maskUb5, __ubuf__ uint32_t * maskUb6,
    __ubuf__ uint32_t * maskUb7, __ubuf__ uint32_t * maskUb8, const uint32_t nPadding,
    const uint32_t blockStride, const uint32_t repeatStride, const uint32_t oriTailN1, const uint32_t oriTailN2, const uint32_t oriTailN3,
    const uint32_t oriTailN4, const uint32_t tailN1, const uint32_t tailN2, const uint32_t tailN3, const uint32_t tailN4,
    uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltOriTailN3, uint32_t pltOriTailN4,
    uint32_t pltTailN1, uint32_t pltTailN2, uint32_t pltTailN3, uint32_t pltTailN4, float divValue, const uint16_t m,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue)
{
    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    ProcessFirstPass512<T, T2, OUTPUT_T, T, pseMode, hasAtten>(
        ureg_max, preg_all, preg_all_b16, maxUb, srcUb, pseUb,
        maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7, maskUb8,
        m, s2BaseSize, pseStride, scale, slopes, posShift, minValue, nPadding,
        pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4,
        pltTailN1, pltTailN2, pltTailN3, pltTailN4);

    ProcessTail512NoUpdate<T>(ureg_max, maxUb);

    ProcessSecondPass512<T, T2>(
        ureg_exp_sum, preg_all, preg_all_b16, maxUbStart,
        expUb1, expUb2, expUb3, expUb4,
        expSumUb, srcUb, m, s2BaseSize, blockStride, repeatStride, preg_all_b16);

    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
}

template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ __no_simd_vf_fusion__ void ProcessVec1NoUpdateGeneralImpl512VFStage1(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ T2 * expUb3, __ubuf__ T2 * expUb4, __ubuf__ OUTPUT_T * pseUb,
    __ubuf__ T * expSumUb, __ubuf__ T * maxUb, __ubuf__ T * maxUbStart, __ubuf__ T * srcUb, __ubuf__ uint32_t * maskUb1, __ubuf__ uint32_t * maskUb2,
    __ubuf__ uint32_t * maskUb3, __ubuf__ uint32_t * maskUb4, __ubuf__ uint32_t * maskUb5, __ubuf__ uint32_t * maskUb6,
    __ubuf__ uint32_t * maskUb7, __ubuf__ uint32_t * maskUb8, const uint32_t nPadding,
    const uint32_t blockStride, const uint32_t repeatStride, const uint32_t oriTailN1, const uint32_t oriTailN2, const uint32_t oriTailN3,
    const uint32_t oriTailN4, const uint32_t tailN1, const uint32_t tailN2, const uint32_t tailN3, const uint32_t tailN4,
    uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltOriTailN3, uint32_t pltOriTailN4,
    uint32_t pltTailN1, uint32_t pltTailN2, uint32_t pltTailN3, uint32_t pltTailN4, float divValue, const uint16_t m,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue)
{
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_ori_tail_n3 = UpdateMask<T>(pltOriTailN3);
    MaskReg preg_ori_tail_n4 = UpdateMask<T>(pltOriTailN4);

    ProcessStage1Pass512<T, T2, OUTPUT_T, pseMode, hasAtten>(
        srcUb, pseUb, maskUb7, maskUb8,
        m, s2BaseSize, pseStride, scale, slopes, posShift, minValue,
        nPadding, preg_all, preg_all_b16, preg_ori_tail_n3, preg_ori_tail_n4);
}

template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ __no_simd_vf_fusion__ void ProcessVec1NoUpdateGeneralImpl512VFStage2(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ T2 * expUb3, __ubuf__ T2 * expUb4, __ubuf__ OUTPUT_T * pseUb,
    __ubuf__ T * expSumUb, __ubuf__ T * maxUb, __ubuf__ T * maxUbStart, __ubuf__ T * srcUb, __ubuf__ uint32_t * maskUb1, __ubuf__ uint32_t * maskUb2,
    __ubuf__ uint32_t * maskUb3, __ubuf__ uint32_t * maskUb4, __ubuf__ uint32_t * maskUb5, __ubuf__ uint32_t * maskUb6,
    __ubuf__ uint32_t * maskUb7, __ubuf__ uint32_t * maskUb8, const uint32_t nPadding,
    const uint32_t blockStride, const uint32_t repeatStride, const uint32_t oriTailN1, const uint32_t oriTailN2, const uint32_t oriTailN3,
    const uint32_t oriTailN4, const uint32_t tailN1, const uint32_t tailN2, const uint32_t tailN3, const uint32_t tailN4,
    uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltOriTailN3, uint32_t pltOriTailN4,
    uint32_t pltTailN1, uint32_t pltTailN2, uint32_t pltTailN3, uint32_t pltTailN4, float divValue, const uint16_t m,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue)
{
    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    ProcessFirstPass512Stage2<T, T2, OUTPUT_T, T, pseMode, hasAtten>(
        ureg_max, preg_all, preg_all_b16, maxUb, srcUb, pseUb,
        maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6,
        maskUb7, maskUb8,
        m, s2BaseSize, pseStride, scale, slopes, posShift, minValue, nPadding,
        pltOriTailN1, pltOriTailN2);

    StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), ureg_max, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    ProcessSecondPass512Stage2<T, T2>(
        ureg_exp_sum, preg_all, preg_all_b16, maxUbStart,
        expUb1, expUb2, expUb3, expUb4,
        expSumUb, srcUb, m, s2BaseSize, blockStride, repeatStride, preg_all_b16);

    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
}

__aicore__ inline void SetupAicoreParams512(
    const uint32_t s1BaseSize, const uint32_t s2BaseSize,
    uint32_t &blockStride, uint32_t &repeatStride,
    const uint32_t originN,
    uint32_t &oriTailN1, uint32_t &oriTailN2, uint32_t &oriTailN3, uint32_t &oriTailN4,
    uint32_t &tailN1, uint32_t &tailN2, uint32_t &tailN3, uint32_t &tailN4,
    uint32_t &pltOriTailN1, uint32_t &pltOriTailN2, uint32_t &pltOriTailN3, uint32_t &pltOriTailN4,
    uint32_t &pltTailN1, uint32_t &pltTailN2, uint32_t &pltTailN3, uint32_t &pltTailN4,
    uint32_t &nPadding, float &divValue, float keepProb)
{
    blockStride = s1BaseSize >> 1 | 0x1;
    repeatStride = 1;
    nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    oriTailN1 = originN - floatRepSize * 4 < floatRepSize ? originN - floatRepSize * 4 : floatRepSize;
    oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 5) <= 0 ? 0 : originN - floatRepSize * 5;
    oriTailN3 = static_cast<int32_t>(originN - floatRepSize * 6) <= 0 ? 0 : originN - floatRepSize * 6;
    oriTailN4 = static_cast<int32_t>(originN - floatRepSize * 7) <= 0 ? 0 : originN - floatRepSize * 7;
    tailN1 = s2BaseSize - floatRepSize * 4;
    tailN2 = s2BaseSize - floatRepSize * 5;
    tailN3 = s2BaseSize - floatRepSize * 6;
    tailN4 = s2BaseSize - floatRepSize * 7;
    pltOriTailN1 = oriTailN1;
    pltOriTailN2 = oriTailN2;
    pltOriTailN3 = oriTailN3;
    pltOriTailN4 = oriTailN4;
    pltTailN1 = tailN1;
    pltTailN2 = tailN2;
    pltTailN3 = tailN3;
    pltTailN4 = tailN4;
    divValue = 1.0f / keepProb;
}

__aicore__ inline void SetupBuffers512NoUpdate(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<OUTPUT_T>& pseTensor,
    uint32_t s1BaseSize,
    __ubuf__ T2 *&expUb1, __ubuf__ T2 *&expUb2, __ubuf__ T2 *&expUb3, __ubuf__ T2 *&expUb4,
    __ubuf__ T *&srcUb, __ubuf__ T *&expSumUb, __ubuf__ T *&maxUb, __ubuf__ T *&maxUbStart,
    __ubuf__ OUTPUT_T *&pseUb,
    __ubuf__ uint32_t *&maskUb1, __ubuf__ uint32_t *&maskUb2, __ubuf__ uint32_t *&maskUb3, __ubuf__ uint32_t *&maskUb4,
    __ubuf__ uint32_t *&maskUb5, __ubuf__ uint32_t *&maskUb6, __ubuf__ uint32_t *&maskUb7, __ubuf__ uint32_t *&maskUb8)
{
    expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    expUb2 = expUb1 + ((s1BaseSize >> 1) + 1) * (128);
    expUb3 = expUb1 + 2 * ((s1BaseSize >> 1) + 1) * (128);
    expUb4 = expUb1 + 3 * ((s1BaseSize >> 1) + 1) * (128);
    srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    maxUbStart = maxUb;
    pseUb = (__ubuf__ OUTPUT_T*)pseTensor.GetPhyAddr();
    __ubuf__ uint32_t * maskBase = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    maskUb1 = maskBase; maskUb2 = maskBase + floatRepSize;
    maskUb3 = maskBase + floatRepSize * 2; maskUb4 = maskBase + floatRepSize * 3;
    maskUb5 = maskBase + floatRepSize * 4; maskUb6 = maskBase + floatRepSize * 5;
    maskUb7 = maskBase + floatRepSize * 6; maskUb8 = maskBase + floatRepSize * 7;
}

// 256 < Orignin N <=512
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1NoUpdateGeneralImpl512(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<OUTPUT_T>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue, float keepProb)
{
    __ubuf__ T2 * expUb1, *expUb2, *expUb3, *expUb4;
    __ubuf__ T * srcUb, *expSumUb, *maxUb, *maxUbStart;
    __ubuf__ OUTPUT_T * pseUb;
    __ubuf__ uint32_t * maskUb1, *maskUb2, *maskUb3, *maskUb4, *maskUb5, *maskUb6, *maskUb7, *maskUb8;
    SetupBuffers512NoUpdate(dstTensor, expSumTensor, maxTensor, srcTensor, maskTensor, pseTensor, s1BaseSize,
        expUb1, expUb2, expUb3, expUb4, srcUb, expSumUb, maxUb, maxUbStart, pseUb,
        maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7, maskUb8);

    uint32_t blockStride, repeatStride, nPadding;
    uint32_t oriTailN1, oriTailN2, oriTailN3, oriTailN4, tailN1, tailN2, tailN3, tailN4;
    uint32_t pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4;
    uint32_t pltTailN1, pltTailN2, pltTailN3, pltTailN4;
    float divValue;
    SetupAicoreParams512(s1BaseSize, s2BaseSize,
        blockStride, repeatStride, originN,
        oriTailN1, oriTailN2, oriTailN3, oriTailN4,
        tailN1, tailN2, tailN3, tailN4,
        pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4,
        pltTailN1, pltTailN2, pltTailN3, pltTailN4,
        nPadding, divValue, keepProb);

    if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE && hasAtten == 1) {
        ProcessVec1NoUpdateGeneralImpl512VFStage1<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
            expUb1, expUb2, expUb3, expUb4, pseUb, expSumUb, maxUb, maxUbStart,
            srcUb, maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6,
            maskUb7, maskUb8, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2, oriTailN3, oriTailN4,
            tailN1, tailN2, tailN3, tailN4,
            pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4,
            pltTailN1, pltTailN2, pltTailN3, pltTailN4,
            divValue, m, pseStride, slopes, posShift, scale, minValue);
        ProcessVec1NoUpdateGeneralImpl512VFStage2<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
            expUb1, expUb2, expUb3, expUb4, pseUb, expSumUb, maxUb, maxUbStart,
            srcUb, maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6,
            maskUb7, maskUb8, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2, oriTailN3, oriTailN4,
            tailN1, tailN2, tailN3, tailN4,
            pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4,
            pltTailN1, pltTailN2, pltTailN3, pltTailN4,
            divValue, m, pseStride, slopes, posShift, scale, minValue);
    } else {
        ProcessVec1NoUpdateGeneralImpl512VF<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
            expUb1, expUb2, expUb3, expUb4, pseUb, expSumUb, maxUb, maxUbStart,
            srcUb, maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6,
            maskUb7, maskUb8, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2, oriTailN3, oriTailN4,
            tailN1, tailN2, tailN3, tailN4,
            pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4,
            pltTailN1, pltTailN2, pltTailN3, pltTailN4,
            divValue, m, pseStride, slopes, posShift, scale, minValue);
    }
}
} // namespace

#endif // VF_BASIC_BLOCK_UNALIGNED512_NO_UPDATE_H
