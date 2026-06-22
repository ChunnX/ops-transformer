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
 * \file vf_basic_block_unaligned1024_update.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED1024_UPDATE_H
#define VF_BASIC_BLOCK_UNALIGNED1024_UPDATE_H

#include "vf_basic_block_utils.h"
#include "vf_basic_block_unaligned1024_common.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 16, uint32_t s2BaseSize = 512,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ void ProcessVec1UpdateGeneralImpl1024VF(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ T2 * expUb3, __ubuf__ T2 * expUb4, __ubuf__ T2 * expUb5, __ubuf__ T2 * expUb6,
    __ubuf__ T2 * expUb7, __ubuf__ T2 * expUb8, __ubuf__ OUTPUT_T * pseUb, __ubuf__ T * maxUb, __ubuf__ T * srcUb, __ubuf__ T * expMaxUb,
    __ubuf__ T * inMaxUb, __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpMaxUb, __ubuf__ T * tmpMaxUb2,
    __ubuf__ uint32_t * maskUb1, __ubuf__ uint32_t * maskUb2, __ubuf__ uint32_t * maskUb3, __ubuf__ uint32_t * maskUb4, __ubuf__ uint32_t * maskUb5,
    __ubuf__ uint32_t * maskUb6, __ubuf__ uint32_t * maskUb7, __ubuf__ uint32_t * maskUb8, __ubuf__ uint32_t * maskUb9, __ubuf__ uint32_t * maskUb10,
    __ubuf__ uint32_t * maskUb11, __ubuf__ uint32_t * maskUb12, __ubuf__ uint32_t * maskUb13, __ubuf__ uint32_t * maskUb14,
    __ubuf__ uint32_t * maskUb15, __ubuf__ uint32_t * maskUb16, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride,
    const uint32_t oriTailN1, const uint32_t oriTailN2, const uint32_t oriTailN3, const uint32_t oriTailN4, const uint32_t oriTailN5,
    const uint32_t oriTailN6, const uint32_t oriTailN7, const uint32_t oriTailN8, const uint32_t tailN1, const uint32_t tailN2, const uint32_t tailN3,
    const uint32_t tailN4, const uint32_t tailN5, const uint32_t tailN6, const uint32_t tailN7, const uint32_t tailN8, uint32_t pltOriTailN1,
    uint32_t pltOriTailN2, uint32_t pltOriTailN3, uint32_t pltOriTailN4, uint32_t pltOriTailN5, uint32_t pltOriTailN6, uint32_t pltOriTailN7, uint32_t pltOriTailN8,
    uint32_t pltTailN1, uint32_t pltTailN2, uint32_t pltTailN3, uint32_t pltTailN4, uint32_t pltTailN5, uint32_t pltTailN6, uint32_t pltTailN7, uint32_t pltTailN8,
    uint32_t pltN, float divValue, const uint16_t m, const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue)
{
    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);

    ProcessFirstPass1024<T, T2, OUTPUT_T, float, pseMode, hasAtten>(
        ureg_max, preg_all, preg_all_b16,
        tmpMaxUb, srcUb, pseUb, maskUb1, maskUb2, maskUb3, maskUb4,
        maskUb5, maskUb6, maskUb7, maskUb8, maskUb9, maskUb10,
        maskUb11, maskUb12, maskUb13, maskUb14, maskUb15, maskUb16,
        m, s2BaseSize, pseStride, scale, slopes, posShift, minValue,
        nPadding,
        pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4,
        pltOriTailN5, pltOriTailN6, pltOriTailN7, pltOriTailN8,
        pltTailN1, pltTailN2, pltTailN3, pltTailN4,
        pltTailN5, pltTailN6, pltTailN7, pltTailN8);

    ProcessTail1024Update<T>(ureg_max, tmpMaxUb, tmpMaxUb2, inMaxUb, preg_all);

    ProcessSecondPass1024<T, T2>(
        ureg_exp_sum, preg_all, preg_all_b16,
        tmpMaxUb2, expUb1, expUb2, expUb3, expUb4, expUb5, expUb6, expUb7, expUb8,
        tmpExpSumUb, srcUb, m, s2BaseSize, blockStride, repeatStride, preg_n_b16);
}

// noupdate, 512 < originN <= 1024
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 16, uint32_t s2BaseSize = 512,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1UpdateGeneralImpl1024(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<OUTPUT_T>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const T minValue, float keepProb)
{
    const uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    const uint32_t oriTailN1 = originN - floatRepSize * 8 < floatRepSize ? originN - floatRepSize * 8 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 9) <= 0 ? 0 : originN - floatRepSize * 9;
    const uint32_t oriTailN3 = static_cast<int32_t>(originN - floatRepSize * 10) <= 0 ? 0 : originN - floatRepSize * 10;
    const uint32_t oriTailN4 = static_cast<int32_t>(originN - floatRepSize * 11) <= 0 ? 0 : originN - floatRepSize * 11;
    const uint32_t oriTailN5 = static_cast<int32_t>(originN - floatRepSize * 12) <= 0 ? 0 : originN - floatRepSize * 12;
    const uint32_t oriTailN6 = static_cast<int32_t>(originN - floatRepSize * 13) <= 0 ? 0 : originN - floatRepSize * 13;
    const uint32_t oriTailN7 = static_cast<int32_t>(originN - floatRepSize * 14) <= 0 ? 0 : originN - floatRepSize * 14;
    const uint32_t oriTailN8 = static_cast<int32_t>(originN - floatRepSize * 15) <= 0 ? 0 : originN - floatRepSize * 15;

    const uint32_t tailN1 = s2BaseSize - floatRepSize * 8;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 9;
    const uint32_t tailN3 = s2BaseSize - floatRepSize * 10;
    const uint32_t tailN4 = s2BaseSize - floatRepSize * 11;
    const uint32_t tailN5 = s2BaseSize - floatRepSize * 12;
    const uint32_t tailN6 = s2BaseSize - floatRepSize * 13;
    const uint32_t tailN7 = s2BaseSize - floatRepSize * 14;
    const uint32_t tailN8 = s2BaseSize - floatRepSize * 15;

    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltOriTailN3 = oriTailN3;
    uint32_t pltOriTailN4 = oriTailN4;
    uint32_t pltOriTailN5 = oriTailN5;
    uint32_t pltOriTailN6 = oriTailN6;
    uint32_t pltOriTailN7 = oriTailN7;
    uint32_t pltOriTailN8 = oriTailN8;

    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    uint32_t pltTailN3 = tailN3;
    uint32_t pltTailN4 = tailN4;
    uint32_t pltTailN5 = tailN5;
    uint32_t pltTailN6 = tailN6;
    uint32_t pltTailN7 = tailN7;
    uint32_t pltTailN8 = tailN8;

    float divValue = 1.0f / keepProb;
    uint32_t pltN = s2BaseSize;

    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb3 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 2 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb4 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 3 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb5 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 4 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb6 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 5 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb7 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 6 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb8 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 7 * ((s1BaseSize >> 1) + 1) * (128);

    __ubuf__ OUTPUT_T * pseUb = (__ubuf__ OUTPUT_T*)pseTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;

    __ubuf__ uint32_t * maskUb1 = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ uint32_t * maskUb2 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize);
    __ubuf__ uint32_t * maskUb3 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 2);
    __ubuf__ uint32_t * maskUb4 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 3);
    __ubuf__ uint32_t * maskUb5 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 4);
    __ubuf__ uint32_t * maskUb6 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 5);
    __ubuf__ uint32_t * maskUb7 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 6);
    __ubuf__ uint32_t * maskUb8 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 7);
    __ubuf__ uint32_t * maskUb9 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 8);
    __ubuf__ uint32_t * maskUb10 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 9);
    __ubuf__ uint32_t * maskUb11 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 10);
    __ubuf__ uint32_t * maskUb12 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 11);
    __ubuf__ uint32_t * maskUb13 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 12);
    __ubuf__ uint32_t * maskUb14 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 13);
    __ubuf__ uint32_t * maskUb15 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 14);
    __ubuf__ uint32_t * maskUb16 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 15);

    ProcessVec1UpdateGeneralImpl1024VF<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
        expUb1, expUb2, expUb3, expUb4, expUb5, expUb6, expUb7, expUb8, pseUb, maxUb, srcUb, expMaxUb,
        inMaxUb, tmpExpSumUb, tmpMaxUb, tmpMaxUb2,
        maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7, maskUb8, maskUb9, maskUb10, maskUb11,
        maskUb12, maskUb13, maskUb14, maskUb15, maskUb16, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2,
        oriTailN3, oriTailN4, oriTailN5, oriTailN6, oriTailN7, oriTailN8, tailN1, tailN2, tailN3, tailN4, tailN5,
        tailN6, tailN7, tailN8, pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4, pltOriTailN5, pltOriTailN6,
        pltOriTailN7, pltOriTailN8, pltTailN1, pltTailN2, pltTailN3, pltTailN4, pltTailN5, pltTailN6, pltTailN7, pltTailN8,
        pltN, divValue, m, pseStride, slopes, posShift, scale, minValue);
}
} // namespace

#endif // VF_BASIC_BLOCK_UNALIGNED1024_UPDATE_H
