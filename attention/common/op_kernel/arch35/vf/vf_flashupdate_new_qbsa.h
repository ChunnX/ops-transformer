/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_flashupdate_new_qbsa.h
 * \brief QBSA empty-row normalization and LSE, with common FlashUpdate reuse.
 */
#ifndef MY_FLASH_UPDATE_NEW_QBSA_INTERFACE_H
#define MY_FLASH_UPDATE_NEW_QBSA_INTERFACE_H

#include "vf_basic_block_utils.h"
#include "vf_flashupdate_new.h"

namespace QbsaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

// Preserve the QBSA interface while reusing the common basic/general kernels.
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, bool isUpdatePre>
__aicore__ inline void FlashUpdateNew(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                      const LocalTensor<T> &preTensor, const LocalTensor<T> &expMaxTensor,
                                      const uint16_t m, const uint16_t d, const float deScaleV, const float deScaleVPre)
{
    // isMlaFullQuant=false never loads rowMaxTensor. Supply the valid expMaxTensor
    // because the common wrapper still obtains the unused tensor's address.
    FaVectorApi::FlashUpdateNew<T, INPUT_T, OUTPUT_T, srcD, isUpdatePre, false>(
        dstTensor, curTensor, preTensor, expMaxTensor, expMaxTensor, m, d, deScaleV, deScaleVPre);
}

// Reuse the common normalization, then restore QBSA's sum-zero output policy.
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, bool isUpdatePre,
          bool zeroWhenSumZero = false>
__aicore__ inline void FlashUpdateLastNew(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                          const LocalTensor<T> &preTensor, const LocalTensor<T> &expMaxTensor,
                                          const LocalTensor<T> &expSumTensor, uint16_t m, uint16_t d,
                                          const float deScaleV, const float deScaleVPre)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateLast, T must be float");
    // isMlaFullQuant=false does not load rowMaxTensor; expMaxTensor supplies its address.
    FaVectorApi::FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, srcD, isUpdatePre, false>(
        dstTensor, curTensor, preTensor, expMaxTensor, expMaxTensor, expSumTensor, m, d, deScaleV, deScaleVPre);
    if constexpr (zeroWhenSumZero) {
        PipeBarrier<PIPE_V>();
        constexpr uint16_t floatRepSize = 64;
        const uint16_t normalizedD = srcD % floatRepSize == 0 ? srcD : d;
        // Compare the compact row sums against +0, rather than the default max sentinel.
        FaVectorApi::RowInvalidUpdateVF<T>(dstTensor, expSumTensor, m, normalizedD, d, 0U);
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD>
__aicore__ inline void LastDivNew(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                  const LocalTensor<T> &expSumTensor, const uint16_t m, const uint16_t d,
                                  const float deScaleV)
{
    static_assert(IsSameType<T, float>::value, "VF LastDivNew, T must be float");
    FaVectorApi::LastDivNew<T, INPUT_T, OUTPUT_T, srcD, false>(dstTensor, curTensor, expSumTensor, m, d, deScaleV);
    PipeBarrier<PIPE_V>();
    // LastDivNew processes complete float registers only; preserve its untouched tail.
    constexpr uint16_t floatRepSize = 64;
    const uint16_t normalizedD = d / floatRepSize * floatRepSize;
    FaVectorApi::RowInvalidUpdateVF<T>(dstTensor, expSumTensor, m, normalizedD, d, 0U);
}

template <typename T, bool negativeInfWhenEmpty = false>
__simd_vf__ inline void ComputeLseOutputVF(__ubuf__ T *srcSumUb, __ubuf__ T *srcMaxUb, __ubuf__ T *dstUb,
                                           const uint32_t dealCount)
{
    Reg::RegTensor<T> vregSum;
    Reg::RegTensor<T> vregMax;
    Reg::RegTensor<T> vregRes;
    Reg::RegTensor<T> vregResFinal;
    Reg::RegTensor<float> vregMinValue;
    Reg::RegTensor<float> vregInfValue;
    Reg::RegTensor<float> vregZero;
    Reg::MaskReg pregCompare;
    Reg::MaskReg pregZeroSum;
    constexpr uint32_t dealRows = 8;
    constexpr uint32_t floatRepSize = 64; // 64: 一个寄存器存64个float
    constexpr float infValue = 3e+99;     // 3e+99 for float inf
    constexpr float negInfValue = -3e+99; // -3e+99 for float -inf
    constexpr uint32_t tmpMin = 0xFF7FFFFF;
    float minValue = *((float *)&tmpMin);
    uint16_t updateLoops = dealCount / dealRows;
    uint16_t tailLSize = (dealCount % dealRows) << 3;
    uint32_t pltTail = static_cast<uint32_t>(tailLSize);

    Reg::MaskReg pregAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregTail = Reg::UpdateMask<T>(pltTail);
    Reg::Duplicate<float, float>(vregMinValue, minValue);
    if constexpr (negativeInfWhenEmpty) {
        Reg::Duplicate<float, float>(vregInfValue, negInfValue);
    } else {
        // 保持原 FP8 实例化的寄存器初始化与指令路径不变。
        Reg::Duplicate<float, float>(vregInfValue, infValue);
    }
    Reg::Duplicate<float, float>(vregZero, 0.0f);

    for (uint16_t i = 0; i < updateLoops; ++i) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_E2B_B32>(vregSum, srcSumUb + (i * dealRows));
        Reg::LoadAlign<T, Reg::LoadDist::DIST_E2B_B32>(vregMax, srcMaxUb + (i * dealRows));

        Reg::Log<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregSum, pregAll);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregAll);

        Reg::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregAll);
        if constexpr (negativeInfWhenEmpty) {
            Reg::Select<T>(vregResFinal, vregInfValue, vregRes, pregCompare);
        } else {
            Reg::Select<T>(vregResFinal, vregMinValue, vregRes, pregCompare);
        }

        // sum == 0 表示空 softmax 行：MX 输出 -Inf，其他路径保留 -FLT_MAX。
        Reg::Compare<float, CMPMODE::EQ>(pregZeroSum, vregSum, vregZero, pregAll);
        if constexpr (negativeInfWhenEmpty) {
            Reg::Select<T>(vregResFinal, vregInfValue, vregResFinal, pregZeroSum);
        } else {
            Reg::Select<T>(vregResFinal, vregMinValue, vregResFinal, pregZeroSum);
        }

        Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(dstUb + (i * floatRepSize), vregResFinal, pregAll);
    }

    if (tailLSize != 0) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_E2B_B32>(vregSum, srcSumUb + dealRows * updateLoops);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_E2B_B32>(vregMax, srcMaxUb + dealRows * updateLoops);

        Reg::Log<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregSum, pregTail);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregTail);

        Reg::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregTail);
        if constexpr (negativeInfWhenEmpty) {
            Reg::Select<T>(vregResFinal, vregInfValue, vregRes, pregCompare);
        } else {
            Reg::Select<T>(vregResFinal, vregMinValue, vregRes, pregCompare);
        }

        // sum == 0 表示空 softmax 行：MX 输出 -Inf，其他路径保留 -FLT_MAX。
        Reg::Compare<float, CMPMODE::EQ>(pregZeroSum, vregSum, vregZero, pregTail);
        if constexpr (negativeInfWhenEmpty) {
            Reg::Select<T>(vregResFinal, vregInfValue, vregResFinal, pregZeroSum);
        } else {
            Reg::Select<T>(vregResFinal, vregMinValue, vregResFinal, pregZeroSum);
        }

        Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(dstUb + floatRepSize * updateLoops, vregResFinal, pregTail);
    }
}

template <typename T, bool negativeInfWhenEmpty = false>
__aicore__ inline void ComputeLseOutputVF(const LocalTensor<T> &dstTensor, const LocalTensor<T> &softmaxSumTensor,
                                          const LocalTensor<T> &softmaxMaxTensor, uint32_t dealCount)
{
    __ubuf__ T *srcSumUb = (__ubuf__ T *)softmaxSumTensor.GetPhyAddr();
    __ubuf__ T *srcMaxUb = (__ubuf__ T *)softmaxMaxTensor.GetPhyAddr();
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();

    ComputeLseOutputVF<T, negativeInfWhenEmpty>(srcSumUb, srcMaxUb, dstUb, dealCount);
}

} // namespace QbsaVectorApi

#endif // MY_FLASH_UPDATE_NEW_QBSA_INTERFACE_H
