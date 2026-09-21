/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_mx_finalize_output_QBSA.h
 * \brief Fuse QBSA MX final V2 arithmetic, invalid-row handling and output conversion.
 */
#ifndef VF_MX_FINALIZE_OUTPUT_QBSA_H
#define VF_MX_FINALIZE_OUTPUT_QBSA_H

#include "kernel_tensor.h"
#include "vf_mx_cast_utils_QBSA.h"

namespace QbsaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

template <typename INPUT_T, typename OUTPUT_T, uint16_t SRC_D, bool HAS_ATTEN>
__simd_vf__ inline void MxLastDivCastOutputVF(__ubuf__ OUTPUT_T *dstUb, __ubuf__ float *curUb, __ubuf__ float *expSumUb,
                                              __ubuf__ float *maxUb, const uint16_t m, const float deScaleV)
{
    static_assert(SRC_D == 128U, "MX fused output only supports D=128");
    static_assert(IsSameType<OUTPUT_T, bfloat16_t>::value || IsSameType<OUTPUT_T, half>::value,
                  "MX fused output only supports 16-bit floating-point output");

    RegTensor<float> vregCurEven;
    RegTensor<float> vregCurOdd;
    RegTensor<float> vregExpSum;
    RegTensor<float> vregMax;
    RegTensor<float> vregMin;
    RegTensor<float> vregZero;
    RegTensor<OUTPUT_T> vregEvenOut;
    RegTensor<OUTPUT_T> vregOddOut;
    RegTensor<OUTPUT_T> vregPackedOut;

    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg pregZeroSum;
    MaskReg pregInvalidRow;

    Duplicate(vregZero, 0.0F);
    if constexpr (HAS_ATTEN) {
        constexpr uint32_t minBits = 0xFF7FFFFFU;
        const float minValue = *((float *)&minBits);
        Duplicate(vregMin, minValue);
    }

    // dst may alias cur: load the complete FP32 row before the compact 16-bit store.
    for (uint16_t i = 0U; i < m; ++i) {
        LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(vregCurEven, vregCurOdd, curUb + i * SRC_D);
        if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                      IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
            Muls(vregCurEven, vregCurEven, deScaleV, pregAll);
            Muls(vregCurOdd, vregCurOdd, deScaleV, pregAll);
        }

        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vregExpSum, expSumUb + i);
        Div(vregCurEven, vregCurEven, vregExpSum, pregAll);
        Div(vregCurOdd, vregCurOdd, vregExpSum, pregAll);
        Compare<float, CMPMODE::EQ>(pregZeroSum, vregExpSum, vregZero, pregAll);
        Select(vregCurEven, vregZero, vregCurEven, pregZeroSum);
        Select(vregCurOdd, vregZero, vregCurOdd, pregZeroSum);

        if constexpr (HAS_ATTEN) {
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vregMax, maxUb + i);
            Compare<float, CMPMODE::EQ>(pregInvalidRow, vregMax, vregMin, pregAll);
            Select(vregCurEven, vregZero, vregCurEven, pregInvalidRow);
            Select(vregCurOdd, vregZero, vregCurOdd, pregInvalidRow);
        }

        Cast<OUTPUT_T, float, castTraitRoundNoSatZero>(vregEvenOut, vregCurEven, pregAll);
        Cast<OUTPUT_T, float, castTraitRoundNoSatOne>(vregOddOut, vregCurOdd, pregAll);
        Or((RegTensor<uint16_t> &)vregPackedOut, (RegTensor<uint16_t> &)vregEvenOut, (RegTensor<uint16_t> &)vregOddOut,
           pregAllB16);
        StoreAlign<OUTPUT_T, Reg::StoreDist::DIST_NORM_B32>(dstUb + i * SRC_D, vregPackedOut, pregAllB16);
    }
}

template <typename INPUT_T, typename OUTPUT_T, uint16_t SRC_D, bool HAS_ATTEN>
__aicore__ inline void MxLastDivCastOutput(const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<float> &curTensor,
                                           const LocalTensor<float> &expSumTensor, const LocalTensor<float> &maxTensor,
                                           const uint16_t m, const float deScaleV)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ float *curUb = (__ubuf__ float *)curTensor.GetPhyAddr();
    __ubuf__ float *expSumUb = (__ubuf__ float *)expSumTensor.GetPhyAddr();
    __ubuf__ float *maxUb = (__ubuf__ float *)maxTensor.GetPhyAddr();
    MxLastDivCastOutputVF<INPUT_T, OUTPUT_T, SRC_D, HAS_ATTEN>(dstUb, curUb, expSumUb, maxUb, m, deScaleV);
}

template <typename INPUT_T, typename OUTPUT_T, uint16_t SRC_D, bool IS_UPDATE_PRE, bool HAS_ATTEN>
__simd_vf__ inline void MxFlashUpdateLastCastOutputVF(__ubuf__ OUTPUT_T *dstUb, __ubuf__ float *curUb,
                                                      __ubuf__ float *preUb, __ubuf__ float *expMaxUb,
                                                      __ubuf__ float *expSumUb, __ubuf__ float *maxUb, const uint16_t m,
                                                      const float deScaleV, const float deScaleVPre)
{
    static_assert(SRC_D == 128U, "MX fused output only supports D=128");
    static_assert(IsSameType<OUTPUT_T, bfloat16_t>::value || IsSameType<OUTPUT_T, half>::value,
                  "MX fused output only supports 16-bit floating-point output");

    RegTensor<float> vregCurEven;
    RegTensor<float> vregCurOdd;
    RegTensor<float> vregPreEven;
    RegTensor<float> vregPreOdd;
    RegTensor<float> vregExpMax;
    RegTensor<float> vregExpSum;
    RegTensor<float> vregMax;
    RegTensor<float> vregMin;
    RegTensor<float> vregZero;
    RegTensor<OUTPUT_T> vregEvenOut;
    RegTensor<OUTPUT_T> vregOddOut;
    RegTensor<OUTPUT_T> vregPackedOut;

    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg pregZeroSum;
    MaskReg pregInvalidRow;

    Duplicate(vregZero, 0.0F);
    if constexpr (HAS_ATTEN) {
        constexpr uint32_t minBits = 0xFF7FFFFFU;
        const float minValue = *((float *)&minBits);
        Duplicate(vregMin, minValue);
    }

    // dst may alias pre: load the complete FP32 row before the compact 16-bit store.
    for (uint16_t i = 0U; i < m; ++i) {
        LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(vregCurEven, vregCurOdd, curUb + i * SRC_D);
        LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(vregPreEven, vregPreOdd, preUb + i * SRC_D);
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vregExpMax, expMaxUb + i);
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vregExpSum, expSumUb + i);

        Mul(vregPreEven, vregExpMax, vregPreEven, pregAll);
        Mul(vregPreOdd, vregExpMax, vregPreOdd, pregAll);
        if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                      IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
            Muls(vregCurEven, vregCurEven, deScaleV, pregAll);
            Muls(vregCurOdd, vregCurOdd, deScaleV, pregAll);
            if constexpr (IS_UPDATE_PRE) {
                Muls(vregPreEven, vregPreEven, deScaleVPre, pregAll);
                Muls(vregPreOdd, vregPreOdd, deScaleVPre, pregAll);
            }
        }
        Add(vregCurEven, vregPreEven, vregCurEven, pregAll);
        Add(vregCurOdd, vregPreOdd, vregCurOdd, pregAll);
        Div(vregCurEven, vregCurEven, vregExpSum, pregAll);
        Div(vregCurOdd, vregCurOdd, vregExpSum, pregAll);
        Compare<float, CMPMODE::EQ>(pregZeroSum, vregExpSum, vregZero, pregAll);
        Select(vregCurEven, vregZero, vregCurEven, pregZeroSum);
        Select(vregCurOdd, vregZero, vregCurOdd, pregZeroSum);

        if constexpr (HAS_ATTEN) {
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vregMax, maxUb + i);
            Compare<float, CMPMODE::EQ>(pregInvalidRow, vregMax, vregMin, pregAll);
            Select(vregCurEven, vregZero, vregCurEven, pregInvalidRow);
            Select(vregCurOdd, vregZero, vregCurOdd, pregInvalidRow);
        }

        Cast<OUTPUT_T, float, castTraitRoundNoSatZero>(vregEvenOut, vregCurEven, pregAll);
        Cast<OUTPUT_T, float, castTraitRoundNoSatOne>(vregOddOut, vregCurOdd, pregAll);
        Or((RegTensor<uint16_t> &)vregPackedOut, (RegTensor<uint16_t> &)vregEvenOut, (RegTensor<uint16_t> &)vregOddOut,
           pregAllB16);
        StoreAlign<OUTPUT_T, Reg::StoreDist::DIST_NORM_B32>(dstUb + i * SRC_D, vregPackedOut, pregAllB16);
    }
}

template <typename INPUT_T, typename OUTPUT_T, uint16_t SRC_D, bool IS_UPDATE_PRE, bool HAS_ATTEN>
__aicore__ inline void MxFlashUpdateLastCastOutput(
    const LocalTensor<OUTPUT_T> &dstTensor, const LocalTensor<float> &curTensor, const LocalTensor<float> &preTensor,
    const LocalTensor<float> &expMaxTensor, const LocalTensor<float> &expSumTensor, const LocalTensor<float> &maxTensor,
    const uint16_t m, const float deScaleV, const float deScaleVPre)
{
    __ubuf__ OUTPUT_T *dstUb = (__ubuf__ OUTPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ float *curUb = (__ubuf__ float *)curTensor.GetPhyAddr();
    __ubuf__ float *preUb = (__ubuf__ float *)preTensor.GetPhyAddr();
    __ubuf__ float *expMaxUb = (__ubuf__ float *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *expSumUb = (__ubuf__ float *)expSumTensor.GetPhyAddr();
    __ubuf__ float *maxUb = (__ubuf__ float *)maxTensor.GetPhyAddr();
    MxFlashUpdateLastCastOutputVF<INPUT_T, OUTPUT_T, SRC_D, IS_UPDATE_PRE, HAS_ATTEN>(
        dstUb, curUb, preUb, expMaxUb, expSumUb, maxUb, m, deScaleV, deScaleVPre);
}

} // namespace QbsaVectorApi

#endif // VF_MX_FINALIZE_OUTPUT_QBSA_H
