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
 * \file vf_flashupdate_new_QBSA.h
 * \brief QBSA empty-row normalization and LSE, with common FlashUpdate reuse.
 */
#ifndef MY_FLASH_UPDATE_NEW_QBSA_INTERFACE_H
#define MY_FLASH_UPDATE_NEW_QBSA_INTERFACE_H

#include "vf_basic_block_utils.h"
#include "vf_flashupdate_new.h"

namespace QbsaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;
using FaVectorApi::REDUCE_SIZE;

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

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre,
          bool zeroWhenSumZero>
__simd_vf__ inline void FlashUpdateLastBasicVF(__ubuf__ float *dstUb, __ubuf__ float *curUb, __ubuf__ float *preUb,
                                               __ubuf__ float *expMaxUb, __ubuf__ float *expSumUb, const uint16_t m,
                                               const uint16_t d, const float deScaleV, const float deScaleVPre)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;
    RegTensor<float> vreg_div;
    RegTensor<half> vreg_cast;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_zero;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_zero_sum;
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    constexpr float fp8e4m3MaxValueRec = 1 / 448.0f;
    constexpr float int8MaxValueRec = 1 / 127.0f;
    constexpr float hifp8MaxValueRec = 1 / 32768.0f;

    if constexpr (zeroWhenSumZero) {
        Reg::Duplicate<float, float>(vreg_zero, 0.0f);
    }

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);
        LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * reduceSize);
        if constexpr (zeroWhenSumZero) {
            Reg::Compare<float, CMPMODE::EQ>(preg_zero_sum, vreg_exp_sum, vreg_zero, preg_all);
        }
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                          IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_all);
            if constexpr (zeroWhenSumZero) {
                Select(vreg_div, vreg_zero, vreg_div, preg_zero_sum);
            }
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div,
                                                         preg_all);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre,
          bool zeroWhenSumZero>
__aicore__ inline void FlashUpdateLastBasic(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                            const LocalTensor<T> &preTensor, const LocalTensor<T> &expMaxTensor,
                                            const LocalTensor<T> &expSumTensor, const uint16_t m, const uint16_t d,
                                            const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ float *curUb = (__ubuf__ T *)curTensor.GetPhyAddr();
    __ubuf__ float *preUb = (__ubuf__ T *)preTensor.GetPhyAddr();
    __ubuf__ float *expMaxUb = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *expSumUb = (__ubuf__ T *)expSumTensor.GetPhyAddr();

    FlashUpdateLastBasicVF<T, INPUT_T, OUTPUT_T, srcD, reduceSize, isUpdatePre, zeroWhenSumZero>(
        dstUb, curUb, preUb, expMaxUb, expSumUb, m, d, deScaleV, deScaleVPre);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t reduceSize, bool isUpdatePre, bool zeroWhenSumZero>
__simd_vf__ inline void FlashUpdateLastGeneralVF(__ubuf__ float *dstUb, __ubuf__ float *curUb, __ubuf__ float *preUb,
                                                 __ubuf__ float *expMaxUb, __ubuf__ float *expSumUb, const uint16_t m,
                                                 const uint16_t d, const float deScaleV, const float deScaleVPre,
                                                 const uint32_t pltTailD, const uint16_t hasTail)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;
    RegTensor<float> vreg_div;
    RegTensor<half> vreg_cast;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_zero;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_zero_sum;
    uint32_t tmpTailD = pltTailD;
    MaskReg preg_tail_d = UpdateMask<float>(tmpTailD);
    constexpr uint16_t floatRepSize = 64;
    uint16_t dLoops = d / floatRepSize;

    if constexpr (zeroWhenSumZero) {
        Reg::Duplicate<float, float>(vreg_zero, 0.0f);
    }

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);
        LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * reduceSize);
        if constexpr (zeroWhenSumZero) {
            Reg::Compare<float, CMPMODE::EQ>(preg_zero_sum, vreg_exp_sum, vreg_zero, preg_all);
        }
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);

            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                          IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_all);
            if constexpr (zeroWhenSumZero) {
                Select(vreg_div, vreg_zero, vreg_div, preg_zero_sum);
            }

            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div,
                                                         preg_all);
        }

        for (uint16_t t = 0; t < hasTail; ++t) {
            LoadAlign(vreg_input_pre, preUb + i * d + dLoops * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + dLoops * floatRepSize);
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_tail_d);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                          IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_tail_d);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_tail_d);
            if constexpr (zeroWhenSumZero) {
                Select(vreg_div, vreg_zero, vreg_div, preg_zero_sum);
            }

            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)dstUb + i * d + dLoops * floatRepSize, vreg_div,
                                                         preg_tail_d);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t reduceSize, bool isUpdatePre, bool zeroWhenSumZero>
__aicore__ inline void FlashUpdateLastGeneral(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                              const LocalTensor<T> &preTensor, const LocalTensor<T> &expMaxTensor,
                                              const LocalTensor<T> &expSumTensor, const uint16_t m, const uint16_t d,
                                              const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ float *curUb = (__ubuf__ T *)curTensor.GetPhyAddr();
    __ubuf__ float *preUb = (__ubuf__ T *)preTensor.GetPhyAddr();
    __ubuf__ float *expMaxUb = (__ubuf__ T *)expMaxTensor.GetPhyAddr();
    __ubuf__ float *expSumUb = (__ubuf__ T *)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = tailD;

    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    FlashUpdateLastGeneralVF<T, INPUT_T, OUTPUT_T, reduceSize, isUpdatePre, zeroWhenSumZero>(
        dstUb, curUb, preUb, expMaxUb, expSumUb, m, d, deScaleV, deScaleVPre, pltTailD, hasTail);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, bool isUpdatePre,
          bool zeroWhenSumZero = false>
__aicore__ inline void FlashUpdateLastNew(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                          const LocalTensor<T> &preTensor, const LocalTensor<T> &expMaxTensor,
                                          const LocalTensor<T> &expSumTensor, uint16_t m, uint16_t d,
                                          const float deScaleV, const float deScaleVPre)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateLast, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if constexpr (srcD % floatRepSize == 0) {
        FlashUpdateLastBasic<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE, isUpdatePre, zeroWhenSumZero>(
            dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, m, d, deScaleV, deScaleVPre);
    } else {
        FlashUpdateLastGeneral<T, INPUT_T, OUTPUT_T, REDUCE_SIZE, isUpdatePre, zeroWhenSumZero>(
            dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, m, d, deScaleV, deScaleVPre);
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD>
__simd_vf__ inline void LastDivNewVF(__ubuf__ float *dstUb, __ubuf__ float *curUb, __ubuf__ float *expSumUb,
                                     const uint16_t m, const uint16_t d, const float deScaleV)
{
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_zero;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_zero_sum;
    constexpr uint16_t floatRepSize = 64;
    const uint16_t dLoops = d >> 6;
    constexpr float fp8e4m3MaxValueRec = 1 / 448.0f;
    constexpr float int8MaxValueRec = 1 / 127.0f;
    constexpr float hifp8MaxValueRec = 1 / 32768.0f;

    Reg::Duplicate<float, float>(vreg_zero, 0.0f);

    for (uint16_t i = 0; i < m; ++i) {
        uint32_t sreg_init = d;
        LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * REDUCE_SIZE);
        Reg::Compare<float, CMPMODE::EQ>(preg_zero_sum, vreg_exp_sum, vreg_zero, preg_all);
        for (uint16_t j = 0; j < dLoops; ++j) {
            MaskReg preg_update = UpdateMask<float>(sreg_init);

            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                          IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
            }
            Div(vreg_div, vreg_input_cur, vreg_exp_sum, preg_update);
            Select(vreg_div, vreg_zero, vreg_div, preg_zero_sum);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div,
                                                         preg_update);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD>
__aicore__ inline void LastDivNew(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                  const LocalTensor<T> &expSumTensor, const uint16_t m, const uint16_t d,
                                  const float deScaleV)
{
    __ubuf__ float *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ float *curUb = (__ubuf__ T *)curTensor.GetPhyAddr();
    __ubuf__ float *expSumUb = (__ubuf__ T *)expSumTensor.GetPhyAddr();

    LastDivNewVF<T, INPUT_T, OUTPUT_T, srcD>(dstUb, curUb, expSumUb, m, d, deScaleV);
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
