/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * \file vf_flashupdate_new_mx.h
 * \brief
 */
#ifndef MY_FLASH_UPDATE_NEW_INTERFACE_MX_H
#define MY_FLASH_UPDATE_NEW_INTERFACE_MX_H

#include "kernel_tensor.h"

namespace FaVectorApi {
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre, bool isMlaFullQuant>
__simd_vf__ inline void FlashUpdateBasicVF_MX(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, __ubuf__ float * rowMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_row_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();

    // dstTensor = preTensor * expMaxTensor + curTensor
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);  // [m,8]
        // if constexpr (isMlaFullQuant) {
        //     LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_row_max, rowMaxUb + i * reduceSize);
        // }

        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);//不对
            // if constexpr (isMlaFullQuant) {
            //     Mul(vreg_input_cur, vreg_input_cur, vreg_row_max, preg_all);
            // }
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_add, preg_all);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre, bool isMlaFullQuant>
__simd_vf__ inline void FlashUpdateLastBasicVF_MX(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * rowMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_row_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;
    RegTensor<float> vreg_div;
    RegTensor<half> vreg_cast;
    RegTensor<float> vreg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    constexpr float fp8e4m3MaxValueRec = 1 / 448.0f;
    constexpr float int8MaxValueRec = 1 / 127.0f;
    constexpr float hifp8MaxValueRec = 1 / 32768.0f;

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * reduceSize);
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_all);
            if constexpr (isMlaFullQuant) {
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value) {
                    Muls(vreg_div, vreg_div, fp8e4m3MaxValueRec, preg_all);
                } else if constexpr (IsSameType<INPUT_T, int8_t>::value) {
                    Muls(vreg_div, vreg_div, int8MaxValueRec, preg_all);
                } else {
                    Muls(vreg_div, vreg_div, hifp8MaxValueRec, preg_all);
                }
            }
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div, preg_all);
        }
    }
}


template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD, bool isMlaFullQuant>
__simd_vf__ inline void LastDivNewVF_MX(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * expSumUb,
    const uint16_t m, const uint16_t d, const float deScaleV)
{
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = 64;
    const uint16_t dLoops = d >> 6;
    constexpr float fp8e4m3MaxValueRec = 1 / 448.0f;
    constexpr float int8MaxValueRec = 1 / 127.0f;
    constexpr float hifp8MaxValueRec = 1 / 32768.0f;

    for (uint16_t i = 0; i < m; ++i) {
        uint32_t sreg_init = d;
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * REDUCE_SIZE);
        for (uint16_t j = 0; j < dLoops; ++j) {
            MaskReg preg_update = UpdateMask<float>(sreg_init);

            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
            }
            Div(vreg_div, vreg_input_cur, vreg_exp_sum, preg_update);
            if constexpr (isMlaFullQuant) {
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value) {
                    Muls(vreg_div, vreg_div, fp8e4m3MaxValueRec, preg_all);
                } else if constexpr (IsSameType<INPUT_T, int8_t>::value) {
                    Muls(vreg_div, vreg_div, int8MaxValueRec, preg_all);
                } else {
                    Muls(vreg_div, vreg_div, hifp8MaxValueRec, preg_all);
                }
            }
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div, preg_update);
        }
    }
}

template <typename T>
__simd_vf__ inline void RowInvalidUpdateVF_MX(__ubuf__ T *finalUb, __ubuf__ float *maxUb, const uint16_t m,
    const uint16_t d, int64_t dSize, const uint32_t pltTailD, const uint16_t hasTail, const uint32_t min = 0xFF7FFFFF)
{
    constexpr uint16_t floatRepSize = 64; // 64: 一个寄存器可以存储64个float类型数据
    const uint16_t dLoops = d / floatRepSize;


    constexpr uint32_t tmpZero = 0x00000000; // zero value of fp16 and fp32
    const T zeroValue = *((T*)&tmpZero);
    const float minValue = *((float*)&min);
    MicroAPI::RegTensor<float> vregMinValue;
    MicroAPI::RegTensor<T> vregZeroValue;
    MicroAPI::RegTensor<float> vregMax;
    MicroAPI::RegTensor<T> vregFinal;
    MicroAPI::RegTensor<T> vregFinalNew;

    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t tmpTailD = pltTailD;
    MicroAPI::MaskReg pregTailD = MicroAPI::UpdateMask<T>(tmpTailD);
    MicroAPI::MaskReg pregCompare;

    MicroAPI::Duplicate<float, float>(vregMinValue, minValue);
    MicroAPI::Duplicate<T, T>(vregZeroValue, zeroValue);
    for (uint16_t i = 0; i < m; ++i) {
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vregMax, maxUb + i);
        MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregAll);
        for (uint16_t j = 0; j < dLoops; ++j) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregFinal, finalUb + i * dSize + j * floatRepSize);
            MicroAPI::Select<T>(vregFinalNew, vregZeroValue, vregFinal, pregCompare);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(finalUb + i * dSize + j * floatRepSize,
                vregFinalNew, pregAll);
        }
        for (uint16_t t = 0; t < hasTail; ++t) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregFinal, finalUb + i * dSize + dLoops * floatRepSize);
            MicroAPI::Select<T>(vregFinalNew, vregZeroValue, vregFinal, pregCompare);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(finalUb + i * dSize + dLoops * floatRepSize,
                vregFinalNew, pregTailD);
        }
    }
}

template <typename T>
__aicore__ inline void RowInvalidUpdateVF_MX(__ubuf__ T *finalUb, __ubuf__ float *maxUb, 
    const uint16_t m, const uint16_t d, int64_t dSize, const uint32_t min = 0xFF7FFFFF)
{
    // __ubuf__ T * finalUb = (__ubuf__ T*)finalTensor.GetPhyAddr();
    // __ubuf__ float * maxUb = (__ubuf__ float*)maxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);
    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    RowInvalidUpdateVF<T>(finalUb, maxUb, m, d, dSize, pltTailD, hasTail, min);
}
} // namespace

#endif // MY_FLASH_UPDATE_INTERFACE_H
