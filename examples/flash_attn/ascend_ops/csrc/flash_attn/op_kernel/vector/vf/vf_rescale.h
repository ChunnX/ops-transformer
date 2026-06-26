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
 * \file vf_rescale.h
 * \brief
 */
#ifndef VF_RESCALE_H
#define VF_RESCALE_H

#include "kernel_tensor.h"
#ifdef __NPU_DEVICE__
namespace FaVectorApi {
#ifndef __CCE_KT_TEST__
// bf16->fp32
static constexpr MicroAPI::CastTrait castTraitFp16_32_update = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr uint16_t REDUCE_SIZE = 1;
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre>
__simd_vf__ inline void FlashUpdateBasicVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, __ubuf__ float * rowMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);

        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_add, preg_all);
        }
    }
}
/* **************************************************************************************************
 * FlashUpdate, fp32
 * ************************************************************************************************* */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre>
__aicore__ inline void FlashUpdateBasic(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor,
    const uint16_t m, const uint16_t d, const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * rowMaxUb = (__ubuf__ T*)rowMaxTensor.GetPhyAddr();

    FlashUpdateBasicVF<T, INPUT_T, OUTPUT_T, srcD, reduceSize, isUpdatePre>(
        dstUb, curUb, preUb, expMaxUb, rowMaxUb, m, d, deScaleV, deScaleVPre);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t reduceSize, bool isUpdatePre>
__simd_vf__ inline void FlashUpdateGeneralVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre, const uint32_t pltTailD, const uint16_t hasTail)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    uint32_t tmpTailD = pltTailD;
    MaskReg preg_tail_d = UpdateMask<float>(tmpTailD);
    constexpr uint16_t floatRepSize = 64;
    const uint16_t dLoops = d / floatRepSize;

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);

        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);

            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_add, preg_all);
        }
        for (uint16_t t = 0; t < hasTail; ++t) {
            LoadAlign(vreg_input_pre, preUb + i * d + dLoops * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + dLoops * floatRepSize);

            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_tail_d);
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_tail_d);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + dLoops * floatRepSize, vreg_add, preg_tail_d);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t reduceSize, bool isUpdatePre>
__aicore__ inline void FlashUpdateGeneral(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);

    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    FlashUpdateGeneralVF<T, INPUT_T, OUTPUT_T, reduceSize, isUpdatePre>(
        dstUb, curUb, preUb, expMaxUb, m, d, deScaleV, deScaleVPre, pltTailD, hasTail);
}

/*
 * @ingroup FlashUpdate
 * @brief compute, dstTensor = preTensor * expMaxTensor + curTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, bool isUpdatePre>
__aicore__ inline void FlashUpdateNew(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdate, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if constexpr(srcD % floatRepSize == 0) {
        FlashUpdateBasic<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE, isUpdatePre>(dstTensor, curTensor, preTensor, expMaxTensor, rowMaxTensor,
        m, d, deScaleV, deScaleVPre);
    } else {

        FlashUpdateGeneral<T, INPUT_T, OUTPUT_T, REDUCE_SIZE, isUpdatePre>(dstTensor, curTensor, preTensor, expMaxTensor, m, d,
        deScaleV, deScaleVPre);
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre>
__simd_vf__ inline void FlashUpdateLastBasicVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * rowMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * reduceSize);
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div, preg_all);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre>
__aicore__ inline void FlashUpdateLastBasic(const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& curTensor, const LocalTensor<T>& preTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m, const uint16_t d, const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * rowMaxUb = (__ubuf__ T*)rowMaxTensor.GetPhyAddr();

    FlashUpdateLastBasicVF<T, INPUT_T, OUTPUT_T, srcD, reduceSize, isUpdatePre>(
        dstUb, curUb, preUb, expMaxUb, expSumUb, rowMaxUb, m, d, deScaleV, deScaleVPre);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t reduceSize, bool isUpdatePre>
__simd_vf__ inline void FlashUpdateLastGeneralVF(__ubuf__ float * dstUb, __ubuf__ float * curUb,
    __ubuf__ float * preUb, __ubuf__ float * expMaxUb, __ubuf__ float * expSumUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre, const uint32_t pltTailD, const uint16_t hasTail)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    uint32_t tmpTailD = pltTailD;
    MaskReg preg_tail_d = UpdateMask<float>(tmpTailD);
    constexpr uint16_t floatRepSize = 64;
    uint16_t dLoops = d / floatRepSize;

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * reduceSize);
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);

            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div, preg_all);
        }

        for (uint16_t t = 0; t < hasTail; ++t) {
            LoadAlign(vreg_input_pre, preUb + i * d + dLoops * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + dLoops * floatRepSize);    
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_tail_d);
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_tail_d);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_tail_d);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + dLoops * floatRepSize, vreg_div, preg_tail_d);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t reduceSize, bool isUpdatePre>
__aicore__ inline void FlashUpdateLastGeneral(const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& curTensor, const LocalTensor<T>& preTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m, const uint16_t d, const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = tailD;

    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    FlashUpdateLastGeneralVF<T, INPUT_T, OUTPUT_T, reduceSize, isUpdatePre>(
        dstUb, curUb, preUb, expMaxUb, expSumUb, m, d, deScaleV, deScaleVPre, pltTailD, hasTail);
}

/*
 * @ingroup FlashUpdateLast
 * @brief compute, dstTensor = (preTensor * expMaxTensor + curTensor) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, 32 bytes align
 */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, bool isUpdatePre>
__aicore__ inline void FlashUpdateLastNew(const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& curTensor, const LocalTensor<T>& preTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor, const LocalTensor<T>& expSumTensor,
    uint16_t m, uint16_t d, const float deScaleV, const float deScaleVPre)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateLast, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if constexpr(srcD % floatRepSize == 0) {
        FlashUpdateLastBasic<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE, isUpdatePre>(
            dstTensor, curTensor, preTensor, expMaxTensor, rowMaxTensor, expSumTensor, m, d, deScaleV, deScaleVPre);
    } else {
        FlashUpdateLastGeneral<T, INPUT_T, OUTPUT_T, REDUCE_SIZE, isUpdatePre>(
            dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, m, d, deScaleV, deScaleVPre);
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD>
__simd_vf__ inline void LastDivNewVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * expSumUb,
    const uint16_t m, const uint16_t d, const float deScaleV)
{
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = 64;
    const uint16_t dLoops = d >> 6;

    for (uint16_t i = 0; i < m; ++i) {
        uint32_t sreg_init = d;
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * REDUCE_SIZE);
        for (uint16_t j = 0; j < dLoops; ++j) {
            MaskReg preg_update = UpdateMask<float>(sreg_init);

            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            Div(vreg_div, vreg_input_cur, vreg_exp_sum, preg_update);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div, preg_update);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD>
__aicore__ inline void LastDivNew(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d, const float deScaleV)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    LastDivNewVF<T, INPUT_T, OUTPUT_T, srcD>(dstUb, curUb, expSumUb, m, d, deScaleV);
}

template <typename T>
__simd_vf__ inline void RowInvalidUpdateVF(__ubuf__ T *finalUb, __ubuf__ float *maxUb, const uint16_t m,
    const uint16_t d, int64_t dSize, const uint32_t pltTailD, const uint16_t hasTail, const uint32_t min = 0xFF7FFFFF)
{
    constexpr uint16_t floatRepSize = 64;
    const uint16_t dLoops = d / floatRepSize;


    constexpr uint32_t tmpZero = 0x00000000;
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
__aicore__ inline void RowInvalidUpdateVF(const LocalTensor<T>& finalTensor, const LocalTensor<float>& maxTensor,
    const uint16_t m, const uint16_t d, int64_t dSize, const uint32_t min = 0xFF7FFFFF)
{
    __ubuf__ T * finalUb = (__ubuf__ T*)finalTensor.GetPhyAddr();
    __ubuf__ float * maxUb = (__ubuf__ float*)maxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);
    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    RowInvalidUpdateVF<T>(finalUb, maxUb, m, d, dSize, pltTailD, hasTail, min);
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__simd_vf__ inline void DivCastImpl64VF(__ubuf__ OUTPUT_T * dstUb, __ubuf__ float * srcUb, __ubuf__ float * expSumUb,
    const uint16_t m)
{
    RegTensor<T> vreg_src;
    RegTensor<T> vreg_div;
    RegTensor<T> vreg_exp_sum;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_div_even_bf16;
    RegTensor<bfloat16_t> vreg_div_bf16;
    RegTensor<bfloat16_t> vreg_dst_even_bf16;
    RegTensor<bfloat16_t> vreg_dst_odd_bf16;
    // half
    RegTensor<half> vreg_div_even_f16;
    RegTensor<half> vreg_div_f16;
    RegTensor<half> vreg_dst_even_f16;
    RegTensor<half> vreg_dst_odd_f16;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i);
        LoadAlign(vreg_src, srcUb + i * srcD);
        Div(vreg_div, vreg_src, vreg_exp_sum, preg_all);

        if constexpr (IsSameType<OUTPUT_T, float>::value) {
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_div, preg_all);
        } else if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
            Cast<OUTPUT_T, T, castTraitZero>(vreg_div_bf16, vreg_div, preg_all_b16);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_PACK_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_div_bf16, preg_all);
        } else {
            Cast<OUTPUT_T, T, castTraitZero>(vreg_div_f16, vreg_div, preg_all_b16);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_PACK_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_div_f16, preg_all);
        }
    }
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void DivCastImpl64(const LocalTensor<OUTPUT_T>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m)
{
    __ubuf__ OUTPUT_T * dstUb = (__ubuf__ OUTPUT_T*)dstTensor.GetPhyAddr();
    __ubuf__ float * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    DivCastImpl64VF<T, OUTPUT_T, srcD>(dstUb, srcUb, expSumUb, m);
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__simd_vf__ inline void DivCastImpl128VF(__ubuf__ OUTPUT_T * dstUb, __ubuf__ float * srcUb, __ubuf__ float * expSumUb,
    const uint16_t m)
{
    RegTensor<float> vreg_src_even, vreg_src_odd;
    RegTensor<float> vreg_div_even, vreg_div_odd;
    RegTensor<float> vreg_exp_sum;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_div_even_bf16;
    RegTensor<bfloat16_t> vreg_div_odd_bf16;
    RegTensor<bfloat16_t> vreg_cast_bf16;
    // half
    RegTensor<half> vreg_div_even_f16;
    RegTensor<half> vreg_div_odd_f16;
    RegTensor<half> vreg_cast_f16;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i);
        if constexpr (IsSameType<OUTPUT_T, float>::value) {
            LoadAlign(vreg_src_even, srcUb + i * srcD);
            LoadAlign(vreg_src_odd, srcUb + i * srcD + (srcD >> 1));
        } else {
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_src_even, vreg_src_odd, srcUb + i * srcD);
        }
        Div(vreg_div_even, vreg_src_even, vreg_exp_sum, preg_all);
        Div(vreg_div_odd, vreg_src_odd, vreg_exp_sum, preg_all);

        if constexpr (IsSameType<OUTPUT_T, float>::value) {
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_div_even, preg_all);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD + (srcD >> 1), vreg_div_odd, preg_all);
        } else if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
            Cast<OUTPUT_T, T, castTraitZero>(vreg_div_even_bf16, vreg_div_even, preg_all);
            Cast<OUTPUT_T, T, castTraitOne>(vreg_div_odd_bf16, vreg_div_odd, preg_all);
            Or((RegTensor<uint16_t>&)vreg_cast_bf16, (RegTensor<uint16_t>&)vreg_div_even_bf16,
                (RegTensor<uint16_t>&)vreg_div_odd_bf16, preg_all_b16);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_cast_bf16, preg_all_b16);
        } else {
            Cast<OUTPUT_T, T, castTraitZero>(vreg_div_even_f16, vreg_div_even, preg_all);
            Cast<OUTPUT_T, T, castTraitOne>(vreg_div_odd_f16, vreg_div_odd, preg_all);
            Or((RegTensor<uint16_t>&)vreg_cast_f16, (RegTensor<uint16_t>&)vreg_div_even_f16, (RegTensor<uint16_t>&)vreg_div_odd_f16, preg_all_b16);
            StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ OUTPUT_T *&)dstUb + i * srcD, vreg_cast_f16, preg_all_b16);
        }
    }
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void DivCastImpl128(const LocalTensor<OUTPUT_T>& dstTensor,
                                      const LocalTensor<T>& srcTensor, const LocalTensor<T>& expSumTensor,
                                      const uint16_t m)
{
    __ubuf__ OUTPUT_T * dstUb = (__ubuf__ OUTPUT_T*)dstTensor.GetPhyAddr();
    __ubuf__ float * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    DivCastImpl128VF<T, OUTPUT_T, srcD>(dstUb, srcUb, expSumUb, m);
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__simd_vf__ inline void DivCastImplGeneralVF(__ubuf__ OUTPUT_T * dstUb, __ubuf__ float * srcUb,
    __ubuf__ float * expSumUb, const uint16_t m)
{
    RegTensor<float> vreg_src;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_div_bf16;
    RegTensor<bfloat16_t> vreg_dst_even_bf16;
    RegTensor<bfloat16_t> vreg_dst_odd_bf16;
    // half
    RegTensor<half> vreg_div_f16;
    RegTensor<half> vreg_dst_even_f16;
    RegTensor<half> vreg_dst_odd_f16;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    constexpr uint16_t dLoops = srcD >> 6;
    constexpr uint16_t floatRepSize = 64;

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i);
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_src, srcUb + i * srcD + j * floatRepSize);
            Div(vreg_div, vreg_src, vreg_exp_sum, preg_all);

            if constexpr (IsSameType<OUTPUT_T, float>::value) {
                StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ OUTPUT_T *&)dstUb + i * srcD + j * floatRepSize, vreg_div, preg_all);
            } else if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
                Cast<OUTPUT_T, T, castTraitZero>(vreg_div_bf16, vreg_div, preg_all_b16);
                StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_PACK_B32>(
                    (__ubuf__ OUTPUT_T *&)dstUb + i * srcD + j * floatRepSize, vreg_div_bf16, preg_all);
            } else {
                Cast<OUTPUT_T, T, castTraitZero>(vreg_div_f16, vreg_div, preg_all_b16);
                StoreAlign<OUTPUT_T, MicroAPI::StoreDist::DIST_PACK_B32>(
                    (__ubuf__ OUTPUT_T *&)dstUb + i * srcD + j * floatRepSize, vreg_div_f16, preg_all);
            }
        }
    }
}

template <typename T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void DivCastImplGeneral(const LocalTensor<OUTPUT_T>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m)
{
    __ubuf__ OUTPUT_T * dstUb = (__ubuf__ OUTPUT_T*)dstTensor.GetPhyAddr();
    __ubuf__ float * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    DivCastImplGeneralVF<T, OUTPUT_T, srcD>(dstUb, srcUb, expSumUb, m);
}

/*
 * @ingroup DivCast
 * @brief compute, dstTensor = cast(srcTensor / expSumTensor)
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor, shape is [m, 8]
 * @param [in] m, input rows
 * @param [in] d, input colums, 32 bytes align
 * @param [in] srcD, should be 64 or 128
 */
template <typename T, typename OUTPUT_T, uint16_t srcD>
__aicore__ inline void DivCast(const LocalTensor<OUTPUT_T>& dstTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m)
{
    if constexpr (srcD == 64) {
        DivCastImpl64<T, OUTPUT_T, srcD>(dstTensor, srcTensor, expSumTensor, m);
    } else if constexpr (srcD == 128) {
        DivCastImpl128<T, OUTPUT_T, srcD>(dstTensor, srcTensor, expSumTensor, m);
    } else {    
        DivCastImplGeneral<T, OUTPUT_T, srcD>(dstTensor, srcTensor, expSumTensor, m);
    }
}
#endif
} // namespace
#endif
#endif
