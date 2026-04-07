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
 * \file vf_mul_sel_softmaxflashv2_cast_nz.h
 * \brief
 */
#ifndef MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_H
#define MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_H

#include "kernel_tensor.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {
constexpr uint32_t floatRepSize = 64;
constexpr uint32_t halfRepSize = 128;
constexpr uint32_t blockBytesU8 = 32;
/* **************************************************************************************************
 * Muls + Select(optional) + SoftmaxFlashV2 + Cast(fp32->fp16/bf16) + ND2NZ
 * ************************************************************************************************* */
using AscendC::LocalTensor;

enum OriginNRange {
    GT_128_AND_LTE_256 = 0,  // 128 < originN <= 256, support for non-alignment (s2BaseSize=256)
    GT_64_AND_LTE_128,  // 64 < originN <= 128, support for non-alignment (s2BaseSize=128)
    EQ_128,                 // originN == 128, better performance than GT_64_AND_LTE_128 (s2BaseSize=128)
    GT_0_AND_LTE_64,        // 0 < originN <= 64 (s2BaseSize <= 64 or tail s2)
    GT_256_AND_LTE_512,     // 256 < originN <= 512 (s2BaseSize <= 512 or tail s2)
    GT_512_AND_LTE_1024,    // 512 < originN <= 1024 (s2BaseSize <= 1024 or tail s2)
    N_INVALID
};
using namespace MicroAPI;
constexpr static AscendC::MicroAPI::CastTrait castTraitZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitTwo = {
    AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitThree = {
    AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
 
constexpr static AscendC::MicroAPI::CastTrait castTraitRintZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
 
constexpr static AscendC::MicroAPI::CastTrait castTraitRintTwo = {
    AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintThree = {
    AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
 
// no update, 128 < originN <= 256
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1NoUpdateGeneralImpl256(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue, float keepProb)
{
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (65*16*2/32)，单位block, low 16bits: repeatStride (1)
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    uint64_t maskUb1 = maskTensor.GetPhyAddr();
    uint64_t maskUb2 = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t maskUb3 = maskTensor.GetPhyAddr() + floatRepSize * 2;
    uint64_t maskUb4 = maskTensor.GetPhyAddr() + floatRepSize * 3;
    uint64_t dropMaskUb1 = dropTensor.GetPhyAddr();
    uint64_t dropMaskUb2 = dropTensor.GetPhyAddr() + s2BaseSize / 16;


    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t oriTailN1 = originN - floatRepSize * 2 < floatRepSize ? originN - floatRepSize * 2 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 3) <= 0 ? 0 : originN - floatRepSize * 3;
    const uint32_t tailN1 = s2BaseSize - floatRepSize * 2;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 3;
    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    float divValue = 1.0f / keepProb;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
		
        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_tail_n1 = UpdateMask<float>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<float>(pltOriTailN1);
        MaskReg preg_tail_n2 = UpdateMask<float>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<float>(pltOriTailN2);
        MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();

        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;

        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();

        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;

        Duplicate(vreg_min, minValue);
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Arange(vreg_alibi1, posShift);
            Arange(vreg_alibi2, posShift + floatRepSize);
            Arange(vreg_alibi3, posShift + floatRepSize * 2);
            Arange(vreg_alibi4, posShift + floatRepSize * 3);
        }
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x1, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
            DataCopy(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
            DataCopy(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);  // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                              pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Abs(vreg_pse1, vreg_alibi1, preg_all);
                    Abs(vreg_pse2, vreg_alibi2, preg_all);
                    Abs(vreg_pse3, vreg_alibi3, preg_all);
                    Abs(vreg_pse4, vreg_alibi4, preg_all);
                    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                        Sqrt(vreg_pse1, vreg_pse1, preg_all);
                        Sqrt(vreg_pse2, vreg_pse2, preg_all);
                        Sqrt(vreg_pse3, vreg_pse3, preg_all);
                        Sqrt(vreg_pse4, vreg_pse4, preg_all);
                    }
                    Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
                    Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
                    Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
                    Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
                    Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
                    Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
                    Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
                    Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
                } else {
                    if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                        DataCopy(vreg_pse_bf16_src1, pseUb + i * pseStride);
                        DataCopy(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                        Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
                        Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
                    } else if constexpr (IsSameType<pseShiftType, half>::value) {
                        DataCopy(vreg_pse_f16_src1, pseUb + i * pseStride);
                        DataCopy(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                        Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
                        Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
                    }
                }
                Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
                Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
                Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
                Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);  // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);  
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);   
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_sel3_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 +  i * s2BaseSize, vreg_sel4_new, preg_tail_n2);
                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp3, preg_all);
            } else {
                Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
                Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_input_x3_new, preg_tail_n1);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_input_x4_new, preg_tail_n2);
                
                Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp3, preg_all);
            }
            DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, maxUb, 0, POST_UPDATE);
        if constexpr (hasDrop == 1) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max_brc, maxUbStart + i);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x1, vreg_input_x2, srcUb + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x3, vreg_input_x4, srcUb + floatRepSize * 2 + i * s2BaseSize);
            FusedExpSub(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            ReduceSum(vreg_exp_sum3, vreg_exp_sum3, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)expSumUb), vreg_exp_sum3, ureg_exp_sum, 1);
            // dropmask compute
            if constexpr (hasDrop == 1) {
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                    preg1, (__ubuf__ uint32_t *&)dropMaskUb1, s2BaseSize >> 3);
                // preg1: 0011223344556677 preg2: 0000000000000000
                MaskInterleave<half>(preg3, preg4, preg1, preg2);
                // preg3: 0000110022003300 preg4: 4400550066007700
                MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
                // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
                Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
                Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
                Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
                Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                    preg1, (__ubuf__ uint32_t *&)dropMaskUb2, s2BaseSize >> 3);
                // preg1: 0011223344556677 preg2: 0000000000000000
                MaskInterleave<half>(preg3, preg4, preg1, preg2);
                // preg3: 0000110022003300 preg4: 4400550066007700
                MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
                // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
                Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
                Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
                Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
                Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            }

            if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
            } else if constexpr (IsSameType<T2, half>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16, (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16, (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_all_b16);
                // fp8_e5m2_t
            }
        }
        vstas(ureg_exp_sum, expSumUb, 0, POST_UPDATE);
    }
}

// no update, 128 < originN <= 256
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1NoUpdateGeneralImpl256RegbaseV2(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue, float keepProb)
{
    using REDUCE_TYPE = float;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (65*16*2/32)，单位block, low 16bits: repeatStride (1)
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    uint64_t maskUb1 = maskTensor.GetPhyAddr();
    uint64_t maskUb2 = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t maskUb3 = maskTensor.GetPhyAddr() + floatRepSize * 2;
    uint64_t maskUb4 = maskTensor.GetPhyAddr() + floatRepSize * 3;
    uint64_t dropMaskUb1 = dropTensor.GetPhyAddr();
    uint64_t dropMaskUb2 = dropTensor.GetPhyAddr() + s2BaseSize / 16;


    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t oriTailN1 = originN - floatRepSize * 2 < floatRepSize ? originN - floatRepSize * 2 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 3) <= 0 ? 0 : originN - floatRepSize * 3;
    const uint32_t tailN1 = s2BaseSize - floatRepSize * 2;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 3;
    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    float divValue = 1.0f / keepProb;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<T> vreg_input_x1;
        RegTensor<T> vreg_input_x2;
        RegTensor<REDUCE_TYPE> vreg_input_x1_b32;
        RegTensor<REDUCE_TYPE> vreg_input_x2_b32;
        RegTensor<REDUCE_TYPE> vreg_input_x3_b32;
        RegTensor<REDUCE_TYPE> vreg_input_x4_b32;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<REDUCE_TYPE> vreg_exp_sum_b16;
        RegTensor<REDUCE_TYPE> vreg_exp_even1;
        RegTensor<REDUCE_TYPE> vreg_exp_odd1;
        RegTensor<REDUCE_TYPE> vreg_exp_even2;
        RegTensor<REDUCE_TYPE> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
		
        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_tail_n1 = UpdateMask<float>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<float>(pltOriTailN1);
        MaskReg preg_tail_n2 = UpdateMask<float>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<float>(pltOriTailN2);
        MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();

        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;

        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();

        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;

        Duplicate(vreg_min, minValue);
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Arange(vreg_alibi1, posShift);
            Arange(vreg_alibi2, posShift + floatRepSize);
            Arange(vreg_alibi3, posShift + floatRepSize * 2);
            Arange(vreg_alibi4, posShift + floatRepSize * 3);
        }
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x1, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x2, srcUb + halfRepSize + i * s2BaseSize);
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all_b16);  // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all_b16);
            }
            
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all_b16);  // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all_b16);
            }

            DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x1, preg_all_b16);
            DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + halfRepSize + i * s2BaseSize, vreg_input_x2, preg_all_b16);
            
            Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all_b16);
            ReduceMax(vreg_input_max, vreg_max_tmp1, preg_all_b16);

            DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, maxUb, 0, POST_UPDATE);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max_brc, maxUbStart + i);
            DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(
                vreg_input_x1, srcUb + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(
                vreg_input_x2, srcUb + halfRepSize + i * s2BaseSize);
            MicroAPI::Cast<REDUCE_TYPE, T, castTraitZero>(vreg_input_x1_b32, vreg_input_x1, preg_all_b16);
            MicroAPI::Cast<REDUCE_TYPE, T, castTraitOne>(vreg_input_x2_b32, vreg_input_x1, preg_all_b16);
            MicroAPI::Cast<REDUCE_TYPE, T, castTraitZero>(vreg_input_x3_b32, vreg_input_x2, preg_all_b16);
            MicroAPI::Cast<REDUCE_TYPE, T, castTraitOne>(vreg_input_x4_b32, vreg_input_x2, preg_all_b16);
            FusedExpSub(vreg_exp_even1, vreg_input_x1_b32, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd1, vreg_input_x2_b32, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even2, vreg_input_x3_b32, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd2, vreg_input_x4_b32, vreg_max_brc, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            ReduceSum(vreg_exp_sum3, vreg_exp_sum3, preg_all);
            MicroAPI::Cast<T, REDUCE_TYPE, castTraitZero>(vreg_exp_sum_b16, vreg_exp_sum3, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)expSumUb), vreg_exp_sum_b16, ureg_exp_sum, 1);

            if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, REDUCE_TYPE, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
            } else if constexpr (IsSameType<T2, half>::value) {
                Cast<T2, REDUCE_TYPE, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16, (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16, (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_all_b16);
                // fp8_e5m2_t
            }
        }
        vstas(ureg_exp_sum, expSumUb, 0, POST_UPDATE);
    }
}

// no update, 64 < originN <= 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1NoUpdateGeneralImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb)
{
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (65*16*2/32)，单位block, low 16bits: repeatStride (1)
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * x_expUb = nullptr;
    if constexpr (IsSameType<T2, float>::value) {
        x_expUb = expUb + ((s1BaseSize >> 1) + 1) * (s2BaseSize >> 1);
    }
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();

    uint64_t maskUb = maskTensor.GetPhyAddr();
    uint64_t maskUbUnroll = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t dropMaskUb = dropTensor.GetPhyAddr();


    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t oriTailN = originN - floatRepSize;
    const uint32_t tailN = s2BaseSize - floatRepSize;
    const float dScale = scale * dScaleQK;
    uint32_t pltOriTailN = oriTailN;
    uint32_t pltTailN = tailN;
    float divValue = 1.0f / keepProb;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel;
        RegTensor<float> vreg_sel_unroll;
        RegTensor<float> vreg_sel_unroll_new;
        RegTensor<float> vreg_input_x;
        RegTensor<float> vreg_input_x_unroll;
        RegTensor<float> vreg_input_x_unroll_new;
        RegTensor<float> vreg_max_tmp;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp_sum;
        RegTensor<float> vreg_exp_even;
        RegTensor<float> vreg_exp_odd;
        RegTensor<float> vreg_pse;
        RegTensor<float> vreg_pse_unroll;
        RegTensor<float> vreg_alibi;
        RegTensor<float> vreg_alibi_unroll;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd_bf16;
        RegTensor<bfloat16_t> vreg_exp_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src;
        RegTensor<bfloat16_t> vreg_pse_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
        // half
        RegTensor<half> vreg_exp_even_f16;
        RegTensor<half> vreg_exp_odd_f16;
        RegTensor<half> vreg_exp_f16;
        RegTensor<half> vreg_pse_f16_src;
        RegTensor<half> vreg_pse_f16;
        RegTensor<half> vreg_pse_f16_unroll;

        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
        MaskReg preg_tail_n = UpdateMask<float>(pltTailN);
        MaskReg preg_ori_tail_n = UpdateMask<float>(pltOriTailN);
        MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();
        MaskReg preg_compare;
        MaskReg preg_compare_unroll;

        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;

        Duplicate(vreg_min, minValue);
        if constexpr (hasAtten == 1 && isMlaSgd) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare, ((__ubuf__ uint32_t*)(maskUb)));
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare_unroll, ((__ubuf__ uint32_t*)(maskUbUnroll)));
        }
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Arange(vreg_alibi, posShift);
            Arange(vreg_alibi_unroll, posShift + 64);
        }
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, dScale, preg_all);  // Muls(scale)
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScale, preg_ori_tail_n);
            } else {
                if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                              IsSameType<T2, hifloat8_t>::value) {
                    Muls(vreg_input_x, vreg_input_x, dScaleQK, preg_all);  // Muls(dScaleQK)
                    Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScaleQK, preg_ori_tail_n);
                }
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                              pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Abs(vreg_pse, vreg_alibi, preg_all);
                    Abs(vreg_pse_unroll, vreg_alibi_unroll, preg_all);
                    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                        Sqrt(vreg_pse, vreg_pse, preg_all);
                        Sqrt(vreg_pse_unroll, vreg_pse_unroll, preg_all);
                    }
                    Muls(vreg_pse, vreg_pse, slopes, preg_all);
                    Muls(vreg_pse_unroll, vreg_pse_unroll, slopes, preg_all);
                    Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all);
                    Adds(vreg_alibi_unroll, vreg_alibi_unroll, -1.0f, preg_all);
                } else {
                    if constexpr (IsSameType<pseShiftType, float>::value) {
                        DataCopy(vreg_pse, pseUb + i * pseStride);
                        DataCopy(vreg_pse_unroll, pseUb + i * pseStride + (s2BaseSize >> 1));
                    } else if constexpr(IsSameType<pseShiftType, bfloat16_t>::value) {
                        DataCopy(vreg_pse_bf16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_bf16_unroll, preg_all_b16);
                    } else {
                        DataCopy(vreg_pse_f16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_f16_unroll, preg_all_b16);
                    }
                }
                Add(vreg_input_x, vreg_input_x, vreg_pse, preg_all);
                Add(vreg_input_x_unroll, vreg_input_x_unroll, vreg_pse_unroll, preg_ori_tail_n);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, scale, preg_all);  // Muls(scale)
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_ori_tail_n);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSgd) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        preg_compare, (__ubuf__ uint32_t *&)maskUb, nPadding);
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, nPadding);    
                }
                Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
                Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll);
                Select(vreg_sel_unroll_new, vreg_sel_unroll, vreg_min, preg_ori_tail_n);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel_unroll_new, preg_tail_n);
                Max(vreg_max_tmp, vreg_sel, vreg_sel_unroll_new, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp, preg_all);
            } else {
                Select(vreg_input_x_unroll_new, vreg_input_x_unroll, vreg_min, preg_ori_tail_n);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x_unroll_new, preg_tail_n);

                Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll_new, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp, preg_all);
            }
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, maxUb, 0, POST_UPDATE);
        if constexpr (hasDrop == 1) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max_brc, maxUbStart + i);
            if constexpr (IsSameType<T2, float>::value) {
                DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
                DataCopy(vreg_input_x_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
            } else {
                DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    vreg_input_x, vreg_input_x_unroll, srcUb + i * s2BaseSize);
            }
            FusedExpSub(vreg_exp_even, vreg_input_x, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max_brc, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
            ReduceSum(vreg_exp_sum, vreg_exp_sum, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)expSumUb), vreg_exp_sum, ureg_exp_sum, 1);
            // dropmask compute
            if constexpr (hasDrop == 1) {
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                    preg1, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
                // preg1: 0011223344556677 preg2: 0000000000000000
                if constexpr (IsSameType<T2, float>::value) {
                    MaskInterleave<half>(preg5, preg6, preg1, preg2);
                    // preg5(even-4bit): 0000110022003300 preg6(odd-4bit): 4400550066007700
                } else {
                    MaskInterleave<half>(preg3, preg4, preg1, preg2);
                    // preg3: 0000110022003300 preg4: 4400550066007700
                    MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
                    // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
                }
                Select(vreg_sel_drop, vreg_exp_even, vreg_zero, preg5);
                Muls(vreg_exp_even, vreg_sel_drop, divValue, preg_all);
                Select(vreg_sel_drop2, vreg_exp_odd, vreg_zero, preg6);
                Muls(vreg_exp_odd, vreg_sel_drop2, divValue, preg_all);
            }

            if constexpr (IsSameType<T2, float>::value) {
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_even, blockStride, repeatStride, preg_all);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_expUb), vreg_exp_odd, blockStride, repeatStride, preg_all);
            } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd_bf16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, repeatStride, preg_all_b16);
            } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
                RegTensor<fp8_e5m2_t> vreg_exp_even_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_odd_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_merge_tmp_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
                RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_even_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_odd_f8e5m2, preg_all_b8);
                DataCopy(vreg_exp_merge_f8e5m2_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e5m2, vreg_exp_merge_tmp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_even_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_odd_f8e4m3, preg_all_b8);
                DataCopy(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
                RegTensor<hifloat8_t> vreg_exp_even_hif8;
                RegTensor<hifloat8_t> vreg_exp_odd_hif8;
                RegTensor<hifloat8_t> vreg_exp_merge_tmp_hif8;
                RegTensor<hifloat8_t> vreg_exp_merge_hif8;
                RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_hif8, (RegTensor<uint8_t>&)vreg_exp_even_hif8, (RegTensor<uint8_t>&)vreg_exp_odd_hif8, preg_all_b8);
                DataCopy(vreg_exp_merge_hif8_indexes, indexesUb);
                Gather(vreg_exp_merge_hif8, vreg_exp_merge_tmp_hif8, vreg_exp_merge_hif8_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
            } else {
                Cast<T2, T, castTraitZero>(vreg_exp_even_f16, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd_f16, vreg_exp_odd, preg_all);
                Or((RegTensor<uint16_t>&)(RegTensor<uint16_t>&)vreg_exp_f16, (RegTensor<uint16_t>&)vreg_exp_even_f16, (RegTensor<uint16_t>&)vreg_exp_odd_f16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_f16, blockStride, repeatStride, preg_all_b16);
            }
        }
        vstas(ureg_exp_sum, expSumUb, 0, POST_UPDATE);
    }
}

// no update, originN == 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1NoUpdateImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb)
{
    float divValue = 1.0f / keepProb;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * x_expUb = nullptr;
    if constexpr (IsSameType<T2, float>::value) {
        x_expUb = expUb + ((s1BaseSize >> 1) + 1) * (s2BaseSize >> 1);
    }
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    uint64_t maskUb = maskTensor.GetPhyAddr();
    uint64_t maskUbUnroll = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t dropMaskUb = dropTensor.GetPhyAddr();
    const float dScale = scale * dScaleQK;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel;
        RegTensor<float> vreg_sel_unroll;
        RegTensor<float> vreg_input_x;
        RegTensor<float> vreg_input_x_unroll;
        RegTensor<float> vreg_max_tmp;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_exp_sum;
        RegTensor<float> vreg_exp_even;
        RegTensor<float> vreg_exp_odd;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_pse;
        RegTensor<float> vreg_pse_unroll;
        RegTensor<float> vreg_alibi;
        RegTensor<float> vreg_alibi_unroll;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd_bf16;
        RegTensor<bfloat16_t> vreg_exp_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src;
        RegTensor<bfloat16_t> vreg_pse_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
        // half
        RegTensor<half> vreg_exp_even_f16;
        RegTensor<half> vreg_exp_odd_f16;
        RegTensor<half> vreg_exp_f16;
        RegTensor<half> vreg_pse_f16_src;
        RegTensor<half> vreg_pse_f16;
        RegTensor<half> vreg_pse_f16_unroll;

        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_compare;
        MaskReg preg_compare_unroll;

        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;

        if constexpr (hasAtten == 1) {
            Duplicate(vreg_min, minValue);
            if constexpr (isMlaSgd) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (preg_compare, ((__ubuf__ uint32_t*)(maskUb)));
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (preg_compare_unroll, ((__ubuf__ uint32_t*)(maskUbUnroll)));
            }
        }
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Arange(vreg_alibi, posShift);
            Arange(vreg_alibi_unroll, posShift + 64);
        }
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, dScale, preg_all);  // Muls(scale)
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScale, preg_all);
            } else {
                if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                              IsSameType<T2, hifloat8_t>::value) {
                    Muls(vreg_input_x, vreg_input_x, dScaleQK, preg_all);  // Muls(dScaleQK)
                    Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScaleQK, preg_all);
                }
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                              pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Abs(vreg_pse, vreg_alibi, preg_all);
                    Abs(vreg_pse_unroll, vreg_alibi_unroll, preg_all);
                    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                        Sqrt(vreg_pse, vreg_pse, preg_all);
                        Sqrt(vreg_pse_unroll, vreg_pse_unroll, preg_all);
                    }
                    Muls(vreg_pse, vreg_pse, slopes, preg_all);
                    Muls(vreg_pse_unroll, vreg_pse_unroll, slopes, preg_all);
                    Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all);
                    Adds(vreg_alibi_unroll, vreg_alibi_unroll, -1.0f, preg_all);
                } else {
                    if constexpr (IsSameType<pseShiftType, float>::value) {
                        DataCopy(vreg_pse, pseUb + i * pseStride);
                        DataCopy(vreg_pse_unroll, pseUb + i * pseStride + (s2BaseSize >> 1));
                    } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                        DataCopy(vreg_pse_bf16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_bf16_unroll, preg_all_b16);
                    } else {
                        DataCopy(vreg_pse_f16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_f16_unroll, preg_all_b16);
                    }
                }
                Add(vreg_input_x, vreg_input_x, vreg_pse, preg_all);
                Add(vreg_input_x_unroll, vreg_input_x_unroll, vreg_pse_unroll, preg_all);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, scale, preg_all);  // Muls(scale)
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_all);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSgd) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        preg_compare, (__ubuf__ uint32_t *&)maskUb, s2BaseSize);
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, s2BaseSize); 
                }

                Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
                Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel_unroll, preg_all);
                Max(vreg_max_tmp, vreg_sel, vreg_sel_unroll, preg_all);
            } else {
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x_unroll, preg_all);
                Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll, preg_all);
            }
            ReduceMax(vreg_input_max, vreg_max_tmp, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, maxUb, 0, POST_UPDATE);
        if constexpr (hasDrop == 1) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            // maxUb is [S1, 1], BRC_B32 is reading one fp32 element and broadcast it to all 64 vreg element
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max_brc, maxUbStart + i);
            if constexpr (IsSameType<T2, float>::value) {
                DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
                DataCopy(vreg_input_x_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
            } else {
                DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    vreg_input_x, vreg_input_x_unroll, srcUb + i * s2BaseSize);
            }
            FusedExpSub(vreg_exp_even, vreg_input_x, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max_brc, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
            ReduceSum(vreg_exp_sum, vreg_exp_sum, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)expSumUb), vreg_exp_sum, ureg_exp_sum, 1);

            // dropmask compute
            if constexpr (hasDrop == 1) {
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                    preg1, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
                if constexpr (IsSameType<T2, float>::value) {
                    MaskInterleave<half>(preg5, preg6, preg1, preg2);
                } else {
                    MaskInterleave<half>(preg3, preg4, preg1, preg2);
                    MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
                }
                Select(vreg_sel_drop, vreg_exp_even, vreg_zero, preg5);
                Muls(vreg_exp_even, vreg_sel_drop, divValue, preg_all);
                Select(vreg_sel_drop2, vreg_exp_odd, vreg_zero, preg6);
                Muls(vreg_exp_odd, vreg_sel_drop2, divValue, preg_all);
            }

            if constexpr (IsSameType<T2, float>::value) {
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_even, blockStride, repeatStride, preg_all);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_expUb), vreg_exp_odd, blockStride, repeatStride, preg_all);
            } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd_bf16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, repeatStride, preg_all_b16);
            } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
                RegTensor<fp8_e5m2_t> vreg_exp_even_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_odd_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_merge_tmp_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
                RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_even_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_odd_f8e5m2, preg_all_b8);
                DataCopy(vreg_exp_merge_f8e5m2_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e5m2, vreg_exp_merge_tmp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_even_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_odd_f8e4m3, preg_all_b8);
                DataCopy(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
                RegTensor<hifloat8_t> vreg_exp_even_hif8;
                RegTensor<hifloat8_t> vreg_exp_odd_hif8;
                RegTensor<hifloat8_t> vreg_exp_merge_tmp_hif8;
                RegTensor<hifloat8_t> vreg_exp_merge_hif8;
                RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_hif8, (RegTensor<uint8_t>&)vreg_exp_even_hif8, (RegTensor<uint8_t>&)vreg_exp_odd_hif8, preg_all_b8);
                DataCopy(vreg_exp_merge_hif8_indexes, indexesUb);
                Gather(vreg_exp_merge_hif8, vreg_exp_merge_tmp_hif8, vreg_exp_merge_hif8_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
            } else {
                Cast<T2, T, castTraitZero>(vreg_exp_even_f16, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd_f16, vreg_exp_odd, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp_f16, (RegTensor<uint16_t>&)vreg_exp_even_f16, (RegTensor<uint16_t>&)vreg_exp_odd_f16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_f16, blockStride, repeatStride, preg_all_b16);
            }
        }
        vstas(ureg_exp_sum, expSumUb, 0, POST_UPDATE);
    }
}

// no update, originN <= 64
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1NoUpdateImpl64(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb)
{
    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();

    uint64_t maskUb = maskTensor.GetPhyAddr();
    uint64_t dropMaskUb = dropTensor.GetPhyAddr();


    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    uint32_t pltOriginalN = originN;

    float divValue = 1.0f / keepProb;
    uint32_t pltSrcN = s2BaseSize;
    uint32_t pltSrcN16 = s2BaseSize;
    const float dScale = scale * dScaleQK;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel;
        RegTensor<float> vreg_input_x;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_exp;
        RegTensor<float> vreg_exp_sum;
        RegTensor<float> vreg_pse;
        RegTensor<float> vreg_alibi;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_zero;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_exp_bf16;
        RegTensor<bfloat16_t> vreg_dst_even_bf16;
        RegTensor<bfloat16_t> vreg_dst_odd_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src;
        RegTensor<bfloat16_t> vreg_pse_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
        // half
        RegTensor<half> vreg_exp_even_f16;
        RegTensor<half> vreg_exp_f16;
        RegTensor<half> vreg_dst_even_f16;
        RegTensor<half> vreg_dst_odd_f16;
        RegTensor<half> vreg_pse_f16_src;
        RegTensor<half> vreg_pse_f16;
        RegTensor<half> vreg_pse_f16_unroll;

        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_src_n = UpdateMask<float>(pltSrcN);
        MaskReg preg_src_n_b16 = UpdateMask<uint16_t>(pltSrcN16);

        MaskReg preg_ori_src_n = UpdateMask<T>(pltOriginalN);
        MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();
        MaskReg preg_compare;
        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
        MaskReg preg3;
        MaskReg preg4;

        if constexpr (hasAtten == 1) {
            Duplicate(vreg_min, minValue);
            if constexpr (isMlaSgd) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (preg_compare, ((__ubuf__ uint32_t*)(maskUb)));
            }
        }
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Arange(vreg_alibi, posShift);
        }
        // x_max = max(src, axis=-1, keepdims=True)
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, dScale, preg_ori_src_n);  // Muls(scale)
            } else {
                if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                              IsSameType<T2, hifloat8_t>::value) {
                    Muls(vreg_input_x, vreg_input_x, dScaleQK, preg_ori_src_n);  // Muls(dScaleQK)
                }
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                              pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Abs(vreg_pse, vreg_alibi, preg_all);
                    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                        Sqrt(vreg_pse, vreg_pse, preg_all);
                    }
                    Muls(vreg_pse, vreg_pse, slopes, preg_all);
                    Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all);
                } else {
                    if constexpr (IsSameType<pseShiftType, float>::value) {
                        DataCopy(vreg_pse, pseUb + i * pseStride);
                    } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                        DataCopy(vreg_pse_bf16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                    } else {
                        DataCopy(vreg_pse_f16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                    }
                }
                Add(vreg_input_x, vreg_input_x, vreg_pse, preg_ori_src_n);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, scale, preg_ori_src_n);  // Muls(scale)
            }
            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSgd) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        preg_compare, (__ubuf__ uint32_t *&)maskUb, nPadding);
                }
                Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_src_n);
                ReduceMax(vreg_input_max, vreg_sel, preg_ori_src_n);
            } else {
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_src_n);
                ReduceMax(vreg_input_max, vreg_input_x, preg_ori_src_n);
            }
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, maxUb, 0, POST_UPDATE);
        if constexpr (hasDrop == 1) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max_brc, maxUbStart + i);
            DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
            FusedExpSub(vreg_exp, vreg_input_x, vreg_max_brc, preg_ori_src_n);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            ReduceSum(vreg_exp_sum, vreg_exp, preg_ori_src_n);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)expSumUb), vreg_exp_sum, ureg_exp_sum, 1);
            // dropmask compute
            if constexpr (hasDrop == 1) {
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                    preg1, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);

                MaskInterleave<half>(preg3, preg4, preg1, preg2);
                Select(vreg_sel_drop, vreg_exp, vreg_zero, preg3);
                Muls(vreg_exp, vreg_sel_drop, divValue, preg_all);
            }

            if constexpr (IsSameType<T2, float>::value) {
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp, blockStride, repeatStride, preg_all);
            } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_bf16, vreg_exp, preg_all_b16);
                DeInterleave(vreg_dst_even_bf16, vreg_dst_odd_bf16,
                        vreg_exp_bf16, vreg_exp_bf16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_dst_even_bf16, blockStride, repeatStride, preg_src_n_b16);
            } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
                RegTensor<fp8_e5m2_t> vreg_exp_f8e5m2;
                RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
                RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
                MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                Cast<T2, T, castTraitRintZero>(vreg_exp_f8e5m2, vreg_exp, preg_all);
                DataCopy(vreg_exp_merge_f8e5m2_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e5m2, vreg_exp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_f8e4m3;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
                MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                Cast<T2, T, castTraitRintZero>(vreg_exp_f8e4m3, vreg_exp, preg_all);
                DataCopy(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3, vreg_exp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
                RegTensor<hifloat8_t> vreg_exp_hif8;
                RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
                RegTensor<hifloat8_t> vreg_exp_merge_hif8;
                MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                Cast<T2, T, castTraitZero>(vreg_exp_hif8, vreg_exp, preg_all);
                DataCopy(vreg_exp_merge_hif8_indexes, indexesUb);
                Gather(vreg_exp_merge_hif8, vreg_exp_hif8, vreg_exp_merge_hif8_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
            } else {
                Cast<T2, T, castTraitZero>(vreg_exp_f16, vreg_exp, preg_all_b16);
                DeInterleave(vreg_dst_even_f16, vreg_dst_odd_f16, vreg_exp_f16, vreg_exp_f16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_dst_even_f16, blockStride, repeatStride, preg_src_n_b16);
            }
        }
        vstas(ureg_exp_sum, expSumUb, 0, POST_UPDATE);
    }
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
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb3 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 2* ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb4 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 3* ((s1BaseSize >> 1) + 1) * (128);

    __ubuf__ OUTPUT_T * pseUb = (__ubuf__ OUTPUT_T*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    uint64_t maskUb1 = maskTensor.GetPhyAddr();
    uint64_t maskUb2 = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t maskUb3 = maskTensor.GetPhyAddr() + floatRepSize * 2;
    uint64_t maskUb4 = maskTensor.GetPhyAddr() + floatRepSize * 3;
    uint64_t maskUb5 = maskTensor.GetPhyAddr() + floatRepSize * 4;
    uint64_t maskUb6 = maskTensor.GetPhyAddr() + floatRepSize * 5;
    uint64_t maskUb7 = maskTensor.GetPhyAddr() + floatRepSize * 6;
    uint64_t maskUb8 = maskTensor.GetPhyAddr() + floatRepSize * 7;

    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t oriTailN1 = originN - floatRepSize * 4 < floatRepSize ? originN - floatRepSize * 4 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 5) <= 0 ? 0 : originN - floatRepSize * 5;
    const uint32_t oriTailN3 = static_cast<int32_t>(originN - floatRepSize * 6) <= 0 ? 0 : originN - floatRepSize * 6;
    const uint32_t oriTailN4 = static_cast<int32_t>(originN - floatRepSize * 7) <= 0 ? 0 : originN - floatRepSize * 7;

    const uint32_t tailN1 = s2BaseSize - floatRepSize * 4;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 5;
    const uint32_t tailN3 = s2BaseSize - floatRepSize * 6;
    const uint32_t tailN4 = s2BaseSize - floatRepSize * 7;

    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltOriTailN3 = oriTailN3;
    uint32_t pltOriTailN4 = oriTailN4;

    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    uint32_t pltTailN3 = tailN3;
    uint32_t pltTailN4 = tailN4;
    float divValue = 1.0f / keepProb;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel5;
        RegTensor<float> vreg_sel6;
        RegTensor<float> vreg_sel7;
        RegTensor<float> vreg_sel8;
        RegTensor<float> vreg_sel5_new;
        RegTensor<float> vreg_sel6_new;
        RegTensor<float> vreg_sel7_new;
        RegTensor<float> vreg_sel8_new;


        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x5;
        RegTensor<float> vreg_input_x6;
        RegTensor<float> vreg_input_x7;
        RegTensor<float> vreg_input_x8;
        RegTensor<float> vreg_input_x5_new;
        RegTensor<float> vreg_input_x6_new;
        RegTensor<float> vreg_input_x7_new;
        RegTensor<float> vreg_input_x8_new;

        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_max_tmp4;

        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;

        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_sum4;

        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_exp_even3;
        RegTensor<float> vreg_exp_odd3;
        RegTensor<float> vreg_exp_even4;
        RegTensor<float> vreg_exp_odd4;

        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_pse5;
        RegTensor<float> vreg_pse6;
        RegTensor<float> vreg_pse7;
        RegTensor<float> vreg_pse8;

        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp_even3_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd3_bf16;
        RegTensor<bfloat16_t> vreg_exp_even4_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd4_bf16;

        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_exp3_bf16;
        RegTensor<bfloat16_t> vreg_exp4_bf16;

        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse_bf16_src3;
        RegTensor<bfloat16_t> vreg_pse_bf16_src4;

        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        RegTensor<bfloat16_t> vreg_pse5_bf16;
        RegTensor<bfloat16_t> vreg_pse6_bf16;
        RegTensor<bfloat16_t> vreg_pse7_bf16;
        RegTensor<bfloat16_t> vreg_pse8_bf16;

        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp_even3_f16;
        RegTensor<half> vreg_exp_odd3_f16;
        RegTensor<half> vreg_exp_even4_f16;
        RegTensor<half> vreg_exp_odd4_f16;

        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_exp3_f16;
        RegTensor<half> vreg_exp4_f16;

        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse_f16_src3;
        RegTensor<half> vreg_pse_f16_src4;

        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
        RegTensor<half> vreg_pse5_f16;
        RegTensor<half> vreg_pse6_f16;
        RegTensor<half> vreg_pse7_f16;
        RegTensor<half> vreg_pse8_f16;
		
        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

        MaskReg preg_tail_n1 = UpdateMask<float>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<float>(pltOriTailN1);

        MaskReg preg_tail_n2 = UpdateMask<float>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<float>(pltOriTailN2);

        MaskReg preg_tail_n3 = UpdateMask<T>(pltTailN3);
        MaskReg preg_ori_tail_n3 = UpdateMask<T>(pltOriTailN3);

        MaskReg preg_tail_n4 = UpdateMask<T>(pltTailN4);
        MaskReg preg_ori_tail_n4 = UpdateMask<T>(pltOriTailN4);

        MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();

        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;
        MaskReg preg_compare5;
        MaskReg preg_compare6;
        MaskReg preg_compare7;
        MaskReg preg_compare8;

        Duplicate(vreg_min, minValue);
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x1, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
            DataCopy(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
            DataCopy(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);
            DataCopy(vreg_input_x5, srcUb + floatRepSize * 4 + i * s2BaseSize);
            DataCopy(vreg_input_x6, srcUb + floatRepSize * 5 + i * s2BaseSize);
            DataCopy(vreg_input_x7, srcUb + floatRepSize * 6 + i * s2BaseSize);
            DataCopy(vreg_input_x8, srcUb + floatRepSize * 7 + i * s2BaseSize);

            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);

                Muls(vreg_input_x5, vreg_input_x5, scale, preg_ori_tail_n1);
                Muls(vreg_input_x6, vreg_input_x6, scale, preg_ori_tail_n2);
                Muls(vreg_input_x7, vreg_input_x7, scale, preg_ori_tail_n3);
                Muls(vreg_input_x8, vreg_input_x8, scale, preg_ori_tail_n4);
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (IsSameType<T2, bfloat16_t>::value) {
                    DataCopy(vreg_pse_bf16_src1, pseUb + i * pseStride);
                    DataCopy(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src3, pseUb + floatRepSize * 4 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src4, pseUb + floatRepSize * 6 + i * pseStride);

                    Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
                    Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
                    Interleave(vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse_bf16_src3, vreg_pse_bf16_src3);
                    Interleave(vreg_pse7_bf16, vreg_pse8_bf16, vreg_pse_bf16_src4, vreg_pse_bf16_src4);

                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse5, vreg_pse5_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse6, vreg_pse6_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse7, vreg_pse7_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse8, vreg_pse8_bf16, preg_all_b16);
                } else if constexpr (IsSameType<T2, half>::value) {
                    DataCopy(vreg_pse_f16_src1, pseUb + i * pseStride);
                    DataCopy(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    DataCopy(vreg_pse_f16_src3, pseUb + floatRepSize * 4 + i * pseStride);
                    DataCopy(vreg_pse_f16_src4, pseUb + floatRepSize * 6 + i * pseStride);

                    Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
                    Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
                    Interleave(vreg_pse5_f16, vreg_pse6_f16, vreg_pse_f16_src3, vreg_pse_f16_src3);
                    Interleave(vreg_pse7_f16, vreg_pse8_f16, vreg_pse_f16_src4, vreg_pse_f16_src4);

                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse5, vreg_pse5_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse6, vreg_pse6_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse7, vreg_pse7_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse8, vreg_pse8_f16, preg_all_b16);
                }
                Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
                Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
                Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_all);
                Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_all);
                Add(vreg_input_x5, vreg_input_x5, vreg_pse5, preg_ori_tail_n1);
                Add(vreg_input_x6, vreg_input_x6, vreg_pse6, preg_ori_tail_n2);
                Add(vreg_input_x7, vreg_input_x7, vreg_pse7, preg_ori_tail_n3);
                Add(vreg_input_x8, vreg_input_x8, vreg_pse8, preg_ori_tail_n4);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);

                Muls(vreg_input_x5, vreg_input_x5, scale, preg_ori_tail_n1);
                Muls(vreg_input_x6, vreg_input_x6, scale, preg_ori_tail_n2);
                Muls(vreg_input_x7, vreg_input_x7, scale, preg_ori_tail_n3);
                Muls(vreg_input_x8, vreg_input_x8, scale, preg_ori_tail_n4);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare5, (__ubuf__ uint32_t *&)maskUb5, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare6, (__ubuf__ uint32_t *&)maskUb6, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare7, (__ubuf__ uint32_t *&)maskUb7, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare8, (__ubuf__ uint32_t *&)maskUb8, nPadding);

                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel5, vreg_min, vreg_input_x5, preg_compare5);
                Select(vreg_sel6, vreg_min, vreg_input_x6, preg_compare6);
                Select(vreg_sel7, vreg_min, vreg_input_x7, preg_compare7);
                Select(vreg_sel8, vreg_min, vreg_input_x8, preg_compare8);

                Select(vreg_sel5_new, vreg_sel5, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel6_new, vreg_sel6, vreg_min, preg_ori_tail_n2);
                Select(vreg_sel7_new, vreg_sel7, vreg_min, preg_ori_tail_n3);
                Select(vreg_sel8_new, vreg_sel8, vreg_min, preg_ori_tail_n4);

                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_sel3, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 +  i * s2BaseSize, vreg_sel4, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 4 +  i * s2BaseSize, vreg_sel5_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 5 +  i * s2BaseSize, vreg_sel6_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 6 +  i * s2BaseSize, vreg_sel7_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 7 +  i * s2BaseSize, vreg_sel8_new, preg_all);

                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3, vreg_sel4, preg_all);
                Max(vreg_max_tmp3, vreg_sel5_new, vreg_sel6_new, preg_all);
                Max(vreg_max_tmp4, vreg_sel7_new, vreg_sel8_new, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);

                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp3, preg_all);

                ReduceMax(vreg_input_max, vreg_max_tmp3, preg_all);
            } else {
                Select(vreg_input_x5_new, vreg_input_x5, vreg_min, preg_ori_tail_n1);
                Select(vreg_input_x6_new, vreg_input_x6, vreg_min, preg_ori_tail_n2);
                Select(vreg_input_x7_new, vreg_input_x7, vreg_min, preg_ori_tail_n3);
                Select(vreg_input_x8_new, vreg_input_x8, vreg_min, preg_ori_tail_n4);

                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_input_x3, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_input_x4, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 4 + i * s2BaseSize, vreg_input_x5_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 5 + i * s2BaseSize, vreg_input_x6_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 6 + i * s2BaseSize, vreg_input_x7_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 7 + i * s2BaseSize, vreg_input_x8_new, preg_all);

                Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                Max(vreg_max_tmp2, vreg_input_x3, vreg_input_x4, preg_all);
                Max(vreg_max_tmp3, vreg_input_x5_new, vreg_input_x6_new, preg_all);
                Max(vreg_max_tmp4, vreg_input_x7_new, vreg_input_x8_new, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);

                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp3, preg_all);

                ReduceMax(vreg_input_max, vreg_max_tmp3, preg_all);
            }
            DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, maxUb, 0, POST_UPDATE);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max_brc, maxUbStart + i);

            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x1, vreg_input_x2, srcUb + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x3, vreg_input_x4, srcUb + floatRepSize * 2 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x5, vreg_input_x6, srcUb + floatRepSize * 4 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x7, vreg_input_x8, srcUb + floatRepSize * 6 + i * s2BaseSize);


            FusedExpSub(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even3, vreg_input_x5, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd3, vreg_input_x6, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even4, vreg_input_x7, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd4, vreg_input_x8, vreg_max_brc, preg_all);

            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_even3, vreg_exp_odd3, preg_all);
            Add(vreg_exp_sum4, vreg_exp_even4, vreg_exp_odd4, preg_all);

            Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum3, vreg_exp_sum4, preg_all);

            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum3, preg_all);

            ReduceSum(vreg_exp_sum3, vreg_exp_sum3, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)expSumUb), vreg_exp_sum3, ureg_exp_sum, 1);

            if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even3_bf16, vreg_exp_even3, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd3_bf16, vreg_exp_odd3, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even4_bf16, vreg_exp_even4, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd4_bf16, vreg_exp_odd4, preg_all);


                Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp3_bf16, (RegTensor<uint16_t>&)vreg_exp_even3_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd3_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp4_bf16, (RegTensor<uint16_t>&)vreg_exp_even4_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd4_bf16, preg_all_b16);


                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb3), vreg_exp3_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb4), vreg_exp4_bf16, blockStride, repeatStride, preg_all_b16);
            } else if constexpr (IsSameType<T2, half>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even3_f16, vreg_exp_even3, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd3_f16, vreg_exp_odd3, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even4_f16, vreg_exp_even4, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd4_f16, vreg_exp_odd4, preg_all);

                Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16, (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16, (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp3_f16, (RegTensor<uint16_t>&)vreg_exp_even3_f16, (RegTensor<uint16_t>&)vreg_exp_odd3_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp4_f16, (RegTensor<uint16_t>&)vreg_exp_even4_f16, (RegTensor<uint16_t>&)vreg_exp_odd4_f16, preg_all_b16);

                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb3), vreg_exp3_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb4), vreg_exp4_f16, blockStride, repeatStride, preg_all_b16);
            }
        }
        vstas(ureg_exp_sum, expSumUb, 0, POST_UPDATE);
    }
}

// no update, 512 < originN <= 1024
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1NoUpdateGeneralImpl1024(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<OUTPUT_T>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue, float keepProb)
{
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb3 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 2 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb4 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 3 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb5 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 4 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb6 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 5 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb7 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 6 * ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb8 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 7 * ((s1BaseSize >> 1) + 1) * (128);

    __ubuf__ OUTPUT_T * pseUb = (__ubuf__ OUTPUT_T*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    uint64_t maskUb1 = maskTensor.GetPhyAddr();
    uint64_t maskUb2 = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t maskUb3 = maskTensor.GetPhyAddr() + floatRepSize * 2;
    uint64_t maskUb4 = maskTensor.GetPhyAddr() + floatRepSize * 3;
    uint64_t maskUb5 = maskTensor.GetPhyAddr() + floatRepSize * 4;
    uint64_t maskUb6 = maskTensor.GetPhyAddr() + floatRepSize * 5;
    uint64_t maskUb7 = maskTensor.GetPhyAddr() + floatRepSize * 6;
    uint64_t maskUb8 = maskTensor.GetPhyAddr() + floatRepSize * 7;
    uint64_t maskUb9 = maskTensor.GetPhyAddr() + floatRepSize * 8;
    uint64_t maskUb10 = maskTensor.GetPhyAddr() + floatRepSize * 9;
    uint64_t maskUb11 = maskTensor.GetPhyAddr() + floatRepSize * 10;
    uint64_t maskUb12 = maskTensor.GetPhyAddr() + floatRepSize * 11;
    uint64_t maskUb13 = maskTensor.GetPhyAddr() + floatRepSize * 12;
    uint64_t maskUb14 = maskTensor.GetPhyAddr() + floatRepSize * 13;
    uint64_t maskUb15 = maskTensor.GetPhyAddr() + floatRepSize * 14;
    uint64_t maskUb16 = maskTensor.GetPhyAddr() + floatRepSize * 15;

    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
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

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel5;
        RegTensor<float> vreg_sel6;
        RegTensor<float> vreg_sel7;
        RegTensor<float> vreg_sel8;
        RegTensor<float> vreg_sel9;
        RegTensor<float> vreg_sel10;
        RegTensor<float> vreg_sel11;
        RegTensor<float> vreg_sel12;
        RegTensor<float> vreg_sel13;
        RegTensor<float> vreg_sel14;
        RegTensor<float> vreg_sel15;
        RegTensor<float> vreg_sel16;

        RegTensor<float> vreg_sel9_new;
        RegTensor<float> vreg_sel10_new;
        RegTensor<float> vreg_sel11_new;
        RegTensor<float> vreg_sel12_new;
        RegTensor<float> vreg_sel13_new;
        RegTensor<float> vreg_sel14_new;
        RegTensor<float> vreg_sel15_new;
        RegTensor<float> vreg_sel16_new;


        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x5;
        RegTensor<float> vreg_input_x6;
        RegTensor<float> vreg_input_x7;
        RegTensor<float> vreg_input_x8;
        RegTensor<float> vreg_input_x9;
        RegTensor<float> vreg_input_x10;
        RegTensor<float> vreg_input_x11;
        RegTensor<float> vreg_input_x12;
        RegTensor<float> vreg_input_x13;
        RegTensor<float> vreg_input_x14;
        RegTensor<float> vreg_input_x15;
        RegTensor<float> vreg_input_x16;
        RegTensor<float> vreg_input_x9_new;
        RegTensor<float> vreg_input_x10_new;
        RegTensor<float> vreg_input_x11_new;
        RegTensor<float> vreg_input_x12_new;
        RegTensor<float> vreg_input_x13_new;
        RegTensor<float> vreg_input_x14_new;
        RegTensor<float> vreg_input_x15_new;
        RegTensor<float> vreg_input_x16_new;


        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_max_tmp4;
        RegTensor<float> vreg_max_tmp5;
        RegTensor<float> vreg_max_tmp6;
        RegTensor<float> vreg_max_tmp7;
        RegTensor<float> vreg_max_tmp8;

        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;

        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_sum4;
        RegTensor<float> vreg_exp_sum5;
        RegTensor<float> vreg_exp_sum6;
        RegTensor<float> vreg_exp_sum7;
        RegTensor<float> vreg_exp_sum8;

        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_exp_even3;
        RegTensor<float> vreg_exp_odd3;
        RegTensor<float> vreg_exp_even4;
        RegTensor<float> vreg_exp_odd4;
        RegTensor<float> vreg_exp_even5;
        RegTensor<float> vreg_exp_odd5;
        RegTensor<float> vreg_exp_even6;
        RegTensor<float> vreg_exp_odd6;
        RegTensor<float> vreg_exp_even7;
        RegTensor<float> vreg_exp_odd7;
        RegTensor<float> vreg_exp_even8;
        RegTensor<float> vreg_exp_odd8;

        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_pse5;
        RegTensor<float> vreg_pse6;
        RegTensor<float> vreg_pse7;
        RegTensor<float> vreg_pse8;
        RegTensor<float> vreg_pse9;
        RegTensor<float> vreg_pse10;
        RegTensor<float> vreg_pse11;
        RegTensor<float> vreg_pse12;
        RegTensor<float> vreg_pse13;
        RegTensor<float> vreg_pse14;
        RegTensor<float> vreg_pse15;
        RegTensor<float> vreg_pse16;

        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp_even3_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd3_bf16;
        RegTensor<bfloat16_t> vreg_exp_even4_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd4_bf16;
        RegTensor<bfloat16_t> vreg_exp_even5_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd5_bf16;
        RegTensor<bfloat16_t> vreg_exp_even6_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd6_bf16;
        RegTensor<bfloat16_t> vreg_exp_even7_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd7_bf16;
        RegTensor<bfloat16_t> vreg_exp_even8_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd8_bf16;

        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_exp3_bf16;
        RegTensor<bfloat16_t> vreg_exp4_bf16;
        RegTensor<bfloat16_t> vreg_exp5_bf16;
        RegTensor<bfloat16_t> vreg_exp6_bf16;
        RegTensor<bfloat16_t> vreg_exp7_bf16;
        RegTensor<bfloat16_t> vreg_exp8_bf16;

        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse_bf16_src3;
        RegTensor<bfloat16_t> vreg_pse_bf16_src4;
        RegTensor<bfloat16_t> vreg_pse_bf16_src5;
        RegTensor<bfloat16_t> vreg_pse_bf16_src6;
        RegTensor<bfloat16_t> vreg_pse_bf16_src7;
        RegTensor<bfloat16_t> vreg_pse_bf16_src8;

        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        RegTensor<bfloat16_t> vreg_pse5_bf16;
        RegTensor<bfloat16_t> vreg_pse6_bf16;
        RegTensor<bfloat16_t> vreg_pse7_bf16;
        RegTensor<bfloat16_t> vreg_pse8_bf16;
        RegTensor<bfloat16_t> vreg_pse9_bf16;
        RegTensor<bfloat16_t> vreg_pse10_bf16;
        RegTensor<bfloat16_t> vreg_pse11_bf16;
        RegTensor<bfloat16_t> vreg_pse12_bf16;
        RegTensor<bfloat16_t> vreg_pse13_bf16;
        RegTensor<bfloat16_t> vreg_pse14_bf16;
        RegTensor<bfloat16_t> vreg_pse15_bf16;
        RegTensor<bfloat16_t> vreg_pse16_bf16;

        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp_even3_f16;
        RegTensor<half> vreg_exp_odd3_f16;
        RegTensor<half> vreg_exp_even4_f16;
        RegTensor<half> vreg_exp_odd4_f16;
        RegTensor<half> vreg_exp_even5_f16;
        RegTensor<half> vreg_exp_odd5_f16;
        RegTensor<half> vreg_exp_even6_f16;
        RegTensor<half> vreg_exp_odd6_f16;
        RegTensor<half> vreg_exp_even7_f16;
        RegTensor<half> vreg_exp_odd7_f16;
        RegTensor<half> vreg_exp_even8_f16;
        RegTensor<half> vreg_exp_odd8_f16;

        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_exp3_f16;
        RegTensor<half> vreg_exp4_f16;
        RegTensor<half> vreg_exp5_f16;
        RegTensor<half> vreg_exp6_f16;
        RegTensor<half> vreg_exp7_f16;
        RegTensor<half> vreg_exp8_f16;

        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse_f16_src3;
        RegTensor<half> vreg_pse_f16_src4;
        RegTensor<half> vreg_pse_f16_src5;
        RegTensor<half> vreg_pse_f16_src6;
        RegTensor<half> vreg_pse_f16_src7;
        RegTensor<half> vreg_pse_f16_src8;

        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
        RegTensor<half> vreg_pse5_f16;
        RegTensor<half> vreg_pse6_f16;
        RegTensor<half> vreg_pse7_f16;
        RegTensor<half> vreg_pse8_f16;
        RegTensor<half> vreg_pse9_f16;
        RegTensor<half> vreg_pse10_f16;
        RegTensor<half> vreg_pse11_f16;
        RegTensor<half> vreg_pse12_f16;
        RegTensor<half> vreg_pse13_f16;
        RegTensor<half> vreg_pse14_f16;
        RegTensor<half> vreg_pse15_f16;
        RegTensor<half> vreg_pse16_f16;
		
        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

        MaskReg preg_tail_n1 = UpdateMask<float>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<float>(pltOriTailN1);

        MaskReg preg_tail_n2 = UpdateMask<float>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<float>(pltOriTailN2);

        MaskReg preg_tail_n3 = UpdateMask<T>(pltTailN3);
        MaskReg preg_ori_tail_n3 = UpdateMask<T>(pltOriTailN3);

        MaskReg preg_tail_n4 = UpdateMask<T>(pltTailN4);
        MaskReg preg_ori_tail_n4 = UpdateMask<T>(pltOriTailN4);

        MaskReg preg_tail_n5 = UpdateMask<float>(pltTailN5);
        MaskReg preg_ori_tail_n5 = UpdateMask<float>(pltOriTailN5);

        MaskReg preg_tail_n6 = UpdateMask<float>(pltTailN6);
        MaskReg preg_ori_tail_n6 = UpdateMask<float>(pltOriTailN6);

        MaskReg preg_tail_n7 = UpdateMask<T>(pltTailN7);
        MaskReg preg_ori_tail_n7 = UpdateMask<T>(pltOriTailN7);

        MaskReg preg_tail_n8 = UpdateMask<T>(pltTailN8);
        MaskReg preg_ori_tail_n8 = UpdateMask<T>(pltOriTailN8);

        MaskReg preg_reduce_n = CreateMask<float, MaskPattern::VL8>();

        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;
        MaskReg preg_compare5;
        MaskReg preg_compare6;
        MaskReg preg_compare7;
        MaskReg preg_compare8;
        MaskReg preg_compare9;
        MaskReg preg_compare10;
        MaskReg preg_compare11;
        MaskReg preg_compare12;
        MaskReg preg_compare13;
        MaskReg preg_compare14;
        MaskReg preg_compare15;
        MaskReg preg_compare16;

        Duplicate(vreg_min, minValue);
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x1, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
            DataCopy(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
            DataCopy(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);
            DataCopy(vreg_input_x5, srcUb + floatRepSize * 4 + i * s2BaseSize);
            DataCopy(vreg_input_x6, srcUb + floatRepSize * 5 + i * s2BaseSize);
            DataCopy(vreg_input_x7, srcUb + floatRepSize * 6 + i * s2BaseSize);
            DataCopy(vreg_input_x8, srcUb + floatRepSize * 7 + i * s2BaseSize);
            DataCopy(vreg_input_x9, srcUb + floatRepSize * 8 + i * s2BaseSize);
            DataCopy(vreg_input_x10, srcUb + floatRepSize * 9 + i * s2BaseSize);
            DataCopy(vreg_input_x11, srcUb + floatRepSize * 10 + i * s2BaseSize);
            DataCopy(vreg_input_x12, srcUb + floatRepSize * 11 + i * s2BaseSize);
            DataCopy(vreg_input_x13, srcUb + floatRepSize * 12 + i * s2BaseSize);
            DataCopy(vreg_input_x14, srcUb + floatRepSize * 13 + i * s2BaseSize);
            DataCopy(vreg_input_x15, srcUb + floatRepSize * 14 + i * s2BaseSize);
            DataCopy(vreg_input_x16, srcUb + floatRepSize * 15 + i * s2BaseSize);


            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);
                Muls(vreg_input_x5, vreg_input_x5, scale, preg_all);
                Muls(vreg_input_x6, vreg_input_x6, scale, preg_all);
                Muls(vreg_input_x7, vreg_input_x7, scale, preg_all);
                Muls(vreg_input_x8, vreg_input_x8, scale, preg_all);

                Muls(vreg_input_x9, vreg_input_x9, scale, preg_ori_tail_n1);
                Muls(vreg_input_x10, vreg_input_x10, scale, preg_ori_tail_n2);
                Muls(vreg_input_x11, vreg_input_x11, scale, preg_ori_tail_n3);
                Muls(vreg_input_x12, vreg_input_x12, scale, preg_ori_tail_n4);
                Muls(vreg_input_x13, vreg_input_x13, scale, preg_ori_tail_n5);
                Muls(vreg_input_x14, vreg_input_x14, scale, preg_ori_tail_n6);
                Muls(vreg_input_x15, vreg_input_x15, scale, preg_ori_tail_n7);
                Muls(vreg_input_x16, vreg_input_x16, scale, preg_ori_tail_n8);
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (IsSameType<T2, bfloat16_t>::value) {
                    DataCopy(vreg_pse_bf16_src1, pseUb + i * pseStride);
                    DataCopy(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src3, pseUb + floatRepSize * 4 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src4, pseUb + floatRepSize * 6 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src5, pseUb + floatRepSize * 8 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src6, pseUb + floatRepSize * 10 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src7, pseUb + floatRepSize * 12 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src8, pseUb + floatRepSize * 14 + i * pseStride);

                    Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
                    Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
                    Interleave(vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse_bf16_src3, vreg_pse_bf16_src3);
                    Interleave(vreg_pse7_bf16, vreg_pse8_bf16, vreg_pse_bf16_src4, vreg_pse_bf16_src4);
                    Interleave(vreg_pse9_bf16, vreg_pse10_bf16, vreg_pse_bf16_src5, vreg_pse_bf16_src5);
                    Interleave(vreg_pse11_bf16, vreg_pse12_bf16, vreg_pse_bf16_src6, vreg_pse_bf16_src6);
                    Interleave(vreg_pse13_bf16, vreg_pse14_bf16, vreg_pse_bf16_src7, vreg_pse_bf16_src7);
                    Interleave(vreg_pse15_bf16, vreg_pse16_bf16, vreg_pse_bf16_src8, vreg_pse_bf16_src8);


                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse5, vreg_pse5_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse6, vreg_pse6_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse7, vreg_pse7_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse8, vreg_pse8_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse9, vreg_pse9_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse10, vreg_pse10_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse11, vreg_pse11_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse12, vreg_pse12_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse13, vreg_pse13_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse14, vreg_pse14_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse15, vreg_pse15_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse16, vreg_pse16_bf16, preg_all_b16);

                } else if constexpr (IsSameType<T2, half>::value) {
                    DataCopy(vreg_pse_f16_src1, pseUb + i * pseStride);
                    DataCopy(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    DataCopy(vreg_pse_f16_src3, pseUb + floatRepSize * 4 + i * pseStride);
                    DataCopy(vreg_pse_f16_src4, pseUb + floatRepSize * 6 + i * pseStride);
                    DataCopy(vreg_pse_f16_src5, pseUb + floatRepSize * 8 + i * pseStride);
                    DataCopy(vreg_pse_f16_src6, pseUb + floatRepSize * 10 + i * pseStride);
                    DataCopy(vreg_pse_f16_src7, pseUb + floatRepSize * 12 + i * pseStride);
                    DataCopy(vreg_pse_f16_src8, pseUb + floatRepSize * 14 + i * pseStride);

                    Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
                    Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
                    Interleave(vreg_pse5_f16, vreg_pse6_f16, vreg_pse_f16_src3, vreg_pse_f16_src3);
                    Interleave(vreg_pse7_f16, vreg_pse8_f16, vreg_pse_f16_src4, vreg_pse_f16_src4);
                    Interleave(vreg_pse9_f16, vreg_pse10_f16, vreg_pse_f16_src5, vreg_pse_f16_src5);
                    Interleave(vreg_pse11_f16, vreg_pse12_f16, vreg_pse_f16_src6, vreg_pse_f16_src6);
                    Interleave(vreg_pse13_f16, vreg_pse14_f16, vreg_pse_f16_src7, vreg_pse_f16_src7);
                    Interleave(vreg_pse15_f16, vreg_pse16_f16, vreg_pse_f16_src8, vreg_pse_f16_src8);

                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse5, vreg_pse5_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse6, vreg_pse6_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse7, vreg_pse7_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse8, vreg_pse8_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse9, vreg_pse9_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse10, vreg_pse10_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse11, vreg_pse11_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse12, vreg_pse12_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse13, vreg_pse13_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse14, vreg_pse14_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse15, vreg_pse15_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse16, vreg_pse16_f16, preg_all_b16);
                }
                Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
                Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
                Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_all);
                Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_all);
                Add(vreg_input_x5, vreg_input_x5, vreg_pse5, preg_all);
                Add(vreg_input_x6, vreg_input_x6, vreg_pse6, preg_all);
                Add(vreg_input_x7, vreg_input_x7, vreg_pse7, preg_all);
                Add(vreg_input_x8, vreg_input_x8, vreg_pse8, preg_all);
                Add(vreg_input_x9, vreg_input_x9, vreg_pse9, preg_ori_tail_n1);
                Add(vreg_input_x10, vreg_input_x10, vreg_pse10, preg_ori_tail_n2);
                Add(vreg_input_x11, vreg_input_x11, vreg_pse11, preg_ori_tail_n3);
                Add(vreg_input_x12, vreg_input_x12, vreg_pse12, preg_ori_tail_n4);
                Add(vreg_input_x13, vreg_input_x13, vreg_pse13, preg_ori_tail_n5);
                Add(vreg_input_x14, vreg_input_x14, vreg_pse14, preg_ori_tail_n6);
                Add(vreg_input_x15, vreg_input_x15, vreg_pse15, preg_ori_tail_n7);
                Add(vreg_input_x16, vreg_input_x16, vreg_pse16, preg_ori_tail_n8);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);
                Muls(vreg_input_x5, vreg_input_x5, scale, preg_all);
                Muls(vreg_input_x6, vreg_input_x6, scale, preg_all);
                Muls(vreg_input_x7, vreg_input_x7, scale, preg_all);
                Muls(vreg_input_x8, vreg_input_x8, scale, preg_all);

                Muls(vreg_input_x9, vreg_input_x9, scale, preg_ori_tail_n1);
                Muls(vreg_input_x10, vreg_input_x10, scale, preg_ori_tail_n2);
                Muls(vreg_input_x11, vreg_input_x11, scale, preg_ori_tail_n3);
                Muls(vreg_input_x12, vreg_input_x12, scale, preg_ori_tail_n4);
                Muls(vreg_input_x13, vreg_input_x13, scale, preg_ori_tail_n5);
                Muls(vreg_input_x14, vreg_input_x14, scale, preg_ori_tail_n6);
                Muls(vreg_input_x15, vreg_input_x15, scale, preg_ori_tail_n7);
                Muls(vreg_input_x16, vreg_input_x16, scale, preg_ori_tail_n8);
            }

            if constexpr (hasAtten == 1) {
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare5, (__ubuf__ uint32_t *&)maskUb5, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare6, (__ubuf__ uint32_t *&)maskUb6, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare7, (__ubuf__ uint32_t *&)maskUb7, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare8, (__ubuf__ uint32_t *&)maskUb8, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare9, (__ubuf__ uint32_t *&)maskUb9, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare10, (__ubuf__ uint32_t *&)maskUb10, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare11, (__ubuf__ uint32_t *&)maskUb11, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare12, (__ubuf__ uint32_t *&)maskUb12, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare13, (__ubuf__ uint32_t *&)maskUb13, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare14, (__ubuf__ uint32_t *&)maskUb14, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare15, (__ubuf__ uint32_t *&)maskUb15, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare16, (__ubuf__ uint32_t *&)maskUb16, nPadding);
                
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel5, vreg_min, vreg_input_x5, preg_compare5);
                Select(vreg_sel6, vreg_min, vreg_input_x6, preg_compare6);
                Select(vreg_sel7, vreg_min, vreg_input_x7, preg_compare7);
                Select(vreg_sel8, vreg_min, vreg_input_x8, preg_compare8);
                Select(vreg_sel9, vreg_min, vreg_input_x9, preg_compare9);
                Select(vreg_sel10, vreg_min, vreg_input_x10, preg_compare10);
                Select(vreg_sel11, vreg_min, vreg_input_x11, preg_compare11);
                Select(vreg_sel12, vreg_min, vreg_input_x12, preg_compare12);
                Select(vreg_sel13, vreg_min, vreg_input_x13, preg_compare13);
                Select(vreg_sel14, vreg_min, vreg_input_x14, preg_compare14);
                Select(vreg_sel15, vreg_min, vreg_input_x15, preg_compare15);
                Select(vreg_sel16, vreg_min, vreg_input_x16, preg_compare16);

                Select(vreg_sel9_new, vreg_sel9, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel10_new, vreg_sel10, vreg_min, preg_ori_tail_n2);
                Select(vreg_sel11_new, vreg_sel11, vreg_min, preg_ori_tail_n3);
                Select(vreg_sel12_new, vreg_sel12, vreg_min, preg_ori_tail_n4);
                Select(vreg_sel13_new, vreg_sel13, vreg_min, preg_ori_tail_n5);
                Select(vreg_sel14_new, vreg_sel14, vreg_min, preg_ori_tail_n6);
                Select(vreg_sel15_new, vreg_sel15, vreg_min, preg_ori_tail_n7);
                Select(vreg_sel16_new, vreg_sel16, vreg_min, preg_ori_tail_n8);

                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_sel3, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 +  i * s2BaseSize, vreg_sel4, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 4 +  i * s2BaseSize, vreg_sel5, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 5 +  i * s2BaseSize, vreg_sel6, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 6 +  i * s2BaseSize, vreg_sel7, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 7 +  i * s2BaseSize, vreg_sel8, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 8 +  i * s2BaseSize, vreg_sel9_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 9 +  i * s2BaseSize, vreg_sel10_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 10 +  i * s2BaseSize, vreg_sel11_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 11 +  i * s2BaseSize, vreg_sel12_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 12 +  i * s2BaseSize, vreg_sel13_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 13 +  i * s2BaseSize, vreg_sel14_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 14 +  i * s2BaseSize, vreg_sel15_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 15 +  i * s2BaseSize, vreg_sel16_new, preg_all);

                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3, vreg_sel4, preg_all);
                Max(vreg_max_tmp3, vreg_sel5, vreg_sel6, preg_all);
                Max(vreg_max_tmp4, vreg_sel7, vreg_sel8, preg_all);
                Max(vreg_max_tmp5, vreg_sel9_new, vreg_sel10_new, preg_all);
                Max(vreg_max_tmp6, vreg_sel11_new, vreg_sel12_new, preg_all);
                Max(vreg_max_tmp7, vreg_sel13_new, vreg_sel14_new, preg_all);
                Max(vreg_max_tmp8, vreg_sel15_new, vreg_sel16_new, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);
                Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp6, preg_all);
                Max(vreg_max_tmp7, vreg_max_tmp7, vreg_max_tmp8, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp3, preg_all);
                Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp7, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp5, preg_all);

                ReduceMax(vreg_input_max, vreg_max_tmp1, preg_all);
            } else {
                Select(vreg_input_x9_new, vreg_input_x9, vreg_min, preg_ori_tail_n1);
                Select(vreg_input_x10_new, vreg_input_x10, vreg_min, preg_ori_tail_n2);
                Select(vreg_input_x11_new, vreg_input_x11, vreg_min, preg_ori_tail_n3);
                Select(vreg_input_x12_new, vreg_input_x12, vreg_min, preg_ori_tail_n4);
                Select(vreg_input_x13_new, vreg_input_x13, vreg_min, preg_ori_tail_n5);
                Select(vreg_input_x14_new, vreg_input_x14, vreg_min, preg_ori_tail_n6);
                Select(vreg_input_x15_new, vreg_input_x15, vreg_min, preg_ori_tail_n7);
                Select(vreg_input_x16_new, vreg_input_x16, vreg_min, preg_ori_tail_n8);


                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_input_x3, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_input_x4, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 4 + i * s2BaseSize, vreg_input_x5, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 5 + i * s2BaseSize, vreg_input_x6, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 6 + i * s2BaseSize, vreg_input_x7, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 7 + i * s2BaseSize, vreg_input_x8, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 8 + i * s2BaseSize, vreg_input_x9_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 9 + i * s2BaseSize, vreg_input_x10_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 10 + i * s2BaseSize, vreg_input_x11_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 11 + i * s2BaseSize, vreg_input_x12_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 12 + i * s2BaseSize, vreg_input_x13_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 13 + i * s2BaseSize, vreg_input_x14_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 14 + i * s2BaseSize, vreg_input_x15_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 15 + i * s2BaseSize, vreg_input_x16_new, preg_all);

                Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                Max(vreg_max_tmp2, vreg_input_x3, vreg_input_x4, preg_all);
                Max(vreg_max_tmp3, vreg_input_x5, vreg_input_x6, preg_all);
                Max(vreg_max_tmp4, vreg_input_x7, vreg_input_x8, preg_all);
                Max(vreg_max_tmp5, vreg_input_x9_new, vreg_input_x10_new, preg_all);
                Max(vreg_max_tmp6, vreg_input_x11_new, vreg_input_x12_new, preg_all);
                Max(vreg_max_tmp7, vreg_input_x13_new, vreg_input_x14_new, preg_all);
                Max(vreg_max_tmp8, vreg_input_x15_new, vreg_input_x16_new, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);
                Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp6, preg_all);
                Max(vreg_max_tmp7, vreg_max_tmp7, vreg_max_tmp8, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp3, preg_all);
                Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp7, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp5, preg_all);

                ReduceMax(vreg_input_max, vreg_max_tmp1, preg_all);
            }
            DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, maxUb, 0, POST_UPDATE);
        if constexpr (hasDrop == 1) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max_brc, maxUbStart + i);

            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x1, vreg_input_x2, srcUb + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x3, vreg_input_x4, srcUb + floatRepSize * 2 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x5, vreg_input_x6, srcUb + floatRepSize * 4 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x7, vreg_input_x8, srcUb + floatRepSize * 6 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x9, vreg_input_x10, srcUb + floatRepSize * 8 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x11, vreg_input_x12, srcUb + floatRepSize * 10 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x13, vreg_input_x14, srcUb + floatRepSize * 12 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x15, vreg_input_x16, srcUb + floatRepSize * 14 + i * s2BaseSize);


            FusedExpSub(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even3, vreg_input_x5, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd3, vreg_input_x6, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even4, vreg_input_x7, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd4, vreg_input_x8, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even5, vreg_input_x9, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd5, vreg_input_x10, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even6, vreg_input_x11, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd6, vreg_input_x12, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even7, vreg_input_x13, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd7, vreg_input_x14, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_even8, vreg_input_x15, vreg_max_brc, preg_all);
            FusedExpSub(vreg_exp_odd8, vreg_input_x16, vreg_max_brc, preg_all);

            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_even3, vreg_exp_odd3, preg_all);
            Add(vreg_exp_sum4, vreg_exp_even4, vreg_exp_odd4, preg_all);
            Add(vreg_exp_sum5, vreg_exp_even5, vreg_exp_odd5, preg_all);
            Add(vreg_exp_sum6, vreg_exp_even6, vreg_exp_odd6, preg_all);
            Add(vreg_exp_sum7, vreg_exp_even7, vreg_exp_odd7, preg_all);
            Add(vreg_exp_sum8, vreg_exp_even8, vreg_exp_odd8, preg_all);

            Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum3, vreg_exp_sum4, preg_all);
            Add(vreg_exp_sum5, vreg_exp_sum5, vreg_exp_sum6, preg_all);
            Add(vreg_exp_sum7, vreg_exp_sum7, vreg_exp_sum8, preg_all);

            Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum3, preg_all);
            Add(vreg_exp_sum5, vreg_exp_sum5, vreg_exp_sum7, preg_all);

            Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum5, preg_all);

            ReduceSum(vreg_exp_sum1, vreg_exp_sum1, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)expSumUb), vreg_exp_sum1, ureg_exp_sum, 1);

            if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even3_bf16, vreg_exp_even3, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd3_bf16, vreg_exp_odd3, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even4_bf16, vreg_exp_even4, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd4_bf16, vreg_exp_odd4, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even5_bf16, vreg_exp_even5, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd5_bf16, vreg_exp_odd5, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even6_bf16, vreg_exp_even6, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd6_bf16, vreg_exp_odd6, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even7_bf16, vreg_exp_even7, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd7_bf16, vreg_exp_odd7, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even8_bf16, vreg_exp_even8, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd8_bf16, vreg_exp_odd8, preg_all);


                Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp3_bf16, (RegTensor<uint16_t>&)vreg_exp_even3_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd3_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp4_bf16, (RegTensor<uint16_t>&)vreg_exp_even4_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd4_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp5_bf16, (RegTensor<uint16_t>&)vreg_exp_even5_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd5_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp6_bf16, (RegTensor<uint16_t>&)vreg_exp_even6_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd6_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp7_bf16, (RegTensor<uint16_t>&)vreg_exp_even7_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd7_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp8_bf16, (RegTensor<uint16_t>&)vreg_exp_even8_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd8_bf16, preg_all_b16);


                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb3), vreg_exp3_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb4), vreg_exp4_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb5), vreg_exp5_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb6), vreg_exp6_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb7), vreg_exp7_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb8), vreg_exp8_bf16, blockStride, repeatStride, preg_all_b16);

            } else if constexpr (IsSameType<T2, half>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even3_f16, vreg_exp_even3, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd3_f16, vreg_exp_odd3, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even4_f16, vreg_exp_even4, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd4_f16, vreg_exp_odd4, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even5_f16, vreg_exp_even5, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd5_f16, vreg_exp_odd5, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even6_f16, vreg_exp_even6, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd6_f16, vreg_exp_odd6, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even7_f16, vreg_exp_even7, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd7_f16, vreg_exp_odd7, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even8_f16, vreg_exp_even8, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd8_f16, vreg_exp_odd8, preg_all);

                Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16, (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16, (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp3_f16, (RegTensor<uint16_t>&)vreg_exp_even3_f16, (RegTensor<uint16_t>&)vreg_exp_odd3_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp4_f16, (RegTensor<uint16_t>&)vreg_exp_even4_f16, (RegTensor<uint16_t>&)vreg_exp_odd4_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp5_f16, (RegTensor<uint16_t>&)vreg_exp_even5_f16, (RegTensor<uint16_t>&)vreg_exp_odd5_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp6_f16, (RegTensor<uint16_t>&)vreg_exp_even6_f16, (RegTensor<uint16_t>&)vreg_exp_odd6_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp7_f16, (RegTensor<uint16_t>&)vreg_exp_even7_f16, (RegTensor<uint16_t>&)vreg_exp_odd7_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp8_f16, (RegTensor<uint16_t>&)vreg_exp_even8_f16, (RegTensor<uint16_t>&)vreg_exp_odd8_f16, preg_all_b16);

                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb3), vreg_exp3_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb4), vreg_exp4_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb5), vreg_exp5_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb6), vreg_exp6_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb7), vreg_exp7_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb8), vreg_exp8_f16, blockStride, repeatStride, preg_all_b16);
            }
        }
        vstas(ureg_exp_sum, expSumUb, 0, POST_UPDATE);
    }
}

template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = GT_64_AND_LTE_128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1NoUpdate(
    const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb)
{
    if constexpr (oriNRange == GT_128_AND_LTE_256) {
    ProcessVec1NoUpdateGeneralImpl256RegbaseV2<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(dstTensor,
        expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, maskTensor, pseTensor,
        dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift, scale, minValue, keepProb);
    } else if constexpr (oriNRange == EQ_128) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1NoUpdateImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop, isMlaSgd>(dstTensor, indexesTensor, expSumTensor,
            maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor,
            sharedTmpBuffer, m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb);
    } else if constexpr (oriNRange == GT_0_AND_LTE_64) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1NoUpdateImpl64<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop, isMlaSgd>(dstTensor, indexesTensor, expSumTensor,
            maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor,
            sharedTmpBuffer, m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb);
    } else if constexpr (oriNRange == GT_256_AND_LTE_512) {
        ProcessVec1NoUpdateGeneralImpl512<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(dstTensor, expSumTensor,
            maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor,
            sharedTmpBuffer, m, originN, pseStride, slopes, posShift, scale, minValue, keepProb);
    } else if constexpr (oriNRange == GT_512_AND_LTE_1024) {
        ProcessVec1NoUpdateGeneralImpl1024<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(dstTensor,
            expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, maskTensor, pseTensor,
            dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift, scale, minValue, keepProb);
    } else { // GT_64_AND_LTE_128
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1NoUpdateGeneralImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop, isMlaSgd>(dstTensor, indexesTensor, 
            expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor, maskTensor, pseTensor,
            dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb);
    }
}


// update, 128 < originN <= 256
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1UpdateGeneralImpl256(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const T minValue, float keepProb)
{
    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    const uint32_t oriTailN1 = originN - floatRepSize * 2 < floatRepSize ? originN - floatRepSize * 2 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 3) <= 0 ? 0 : originN - floatRepSize * 3;
    const uint32_t tailN1 = s2BaseSize - floatRepSize * 2;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 3;
    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    float divValue = 1.0f / keepProb;
    uint32_t pltN = s2BaseSize;

    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    uint64_t maskUb1 = maskTensor.GetPhyAddr();
    uint64_t maskUb2 = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t maskUb3 = maskTensor.GetPhyAddr() + floatRepSize * 2;
    uint64_t maskUb4 = maskTensor.GetPhyAddr() + floatRepSize * 3;
    uint64_t dropMaskUb1 = dropTensor.GetPhyAddr();
    uint64_t dropMaskUb2 = dropTensor.GetPhyAddr() + s2BaseSize / 16;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_zero;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;

        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
        MaskReg preg_tail_n1 = UpdateMask<T>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<T>(pltOriTailN1);
        MaskReg preg_tail_n2 = UpdateMask<T>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<T>(pltOriTailN2);
        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;
        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;

        Duplicate(vreg_min, minValue);
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Arange(vreg_alibi1, posShift);
            Arange(vreg_alibi2, posShift + floatRepSize);
            Arange(vreg_alibi3, posShift + floatRepSize * 2);
            Arange(vreg_alibi4, posShift + floatRepSize * 3);
        }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x1, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
            DataCopy(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
            DataCopy(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);  // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                              pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Abs(vreg_pse1, vreg_alibi1, preg_all);
                    Abs(vreg_pse2, vreg_alibi2, preg_all);
                    Abs(vreg_pse3, vreg_alibi3, preg_all);
                    Abs(vreg_pse4, vreg_alibi4, preg_all);
                    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                        Sqrt(vreg_pse1, vreg_pse1, preg_all);
                        Sqrt(vreg_pse2, vreg_pse2, preg_all);
                        Sqrt(vreg_pse3, vreg_pse3, preg_all);
                        Sqrt(vreg_pse4, vreg_pse4, preg_all);
                    }
                    Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
                    Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
                    Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
                    Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
                    Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
                    Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
                    Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
                    Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
                } else {
                    if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                        DataCopy(vreg_pse_bf16_src1, pseUb + i * pseStride);
                        DataCopy(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                        Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
                        Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
                    } else if constexpr (IsSameType<pseShiftType, half>::value) {
                        DataCopy(vreg_pse_f16_src1, pseUb + i * pseStride);
                        DataCopy(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                        Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
                        Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
                    }
                }
                Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
                Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
                Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
                Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);  // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);  
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);              
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_sel3_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp3, preg_all);
            } else {
                Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
                Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_input_x3_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_input_x4_new, preg_all);
                Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp3, preg_all);
            }

            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, tmpMaxUb, 0, POST_UPDATE);
        DataCopy(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy(vreg_input_max, tmpMaxUb2); // 获取新的max[s1, 1]
        
        Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all); // 计算新、旧max的最大值
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
        if constexpr (hasDrop == 1) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max, tmpMaxUb2 + i);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x1, vreg_input_x2, srcUb + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x3, vreg_input_x4, srcUb + floatRepSize * 2 + i * s2BaseSize);
            FusedExpSub(vreg_exp_even1, vreg_input_x1, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even2, vreg_input_x3, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            ReduceSum(vreg_exp_sum3, vreg_exp_sum3, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum3, ureg_exp_sum, 1);

            // dropmask compute
            if constexpr (hasDrop == 1) {
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                    preg1, (__ubuf__ uint32_t *&)dropMaskUb1, s2BaseSize >> 3);
                MaskInterleave<half>(preg3, preg4, preg1, preg2);
                MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
                Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
                Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
                Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
                Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                    preg1, (__ubuf__ uint32_t *&)dropMaskUb2, s2BaseSize >> 3);
                MaskInterleave<half>(preg3, preg4, preg1, preg2);
                MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
                Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
                Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
                Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
                Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            }

            if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_n_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_n_b16);
            } else if constexpr (IsSameType<T2, half>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16, (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16, (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_n_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_n_b16);
            }
        }
        vstas(ureg_exp_sum, tmpExpSumUb, 0, POST_UPDATE);
    }
}

// update, 128 < originN <= 256
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__aicore__ inline void ProcessVec1UpdateGeneralImpl256RegbaseV2(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const T minValue, float keepProb)
{
    using REDUCE_TYPE = float;
    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    const uint32_t oriTailN1 = originN - floatRepSize * 2 < floatRepSize ? originN - floatRepSize * 2 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 3) <= 0 ? 0 : originN - floatRepSize * 3;
    const uint32_t tailN1 = s2BaseSize - floatRepSize * 2;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 3;
    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    float divValue = 1.0f / keepProb;
    uint32_t pltN = s2BaseSize;

    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    uint64_t maskUb1 = maskTensor.GetPhyAddr();
    uint64_t maskUb2 = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t maskUb3 = maskTensor.GetPhyAddr() + floatRepSize * 2;
    uint64_t maskUb4 = maskTensor.GetPhyAddr() + floatRepSize * 3;
    uint64_t dropMaskUb1 = dropTensor.GetPhyAddr();
    uint64_t dropMaskUb2 = dropTensor.GetPhyAddr() + s2BaseSize / 16;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<T> vreg_input_x1;
        RegTensor<T> vreg_input_x2;
        RegTensor<REDUCE_TYPE> vreg_input_x1_b32;
        RegTensor<REDUCE_TYPE> vreg_input_x2_b32;
        RegTensor<REDUCE_TYPE> vreg_input_x3_b32;
        RegTensor<REDUCE_TYPE> vreg_input_x4_b32;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;
        RegTensor<REDUCE_TYPE> vreg_exp_sum_b16;
        RegTensor<REDUCE_TYPE> vreg_exp_even1;
        RegTensor<REDUCE_TYPE> vreg_exp_odd1;
        RegTensor<REDUCE_TYPE> vreg_exp_even2;
        RegTensor<REDUCE_TYPE> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_zero;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;

        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
        MaskReg preg_tail_n1 = UpdateMask<T>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<T>(pltOriTailN1);
        MaskReg preg_tail_n2 = UpdateMask<T>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<T>(pltOriTailN2);
        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;
        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;

        Duplicate(vreg_min, minValue);
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Arange(vreg_alibi1, posShift);
            Arange(vreg_alibi2, posShift + floatRepSize);
            Arange(vreg_alibi3, posShift + floatRepSize * 2);
            Arange(vreg_alibi4, posShift + floatRepSize * 3);
        }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x1, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x2, srcUb + halfRepSize + i * s2BaseSize);
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all_b16);  // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all_b16);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all_b16);  // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all_b16);
            }

            DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x1, preg_all_b16);
            DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + halfRepSize + i * s2BaseSize, vreg_input_x2, preg_all_b16);
            
            Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all_b16);
            ReduceMax(vreg_input_max, vreg_max_tmp1, preg_all_b16);


            DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, tmpMaxUb, 0, POST_UPDATE);
        DataCopy(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy(vreg_input_max, tmpMaxUb2); // 获取新的max[s1, 1]
        
        Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all_b16); // 计算新、旧max的最大值
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all_b16);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max, tmpMaxUb2 + i);
            DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(
                vreg_input_x1, srcUb + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(
                vreg_input_x2, srcUb + halfRepSize + i * s2BaseSize);
            MicroAPI::Cast<REDUCE_TYPE, T, castTraitZero>(vreg_input_x1_b32, vreg_input_x1, preg_all_b16);
            MicroAPI::Cast<REDUCE_TYPE, T, castTraitOne>(vreg_input_x2_b32, vreg_input_x1, preg_all_b16);
            MicroAPI::Cast<REDUCE_TYPE, T, castTraitZero>(vreg_input_x3_b32, vreg_input_x2, preg_all_b16);
            MicroAPI::Cast<REDUCE_TYPE, T, castTraitOne>(vreg_input_x4_b32, vreg_input_x2, preg_all_b16);
            FusedExpSub(vreg_exp_even1, vreg_input_x1_b32, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd1, vreg_input_x2_b32, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even2, vreg_input_x3_b32, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd2, vreg_input_x4_b32, vreg_max, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            ReduceSum(vreg_exp_sum3, vreg_exp_sum3, preg_all);
            MicroAPI::Cast<T, REDUCE_TYPE, castTraitZero>(vreg_exp_sum_b16, vreg_exp_sum3, preg_all);
            DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum_b16, ureg_exp_sum, 1);

            if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, REDUCE_TYPE, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_n_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_n_b16);
            } else if constexpr (IsSameType<T2, half>::value) {
                Cast<T2, REDUCE_TYPE, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
                Cast<T2, REDUCE_TYPE, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16, (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16, (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_n_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_n_b16);
            }
        }
        vstas(ureg_exp_sum, tmpExpSumUb, 0, POST_UPDATE);
    }
}

// update, 64 < originN <= 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1UpdateGeneralImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb)
{
    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    const uint32_t oriTailN = originN - floatRepSize;
    const uint32_t tailN = s2BaseSize - floatRepSize;
    const float dScale = scale * dScaleQK;
    uint32_t pltOriTailN = oriTailN;
    uint32_t pltTailN = tailN;
    float divValue = 1.0f / keepProb;
    uint32_t pltN = s2BaseSize;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * x_expUb = nullptr;
    if constexpr (IsSameType<T2, float>::value) {
        x_expUb = expUb + ((s1BaseSize >> 1) + 1) * (s2BaseSize >> 1);
    }
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    uint64_t maskUb = maskTensor.GetPhyAddr();
    uint64_t maskUbUnroll = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t dropMaskUb = dropTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel;
        RegTensor<float> vreg_sel_unroll;
        RegTensor<float> vreg_sel_unroll_new;
        RegTensor<float> vreg_input_x;
        RegTensor<float> vreg_input_x_unroll;
        RegTensor<float> vreg_input_x_unroll_new;
        RegTensor<float> vreg_max_tmp;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_exp_sum;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_exp_even;
        RegTensor<float> vreg_exp_odd;
        RegTensor<float> vreg_pse;
        RegTensor<float> vreg_pse_unroll;
        RegTensor<float> vreg_alibi;
        RegTensor<float> vreg_alibi_unroll;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_zero;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd_bf16;
        RegTensor<bfloat16_t> vreg_exp_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src;
        RegTensor<bfloat16_t> vreg_pse_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
        // half
        RegTensor<half> vreg_exp_even_f16;
        RegTensor<half> vreg_exp_odd_f16;
        RegTensor<half> vreg_exp_f16;
        RegTensor<half> vreg_pse_f16_src;
        RegTensor<half> vreg_pse_f16;
        RegTensor<half> vreg_pse_f16_unroll;

        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
        MaskReg preg_tail_n = UpdateMask<T>(pltTailN);
        MaskReg preg_ori_tail_n = UpdateMask<T>(pltOriTailN);
        MaskReg preg_compare;
        MaskReg preg_compare_unroll;
        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;

        Duplicate(vreg_min, minValue);
        if constexpr (hasAtten == 1 && isMlaSgd) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare, ((__ubuf__ uint32_t*)(maskUb)));
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare_unroll, ((__ubuf__ uint32_t*)(maskUbUnroll)));
        }
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Arange(vreg_alibi, posShift);
            Arange(vreg_alibi_unroll, posShift + 64);
        }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, dScale, preg_all);  // Muls(scale)
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScale, preg_ori_tail_n);
            } else {
                if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                              IsSameType<T2, hifloat8_t>::value) {
                    Muls(vreg_input_x, vreg_input_x, dScaleQK, preg_all);  // Muls(dScaleQK)
                    Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScaleQK, preg_ori_tail_n);
                }
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                              pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Abs(vreg_pse, vreg_alibi, preg_all);
                    Abs(vreg_pse_unroll, vreg_alibi_unroll, preg_all);
                    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                        Sqrt(vreg_pse, vreg_pse, preg_all);
                        Sqrt(vreg_pse_unroll, vreg_pse_unroll, preg_all);
                    }
                    Muls(vreg_pse, vreg_pse, slopes, preg_all);
                    Muls(vreg_pse_unroll, vreg_pse_unroll, slopes, preg_all);
                    Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all);
                    Adds(vreg_alibi_unroll, vreg_alibi_unroll, -1.0f, preg_all);
                } else {
                    if constexpr (IsSameType<pseShiftType, float>::value) {
                        DataCopy(vreg_pse, pseUb + i * pseStride);
                        DataCopy(vreg_pse_unroll, pseUb + i * pseStride + (s2BaseSize >> 1));
                    } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                        DataCopy(vreg_pse_bf16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_bf16_unroll, preg_all_b16);
                    } else {
                        DataCopy(vreg_pse_f16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_f16_unroll, preg_all_b16);
                    }
                }
                Add(vreg_input_x, vreg_input_x, vreg_pse, preg_all);
                Add(vreg_input_x_unroll, vreg_input_x_unroll, vreg_pse_unroll, preg_ori_tail_n);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, scale, preg_all);  // Muls(scale)
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_ori_tail_n);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if (!isMlaSgd) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        preg_compare, (__ubuf__ uint32_t *&)maskUb, nPadding);
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, nPadding);    
                }
                Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
                Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll);
                Select(vreg_sel_unroll_new, vreg_sel_unroll, vreg_min, preg_ori_tail_n);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel_unroll_new, preg_tail_n);
                Max(vreg_max_tmp, vreg_sel, vreg_sel_unroll_new, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp, preg_all);
            } else {
                Select(vreg_input_x_unroll_new, vreg_input_x_unroll, vreg_min, preg_ori_tail_n);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x_unroll_new, preg_tail_n);    
                Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll_new, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp, preg_all);
            }

            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, tmpMaxUb, 0, POST_UPDATE);
        DataCopy(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy(vreg_input_max, tmpMaxUb2); // 获取新的max[s1, 1]
        Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all); // 计算新、旧max的最大值
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
        if constexpr (hasDrop == 1) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max, tmpMaxUb2 + i);

            if constexpr (IsSameType<T2, float>::value) {
                DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
                DataCopy(vreg_input_x_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
            } else {
                DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    vreg_input_x, vreg_input_x_unroll, srcUb + i * s2BaseSize);
            }
            FusedExpSub(vreg_exp_even, vreg_input_x, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
            ReduceSum(vreg_exp_sum, vreg_exp_sum, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum, ureg_exp_sum, 1);

            // dropmask compute
            if constexpr (hasDrop == 1) {
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                    preg1, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
                if constexpr (IsSameType<T2, float>::value) {
                    MaskInterleave<half>(preg5, preg6, preg1, preg2);
                } else {
                    MaskInterleave<half>(preg3, preg4, preg1, preg2);
                    MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
                }
                Select(vreg_sel_drop, vreg_exp_even, vreg_zero, preg5);
                Muls(vreg_exp_even, vreg_sel_drop, divValue, preg_all);
                Select(vreg_sel_drop2, vreg_exp_odd, vreg_zero, preg6);
                Muls(vreg_exp_odd, vreg_sel_drop2, divValue, preg_all);
            }

            if constexpr (IsSameType<T2, float>::value) {
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_even, blockStride, repeatStride, preg_all);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_expUb), vreg_exp_odd, blockStride, repeatStride, preg_all);
            } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd_bf16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, repeatStride, preg_n_b16);
            } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
                RegTensor<fp8_e5m2_t> vreg_exp_even_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_odd_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_merge_tmp_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
                RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_even_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_odd_f8e5m2, preg_all_b8);
                DataCopy(vreg_exp_merge_f8e5m2_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e5m2, vreg_exp_merge_tmp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_even_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_odd_f8e4m3, preg_all_b8);
                DataCopy(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
                RegTensor<hifloat8_t> vreg_exp_even_hif8;
                RegTensor<hifloat8_t> vreg_exp_odd_hif8;
                RegTensor<hifloat8_t> vreg_exp_merge_tmp_hif8;
                RegTensor<hifloat8_t> vreg_exp_merge_hif8;
                RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_hif8, (RegTensor<uint8_t>&)vreg_exp_even_hif8, (RegTensor<uint8_t>&)vreg_exp_odd_hif8, preg_all_b8);
                DataCopy(vreg_exp_merge_hif8_indexes, indexesUb);
                Gather(vreg_exp_merge_hif8, vreg_exp_merge_tmp_hif8, vreg_exp_merge_hif8_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
            } else {
                Cast<T2, T, castTraitZero>(vreg_exp_even_f16, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd_f16, vreg_exp_odd, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp_f16, (RegTensor<uint16_t>&)vreg_exp_even_f16, (RegTensor<uint16_t>&)vreg_exp_odd_f16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_f16, blockStride, repeatStride, preg_n_b16);
            }
        }
        vstas(ureg_exp_sum, tmpExpSumUb, 0, POST_UPDATE);
    }
}

// update, originN == 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1UpdateImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb)
{
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    const float dScale = scale * dScaleQK;
    float divValue = 1.0f / keepProb;

    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * x_expUb = nullptr;
    if constexpr (IsSameType<T2, float>::value) {
        x_expUb = expUb + ((s1BaseSize >> 1) + 1) * (s2BaseSize >> 1);
    }
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    uint64_t maskUb = maskTensor.GetPhyAddr();
    uint64_t maskUbUnroll = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t dropMaskUb = dropTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel;
        RegTensor<float> vreg_sel_unroll;
        RegTensor<float> vreg_input_x;
        RegTensor<float> vreg_input_x_unroll;
        RegTensor<float> vreg_max_tmp;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp_sum;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_even;
        RegTensor<float> vreg_exp_odd;
        RegTensor<float> vreg_pse;
        RegTensor<float> vreg_pse_unroll;
        RegTensor<float> vreg_alibi;
        RegTensor<float> vreg_alibi_unroll;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd_bf16;
        RegTensor<bfloat16_t> vreg_exp_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src;
        RegTensor<bfloat16_t> vreg_pse_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
        // half
        RegTensor<half> vreg_exp_even_f16;
        RegTensor<half> vreg_exp_odd_f16;
        RegTensor<half> vreg_exp_f16;
        RegTensor<half> vreg_pse_f16_src;
        RegTensor<half> vreg_pse_f16;
        RegTensor<half> vreg_pse_f16_unroll;

        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_compare;
        MaskReg preg_compare_unroll;
        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;
        if constexpr (hasAtten == 1) {
            Duplicate(vreg_min, minValue);
            if constexpr (isMlaSgd) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (preg_compare, ((__ubuf__ uint32_t*)(maskUb)));
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (preg_compare_unroll, ((__ubuf__ uint32_t*)(maskUbUnroll)));
            }
        }
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Arange(vreg_alibi, posShift);
            Arange(vreg_alibi_unroll, posShift + 64);
        }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, dScale, preg_all);  // Muls(scale)
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScale, preg_all);
            } else {
                if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                              IsSameType<T2, hifloat8_t>::value) {
                    Muls(vreg_input_x, vreg_input_x, dScaleQK, preg_all);  // Muls(dScaleQK)
                    Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScaleQK, preg_all);
                }
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                              pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Abs(vreg_pse, vreg_alibi, preg_all);
                    Abs(vreg_pse_unroll, vreg_alibi_unroll, preg_all);
                    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                        Sqrt(vreg_pse, vreg_pse, preg_all);
                        Sqrt(vreg_pse_unroll, vreg_pse_unroll, preg_all);
                    }
                    Muls(vreg_pse, vreg_pse, slopes, preg_all);
                    Muls(vreg_pse_unroll, vreg_pse_unroll, slopes, preg_all);
                    Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all);
                    Adds(vreg_alibi_unroll, vreg_alibi_unroll, -1.0f, preg_all);
                } else {
                    if constexpr (IsSameType<pseShiftType, float>::value) {
                        DataCopy(vreg_pse, pseUb + i * pseStride);
                        DataCopy(vreg_pse_unroll, pseUb + i * pseStride + (s2BaseSize >> 1));
                    } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                        DataCopy(vreg_pse_bf16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_bf16_unroll, preg_all_b16);
                    } else {
                        DataCopy(vreg_pse_f16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_f16_unroll, preg_all_b16);
                    }
                }
                Add(vreg_input_x, vreg_input_x, vreg_pse, preg_all);
                Add(vreg_input_x_unroll, vreg_input_x_unroll, vreg_pse_unroll, preg_all);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, scale, preg_all);  // Muls(scale)
                Muls(vreg_input_x_unroll, vreg_input_x_unroll, scale, preg_all);
            }

            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSgd) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        preg_compare, (__ubuf__ uint32_t *&)maskUb, s2BaseSize);
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, s2BaseSize);
                }
                Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
                Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel_unroll, preg_all);
                Max(vreg_max_tmp, vreg_sel, vreg_sel_unroll, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp, preg_all);
            } else {
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x_unroll, preg_all);
                Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp, preg_all);
            }

            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, tmpMaxUb, 0, POST_UPDATE);
        DataCopy(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy(vreg_input_max, tmpMaxUb2); // 获取新的max[s1, 1]
        Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all); // 计算新、旧max的最大值
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
        if constexpr (hasDrop == 1) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max, tmpMaxUb2 + i);
            if constexpr (IsSameType<T2, float>::value) {
                DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
                DataCopy(vreg_input_x_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
            } else {
                DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    vreg_input_x, vreg_input_x_unroll, srcUb + i * s2BaseSize);
            }
            FusedExpSub(vreg_exp_even, vreg_input_x, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
            ReduceSum(vreg_exp_sum, vreg_exp_sum, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum, ureg_exp_sum, 1);

            // dropmask compute
            if constexpr (hasDrop == 1) {
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                    preg1, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
                if constexpr (IsSameType<T2, float>::value) {
                    MaskInterleave<half>(preg5, preg6, preg1, preg2);
                } else {
                    MaskInterleave<half>(preg3, preg4, preg1, preg2);
                    MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
                }
                Select(vreg_sel_drop, vreg_exp_even, vreg_zero, preg5);
                Muls(vreg_exp_even, vreg_sel_drop, divValue, preg_all);
                Select(vreg_sel_drop2, vreg_exp_odd, vreg_zero, preg6);
                Muls(vreg_exp_odd, vreg_sel_drop2, divValue, preg_all);
            }

            if constexpr (IsSameType<T2, float>::value) {
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_even, blockStride, repeatStride, preg_all);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_expUb), vreg_exp_odd, blockStride, repeatStride, preg_all);
            } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd_bf16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_bf16, blockStride, repeatStride, preg_all_b16);
            } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
                RegTensor<fp8_e5m2_t> vreg_exp_even_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_odd_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_merge_tmp_f8e5m2;
                RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
                RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_even_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_odd_f8e5m2, preg_all_b8);
                DataCopy(vreg_exp_merge_f8e5m2_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e5m2, vreg_exp_merge_tmp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_even_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_odd_f8e4m3, preg_all_b8);
                DataCopy(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
                RegTensor<hifloat8_t> vreg_exp_even_hif8;
                RegTensor<hifloat8_t> vreg_exp_odd_hif8;
                RegTensor<hifloat8_t> vreg_exp_merge_tmp_hif8;
                RegTensor<hifloat8_t> vreg_exp_merge_hif8;
                RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all);
                Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_hif8, (RegTensor<uint8_t>&)vreg_exp_even_hif8, (RegTensor<uint8_t>&)vreg_exp_odd_hif8, preg_all_b8);
                DataCopy(vreg_exp_merge_hif8_indexes, indexesUb);
                Gather(vreg_exp_merge_hif8, vreg_exp_merge_tmp_hif8, vreg_exp_merge_hif8_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
            } else {
                Cast<T2, T, castTraitZero>(vreg_exp_even_f16, vreg_exp_even, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd_f16, vreg_exp_odd, preg_all);
                Or((RegTensor<uint16_t>&)vreg_exp_f16, (RegTensor<uint16_t>&)vreg_exp_even_f16, (RegTensor<uint16_t>&)vreg_exp_odd_f16, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_f16, blockStride, repeatStride, preg_all_b16);
            }
        }
        vstas(ureg_exp_sum, tmpExpSumUb, 0, POST_UPDATE);
    }
}

// update, originN <= 64
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1UpdateImpl64(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb)
{
    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    const float dScale = scale * dScaleQK;
    uint32_t pltOriginalN = originN;
    float divValue = 1.0f / keepProb;
    uint32_t pltSrcN = s2BaseSize;
    uint32_t pltSrcN16 = s2BaseSize;
    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    uint64_t maskUb = maskTensor.GetPhyAddr();
    uint64_t dropMaskUb = dropTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel;
        RegTensor<float> vreg_input_x;
        RegTensor<float> vreg_input_x_unroll;
        RegTensor<float> vreg_max_tmp;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp;
        RegTensor<float> vreg_exp_sum;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_pse;
        RegTensor<float> vreg_alibi;
        RegTensor<float> vreg_sel_drop;

        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_exp_bf16;
        RegTensor<bfloat16_t> vreg_dst_even_bf16;
        RegTensor<bfloat16_t> vreg_dst_odd_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src;
        RegTensor<bfloat16_t> vreg_pse_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
        // half
        RegTensor<half> vreg_exp_even_f16;
        RegTensor<half> vreg_exp_f16;
        RegTensor<half> vreg_dst_even_f16;
        RegTensor<half> vreg_dst_odd_f16;
        RegTensor<half> vreg_pse_f16_src;
        RegTensor<half> vreg_pse_f16;
        RegTensor<half> vreg_pse_f16_unroll;

        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

        MaskReg preg_src_n = UpdateMask<T>(pltSrcN);
        MaskReg preg_src_n_b16 = UpdateMask<uint16_t>(pltSrcN16);
        MaskReg preg_ori_src_n = UpdateMask<T>(pltOriginalN);
        MaskReg preg_compare;
        MaskReg preg1;
        MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
        MaskReg preg3;
        MaskReg preg4;

        if constexpr (hasAtten == 1) {
            Duplicate(vreg_min, minValue);
            if constexpr (isMlaSgd) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_DS>
                    (preg_compare, ((__ubuf__ uint32_t*)(maskUb)));
            }
        }
        if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                      pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Arange(vreg_alibi, posShift);
        }
        // x_max = max(src, axis=-1, keepdims=True)
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, dScale, preg_ori_src_n);  // Muls(scale)
            } else {
                if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                              IsSameType<T2, hifloat8_t>::value) {
                    Muls(vreg_input_x, vreg_input_x, dScaleQK, preg_ori_src_n);  // Muls(dScaleQK)
                }
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                              pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Abs(vreg_pse, vreg_alibi, preg_all);
                    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                        Sqrt(vreg_pse, vreg_pse, preg_all);
                    }
                    Muls(vreg_pse, vreg_pse, slopes, preg_all);
                    Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all);
                } else {
                    if constexpr (IsSameType<pseShiftType, float>::value) {
                        DataCopy(vreg_pse, pseUb + i * pseStride);
                    } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                        DataCopy(vreg_pse_bf16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                    } else {
                        DataCopy(vreg_pse_f16_src, pseUb + i * pseStride);
                        Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                        Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                    }
                }
                Add(vreg_input_x, vreg_input_x, vreg_pse, preg_ori_src_n);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, scale, preg_ori_src_n);  // Muls(scale)
            }
            if constexpr (hasAtten == 1) {
                // atten mask
                if constexpr (!isMlaSgd) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        preg_compare, (__ubuf__ uint32_t *&)maskUb, nPadding);
                }
                Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_src_n);
                ReduceMax(vreg_input_max, vreg_sel, preg_ori_src_n);
            } else {
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_src_n);
                ReduceMax(vreg_input_max, vreg_input_x, preg_ori_src_n);
            }

            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, tmpMaxUb, 0, POST_UPDATE);
        DataCopy(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy(vreg_input_max, tmpMaxUb2);
        Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all); // 计算新、旧的最大值
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
        if constexpr (hasDrop == 1) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max, tmpMaxUb2 + i);
            DataCopy(vreg_input_x, srcUb + i * s2BaseSize);
            FusedExpSub(vreg_exp, vreg_input_x, vreg_max, preg_ori_src_n);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            ReduceSum(vreg_exp_sum, vreg_exp, preg_ori_src_n);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum, ureg_exp_sum, 1);

            // dropmask compute
            if constexpr (hasDrop == 1) {
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                    preg1, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
                MaskInterleave<half>(preg3, preg4, preg1, preg2);
                Select(vreg_sel_drop, vreg_exp, vreg_zero, preg3);
                Muls(vreg_exp, vreg_sel_drop, divValue, preg_all);
            }

            if constexpr (IsSameType<T2, float>::value) {
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp, blockStride, repeatStride, preg_all);
            } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_bf16, vreg_exp, preg_all_b16);
                DeInterleave(vreg_dst_even_bf16, vreg_dst_odd_bf16,
                        vreg_exp_bf16, vreg_exp_bf16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_dst_even_bf16, blockStride, repeatStride, preg_src_n_b16);
            }  else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
                RegTensor<fp8_e5m2_t> vreg_exp_f8e5m2;
                RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
                RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_f8e5m2, vreg_exp, preg_all);
                DataCopy(vreg_exp_merge_f8e5m2_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e5m2, vreg_exp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_f8e4m3;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_f8e4m3, vreg_exp, preg_all);
                DataCopy(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3, vreg_exp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
            } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
                RegTensor<hifloat8_t> vreg_exp_hif8;
                RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
                RegTensor<hifloat8_t> vreg_exp_merge_hif8;
                __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
                MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitZero>(vreg_exp_hif8, vreg_exp, preg_all);
                DataCopy(vreg_exp_merge_hif8_indexes, indexesUb);
                Gather(vreg_exp_merge_hif8, vreg_exp_hif8, vreg_exp_merge_hif8_indexes);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
            } else {
                Cast<T2, T, castTraitZero>(vreg_exp_f16, vreg_exp, preg_all_b16);
                DeInterleave(vreg_dst_even_f16, vreg_dst_odd_f16, vreg_exp_f16, vreg_exp_f16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb), vreg_dst_even_f16, blockStride, repeatStride, preg_src_n_b16);
            }
        }
        vstas(ureg_exp_sum, tmpExpSumUb, 0, POST_UPDATE);
    }
}


template <typename T>
__aicore__ inline void UpdateExpSumAndExpMaxImpl(
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor,  const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m)
{
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();

    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();

    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __VEC_SCOPE__
    {
        RegTensor<float> vreg_input_x;
        RegTensor<float> vreg_input_x_unroll;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_exp_sum;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_exp_sum_update;
        UnalignReg ureg_exp_sum;
        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        // 注意：当m大于64的时候需要开启循环
        DataCopy(vreg_max, tmpMaxUb);
        DataCopy(vreg_in_max, inMaxUb);
        FusedExpSub(vreg_exp_max, vreg_in_max, vreg_max, preg_all);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)maxUb, vreg_max, preg_all);
        DataCopy(vreg_in_exp_sum, inExpSumUb);

        // x_sum = exp_max * insum + x_sum
        DataCopy(vreg_exp_sum_brc, tmpExpSumUb);
        Mul(vreg_exp_sum_update, vreg_exp_max, vreg_in_exp_sum, preg_all);
        Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_exp_sum_brc, preg_all);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)expSumUb, vreg_exp_sum_update, preg_all);
    }
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
    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
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

    uint64_t maskUb1 = maskTensor.GetPhyAddr();
    uint64_t maskUb2 = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t maskUb3 = maskTensor.GetPhyAddr() + floatRepSize * 2;
    uint64_t maskUb4 = maskTensor.GetPhyAddr() + floatRepSize * 3;
    uint64_t maskUb5 = maskTensor.GetPhyAddr() + floatRepSize * 4;
    uint64_t maskUb6 = maskTensor.GetPhyAddr() + floatRepSize * 5;
    uint64_t maskUb7 = maskTensor.GetPhyAddr() + floatRepSize * 6;
    uint64_t maskUb8 = maskTensor.GetPhyAddr() + floatRepSize * 7;
    uint64_t maskUb9 = maskTensor.GetPhyAddr() + floatRepSize * 8;
    uint64_t maskUb10 = maskTensor.GetPhyAddr() + floatRepSize * 9;
    uint64_t maskUb11 = maskTensor.GetPhyAddr() + floatRepSize * 10;
    uint64_t maskUb12 = maskTensor.GetPhyAddr() + floatRepSize * 11;
    uint64_t maskUb13 = maskTensor.GetPhyAddr() + floatRepSize * 12;
    uint64_t maskUb14 = maskTensor.GetPhyAddr() + floatRepSize * 13;
    uint64_t maskUb15 = maskTensor.GetPhyAddr() + floatRepSize * 14;
    uint64_t maskUb16 = maskTensor.GetPhyAddr() + floatRepSize * 15;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel5;
        RegTensor<float> vreg_sel6;
        RegTensor<float> vreg_sel7;
        RegTensor<float> vreg_sel8;
        RegTensor<float> vreg_sel9;
        RegTensor<float> vreg_sel10;
        RegTensor<float> vreg_sel11;
        RegTensor<float> vreg_sel12;
        RegTensor<float> vreg_sel13;
        RegTensor<float> vreg_sel14;
        RegTensor<float> vreg_sel15;
        RegTensor<float> vreg_sel16;

        RegTensor<float> vreg_sel9_new;
        RegTensor<float> vreg_sel10_new;
        RegTensor<float> vreg_sel11_new;
        RegTensor<float> vreg_sel12_new;
        RegTensor<float> vreg_sel13_new;
        RegTensor<float> vreg_sel14_new;
        RegTensor<float> vreg_sel15_new;
        RegTensor<float> vreg_sel16_new;


        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x5;
        RegTensor<float> vreg_input_x6;
        RegTensor<float> vreg_input_x7;
        RegTensor<float> vreg_input_x8;
        RegTensor<float> vreg_input_x9;
        RegTensor<float> vreg_input_x10;
        RegTensor<float> vreg_input_x11;
        RegTensor<float> vreg_input_x12;
        RegTensor<float> vreg_input_x13;
        RegTensor<float> vreg_input_x14;
        RegTensor<float> vreg_input_x15;
        RegTensor<float> vreg_input_x16;

        RegTensor<float> vreg_input_x9_new;
        RegTensor<float> vreg_input_x10_new;
        RegTensor<float> vreg_input_x11_new;
        RegTensor<float> vreg_input_x12_new;
        RegTensor<float> vreg_input_x13_new;
        RegTensor<float> vreg_input_x14_new;
        RegTensor<float> vreg_input_x15_new;
        RegTensor<float> vreg_input_x16_new;

        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_max_tmp4;
        RegTensor<float> vreg_max_tmp5;
        RegTensor<float> vreg_max_tmp6;
        RegTensor<float> vreg_max_tmp7;
        RegTensor<float> vreg_max_tmp8;

        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_new;

        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_sum4;
        RegTensor<float> vreg_exp_sum5;
        RegTensor<float> vreg_exp_sum6;
        RegTensor<float> vreg_exp_sum7;
        RegTensor<float> vreg_exp_sum8;

        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;

        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_exp_even3;
        RegTensor<float> vreg_exp_odd3;
        RegTensor<float> vreg_exp_even4;
        RegTensor<float> vreg_exp_odd4;
        RegTensor<float> vreg_exp_even5;
        RegTensor<float> vreg_exp_odd5;
        RegTensor<float> vreg_exp_even6;
        RegTensor<float> vreg_exp_odd6;
        RegTensor<float> vreg_exp_even7;
        RegTensor<float> vreg_exp_odd7;
        RegTensor<float> vreg_exp_even8;
        RegTensor<float> vreg_exp_odd8; 

        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_pse5;
        RegTensor<float> vreg_pse6;
        RegTensor<float> vreg_pse7;
        RegTensor<float> vreg_pse8;
        RegTensor<float> vreg_pse9;
        RegTensor<float> vreg_pse10;
        RegTensor<float> vreg_pse11;
        RegTensor<float> vreg_pse12;
        RegTensor<float> vreg_pse13;
        RegTensor<float> vreg_pse14;
        RegTensor<float> vreg_pse15;
        RegTensor<float> vreg_pse16;

        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp_even3_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd3_bf16;
        RegTensor<bfloat16_t> vreg_exp_even4_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd4_bf16;
        RegTensor<bfloat16_t> vreg_exp_even5_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd5_bf16;
        RegTensor<bfloat16_t> vreg_exp_even6_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd6_bf16;
        RegTensor<bfloat16_t> vreg_exp_even7_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd7_bf16;
        RegTensor<bfloat16_t> vreg_exp_even8_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd8_bf16;

        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_exp3_bf16;
        RegTensor<bfloat16_t> vreg_exp4_bf16;
        RegTensor<bfloat16_t> vreg_exp5_bf16;
        RegTensor<bfloat16_t> vreg_exp6_bf16;
        RegTensor<bfloat16_t> vreg_exp7_bf16;
        RegTensor<bfloat16_t> vreg_exp8_bf16;

        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse_bf16_src3;
        RegTensor<bfloat16_t> vreg_pse_bf16_src4;
        RegTensor<bfloat16_t> vreg_pse_bf16_src5;
        RegTensor<bfloat16_t> vreg_pse_bf16_src6;
        RegTensor<bfloat16_t> vreg_pse_bf16_src7;
        RegTensor<bfloat16_t> vreg_pse_bf16_src8;

        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        RegTensor<bfloat16_t> vreg_pse5_bf16;
        RegTensor<bfloat16_t> vreg_pse6_bf16;
        RegTensor<bfloat16_t> vreg_pse7_bf16;
        RegTensor<bfloat16_t> vreg_pse8_bf16;
        RegTensor<bfloat16_t> vreg_pse9_bf16;
        RegTensor<bfloat16_t> vreg_pse10_bf16;
        RegTensor<bfloat16_t> vreg_pse11_bf16;
        RegTensor<bfloat16_t> vreg_pse12_bf16;
        RegTensor<bfloat16_t> vreg_pse13_bf16;
        RegTensor<bfloat16_t> vreg_pse14_bf16;
        RegTensor<bfloat16_t> vreg_pse15_bf16;
        RegTensor<bfloat16_t> vreg_pse16_bf16;

        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp_even3_f16;
        RegTensor<half> vreg_exp_odd3_f16;
        RegTensor<half> vreg_exp_even4_f16;
        RegTensor<half> vreg_exp_odd4_f16;
        RegTensor<half> vreg_exp_even5_f16;
        RegTensor<half> vreg_exp_odd5_f16;
        RegTensor<half> vreg_exp_even6_f16;
        RegTensor<half> vreg_exp_odd6_f16;
        RegTensor<half> vreg_exp_even7_f16;
        RegTensor<half> vreg_exp_odd7_f16;
        RegTensor<half> vreg_exp_even8_f16;
        RegTensor<half> vreg_exp_odd8_f16;

        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_exp3_f16;
        RegTensor<half> vreg_exp4_f16;
        RegTensor<half> vreg_exp5_f16;
        RegTensor<half> vreg_exp6_f16;
        RegTensor<half> vreg_exp7_f16;
        RegTensor<half> vreg_exp8_f16;

        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse_f16_src3;
        RegTensor<half> vreg_pse_f16_src4;
        RegTensor<half> vreg_pse_f16_src5;
        RegTensor<half> vreg_pse_f16_src6;
        RegTensor<half> vreg_pse_f16_src7;
        RegTensor<half> vreg_pse_f16_src8;

        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
        RegTensor<half> vreg_pse5_f16;
        RegTensor<half> vreg_pse6_f16;
        RegTensor<half> vreg_pse7_f16;
        RegTensor<half> vreg_pse8_f16;
        RegTensor<half> vreg_pse9_f16;
        RegTensor<half> vreg_pse10_f16;
        RegTensor<half> vreg_pse11_f16;
        RegTensor<half> vreg_pse12_f16;
        RegTensor<half> vreg_pse13_f16;
        RegTensor<half> vreg_pse14_f16;
        RegTensor<half> vreg_pse15_f16;
        RegTensor<half> vreg_pse16_f16;

        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);

        MaskReg preg_tail_n1 = UpdateMask<T>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<T>(pltOriTailN1);
        MaskReg preg_tail_n2 = UpdateMask<T>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<T>(pltOriTailN2);
        MaskReg preg_tail_n3 = UpdateMask<T>(pltTailN3);
        MaskReg preg_ori_tail_n3 = UpdateMask<T>(pltOriTailN3);
        MaskReg preg_tail_n4 = UpdateMask<T>(pltTailN4);
        MaskReg preg_ori_tail_n4 = UpdateMask<T>(pltOriTailN4);
        MaskReg preg_tail_n5 = UpdateMask<T>(pltTailN5);
        MaskReg preg_ori_tail_n5 = UpdateMask<T>(pltOriTailN5);
        MaskReg preg_tail_n6 = UpdateMask<T>(pltTailN6);
        MaskReg preg_ori_tail_n6 = UpdateMask<T>(pltOriTailN6);
        MaskReg preg_tail_n7 = UpdateMask<T>(pltTailN7);
        MaskReg preg_ori_tail_n7 = UpdateMask<T>(pltOriTailN7);
        MaskReg preg_tail_n8 = UpdateMask<T>(pltTailN8);
        MaskReg preg_ori_tail_n8 = UpdateMask<T>(pltOriTailN8);


        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;
        MaskReg preg_compare5;
        MaskReg preg_compare6;
        MaskReg preg_compare7;
        MaskReg preg_compare8;
        MaskReg preg_compare9;
        MaskReg preg_compare10;
        MaskReg preg_compare11;
        MaskReg preg_compare12;
        MaskReg preg_compare13;
        MaskReg preg_compare14;
        MaskReg preg_compare15;
        MaskReg preg_compare16;

        Duplicate(vreg_min, minValue);
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x1, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
            DataCopy(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
            DataCopy(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);
            DataCopy(vreg_input_x5, srcUb + floatRepSize * 4 + i * s2BaseSize);
            DataCopy(vreg_input_x6, srcUb + floatRepSize * 5 + i * s2BaseSize);
            DataCopy(vreg_input_x7, srcUb + floatRepSize * 6 + i * s2BaseSize);
            DataCopy(vreg_input_x8, srcUb + floatRepSize * 7 + i * s2BaseSize);
            DataCopy(vreg_input_x9, srcUb + floatRepSize * 8 + i * s2BaseSize);
            DataCopy(vreg_input_x10, srcUb + floatRepSize * 9 + i * s2BaseSize);
            DataCopy(vreg_input_x11, srcUb + floatRepSize * 10 + i * s2BaseSize);
            DataCopy(vreg_input_x12, srcUb + floatRepSize * 11 + i * s2BaseSize);
            DataCopy(vreg_input_x13, srcUb + floatRepSize * 12 + i * s2BaseSize);
            DataCopy(vreg_input_x14, srcUb + floatRepSize * 13 + i * s2BaseSize);
            DataCopy(vreg_input_x15, srcUb + floatRepSize * 14 + i * s2BaseSize);
            DataCopy(vreg_input_x16, srcUb + floatRepSize * 15 + i * s2BaseSize);

            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);
                Muls(vreg_input_x5, vreg_input_x5, scale, preg_all);
                Muls(vreg_input_x6, vreg_input_x6, scale, preg_all);
                Muls(vreg_input_x7, vreg_input_x7, scale, preg_all);
                Muls(vreg_input_x8, vreg_input_x8, scale, preg_all);

                Muls(vreg_input_x9, vreg_input_x9, scale, preg_ori_tail_n1);
                Muls(vreg_input_x10, vreg_input_x10, scale, preg_ori_tail_n2);
                Muls(vreg_input_x11, vreg_input_x11, scale, preg_ori_tail_n3);
                Muls(vreg_input_x12, vreg_input_x12, scale, preg_ori_tail_n4);
                Muls(vreg_input_x13, vreg_input_x13, scale, preg_ori_tail_n5);
                Muls(vreg_input_x14, vreg_input_x14, scale, preg_ori_tail_n6);
                Muls(vreg_input_x15, vreg_input_x15, scale, preg_ori_tail_n7);
                Muls(vreg_input_x16, vreg_input_x16, scale, preg_ori_tail_n8);
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (IsSameType<T2, bfloat16_t>::value) {
                    DataCopy(vreg_pse_bf16_src1, pseUb + i * pseStride);
                    DataCopy(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src3, pseUb + floatRepSize * 4 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src4, pseUb + floatRepSize * 6 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src5, pseUb + floatRepSize * 8 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src6, pseUb + floatRepSize * 10 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src7, pseUb + floatRepSize * 12 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src8, pseUb + floatRepSize * 14 + i * pseStride);

                    Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
                    Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
                    Interleave(vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse_bf16_src3, vreg_pse_bf16_src3);
                    Interleave(vreg_pse7_bf16, vreg_pse8_bf16, vreg_pse_bf16_src4, vreg_pse_bf16_src4);
                    Interleave(vreg_pse9_bf16, vreg_pse10_bf16, vreg_pse_bf16_src5, vreg_pse_bf16_src5);
                    Interleave(vreg_pse11_bf16, vreg_pse12_bf16, vreg_pse_bf16_src6, vreg_pse_bf16_src6);
                    Interleave(vreg_pse13_bf16, vreg_pse14_bf16, vreg_pse_bf16_src7, vreg_pse_bf16_src7);
                    Interleave(vreg_pse15_bf16, vreg_pse16_bf16, vreg_pse_bf16_src8, vreg_pse_bf16_src8);

                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse5, vreg_pse5_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse6, vreg_pse6_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse7, vreg_pse7_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse8, vreg_pse8_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse9, vreg_pse9_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse10, vreg_pse10_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse11, vreg_pse11_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse12, vreg_pse12_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse13, vreg_pse13_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse14, vreg_pse14_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse15, vreg_pse15_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse16, vreg_pse16_bf16, preg_all_b16);
                } else if constexpr (IsSameType<T2, half>::value) {
                    DataCopy(vreg_pse_f16_src1, pseUb + i * pseStride);
                    DataCopy(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    DataCopy(vreg_pse_f16_src3, pseUb + floatRepSize * 4 + i * pseStride);
                    DataCopy(vreg_pse_f16_src4, pseUb + floatRepSize * 6 + i * pseStride);
                    DataCopy(vreg_pse_f16_src5, pseUb + floatRepSize * 8 + i * pseStride);
                    DataCopy(vreg_pse_f16_src6, pseUb + floatRepSize * 10 + i * pseStride);
                    DataCopy(vreg_pse_f16_src7, pseUb + floatRepSize * 12 + i * pseStride);
                    DataCopy(vreg_pse_f16_src8, pseUb + floatRepSize * 14 + i * pseStride);

                    Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
                    Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
                    Interleave(vreg_pse5_f16, vreg_pse6_f16, vreg_pse_f16_src3, vreg_pse_f16_src3);
                    Interleave(vreg_pse7_f16, vreg_pse8_f16, vreg_pse_f16_src4, vreg_pse_f16_src4);
                    Interleave(vreg_pse9_f16, vreg_pse10_f16, vreg_pse_f16_src5, vreg_pse_f16_src5);
                    Interleave(vreg_pse11_f16, vreg_pse12_f16, vreg_pse_f16_src6, vreg_pse_f16_src6);
                    Interleave(vreg_pse13_f16, vreg_pse14_f16, vreg_pse_f16_src7, vreg_pse_f16_src7);
                    Interleave(vreg_pse15_f16, vreg_pse16_f16, vreg_pse_f16_src8, vreg_pse_f16_src8);

                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse5, vreg_pse5_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse6, vreg_pse6_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse7, vreg_pse7_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse8, vreg_pse8_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse9, vreg_pse9_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse10, vreg_pse10_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse11, vreg_pse11_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse12, vreg_pse12_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse13, vreg_pse13_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse14, vreg_pse14_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse15, vreg_pse15_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse16, vreg_pse16_f16, preg_all_b16);
                }
                Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
                Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
                Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_all);
                Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_all);
                Add(vreg_input_x5, vreg_input_x5, vreg_pse5, preg_all);
                Add(vreg_input_x6, vreg_input_x6, vreg_pse6, preg_all);
                Add(vreg_input_x7, vreg_input_x7, vreg_pse7, preg_all);
                Add(vreg_input_x8, vreg_input_x8, vreg_pse8, preg_all);
                Add(vreg_input_x9, vreg_input_x9, vreg_pse9, preg_ori_tail_n1);
                Add(vreg_input_x10, vreg_input_x10, vreg_pse10, preg_ori_tail_n2);
                Add(vreg_input_x11, vreg_input_x11, vreg_pse11, preg_ori_tail_n3);
                Add(vreg_input_x12, vreg_input_x12, vreg_pse12, preg_ori_tail_n4);
                Add(vreg_input_x13, vreg_input_x13, vreg_pse13, preg_ori_tail_n5);
                Add(vreg_input_x14, vreg_input_x14, vreg_pse14, preg_ori_tail_n6);
                Add(vreg_input_x15, vreg_input_x15, vreg_pse15, preg_ori_tail_n7);
                Add(vreg_input_x16, vreg_input_x16, vreg_pse16, preg_ori_tail_n8);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);
                Muls(vreg_input_x5, vreg_input_x5, scale, preg_all);
                Muls(vreg_input_x6, vreg_input_x6, scale, preg_all);
                Muls(vreg_input_x7, vreg_input_x7, scale, preg_all);
                Muls(vreg_input_x8, vreg_input_x8, scale, preg_all);

                Muls(vreg_input_x9, vreg_input_x9, scale, preg_ori_tail_n1);
                Muls(vreg_input_x10, vreg_input_x10, scale, preg_ori_tail_n2);
                Muls(vreg_input_x11, vreg_input_x11, scale, preg_ori_tail_n3);
                Muls(vreg_input_x12, vreg_input_x12, scale, preg_ori_tail_n4);
                Muls(vreg_input_x13, vreg_input_x13, scale, preg_ori_tail_n5);
                Muls(vreg_input_x14, vreg_input_x14, scale, preg_ori_tail_n6);
                Muls(vreg_input_x15, vreg_input_x15, scale, preg_ori_tail_n7);
                Muls(vreg_input_x16, vreg_input_x16, scale, preg_ori_tail_n8);
            }
            if constexpr (hasAtten == 1) {
                // atten mask
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare5, (__ubuf__ uint32_t *&)maskUb5, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare6, (__ubuf__ uint32_t *&)maskUb6, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare7, (__ubuf__ uint32_t *&)maskUb7, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare8, (__ubuf__ uint32_t *&)maskUb8, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare9, (__ubuf__ uint32_t *&)maskUb9, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare10, (__ubuf__ uint32_t *&)maskUb10, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare11, (__ubuf__ uint32_t *&)maskUb11, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare12, (__ubuf__ uint32_t *&)maskUb12, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare13, (__ubuf__ uint32_t *&)maskUb13, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare14, (__ubuf__ uint32_t *&)maskUb14, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare15, (__ubuf__ uint32_t *&)maskUb15, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare16, (__ubuf__ uint32_t *&)maskUb16, nPadding);

                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel5, vreg_min, vreg_input_x5, preg_compare5);
                Select(vreg_sel6, vreg_min, vreg_input_x6, preg_compare6);
                Select(vreg_sel7, vreg_min, vreg_input_x7, preg_compare7);
                Select(vreg_sel8, vreg_min, vreg_input_x8, preg_compare8);
                Select(vreg_sel9, vreg_min, vreg_input_x9, preg_compare9);
                Select(vreg_sel10, vreg_min, vreg_input_x10, preg_compare10);
                Select(vreg_sel11, vreg_min, vreg_input_x11, preg_compare11);
                Select(vreg_sel12, vreg_min, vreg_input_x12, preg_compare12);
                Select(vreg_sel13, vreg_min, vreg_input_x13, preg_compare13);
                Select(vreg_sel14, vreg_min, vreg_input_x14, preg_compare14);
                Select(vreg_sel15, vreg_min, vreg_input_x15, preg_compare15);
                Select(vreg_sel16, vreg_min, vreg_input_x16, preg_compare16);

                Select(vreg_sel9_new, vreg_sel9, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel10_new, vreg_sel10, vreg_min, preg_ori_tail_n2);
                Select(vreg_sel11_new, vreg_sel11, vreg_min, preg_ori_tail_n3);
                Select(vreg_sel12_new, vreg_sel12, vreg_min, preg_ori_tail_n4);
                Select(vreg_sel13_new, vreg_sel13, vreg_min, preg_ori_tail_n5);
                Select(vreg_sel14_new, vreg_sel14, vreg_min, preg_ori_tail_n6);
                Select(vreg_sel15_new, vreg_sel15, vreg_min, preg_ori_tail_n7);
                Select(vreg_sel16_new, vreg_sel16, vreg_min, preg_ori_tail_n8);

                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_sel3, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 +  i * s2BaseSize, vreg_sel4, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 4 +  i * s2BaseSize, vreg_sel5, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 5 +  i * s2BaseSize, vreg_sel6, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 6 +  i * s2BaseSize, vreg_sel7, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 7 +  i * s2BaseSize, vreg_sel8, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 8 +  i * s2BaseSize, vreg_sel9_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 9 +  i * s2BaseSize, vreg_sel10_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 10 +  i * s2BaseSize, vreg_sel11_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 11 +  i * s2BaseSize, vreg_sel12_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 12 +  i * s2BaseSize, vreg_sel13_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 13 +  i * s2BaseSize, vreg_sel14_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 14 +  i * s2BaseSize, vreg_sel15_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 15 +  i * s2BaseSize, vreg_sel16_new, preg_all);

                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3, vreg_sel4, preg_all);
                Max(vreg_max_tmp3, vreg_sel5, vreg_sel6, preg_all);
                Max(vreg_max_tmp4, vreg_sel7, vreg_sel8, preg_all);
                Max(vreg_max_tmp5, vreg_sel9_new, vreg_sel10_new, preg_all);
                Max(vreg_max_tmp6, vreg_sel11_new, vreg_sel12_new, preg_all);
                Max(vreg_max_tmp7, vreg_sel13_new, vreg_sel14_new, preg_all);
                Max(vreg_max_tmp8, vreg_sel15_new, vreg_sel16_new, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);
                Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp6, preg_all);
                Max(vreg_max_tmp7, vreg_max_tmp7, vreg_max_tmp8, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp3, preg_all);
                Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp7, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp5, preg_all);

                ReduceMax(vreg_input_max, vreg_max_tmp1, preg_all);
            } else {
                Select(vreg_input_x9_new, vreg_input_x9, vreg_min, preg_ori_tail_n1);
                Select(vreg_input_x10_new, vreg_input_x10, vreg_min, preg_ori_tail_n2);
                Select(vreg_input_x11_new, vreg_input_x11, vreg_min, preg_ori_tail_n3);
                Select(vreg_input_x12_new, vreg_input_x12, vreg_min, preg_ori_tail_n4);
                Select(vreg_input_x13_new, vreg_input_x13, vreg_min, preg_ori_tail_n5);
                Select(vreg_input_x14_new, vreg_input_x14, vreg_min, preg_ori_tail_n6);
                Select(vreg_input_x15_new, vreg_input_x15, vreg_min, preg_ori_tail_n7);
                Select(vreg_input_x16_new, vreg_input_x16, vreg_min, preg_ori_tail_n8);
                
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_input_x3, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_input_x4, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 4 + i * s2BaseSize, vreg_input_x5, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 5 + i * s2BaseSize, vreg_input_x6, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 6 + i * s2BaseSize, vreg_input_x7, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 7 + i * s2BaseSize, vreg_input_x8, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 8 + i * s2BaseSize, vreg_input_x9_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 9 + i * s2BaseSize, vreg_input_x10_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 10 + i * s2BaseSize, vreg_input_x11_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 11 + i * s2BaseSize, vreg_input_x12_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 12 + i * s2BaseSize, vreg_input_x13_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 13 + i * s2BaseSize, vreg_input_x14_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 14 + i * s2BaseSize, vreg_input_x15_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 15 + i * s2BaseSize, vreg_input_x16_new, preg_all);

                Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);    //1 2 取max
                Max(vreg_max_tmp2, vreg_input_x3, vreg_input_x4, preg_all);    //3 4 取max
                Max(vreg_max_tmp3, vreg_input_x5, vreg_input_x6, preg_all);
                Max(vreg_max_tmp4, vreg_input_x7, vreg_input_x8, preg_all);
                Max(vreg_max_tmp5, vreg_input_x9_new, vreg_input_x10_new, preg_all);
                Max(vreg_max_tmp6, vreg_input_x11_new, vreg_input_x12_new, preg_all);
                Max(vreg_max_tmp7, vreg_input_x13_new, vreg_input_x14_new, preg_all);
                Max(vreg_max_tmp8, vreg_input_x15_new, vreg_input_x16_new, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);
                Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp6, preg_all);
                Max(vreg_max_tmp7, vreg_max_tmp7, vreg_max_tmp8, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp3, preg_all);
                Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp7, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp5, preg_all);

                ReduceMax(vreg_input_max, vreg_max_tmp1, preg_all);
            }

            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, tmpMaxUb, 0, POST_UPDATE);
        DataCopy(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy(vreg_input_max, tmpMaxUb2);
        
        Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max, tmpMaxUb2 + i);

            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x1, vreg_input_x2, srcUb + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x3, vreg_input_x4, srcUb + floatRepSize * 2 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x5, vreg_input_x6, srcUb + floatRepSize * 4 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x7, vreg_input_x8, srcUb + floatRepSize * 6 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x9, vreg_input_x10, srcUb + floatRepSize * 8 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x11, vreg_input_x12, srcUb + floatRepSize * 10 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x13, vreg_input_x14, srcUb + floatRepSize * 12 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x15, vreg_input_x16, srcUb + floatRepSize * 14 + i * s2BaseSize);

            FusedExpSub(vreg_exp_even1, vreg_input_x1, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even2, vreg_input_x3, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even3, vreg_input_x5, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd3, vreg_input_x6, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even4, vreg_input_x7, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd4, vreg_input_x8, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even5, vreg_input_x9, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd5, vreg_input_x10, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even6, vreg_input_x11, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd6, vreg_input_x12, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even7, vreg_input_x13, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd7, vreg_input_x14, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even8, vreg_input_x15, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd8, vreg_input_x16, vreg_max, preg_all);

            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_even3, vreg_exp_odd3, preg_all);
            Add(vreg_exp_sum4, vreg_exp_even4, vreg_exp_odd4, preg_all);
            Add(vreg_exp_sum5, vreg_exp_even5, vreg_exp_odd5, preg_all);
            Add(vreg_exp_sum6, vreg_exp_even6, vreg_exp_odd6, preg_all);
            Add(vreg_exp_sum7, vreg_exp_even7, vreg_exp_odd7, preg_all);
            Add(vreg_exp_sum8, vreg_exp_even8, vreg_exp_odd8, preg_all);

            Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum3, vreg_exp_sum4, preg_all);
            Add(vreg_exp_sum5, vreg_exp_sum5, vreg_exp_sum6, preg_all);
            Add(vreg_exp_sum7, vreg_exp_sum7, vreg_exp_sum8, preg_all);

            Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum3, preg_all);
            Add(vreg_exp_sum5, vreg_exp_sum5, vreg_exp_sum7, preg_all);

            Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum5, preg_all);

            ReduceSum(vreg_exp_sum1, vreg_exp_sum1, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum1, ureg_exp_sum, 1);

            if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even3_bf16, vreg_exp_even3, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd3_bf16, vreg_exp_odd3, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even4_bf16, vreg_exp_even4, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd4_bf16, vreg_exp_odd4, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even5_bf16, vreg_exp_even5, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd5_bf16, vreg_exp_odd5, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even6_bf16, vreg_exp_even6, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd6_bf16, vreg_exp_odd6, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even7_bf16, vreg_exp_even7, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd7_bf16, vreg_exp_odd7, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even8_bf16, vreg_exp_even8, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd8_bf16, vreg_exp_odd8, preg_all);

                Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp3_bf16, (RegTensor<uint16_t>&)vreg_exp_even3_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd3_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp4_bf16, (RegTensor<uint16_t>&)vreg_exp_even4_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd4_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp5_bf16, (RegTensor<uint16_t>&)vreg_exp_even5_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd5_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp6_bf16, (RegTensor<uint16_t>&)vreg_exp_even6_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd6_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp7_bf16, (RegTensor<uint16_t>&)vreg_exp_even7_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd7_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp8_bf16, (RegTensor<uint16_t>&)vreg_exp_even8_bf16,
                    (RegTensor<uint16_t>&)vreg_exp_odd8_bf16, preg_all_b16);

                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb3), vreg_exp3_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb4), vreg_exp4_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb5), vreg_exp5_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb6), vreg_exp6_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb7), vreg_exp7_bf16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb8), vreg_exp8_bf16, blockStride, repeatStride, preg_all_b16);

            } else if constexpr (IsSameType<T2, half>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even3_f16, vreg_exp_even3, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd3_f16, vreg_exp_odd3, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even4_f16, vreg_exp_even4, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd4_f16, vreg_exp_odd4, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even5_f16, vreg_exp_even5, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd5_f16, vreg_exp_odd5, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even6_f16, vreg_exp_even6, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd6_f16, vreg_exp_odd6, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even7_f16, vreg_exp_even7, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd7_f16, vreg_exp_odd7, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even8_f16, vreg_exp_even8, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd8_f16, vreg_exp_odd8, preg_all);

                Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16, (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16, (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp3_f16, (RegTensor<uint16_t>&)vreg_exp_even3_f16, (RegTensor<uint16_t>&)vreg_exp_odd3_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp4_f16, (RegTensor<uint16_t>&)vreg_exp_even4_f16, (RegTensor<uint16_t>&)vreg_exp_odd4_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp5_f16, (RegTensor<uint16_t>&)vreg_exp_even5_f16, (RegTensor<uint16_t>&)vreg_exp_odd5_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp6_f16, (RegTensor<uint16_t>&)vreg_exp_even6_f16, (RegTensor<uint16_t>&)vreg_exp_odd6_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp7_f16, (RegTensor<uint16_t>&)vreg_exp_even7_f16, (RegTensor<uint16_t>&)vreg_exp_odd7_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp8_f16, (RegTensor<uint16_t>&)vreg_exp_even8_f16, (RegTensor<uint16_t>&)vreg_exp_odd8_f16, preg_all_b16);

                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb3), vreg_exp3_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb4), vreg_exp4_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb5), vreg_exp5_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb6), vreg_exp6_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb7), vreg_exp7_f16, blockStride, repeatStride, preg_all_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb8), vreg_exp8_f16, blockStride, repeatStride, preg_all_b16);
            }
        }
        vstas(ureg_exp_sum, tmpExpSumUb, 0, POST_UPDATE);
    }
}

// update, 512 < originN <= 1024
template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 16, uint32_t s2BaseSize = 512,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1UpdateGeneralImpl512(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<OUTPUT_T>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const T minValue, float keepProb)
{
    constexpr uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    constexpr uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    const uint32_t oriTailN1 = originN - floatRepSize * 4 < floatRepSize ? originN - floatRepSize * 4 : floatRepSize;
    const uint32_t oriTailN2 = static_cast<int32_t>(originN - floatRepSize * 5) <= 0 ? 0 : originN - floatRepSize * 5;
    const uint32_t oriTailN3 = static_cast<int32_t>(originN - floatRepSize * 6) <= 0 ? 0 : originN - floatRepSize * 6;
    const uint32_t oriTailN4 = static_cast<int32_t>(originN - floatRepSize * 7) <= 0 ? 0 : originN - floatRepSize * 7;
    const uint32_t tailN1 = s2BaseSize - floatRepSize * 4;
    const uint32_t tailN2 = s2BaseSize - floatRepSize * 5;
    const uint32_t tailN3 = s2BaseSize - floatRepSize * 6;
    const uint32_t tailN4 = s2BaseSize - floatRepSize * 7;
    uint32_t pltOriTailN1 = oriTailN1;
    uint32_t pltOriTailN2 = oriTailN2;
    uint32_t pltOriTailN3 = oriTailN3;
    uint32_t pltOriTailN4 = oriTailN4;
    uint32_t pltTailN1 = tailN1;
    uint32_t pltTailN2 = tailN2;
    uint32_t pltTailN3 = tailN3;
    uint32_t pltTailN4 = tailN4;
    float divValue = 1.0f / keepProb;
    uint32_t pltN = s2BaseSize;

    __ubuf__ T2 * expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb3 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 2* ((s1BaseSize >> 1) + 1) * (128);
    __ubuf__ T2 * expUb4 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + 3* ((s1BaseSize >> 1) + 1) * (128);

    __ubuf__ OUTPUT_T * pseUb = (__ubuf__ OUTPUT_T*)pseTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;

    uint64_t maskUb1 = maskTensor.GetPhyAddr();
    uint64_t maskUb2 = maskTensor.GetPhyAddr() + floatRepSize;
    uint64_t maskUb3 = maskTensor.GetPhyAddr() + floatRepSize * 2;
    uint64_t maskUb4 = maskTensor.GetPhyAddr() + floatRepSize * 3;
    uint64_t maskUb5 = maskTensor.GetPhyAddr() + floatRepSize * 4;
    uint64_t maskUb6 = maskTensor.GetPhyAddr() + floatRepSize * 5;
    uint64_t maskUb7 = maskTensor.GetPhyAddr() + floatRepSize * 6;
    uint64_t maskUb8 = maskTensor.GetPhyAddr() + floatRepSize * 7;

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel5;
        RegTensor<float> vreg_sel6;
        RegTensor<float> vreg_sel7;
        RegTensor<float> vreg_sel8;
        RegTensor<float> vreg_sel5_new;
        RegTensor<float> vreg_sel6_new;
        RegTensor<float> vreg_sel7_new;
        RegTensor<float> vreg_sel8_new;


        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x5;
        RegTensor<float> vreg_input_x6;
        RegTensor<float> vreg_input_x7;
        RegTensor<float> vreg_input_x8;
        RegTensor<float> vreg_input_x5_new;
        RegTensor<float> vreg_input_x6_new;
        RegTensor<float> vreg_input_x7_new;
        RegTensor<float> vreg_input_x8_new;


        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_max_tmp4;

        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_max_new;

        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_sum4;

        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;

        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_exp_even3;
        RegTensor<float> vreg_exp_odd3;
        RegTensor<float> vreg_exp_even4;
        RegTensor<float> vreg_exp_odd4; 

        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_pse5;
        RegTensor<float> vreg_pse6;
        RegTensor<float> vreg_pse7;
        RegTensor<float> vreg_pse8;

        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp_even3_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd3_bf16;
        RegTensor<bfloat16_t> vreg_exp_even4_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd4_bf16; 

        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_exp3_bf16;
        RegTensor<bfloat16_t> vreg_exp4_bf16;

        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse_bf16_src3;
        RegTensor<bfloat16_t> vreg_pse_bf16_src4;

        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        RegTensor<bfloat16_t> vreg_pse5_bf16;
        RegTensor<bfloat16_t> vreg_pse6_bf16;
        RegTensor<bfloat16_t> vreg_pse7_bf16;
        RegTensor<bfloat16_t> vreg_pse8_bf16;

        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp_even3_f16;
        RegTensor<half> vreg_exp_odd3_f16;
        RegTensor<half> vreg_exp_even4_f16;
        RegTensor<half> vreg_exp_odd4_f16;

        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_exp3_f16;
        RegTensor<half> vreg_exp4_f16;

        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse_f16_src3;
        RegTensor<half> vreg_pse_f16_src4;

        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
        RegTensor<half> vreg_pse5_f16;
        RegTensor<half> vreg_pse6_f16;
        RegTensor<half> vreg_pse7_f16;
        RegTensor<half> vreg_pse8_f16;

        UnalignReg ureg_max;
        UnalignReg ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);

        MaskReg preg_tail_n1 = UpdateMask<T>(pltTailN1);
        MaskReg preg_ori_tail_n1 = UpdateMask<T>(pltOriTailN1);
        MaskReg preg_tail_n2 = UpdateMask<T>(pltTailN2);
        MaskReg preg_ori_tail_n2 = UpdateMask<T>(pltOriTailN2);
        MaskReg preg_tail_n3 = UpdateMask<T>(pltTailN3);
        MaskReg preg_ori_tail_n3 = UpdateMask<T>(pltOriTailN3);
        MaskReg preg_tail_n4 = UpdateMask<T>(pltTailN4);
        MaskReg preg_ori_tail_n4 = UpdateMask<T>(pltOriTailN4);


        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;
        MaskReg preg_compare5;
        MaskReg preg_compare6;
        MaskReg preg_compare7;
        MaskReg preg_compare8;

        Duplicate(vreg_min, minValue);
        for (uint16_t i = 0; i < m; ++i) {
            DataCopy(vreg_input_x1, srcUb + i * s2BaseSize);
            DataCopy(vreg_input_x2, srcUb + floatRepSize + i * s2BaseSize);
            DataCopy(vreg_input_x3, srcUb + floatRepSize * 2 + i * s2BaseSize);
            DataCopy(vreg_input_x4, srcUb + floatRepSize * 3 + i * s2BaseSize);
            DataCopy(vreg_input_x5, srcUb + floatRepSize * 4 + i * s2BaseSize);
            DataCopy(vreg_input_x6, srcUb + floatRepSize * 5 + i * s2BaseSize);
            DataCopy(vreg_input_x7, srcUb + floatRepSize * 6 + i * s2BaseSize);
            DataCopy(vreg_input_x8, srcUb + floatRepSize * 7 + i * s2BaseSize);

            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);

                Muls(vreg_input_x5, vreg_input_x5, scale, preg_ori_tail_n1);
                Muls(vreg_input_x6, vreg_input_x6, scale, preg_ori_tail_n2);
                Muls(vreg_input_x7, vreg_input_x7, scale, preg_ori_tail_n3);
                Muls(vreg_input_x8, vreg_input_x8, scale, preg_ori_tail_n4);
            }
            if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
                if constexpr (IsSameType<T2, bfloat16_t>::value) {
                    DataCopy(vreg_pse_bf16_src1, pseUb + i * pseStride);
                    DataCopy(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src3, pseUb + floatRepSize * 4 + i * pseStride);
                    DataCopy(vreg_pse_bf16_src4, pseUb + floatRepSize * 6 + i * pseStride);

                    Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
                    Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
                    Interleave(vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse_bf16_src3, vreg_pse_bf16_src3);
                    Interleave(vreg_pse7_bf16, vreg_pse8_bf16, vreg_pse_bf16_src4, vreg_pse_bf16_src4);

                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse5, vreg_pse5_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse6, vreg_pse6_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse7, vreg_pse7_bf16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse8, vreg_pse8_bf16, preg_all_b16);
                } else if constexpr (IsSameType<T2, half>::value) {
                    DataCopy(vreg_pse_f16_src1, pseUb + i * pseStride);
                    DataCopy(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
                    DataCopy(vreg_pse_f16_src3, pseUb + floatRepSize * 4 + i * pseStride);
                    DataCopy(vreg_pse_f16_src4, pseUb + floatRepSize * 6 + i * pseStride);

                    Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
                    Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
                    Interleave(vreg_pse5_f16, vreg_pse6_f16, vreg_pse_f16_src3, vreg_pse_f16_src3);
                    Interleave(vreg_pse7_f16, vreg_pse8_f16, vreg_pse_f16_src4, vreg_pse_f16_src4);

                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse5, vreg_pse5_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse6, vreg_pse6_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse7, vreg_pse7_f16, preg_all_b16);
                    Cast<T, OUTPUT_T, castTraitZero>(vreg_pse8, vreg_pse8_f16, preg_all_b16);
                }
                Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
                Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
                Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_all);
                Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_all);
                Add(vreg_input_x5, vreg_input_x5, vreg_pse5, preg_ori_tail_n1);
                Add(vreg_input_x6, vreg_input_x6, vreg_pse6, preg_ori_tail_n2);
                Add(vreg_input_x7, vreg_input_x7, vreg_pse7, preg_ori_tail_n3);
                Add(vreg_input_x8, vreg_input_x8, vreg_pse8, preg_ori_tail_n4);
            }
            if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);

                Muls(vreg_input_x5, vreg_input_x5, scale, preg_ori_tail_n1);
                Muls(vreg_input_x6, vreg_input_x6, scale, preg_ori_tail_n2);
                Muls(vreg_input_x7, vreg_input_x7, scale, preg_ori_tail_n3);
                Muls(vreg_input_x8, vreg_input_x8, scale, preg_ori_tail_n4);
            }
            if constexpr (hasAtten == 1) {
                // atten mask
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare5, (__ubuf__ uint32_t *&)maskUb5, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare6, (__ubuf__ uint32_t *&)maskUb6, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare7, (__ubuf__ uint32_t *&)maskUb7, nPadding);
                DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare8, (__ubuf__ uint32_t *&)maskUb8, nPadding);
                
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel5, vreg_min, vreg_input_x5, preg_compare5);
                Select(vreg_sel6, vreg_min, vreg_input_x6, preg_compare6);
                Select(vreg_sel7, vreg_min, vreg_input_x7, preg_compare7);
                Select(vreg_sel8, vreg_min, vreg_input_x8, preg_compare8);

                Select(vreg_sel5_new, vreg_sel5, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel6_new, vreg_sel6, vreg_min, preg_ori_tail_n2);
                Select(vreg_sel7_new, vreg_sel7, vreg_min, preg_ori_tail_n3);
                Select(vreg_sel8_new, vreg_sel8, vreg_min, preg_ori_tail_n4);
                
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_sel3, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 +  i * s2BaseSize, vreg_sel4, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 4 +  i * s2BaseSize, vreg_sel5_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 5 +  i * s2BaseSize, vreg_sel6_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 6 +  i * s2BaseSize, vreg_sel7_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 7 +  i * s2BaseSize, vreg_sel8_new, preg_all);

                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3, vreg_sel4, preg_all);
                Max(vreg_max_tmp3, vreg_sel5_new, vreg_sel6_new, preg_all);
                Max(vreg_max_tmp4, vreg_sel7_new, vreg_sel8_new, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);

                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp3, preg_all);

                ReduceMax(vreg_input_max, vreg_max_tmp3, preg_all);
            } else {
                Select(vreg_input_x5_new, vreg_input_x5, vreg_min, preg_ori_tail_n1);
                Select(vreg_input_x6_new, vreg_input_x6, vreg_min, preg_ori_tail_n2);
                Select(vreg_input_x7_new, vreg_input_x7, vreg_min, preg_ori_tail_n3);
                Select(vreg_input_x8_new, vreg_input_x8, vreg_min, preg_ori_tail_n4);

                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x1, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x2, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_input_x3, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_input_x4, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 4 + i * s2BaseSize, vreg_input_x5_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 5 + i * s2BaseSize, vreg_input_x6_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 6 + i * s2BaseSize, vreg_input_x7_new, preg_all);
                DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T *&)srcUb + floatRepSize * 7 + i * s2BaseSize, vreg_input_x8_new, preg_all);

                Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                Max(vreg_max_tmp2, vreg_input_x3, vreg_input_x4, preg_all);
                Max(vreg_max_tmp3, vreg_input_x5_new, vreg_input_x6_new, preg_all);
                Max(vreg_max_tmp4, vreg_input_x7_new, vreg_input_x8_new, preg_all);

                Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);

                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp3, preg_all);
                ReduceMax(vreg_input_max, vreg_max_tmp3, preg_all);
            }

            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1);
        }
        vstas(ureg_max, tmpMaxUb, 0, POST_UPDATE);
        DataCopy(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy(vreg_input_max, tmpMaxUb2);
        
        Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg_max, tmpMaxUb2 + i);

            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x1, vreg_input_x2, srcUb + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x3, vreg_input_x4, srcUb + floatRepSize * 2 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x5, vreg_input_x6, srcUb + floatRepSize * 4 + i * s2BaseSize);
            DataCopy<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x7, vreg_input_x8, srcUb + floatRepSize * 6 + i * s2BaseSize);

            FusedExpSub(vreg_exp_even1, vreg_input_x1, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even2, vreg_input_x3, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even3, vreg_input_x5, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd3, vreg_input_x6, vreg_max, preg_all);
            FusedExpSub(vreg_exp_even4, vreg_input_x7, vreg_max, preg_all);
            FusedExpSub(vreg_exp_odd4, vreg_input_x8, vreg_max, preg_all);

            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_even3, vreg_exp_odd3, preg_all);
            Add(vreg_exp_sum4, vreg_exp_even4, vreg_exp_odd4, preg_all);

            Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum3, vreg_exp_sum4, preg_all);

            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum3, preg_all);

            ReduceSum(vreg_exp_sum3, vreg_exp_sum3, preg_all);
            DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum3, ureg_exp_sum, 1);

            if constexpr (IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even3_bf16, vreg_exp_even3, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd3_bf16, vreg_exp_odd3, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even4_bf16, vreg_exp_even4, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd4_bf16, vreg_exp_odd4, preg_all);

                Or((RegTensor<uint16_t>&)vreg_exp1_bf16, (RegTensor<uint16_t>&)vreg_exp_even1_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd1_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_bf16, (RegTensor<uint16_t>&)vreg_exp_even2_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd2_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp3_bf16, (RegTensor<uint16_t>&)vreg_exp_even3_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd3_bf16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp4_bf16, (RegTensor<uint16_t>&)vreg_exp_even4_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd4_bf16, preg_all_b16);

                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_n_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_n_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb3), vreg_exp3_bf16, blockStride, repeatStride, preg_n_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb4), vreg_exp4_bf16, blockStride, repeatStride, preg_n_b16);

            } else if constexpr (IsSameType<T2, half>::value) {
                Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even3_f16, vreg_exp_even3, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd3_f16, vreg_exp_odd3, preg_all);
                Cast<T2, T, castTraitZero>(vreg_exp_even4_f16, vreg_exp_even4, preg_all);
                Cast<T2, T, castTraitOne>(vreg_exp_odd4_f16, vreg_exp_odd4, preg_all);

                Or((RegTensor<uint16_t>&)vreg_exp1_f16, (RegTensor<uint16_t>&)vreg_exp_even1_f16, (RegTensor<uint16_t>&)vreg_exp_odd1_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp2_f16, (RegTensor<uint16_t>&)vreg_exp_even2_f16, (RegTensor<uint16_t>&)vreg_exp_odd2_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp3_f16, (RegTensor<uint16_t>&)vreg_exp_even3_f16, (RegTensor<uint16_t>&)vreg_exp_odd3_f16, preg_all_b16);
                Or((RegTensor<uint16_t>&)vreg_exp4_f16, (RegTensor<uint16_t>&)vreg_exp_even4_f16, (RegTensor<uint16_t>&)vreg_exp_odd4_f16, preg_all_b16);

                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_n_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_n_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb3), vreg_exp3_f16, blockStride, repeatStride, preg_n_b16);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb4), vreg_exp4_f16, blockStride, repeatStride, preg_n_b16);
            }
        }
        vstas(ureg_exp_sum, tmpExpSumUb, 0, POST_UPDATE);
    }
}

template <typename T>
__aicore__ inline void UpdateExpSumAndExpMax(const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m)
{
    UpdateExpSumAndExpMaxImpl<T>(expSumTensor, maxTensor, expMaxTensor,
                                 inExpSumTensor, inMaxTensor, sharedTmpBuffer, m);
}

template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = GT_64_AND_LTE_128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1Update(
    const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb)
{
    if constexpr (oriNRange == GT_128_AND_LTE_256) {
    ProcessVec1UpdateGeneralImpl256RegbaseV2<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
        dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
        inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN,
        pseStride, slopes, posShift, scale, minValue, keepProb);
    } else if constexpr (oriNRange == EQ_128) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1UpdateImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop, isMlaSgd>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN,
            pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb);
    } else if constexpr (oriNRange == GT_0_AND_LTE_64) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1UpdateImpl64<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop, isMlaSgd>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN,
            pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb);
    } else if constexpr (oriNRange == GT_256_AND_LTE_512) {
        ProcessVec1UpdateGeneralImpl512<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN,
            pseStride, slopes, posShift, scale, minValue, keepProb);
    } else if constexpr (oriNRange == GT_512_AND_LTE_1024) {
        ProcessVec1UpdateGeneralImpl1024<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN,
            pseStride, slopes, posShift, scale, minValue, keepProb);
    } else { // GT_64_AND_LTE_128
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1UpdateGeneralImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop, isMlaSgd>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN,
            pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb);
    }
}

/*
 * @ingroup ProcessVec1Vf
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [in] srcTensor, input LocalTensor
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] inExpSumTensor, in sum(exp(x-max)) of last softmax result
 * @param [in] inMaxTensor, in max value of last softmax result
 * @param [in] maskTensor, atten mask LocalTensor, each line padding to 32, padding value is 1
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] s2BaseSize, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] originN, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isUpdate, enable flash mode
 * @param [in] oriNRange, originN range
 * @param [in] hasAtten, indicates whether there is atten_mask
 */
template <typename T, typename T2, typename pseShiftType, bool isUpdate = false, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
          OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0,
          PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1Vf(const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<pseShiftType>& pseTensor, const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb)
{
    static_assert(IsSameType<T, float>::value, "VF mul_sel_softmaxflashv2_cast_nz, T must be float");
    static_assert(IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value || IsSameType<T2, float>::value ||
                  IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value || IsSameType<T2, hifloat8_t>::value,
                  "VF mul_sel_softmaxflashv2_cast_nz, T2 must be half, bfloat16 or float or fp8");
    static_assert(oriNRange < N_INVALID, "VF mul_sel_softmaxflashv2_cast_nz, oriNRange is invalid");

    if constexpr (!isUpdate) {
        ProcessVec1NoUpdate<T, T2, pseShiftType, s1BaseSize, s2BaseSize, oriNRange, hasAtten, pseMode, hasDrop, isMlaSgd>(
            dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer,
            m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb);
    } else {
        ProcessVec1Update<T, T2, pseShiftType, s1BaseSize, s2BaseSize, oriNRange, hasAtten, pseMode, hasDrop, isMlaSgd>(
            dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer,
            m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb);
    }
}

template <typename T>
__aicore__ inline void SoftmaxSumUpdate(const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor,
    const uint32_t m, const T minValue, const T maxValue)
{
    __ubuf__ T * sumUb = (__ubuf__ T*)sumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> vreg_max_value;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_sum;
        RegTensor<float> vreg_sum_new;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_compare;

        Duplicate(vreg_max_value, maxValue);
        // 注意：当m大于64的时候需要开启循环
        DataCopy(vreg_max, maxUb);
        DataCopy(vreg_sum, sumUb);
        CompareScalar<T, CMPMODE::EQ>(preg_compare, vreg_max, minValue, preg_all);
        Select(vreg_sum_new, vreg_max_value, vreg_sum, preg_compare);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)sumUb, vreg_sum_new, preg_all);
    }
}
} // namespace

#endif // MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_H
