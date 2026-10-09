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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_dn_arch92.h
 * \brief
 */

#ifndef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_ARCH92_H_
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_ARCH92_H_

#include "kernel_tensor.h"
#include "../../arch35/vf/vf_basic_block_utils.h"

namespace FaVectorApi {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

template <typename T, typename T2, uint16_t s2BaseSize = 128, bool isUpdate = false, bool hasAtten = 0>
__simd_vf__ inline void ProcessVec1Dn128V2VF(__ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB,
    __ubuf__ float *exp_max, __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max,
    __ubuf__ uint8_t *maskUb, const uint32_t m, const uint32_t n, const T scale, const T minValue,
    const uint32_t blockStride, const uint32_t repeatStride)
{
    __ubuf__ T *src_ub_0 = input_x_local_UB;
    __ubuf__ T *src_ub_1 = src_ub_0 + 64;
    __ubuf__ T *src_ub_2 = src_ub_0 + m * 1;
    __ubuf__ T *src_ub_3 = src_ub_0 + 64 + m * 1;
    __ubuf__ T *src_ub_4 = src_ub_0 + m * 2;
    __ubuf__ T *src_ub_5 = src_ub_0 + 64 + m * 2;
    __ubuf__ T *src_ub_6 = src_ub_0 + m * 3;
    __ubuf__ T *src_ub_7 = src_ub_0 + 64 + m * 3;

    __ubuf__ uint8_t *mask_ub_0 = maskUb;
    __ubuf__ uint8_t *mask_ub_1 = maskUb + 64;
    __ubuf__ uint8_t *mask_ub_2 = maskUb + 64 * 2;
    __ubuf__ uint8_t *mask_ub_3 = maskUb + 64 * 3;

    MaskReg preg_input = CreateMask<T2, MaskPattern::ALL>();
    MaskReg preg_out = CreateMask<T2, MaskPattern::ALL>();

    RegTensor<T> vreg_src_0, vreg_src_1, vreg_src_2, vreg_src_3;
    RegTensor<T> vreg_src_4, vreg_src_5, vreg_src_6, vreg_src_7;
    RegTensor<T> vreg_max_0, vreg_max_1;
    RegTensor<T> vreg_max_pack_0, vreg_max_pack_1;
    RegTensor<T> vreg_sum_0, vreg_sum_1;
    RegTensor<T> vreg_sum_pack_0, vreg_sum_pack_1;
    RegTensor<T> vreg_exp_0, vreg_exp_1;
    RegTensor<T2> vreg_exp_cast_0, vreg_exp_cast_1;
    RegTensor<T> vreg_max_last_0, vreg_max_last_1;
    RegTensor<T> vreg_sum_last_0, vreg_sum_last_1;
    RegTensor<T> vreg_exp_max_0, vreg_exp_max_1;
    RegTensor<T> vreg_min;

    MaskReg preg_mask_0, preg_mask_1, preg_mask_2, preg_mask_3;

    Duplicate(vreg_max_0, minValue);
    Duplicate(vreg_max_1, minValue);
    Duplicate(vreg_min, minValue);

    for (uint16_t i = n; i < s2BaseSize; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
            input_x_local_UB + i * m, vreg_min, preg_input);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
            input_x_local_UB + i * m + 64, vreg_min, preg_input);
    }
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        for (uint16_t iter_m = 0; iter_m < uint16_t(s2BaseSize / 2); ++iter_m) {
            LoadAlign(vreg_src_0, src_ub_0 + iter_m * m * 2);
            LoadAlign(vreg_src_1, src_ub_1 + iter_m * m * 2);
            LoadAlign(vreg_src_2, src_ub_2 + iter_m * m * 2);
            LoadAlign(vreg_src_3, src_ub_3 + iter_m * m * 2);
            
            Muls(vreg_src_0, vreg_src_0, scale, preg_input);
            Muls(vreg_src_1, vreg_src_1, scale, preg_input);
            Muls(vreg_src_2, vreg_src_2, scale, preg_input);
            Muls(vreg_src_3, vreg_src_3, scale, preg_input);

            LoadAlign<uint8_t, Reg::MaskDist::DIST_DS>(
                preg_mask_0, mask_ub_0 + iter_m * m * 2);
            LoadAlign<uint8_t, Reg::MaskDist::DIST_DS>(
                preg_mask_1, mask_ub_1 + iter_m * m * 2);
            LoadAlign<uint8_t, Reg::MaskDist::DIST_DS>(
                preg_mask_2, mask_ub_2 + iter_m * m * 2);
            LoadAlign<uint8_t, Reg::MaskDist::DIST_DS>(
                preg_mask_3, mask_ub_3 + iter_m * m * 2);
            
            Select(vreg_src_0, vreg_src_0, vreg_min, preg_mask_0);
            Select(vreg_src_1, vreg_src_1, vreg_min, preg_mask_1);
            Select(vreg_src_2, vreg_src_2, vreg_min, preg_mask_2);
            Select(vreg_src_3, vreg_src_3, vreg_min, preg_mask_3);

            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
                src_ub_0 + iter_m * m * 2, vreg_src_0, preg_input);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
                src_ub_1 + iter_m * m * 2, vreg_src_1, preg_input);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
                src_ub_2 + iter_m * m * 2, vreg_src_2, preg_input);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
                src_ub_3 + iter_m * m * 2, vreg_src_3, preg_input);

            Max(vreg_src_0, vreg_src_0, vreg_src_2, preg_input);
            Max(vreg_src_1, vreg_src_1, vreg_src_3, preg_input);
            Max(vreg_max_0, vreg_max_0, vreg_src_0, preg_input);
            Max(vreg_max_1, vreg_max_1, vreg_src_1, preg_input);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(s2BaseSize / 4); ++iter_m) {
            LoadAlign(vreg_src_0, src_ub_0 + iter_m * m * 4);
            LoadAlign(vreg_src_1, src_ub_1 + iter_m * m * 4);
            LoadAlign(vreg_src_2, src_ub_2 + iter_m * m * 4);
            LoadAlign(vreg_src_3, src_ub_3 + iter_m * m * 4);
            LoadAlign(vreg_src_4, src_ub_4 + iter_m * m * 4);
            LoadAlign(vreg_src_5, src_ub_5 + iter_m * m * 4);
            LoadAlign(vreg_src_6, src_ub_6 + iter_m * m * 4);
            LoadAlign(vreg_src_7, src_ub_7 + iter_m * m * 4);

            Max(vreg_src_0, vreg_src_0, vreg_src_2, preg_input);
            Max(vreg_src_4, vreg_src_4, vreg_src_6, preg_input);
            Max(vreg_src_1, vreg_src_1, vreg_src_3, preg_input);
            Max(vreg_src_5, vreg_src_5, vreg_src_7, preg_input);
            Max(vreg_src_0, vreg_src_0, vreg_src_4, preg_input);
            Max(vreg_src_1, vreg_src_1, vreg_src_5, preg_input);
            Max(vreg_max_0, vreg_max_0, vreg_src_0, preg_input);
            Max(vreg_max_1, vreg_max_1, vreg_src_1, preg_input);
        }            
    }
    if constexpr (!hasAtten) {
        Muls(vreg_max_0, vreg_max_0, scale, preg_input);
        Muls(vreg_max_1, vreg_max_1, scale, preg_input);
    }

    if constexpr (isUpdate) {
        LoadAlign(vreg_max_last_0, new_global_max);
        LoadAlign(vreg_max_last_1, new_global_max + 64);
        Max(vreg_max_0, vreg_max_0, vreg_max_last_0, preg_input);
        Max(vreg_max_1, vreg_max_1, vreg_max_last_1, preg_input);
    }

    DeInterleave(vreg_max_pack_0, vreg_max_pack_1, vreg_max_0, vreg_max_1);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(new_global_max, vreg_max_0, preg_input);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(new_global_max + 64, vreg_max_1, preg_input);

    if constexpr (isUpdate) {
        FusedExpSub(vreg_exp_max_0, vreg_max_last_0, vreg_max_0, preg_input);
        FusedExpSub(vreg_exp_max_1, vreg_max_last_1, vreg_max_1, preg_input);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(exp_max, vreg_exp_max_0, preg_input);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(exp_max + 64, vreg_exp_max_1, preg_input);
    }

    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_sum_0, 0, preg_input);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_sum_1, 0, preg_input);
    if constexpr (hasAtten) {
        mem_bar(VST_VLD);
    }
    for (uint16_t i = 0; i < uint16_t(s2BaseSize); ++i) {
        LoadAlign<T, Reg::LoadDist::DIST_DINTLV_B32>(
            vreg_src_0, vreg_src_1, input_x_local_UB + i * m);

        if constexpr (!hasAtten) {
            Muls(vreg_src_0, vreg_src_0, scale, preg_input);
            Muls(vreg_src_1, vreg_src_1, scale, preg_input);
        }

        FusedExpSub(vreg_exp_0, vreg_src_0, vreg_max_pack_0, preg_input);
        FusedExpSub(vreg_exp_1, vreg_src_1, vreg_max_pack_1, preg_input);
        Add(vreg_sum_0, vreg_exp_0, vreg_sum_0, preg_input);
        Add(vreg_sum_1, vreg_exp_1, vreg_sum_1, preg_input);

        Cast<T2, T, castTraitOne>(vreg_exp_cast_0, vreg_exp_1, preg_input);
        Cast<T2, T, castTraitZeroMerge>(vreg_exp_cast_0, vreg_exp_0, preg_input);

        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                x_exp, vreg_exp_cast_0, blockStride, repeatStride, preg_out);
    }
    Interleave(vreg_sum_pack_0, vreg_sum_pack_1, vreg_sum_0, vreg_sum_1);
    if constexpr (isUpdate) {
        LoadAlign(vreg_sum_last_0, new_global_sum);
        LoadAlign(vreg_sum_last_1, new_global_sum + 64);
        Mul(vreg_sum_last_0, vreg_exp_max_0, vreg_sum_last_0, preg_input);
        Mul(vreg_sum_last_1, vreg_exp_max_1, vreg_sum_last_1, preg_input);
        Add(vreg_sum_pack_0, vreg_sum_last_0, vreg_sum_pack_0, preg_input);
        Add(vreg_sum_pack_1, vreg_sum_last_1, vreg_sum_pack_1, preg_input);
    }
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(new_global_sum, vreg_sum_pack_0, preg_input);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(new_global_sum + 64, vreg_sum_pack_1, preg_input);
}

template <typename T, typename T2, uint16_t mBaseSize = 128, uint16_t s2BaseSize = 128, bool isUpdate = false, bool hasAtten = 0>
__aicore__ inline void ProcessVec1VfDn(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                       const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t>& maskTensor, 
                                       const uint32_t m, const uint32_t n, const T scale, const T minValue,
                                       const float sinkValue = 0.0f)
{
    constexpr uint32_t repeatStride = 1;
    constexpr uint32_t blockStride = s2BaseSize | 0x1;

    __ubuf__ T2* x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ T* input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ T* exp_max = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T* new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ T* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint8_t* maskUb = (__ubuf__ uint8_t*)maskTensor.GetPhyAddr();

    ProcessVec1Dn128V2VF<T, T2, s2BaseSize, isUpdate, hasAtten>(x_exp, input_x_local_UB, exp_max, new_global_sum, new_global_max,
        maskUb, m, n, scale, minValue, blockStride, repeatStride);
}

template <typename T>
__simd_vf__ inline void BroadCastMaxSumVF(__ubuf__ float *out_ub, __ubuf__ float *ori_ub, const uint16_t loopM)
{
    RegTensor<float> broadcast_reg;
    MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
    for (uint16_t i = 0; i < loopM; ++i) {
        LoadAlign<T, Reg::LoadDist::DIST_E2B_B32>(broadcast_reg, ori_ub + i * 8);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)out_ub + i * 64, broadcast_reg, preg_all);
    }
}

template <typename T>
__aicore__ inline void BroadcastMaxSum(const LocalTensor<T>& outTensor, const LocalTensor<T> &oriTensor,
                                       uint32_t vecS1RealSize)
{
    __ubuf__ float *out_ub = (__ubuf__ T*)outTensor.GetPhyAddr();
    __ubuf__ float *ori_ub = (__ubuf__ T*)oriTensor.GetPhyAddr();

    // Align8, broadcast one element to 8 elements, one register can store 64 elements,
    // so we can handle 64 / 8 = 8 elements per loop.
    uint16_t loopM = (vecS1RealSize + 7) >> 3;
    BroadCastMaxSumVF<T>(out_ub, ori_ub, loopM);
}

}
#endif // MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_ARCH92_H_