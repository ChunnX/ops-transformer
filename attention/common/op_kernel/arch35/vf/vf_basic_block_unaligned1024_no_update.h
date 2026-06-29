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
 * \file vf_basic_block_unaligned1024_no_update.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED1024_NO_UPDATE_H
#define VF_BASIC_BLOCK_UNALIGNED1024_NO_UPDATE_H

#include "vf_basic_block_utils.h"
#include "vf_basic_block_unaligned1024_common.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

template <typename T, typename T2, typename OUTPUT_T, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0>
__simd_vf__ void ProcessVec1NoUpdateGeneralImpl1024VF(
    __ubuf__ T2 * expUb1, __ubuf__ T2 * expUb2, __ubuf__ T2 * expUb3, __ubuf__ T2 * expUb4, __ubuf__ T2 * expUb5, __ubuf__ T2 * expUb6,
    __ubuf__ T2 * expUb7, __ubuf__ T2 * expUb8, __ubuf__ OUTPUT_T * pseUb, __ubuf__ T * expSumUb, __ubuf__ T * maxUb, __ubuf__ T * maxUbStart,
    __ubuf__ T * srcUb, __ubuf__ uint32_t * maskUb1, __ubuf__ uint32_t * maskUb2, __ubuf__ uint32_t * maskUb3, __ubuf__ uint32_t * maskUb4,
    __ubuf__ uint32_t * maskUb5, __ubuf__ uint32_t * maskUb6, __ubuf__ uint32_t * maskUb7, __ubuf__ uint32_t * maskUb8, __ubuf__ uint32_t * maskUb9,
    __ubuf__ uint32_t * maskUb10, __ubuf__ uint32_t * maskUb11, __ubuf__ uint32_t * maskUb12, __ubuf__ uint32_t * maskUb13, __ubuf__ uint32_t * maskUb14,
    __ubuf__ uint32_t * maskUb15, __ubuf__ uint32_t * maskUb16, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride,
    const uint32_t oriTailN1, const uint32_t oriTailN2, const uint32_t oriTailN3, const uint32_t oriTailN4, const uint32_t oriTailN5,
    const uint32_t oriTailN6, const uint32_t oriTailN7, const uint32_t oriTailN8, const uint32_t tailN1, const uint32_t tailN2, const uint32_t tailN3,
    const uint32_t tailN4, const uint32_t tailN5, const uint32_t tailN6, const uint32_t tailN7, const uint32_t tailN8, uint32_t pltOriTailN1,
    uint32_t pltOriTailN2, uint32_t pltOriTailN3, uint32_t pltOriTailN4, uint32_t pltOriTailN5, uint32_t pltOriTailN6, uint32_t pltOriTailN7, uint32_t pltOriTailN8,
    uint32_t pltTailN1, uint32_t pltTailN2, uint32_t pltTailN3, uint32_t pltTailN4, uint32_t pltTailN5, uint32_t pltTailN6, uint32_t pltTailN7, uint32_t pltTailN8,
    float divValue, const uint16_t m, const uint32_t pseStride, const float slopes, const float posShift, const T scale, const T minValue)
{
    RegTensor<float> vreg_min;
    VREG_FLOAT_DECL_16(vreg_sel);
    RegTensor<float> vreg_sel9_new;
    RegTensor<float> vreg_sel10_new;
    RegTensor<float> vreg_sel11_new;
    RegTensor<float> vreg_sel12_new;
    RegTensor<float> vreg_sel13_new;
    RegTensor<float> vreg_sel14_new;
    RegTensor<float> vreg_sel15_new;
    RegTensor<float> vreg_sel16_new;

    VREG_FLOAT_DECL_16(vreg_input_x);
    RegTensor<float> vreg_input_x9_new;
    RegTensor<float> vreg_input_x10_new;
    RegTensor<float> vreg_input_x11_new;
    RegTensor<float> vreg_input_x12_new;
    RegTensor<float> vreg_input_x13_new;
    RegTensor<float> vreg_input_x14_new;
    RegTensor<float> vreg_input_x15_new;
    RegTensor<float> vreg_input_x16_new;

    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_max_brc;
    RegTensor<float> vreg_zero;

    RegTensor<float> vreg_exp_sum1;

    VREG_FLOAT_DECL_EXP_PAIRS_8(vreg_exp);

    VREG_FLOAT_DECL_16(vreg_pse);

    VREG_FLOAT_DECL_16(vreg_alibi);

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

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

    VREG_PREG_COMPARE_DECL_16();

    Duplicate(vreg_min, minValue);
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                    pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        InitAlibiOffsets(vreg_alibi1, vreg_alibi2, vreg_alibi3, vreg_alibi4,
            vreg_alibi5, vreg_alibi6, vreg_alibi7, vreg_alibi8,
            vreg_alibi9, vreg_alibi10, vreg_alibi11, vreg_alibi12,
            vreg_alibi13, vreg_alibi14, vreg_alibi15, vreg_alibi16,
            posShift, preg_all);
    }
    for (uint16_t i = 0; i < m; ++i) {
        LoadInput16<T>(vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
            vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
            vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12,
            vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16,
            srcUb, i, s2BaseSize);

        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            ScaleInput16(vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
                vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
                vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12,
                vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16,
                scale, preg_all, preg_ori_tail_n1, preg_ori_tail_n2,
                preg_ori_tail_n3, preg_ori_tail_n4, preg_ori_tail_n5, preg_ori_tail_n6,
                preg_ori_tail_n7, preg_ori_tail_n8);
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    ComputePseInnerMulAddSqrt16(vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4,
                        vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8,
                        vreg_pse9, vreg_pse10, vreg_pse11, vreg_pse12,
                        vreg_pse13, vreg_pse14, vreg_pse15, vreg_pse16,
                        vreg_alibi1, vreg_alibi2, vreg_alibi3, vreg_alibi4,
                        vreg_alibi5, vreg_alibi6, vreg_alibi7, vreg_alibi8,
                        vreg_alibi9, vreg_alibi10, vreg_alibi11, vreg_alibi12,
                        vreg_alibi13, vreg_alibi14, vreg_alibi15, vreg_alibi16,
                        slopes, preg_all);
                } else {
                    ComputePseInnerMulAdd16(vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4,
                        vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8,
                        vreg_pse9, vreg_pse10, vreg_pse11, vreg_pse12,
                        vreg_pse13, vreg_pse14, vreg_pse15, vreg_pse16,
                        vreg_alibi1, vreg_alibi2, vreg_alibi3, vreg_alibi4,
                        vreg_alibi5, vreg_alibi6, vreg_alibi7, vreg_alibi8,
                        vreg_alibi9, vreg_alibi10, vreg_alibi11, vreg_alibi12,
                        vreg_alibi13, vreg_alibi14, vreg_alibi15, vreg_alibi16,
                        slopes, preg_all);
                }
            } else {
                if constexpr (IsSameType<T2, bfloat16_t>::value) {
                    LoadCastPseBf16_16(vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4,
                        vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8,
                        vreg_pse9, vreg_pse10, vreg_pse11, vreg_pse12,
                        vreg_pse13, vreg_pse14, vreg_pse15, vreg_pse16,
                        pseUb, i, pseStride, preg_all_b16);
                } else if constexpr (IsSameType<T2, half>::value) {
                    LoadCastPseF16_16(vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4,
                        vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8,
                        vreg_pse9, vreg_pse10, vreg_pse11, vreg_pse12,
                        vreg_pse13, vreg_pse14, vreg_pse15, vreg_pse16,
                        pseUb, i, pseStride, preg_all_b16);
                }
            }
            AddPseToInput16(vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
                vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
                vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12,
                vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16,
                vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4,
                vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8,
                vreg_pse9, vreg_pse10, vreg_pse11, vreg_pse12,
                vreg_pse13, vreg_pse14, vreg_pse15, vreg_pse16,
                preg_all, preg_ori_tail_n1, preg_ori_tail_n2,
                preg_ori_tail_n3, preg_ori_tail_n4,
                preg_ori_tail_n5, preg_ori_tail_n6,
                preg_ori_tail_n7, preg_ori_tail_n8);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            ScaleInput16(vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
                vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
                vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12,
                vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16,
                scale, preg_all, preg_ori_tail_n1, preg_ori_tail_n2,
                preg_ori_tail_n3, preg_ori_tail_n4, preg_ori_tail_n5, preg_ori_tail_n6,
                preg_ori_tail_n7, preg_ori_tail_n8);
        }

        if constexpr (hasAtten == 1) {
            LoadAttenMask16(preg_compare1, preg_compare2, preg_compare3, preg_compare4,
                preg_compare5, preg_compare6, preg_compare7, preg_compare8,
                preg_compare9, preg_compare10, preg_compare11, preg_compare12,
                preg_compare13, preg_compare14, preg_compare15, preg_compare16,
                maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7, maskUb8,
                maskUb9, maskUb10, maskUb11, maskUb12, maskUb13, maskUb14, maskUb15, maskUb16,
                nPadding);

            ApplyAttenMaskSelect16(vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4,
                vreg_sel5, vreg_sel6, vreg_sel7, vreg_sel8,
                vreg_sel9, vreg_sel10, vreg_sel11, vreg_sel12,
                vreg_sel13, vreg_sel14, vreg_sel15, vreg_sel16,
                vreg_min,
                vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
                vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
                vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12,
                vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16,
                preg_compare1, preg_compare2, preg_compare3, preg_compare4,
                preg_compare5, preg_compare6, preg_compare7, preg_compare8,
                preg_compare9, preg_compare10, preg_compare11, preg_compare12,
                preg_compare13, preg_compare14, preg_compare15, preg_compare16);

            Select(vreg_sel9_new, vreg_sel9, vreg_min, preg_ori_tail_n1);
            Select(vreg_sel10_new, vreg_sel10, vreg_min, preg_ori_tail_n2);
            Select(vreg_sel11_new, vreg_sel11, vreg_min, preg_ori_tail_n3);
            Select(vreg_sel12_new, vreg_sel12, vreg_min, preg_ori_tail_n4);
            Select(vreg_sel13_new, vreg_sel13, vreg_min, preg_ori_tail_n5);
            Select(vreg_sel14_new, vreg_sel14, vreg_min, preg_ori_tail_n6);
            Select(vreg_sel15_new, vreg_sel15, vreg_min, preg_ori_tail_n7);
            Select(vreg_sel16_new, vreg_sel16, vreg_min, preg_ori_tail_n8);

            StoreAlign16<T>(vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4,
                vreg_sel5, vreg_sel6, vreg_sel7, vreg_sel8,
                vreg_sel9_new, vreg_sel10_new, vreg_sel11_new, vreg_sel12_new,
                vreg_sel13_new, vreg_sel14_new, vreg_sel15_new, vreg_sel16_new,
                srcUb, i, s2BaseSize, preg_all);

            MaxReduce16(vreg_input_max,
                vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4,
                vreg_sel5, vreg_sel6, vreg_sel7, vreg_sel8,
                vreg_sel9_new, vreg_sel10_new, vreg_sel11_new, vreg_sel12_new,
                vreg_sel13_new, vreg_sel14_new, vreg_sel15_new, vreg_sel16_new,
                preg_all);
        } else {
            Select(vreg_input_x9_new, vreg_input_x9, vreg_min, preg_ori_tail_n1);
            Select(vreg_input_x10_new, vreg_input_x10, vreg_min, preg_ori_tail_n2);
            Select(vreg_input_x11_new, vreg_input_x11, vreg_min, preg_ori_tail_n3);
            Select(vreg_input_x12_new, vreg_input_x12, vreg_min, preg_ori_tail_n4);
            Select(vreg_input_x13_new, vreg_input_x13, vreg_min, preg_ori_tail_n5);
            Select(vreg_input_x14_new, vreg_input_x14, vreg_min, preg_ori_tail_n6);
            Select(vreg_input_x15_new, vreg_input_x15, vreg_min, preg_ori_tail_n7);
            Select(vreg_input_x16_new, vreg_input_x16, vreg_min, preg_ori_tail_n8);

            StoreAlign16<T>(vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
                vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
                vreg_input_x9_new, vreg_input_x10_new, vreg_input_x11_new, vreg_input_x12_new,
                vreg_input_x13_new, vreg_input_x14_new, vreg_input_x15_new, vreg_input_x16_new,
                srcUb, i, s2BaseSize, preg_all);

            MaxReduce16(vreg_input_max,
                vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
                vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
                vreg_input_x9_new, vreg_input_x10_new, vreg_input_x11_new, vreg_input_x12_new,
                vreg_input_x13_new, vreg_input_x14_new, vreg_input_x15_new, vreg_input_x16_new,
                preg_all);
        }
        StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
    }
    StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), ureg_max, 0);
    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg_max_brc, maxUbStart + i);

        LoadInputDinterleave8<T>(vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
            vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
            vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12,
            vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16,
            srcUb, i, s2BaseSize);

        ExpSub16(vreg_exp_even1, vreg_exp_odd1, vreg_exp_even2, vreg_exp_odd2,
            vreg_exp_even3, vreg_exp_odd3, vreg_exp_even4, vreg_exp_odd4,
            vreg_exp_even5, vreg_exp_odd5, vreg_exp_even6, vreg_exp_odd6,
            vreg_exp_even7, vreg_exp_odd7, vreg_exp_even8, vreg_exp_odd8,
            vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
            vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
            vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12,
            vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16,
            vreg_max_brc, preg_all);

        ExpSumReduceStore1024<float, T2>(vreg_exp_sum1,
            vreg_exp_even1, vreg_exp_odd1, vreg_exp_even2, vreg_exp_odd2,
            vreg_exp_even3, vreg_exp_odd3, vreg_exp_even4, vreg_exp_odd4,
            vreg_exp_even5, vreg_exp_odd5, vreg_exp_even6, vreg_exp_odd6,
            vreg_exp_even7, vreg_exp_odd7, vreg_exp_even8, vreg_exp_odd8,
            ureg_exp_sum, expSumUb, preg_all);

        if constexpr (IsSameType<T2, bfloat16_t>::value) {
            CastStoreExpBf16_1024<T2>(vreg_exp_even1, vreg_exp_odd1,
                vreg_exp_even2, vreg_exp_odd2, vreg_exp_even3, vreg_exp_odd3,
                vreg_exp_even4, vreg_exp_odd4, vreg_exp_even5, vreg_exp_odd5,
                vreg_exp_even6, vreg_exp_odd6, vreg_exp_even7, vreg_exp_odd7,
                vreg_exp_even8, vreg_exp_odd8,
                expUb1, expUb2, expUb3, expUb4, expUb5, expUb6, expUb7, expUb8,
                blockStride, repeatStride, preg_all, preg_all_b16);
        } else if constexpr (IsSameType<T2, half>::value) {
            CastStoreExpF16_1024<T2>(vreg_exp_even1, vreg_exp_odd1,
                vreg_exp_even2, vreg_exp_odd2, vreg_exp_even3, vreg_exp_odd3,
                vreg_exp_even4, vreg_exp_odd4, vreg_exp_even5, vreg_exp_odd5,
                vreg_exp_even6, vreg_exp_odd6, vreg_exp_even7, vreg_exp_odd7,
                vreg_exp_even8, vreg_exp_odd8,
                expUb1, expUb2, expUb3, expUb4, expUb5, expUb6, expUb7, expUb8,
                blockStride, repeatStride, preg_all, preg_all_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
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
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
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

    const uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
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

    ProcessVec1NoUpdateGeneralImpl1024VF<T, T2, OUTPUT_T, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
        expUb1, expUb2, expUb3, expUb4, expUb5, expUb6, expUb7, expUb8, pseUb, expSumUb, maxUb, maxUbStart, srcUb,
        maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7, maskUb8, maskUb9, maskUb10, maskUb11,
        maskUb12, maskUb13, maskUb14, maskUb15, maskUb16, nPadding, blockStride, repeatStride, oriTailN1, oriTailN2,
        oriTailN3, oriTailN4, oriTailN5, oriTailN6, oriTailN7, oriTailN8, tailN1, tailN2, tailN3, tailN4, tailN5,
        tailN6, tailN7, tailN8, pltOriTailN1, pltOriTailN2, pltOriTailN3, pltOriTailN4, pltOriTailN5, pltOriTailN6,
        pltOriTailN7, pltOriTailN8, pltTailN1, pltTailN2, pltTailN3, pltTailN4, pltTailN5, pltTailN6, pltTailN7, pltTailN8,
        divValue, m, pseStride, slopes, posShift, scale, minValue);
}
} // namespace

#endif // VF_BASIC_BLOCK_UNALIGNED1024_NO_UPDATE_H
