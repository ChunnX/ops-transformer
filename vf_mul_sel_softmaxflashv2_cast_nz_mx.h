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
#ifndef MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_MX_H
#define MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_MX_H

#include "kernel_tensor.h"
using namespace regbaseutil;

namespace FaVectorApi {
/* **************************************************************************************************
 * Muls + Select(optional) + SoftmaxFlashV2 + Cast(fp32->fp16/bf16) + ND2NZ
 * ************************************************************************************************* */
    template <typename T, typename T2>
    __simd_vf__ void process_vec1_noupdate_subloop0(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float *expSumUb, __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN1, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
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
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_sum_brc;
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
        RegTensor<float> vreg_sink_input;
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
        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

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
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        Duplicate(vreg_p_scale, static_cast<float>(pScale));
        Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

        Duplicate(vreg_min, minValue);
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(vreg_input_x1, srcUb + i * 256);
            LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * 256);
            LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * 256);
            LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * 256);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // } else {
            //     if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            //                   IsSameType<T2, hifloat8_t>::value) {
            //         Muls(vreg_input_x1, vreg_input_x1, dScaleQK, preg_all); // Muls(dScaleQK)
            //         Muls(vreg_input_x2, vreg_input_x2, dScaleQK, preg_all);
            //         Muls(vreg_input_x3, vreg_input_x3, dScaleQK, preg_all);
            //         Muls(vreg_input_x4, vreg_input_x4, dScaleQK, preg_all);
            //     }
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_sel1,
                                                                  preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                                                                  vreg_sel2, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                                                                  vreg_sel3_new, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                                                                  vreg_sel4_new, preg_tail_n2);
                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                                  vreg_max_tmp3, preg_all);
            // } 
            // else {
                // Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
                // Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
                //                                                   preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                //                                                   vreg_input_x2, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                //                                                   vreg_input_x3_new, preg_tail_n1);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                //                                                   vreg_input_x4_new, preg_tail_n2);

                // Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                // Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                // Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                //                                                                                   vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
        }
        StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)maxUb), ureg_max, 0);
        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
        // }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max_brc, maxUbStart + i);
            Sub(vreg_max_brc, vreg_max_brc, vreg_ln_p_scale, preg_all);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x1, vreg_input_x2, srcUb + i * 256);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x3, vreg_input_x4, srcUb + floatRepSize * 2 + i * 256);
            ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum3, vreg_exp_sum3,
                                                                                              preg_all);
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)expSumUb), vreg_exp_sum3,
                                                                         ureg_exp_sum, 1);
            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     // preg1: 0011223344556677 preg2: 0000000000000000
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     // preg3: 0000110022003300 preg4: 4400550066007700
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     // preg1: 0011223344556677 preg2: 0000000000000000
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     // preg3: 0000110022003300 preg4: 4400550066007700
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8);
                LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3_1, vreg_exp_merge_tmp_f8e4m3_1, vreg_exp_merge_f8e4m3_indexes);
                Gather(vreg_exp_merge_f8e4m3_2, vreg_exp_merge_tmp_f8e4m3_2, vreg_exp_merge_f8e4m3_indexes);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp_merge_f8e4m3_1, blockStride, repeatStride, preg_all_b8_128);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp_merge_f8e4m3_2, blockStride, repeatStride, preg_all_b8_128);
            // }
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
    }

    template <typename T, typename T2>
    __simd_vf__ void process_vec1_noupdate_subloop1(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float * expMaxUb, __ubuf__ float * inMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * inExpSumUb, __ubuf__ float * tmpExpSumUb,
        __ubuf__ float * tmpExpSumUb2, __ubuf__ float * tmpMaxUb, __ubuf__ float * tmpMaxUb2,
        __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN1, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
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
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_sum_brc;
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
        RegTensor<float> vreg_sink_input;
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
        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

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
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        Duplicate(vreg_p_scale, static_cast<float>(pScale));
        Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

        Duplicate(vreg_min, minValue);
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(vreg_input_x1, srcUb + i * 256);
            LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * 256);
            LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * 256);
            LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * 256);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // } else {
            //     if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            //                   IsSameType<T2, hifloat8_t>::value) {
            //         Muls(vreg_input_x1, vreg_input_x1, dScaleQK, preg_all); // Muls(dScaleQK)
            //         Muls(vreg_input_x2, vreg_input_x2, dScaleQK, preg_all);
            //         Muls(vreg_input_x3, vreg_input_x3, dScaleQK, preg_all);
            //         Muls(vreg_input_x4, vreg_input_x4, dScaleQK, preg_all);
            //     }
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_sel1,
                                                                  preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                                                                  vreg_sel2, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                                                                  vreg_sel3_new, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                                                                  vreg_sel4_new, preg_tail_n2);
                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                                  vreg_max_tmp3, preg_all);
            // }
            // else {
                // Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
                // Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
                //                                                   preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                //                                                   vreg_input_x2, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                //                                                   vreg_input_x3_new, preg_tail_n1);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                //                                                   vreg_input_x4_new, preg_tail_n2);

                // Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                // Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                // Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                //                                                                                   vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max,
                                                                     1);
        }
        StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
        RegTensor<float> vreg_in_max;
        LoadAlign(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(vreg_input_max, tmpMaxUb2);
        Max(vreg_input_max, vreg_in_max, vreg_input_max,
            preg_all); // loop1的max需要和loop0的max取一个max，保证loop1的max是当前的全局最大max，不要被loop1的max覆盖掉
        ExpSub(vreg_exp_max, vreg_in_max, vreg_input_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_input_max, preg_all);

        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
        // }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max_brc, maxUbStart + i);
            Sub(vreg_max_brc, vreg_max_brc, vreg_ln_p_scale, preg_all);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x1, vreg_input_x2, srcUb + i * 256);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x3, vreg_input_x4,
                                                              srcUb + floatRepSize * 2 + i * 256);
            ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all);
            ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum3, vreg_exp_sum3,
                                                                                              preg_all);
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum3,
                                                                         ureg_exp_sum, 1);
            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     // preg1: 0011223344556677 preg2: 0000000000000000
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     // preg3: 0000110022003300 preg4: 4400550066007700
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     // preg1: 0011223344556677 preg2: 0000000000000000
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     // preg3: 0000110022003300 preg4: 4400550066007700
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8);
                LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3_1, vreg_exp_merge_tmp_f8e4m3_1, vreg_exp_merge_f8e4m3_indexes);
                Gather(vreg_exp_merge_f8e4m3_2, vreg_exp_merge_tmp_f8e4m3_2, vreg_exp_merge_f8e4m3_indexes);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp_merge_f8e4m3_1, blockStride, repeatStride, preg_all_b8_128);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp_merge_f8e4m3_2, blockStride, repeatStride, preg_all_b8_128);
            // }
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> first_loop_sum;
        LoadAlign(first_loop_sum, inExpSumUb);
        LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2);
        Mul(first_loop_sum, vreg_exp_max, first_loop_sum, preg_all);
        Add(vreg_exp_sum3, first_loop_sum, vreg_exp_sum_brc, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_exp_sum3, preg_all);
    }


    template <typename T, typename T2>
    __simd_vf__ void process_vec1_update_subloop0(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float * expMaxUb, __ubuf__ float * inMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * inExpSumUb, __ubuf__ float * tmpExpSumUb,
        __ubuf__ float * tmpExpSumUb2, __ubuf__ float * tmpMaxUb, __ubuf__ float * tmpMaxUb2,
        __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4,  __ubuf__ float *preLoopMaxUb, __ubuf__ float *preLoopSumUb, __ubuf__ float *firstLoopSumUb,
        const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN1, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
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
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_subloop_update;
        RegTensor<float> vreg_pre_loop_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_pre_sum;
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
        RegTensor<float> vreg_sink_input;
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

        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        // MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
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
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        Duplicate(vreg_p_scale, static_cast<float>(pScale));
        Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

        Duplicate(vreg_min, minValue);
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(vreg_input_x1, srcUb + i * 256);
            LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * 256);
            LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * 256);
            LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * 256);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                // atten mask
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_sel1,
                                                                  preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                                                                  vreg_sel2, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                                                                  vreg_sel3_new, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                                                                  vreg_sel4_new, preg_all);
                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                                  vreg_max_tmp3, preg_all);
            // } else {
                // Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
                // Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
                //                                                   preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                //                                                   vreg_input_x2, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                //                                                   vreg_input_x3_new, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                //                                                   vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                // Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                // Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                //                                                                                   vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), vreg_input_max,
                                                                         ureg_max, 1);
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);

        LoadAlign(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(vreg_input_max, tmpMaxUb2);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)preLoopMaxUb, vreg_in_max,
                                                          preg_all);
        Max(vreg_max_new, vreg_in_max, vreg_input_max, preg_all);
        ExpSub(vreg_exp_max, vreg_in_max, vreg_max_new, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_max_new, preg_all);


        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        // }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, maxUbStart + i);
            Sub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x1, vreg_input_x2, srcUb + i * 256);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x3, vreg_input_x4,
                                                              srcUb + floatRepSize * 2 + i * 256);
            ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max, preg_all);
            ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max, preg_all);
            ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max, preg_all);
            ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum3, vreg_exp_sum3,
                                                                                              preg_all);
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum3,
                                                                         ureg_exp_sum, 1);

            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_n_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8);
                LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3_1, vreg_exp_merge_tmp_f8e4m3_1, vreg_exp_merge_f8e4m3_indexes);
                Gather(vreg_exp_merge_f8e4m3_2, vreg_exp_merge_tmp_f8e4m3_2, vreg_exp_merge_f8e4m3_indexes);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp_merge_f8e4m3_1, blockStride, repeatStride, preg_all_b8_128);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp_merge_f8e4m3_2, blockStride, repeatStride, preg_all_b8_128);
            // } else if constexpr (IsSameType<T2, half>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_f16, (RegTensor<uint16_t> &)vreg_exp_even1_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_f16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_f16, (RegTensor<uint16_t> &)vreg_exp_even2_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_f16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_n_b16);
            // }
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);

        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(vreg_in_exp_sum, inExpSumUb);
        LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)preLoopSumUb, vreg_in_exp_sum,
                                                          preg_all);
        Mul(vreg_exp_max, vreg_exp_max, vreg_in_exp_sum, preg_all);
        Add(vreg_exp_max, vreg_exp_max, vreg_exp_sum_brc,
            preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_exp_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)firstLoopSumUb, vreg_exp_sum_brc,
                                                          preg_all); // 传给loop 1用于更新rowSum
    }

    template <typename T, typename T2>
    __simd_vf__ void process_vec1_update_subloop1(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float * expMaxUb, __ubuf__ float * inMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * inExpSumUb, __ubuf__ float * tmpExpSumUb,
        __ubuf__ float * tmpExpSumUb2, __ubuf__ float * tmpMaxUb, __ubuf__ float * tmpMaxUb2,
        __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4,  __ubuf__ float *preLoopMaxUb, __ubuf__ float *preLoopSumUb, __ubuf__ float *firstLoopSumUb,
        const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN1, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
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
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_subloop_update;
        RegTensor<float> vreg_pre_loop_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_pre_sum;
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
        RegTensor<float> vreg_sink_input;
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

        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        // MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
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
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        Duplicate(vreg_p_scale, static_cast<float>(pScale));
        Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

        Duplicate(vreg_min, minValue);
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(vreg_input_x1, srcUb + i * 256);
            LoadAlign(vreg_input_x2, srcUb + floatRepSize + i * 256);
            LoadAlign(vreg_input_x3, srcUb + floatRepSize * 2 + i * 256);
            LoadAlign(vreg_input_x4, srcUb + floatRepSize * 3 + i * 256);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
                Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
                Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
                Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding);
                Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                Select(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                Select(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_sel1,
                                                                  preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                                                                  vreg_sel2, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                                                                  vreg_sel3_new, preg_all);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                                                                  vreg_sel4_new, preg_all);
                Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all);
                Max(vreg_max_tmp2, vreg_sel3_new, vreg_sel4_new, preg_all);
                Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                                                                                                  vreg_max_tmp3, preg_all);
            // } else {
                // Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
                // Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
                //                                                   preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                //                                                   vreg_input_x2, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                //                                                   vreg_input_x3_new, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                //                                                   vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                // Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                // Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                //                                                                                   vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), vreg_input_max,
                                                                         ureg_max, 1);
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);

        LoadAlign(vreg_in_max, inMaxUb);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(vreg_input_max, tmpMaxUb2);
        Max(vreg_max_new, vreg_in_max, vreg_input_max, preg_all);
        ExpSub(vreg_subloop_update, vreg_in_max, vreg_max_new, preg_all);
        LoadAlign(vreg_pre_loop_max, preLoopMaxUb);
        ExpSub(vreg_exp_max, vreg_pre_loop_max, vreg_max_new, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)maxUb, vreg_max_new, preg_all);

        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        // }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, maxUbStart + i);
            Sub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x1, vreg_input_x2, srcUb + i * 256);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x3, vreg_input_x4,
                                                              srcUb + floatRepSize * 2 + i * 256);
            ExpSub(vreg_exp_even1, vreg_input_x1, vreg_max, preg_all);
            ExpSub(vreg_exp_odd1, vreg_input_x2, vreg_max, preg_all);
            ExpSub(vreg_exp_even2, vreg_input_x3, vreg_max, preg_all);
            ExpSub(vreg_exp_odd2, vreg_input_x4, vreg_max, preg_all);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
            Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
            Add(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all);
            Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum3, vreg_exp_sum3,
                                                                                              preg_all);
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum3,
                                                                         ureg_exp_sum, 1);

            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_n_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all);
                Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8);
                Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8);
                LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
                Gather(vreg_exp_merge_f8e4m3_1, vreg_exp_merge_tmp_f8e4m3_1, vreg_exp_merge_f8e4m3_indexes);
                Gather(vreg_exp_merge_f8e4m3_2, vreg_exp_merge_tmp_f8e4m3_2, vreg_exp_merge_f8e4m3_indexes);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb1), vreg_exp_merge_f8e4m3_1, blockStride, repeatStride, preg_all_b8_128);
                StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)expUb2), vreg_exp_merge_f8e4m3_2, blockStride, repeatStride, preg_all_b8_128);
            // } else if constexpr (IsSameType<T2, half>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_f16, (RegTensor<uint16_t> &)vreg_exp_even1_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_f16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_f16, (RegTensor<uint16_t> &)vreg_exp_even2_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_f16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_n_b16);
            // }
        }
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);

        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2);
        RegTensor<float> vreg_first_sum;
        LoadAlign(vreg_first_sum, firstLoopSumUb);
        LoadAlign(vreg_pre_sum, preLoopSumUb);
        Mul(vreg_first_sum, vreg_subloop_update, vreg_first_sum, preg_all);
        Add(vreg_first_sum, vreg_first_sum, vreg_exp_sum_brc, preg_all);
        Mul(vreg_pre_sum, vreg_exp_max, vreg_pre_sum, preg_all);
        Add(vreg_pre_sum, vreg_first_sum, vreg_pre_sum, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_pre_sum, preg_all);
    }

} // namespace

#endif // MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_H
