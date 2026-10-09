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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_dn.h
 * \brief
 */

#ifndef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#include "kernel_tensor.h"
namespace FaVectorApi {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Reg;

#define VMULSCVT false
#define DROPOUT false

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__simd_vf__ inline void ProcessVec1DnNoUpdateVF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                                __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                                __ubuf__ float* new_global_max, __ubuf__ uint32_t* maskUb,
                                                __ubuf__ uint8_t* indexesUb, const uint32_t m, const uint32_t n,
                                                const uint32_t originN, const T scale, float deScaleQK,
                                                const T minValue, float keepProb, bool needAtten, const float dScale,
                                                const uint32_t blockStride, const uint32_t repeatStride,
                                                const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;
    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }

    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t* mask_ub0 = maskUb;
    __ubuf__ uint32_t* mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t* mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t* mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    uint32_t paddingRows = m;
    MaskReg paddingMask = UpdateMask<T>(paddingRows);
    for (uint16_t i = originN; i < ubN; ++i) {
        // 仅写实际 pitch 对应的 M lanes, 避免全宽写入越过相邻通信缓冲
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, paddingMask);
    }
    mem_bar(VST_VLD);

    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                   IsSameType<T2, hifloat8_t>::value) &&
                  hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);

    if constexpr (hasSink) {
        Max(max0, max0, vreg_sink_input, preg_108);
    }

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
            Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
            Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
            Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }

        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1,
                         vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                             IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__aicore__ inline void ProcessVec1DnNoUpdate(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                             const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                             const LocalTensor<T>& expMaxTensor,
                                             const LocalTensor<uint8_t>& vselrIndexesBuf,
                                             const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n,
                                             const uint32_t originN, const T scale, float deScaleQK, const T minValue,
                                             float keepProb, bool needAtten, const float pScale = 1.0f,
                                             const float sinkValue = 0.0f)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t* maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    ProcessVec1DnNoUpdateVF<T, T2, hasAtten, ubN, hasSink>(
        x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, indexesUb, m, n, originN, scale,
        deScaleQK, minValue, keepProb, needAtten, dScale, blockStride, repeatStride, pScale, sinkValue);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__simd_vf__ inline void ProcessVec1DnUpdateVF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                              __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                              __ubuf__ float* new_global_max, __ubuf__ uint32_t* maskUb,
                                              __ubuf__ uint8_t* indexesUb, const uint32_t m, const uint32_t n,
                                              const uint32_t originN, const T scale, float deScaleQK, const T minValue,
                                              float keepProb, bool needAtten, const float dScale,
                                              const uint32_t blockStride, const uint32_t repeatStride,
                                              const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;

    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t* mask_ub0 = maskUb;
    __ubuf__ uint32_t* mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t* mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t* mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    uint32_t paddingRows = m;
    MaskReg paddingMask = UpdateMask<T>(paddingRows);
    for (uint16_t i = originN; i < ubN; ++i) {
        // 仅写实际 pitch 对应的 M lanes, 避免全宽写入越过相邻通信缓冲
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, paddingMask);
    }
    mem_bar(VST_VLD);

    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                   IsSameType<T2, hifloat8_t>::value) &&
                  hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);
    if constexpr (hasSink) {
        Max(max0, max0, vreg_sink_input, preg_108);
    }

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
            Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
            Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
            Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }
        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1,
                         vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储156在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                             IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);

            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }
    RegTensor<float> vreg_l0;
    LoadAlign(vreg_l0, new_global_sum);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__aicore__ inline void ProcessVec1DnUpdate(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                           const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                           const LocalTensor<T>& expMaxTensor,
                                           const LocalTensor<uint8_t>& vselrIndexesBuf,
                                           const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n,
                                           const uint32_t originN, const T scale, float deScaleQK, const T minValue,
                                           float keepProb, bool needAtten, const float pScale = 1.0f,
                                           const float sinkValue = 0.0f)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t* maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    ProcessVec1DnUpdateVF<T, T2, hasAtten, ubN, hasSink>(
        x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, indexesUb, m, n, originN, scale,
        deScaleQK, minValue, keepProb, needAtten, dScale, blockStride, repeatStride, pScale, sinkValue);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnNoUpdateMxfp8VF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                                     __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                                     __ubuf__ float* new_global_max, __ubuf__ uint8_t* indexesUb,
                                                     __ubuf__ fp8_e8m0_t* pScaleSubLoop0, const uint32_t m,
                                                     const uint32_t n, const uint32_t originN, const T scale,
                                                     float deScaleQK, float pScale, const T minValue, float keepProb,
                                                     const float dScale, const uint32_t blockStride,
                                                     const uint32_t repeatStride, int32_t subLoop, int64_t maskLine)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3, preg_compare_all;
    preg_compare_all = CreateMask<T, MaskPattern::ALLF>();
    MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }

    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        Arange(vreg_chunk0, (float)maskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Compare<T, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);
    Muls(max0, max0, INV_LN2, preg_108);
    Truncate<T, RoundMode::CAST_CEIL>(max0, max0, preg_108);
    Muls(max0, max0, LN2, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    if (subLoop == 1) {
        LoadAlign(vreg_x_max_f32_b, new_global_max);
        Max(max0, max0, vreg_x_max_f32_b, preg_108);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    }

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    LoadAlign(idx_nd2nz, indexesUb);
    uint16_t loopNum = ubN / 8; // ubN：s2BaseSize = 256

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

        Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
        Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
        Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
        Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134); // vreg_x_exp_i softmax分子

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);

        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum_2, vreg_x_sum_2, vreg_x_sum_3, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum_6, vreg_x_sum_6, vreg_x_sum_7, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_2, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_6, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_4, preg_134);

    if (subLoop == 0) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum_0, preg_134);
    } else {
        RegTensor<float> first_loop_sum;
        LoadAlign(first_loop_sum, new_global_sum);
        Mul(first_loop_sum, vreg_x_max_f32_b, first_loop_sum, preg_134);
        Add(vreg_x_sum_0, first_loop_sum, vreg_x_sum_0, preg_134);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum_0, preg_134);
        // pscale update
        RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
        RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;
        Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_x_max_f32_b, preg_135);
        Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_x_max_f32_b, preg_135);
        Or((RegTensor<uint16_t>&)vreg_p_scale_bf16_0, (RegTensor<uint16_t>&)vreg_p_scale_bf16_0,
           (RegTensor<uint16_t>&)vreg_p_scale_bf16_1, preg_108);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0, vreg_p_scale_bf16_0, preg_108);
        StoreAlign<fp8_e8m0_t, Reg::StoreDist::DIST_PACK_B16>(((__ubuf__ fp8_e8m0_t*&)pScaleSubLoop0),
                                                              vreg_p_scale_f8e8m0, preg_108);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnNoUpdateMxfp8(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                                  const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                                  const LocalTensor<T>& expMaxTensor,
                                                  const LocalTensor<uint8_t>& vselrIndexesBuf, const uint32_t m,
                                                  const uint32_t n, const uint32_t originN, const T scale,
                                                  float deScaleQK, float pScale, const T minValue, float keepProb,
                                                  int32_t subLoop, const LocalTensor<fp8_e8m0_t>& pScaleSubLoop0Tensor,
                                                  int64_t maskLine)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t* pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t*)pScaleSubLoop0Tensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    ProcessVec1DnNoUpdateMxfp8VF<T, T2, hasAtten, ubN>(
        x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, indexesUb, pScaleSubLoop0Ub, m, n,
        originN, scale, deScaleQK, pScale, minValue, keepProb, dScale, blockStride, repeatStride, subLoop, maskLine);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnUpdateMxfp8VF(
    __ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB, __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
    __ubuf__ float* new_global_max, __ubuf__ uint8_t* indexesUb, __ubuf__ fp8_e8m0_t* pScaleSubLoop0, const uint32_t m,
    const uint32_t n, const uint32_t originN, const T scale, float deScaleQK, float pScale, const T minValue,
    float keepProb, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    __ubuf__ float* pre_loop_max, __ubuf__ float* pre_loop_sum, __ubuf__ float* first_loop_sum, int32_t subLoop,
    int64_t maskLine)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<float> vreg_subloop_update;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3, preg_compare_all;
    preg_compare_all = CreateMask<T, MaskPattern::ALLF>();
    MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        Arange(vreg_chunk0, (float)maskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Compare<T, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);
    Muls(max0, max0, INV_LN2, preg_108);
    Truncate<T, RoundMode::CAST_CEIL>(max0, max0, preg_108);
    Muls(max0, max0, LN2, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    if (subLoop == 0) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)pre_loop_max, vreg_x_max_f32_b, preg_108);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    } else {
        FusedExpSub(vreg_subloop_update, vreg_x_max_f32_b, max0, preg_134);
        LoadAlign(vreg_x_max_f32_b, pre_loop_max);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    }

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);
    loopNum = ubN / 8;

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

        Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
        Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
        Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
        Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134); // vreg_x_exp_i softmax分子

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);

        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum_2, vreg_x_sum_2, vreg_x_sum_3, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum_6, vreg_x_sum_6, vreg_x_sum_7, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_2, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_6, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_4, preg_134);

    RegTensor<float> vreg_l0;
    if (subLoop == 0) {
        LoadAlign(vreg_l0, new_global_sum);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)pre_loop_sum, vreg_l0, preg_134);
        Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
        Add(vreg_l0, vreg_l0, vreg_x_sum_0, preg_134);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_l0, preg_134);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)first_loop_sum, vreg_x_sum_0, preg_134);
    } else {
        RegTensor<float> vreg_l1;
        LoadAlign(vreg_l0, first_loop_sum);
        LoadAlign(vreg_l1, pre_loop_sum);
        Mul(vreg_l0, vreg_subloop_update, vreg_l0, preg_134);
        Add(vreg_x_sum_0, vreg_l0, vreg_x_sum_0, preg_134);
        Mul(vreg_l1, vreg_x_max_f32_b, vreg_l1, preg_134);
        Add(vreg_l0, vreg_x_sum_0, vreg_l1, preg_134);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_l0, preg_134);
        // pscale update
        RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
        RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;
        Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_subloop_update, preg_135);
        Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_subloop_update, preg_135);
        Or((RegTensor<uint16_t>&)vreg_p_scale_bf16_0, (RegTensor<uint16_t>&)vreg_p_scale_bf16_0,
           (RegTensor<uint16_t>&)vreg_p_scale_bf16_1, preg_108);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0, vreg_p_scale_bf16_0, preg_108);
        StoreAlign<fp8_e8m0_t, Reg::StoreDist::DIST_PACK_B16>(((__ubuf__ fp8_e8m0_t*&)pScaleSubLoop0),
                                                              vreg_p_scale_f8e8m0, preg_108);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnUpdateMxfp8(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t>& vselrIndexesBuf,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK, float pScale,
    const T minValue, float keepProb, const LocalTensor<float>& preLoopMaxTensor,
    const LocalTensor<float>& preLoopSumTensor, const LocalTensor<float>& firstLoopSumTensor, int32_t subLoop,
    const LocalTensor<fp8_e8m0_t>& pScaleSubLoop0Tensor, int64_t maskLine)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ float* pre_loop_max = (__ubuf__ T*)preLoopMaxTensor.GetPhyAddr();
    __ubuf__ float* pre_loop_sum = (__ubuf__ T*)preLoopSumTensor.GetPhyAddr();
    __ubuf__ float* first_loop_sum = (__ubuf__ T*)firstLoopSumTensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t* pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t*)pScaleSubLoop0Tensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = nullptr;
    __ubuf__ uint8_t* indexesPScaleUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    ProcessVec1DnUpdateMxfp8VF<T, T2, hasAtten, ubN>(
        x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, indexesUb, pScaleSubLoop0Ub, m, n,
        originN, scale, deScaleQK, pScale, minValue, keepProb, dScale, blockStride, repeatStride, pre_loop_max,
        pre_loop_sum, first_loop_sum, subLoop, maskLine);
}

/*
 * @ingroup ProcessVec1Vf
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [in] srcTensor, input LocalTensor
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] n, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] originN, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isUpdate, enable flash mode
 * @param [in] oriNRange, originN range
 */

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1VfDnMxfp8(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                            const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                            const LocalTensor<T>& expMaxTensor, TBuf<>* vselrIndexesBuf,
                                            const LocalTensor<fp8_e8m0_t>& pScaleSubLoop0Tensor, const uint32_t m,
                                            const uint32_t n, const uint32_t originN, const T scale, float deScaleQK,
                                            float pScale, const T minValue, float keepProb,
                                            const LocalTensor<T>& preLoopMaxTensor,
                                            const LocalTensor<T>& preLoopSumTensor,
                                            const LocalTensor<T>& firstLoopSumTensor, int32_t subLoop, int64_t maskLine)
{
    if constexpr (!isUpdate) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnNoUpdateMxfp8<T, T2, hasAtten, ubN>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                                                         indexesTensor, m, n, originN, scale, deScaleQK, pScale,
                                                         minValue, keepProb, subLoop, pScaleSubLoop0Tensor, maskLine);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnUpdateMxfp8<T, T2, hasAtten, ubN>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                                                       indexesTensor, m, n, originN, scale, deScaleQK, pScale, minValue,
                                                       keepProb, preLoopMaxTensor, preLoopSumTensor, firstLoopSumTensor,
                                                       subLoop, pScaleSubLoop0Tensor, maskLine);
    }
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 128,
          bool hasSink = false, bool hasInvalidLine = false>
__simd_vf__ inline void ProcessVec1DnCausalVF(
    __ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB, __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
    __ubuf__ float* new_global_max, __ubuf__ uint32_t* maskUb, __ubuf__ T* tmpExpSumUb, __ubuf__ T* tmpMaxUb,
    __ubuf__ uint8_t* indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN, const T scale,
    float deScaleQK, const T minValue, float keepProb, bool needAtten, const float dScale, const uint32_t blockStride,
    const uint32_t repeatStride, const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;

    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t* mask_ub0 = maskUb;
    __ubuf__ uint32_t* mask_ub1 = maskUb + (m >> 2);
    __ubuf__ uint32_t* mask_ub2 = maskUb + (m >> 2) * 2;
    __ubuf__ uint32_t* mask_ub3 = maskUb + (m >> 2) * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                   IsSameType<T2, hifloat8_t>::value || IsSameType<T2, bfloat16_t>::value ||
                   IsSameType<T2, half>::value) &&
                  hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                if constexpr (hasInvalidLine) {
                    Muls(src0, src0, dScale, preg_108);
                    Muls(src1, src1, dScale, preg_108);
                    Muls(src2, src2, dScale, preg_108);
                    Muls(src3, src3, dScale, preg_108);
                }
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, Reg::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                if constexpr (hasInvalidLine) {
                    Muls(src0, src0, dScale, preg_108);
                    Muls(src1, src1, dScale, preg_108);
                    Muls(src2, src2, dScale, preg_108);
                    Muls(src3, src3, dScale, preg_108);
                    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                }
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            if constexpr (hasInvalidLine) {
                Muls(src0, src0, dScale, preg_108);
                Muls(src1, src1, dScale, preg_108);
                Muls(src2, src2, dScale, preg_108);
                Muls(src3, src3, dScale, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
            }
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }
    if constexpr (isUpdate) {
        LoadAlign(vreg_x_max_f32_b, new_global_max);
    }
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    if constexpr (!hasInvalidLine) {
        Muls(max0, max0, dScale, preg_108);
    }
    if constexpr (hasSink) {
        Max(max0, max0, vreg_sink_input, preg_108);
    }

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    if constexpr (isUpdate) {
        Max(max0, max0, vreg_x_max_f32_b, preg_108);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)tmpMaxUb, max0, preg_108);
    } else {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);
    }
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }
    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }
        if constexpr (!hasInvalidLine) {
            Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
            Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
            Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
            Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);
        }

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            if constexpr (!hasInvalidLine) {
                Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
                Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
                Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
                Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);
            }

            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }
        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1,
                         vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储156在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                             IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);

            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }
    if constexpr (isUpdate) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)tmpExpSumUb, vreg_x_sum0, preg_134);
    } else {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum0, preg_134);
    }
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 128,
          bool hasSink = false, bool hasInvalidLine = false>
__aicore__ inline void ProcessVec1DnCausal(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t>& vselrIndexesBuf,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t n, const uint32_t originN, const T scale, float deScaleQK, const T minValue, float keepProb,
    bool needAtten, const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t* maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = nullptr;
    __ubuf__ T* tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T* tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    ProcessVec1DnCausalVF<T, T2, isUpdate, hasAtten, ubN, hasSink, hasInvalidLine>(
        x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, maskUb, tmpExpSumUb, tmpMaxUb, indexesUb,
        m, n, originN, scale, deScaleQK, minValue, keepProb, needAtten, dScale, blockStride, repeatStride, pScale,
        sinkValue);
}
// DN tail softmax initialization with BF16 P output.
template <typename T, typename T2, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnTailNoUpdateMxfp8VF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                                         __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                                         __ubuf__ float* new_global_max,
                                                         __ubuf__ fp8_e8m0_t* pScaleSubLoop0, const uint32_t m,
                                                         const uint32_t originN, const T scale, const float deScaleQK,
                                                         const T minValue, const T pScale, const bool needAtten = false,
                                                         const int64_t maskLine = 0)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum0;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;
    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

    MaskReg preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_135 = CreateMask<T, MaskPattern::ALL>();
    // C1 uses a dynamic row pitch (m <= 64). Only the first m lanes belong
    // to this score row; full-vector stores overlap the following rows and
    // can overwrite their causal mask with predicates for different queries.
    uint32_t scoreRowWidth = m;
    MaskReg scoreRowMask = UpdateMask<T>(scoreRowWidth);
    uint32_t sreg_92 = (uint32_t)128ULL;
    MaskReg preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    RegTensor<float> vreg_min;

    // bf16双指针: x_exp_1 = x_exp + ubN*4
    __ubuf__ T2* x_exp_1 = x_exp + (ubN * 4);
    const uint32_t blockStride = ubN >> 1 | 0x1; // 65
    const uint32_t repeatStride = 1;
    const float dScale = scale * deScaleQK; // Mxfp8: 量化QK反量化

    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min,
                                                          scoreRowMask);
    }
    mem_bar(VST_VLD);
    // Remaining score rows are padding initialized to minValue above.
    const uint16_t scoreLoopNum = static_cast<uint16_t>((originN + 3U) / 4U);

    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;
    MaskReg preg_compare0;
    MaskReg preg_compare1;
    MaskReg preg_compare2;
    MaskReg preg_compare3;
    if (needAtten) {
        Arange(vreg_chunk0, (float)maskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_m = 0; iter_m < scoreLoopNum; ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Compare<T, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, scoreRowMask);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, scoreRowMask);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, scoreRowMask);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, scoreRowMask);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < scoreLoopNum; ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    Max(max0, max0, max2, preg_108);
    Max(max0, max0, max1, preg_108);
    Max(max0, max0, max3, preg_108);
    Muls(max0, max0, dScale, preg_108);
    // ln2格点化max(为e8m0 2幂scale折叠)
    Muls(max0, max0, INV_LN2, preg_108);
    Truncate<T, RoundMode::CAST_CEIL>(max0, max0, preg_108);
    Muls(max0, max0, LN2, preg_108);

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    // Match the main MXFP8 path: persist max after folding -ln(pScale).
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    uint16_t loopNum = ubN / 4;
    if (needAtten) {
        // Exp reloads the scores written by the causal-mask loop.
        mem_bar(VST_VLD);
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        // 双pack段式store
        Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
        Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
        DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

        Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
        Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
        DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);

        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum_3, preg_134);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum0, preg_134);
    (void)pScaleSubLoop0; // NoUpdate无pScale更新(与Mxfp8 NoUpdate一致)
}

// DN tail softmax update with BF16 P output.
template <typename T, typename T2, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnTailUpdateMxfp8VF(
    __ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB, __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
    __ubuf__ float* new_global_max, __ubuf__ fp8_e8m0_t* pScaleSubLoop0, __ubuf__ float* pre_loop_max,
    __ubuf__ float* pre_loop_sum, __ubuf__ float* first_loop_sum, int32_t subLoop, const uint32_t m,
    const uint32_t originN, const T scale, const float deScaleQK, const T minValue, const T pScale,
    const bool needAtten = false, const int64_t maskLine = 0)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_l0;
    RegTensor<float> vreg_l1;
    RegTensor<float> vreg_subloop_update;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;
    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

    MaskReg preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_135 = CreateMask<T, MaskPattern::ALL>();
    // C1 uses a dynamic row pitch (m <= 64). Only the first m lanes belong
    // to this score row; full-vector stores overlap the following rows and
    // can overwrite their causal mask with predicates for different queries.
    uint32_t scoreRowWidth = m;
    MaskReg scoreRowMask = UpdateMask<T>(scoreRowWidth);
    uint32_t sreg_92 = (uint32_t)128ULL;
    MaskReg preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    RegTensor<float> vreg_min;

    __ubuf__ T2* x_exp_1 = x_exp + (ubN * 4);
    const uint32_t blockStride = ubN >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    const float dScale = scale * deScaleQK;

    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min,
                                                          scoreRowMask);
    }
    mem_bar(VST_VLD);
    // Remaining score rows are padding initialized to minValue above.
    const uint16_t scoreLoopNum = static_cast<uint16_t>((originN + 3U) / 4U);

    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;
    MaskReg preg_compare0;
    MaskReg preg_compare1;
    MaskReg preg_compare2;
    MaskReg preg_compare3;
    if (needAtten) {
        Arange(vreg_chunk0, (float)maskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_m = 0; iter_m < scoreLoopNum; ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Compare<T, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, scoreRowMask);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, scoreRowMask);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, scoreRowMask);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, scoreRowMask);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < scoreLoopNum; ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max0, max0, max1, preg_108);
    Max(max0, max0, max3, preg_108);
    Muls(max0, max0, dScale, preg_108);
    Muls(max0, max0, INV_LN2, preg_108);
    Truncate<T, RoundMode::CAST_CEIL>(max0, max0, preg_108);
    Muls(max0, max0, LN2, preg_108);
    // The previous max already includes -ln(pScale). Adjust the local max
    // before merging so main and tail use the same softmax state.
    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)pre_loop_max, vreg_x_max_f32_b, preg_108);
    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    uint16_t loopNum = ubN / 4;
    if (needAtten) {
        // Exp reloads the scores written by the causal-mask loop.
        mem_bar(VST_VLD);
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
        Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
        DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

        Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
        Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
        DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);

        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

        StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum_3, preg_134);

    // subLoop==0: 折叠preLoop/firstLoop并接力sum
    LoadAlign(vreg_l0, new_global_sum);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)pre_loop_sum, vreg_l0, preg_134);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_l0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)first_loop_sum, vreg_x_sum0, preg_134);
    (void)subLoop;
    (void)pScaleSubLoop0;
}

// DN tail softmax with BF16 P output and MXFP8 max/sum state.
template <typename T, typename T2, bool isUpdate = false>
__aicore__ inline void ProcessVec1VfDnTailMxfp8(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, TBuf<>* vselrIndexesBuf,
    const LocalTensor<fp8_e8m0_t>& pScaleSubLoop0Tensor, const LocalTensor<float>& preLoopMaxTensor,
    const LocalTensor<float>& preLoopSumTensor, const LocalTensor<float>& firstLoopSumTensor, uint32_t subLoop,
    const uint16_t m, const uint32_t originN, const float scale, const float deScaleQK, const T minValue,
    const float pScale, const bool needAtten = false, const int64_t maskLine = 0)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t* pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t*)pScaleSubLoop0Tensor.GetPhyAddr();
    __ubuf__ float* pre_loop_max = (__ubuf__ float*)preLoopMaxTensor.GetPhyAddr();
    __ubuf__ float* pre_loop_sum = (__ubuf__ float*)preLoopSumTensor.GetPhyAddr();
    __ubuf__ float* first_loop_sum = (__ubuf__ float*)firstLoopSumTensor.GetPhyAddr();
    if constexpr (!isUpdate) {
        ProcessVec1DnTailNoUpdateMxfp8VF<T, T2, 128>(x_exp, input_x_local_UB, exp_max_fp32, new_global_sum,
                                                     new_global_max, pScaleSubLoop0Ub, m, originN, scale, deScaleQK,
                                                     minValue, pScale, needAtten, maskLine);
    } else {
        ProcessVec1DnTailUpdateMxfp8VF<T, T2, 128>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, pScaleSubLoop0Ub, pre_loop_max,
            pre_loop_sum, first_loop_sum, subLoop, m, originN, scale, deScaleQK, minValue, pScale, needAtten, maskLine);
    }
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 128,
          bool hasSink = false>
__aicore__ inline void ProcessVec1VfDn(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                       const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& expMaxTensor, TBuf<>* vselrIndexesBuf,
                                       const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n,
                                       const uint32_t originN, const T scale, float deScaleQK, const T minValue,
                                       float keepProb, bool needAtten, const float pScale = 1.0f,
                                       const float sinkValue = 0.0f)
{
    if constexpr (!isUpdate) {
        // 索引序列，用于在VF代码中量化cast后的gather指令
        // 在对计算结果进行cast之后，数据从fp32转换成了fp8，可能会得到一个乱序的非紧密排布的结果，需要定义一个索引序列来把结果调整正确
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnNoUpdate<T, T2, hasAtten, ubN, hasSink>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, m, n, originN,
            scale, deScaleQK, minValue, keepProb, needAtten, pScale, sinkValue);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnUpdate<T, T2, hasAtten, ubN, hasSink>(dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                                                           indexesTensor, maskTensor, m, n, originN, scale, deScaleQK,
                                                           minValue, keepProb, needAtten, pScale, sinkValue);
    }
}
template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256,
          bool hasSink = false, bool hasInvalidLine = false>
__aicore__ inline void ProcessVec1VfDnCausal(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                             const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                             const LocalTensor<T>& expMaxTensor, TBuf<>* vselrIndexesBuf,
                                             const LocalTensor<uint8_t>& maskTensor,
                                             const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
                                             const uint32_t n, const uint32_t originN, const T scale, float deScaleQK,
                                             const T minValue, float keepProb, bool needAtten,
                                             const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    LocalTensor<uint8_t> indexesTensor;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
    }
    ProcessVec1DnCausal<T, T2, isUpdate, hasAtten, ubN, hasSink, hasInvalidLine>(
        dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, sharedTmpBuffer, m, n,
        originN, scale, deScaleQK, minValue, keepProb, needAtten, pScale, sinkValue);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnNoUpdatePerTokenHeadVF(
    __ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB, __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
    __ubuf__ float* new_global_max, __ubuf__ uint8_t* indexesUb, __ubuf__ float* qScaleUb, __ubuf__ float* kScaleUb,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK, const T minValue,
    float keepProb, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, int64_t maskLine = 0)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, vreg_local_max;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }

    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;

    Duplicate(vreg_min, minValue);
    Duplicate(vreg_local_max, minValue);

    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_108);
    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                   IsSameType<T2, hifloat8_t>::value) &&
                  hasAtten) {
        Arange(vreg_chunk0, (float)maskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            uint32_t base = (uint32_t)iter_m * 256;
            LoadAlign(src0, src_ub0 + base);
            LoadAlign(src1, src_ub1 + base);
            LoadAlign(src2, src_ub2 + base);
            LoadAlign(src3, src_ub3 + base);

            Mul(src0, src0, vreg_qscale_vec, preg_135);
            Mul(src1, src1, vreg_qscale_vec, preg_135);
            Mul(src2, src2, vreg_qscale_vec, preg_135);
            Mul(src3, src3, vreg_qscale_vec, preg_135);
            // 4份ub，解bank冲突
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
            Mul(src0, src0, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 + iter_m * 4 + 1);
            Mul(src1, src1, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 2 + iter_m * 4 + 2);
            Mul(src2, src2, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 3 + iter_m * 4 + 3);
            Mul(src3, src3, vreg_kscale_val, preg_135);

            Compare<T, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + base, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + base, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + base, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + base, src3, preg_108);
            Max(max0, src0, src1, preg_108);
            Max(max1, src2, src3, preg_108);
            Max(max0, max0, max1, preg_108);
            Max(vreg_local_max, max0, vreg_local_max, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            uint32_t base = (uint32_t)iter_m * 256;
            LoadAlign(src0, src_ub0 + base);
            LoadAlign(src1, src_ub1 + base);
            LoadAlign(src2, src_ub2 + base);
            LoadAlign(src3, src_ub3 + base);

            Mul(src0, src0, vreg_qscale_vec, preg_135);
            Mul(src1, src1, vreg_qscale_vec, preg_135);
            Mul(src2, src2, vreg_qscale_vec, preg_135);
            Mul(src3, src3, vreg_qscale_vec, preg_135);
            // 4份ub，解bank冲突
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
            Mul(src0, src0, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 + iter_m * 4 + 1);
            Mul(src1, src1, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 2 + iter_m * 4 + 2);
            Mul(src2, src2, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 3 + iter_m * 4 + 3);
            Mul(src3, src3, vreg_kscale_val, preg_135);

            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + base, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + base, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + base, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + base, src3, preg_108);
            Max(max0, src0, src1, preg_108);
            Max(max1, src2, src3, preg_108);
            Max(max0, max0, max1, preg_108);
            Max(vreg_local_max, max0, vreg_local_max, preg_108);
        }
    }

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(vreg_local_max, vreg_local_max, vreg_ln_p_scale, preg_108);

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, vreg_local_max, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    LoadAlign(idx_nd2nz, indexesUb);
    constexpr uint16_t loopNum = 32;

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, vreg_local_max, preg_134);

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

        // -----------------------------------------------------------------------------//
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, vreg_local_max, preg_134);

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);

        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum_2, vreg_x_sum_2, vreg_x_sum_3, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum_6, vreg_x_sum_6, vreg_x_sum_7, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_2, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_6, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_4, preg_134);

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum_0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnNoUpdatePerTokenHeadTailVF(
    __ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB, __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
    __ubuf__ float* new_global_max, __ubuf__ uint8_t* indexesUb, __ubuf__ float* qScaleUb, __ubuf__ float* kScaleUb,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK, const T minValue,
    float keepProb, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, int64_t maskLine = 0)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, vreg_local_max;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;
    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }

    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;

    Duplicate(vreg_min, minValue);
    Duplicate(vreg_local_max, minValue);

    // qscale和kscale应用循环
    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_108);
    for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
        uint32_t base = (uint32_t)iter_m * 256;
        LoadAlign(vreg_data_tmp0, src_ub0 + base);
        LoadAlign(vreg_data_tmp1, src_ub1 + base);
        LoadAlign(vreg_data_tmp2, src_ub2 + base);
        LoadAlign(vreg_data_tmp3, src_ub3 + base);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);
        // 4份ub，解bank冲突
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 2 + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 3 + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + base, vreg_data_tmp0, preg_135);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + base, vreg_data_tmp1, preg_135);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + base, vreg_data_tmp2, preg_135);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + base, vreg_data_tmp3, preg_135);
    }
    mem_bar(VST_VLD);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);
    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                   IsSameType<T2, hifloat8_t>::value) &&
                  hasAtten) {
        Arange(vreg_chunk0, (float)maskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            uint32_t base = (uint32_t)iter_m * 256;
            LoadAlign(src0, src_ub0 + base);
            LoadAlign(src1, src_ub1 + base);
            LoadAlign(src2, src_ub2 + base);
            LoadAlign(src3, src_ub3 + base);
            Compare<T, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + base, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + base, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + base, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + base, src3, preg_108);
            Max(max0, src0, src1, preg_108);
            Max(max1, src2, src3, preg_108);
            Max(max0, max0, max1, preg_108);
            Max(vreg_local_max, max0, vreg_local_max, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            uint32_t base = (uint32_t)iter_m * 256;
            LoadAlign(src0, src_ub0 + base);
            LoadAlign(src1, src_ub1 + base);
            LoadAlign(src2, src_ub2 + base);
            LoadAlign(src3, src_ub3 + base);
            Max(max0, src0, src1, preg_108);
            Max(max1, src2, src3, preg_108);
            Max(max0, max0, max1, preg_108);
            Max(vreg_local_max, max0, vreg_local_max, preg_108);
        }
    }

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(vreg_local_max, vreg_local_max, vreg_ln_p_scale, preg_108);

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, vreg_local_max, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    LoadAlign(idx_nd2nz, indexesUb);
    constexpr uint16_t loopNum = 32;

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, vreg_local_max, preg_134);

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

        // -----------------------------------------------------------------------------//
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, vreg_local_max, preg_134);

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);

        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum_2, vreg_x_sum_2, vreg_x_sum_3, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum_6, vreg_x_sum_6, vreg_x_sum_7, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_2, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_6, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_4, preg_134);

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum_0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnUpdatePerTokenHeadVF(
    __ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB, __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
    __ubuf__ float* new_global_max, __ubuf__ uint8_t* indexesUb, __ubuf__ float* qScaleUb, __ubuf__ float* kScaleUb,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK, const T minValue,
    float keepProb, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, int64_t maskLine = 0)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, vreg_local_max;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;

    Duplicate(vreg_min, minValue);
    Duplicate(vreg_local_max, minValue);

    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_108);
    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                   IsSameType<T2, hifloat8_t>::value) &&
                  hasAtten) {
        Arange(vreg_chunk0, (float)maskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            uint32_t base = (uint32_t)iter_m * 256;
            LoadAlign(src0, src_ub0 + base);
            LoadAlign(src1, src_ub1 + base);
            LoadAlign(src2, src_ub2 + base);
            LoadAlign(src3, src_ub3 + base);

            Mul(src0, src0, vreg_qscale_vec, preg_135);
            Mul(src1, src1, vreg_qscale_vec, preg_135);
            Mul(src2, src2, vreg_qscale_vec, preg_135);
            Mul(src3, src3, vreg_qscale_vec, preg_135);
            // 4份ub，解bank冲突
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
            Mul(src0, src0, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 + iter_m * 4 + 1);
            Mul(src1, src1, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 2 + iter_m * 4 + 2);
            Mul(src2, src2, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 3 + iter_m * 4 + 3);
            Mul(src3, src3, vreg_kscale_val, preg_135);

            Compare<T, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + base, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + base, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + base, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + base, src3, preg_108);
            Max(max0, src0, src1, preg_108);
            Max(max1, src2, src3, preg_108);
            Max(max0, max0, max1, preg_108);
            Max(vreg_local_max, max0, vreg_local_max, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            uint32_t base = (uint32_t)iter_m * 256;
            LoadAlign(src0, src_ub0 + base);
            LoadAlign(src1, src_ub1 + base);
            LoadAlign(src2, src_ub2 + base);
            LoadAlign(src3, src_ub3 + base);

            Mul(src0, src0, vreg_qscale_vec, preg_135);
            Mul(src1, src1, vreg_qscale_vec, preg_135);
            Mul(src2, src2, vreg_qscale_vec, preg_135);
            Mul(src3, src3, vreg_qscale_vec, preg_135);
            // 4份ub，解bank冲突
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
            Mul(src0, src0, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 + iter_m * 4 + 1);
            Mul(src1, src1, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 2 + iter_m * 4 + 2);
            Mul(src2, src2, vreg_kscale_val, preg_135);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 3 + iter_m * 4 + 3);
            Mul(src3, src3, vreg_kscale_val, preg_135);

            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + base, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + base, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + base, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + base, src3, preg_108);
            Max(max0, src0, src1, preg_108);
            Max(max1, src2, src3, preg_108);
            Max(max0, max0, max1, preg_108);
            Max(vreg_local_max, max0, vreg_local_max, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(vreg_local_max, vreg_local_max, vreg_ln_p_scale, preg_108);
    Max(vreg_local_max, vreg_local_max, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, vreg_local_max, preg_134);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, vreg_local_max, preg_108);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    LoadAlign(idx_nd2nz, indexesUb);
    constexpr uint16_t loopNum = 32;

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, vreg_local_max, preg_134);

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

        // -----------------------------------------------------------------------------//
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, vreg_local_max, preg_134);

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);

        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum_2, vreg_x_sum_2, vreg_x_sum_3, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum_6, vreg_x_sum_6, vreg_x_sum_7, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_2, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_6, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_4, preg_134);

    RegTensor<float> vreg_l0;
    LoadAlign(vreg_l0, new_global_sum);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Add(vreg_l0, vreg_l0, vreg_x_sum_0, preg_134);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnUpdatePerTokenHeadTailVF(
    __ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB, __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
    __ubuf__ float* new_global_max, __ubuf__ uint8_t* indexesUb, __ubuf__ float* qScaleUb, __ubuf__ float* kScaleUb,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK, const T minValue,
    float keepProb, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, int64_t maskLine = 0)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, vreg_local_max;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;
    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;

    __ubuf__ T2* x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;

    Duplicate(vreg_min, minValue);
    Duplicate(vreg_local_max, minValue);

    // qscale和kscale应用循环
    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_108);
    for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
        uint32_t base = (uint32_t)iter_m * 256;
        LoadAlign(vreg_data_tmp0, src_ub0 + base);
        LoadAlign(vreg_data_tmp1, src_ub1 + base);
        LoadAlign(vreg_data_tmp2, src_ub2 + base);
        LoadAlign(vreg_data_tmp3, src_ub3 + base);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);
        // 4份ub，解bank冲突
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 2 + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + NUM_264 * 3 + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + base, vreg_data_tmp0, preg_135);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + base, vreg_data_tmp1, preg_135);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + base, vreg_data_tmp2, preg_135);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + base, vreg_data_tmp3, preg_135);
    }
    mem_bar(VST_VLD);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);
    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                   IsSameType<T2, hifloat8_t>::value) &&
                  hasAtten) {
        Arange(vreg_chunk0, (float)maskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            uint32_t base = (uint32_t)iter_m * 256;
            LoadAlign(src0, src_ub0 + base);
            LoadAlign(src1, src_ub1 + base);
            LoadAlign(src2, src_ub2 + base);
            LoadAlign(src3, src_ub3 + base);
            Compare<T, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + base, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + base, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + base, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + base, src3, preg_108);
            Max(max0, src0, src1, preg_108);
            Max(max1, src2, src3, preg_108);
            Max(max0, max0, max1, preg_108);
            Max(vreg_local_max, max0, vreg_local_max, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            uint32_t base = (uint32_t)iter_m * 256;
            LoadAlign(src0, src_ub0 + base);
            LoadAlign(src1, src_ub1 + base);
            LoadAlign(src2, src_ub2 + base);
            LoadAlign(src3, src_ub3 + base);
            Max(max0, src0, src1, preg_108);
            Max(max1, src2, src3, preg_108);
            Max(max0, max0, max1, preg_108);
            Max(vreg_local_max, max0, vreg_local_max, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(vreg_local_max, vreg_local_max, vreg_ln_p_scale, preg_108);
    Max(vreg_local_max, vreg_local_max, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, vreg_local_max, preg_134);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, vreg_local_max, preg_108);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    LoadAlign(idx_nd2nz, indexesUb);
    constexpr uint16_t loopNum = 32;

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, vreg_local_max, preg_134);

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

        // -----------------------------------------------------------------------------//
        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, vreg_local_max, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, vreg_local_max, preg_134);

        Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);

        Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);

        Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);

        Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);

        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum_2, vreg_x_sum_2, vreg_x_sum_3, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum_6, vreg_x_sum_6, vreg_x_sum_7, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_2, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_6, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_4, preg_134);

    RegTensor<float> vreg_l0;
    LoadAlign(vreg_l0, new_global_sum);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Add(vreg_l0, vreg_l0, vreg_x_sum_0, preg_134);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool isTail = false>
__aicore__ inline void ProcessVec1DnNoUpdatePerTokenHead(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t>& vselrIndexesBuf,
    const LocalTensor<T>& qScaleUbTensor, const LocalTensor<T>& kScaleUbTensor, const uint32_t m, const uint32_t n,
    const uint32_t originN, const T scale, float deScaleQK, float pScale, const T minValue, float keepProb,
    int64_t maskLine)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = nullptr;
    __ubuf__ float* qScaleUb = (__ubuf__ float*)qScaleUbTensor.GetPhyAddr();
    __ubuf__ float* kScaleUb = (__ubuf__ float*)kScaleUbTensor.GetPhyAddr();
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    if constexpr (isTail) {
        ProcessVec1DnNoUpdatePerTokenHeadTailVF<T, T2, hasAtten, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, indexesUb, qScaleUb, kScaleUb, m, n,
            originN, scale, deScaleQK, minValue, keepProb, dScale, blockStride, repeatStride, pScale, maskLine);
    } else {
        ProcessVec1DnNoUpdatePerTokenHeadVF<T, T2, hasAtten, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, indexesUb, qScaleUb, kScaleUb, m, n,
            originN, scale, deScaleQK, minValue, keepProb, dScale, blockStride, repeatStride, pScale, maskLine);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool isTail = false>
__aicore__ inline void ProcessVec1DnUpdatePerTokenHead(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t>& vselrIndexesBuf,
    const LocalTensor<T>& qScaleUbTensor, const LocalTensor<T>& kScaleUbTensor, const uint32_t m, const uint32_t n,
    const uint32_t originN, const T scale, float deScaleQK, float pScale, const T minValue, float keepProb,
    int64_t maskLine)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = nullptr;
    __ubuf__ float* qScaleUb = (__ubuf__ float*)qScaleUbTensor.GetPhyAddr();
    __ubuf__ float* kScaleUb = (__ubuf__ float*)kScaleUbTensor.GetPhyAddr();
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    if constexpr (isTail) {
        ProcessVec1DnUpdatePerTokenHeadTailVF<T, T2, hasAtten, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, indexesUb, qScaleUb, kScaleUb, m, n,
            originN, scale, deScaleQK, minValue, keepProb, dScale, blockStride, repeatStride, pScale, maskLine);
    } else {
        ProcessVec1DnUpdatePerTokenHeadVF<T, T2, hasAtten, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, indexesUb, qScaleUb, kScaleUb, m, n,
            originN, scale, deScaleQK, minValue, keepProb, dScale, blockStride, repeatStride, pScale, maskLine);
    }
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256,
          bool isTail = false>
__aicore__ inline void ProcessVec1VfDnPerTokenHead(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                                   const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                                   const LocalTensor<T>& expMaxTensor, TBuf<>* vselrIndexesBuf,
                                                   const LocalTensor<T>& qScaleUbTensor,
                                                   const LocalTensor<T>& kScaleUbTensor, const uint32_t m,
                                                   const uint32_t n, const uint32_t originN, const T scale,
                                                   float deScaleQK, float pScale, const T minValue, float keepProb,
                                                   int64_t maskLine)
{
    if constexpr (!isUpdate) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnNoUpdatePerTokenHead<T, T2, hasAtten, ubN, isTail>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, qScaleUbTensor, kScaleUbTensor,
            m, n, originN, scale, deScaleQK, pScale, minValue, keepProb, maskLine);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                      IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnUpdatePerTokenHead<T, T2, hasAtten, ubN, isTail>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, qScaleUbTensor, kScaleUbTensor,
            m, n, originN, scale, deScaleQK, pScale, minValue, keepProb, maskLine);
    }
}

template <typename T>
__simd_vf__ inline void BroadCastMaxSumVF(__ubuf__ float* out_ub, __ubuf__ float* ori_ub, const uint16_t loopM)
{
    RegTensor<float> broadcast_reg;
    MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
    for (uint16_t i = 0; i < loopM; ++i) {
        LoadAlign<T, Reg::LoadDist::DIST_E2B_B32>(broadcast_reg, ori_ub + i * 8);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)out_ub + i * 64, broadcast_reg, preg_all);
    }
}

template <typename T>
__aicore__ inline void BroadcastMaxSum(const LocalTensor<T>& outTensor, const LocalTensor<T>& oriTensor,
                                       uint32_t vecS1RealSize)
{
    __ubuf__ float* out_ub = (__ubuf__ T*)outTensor.GetPhyAddr();
    __ubuf__ float* ori_ub = (__ubuf__ T*)oriTensor.GetPhyAddr();

    // Align8, broadcast one element to 8 elements, one register can store 64 elements,
    // so we can handle 64 / 8 = 8 elements per loop.
    uint16_t loopM = (vecS1RealSize + 7) >> 3;
    BroadCastMaxSumVF<T>(out_ub, ori_ub, loopM);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__simd_vf__ inline void ProcessVec1DnNoUpdateHif8VF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                                    __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                                    __ubuf__ float* new_global_max, __ubuf__ uint8_t* indexesUb,
                                                    const uint32_t m, const uint32_t n, const uint32_t originN,
                                                    const T scale, float deScaleQK, float pScale, const T minValue,
                                                    float keepProb, const float dScale, const uint32_t blockStride,
                                                    const uint32_t repeatStride, int64_t maskLine)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;

    __ubuf__ T2* x_exp_1;
    x_exp_1 = x_exp + 32;

    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        Arange(vreg_chunk0, (float)maskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Compare<T, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);

    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    LoadAlign(idx_nd2nz, indexesUb);
    uint16_t loopNum = ubN / 8; // ubN：s2BaseSize = 256

    if constexpr (hasAtten) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);

        Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);

        Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

        Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

        Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
        Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
        Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
        Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134); // vreg_x_exp_i softmax分子

        Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);

        Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);

        Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);

        Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);

        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum_2, vreg_x_sum_2, vreg_x_sum_3, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum_6, vreg_x_sum_6, vreg_x_sum_7, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_2, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_6, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_4, preg_134);

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_x_sum_0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__simd_vf__ inline void ProcessVec1DnUpdateHif8VF(__ubuf__ T2* x_exp, __ubuf__ float* input_x_local_UB,
                                                  __ubuf__ float* exp_max_fp32, __ubuf__ float* new_global_sum,
                                                  __ubuf__ float* new_global_max, __ubuf__ uint8_t* indexesUb,
                                                  const uint32_t m, const uint32_t n, const uint32_t originN,
                                                  const T scale, float deScaleQK, float pScale, const T minValue,
                                                  float keepProb, const float dScale, const uint32_t blockStride,
                                                  const uint32_t repeatStride, int64_t maskLine)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_chunk0;
    RegTensor<float> vreg_thresh;

    __ubuf__ T2* x_exp_1;
    x_exp_1 = x_exp + 32;

    __ubuf__ float* src_ub0 = input_x_local_UB;
    __ubuf__ float* src_ub1 = src_ub0 + m;
    __ubuf__ float* src_ub2 = src_ub0 + m * 2;
    __ubuf__ float* src_ub3 = src_ub0 + m * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr (hasAtten) {
        Arange(vreg_chunk0, (float)maskLine);
        Duplicate(vreg_thresh, 0.0f);
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Compare<T, CMPMODE::GE>(preg_compare0, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare1, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare2, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Compare<T, CMPMODE::GE>(preg_compare3, vreg_chunk0, vreg_thresh, preg_135);
            Adds(vreg_thresh, vreg_thresh, 1.0f, preg_135);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);
    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)new_global_max, max0, preg_108);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>((__ubuf__ T*&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
    LoadAlign(idx_nd2nz, indexesUb);
    loopNum = ubN / 8;

    if constexpr (hasAtten) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
        LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
        LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
        LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
        LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
        Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);

        Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
        Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);

        Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
        Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

        Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
           (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
        Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);

        Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

        Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
        Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
        Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
        Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

        FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
        FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
        FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
        FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134); // vreg_x_exp_i softmax分子

        Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
        Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);

        Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
        Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);

        Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
        Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);

        Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
        Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
           (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
        Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);

        Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
        StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T2*&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
    }
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_1, preg_134);
    Add(vreg_x_sum_2, vreg_x_sum_2, vreg_x_sum_3, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_5, preg_134);
    Add(vreg_x_sum_6, vreg_x_sum_6, vreg_x_sum_7, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_2, preg_134);
    Add(vreg_x_sum_4, vreg_x_sum_4, vreg_x_sum_6, preg_134);
    Add(vreg_x_sum_0, vreg_x_sum_0, vreg_x_sum_4, preg_134);

    RegTensor<float> vreg_l0;
    LoadAlign(vreg_l0, new_global_sum);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Add(vreg_l0, vreg_l0, vreg_x_sum_0, preg_134);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__aicore__ inline void ProcessVec1DnNoUpdateHif8(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                                 const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                                 const LocalTensor<T>& expMaxTensor,
                                                 const LocalTensor<uint8_t>& vselrIndexesBuf, const uint32_t m,
                                                 const uint32_t n, const uint32_t originN, const T scale,
                                                 float deScaleQK, float pScale, const T minValue, float keepProb,
                                                 int64_t maskLine)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    float dScale = scale * deScaleQK;
    uint32_t blockStride = ubN >> 2 | 0x1;
    uint32_t repeatStride = 2;

    ProcessVec1DnNoUpdateHif8VF<T, T2, hasAtten, ubN, hasSink>(
        x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, indexesUb, m, n, originN, scale,
        deScaleQK, pScale, minValue, keepProb, dScale, blockStride, repeatStride, maskLine);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__aicore__ inline void ProcessVec1DnUpdateHif8(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                               const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                               const LocalTensor<T>& expMaxTensor,
                                               const LocalTensor<uint8_t>& vselrIndexesBuf, const uint32_t m,
                                               const uint32_t n, const uint32_t originN, const T scale, float deScaleQK,
                                               float pScale, const T minValue, float keepProb, int64_t maskLine)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    float dScale = scale * deScaleQK;
    uint32_t blockStride = ubN >> 2 | 0x1;
    uint32_t repeatStride = 2;

    ProcessVec1DnUpdateHif8VF<T, T2, hasAtten, ubN, hasSink>(
        x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max, indexesUb, m, n, originN, scale,
        deScaleQK, pScale, minValue, keepProb, dScale, blockStride, repeatStride, maskLine);
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256,
          bool hasSink = false>
__aicore__ inline void ProcessVec1VfDnHif8(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                           const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                           const LocalTensor<T>& expMaxTensor, TBuf<>* vselrIndexesBuf,
                                           const uint32_t m, const uint32_t n, const uint32_t originN, const T scale,
                                           float deScaleQK, float pScale, const T minValue, float keepProb,
                                           int64_t maskLine)
{
    LocalTensor<uint8_t> indexesTensor;
    indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
    if constexpr (!isUpdate) {
        ProcessVec1DnNoUpdateHif8<T, T2, hasAtten, ubN, hasSink>(dstTensor, expSumTensor, maxTensor, srcTensor,
                                                                 expMaxTensor, indexesTensor, m, n, originN, scale,
                                                                 deScaleQK, pScale, minValue, keepProb, maskLine);
    } else {
        ProcessVec1DnUpdateHif8<T, T2, hasAtten, ubN, hasSink>(dstTensor, expSumTensor, maxTensor, srcTensor,
                                                               expMaxTensor, indexesTensor, m, n, originN, scale,
                                                               deScaleQK, pScale, minValue, keepProb, maskLine);
    }
}

template <typename T, typename T2, uint16_t s2BaseSize = 128, bool isUpdate = false,
          bool hasAtten = 0, bool hasDrop = false, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
          bool hasSink = false>
__simd_vf__ inline void ProcessVec1Dn64VF(__ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB,
    __ubuf__ float *exp_max, __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max,
    __ubuf__ uint8_t *maskUb, __ubuf__ uint8_t *dropUb, const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, const T minValue, float keepProb, const uint32_t blockStride, const uint32_t repeatStride, float slopes,
    float posShift, const float sinkValue = 0.0f)
{
    float divValue = 1.0f / keepProb;
    float posShift_0 = posShift;
    float posShift_1 = posShift - 1;
    float posShift_2 = posShift - 2;
    float posShift_3 = posShift - 3;

    __ubuf__ T *src_ub_0 = input_x_local_UB;
    __ubuf__ T *src_ub_1 = src_ub_0 + m;
    __ubuf__ T *src_ub_2 = src_ub_0 + m * 2;
    __ubuf__ T *src_ub_3 = src_ub_0 + m * 3;
    __ubuf__ T *src_ub_4 = src_ub_0 + m * 4;
    __ubuf__ T *src_ub_5 = src_ub_0 + m * 5;
    __ubuf__ T *src_ub_6 = src_ub_0 + m * 6;
    __ubuf__ T *src_ub_7 = src_ub_0 + m * 7;

    __ubuf__ T2 *x_exp_0 = x_exp;
    __ubuf__ T2 *x_exp_1 = x_exp_0 + (n * 4);

    __ubuf__ uint8_t *mask_ub_0 = maskUb;
    __ubuf__ uint8_t *mask_ub_1 = maskUb + m;
    __ubuf__ uint8_t *mask_ub_2 = maskUb + m * 2;
    __ubuf__ uint8_t *mask_ub_3 = maskUb + m * 3;

    __ubuf__ uint8_t *drop_ub_0 = dropUb;
    __ubuf__ uint8_t *drop_ub_1 = dropUb + (n * m / 2) / 4;
    __ubuf__ uint8_t *drop_ub_2 = dropUb + (n * m / 4) / 4;
    __ubuf__ uint8_t *drop_ub_3 = dropUb + (n * m * 3 / 4) / 4;

    MaskReg preg_input = CreateMask<T, MaskPattern::ALL>();
    MaskReg preg_out = CreateMask<T2, MaskPattern::ALL>();

    RegTensor<T> vreg_src_0, vreg_src_1, vreg_src_2, vreg_src_3;
    RegTensor<T> vreg_src_4, vreg_src_5, vreg_src_6, vreg_src_7;
    RegTensor<T> vreg_max_0;
    RegTensor<T> vreg_sum_0, vreg_sum_1, vreg_sum_2, vreg_sum_3;
    RegTensor<T> vreg_exp_0, vreg_exp_1, vreg_exp_2, vreg_exp_3;
    RegTensor<T2> vreg_exp_cast_0, vreg_exp_cast_1, vreg_exp_cast_2, vreg_exp_cast_3;
    RegTensor<T2> vreg_exp_pack_0, vreg_exp_pack_1;
    RegTensor<T> vreg_max_last;
    RegTensor<T> vreg_sum_last;
    RegTensor<T> vreg_exp_max;
    RegTensor<T> vreg_min;
    RegTensor<T> vreg_sink_input;

    MaskReg preg_mask_0, preg_mask_1, preg_mask_2, preg_mask_3;

    RegTensor<T> vreg_alibi_0, vreg_alibi_1, vreg_alibi_2, vreg_alibi_3;
    RegTensor<T> vreg_pse_0, vreg_pse_1, vreg_pse_2, vreg_pse_3;

    MaskReg preg_drop_0, preg_drop_1, preg_drop_2, preg_drop_3;

    Duplicate(vreg_max_0, minValue);
    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    for (uint16_t i = originN; i < uint16_t(n); ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_input);
    }
    mem_bar(VST_VLD);

    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE || pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi_0, posShift_0);
        Arange(vreg_alibi_1, posShift_1);
        Arange(vreg_alibi_2, posShift_2);
        Arange(vreg_alibi_3, posShift_3);
    }

    if constexpr (hasAtten == true || pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        for (uint16_t iter_m = 0; iter_m < uint16_t(n / 4); ++iter_m) {
            LoadAlign(vreg_src_0, src_ub_0 + iter_m * m * 4);
            LoadAlign(vreg_src_1, src_ub_1 + iter_m * m * 4);
            LoadAlign(vreg_src_2, src_ub_2 + iter_m * m * 4);
            LoadAlign(vreg_src_3, src_ub_3 + iter_m * m * 4);

            Muls(vreg_src_0, vreg_src_0, scale, preg_input);
            Muls(vreg_src_1, vreg_src_1, scale, preg_input);
            Muls(vreg_src_2, vreg_src_2, scale, preg_input);
            Muls(vreg_src_3, vreg_src_3, scale, preg_input);

            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE || pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                Abs(vreg_pse_0, vreg_alibi_0, preg_input);
                Abs(vreg_pse_1, vreg_alibi_1, preg_input);
                Abs(vreg_pse_2, vreg_alibi_2, preg_input);
                Abs(vreg_pse_3, vreg_alibi_3, preg_input);

                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Sqrt(vreg_pse_0, vreg_pse_0, preg_input);
                    Sqrt(vreg_pse_1, vreg_pse_1, preg_input);
                    Sqrt(vreg_pse_2, vreg_pse_2, preg_input);
                    Sqrt(vreg_pse_3, vreg_pse_3, preg_input);
                }

                Axpy(vreg_src_0, vreg_pse_0, slopes, preg_input);
                Axpy(vreg_src_1, vreg_pse_1, slopes, preg_input);
                Axpy(vreg_src_2, vreg_pse_2, slopes, preg_input);
                Axpy(vreg_src_3, vreg_pse_3, slopes, preg_input);

                Adds(vreg_alibi_0, vreg_alibi_0, -4, preg_input);
                Adds(vreg_alibi_1, vreg_alibi_1, -4, preg_input);
                Adds(vreg_alibi_2, vreg_alibi_2, -4, preg_input);
                Adds(vreg_alibi_3, vreg_alibi_3, -4, preg_input);
            }

            if constexpr (hasAtten == true) {
                LoadAlign<uint8_t, Reg::MaskDist::DIST_DS>(
                    preg_mask_0, mask_ub_0 + iter_m * m * 4);
                LoadAlign<uint8_t, Reg::MaskDist::DIST_DS>(
                    preg_mask_1, mask_ub_1 + iter_m * m * 4);
                LoadAlign<uint8_t, Reg::MaskDist::DIST_DS>(
                    preg_mask_2, mask_ub_2 + iter_m * m * 4);
                LoadAlign<uint8_t, Reg::MaskDist::DIST_DS>(
                    preg_mask_3, mask_ub_3 + iter_m * m * 4);

                Select(vreg_src_0, vreg_src_0, vreg_min, preg_mask_0);
                Select(vreg_src_1, vreg_src_1, vreg_min, preg_mask_1);
                Select(vreg_src_2, vreg_src_2, vreg_min, preg_mask_2);
                Select(vreg_src_3, vreg_src_3, vreg_min, preg_mask_3);
            }

            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
                src_ub_0 + iter_m * m * 4, vreg_src_0, preg_input);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
                src_ub_1 + iter_m * m * 4, vreg_src_1, preg_input);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
                src_ub_2 + iter_m * m * 4, vreg_src_2, preg_input);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
                src_ub_3 + iter_m * m * 4, vreg_src_3, preg_input);

            Max(vreg_src_0, vreg_src_0, vreg_src_1, preg_input);
            Max(vreg_src_2, vreg_src_2, vreg_src_3, preg_input);
            Max(vreg_src_0, vreg_src_0, vreg_src_2, preg_input);
            Max(vreg_max_0, vreg_max_0, vreg_src_0, preg_input);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(n / 8); ++iter_m) {
            LoadAlign(vreg_src_0, src_ub_0 + iter_m * m * 8);
            LoadAlign(vreg_src_1, src_ub_1 + iter_m * m * 8);
            LoadAlign(vreg_src_2, src_ub_2 + iter_m * m * 8);
            LoadAlign(vreg_src_3, src_ub_3 + iter_m * m * 8);
            LoadAlign(vreg_src_4, src_ub_4 + iter_m * m * 8);
            LoadAlign(vreg_src_5, src_ub_5 + iter_m * m * 8);
            LoadAlign(vreg_src_6, src_ub_6 + iter_m * m * 8);
            LoadAlign(vreg_src_7, src_ub_7 + iter_m * m * 8);

            Max(vreg_src_0, vreg_src_0, vreg_src_1, preg_input);
            Max(vreg_src_2, vreg_src_2, vreg_src_3, preg_input);
            Max(vreg_src_4, vreg_src_4, vreg_src_5, preg_input);
            Max(vreg_src_6, vreg_src_6, vreg_src_7, preg_input);
            Max(vreg_src_0, vreg_src_0, vreg_src_2, preg_input);
            Max(vreg_src_4, vreg_src_4, vreg_src_6, preg_input);
            Max(vreg_src_0, vreg_src_0, vreg_src_4, preg_input);
            Max(vreg_max_0, vreg_max_0, vreg_src_0, preg_input);
        }
    }

    if constexpr (!hasAtten && pseMode == PseTypeEnum::PSE_NONE_TYPE) {
        Muls(vreg_max_0, vreg_max_0, scale, preg_input);
    }

    if constexpr (isUpdate) {
        LoadAlign(vreg_max_last, new_global_max);
        Max(vreg_max_0, vreg_max_0, vreg_max_last, preg_input);
    }

    if constexpr (hasSink) {
        Max(vreg_max_0, vreg_max_0, vreg_sink_input, preg_input);
    }

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)new_global_max, vreg_max_0, preg_input);
    
    if constexpr (isUpdate) {
        FusedExpSub(vreg_exp_max, vreg_max_last, vreg_max_0, preg_input);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)exp_max, vreg_exp_max, preg_input);
    }

    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_sum_0, 0, preg_input);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_sum_1, 0, preg_input);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_sum_2, 0, preg_input);
    Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vreg_sum_3, 0, preg_input);
    if constexpr (hasAtten == true || pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        mem_bar(VST_VLD);
    }
    for (uint16_t i = 0; i < uint16_t(n / 4); ++i) {
        LoadAlign(vreg_src_0, input_x_local_UB + i * m);
        LoadAlign(vreg_src_1, input_x_local_UB + n * m / 2 + i * m);
        LoadAlign(vreg_src_2, input_x_local_UB + n * m / 4 + i * m);
        LoadAlign(vreg_src_3, input_x_local_UB + n * m / 2 + n * m / 4 + i * m);

        if constexpr (!hasAtten && pseMode == PseTypeEnum::PSE_NONE_TYPE) {
            Muls(vreg_src_0, vreg_src_0, scale, preg_input);
            Muls(vreg_src_1, vreg_src_1, scale, preg_input);
            Muls(vreg_src_2, vreg_src_2, scale, preg_input);
            Muls(vreg_src_3, vreg_src_3, scale, preg_input);
        }

        FusedExpSub(vreg_exp_0, vreg_src_0, vreg_max_0, preg_input);
        FusedExpSub(vreg_exp_1, vreg_src_1, vreg_max_0, preg_input);
        FusedExpSub(vreg_exp_2, vreg_src_2, vreg_max_0, preg_input);
        FusedExpSub(vreg_exp_3, vreg_src_3, vreg_max_0, preg_input);

        Add(vreg_sum_0, vreg_exp_0, vreg_sum_0, preg_input);
        Add(vreg_sum_1, vreg_exp_1, vreg_sum_1, preg_input);
        Add(vreg_sum_2, vreg_exp_2, vreg_sum_2, preg_input);
        Add(vreg_sum_3, vreg_exp_3, vreg_sum_3, preg_input);

        if constexpr (hasDrop) {
            LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::MaskDist::DIST_US>(preg_drop_0, drop_ub_0, 128 >> 3);
            LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::MaskDist::DIST_US>(preg_drop_1, drop_ub_1, 128 >> 3);
            LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::MaskDist::DIST_US>(preg_drop_2, drop_ub_2, 128 >> 3);
            LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::MaskDist::DIST_US>(preg_drop_3, drop_ub_3, 128 >> 3);
            if constexpr (AscendC::IsSameType<T2, half>::value) {
                vmulscvt(vreg_exp_cast_0, vreg_exp_0, divValue, preg_drop_0, PART_EVEN);
                vmulscvt(vreg_exp_cast_1, vreg_exp_1, divValue, preg_drop_1, PART_EVEN);
                vmulscvt(vreg_exp_cast_2, vreg_exp_2, divValue, preg_drop_2, PART_EVEN);
                vmulscvt(vreg_exp_cast_3, vreg_exp_3, divValue, preg_drop_3, PART_EVEN);
            } else {
                Muls(vreg_exp_0, vreg_exp_0, divValue, preg_drop_0);
                Muls(vreg_exp_1, vreg_exp_1, divValue, preg_drop_1);
                Muls(vreg_exp_2, vreg_exp_2, divValue, preg_drop_2);
                Muls(vreg_exp_3, vreg_exp_3, divValue, preg_drop_3);
            }
        }

        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value || AscendC::IsSameType<T2, half>::value) {
            if constexpr (!(AscendC::IsSameType<T2, half>::value && hasDrop)) {
                Cast<T2, T, castTraitZero>(vreg_exp_cast_0, vreg_exp_0, preg_input);
                Cast<T2, T, castTraitZero>(vreg_exp_cast_1, vreg_exp_1, preg_input);
                Cast<T2, T, castTraitZero>(vreg_exp_cast_2, vreg_exp_2, preg_input);
                Cast<T2, T, castTraitZero>(vreg_exp_cast_3, vreg_exp_3, preg_input);
            }
            DeInterleaveV2<T2, Reg::HighLowPart::LOWEST>(vreg_exp_pack_0, vreg_exp_cast_0, vreg_exp_cast_1);
            DeInterleaveV2<T2, Reg::HighLowPart::LOWEST>(vreg_exp_pack_1, vreg_exp_cast_2, vreg_exp_cast_3);

            /* vreg_exp_pack_0会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_exp_pack_1的 */
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_0), vreg_exp_pack_0, blockStride, repeatStride, preg_out);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_exp_pack_1, blockStride, repeatStride, preg_out);
        }
    }
    Add(vreg_sum_0, vreg_sum_1, vreg_sum_0, preg_input);
    Add(vreg_sum_2, vreg_sum_3, vreg_sum_2, preg_input);
    Add(vreg_sum_0, vreg_sum_0, vreg_sum_2, preg_input);

    if constexpr (isUpdate) {
        LoadAlign(vreg_sum_last, new_global_sum);
        Mul(vreg_sum_last, vreg_exp_max, vreg_sum_last, preg_input);
        Add(vreg_sum_0, vreg_sum_last, vreg_sum_0, preg_input);
    }

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)new_global_sum, vreg_sum_0, preg_input);
}

template <typename T, typename T2, uint16_t s2BaseSize = 128, bool isUpdate = false,
          bool hasAtten = 0, bool hasDrop = false, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
          bool hasSink = false>
__simd_vf__ inline void ProcessVec1Dn128VF(__ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB,
    __ubuf__ float *exp_max, __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max,
    __ubuf__ uint8_t *maskUb, __ubuf__ uint8_t *dropUb, __ubuf__ uint8_t * indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, const T minValue, float keepProb, const uint32_t blockStride, const uint32_t repeatStride, float slopes, float posShift,
    const float sinkValue = 0.0f)
{
    float divValue = 1.0f / keepProb;
    float posShift_0 = posShift;
    float posShift_1 = posShift_0 + 64;
    float posShift_2 = posShift - 1;
    float posShift_3 = posShift_2 + 64;

    __ubuf__ T *src_ub_0 = input_x_local_UB;
    __ubuf__ T *src_ub_1 = src_ub_0 + (m >> 1);
    __ubuf__ T *src_ub_2 = src_ub_0 + (m >> 1) * 2;
    __ubuf__ T *src_ub_3 = src_ub_0 + (m >> 1) * 3;
    __ubuf__ T *src_ub_4 = src_ub_0 + (m >> 1) * 4;
    __ubuf__ T *src_ub_5 = src_ub_0 + (m >> 1) * 5;
    __ubuf__ T *src_ub_6 = src_ub_0 + (m >> 1) * 6;
    __ubuf__ T *src_ub_7 = src_ub_0 + (m >> 1) * 7;

    __ubuf__ uint8_t *mask_ub_0 = maskUb;
    __ubuf__ uint8_t *mask_ub_1 = maskUb + (m >> 1);
    __ubuf__ uint8_t *mask_ub_2 = maskUb + (m >> 1) * 2;
    __ubuf__ uint8_t *mask_ub_3 = maskUb + (m >> 1) * 3;

    __ubuf__ uint8_t *drop_ub_0 = dropUb;
    __ubuf__ uint8_t *drop_ub_1 = dropUb + (n * m / 2) / 4;

    MaskReg preg_input = CreateMask<T, MaskPattern::ALL>();
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
    RegTensor<T> vreg_sink_input;

    MaskReg preg_mask_0, preg_mask_1, preg_mask_2, preg_mask_3;

    RegTensor<T> vreg_alibi_0, vreg_alibi_1, vreg_alibi_2, vreg_alibi_3;
    RegTensor<T> vreg_pse_0, vreg_pse_1, vreg_pse_2, vreg_pse_3;

    MaskReg preg_drop_0, preg_drop_1, preg_drop_2, preg_drop_3;

    Duplicate(vreg_max_0, minValue);
    Duplicate(vreg_max_1, minValue);
    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    for (uint16_t i = originN; i < uint16_t(n); ++i) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
            input_x_local_UB + i * m, vreg_min, preg_input);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(
            input_x_local_UB + i * m + 64, vreg_min, preg_input);
    }
    mem_bar(VST_VLD);

    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE || pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi_0, posShift_0);
        Arange(vreg_alibi_1, posShift_1);
        Arange(vreg_alibi_2, posShift_2);
        Arange(vreg_alibi_3, posShift_3);
    }

    if constexpr (hasAtten == true || pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        for (uint16_t iter_m = 0; iter_m < uint16_t(n / 2); ++iter_m) {
            LoadAlign(vreg_src_0, src_ub_0 + iter_m * m * 2);
            LoadAlign(vreg_src_1, src_ub_1 + iter_m * m * 2);
            LoadAlign(vreg_src_2, src_ub_2 + iter_m * m * 2);
            LoadAlign(vreg_src_3, src_ub_3 + iter_m * m * 2);
            
            Muls(vreg_src_0, vreg_src_0, scale, preg_input);
            Muls(vreg_src_1, vreg_src_1, scale, preg_input);
            Muls(vreg_src_2, vreg_src_2, scale, preg_input);
            Muls(vreg_src_3, vreg_src_3, scale, preg_input);

            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE || pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                Abs(vreg_pse_0, vreg_alibi_0, preg_input);
                Abs(vreg_pse_1, vreg_alibi_1, preg_input);
                Abs(vreg_pse_2, vreg_alibi_2, preg_input);
                Abs(vreg_pse_3, vreg_alibi_3, preg_input);

                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Sqrt(vreg_pse_0, vreg_pse_0, preg_input);
                    Sqrt(vreg_pse_1, vreg_pse_1, preg_input);
                    Sqrt(vreg_pse_2, vreg_pse_2, preg_input);
                    Sqrt(vreg_pse_3, vreg_pse_3, preg_input);
                }

                Axpy(vreg_src_0, vreg_pse_0, slopes, preg_input);
                Axpy(vreg_src_1, vreg_pse_1, slopes, preg_input);
                Axpy(vreg_src_2, vreg_pse_2, slopes, preg_input);
                Axpy(vreg_src_3, vreg_pse_3, slopes, preg_input);

                Adds(vreg_alibi_0, vreg_alibi_0, -2, preg_input);
                Adds(vreg_alibi_1, vreg_alibi_1, -2, preg_input);
                Adds(vreg_alibi_2, vreg_alibi_2, -2, preg_input);
                Adds(vreg_alibi_3, vreg_alibi_3, -2, preg_input);
            }

            if constexpr (hasAtten) {
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
            }

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
        for (uint16_t iter_m = 0; iter_m < uint16_t(n / 4); ++iter_m) {
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
    if constexpr (!hasAtten && pseMode == PseTypeEnum::PSE_NONE_TYPE) {
        Muls(vreg_max_0, vreg_max_0, scale, preg_input);
        Muls(vreg_max_1, vreg_max_1, scale, preg_input);
    }

    if constexpr (isUpdate) {
        LoadAlign(vreg_max_last_0, new_global_max);
        LoadAlign(vreg_max_last_1, new_global_max + 64);
        Max(vreg_max_0, vreg_max_0, vreg_max_last_0, preg_input);
        Max(vreg_max_1, vreg_max_1, vreg_max_last_1, preg_input);
    }

    if constexpr (hasSink) {
        Max(vreg_max_0, vreg_max_0, vreg_sink_input, preg_input);
        Max(vreg_max_1, vreg_max_1, vreg_sink_input, preg_input);
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
    if constexpr (hasAtten == true || pseMode != PseTypeEnum::PSE_NONE_TYPE) {
        mem_bar(VST_VLD);
    }
    for (uint16_t i = 0; i < uint16_t(n); ++i) {
        LoadAlign<T, Reg::LoadDist::DIST_DINTLV_B32>(
            vreg_src_0, vreg_src_1, input_x_local_UB + i * m);

        if constexpr (!hasAtten && pseMode == PseTypeEnum::PSE_NONE_TYPE) {
            Muls(vreg_src_0, vreg_src_0, scale, preg_input);
            Muls(vreg_src_1, vreg_src_1, scale, preg_input);
        }

        FusedExpSub(vreg_exp_0, vreg_src_0, vreg_max_pack_0, preg_input);
        FusedExpSub(vreg_exp_1, vreg_src_1, vreg_max_pack_1, preg_input);
        Add(vreg_sum_0, vreg_exp_0, vreg_sum_0, preg_input);
        Add(vreg_sum_1, vreg_exp_1, vreg_sum_1, preg_input);
        if constexpr (hasDrop) {
            LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::MaskDist::DIST_US>(preg_drop_0, drop_ub_0, 128 >> 3);
            LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::MaskDist::DIST_US>(preg_drop_1, drop_ub_1, 128 >> 3);
            MaskDeInterleave<T>(preg_drop_2, preg_drop_3, preg_drop_0, preg_drop_1);
            if constexpr (AscendC::IsSameType<T2, half>::value) {
                vmulscvt(vreg_exp_cast_0, vreg_exp_0, divValue, preg_drop_2, PART_EVEN);
                vmulscvt(vreg_exp_cast_1, vreg_exp_1, divValue, preg_drop_3, PART_ODD);
            } else {
                Muls(vreg_exp_0, vreg_exp_0, divValue, preg_drop_2);
                Muls(vreg_exp_1, vreg_exp_1, divValue, preg_drop_3);
            }
        }

        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value || AscendC::IsSameType<T2, half>::value) {
            if constexpr (!(AscendC::IsSameType<T2, half>::value && hasDrop)) {
                Cast<T2, T, castTraitOne>(vreg_exp_cast_0, vreg_exp_1, preg_input);
                Cast<T2, T, castTraitZeroMerge>(vreg_exp_cast_0, vreg_exp_0, preg_input);
            } else {
                Or((RegTensor<uint16_t>&)vreg_exp_cast_0, (RegTensor<uint16_t>&)vreg_exp_cast_0,
                    (RegTensor<uint16_t>&)vreg_exp_cast_1, preg_out);
            }
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                x_exp, vreg_exp_cast_0, blockStride, repeatStride, preg_out);
        } else if constexpr (IsSameType<T2, float8_e5m2_t>::value ||
                                IsSameType<T2, float8_e4m3_t>::value ||
                                IsSameType<T2, hifloat8_t>::value) {
            uint32_t maskLen = 128;
            MaskReg preg_out_half = UpdateMask<T2>(maskLen);
            RegTensor<T2> vreg_exp_merge_tmp_fp8;
            RegTensor<T2> vreg_exp_merge_fp8;
            RegTensor<uint8_t> vreg_exp_merge_fp8_indexes;
            if constexpr (IsSameType<T2, float8_e5m2_t>::value || IsSameType<T2, float8_e4m3_t>::value) {
                Cast<T2, T, castTraitRintZero>(vreg_exp_cast_0, vreg_exp_0, preg_input);
                Cast<T2, T, castTraitRintTwo>(vreg_exp_cast_1, vreg_exp_1, preg_input);
            } else {
                Cast<T2, T, castTraitZero>(vreg_exp_cast_0, vreg_exp_0, preg_input);
                Cast<T2, T, castTraitTwo>(vreg_exp_cast_1, vreg_exp_1, preg_input);
            }
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_fp8, (RegTensor<uint8_t>&)vreg_exp_cast_0,
                (RegTensor<uint8_t>&)vreg_exp_cast_1, preg_out);
            LoadAlign(vreg_exp_merge_fp8_indexes, indexesUb);
            Gather(vreg_exp_merge_fp8, vreg_exp_merge_tmp_fp8, vreg_exp_merge_fp8_indexes);
            StoreAlign<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                x_exp, vreg_exp_merge_fp8, blockStride, repeatStride, preg_out_half);
        }
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

template <typename T, typename T2, uint16_t s2BaseSize = 128, bool isUpdate = false,
          bool hasAtten = 0, bool hasDrop = false, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
          bool hasSink = false>
__aicore__ inline void ProcessVec1Dn64(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& indexesTensor,
    const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, const T minValue, float keepProb, const float slopes, const float posShift,
    const float sinkValue = 0.0f)
{
    constexpr uint32_t repeatStride = 1;
    constexpr uint32_t blockStride = (s2BaseSize >> 1) | 0x1;

    float divValue = 1.0f / keepProb;

    __ubuf__ T2* x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ T* input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ T* exp_max = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T* new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ T* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint8_t* maskUb = (__ubuf__ uint8_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t* dropUb = (__ubuf__ uint8_t*)dropTensor.GetPhyAddr();

    ProcessVec1Dn64VF<T, T2, s2BaseSize, isUpdate, hasAtten, hasDrop, pseMode, hasSink>(x_exp, input_x_local_UB, exp_max, new_global_sum, new_global_max,
        maskUb, dropUb, m, n, originN, scale, minValue, keepProb, blockStride, repeatStride, slopes, posShift, sinkValue);
}

// T = INPUT TYPE, T2 = OUTPUT TYPE
template <typename T, typename T2, uint16_t s2BaseSize = 128, bool isUpdate = false,
          bool hasAtten = 0, bool hasDrop = false, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
          bool hasSink = false>
__aicore__ inline void ProcessVec1Dn128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& indexesTensor,
    const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, const T minValue, float keepProb, const float slopes, const float posShift,
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
    __ubuf__ uint8_t* dropUb = (__ubuf__ uint8_t*)dropTensor.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();

    ProcessVec1Dn128VF<T, T2, s2BaseSize, isUpdate, hasAtten, hasDrop, pseMode, hasSink>(x_exp, input_x_local_UB, exp_max, new_global_sum, new_global_max,
        maskUb, dropUb, indexesUb, m, n, originN, scale, minValue, keepProb, blockStride, repeatStride, slopes, posShift, sinkValue);
}

template <typename T, typename T2, bool isUpdate = false, uint16_t s1BaseSize = 128, uint16_t s2BaseSize = 128,
          bool hasAtten = false, bool hasDrop = false, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
          bool hasSink = false>
__aicore__ inline void ProcessVec1VfDn(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                       const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& expMaxTensor,
                                       const LocalTensor<uint8_t>& maskTensor,
                                       const LocalTensor<uint8_t>& dropTensor,
                                       const LocalTensor<uint8_t>& indexesTensor,
                                       const uint32_t m, const uint32_t n, const uint32_t originN,
                                       const T scale, const T minValue, float keepProb,
                                       const float slopes, const float posShift,
                                       const float sinkValue = 0.0f)
{
    if constexpr (s1BaseSize == 64) {
        ProcessVec1Dn64<T, T2, s2BaseSize, isUpdate, hasAtten, hasDrop, pseMode, hasSink>(
        dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
        maskTensor, dropTensor, indexesTensor, m, n, originN, scale, minValue, keepProb, slopes, posShift, sinkValue); 
    } else {
        ProcessVec1Dn128<T, T2, s2BaseSize, isUpdate, hasAtten, hasDrop, pseMode, hasSink>(
        dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
        maskTensor, dropTensor, indexesTensor, m, n, originN, scale, minValue, keepProb, slopes, posShift, sinkValue);            
    }
}

} // namespace FaVectorApi
#endif // MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
