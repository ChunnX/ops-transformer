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
using namespace MicroAPI;

#define VMULSCVT false
#define DROPOUT false


template <typename T, typename T2, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnNoUpdate(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, float deScaleQK, const T minValue, float keepProb)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    __VEC_SCOPE__{
        RegTensor<float> vreg_x_sum_even;
        RegTensor<float> vreg_x_sum_odd;
        RegTensor<float> vreg_x_sum_1_even;
        RegTensor<float> vreg_x_sum_1_odd;
        RegTensor<float> vreg_x_sum_2_even;
        RegTensor<float> vreg_x_sum_2_odd;
        RegTensor<float> vreg_x_sum_3_even;
        RegTensor<float> vreg_x_sum_3_odd;
        RegTensor<float> vreg_x_sum0;
        RegTensor<float> vreg_x_sum1;
        RegTensor<float> vreg_x_sum2;
        RegTensor<float> vreg_x_sum3;
        RegTensor<half> vreg_x_exp_even_f16;
        RegTensor<half> vreg_x_exp_odd_f16;
        RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;

        RegTensor<float> vreg_x_exp_even;
        RegTensor<float> vreg_x_exp_odd;
        RegTensor<float> vreg_x_f32_a;
        RegTensor<float> vreg_x_f32_b;
        RegTensor<float> vreg_x_exp_even_1;
        RegTensor<float> vreg_x_exp_odd_1;
        RegTensor<float> vreg_x_exp_even_2;
        RegTensor<float> vreg_x_exp_odd_2;
        RegTensor<float> vreg_x_exp_even_3;
        RegTensor<float> vreg_x_exp_odd_3;
        RegTensor<half> vreg_x_exp_even_f16_1;
        RegTensor<half> vreg_x_exp_odd_f16_1;
        RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
        RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

        RegTensor<float> vreg_x_f32_1_a;
        RegTensor<float> vreg_x_f32_1_b;
        RegTensor<float> vreg_x_f32_2_a;
        RegTensor<float> vreg_x_f32_2_b;
        RegTensor<float> vreg_x_f32_3_a;
        RegTensor<float> vreg_x_f32_3_b;
        RegTensor<half> vreg_x_exp_f16_pack;
        RegTensor<half> vreg_x_exp_f16_1_pack;
        RegTensor<half> vreg_x_exp_f16_packa;
        RegTensor<half> vreg_x_exp_f16_1_packa;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;
        MaskReg preg_100;
        MaskReg preg_101;
        MaskReg preg_134;
        MaskReg preg_135;
        MaskReg preg_136;
        preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
        preg_135 = CreateMask<T, MaskPattern::ALL>();
        uint32_t sreg_92 = (uint32_t)128ULL;
        preg_136 = UpdateMask<uint16_t>(sreg_92);
        MaskReg preg_108;
        RegTensor<float> src_00a, src_01a, src_02a, src_03a;
        RegTensor<float> src_00b, src_01b, src_02b, src_03b;
        RegTensor<float> src_10a, src_11a, src_12a, src_13a;
        RegTensor<float> src_10b, src_11b, src_12b, src_13b;
        RegTensor<float> max_0a, max_1a, max_2a, max_3a;
        RegTensor<float> max_0b, max_1b, max_2b, max_3b;
        RegTensor<float> vreg_min;

        RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
        RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;
 
        __ubuf__ float *src0_ub = (__ubuf__ float*) input_x_local_UB;
        __ubuf__ float *src0_ub_1 = src0_ub + m;
        __ubuf__ T2 *x_exp_1;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
            x_exp_1 = x_exp + 32;
        } else {
            x_exp_1 = x_exp + (ubN * 4);
        }
        __ubuf__ float *src0_ub1 = src0_ub + m * 2;
        __ubuf__ float *src0_ub1_1 = src0_ub + m * 3;
        __ubuf__ float *src0_ub2 = src0_ub + m * 4;
        __ubuf__ float *src0_ub2_1 = src0_ub + m * 5;
        __ubuf__ float *src0_ub3 = src0_ub + m * 6;
        __ubuf__ float *src0_ub3_1 = src0_ub + m * 7;

        Duplicate(max_0a, minValue);
        Duplicate(max_0b, minValue);
        Duplicate(max_1a, minValue);
        Duplicate(max_1b, minValue);
        Duplicate(max_2a, minValue);
        Duplicate(max_2b, minValue);
        Duplicate(max_3a, minValue);
        Duplicate(max_3b, minValue);
        Duplicate(vreg_min, minValue);
        for (uint16_t i = originN; i < ubN; ++i) {
            DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
        }
        mem_bar(VST_VLD);

        preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8); ++iter_m) {
            DataCopy(src_00a, src0_ub + iter_m * m * 8);
            Max(max_0a, max_0a, src_00a, preg_108);
            DataCopy(src_00b, src0_ub_1 + iter_m * m * 8);
            Max(max_0b, max_0b, src_00b, preg_108);
            DataCopy(src_01a, src0_ub1 + iter_m * m * 8);
            Max(max_1a, max_1a, src_01a, preg_108);
            DataCopy(src_01b, src0_ub1_1 + iter_m * m * 8);
            Max(max_1b, max_1b, src_01b, preg_108);
            DataCopy(src_02a, src0_ub2 + iter_m * m * 8);
            Max(max_2a, max_2a, src_02a, preg_108);
            DataCopy(src_02b, src0_ub2_1 + iter_m * m * 8);
            Max(max_2b, max_2b, src_02b, preg_108);
            DataCopy(src_03a, src0_ub3 + iter_m * m * 8);
            Max(max_3a, max_3a, src_03a, preg_108);
            DataCopy(src_03b, src0_ub3_1 + iter_m * m * 8);
            Max(max_3b, max_3b, src_03b, preg_108);
        }

        Max(max_0a, max_0a, max_1a, preg_108);
        Max(max_0b, max_0b, max_1b, preg_108);
        Max(max_2a, max_2a, max_3a, preg_108);
        Max(max_2b, max_2b, max_3b, preg_108);
        Max(max_0a, max_0a, max_2a, preg_108);
        Max(max_0b, max_0b, max_2b, preg_108);
        Max(max_0a, max_0a, max_0b, preg_108);
        Muls(max_0a, max_0a, dScale, preg_108);

        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            (__ubuf__ T *&)new_global_max, max_0a, preg_108);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_even, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_odd, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_even, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_odd, 0, preg_134);
        RegTensor<uint8_t> idx_nd2nz;
        uint16_t loopNum;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2_even, 0, preg_134);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2_odd, 0, preg_134);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3_even, 0, preg_134);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3_odd, 0, preg_134);
            __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
            DataCopy(idx_nd2nz, indexesUb);
            loopNum = ubN / 8;
        } else {
            loopNum = ubN / 4;
        }

        for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
                DataCopy(vreg_x_f32_a, input_x_local_UB + i0 * m * 2);
                DataCopy(vreg_x_f32_b, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
                DataCopy(vreg_x_f32_1_a, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
                DataCopy(vreg_x_f32_1_b, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

                DataCopy(vreg_x_f32_2_a, input_x_local_UB + i0 * m * 2 + 64);
                DataCopy(vreg_x_f32_2_b, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
                DataCopy(vreg_x_f32_3_a, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
                DataCopy(vreg_x_f32_3_b, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
            } else {
                DataCopy(vreg_x_f32_a, input_x_local_UB + i0 * m);
                DataCopy(vreg_x_f32_b, input_x_local_UB + ubN * m / 2 + i0 * m);
                DataCopy(vreg_x_f32_1_a, input_x_local_UB + ubN * m / 4 + i0 * m);
                DataCopy(vreg_x_f32_1_b, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
            }

            Muls(vreg_x_f32_a, vreg_x_f32_a, dScale, preg_108);
            Muls(vreg_x_f32_b, vreg_x_f32_b, dScale, preg_108);
            FusedExpSub(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134);    // 1
            FusedExpSub(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134);     // 5
 
            Muls(vreg_x_f32_1_a, vreg_x_f32_1_a, dScale, preg_108);
            Muls(vreg_x_f32_1_b, vreg_x_f32_1_b, dScale, preg_108);
            FusedExpSub(vreg_x_exp_even_1, vreg_x_f32_1_a, max_0a, preg_134);// 3
            FusedExpSub(vreg_x_exp_odd_1, vreg_x_f32_1_b, max_0a, preg_134); // 7

 
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
                Muls(vreg_x_f32_2_a, vreg_x_f32_2_a, dScale, preg_108);
                Muls(vreg_x_f32_2_b, vreg_x_f32_2_b, dScale, preg_108);
                FusedExpSub(vreg_x_exp_even_2, vreg_x_f32_2_a, max_0a, preg_134);// 2
                FusedExpSub(vreg_x_exp_odd_2, vreg_x_f32_2_b, max_0a, preg_134); // 6
 
                Muls(vreg_x_f32_3_a, vreg_x_f32_3_a, dScale, preg_108);
                Muls(vreg_x_f32_3_b, vreg_x_f32_3_b, dScale, preg_108);
                FusedExpSub(vreg_x_exp_even_3, vreg_x_f32_3_a, max_0a, preg_134);// 4
                FusedExpSub(vreg_x_exp_odd_3, vreg_x_f32_3_b, max_0a, preg_134); // 8
            }
 
            if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_even, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_odd, preg_135);
                DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

                Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_even_1, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_odd_1, preg_135);
                DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
                /* vreg_x_exp_bf16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                  个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
                Add(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
                Add(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);

                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
                Add(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
                Add(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
            } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
                Add(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_even, preg_135);
                } else {
                    Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_even, preg_135);
                }
 
                Add(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_even, vreg_x_exp_even_1, preg_135);
                } else {
                    Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_even, vreg_x_exp_even_1, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_even, preg_134);
 
                Add(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_even_1, vreg_x_exp_odd, preg_135);
                } else {
                    Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_even_1, vreg_x_exp_odd, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_even_1, preg_134);
 
                Add(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_odd, vreg_x_exp_odd_1, preg_135);
                } else {
                    Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_odd, vreg_x_exp_odd_1, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_odd, preg_134);
 
                Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
 
                // -----------------------------------------------------------------------------//
                Add(vreg_x_sum_2_even, vreg_x_exp_even_2, vreg_x_sum_2_even, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_even_2, preg_135);
                } else {
                    Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_even_2, preg_135);
                }
 
                Add(vreg_x_sum_3_even, vreg_x_exp_even_3, vreg_x_sum_3_even, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_even_2, vreg_x_exp_even_3, preg_135);
                } else {
                    Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_even_2, vreg_x_exp_even_3, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_even_2, preg_134);
 
                Add(vreg_x_sum_2_odd, vreg_x_exp_odd_2, vreg_x_sum_2_odd, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_even_3, vreg_x_exp_odd_2, preg_135);
                } else {
                    Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_even_3, vreg_x_exp_odd_2, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_even_3, preg_134);
 
                Add(vreg_x_sum_3_odd, vreg_x_exp_odd_3, vreg_x_sum_3_odd, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_odd_2, vreg_x_exp_odd_3, preg_135);
                } else {
                    Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_odd_2, vreg_x_exp_odd_3, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_odd_2, preg_134);
 
                Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
            } else {
                Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_even, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_odd, preg_135);
                DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

                Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, preg_135);
                DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
                /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                  个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);    
                Add(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
                Add(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);    
                Add(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
                Add(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
            }
        }
        Add(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134);
        Add(vreg_x_sum1, vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134);
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum2, vreg_x_sum_2_odd, vreg_x_sum_2_even, preg_134);
            Add(vreg_x_sum3, vreg_x_sum_3_odd, vreg_x_sum_3_even, preg_134);
        }
 
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
            Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
        }
 
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
    }
}

template <typename T, typename T2, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnUpdate(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, float deScaleQK, const T minValue, float keepProb)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    __VEC_SCOPE__{
        RegTensor<float> vreg_x_sum_even;
        RegTensor<float> vreg_x_sum_odd;
        RegTensor<float> vreg_x_sum_1_even;
        RegTensor<float> vreg_x_sum_1_odd;
        RegTensor<float> vreg_x_sum_2_even;
        RegTensor<float> vreg_x_sum_2_odd;
        RegTensor<float> vreg_x_sum_3_even;
        RegTensor<float> vreg_x_sum_3_odd;
        RegTensor<float> vreg_x_sum0;
        RegTensor<float> vreg_x_sum1;
        RegTensor<float> vreg_x_sum2;
        RegTensor<float> vreg_x_sum3;
        RegTensor<half> vreg_x_exp_even_f16;
        RegTensor<half> vreg_x_exp_odd_f16;
        RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;

        RegTensor<float> vreg_x_exp_even;
        RegTensor<float> vreg_x_exp_odd;
        RegTensor<float> vreg_x_f32_a;
        RegTensor<float> vreg_x_f32_b;
        RegTensor<float> vreg_x_exp_even_1;
        RegTensor<float> vreg_x_exp_odd_1;
        RegTensor<float> vreg_x_exp_even_2;
        RegTensor<float> vreg_x_exp_odd_2;
        RegTensor<float> vreg_x_exp_even_3;
        RegTensor<float> vreg_x_exp_odd_3;
        RegTensor<half> vreg_x_exp_even_f16_1;
        RegTensor<half> vreg_x_exp_odd_f16_1;
        RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
        RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

        RegTensor<float> vreg_x_f32_1_a;
        RegTensor<float> vreg_x_f32_1_b;
        RegTensor<float> vreg_x_f32_2_a;
        RegTensor<float> vreg_x_f32_2_b;
        RegTensor<float> vreg_x_f32_3_a;
        RegTensor<float> vreg_x_f32_3_b;
        RegTensor<float> vreg_x_max_f32_b;
        RegTensor<half> vreg_x_exp_f16_pack;
        RegTensor<half> vreg_x_exp_f16_1_pack;
        RegTensor<half> vreg_x_exp_f16_packa;
        RegTensor<half> vreg_x_exp_f16_1_packa;

        RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

        MaskReg preg_100;
        MaskReg preg_101;
        MaskReg preg_134;
        MaskReg preg_135;
        MaskReg preg_136;
        preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
        preg_135 = CreateMask<T, MaskPattern::ALL>();
        uint32_t sreg_92 = (uint32_t)128ULL;
        preg_136 = UpdateMask<uint16_t>(sreg_92);
        MaskReg preg_108;
        RegTensor<float> src_00a, src_01a, src_02a, src_03a;
        RegTensor<float> src_00b, src_01b, src_02b, src_03b;
        RegTensor<float> src_10a, src_11a, src_12a, src_13a;
        RegTensor<float> src_10b, src_11b, src_12b, src_13b;
        RegTensor<float> max_0a, max_1a, max_2a, max_3a;
        RegTensor<float> max_0b, max_1b, max_2b, max_3b;
        RegTensor<float> vreg_min;

        RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
        RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;
 
        __ubuf__ float *src0_ub = (__ubuf__ float*) input_x_local_UB;
        __ubuf__ float *src0_ub_1 = (__ubuf__ float*) input_x_local_UB + m;
        __ubuf__ T2 *x_exp_1;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
            x_exp_1 = x_exp + 32;
        } else {
            x_exp_1 = x_exp + (ubN * 4);
        }
        __ubuf__ float *src0_ub1 = src0_ub + m * 2;
        __ubuf__ float *src0_ub1_1 = src0_ub + m * 3;
        __ubuf__ float *src0_ub2 = src0_ub + m * 4;
        __ubuf__ float *src0_ub2_1 = src0_ub + m * 5;
        __ubuf__ float *src0_ub3 = src0_ub + m * 6;
        __ubuf__ float *src0_ub3_1 = src0_ub + m * 7;

        Duplicate(max_0a, minValue);
        Duplicate(max_0b, minValue);
        Duplicate(max_1a, minValue);
        Duplicate(max_1b, minValue);
        Duplicate(max_2a, minValue);
        Duplicate(max_2b, minValue);
        Duplicate(max_3a, minValue);
        Duplicate(max_3b, minValue);
        Duplicate(vreg_min, minValue);
        for (uint16_t i = originN; i < ubN; ++i) {
            DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
        }
        mem_bar(VST_VLD);

        preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8); ++iter_m) {
            DataCopy(src_00a, src0_ub + iter_m * m * 8);
            Max(max_0a, max_0a, src_00a, preg_108);
            DataCopy(src_00b, src0_ub_1 + iter_m * m * 8);
            Max(max_0b, max_0b, src_00b, preg_108);
            DataCopy(src_01a, src0_ub1 + iter_m * m * 8);
            Max(max_1a, max_1a, src_01a, preg_108);
            DataCopy(src_01b, src0_ub1_1 + iter_m * m * 8);
            Max(max_1b, max_1b, src_01b, preg_108);
            DataCopy(src_02a, src0_ub2 + iter_m * m * 8);
            Max(max_2a, max_2a, src_02a, preg_108);
            DataCopy(src_02b, src0_ub2_1 + iter_m * m * 8);
            Max(max_2b, max_2b, src_02b, preg_108);
            DataCopy(src_03a, src0_ub3 + iter_m * m * 8);
            Max(max_3a, max_3a, src_03a, preg_108);
            DataCopy(src_03b, src0_ub3_1 + iter_m * m * 8);
            Max(max_3b, max_3b, src_03b, preg_108);
        }
        DataCopy(vreg_x_max_f32_b, new_global_max);
        Max(max_0a, max_0a, max_1a, preg_108);
        Max(max_0b, max_0b, max_1b, preg_108);
        Max(max_2a, max_2a, max_3a, preg_108);
        Max(max_2b, max_2b, max_3b, preg_108);
        Max(max_0a, max_0a, max_2a, preg_108);
        Max(max_0b, max_0b, max_2b, preg_108);
        Max(max_0a, max_0a, max_0b, preg_108);
        Muls(max_0a, max_0a, dScale, preg_108);
        Max(max_0a, max_0a, vreg_x_max_f32_b, preg_108);

        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max_0a, preg_134);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            (__ubuf__ T *&)new_global_max, max_0a, preg_108);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            (__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);    

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_even, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_odd, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1_even, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1_odd, 0, preg_134);
        RegTensor<uint8_t> idx_nd2nz;
        uint16_t loopNum;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2_even, 0, preg_134);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2_odd, 0, preg_134);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3_even, 0, preg_134);
            Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3_odd, 0, preg_134);
            __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
            DataCopy(idx_nd2nz, indexesUb);
            loopNum = ubN / 8;
        } else {
            loopNum = ubN / 4;
        }
  
        for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
                DataCopy(vreg_x_f32_a, input_x_local_UB + i0 * m * 2);
                DataCopy(vreg_x_f32_b, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
                DataCopy(vreg_x_f32_1_a, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
                DataCopy(vreg_x_f32_1_b, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);
 
                DataCopy(vreg_x_f32_2_a, input_x_local_UB + i0 * m * 2 + 64);
                DataCopy(vreg_x_f32_3_a, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
                DataCopy(vreg_x_f32_2_b, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
                DataCopy(vreg_x_f32_3_b, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
            } else {
                DataCopy(vreg_x_f32_a, input_x_local_UB + i0 * m);
                DataCopy(vreg_x_f32_b, input_x_local_UB + ubN * m / 2 + i0 * m);
                DataCopy(vreg_x_f32_1_a, input_x_local_UB + ubN * m / 4 + i0 * m);
                DataCopy(vreg_x_f32_1_b, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
            }
 
 
            Muls(vreg_x_f32_a, vreg_x_f32_a, dScale, preg_108);
            Muls(vreg_x_f32_b, vreg_x_f32_b, dScale, preg_108);
            FusedExpSub(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134);    // 1
            FusedExpSub(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134);     // 5
 
            Muls(vreg_x_f32_1_a, vreg_x_f32_1_a, dScale, preg_108);
            Muls(vreg_x_f32_1_b, vreg_x_f32_1_b, dScale, preg_108);
            FusedExpSub(vreg_x_exp_even_1, vreg_x_f32_1_a, max_0a, preg_134);// 3
            FusedExpSub(vreg_x_exp_odd_1, vreg_x_f32_1_b, max_0a, preg_134); // 7
 
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
                Muls(vreg_x_f32_2_a, vreg_x_f32_2_a, dScale, preg_108);
                Muls(vreg_x_f32_2_b, vreg_x_f32_2_b, dScale, preg_108);
                FusedExpSub(vreg_x_exp_even_2, vreg_x_f32_2_a, max_0a, preg_134);// 2
                FusedExpSub(vreg_x_exp_odd_2, vreg_x_f32_2_b, max_0a, preg_134); // 6
 
                Muls(vreg_x_f32_3_a, vreg_x_f32_3_a, dScale, preg_108);
                Muls(vreg_x_f32_3_b, vreg_x_f32_3_b, dScale, preg_108);
                FusedExpSub(vreg_x_exp_even_3, vreg_x_f32_3_a, max_0a, preg_134);// 4
                FusedExpSub(vreg_x_exp_odd_3, vreg_x_f32_3_b, max_0a, preg_134); // 8
            }
            if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_even, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_odd, preg_135);
                DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

                Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_even_1, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_odd_1, preg_135);
                DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
                /* vreg_x_exp_bf16_pack会不连续的存储156在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                  个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);     
                Add(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
                Add(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136); 
                Add(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
                Add(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
            } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
                Add(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_even, preg_135);
                } else {
                    Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_even, preg_135);
                }
 
                Add(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_even, vreg_x_exp_even_1, preg_135);
                } else {
                    Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_even, vreg_x_exp_even_1, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_even, preg_134);
 
                Add(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_even_1, vreg_x_exp_odd, preg_135);
                } else {
                    Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_even_1, vreg_x_exp_odd, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_even_1, preg_134);
 
                Add(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_odd, vreg_x_exp_odd_1, preg_135);
                } else {
                    Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_odd, vreg_x_exp_odd_1, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_odd, preg_134);
 
                Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);
 
                // -----------------------------------------------------------------------------//
                Add(vreg_x_sum_2_even, vreg_x_exp_even_2, vreg_x_sum_2_even, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_even_2, preg_135);
                } else {
                    Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_even_2, preg_135);
                }
                Add(vreg_x_sum_3_even, vreg_x_exp_even_3, vreg_x_sum_3_even, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_even_2, vreg_x_exp_even_3, preg_135);
                } else {
                    Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_even_2, vreg_x_exp_even_3, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_even_2, preg_134);
 
                Add(vreg_x_sum_2_odd, vreg_x_exp_odd_2, vreg_x_sum_2_odd, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_even_3, vreg_x_exp_odd_2, preg_135);
                } else {
                    Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_even_3, vreg_x_exp_odd_2, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_even_3, preg_134);
 
                Add(vreg_x_sum_3_odd, vreg_x_exp_odd_3, vreg_x_sum_3_odd, preg_134);
                if constexpr (IsSameType<T2, hifloat8_t>::value) {
                    Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_odd_2, vreg_x_exp_odd_3, preg_135);
                } else {
                    Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_odd_2, vreg_x_exp_odd_3, preg_135);
                }
                Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_odd_2, preg_134);
 
                Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
            } else {
                Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_even, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_odd, preg_135);
                DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

                Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, preg_135);

                DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

                /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                  个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136); 
                Add(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
                Add(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);

                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136); 
                Add(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
                Add(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
            }
        }
        Add(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134);
        Add(vreg_x_sum1, vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134);
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum2, vreg_x_sum_2_odd, vreg_x_sum_2_even, preg_134);
            Add(vreg_x_sum3, vreg_x_sum_3_odd, vreg_x_sum_3_even, preg_134);
        }
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
            Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
        }
        RegTensor<float> vreg_l0;
        DataCopy(vreg_l0, new_global_sum);
        Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
        Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)new_global_sum, vreg_l0, preg_134); 
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
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] n, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] originN, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isUpdate, enable flash mode
 * @param [in] oriNRange, originN range
 */

template <typename T, typename T2, bool isUpdate = false, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1VfDn(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                       const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& expMaxTensor, TBuf<> *vselrIndexesBuf,
                                       const uint32_t m, const uint32_t n, const uint32_t originN,
                                       const T scale, float deScaleQK, const T minValue, float keepProb)
{
    if constexpr (!isUpdate) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnNoUpdate<T, T2, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor,
            m, n, originN, scale, deScaleQK, minValue, keepProb);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnUpdate<T, T2, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor,
            m, n, originN, scale, deScaleQK, minValue, keepProb);
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
    __VEC_SCOPE__{
        RegTensor<float> broadcast_reg;
        MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopM; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(
                broadcast_reg, ori_ub + i * 8);
            DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)out_ub + i * 64, broadcast_reg, preg_all); 
        }
    }
}
}
#endif // MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_