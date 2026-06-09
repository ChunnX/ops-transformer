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
 * \file vf_basic_block_aligned128_no_update.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_H
#define VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_H

#include "vf_basic_block_utils.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {
// no update, originN == 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
    bool isMlaFullQuant = false, bool hasSink = false>
__simd_vf__ void ProcessVec1NoUpdateImpl128VF(
    __ubuf__ T2 * expUb, __ubuf__ T2 * x_expUb, __ubuf__ pseShiftType * pseUb, __ubuf__ T * expSumUb,
    __ubuf__ T * maxUb, __ubuf__ T * maxUbStart, __ubuf__ T * srcUb, __ubuf__ T * qScaleUb,
    __ubuf__ uint8_t * indexesUb, __ubuf__ uint32_t * maskUb, __ubuf__ uint32_t * maskUb1, __ubuf__ uint32_t * maskUb2, __ubuf__ uint32_t * maskUb3, __ubuf__ uint32_t * maskUb4, __ubuf__ uint32_t * maskUb5, __ubuf__ uint32_t * maskUb6, __ubuf__ uint32_t * maskUb7,
    __ubuf__ uint32_t * dropMaskUb, float divValue, const uint32_t blockStride,
    const uint32_t repeatStride, const float dScale, const uint16_t m, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue,
    const float deSCaleKValue = 1.0f, const float sinkValue = 0.0f)
{
    RegTensor<float> vreg_min;

    // 记作两两比较存放mask掩码后的数据，用8个寄存器来存放一行128个fp32的数据
    RegTensor<float> vreg_sel;
    RegTensor<float> vreg_sel1;
    RegTensor<float> vreg_sel2;
    RegTensor<float> vreg_sel3;
    RegTensor<float> vreg_sel4;
    RegTensor<float> vreg_sel5;
    RegTensor<float> vreg_sel6;
    RegTensor<float> vreg_sel7;
    RegTensor<float> vreg_sel_unroll; // 不破坏原来的

    // 输入x原始数据，依旧是8个寄存器来存放一行128个fp32的数据
    RegTensor<float> vreg_input_x;
    RegTensor<float> vreg_input_x1;
    RegTensor<float> vreg_input_x2;
    RegTensor<float> vreg_input_x3;
    RegTensor<float> vreg_input_x4;
    RegTensor<float> vreg_input_x5;
    RegTensor<float> vreg_input_x6;
    RegTensor<float> vreg_input_x7;
    RegTensor<float> vreg_input_x_unroll; // 不破坏原来的

    // 存放两两寄存器逐位比较max的结果，由于一行用到了8个寄存器，因此申请四个足以
    RegTensor<float> vreg_max_tmp;
    RegTensor<float> vreg_max_tmp1;
    RegTensor<float> vreg_max_tmp2;
    RegTensor<float> vreg_max_tmp3;

    // 最后reduce的一行里的全局max值，依旧一个保持不变
    RegTensor<float> vreg_input_max;

    // 这个是广播后的每一行的max值（对于noupdate分支来说，存放的是每一行的max会被广播到寄存器中），由于是一行行处理，因此申请一个足以
    RegTensor<float> vreg_max_brc;

    // 这是寄存器里广播一行的expsum值，由于原来是两个寄存器两两比较+reduce；现在fpga需要七次(4+2+1)两两寄存器比较+最后一次reduce（可以复用4次里面的一个寄存器，不需要额外开辟）
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_exp_sum1;
    RegTensor<float> vreg_exp_sum2;
    RegTensor<float> vreg_exp_sum3;

    // enen和old原意是将一行128拆分成两个寄存器，even给前者,old给后者，我们现在一行128被拆分成了8个寄存器，因此需要额外扩容3组
    RegTensor<float> vreg_exp_even;
    RegTensor<float> vreg_exp_odd;
    RegTensor<float> vreg_exp_even1;
    RegTensor<float> vreg_exp_odd1;
    RegTensor<float> vreg_exp_even2;
    RegTensor<float> vreg_exp_odd2;
    RegTensor<float> vreg_exp_even3;
    RegTensor<float> vreg_exp_odd3;

    // drop用的，不需要适配
    RegTensor<float> vreg_zero;
    // pse用的，不需要适配
    RegTensor<float> vreg_pse;
    RegTensor<float> vreg_pse_unroll;
    // 也是pse用的，不需要适配
    RegTensor<float> vreg_alibi;
    RegTensor<float> vreg_alibi_unroll;
    // drop用的，不需要适配
    RegTensor<float> vreg_sel_drop;
    RegTensor<float> vreg_sel_drop2;
    // 不知道干啥用的，不需要适配
    RegTensor<float> vreg_rowmax_p;
    // qscale相关的，一行对应一个qscale，这里寄存器存放的是被广播后的值
    RegTensor<float> vreg_scale_qk;
    // 跟hassink相关的，不需要适配
    RegTensor<float> vreg_sink_input;
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
    // 调用StoreUnAlign能力的时候，需要结合使用的，一行reduce后的max搬出会用到，因此不需要额外适配
    UnalignRegForStore ureg_max;
    // expsum同理
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
    // mask掩码寄存器，一行中16个一组，因此需要八个寄存器
    MaskReg preg_compare;
    MaskReg preg_compare1;
    MaskReg preg_compare2;
    MaskReg preg_compare3;
    MaskReg preg_compare4;
    MaskReg preg_compare5;
    MaskReg preg_compare6;
    MaskReg preg_compare7;
    MaskReg preg_compare_unroll;// 不破坏原来的逻辑
    // 跟drop有关的，无需适配
    MaskReg preg1;
    MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
    MaskReg preg3;
    MaskReg preg4;
    MaskReg preg5;
    MaskReg preg6;

    // todo
    // 3. 基于 expUb 进行指针运算
    __ubuf__ T2 * expUb1 = expUb + ((s1BaseSize + 1) * 128) / 4 * 1;
    __ubuf__ T2 * expUb2 = expUb + ((s1BaseSize + 1) * 128) / 4 * 2;
    __ubuf__ T2 * expUb3 = expUb + ((s1BaseSize + 1) * 128) / 4 * 3;

    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    if constexpr (hasAtten == 1) {
        Duplicate(vreg_min, minValue);
        if constexpr (isMlaSgd) {
            // todo
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare, ((__ubuf__ uint32_t*)(maskUb)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare1, ((__ubuf__ uint32_t*)(maskUb1)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare2, ((__ubuf__ uint32_t*)(maskUb2)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare3, ((__ubuf__ uint32_t*)(maskUb3)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare4, ((__ubuf__ uint32_t*)(maskUb4)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare5, ((__ubuf__ uint32_t*)(maskUb5)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare6, ((__ubuf__ uint32_t*)(maskUb6)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare7, ((__ubuf__ uint32_t*)(maskUb7)));
        }
    }
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                    pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi, posShift);
        Arange(vreg_alibi_unroll, posShift + 64);
    }
    for (uint16_t i = 0; i < m; ++i) {
        // todo
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        LoadAlign(vreg_input_x1, srcUb + floatRepSize + i * s2BaseSize);
        LoadAlign(vreg_input_x2, srcUb + floatRepSize * 2 + i * s2BaseSize);
        LoadAlign(vreg_input_x3, srcUb + floatRepSize * 3 + i * s2BaseSize);
        LoadAlign(vreg_input_x4, srcUb + floatRepSize * 4 + i * s2BaseSize);
        LoadAlign(vreg_input_x5, srcUb + floatRepSize * 5 + i * s2BaseSize);
        LoadAlign(vreg_input_x6, srcUb + floatRepSize * 6 + i * s2BaseSize);
        LoadAlign(vreg_input_x7, srcUb + floatRepSize * 7 + i * s2BaseSize);
        if constexpr (isMlaFullQuant) {
            // todo
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_scale_qk, qScaleUb + i);
            Muls(vreg_scale_qk, vreg_scale_qk, scale, preg_all);
            Muls(vreg_scale_qk, vreg_scale_qk, deSCaleKValue, preg_all);
            Mul(vreg_input_x, vreg_input_x, vreg_scale_qk, preg_all);
            Mul(vreg_input_x1, vreg_input_x1, vreg_scale_qk, preg_all);
            Mul(vreg_input_x2, vreg_input_x2, vreg_scale_qk, preg_all);
            Mul(vreg_input_x3, vreg_input_x3, vreg_scale_qk, preg_all);
            Mul(vreg_input_x4, vreg_input_x4, vreg_scale_qk, preg_all);
            Mul(vreg_input_x5, vreg_input_x5, vreg_scale_qk, preg_all);
            Mul(vreg_input_x6, vreg_input_x6, vreg_scale_qk, preg_all);
            Mul(vreg_input_x7, vreg_input_x7, vreg_scale_qk, preg_all);
        } else {
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
                    LoadAlign(vreg_pse, pseUb + i * pseStride);
                    LoadAlign(vreg_pse_unroll, pseUb + i * pseStride + (s2BaseSize >> 1));
                } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                    LoadAlign(vreg_pse_bf16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse_unroll, vreg_pse_bf16_unroll, preg_all_b16);
                } else {
                    LoadAlign(vreg_pse_f16_src, pseUb + i * pseStride);
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
                // todo
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare, (__ubuf__ uint32_t *&)maskUb, s2BaseSize);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare1, (__ubuf__ uint32_t *&)maskUb1, s2BaseSize); 
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare2, (__ubuf__ uint32_t *&)maskUb2, s2BaseSize); 
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare3, (__ubuf__ uint32_t *&)maskUb3, s2BaseSize); 
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare4, (__ubuf__ uint32_t *&)maskUb4, s2BaseSize); 
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare5, (__ubuf__ uint32_t *&)maskUb5, s2BaseSize); 
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare6, (__ubuf__ uint32_t *&)maskUb6, s2BaseSize); 
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare7, (__ubuf__ uint32_t *&)maskUb7, s2BaseSize); 
            }
            // todo
            Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
            Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
            Select(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
            Select(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
            Select(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
            Select(vreg_sel5, vreg_min, vreg_input_x5, preg_compare5);
            Select(vreg_sel6, vreg_min, vreg_input_x6, preg_compare6);
            Select(vreg_sel7, vreg_min, vreg_input_x7, preg_compare7);
            // todo
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 1 + i * s2BaseSize, vreg_sel1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_sel2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_sel3, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 4 + i * s2BaseSize, vreg_sel4, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 5 + i * s2BaseSize, vreg_sel5, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 6 + i * s2BaseSize, vreg_sel6, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 7 + i * s2BaseSize, vreg_sel7, preg_all);
            // todo
            Max(vreg_max_tmp, vreg_sel, vreg_sel1, preg_all);
            Max(vreg_max_tmp1, vreg_sel2, vreg_sel3, preg_all);
            Max(vreg_max_tmp2, vreg_sel4, vreg_sel5, preg_all);
            Max(vreg_max_tmp3, vreg_sel6, vreg_sel7, preg_all);
            Max(vreg_max_tmp, vreg_max_tmp, vreg_max_tmp1, preg_all);
            Max(vreg_max_tmp2, vreg_max_tmp2, vreg_max_tmp3, preg_all);
            Max(vreg_max_tmp, vreg_max_tmp, vreg_max_tmp2, preg_all);
        } else {
            // todo
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 1 + i * s2BaseSize, vreg_input_x1, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 2 + i * s2BaseSize, vreg_input_x2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 3 + i * s2BaseSize, vreg_input_x3, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 4 + i * s2BaseSize, vreg_input_x4, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 5 + i * s2BaseSize, vreg_input_x5, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 6 + i * s2BaseSize, vreg_input_x6, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize * 7 + i * s2BaseSize, vreg_input_x7, preg_all);
            Max(vreg_max_tmp, vreg_input_x, vreg_input_x1, preg_all);
            Max(vreg_max_tmp1, vreg_input_x2, vreg_input_x3, preg_all);
            Max(vreg_max_tmp2, vreg_input_x4, vreg_input_x5, preg_all);
            Max(vreg_max_tmp3, vreg_input_x6, vreg_input_x7, preg_all);
            Max(vreg_max_tmp, vreg_max_tmp, vreg_max_tmp1, preg_all);
            Max(vreg_max_tmp2, vreg_max_tmp2, vreg_max_tmp3, preg_all);
            Max(vreg_max_tmp, vreg_max_tmp, vreg_max_tmp2, preg_all);
        }
        Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_input_max, vreg_max_tmp, preg_all);
        if constexpr (hasSink) {
            Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
        }
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)maxUb), ureg_max, 0);
    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        // maxUb is [S1, 1], BRC_B32 is reading one fp32 element and broadcast it to all 64 vreg element
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg_max_brc, maxUbStart + i);
        if constexpr (IsSameType<T2, float>::value) {
            LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
            LoadAlign(vreg_input_x_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
        } else {
            // todo
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x, vreg_input_x1, srcUb + i * s2BaseSize);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x2, vreg_input_x3, srcUb + i * s2BaseSize + floatRepSize * 2);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x4, vreg_input_x5, srcUb + i * s2BaseSize + floatRepSize * 4);
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x6, vreg_input_x7, srcUb + i * s2BaseSize + floatRepSize * 6);
        }
        // todo
        ExpSub(vreg_exp_even, vreg_input_x, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd, vreg_input_x1, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even1, vreg_input_x2, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd1, vreg_input_x3, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even2, vreg_input_x4, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd2, vreg_input_x5, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_even3, vreg_input_x6, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd3, vreg_input_x7, vreg_max_brc, preg_all);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        // todo
        Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
        Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
        Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
        Add(vreg_exp_sum3, vreg_exp_even3, vreg_exp_odd3, preg_all);
        Add(vreg_exp_sum, vreg_exp_sum, vreg_exp_sum1, preg_all);
        Add(vreg_exp_sum1, vreg_exp_sum2, vreg_exp_sum3, preg_all);
        Add(vreg_exp_sum, vreg_exp_sum, vreg_exp_sum1, preg_all);
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_sum, vreg_exp_sum, preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)expSumUb), vreg_exp_sum, ureg_exp_sum, 1);
        
        if constexpr (isMlaFullQuant) {
            if constexpr (IsSameType<T2, int8_t>::value) {
                // INT8: × 127
                Muls(vreg_exp_even, vreg_exp_even, int8MaxValue, preg_all);
                Muls(vreg_exp_odd, vreg_exp_odd, int8MaxValue, preg_all);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                // FP8: × 448
                Muls(vreg_exp_even, vreg_exp_even, fp8e4m3MaxValue, preg_all);
                Muls(vreg_exp_even1, vreg_exp_even1, fp8e4m3MaxValue, preg_all);
                Muls(vreg_exp_even2, vreg_exp_even2, fp8e4m3MaxValue, preg_all);
                Muls(vreg_exp_even3, vreg_exp_even3, fp8e4m3MaxValue, preg_all);
                Muls(vreg_exp_odd, vreg_exp_odd, fp8e4m3MaxValue, preg_all);
                Muls(vreg_exp_odd1, vreg_exp_odd1, fp8e4m3MaxValue, preg_all);
                Muls(vreg_exp_odd2, vreg_exp_odd2, fp8e4m3MaxValue, preg_all);
                Muls(vreg_exp_odd3, vreg_exp_odd3, fp8e4m3MaxValue, preg_all);
            } else {
                // HIFP8: × 32768
                Muls(vreg_exp_even, vreg_exp_even, hifp8MaxValue, preg_all);
                Muls(vreg_exp_odd, vreg_exp_odd, hifp8MaxValue, preg_all);
            }
        }

        // dropmask compute
        if constexpr (hasDrop == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
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
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_even, blockStride, repeatStride, preg_all);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_expUb), vreg_exp_odd, blockStride, repeatStride, preg_all);
        } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_even_bf16, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd_bf16, vreg_exp_odd, preg_all);
            Or((RegTensor<uint16_t>&)vreg_exp_bf16, (RegTensor<uint16_t>&)vreg_exp_even_bf16,
                (RegTensor<uint16_t>&)vreg_exp_odd_bf16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
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
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_even_f8e5m2, (RegTensor<uint8_t>&)vreg_exp_odd_f8e5m2, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e5m2_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e5m2, vreg_exp_merge_tmp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
            // todo
            RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
            RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
            RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
            RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
            RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_3;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_1;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_3;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 32;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all);
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3_3, vreg_exp_even3, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3_3, vreg_exp_odd3, preg_all);
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_even_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_odd_f8e4m3, preg_all_b8);
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t>&)vreg_exp_even_f8e4m3_1, (RegTensor<uint8_t>&)vreg_exp_odd_f8e4m3_1, preg_all_b8);
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t>&)vreg_exp_even_f8e4m3_2, (RegTensor<uint8_t>&)vreg_exp_odd_f8e4m3_2, preg_all_b8);
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e4m3_3, (RegTensor<uint8_t>&)vreg_exp_even_f8e4m3_3, (RegTensor<uint8_t>&)vreg_exp_odd_f8e4m3_3, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
            Gather(vreg_exp_merge_f8e4m3_1, vreg_exp_merge_tmp_f8e4m3_1, vreg_exp_merge_f8e4m3_indexes);
            Gather(vreg_exp_merge_f8e4m3_2, vreg_exp_merge_tmp_f8e4m3_2, vreg_exp_merge_f8e4m3_indexes);
            Gather(vreg_exp_merge_f8e4m3_3, vreg_exp_merge_tmp_f8e4m3_3, vreg_exp_merge_f8e4m3_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb1), vreg_exp_merge_f8e4m3_1, blockStride, repeatStride, preg_all_b8_128);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb2), vreg_exp_merge_f8e4m3_2, blockStride, repeatStride, preg_all_b8_128);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb3), vreg_exp_merge_f8e4m3_3, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, int8_t>::value) {
            // 硬件不支持 float → int8 直接转换，需要分两步：float → half → int8
            RegTensor<int8_t> vreg_exp_merge_tmp_int8;
            RegTensor<int8_t> vreg_exp_merge_int8;
            MaskReg preg_all_f16 = CreateMask<half, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            // float → half → Or → half → int8 → Gather
            static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Cast<half, T, castTrait0>(vreg_exp_even_f16, vreg_exp_even, preg_all);
            Cast<half, T, castTrait1>(vreg_exp_odd_f16, vreg_exp_odd, preg_all);
            Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((MicroAPI::RegTensor<uint16_t>&)vreg_exp_f16,
                (MicroAPI::RegTensor<uint16_t>&)vreg_exp_even_f16,
                (MicroAPI::RegTensor<uint16_t>&)vreg_exp_odd_f16, preg_all_f16);
            Cast<T2, half, castTrait0>(vreg_exp_merge_tmp_int8, vreg_exp_f16, preg_all_f16);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vreg_exp_merge_int8,
                    (MicroAPI::RegTensor<uint16_t>&)vreg_exp_merge_tmp_int8);   
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ int8_t *&) expUb), vreg_exp_merge_int8, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
            RegTensor<hifloat8_t> vreg_exp_even_hif8;
            RegTensor<hifloat8_t> vreg_exp_odd_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_tmp_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_hif8;
            RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_hif8, (RegTensor<uint8_t>&)vreg_exp_even_hif8, (RegTensor<uint8_t>&)vreg_exp_odd_hif8, preg_all_b8);
            LoadAlign(vreg_exp_merge_hif8_indexes, indexesUb);
            Gather(vreg_exp_merge_hif8, vreg_exp_merge_tmp_hif8, vreg_exp_merge_hif8_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
        } else {
            Cast<T2, T, castTraitZero>(vreg_exp_even_f16, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitOne>(vreg_exp_odd_f16, vreg_exp_odd, preg_all);
            Or((RegTensor<uint16_t>&)vreg_exp_f16, (RegTensor<uint16_t>&)vreg_exp_even_f16, (RegTensor<uint16_t>&)vreg_exp_odd_f16, preg_all_b16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_f16, blockStride, repeatStride, preg_all_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
}

// no update, originN == 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
    bool isMlaFullQuant = false, bool hasSink = false>
__aicore__ inline void ProcessVec1NoUpdateImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<pseShiftType>& pseTensor, const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb,
    const LocalTensor<T>& queryScaleUb = LocalTensor<T>(), const float deSCaleKValue = 1.0f,
    const float sinkValue = 0.0f)
{
    float divValue = 1.0f / keepProb;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = (s1BaseSize / CV_RATIO) | 0x1;
    const uint32_t repeatStride = 1;
    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * x_expUb = nullptr;
    if constexpr (IsSameType<T2, float>::value) {
        x_expUb = expUb + ((s1BaseSize / CV_RATIO) + 1) * (s2BaseSize >> 1);
    }
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * qScaleUb = (__ubuf__ T*)queryScaleUb.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
    __ubuf__ uint32_t * maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    // todo
    __ubuf__ uint32_t * maskUb1 = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize);
    __ubuf__ uint32_t * maskUb2 = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize * 2);
    __ubuf__ uint32_t * maskUb3 = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize * 3);
    __ubuf__ uint32_t * maskUb4 = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize * 4);
    __ubuf__ uint32_t * maskUb5 = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize * 5);
    __ubuf__ uint32_t * maskUb6 = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize * 6);
    __ubuf__ uint32_t * maskUb7 = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize * 7);
    __ubuf__ uint32_t *  dropMaskUb = (__ubuf__ uint32_t*)dropTensor.GetPhyAddr();
    const float dScale = scale * dScaleQK;

    ProcessVec1NoUpdateImpl128VF<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop,
        isMlaSgd, isMlaFullQuant, hasSink>(
        expUb, x_expUb, pseUb, expSumUb, maxUb, maxUbStart, srcUb, qScaleUb, indexesUb, maskUb, maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7,
        dropMaskUb, divValue, blockStride, repeatStride, dScale, m, pseStride, slopes, posShift, scale, dScaleQK,
        minValue, deSCaleKValue, sinkValue);
}
} // namespace

#endif // VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_H
