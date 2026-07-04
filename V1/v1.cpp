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
#ifndef VF_BASIC_BLOCK_FULLQUANT_MX_ALIGNED128_NO_UPDATE_FULL_QUANT_H
#define VF_BASIC_BLOCK_FULLQUANT_MX_ALIGNED128_NO_UPDATE_FULL_QUANT_H

#include "vf_basic_block_utils.h"
#include "../pse.h"

namespace fullQuant {
using namespace regbaseutilFullQuant;
namespace FaVectorApiFullQuant {
// no update, originN == 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__simd_vf__ void ProcessVec1NoUpdateImpl128Mxfp8FullquantVF(
    __ubuf__ T2 * expUb, __ubuf__ T2 * x_expUb, __ubuf__ pseShiftType * pseUb, __ubuf__ T * expSumUb, 
    __ubuf__ T * maxUb, __ubuf__ T * maxUbStart, __ubuf__ T * srcUb, __ubuf__ T * qScaleUb, 
    __ubuf__ uint8_t * indexesUb, __ubuf__ uint32_t * maskUb, __ubuf__ uint32_t * maskUbUnroll, __ubuf__ uint32_t * dropMaskUb,
    __ubuf__ fp8_e8m0_t *pScaleSubLoop0, __ubuf__ float *pre_loop_max, __ubuf__ float *pre_loop_sum, __ubuf__ float *first_loop_sum, uint32_t subLoop,
    __ubuf__ T * inExpSumUb, __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpExpSumUb2, __ubuf__ T * expMaxUb,
    __ubuf__ T * inMaxUb, __ubuf__ T * tmpMaxUb, __ubuf__ T * tmpMaxUb2,
    float divValue, const uint32_t blockStride, const uint32_t repeatStride, const float dScale, 
    const uint16_t m, const uint32_t pseStride, const float slopes, const float posShift, const T scale, 
    const float dScaleQK, const T minValue, const float deSCaleKValue = 1.0f, const float pScale = 1.0f)
{
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_sel;
    RegTensor<float> vreg_sel_unroll;
    RegTensor<float> vreg_input_x;
    RegTensor<float> vreg_input_x_unroll;
    RegTensor<float> vreg_max_tmp;
    RegTensor<float> vreg_input_max;
    // 新增：
    RegTensor<float> vreg_input_max_v2;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_exp_max_v2;
    // end
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
    RegTensor<float> vreg_rowmax_p;
    RegTensor<float> vreg_scale_qk;
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
    // pscale 新增
    // fp8_e8m0_t
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;
    // pscale----subloop1会用到的，最后pscale会从fp32--bf16---fp8e8m0
    RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
    RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_0;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_1;
    // 后64行新增的
    RegTensor<bfloat16_t> vreg_p_scale_bf16_v2_0;
    RegTensor<bfloat16_t> vreg_p_scale_bf16_v2_1;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_v2_0;
    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_v2_1;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

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
    // 新增pscale
    //pScale
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

    if constexpr (hasAtten == 1) {
        Duplicate(vreg_min, minValue);
        if constexpr (isMlaSgd) {
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare, ((__ubuf__ uint32_t*)(maskUb)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare_unroll, ((__ubuf__ uint32_t*)(maskUbUnroll)));
        }
    }
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                    pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi, posShift);
        Arange(vreg_alibi_unroll, posShift + 64);
    }
    // 开始每一行的循环
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        LoadAlign(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);
        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            // 此步已经将input_x * scale了，由于QKSCALE = 1，因此这里dScale =scale
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
        // 如果涉及到掩码操作
        if constexpr (hasAtten == 1) {
            // atten mask
            if constexpr (!isMlaSgd) {
                // 逐行去拷贝mak矩阵
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare, (__ubuf__ uint32_t *&)maskUb, s2BaseSize);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, s2BaseSize); 
            }
            // 对mask矩阵进行掩码
            Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
            Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll);
            // 搬出到UB上
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_sel_unroll, preg_all);
            // 将一行(两个寄存器存储)用max，求出一个寄存器的max，记作vreg_max_tmp
            Max(vreg_max_tmp, vreg_sel, vreg_sel_unroll, preg_all);
            // 新增pscale的相关计算：
            //TODO: pScale,preg均需要关注
            Sub(vreg_max_tmp, vreg_max_tmp, vreg_ln_p_scale, preg_all);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp, preg_all); // 得到一个标量的max值，记作：vreg_input_max，此时max已经跟pscale相乘了
        } else {
            // 如果不涉及掩码操作，就把 *scale之后的inputx搬出即可
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize, vreg_input_x_unroll, preg_all);
            // 同样求一个寄存器的max
            Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll, preg_all);
            // 新增pscale的相关计算：
            //TODO: pScale,preg均需要关注
            Sub(vreg_max_tmp, vreg_max_tmp, vreg_ln_p_scale, preg_all);
            // 得到一行里max的标量：vreg_max_tmp
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_max_tmp, preg_all);
        }
        // pscale新增---为了将每一行的max对齐到ln2的倍数
        // 目的是为了fp32 --- cast --- fp8e8m0,没有精度损失
        Muls(vreg_input_max, vreg_input_max, INV_LN2, preg_all);
        Truncate<T, RoundMode::CAST_CEIL>(vreg_input_max, vreg_input_max, preg_all); // 向上取整
        Muls(vreg_input_max, vreg_input_max, LN2, preg_all);
        // todo
        // 连续非对齐搬出，不断地更新UB地址+=1，每次搬运一个数据量
        // subLoop等于0时，先赋值给maxUb
        // subLoop等于1时，再赋值给tmpMaxUb（局部的）
        if (subLoop == 0) {
            StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
        } else if (subLoop == 1) {
            StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1); // subloop=1时，当前块的max，存到了tmpMaxUb
        }
    }
    // TODO：如果subLoop==1，需要进行更新maxub，和计算更新因子expmax的操作
    if (subLoop == 0) {
        // 配套StoreUnAlign使用
        StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), ureg_max, 0);
    } else if (subLoop == 1) {
        if (m <= 64) { // max可以用一个寄存器存下来
            // 配套StoreUnAlign使用
            StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
            RegTensor<float> vreg_in_max;
            LoadAlign(vreg_in_max, inMaxUb);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();// 同步
            LoadAlign(vreg_input_max, tmpMaxUb2);
            Max(vreg_input_max, vreg_in_max, vreg_input_max, preg_all); // 此步就是为了将同一个基本块的subloop1的max更新为subloop0和1之间的全局最大
            ExpSub(vreg_exp_max, vreg_in_max, vreg_input_max, preg_all);// 更新因子：exp(上一个全局-当前全局（已经更新）
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);// 搬出到寄存器expMaxUb上，代表更新因子
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)maxUb, vreg_input_max, preg_all); // 当前基本块的全局最大值，搬运到maxUb上
        } else { // s1方向行数超过了64，因此需要两个寄存器存max
            // 配套StoreUnAlign使用
            StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
            RegTensor<float> vreg_in_max;
            RegTensor<float> vreg_in_max_v2;
            LoadAlign(vreg_in_max, inMaxUb);// in是历史的全局最大值，这里的inMaxUb相当于maxub，可以从最外部传参看出，但是这里有个问题，由于m=128，因此，一次寄存器放不下，需要两个寄存器去存储
            LoadAlign(vreg_in_max_v2, inMaxUb + 64);// 64相当于将inMaxUb偏移64个元素之后，赋值给第二个vreg_in_max_v2
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();// 同步
            LoadAlign(vreg_input_max, tmpMaxUb2);//tmp是当前的行最大值，tmpMaxUb2和tmpMaxUb共同一个地址
            LoadAlign(vreg_input_max_v2, tmpMaxUb2 + 64);// 64相当于将tmpMaxUb2偏移64个元素之后，赋值给第二个vreg_input_max_v2
            Max(vreg_input_max, vreg_in_max, vreg_input_max, preg_all); // loop1的max需要和loop0的max取一个max，保证loop1的max是当前的全局最大max，不要被loop1的max覆盖掉
            // 针对s1方向基本块新增：
            // 两个128基本块s1方向的后半段
            // int32_t pltOriTailMHalfRealSize = m - 64;
            Max(vreg_input_max_v2, vreg_in_max_v2, vreg_input_max_v2, preg_all);
            ExpSub(vreg_exp_max, vreg_in_max, vreg_input_max, preg_all);// 更新因子：exp(上一个全局-当前全局（已经更新）)
            // 新增：
            // MaskReg preg_s1half_tail_m = UpdateMask<float>(pltOriTailMHalfRealSize); // 后半段的S1方向真实长度
            ExpSub(vreg_exp_max_v2, vreg_in_max_v2, vreg_input_max_v2, preg_all);// 更新因子：exp(上一个全局-当前全局（已经更新）)
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);// 搬出到寄存器expMaxUb上，代表更新因子
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)maxUb, vreg_input_max, preg_all); // 当前基本块的全局最大值，搬运到maxUb上
            // 新增：
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)expMaxUb + 64, vreg_exp_max_v2, preg_all);// 搬出到寄存器expMaxUb上，代表更新因子，64：128的后半段
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)maxUb + 64, vreg_input_max_v2, preg_all); // 当前基本块的全局最大值，搬运到maxUb上:64：128的后半段
        }
    }

    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    // 遍历行数
    for (uint16_t i = 0; i < m; ++i) {
        // maxUb is [S1, 1], BRC_B32 is reading one fp32 element and broadcast it to all 64 vreg element
        // 广播搬运，搬运一个元素，并广播整个寄存器，vreg_max_brc
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg_max_brc, maxUbStart + i);

        if constexpr (IsSameType<T2, float>::value) {
            LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
            LoadAlign(vreg_input_x_unroll, srcUb + i * s2BaseSize + (s2BaseSize >> 1));
        } else {
            // 双搬入模式，vreg_input_x 和 vreg_input_x_unroll，按照奇偶索引，存储一行数据
            LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg_input_x, vreg_input_x_unroll, srcUb + i * s2BaseSize);
        }
        // 计算softmax分子（exp(x-max)）
        ExpSub(vreg_exp_even, vreg_input_x, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max_brc, preg_all);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_sum, vreg_exp_sum, preg_all); // 广播标量
        //todo
        // 搬出当前块的局部expsum
        if (subLoop == 0) {
            // 连续非对齐搬出，到expSumUb上，expSumUb+=1
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)expSumUb), vreg_exp_sum, ureg_exp_sum, 1);
        } else if (subLoop == 1) {
            StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum, ureg_exp_sum, 1); //subloop =1的块时，将softmax分子（exp(x-max)），搬出到tmpExpSumUb
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
            RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
            MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            // cast计算
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
            // or成一行
            Or((RegTensor<uint8_t>&)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_even_f8e4m3, (RegTensor<uint8_t>&)vreg_exp_odd_f8e4m3, preg_all_b8);
            // 加载索引index
            LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
            // 通过gather重排得到vreg_exp_merge_f8e4m3（cast成fp8之后的）
            Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
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
    // todo
    // 开始更新expsum
    if (subLoop == 0) {
        StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0); //将当前的expsum搬出到expSumUb上，配套使用StoreUnAlign
        // pscale新增
        // 给subloop0占位填充1.0即可，后续会被subloop1复写
        // 寄存器存放了256个1.0
        MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
        Duplicate(vreg_p_scale_f8e8m0, 0x7f, preg_all_b8);
        StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
 	                                                               vreg_p_scale_f8e8m0, preg_all_b8);
    } else if (subLoop == 1) {
        MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
        if (m <= 64) {
            StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            RegTensor<float> first_loop_sum;
            LoadAlign(first_loop_sum, inExpSumUb); // 加载旧的expsum
            RegTensor<float> vreg_exp_sum_brc;
            LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2); // 加载当前块的局部expsum
            Mul(first_loop_sum, vreg_exp_max, first_loop_sum, preg_all);//通过更新因子，计算expsum
            Add(vreg_exp_sum, first_loop_sum, vreg_exp_sum_brc, preg_all);//完成累加，从而更新expsum
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_exp_sum, preg_all);
            // pscale update
            RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
            RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
            Duplicate(vreg_p_scale, static_cast<float>(1.0f)); // 初始化pscale为1.0
            Mul(vreg_p_scale, vreg_exp_max, vreg_p_scale, preg_all); // 1.0 * 更新因子，放回到vreg_p_scale里，感觉有点多余了，其实就是exp_max
            Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_p_scale, preg_all); // fp32--bf16，四字节里的前2字节
            Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_0, vreg_p_scale_bf16_0, preg_all_b16); // bf16---fp8，两字节里的首字节
            Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_p_scale, preg_all); // fp32--bf16，四字节里的后2字节
            Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_all_b16); // bf16---fp8，两字节里的首字节
            // 经过or之后，就是四字节里第一个字节和第三个字节有数据
            Or((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0,
                (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, preg_all_b8);
            Duplicate(vreg_p_scale_f8e8m0_1, 0x7f, preg_all_b8); // 填充1.0
            // 调用交织的算法，两个src，两个dst，按照奇偶分别往两个dst寄存器搬出，dst0存的是两个src的奇数，dst1存的是两个src的偶数
            // 最终呈现效果：dst0存了256个数，前128个是有效位（更新因子，这128位里面两两是一样的，每两个相同的代表一行的更新因子），后128位是1;dst1其实没啥用就可以废弃掉
            DeInterleave(vreg_p_scale_f8e8m0_dst0, vreg_p_scale_f8e8m0_dst1, vreg_p_scale_f8e8m0_0, vreg_p_scale_f8e8m0_1);
            //连续对齐搬出，
            StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                                    vreg_p_scale_f8e8m0_dst0, preg_all_b8);

        } else {
            StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            RegTensor<float> first_loop_sum;
            RegTensor<float> first_loop_sum_v2;

            LoadAlign(first_loop_sum, inExpSumUb); // 加载旧的expsum
            LoadAlign(first_loop_sum_v2, inExpSumUb + 64); // 加载旧的expsum,128的后半段
            
            RegTensor<float> vreg_exp_sum_brc;
            RegTensor<float> vreg_exp_sum_brc_v2;
            
            LoadAlign(vreg_exp_sum_brc, tmpExpSumUb2); // 加载当前块的局部expsum
            LoadAlign(vreg_exp_sum_brc_v2, tmpExpSumUb2 + 64); // 加载当前块的局部expsum，128的后半段

            Mul(first_loop_sum, vreg_exp_max, first_loop_sum, preg_all);//通过更新因子，计算expsum
            Mul(first_loop_sum_v2, vreg_exp_max_v2, first_loop_sum_v2, preg_all);//通过更新因子，计算expsum

            Add(vreg_exp_sum, first_loop_sum, vreg_exp_sum_brc, preg_all);//完成累加，从而更新expsum
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb, vreg_exp_sum, preg_all);

            Add(vreg_exp_sum, first_loop_sum_v2, vreg_exp_sum_brc_v2, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)expSumUb + 64, vreg_exp_sum, preg_all);
            // pscale update
            RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
            RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
            // 前64行的cast，用的都是vreg_exp_max
            Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_exp_max, preg_all); // fp32--bf16，四字节里的前2字节
            Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_0, vreg_p_scale_bf16_0, preg_all_b16); // bf16---fp8，两字节里的首字节
            Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_exp_max, preg_all); // fp32--bf16，四字节里的后2字节
            Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_all_b16); // bf16---fp8，两字节里的首字节
            // 经过or之后，就是四字节里第一个字节和第三个字节有数据，存放到vreg_p_scale_f8e8m0_0里
            Or((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0,
                (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, preg_all_b8);
            // 后64行的cast，用的都是vreg_exp_max_v2
            Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_v2_0, vreg_exp_max_v2, preg_all); // fp32--bf16，四字节里的前2字节
            Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_v2_0, vreg_p_scale_bf16_v2_0, preg_all_b16); // bf16---fp8，两字节里的首字节
            Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_v2_1, vreg_exp_max_v2, preg_all); // fp32--bf16，四字节里的后2字节
            Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_v2_1, vreg_p_scale_bf16_v2_1, preg_all_b16); // bf16---fp8，两字节里的首字节
            // 经过or之后，就是四字节里第一个字节和第三个字节有数据，存放到vreg_p_scale_f8e8m0_v2_0里
            Or((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_v2_0, (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_v2_0,
                (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_v2_1, preg_all_b8);
            // 交织，将vreg_p_scale_f8e8m0_0和vreg_p_scale_f8e8m0_v2_0中的有效数据，紧密排布且连续排布，存放在vreg_p_scale_f8e8m0_dst0
            // 此时的vreg_p_scale_f8e8m0_dst0就是256个数据，前128个数据对应前64行，后128个数据对应后64行
            // vreg_p_scale_f8e8m0_dst1是无效数据，不需要关注
            DeInterleave(vreg_p_scale_f8e8m0_dst0, vreg_p_scale_f8e8m0_dst1, vreg_p_scale_f8e8m0_0, vreg_p_scale_f8e8m0_v2_0);
            // 需要填充subloop1所需要的pscale（256个全1）
            Duplicate(vreg_p_scale_f8e8m0_v2_1, 0x7f, preg_all_b8); // 填充1.0
            // 连续对齐搬出，将subloop0对应的256个有效数据搬运到pScaleSubLoop0前256B中
            StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
                                                                    vreg_p_scale_f8e8m0_dst0, preg_all_b8);
            // 连续对齐搬出，将subloop1对应的256个全1数据搬运到pScaleSubLoop0后256B中
            StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0 + 256),
                                                                    vreg_p_scale_f8e8m0_v2_1, preg_all_b8);
        }
        
    }
}

// no update, originN == 128
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false>
__aicore__ inline void ProcessVec1NoUpdateImpl128Mxfp8Fullquant(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, 
    const LocalTensor<float>& preLoopMaxTensor, const LocalTensor<float>& preLoopSumTensor, const LocalTensor<float>& firstLoopSumTensor, uint32_t subLoop,
    const uint16_t m, const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb, const LocalTensor<T>& queryScaleUb = LocalTensor<T>(), const float deSCaleKValue = 1.0f, const float pScale = 1.0f)
{
    float divValue = 1.0f / keepProb;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = s1BaseSize  | 0x1;
    const uint32_t repeatStride = 1;
    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2 * x_expUb = nullptr;
    if constexpr (IsSameType<T2, float>::value) {
        x_expUb = expUb + ((s1BaseSize ) + 1) * (s2BaseSize >> 1);
    }
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * qScaleUb = (__ubuf__ T*)queryScaleUb.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();
    __ubuf__ uint32_t * maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint32_t * maskUbUnroll = (__ubuf__ uint32_t*)(maskTensor.GetPhyAddr() + floatRepSize);
    __ubuf__ uint32_t *  dropMaskUb = (__ubuf__ uint32_t*)dropTensor.GetPhyAddr();
    const float dScale = scale * dScaleQK;
    // 新增：
    // subloop新增：
    __ubuf__ T * inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 128;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 128;
    // 在noupdate没有用到
    __ubuf__ float *pre_loop_max = (__ubuf__ T*)preLoopMaxTensor.GetPhyAddr();
    __ubuf__ float* pre_loop_sum = (__ubuf__ T*) preLoopSumTensor.GetPhyAddr();
    __ubuf__ float* first_loop_sum = (__ubuf__ T*) firstLoopSumTensor.GetPhyAddr();
    // pscale 新增
    __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0Tensor.GetPhyAddr();

    ProcessVec1NoUpdateImpl128Mxfp8FullquantVF<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop, isMlaSgd>(
        expUb, x_expUb, pseUb, expSumUb, maxUb, maxUbStart, srcUb, qScaleUb, indexesUb, maskUb, maskUbUnroll, dropMaskUb, pScaleSubLoop0Ub,
        pre_loop_max, pre_loop_sum, first_loop_sum, subLoop, inExpSumUb, tmpExpSumUb, tmpExpSumUb2, expMaxUb, inMaxUb, tmpMaxUb, tmpMaxUb2,
        divValue, blockStride, repeatStride, dScale, m, pseStride, slopes, posShift, scale, dScaleQK, minValue, deSCaleKValue, pScale);
}
} // namespace
}
#endif // VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_H
