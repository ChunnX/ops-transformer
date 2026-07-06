/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
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

using namespace regbaseutil;

namespace FaVectorApi {
// no update, originN == 128
template <typename T, typename T2, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128, bool hasAtten = 0,
          bool isMlaSgd = false>
__simd_vf__ void
ProcessVec1NoUpdateImpl128VF(__ubuf__ T2 *expUb, __ubuf__ T2 *x_expUb, __ubuf__ T *expSumUb, __ubuf__ T *maxUb,
                             __ubuf__ T *maxUbStart, __ubuf__ T *srcUb, __ubuf__ T *qScaleUb,
                             __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb, __ubuf__ uint32_t *maskUbUnroll,
                             float divValue, const uint32_t blockStride, const uint32_t repeatStride,
                             const float dScale, const uint16_t m, const T scale, const float dScaleQK,
                             const T minValue, const float deSCaleKValue = 1.0f, const float pScale = 1.0f)
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
    RegTensor<float> vreg_alibi;
    RegTensor<float> vreg_alibi_unroll;
    RegTensor<float> vreg_sel_drop;
    RegTensor<float> vreg_sel_drop2;
    RegTensor<float> vreg_rowmax_p;
    RegTensor<float> vreg_scale_qk;
    RegTensor<float> vreg_sink_input;
    // bfloat16_t
    RegTensor<bfloat16_t> vreg_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_exp_odd_bf16;
    RegTensor<bfloat16_t> vreg_exp_bf16;
    // half
    RegTensor<half> vreg_exp_even_f16;
    RegTensor<half> vreg_exp_odd_f16;
    RegTensor<half> vreg_exp_f16;

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
    // pScale
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_all);

    if constexpr (hasAtten == 1) {
        Duplicate(vreg_min, minValue);
        if constexpr (isMlaSgd) {
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare, ((__ubuf__ uint32_t *)(maskUb)));
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare_unroll,
                                                                       ((__ubuf__ uint32_t *)(maskUbUnroll)));
        }
    }
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        LoadAlign(vreg_input_x_unroll, srcUb + floatRepSize + i * s2BaseSize);

        Muls(vreg_input_x, vreg_input_x, dScale, preg_all); // Muls(scale)
        Muls(vreg_input_x_unroll, vreg_input_x_unroll, dScale, preg_all);

        if constexpr (hasAtten == 1) {
            // atten mask
            if constexpr (!isMlaSgd) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare, (__ubuf__ uint32_t *&)maskUb, s2BaseSize);
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare_unroll, (__ubuf__ uint32_t *&)maskUbUnroll, s2BaseSize);
            }

            Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
            Select(vreg_sel_unroll, vreg_min, vreg_input_x_unroll, preg_compare_unroll);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel,
                                                              preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_sel_unroll, preg_all);
            Max(vreg_max_tmp, vreg_sel, vreg_sel_unroll, preg_all);
            Sub(vreg_max_tmp, vreg_max_tmp, vreg_ln_p_scale, preg_all);
        } else {
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x,
                                                              preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * s2BaseSize,
                                                              vreg_input_x_unroll, preg_all);
            Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll, preg_all);
            Sub(vreg_max_tmp, vreg_max_tmp, vreg_ln_p_scale, preg_all);
        }
        Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max, vreg_max_tmp,
                                                                                          preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max,
                                                                     1);
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)maxUb), ureg_max, 0);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // 以下 mask 在循环内不变，提取到循环外避免重复创建
    constexpr uint32_t maskLen = 128;
    MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
    MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
    for (uint16_t i = 0; i < m; ++i) {
        // maxUb is [S1, 1], BRC_B32 is reading one fp32 element and broadcast it to all 64 vreg element
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max_brc, maxUbStart + i);
        LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg_input_x, vreg_input_x_unroll, srcUb + i * s2BaseSize);
        ExpSub(vreg_exp_even, vreg_input_x, vreg_max_brc, preg_all);
        ExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max_brc, preg_all);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, preg_all);
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_exp_sum, vreg_exp_sum,
                                                                                          preg_all);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)expSumUb), vreg_exp_sum,
                                                                     ureg_exp_sum, 1);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
            RegTensor<fp8_e5m2_t> vreg_exp_even_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_odd_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_tmp_f8e5m2;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
            RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e5m2, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e5m2, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e5m2, (RegTensor<uint8_t> &)vreg_exp_even_f8e5m2,
               (RegTensor<uint8_t> &)vreg_exp_odd_f8e5m2, preg_all_b8);
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
            Cast<T2, T, castTraitRintZero>(vreg_exp_even_f8e4m3, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitRintTwo>(vreg_exp_odd_f8e4m3, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3,
               (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3, preg_all_b8);
            LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e4m3, vreg_exp_merge_tmp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
            RegTensor<hifloat8_t> vreg_exp_even_hif8;
            RegTensor<hifloat8_t> vreg_exp_odd_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_tmp_hif8;
            RegTensor<hifloat8_t> vreg_exp_merge_hif8;
            RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
            Cast<T2, T, castTraitZero>(vreg_exp_even_hif8, vreg_exp_even, preg_all);
            Cast<T2, T, castTraitTwo>(vreg_exp_odd_hif8, vreg_exp_odd, preg_all);
            Or((RegTensor<uint8_t> &)vreg_exp_merge_tmp_hif8, (RegTensor<uint8_t> &)vreg_exp_even_hif8,
               (RegTensor<uint8_t> &)vreg_exp_odd_hif8, preg_all_b8);
            LoadAlign(vreg_exp_merge_hif8_indexes, indexesUb);
            Gather(vreg_exp_merge_hif8, vreg_exp_merge_tmp_hif8, vreg_exp_merge_hif8_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
}

// no update, originN == 128
template <typename T, typename T2, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128, bool hasAtten = 0,
          bool isMlaSgd = false>
__aicore__ inline void ProcessVec1NoUpdateImpl128(
    const LocalTensor<T2> &dstTensor, const LocalTensor<uint8_t> &indexesTensor, const LocalTensor<T> &expSumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inExpSumTensor, const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &maskTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint16_t m, const uint32_t originN, const T scale,
    const float dScaleQK, const T minValue, float keepProb, const LocalTensor<T> &queryScaleUb = LocalTensor<T>(),
    const float deSCaleKValue = 1.0f, const float pScale = 1.0f)
{
    float divValue = 1.0f / keepProb;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    __ubuf__ T2 *expUb = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T2 *x_expUb = nullptr;
    __ubuf__ T *expSumUb = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *maxUbStart = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *qScaleUb = (__ubuf__ T *)queryScaleUb.GetPhyAddr();
    __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)indexesTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUbUnroll = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize);
    const float dScale = scale * dScaleQK;

    ProcessVec1NoUpdateImpl128VF<T, T2, s1BaseSize, s2BaseSize, hasAtten, isMlaSgd>(
        expUb, x_expUb, expSumUb, maxUb, maxUbStart, srcUb, qScaleUb, indexesUb, maskUb, maskUbUnroll, divValue,
        blockStride, repeatStride, dScale, m, scale, dScaleQK, minValue, deSCaleKValue, pScale);
}
} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_H
