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
 * \file vf_softmax_unaligned64_update.h
 * \brief
 */
#ifndef VF_SOFTMAX_UNALIGNED64_UPDATE_H
#define VF_SOFTMAX_UNALIGNED64_UPDATE_H

#include "vf_softmax_const.h"
#include "../../../fa_kernel_public.h"

namespace FaVectorApi {
#ifndef __CCE_KT_TEST__
template <typename T, typename T2, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128, bool hasAttn = 0>
__simd_vf__ void ProcessVec1UpdateImpl64VF(
    __ubuf__ T2 * expUb, __ubuf__ T * maxUb, __ubuf__ T * srcUb, __ubuf__ T * expMaxUb,
    __ubuf__ T * inMaxUb, __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpMaxUb, __ubuf__ T * tmpMaxUb2,
    __ubuf__ uint8_t * indexesUb, __ubuf__ uint32_t * maskUb,
    const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride,
    const float dScale, uint32_t pltOriginalN, uint32_t pltSrcN, uint32_t pltSrcN16, const uint16_t m,
    const T scale, const T minValue)
{
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_sel;
    RegTensor<float> vreg_input_x;
    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_max_new;
    RegTensor<float> vreg_exp;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_max;

    // bfloat16_t
    RegTensor<bfloat16_t> vreg_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_exp_bf16;
    RegTensor<bfloat16_t> vreg_dst_even_bf16;
    RegTensor<bfloat16_t> vreg_dst_odd_bf16;
    // half
    RegTensor<half> vreg_exp_even_f16;
    RegTensor<half> vreg_exp_f16;
    RegTensor<half> vreg_dst_even_f16;
    RegTensor<half> vreg_dst_odd_f16;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    MaskReg preg_src_n = UpdateMask<T>(pltSrcN);
    MaskReg preg_src_n_b16 = UpdateMask<uint16_t>(pltSrcN16);
    MaskReg preg_ori_src_n = UpdateMask<T>(pltOriginalN);
    MaskReg preg_compare;

    if constexpr (hasAttn == 1) {
        Duplicate(vreg_min, minValue);
    }
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        Muls(vreg_input_x, vreg_input_x, dScale, preg_ori_src_n);
        if constexpr (hasAttn == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                preg_compare, (__ubuf__ uint32_t *&)maskUb, nPadding);
            Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_src_n);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_sel, preg_ori_src_n);
        } else {
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_src_n);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_input_x, preg_ori_src_n);
        }
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1);
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
    LoadAlign(vreg_in_max, inMaxUb);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_input_max, tmpMaxUb2);
    Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg_max, tmpMaxUb2 + i);
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        ExpSub(vreg_exp, vreg_input_x, vreg_max, preg_ori_src_n);

        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_sum, vreg_exp, preg_ori_src_n);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum, ureg_exp_sum, 1);

        if constexpr (IsSameType<T2, float>::value) {
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp, blockStride, repeatStride, preg_all);
        } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_bf16, vreg_exp, preg_all_b16);
            DeInterleave(vreg_dst_even_bf16, vreg_dst_odd_bf16,
                    vreg_exp_bf16, vreg_exp_bf16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_dst_even_bf16, blockStride, repeatStride, preg_src_n_b16);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
            RegTensor<fp8_e5m2_t> vreg_exp_f8e5m2;
            RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
            MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_f8e5m2, vreg_exp, preg_all);
            LoadAlign(vreg_exp_merge_f8e5m2_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e5m2, vreg_exp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
            RegTensor<fp8_e4m3fn_t> vreg_exp_f8e4m3;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
            MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_f8e4m3, vreg_exp, preg_all);
            LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e4m3, vreg_exp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, int8_t>::value) {
            RegTensor<int8_t> vreg_exp_merge_tmp_int8;
            RegTensor<int8_t> vreg_exp_merge_int8;
            RegTensor<half> vreg_cast_res;
            MaskReg preg_all_f16 = CreateMask<half, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Cast<half, T, castTrait0>(vreg_exp_f16, vreg_exp, preg_all);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t>&)vreg_cast_res,
                    (MicroAPI::RegTensor<uint32_t>&)vreg_exp_f16);     
            Cast<T2, half, castTrait0>(vreg_exp_merge_tmp_int8, vreg_cast_res, preg_all_f16);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vreg_exp_merge_int8,
                    (MicroAPI::RegTensor<uint16_t>&)vreg_exp_merge_tmp_int8);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ int8_t *&) expUb), vreg_exp_merge_int8, blockStride, repeatStride, preg_all_b8_128);                
        } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
            RegTensor<hifloat8_t> vreg_exp_hif8;
            RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
            RegTensor<hifloat8_t> vreg_exp_merge_hif8;
            MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitZero>(vreg_exp_hif8, vreg_exp, preg_all);
            LoadAlign(vreg_exp_merge_hif8_indexes, indexesUb);
            Gather(vreg_exp_merge_hif8, vreg_exp_hif8, vreg_exp_merge_hif8_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
        } else {
            Cast<T2, T, castTraitZero>(vreg_exp_f16, vreg_exp, preg_all_b16);
            DeInterleave(vreg_dst_even_f16, vreg_dst_odd_f16, vreg_exp_f16, vreg_exp_f16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_dst_even_f16, blockStride, repeatStride, preg_src_n_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);
}
// update, originN <= 64
template <typename T, typename T2, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128, bool hasAttn = 0>
__aicore__ inline void ProcessVec1UpdateImpl64(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m,
    const uint32_t originN, const T scale, const T minValue)
{
    const uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    uint32_t pltOriginalN = originN;
    uint32_t pltSrcN = s2BaseSize;
    uint32_t pltSrcN16 = s2BaseSize;
    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();

    __ubuf__ uint32_t * maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();

    ProcessVec1UpdateImpl64VF<T, T2, s1BaseSize, s2BaseSize, hasAttn>(
        expUb, maxUb, srcUb, expMaxUb, inMaxUb, tmpExpSumUb, tmpMaxUb, tmpMaxUb2,
        indexesUb, maskUb, nPadding, blockStride, repeatStride, scale, pltOriginalN, pltSrcN,
        pltSrcN16, m, scale, minValue);
}
#endif
} // namespace

#endif // VF_SOFTMAX_UNALIGNED64_UPDATE_H
