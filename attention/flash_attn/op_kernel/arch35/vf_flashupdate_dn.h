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
 * \file vf_flashupdate_dn.h
 * \brief DN Vec2 update and output vector functions.
 */
#ifndef FLASH_ATTN_VF_FLASHUPDATE_DN_H_
#define FLASH_ATTN_VF_FLASHUPDATE_DN_H_

#include "kernel_tensor.h"

namespace FaVectorApi {

// PV is DN: each vector holds64 query rows for one value column.
// GetPhyAddr is __aicore__ only; VF uses the UB address carried by LocalTensor.
template <bool last>
__simd_vf__ inline void FlashUpdateDnRows64VF(LocalTensor<float> accumUb, LocalTensor<float> curUb,
                                              LocalTensor<float> factorUb, uint16_t columns)
{
    __ubuf__ float *accum = (__ubuf__ float *)static_cast<uint64_t>(accumUb.address_.bufferAddr);
    __ubuf__ float *cur = (__ubuf__ float *)static_cast<uint64_t>(curUb.address_.bufferAddr);
    __ubuf__ float *factor = (__ubuf__ float *)static_cast<uint64_t>(factorUb.address_.bufferAddr);
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> scale, p0, p1, p2, p3, c0, c1, c2, c3;
    LoadAlign(scale, factor);
    for (uint16_t i = 0; i < columns / 4; ++i) {
        LoadAlign(p0, accum + (i * 4 + 0) * 64);
        LoadAlign(c0, cur + (i * 4 + 0) * 64);
        LoadAlign(p1, accum + (i * 4 + 1) * 64);
        LoadAlign(c1, cur + (i * 4 + 1) * 64);
        LoadAlign(p2, accum + (i * 4 + 2) * 64);
        LoadAlign(c2, cur + (i * 4 + 2) * 64);
        LoadAlign(p3, accum + (i * 4 + 3) * 64);
        LoadAlign(c3, cur + (i * 4 + 3) * 64);
        if constexpr (last) {
            Mul(p0, p0, scale, mask);
            Add(c0, c0, p0, mask);
            Mul(p1, p1, scale, mask);
            Add(c1, c1, p1, mask);
            Mul(p2, p2, scale, mask);
            Add(c2, c2, p2, mask);
            Mul(p3, p3, scale, mask);
            Add(c3, c3, p3, mask);
        } else {
            MulAddDst(c0, p0, scale, mask);
            MulAddDst(c1, p1, scale, mask);
            MulAddDst(c2, p2, scale, mask);
            MulAddDst(c3, p3, scale, mask);
        }
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(accum + (i * 4 + 0) * 64, c0, mask);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(accum + (i * 4 + 1) * 64, c1, mask);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(accum + (i * 4 + 2) * 64, c2, mask);
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(accum + (i * 4 + 3) * 64, c3, mask);
    }
    for (uint16_t col = columns / 4 * 4; col < columns; ++col) {
        LoadAlign(p0, accum + col * 64);
        LoadAlign(c0, cur + col * 64);
        if constexpr (last) {
            Mul(p0, p0, scale, mask);
            Add(c0, c0, p0, mask);
        } else {
            MulAddDst(c0, p0, scale, mask);
        }
        StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(accum + col * 64, c0, mask);
    }
}
// Split-KV workspace remains FP32 row-major for the existing reduction block.
__simd_vf__ inline void FlashOutputDnWorkspaceVF(LocalTensor<float> outUb, LocalTensor<float> accumUb,
                                                 LocalTensor<float> sumUb, uint16_t rows, uint16_t columns,
                                                 uint16_t stride)
{
    __ubuf__ float *out = (__ubuf__ float *)static_cast<uint64_t>(outUb.address_.bufferAddr);
    __ubuf__ float *accum = (__ubuf__ float *)static_cast<uint64_t>(accumUb.address_.bufferAddr);
    __ubuf__ float *sum = (__ubuf__ float *)static_cast<uint64_t>(sumUb.address_.bufferAddr);
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<uint32_t> lane, base, index, rowIndex;
    RegTensor<float> value, denominator;
    Arange((RegTensor<int32_t> &)lane, 0);
    Muls(base, lane, 64U, all);
    for (uint16_t row = 0; row < rows; ++row) {
        Duplicate(rowIndex, static_cast<uint32_t>(row));
        Gather(denominator, sum, rowIndex, all);
        for (uint16_t col = 0; col < columns; col += 64) {
            uint32_t count = columns - col;
            MaskReg mask = UpdateMask<float>(count);
            Adds(index, base, static_cast<uint32_t>(col) * 64U + row, all);
            Gather(value, accum, index, mask);
            Div(value, value, denominator, mask);
            StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(out + row * stride + col, value, mask);
        }
    }
}
static constexpr Reg::CastTrait castDnOutput = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                                RoundMode::CAST_ROUND};
template <typename OUT_T>
__simd_vf__ inline void FlashPackOutputDnRows64VF(LocalTensor<OUT_T> outUb, LocalTensor<float> accumUb,
                                                  LocalTensor<float> sumUb, uint16_t columns)
{
    __ubuf__ OUT_T *out = (__ubuf__ OUT_T *)static_cast<uint64_t>(outUb.address_.bufferAddr);
    __ubuf__ float *accum = (__ubuf__ float *)static_cast<uint64_t>(accumUb.address_.bufferAddr);
    __ubuf__ float *sum = (__ubuf__ float *)static_cast<uint64_t>(sumUb.address_.bufferAddr);
    MaskReg mf = CreateMask<float, MaskPattern::ALL>();
    MaskReg half = CreateMask<uint16_t, MaskPattern::VL64>();
    RegTensor<float> denominator, x;
    RegTensor<OUT_T> cast, packed;
    LoadAlign(denominator, sum);
    for (uint16_t col = 0; col < columns; ++col) {
        LoadAlign(x, accum + col * 64);
        Div(x, x, denominator, mf);
        Cast<OUT_T, float, castDnOutput>(cast, x, mf);
        Pack<uint16_t, uint32_t, HighLowPart::LOWEST>((RegTensor<uint16_t> &)packed, (RegTensor<uint32_t> &)cast);
        StoreAlign<OUT_T, Reg::StoreDist::DIST_NORM_B16>(out + col * 64, packed, half);
    }
}

} // namespace FaVectorApi

#endif // FLASH_ATTN_VF_FLASHUPDATE_DN_H_
