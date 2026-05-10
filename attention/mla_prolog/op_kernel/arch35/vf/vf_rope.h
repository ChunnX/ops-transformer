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
 * \file vf_rope.mod.h
 * \brief VF implementation for mla_prolog rope path (interleave-half mode).
 */

#ifndef VF_ROPE_MOD_H
#define VF_ROPE_MOD_H

#include "kernel_tensor.h"

namespace MlaProlog {
constexpr uint32_t ROPE_VF_COL = 64;

__simd_vf__ inline void RopeVFImpl(__ubuf__ float *outputUb, __ubuf__ float *inputUb, __ubuf__ float *sinUb,
                                   __ubuf__ float *cosUb, uint16_t row, uint64_t srcStride, uint64_t dstStride, uint64_t sinCosStride)
{
    MicroAPI::RegTensor<float> vregX;
    MicroAPI::RegTensor<float> vregSin;
    MicroAPI::RegTensor<float> vregCos;
    MicroAPI::RegTensor<float> vregEven;
    MicroAPI::RegTensor<float> vregOdd;

    
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskL32 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    MicroAPI::MaskReg maskH32;

    uint32_t halfReg = ROPE_VF_COL / 2;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskLowHalf = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    MicroAPI::MaskReg maskHighHalf;
    MicroAPI::Not(maskHighHalf, maskLowHalf, maskAll);

    for (uint16_t i = 0; i < row; ++i) {
        __ubuf__ float *curXUb = inputUb + i * srcStride;
        __ubuf__ float *curResUb = outputUb + i * dstStride;
        __ubuf__ float *curSinUb = sinUb + i * sinCosStride;
        __ubuf__ float *curCosUb = cosUb + i * sinCosStride;

        MicroAPI::DataCopy(vregX, inputUb);
        MicroAPI::DataCopy(vregSin, curSinUb);
        MicroAPI::DataCopy(vregCos, curCosUb);

        // vregEven = [evens(0..31), evens(32..63)]
        // vregOdd = [odds(0..31),  odds(32..63)]
        MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregX, vregX);

        // Part1 low  = cos * evens,        Part1 high preserved
        MicroAPI::Mul(vregEven, vregCos, vregEven, maskLowHalf);
        // Part2 low  = sin(-) * odds,      Part2 high preserved
        MicroAPI::Mul(vregOdd, vregSin, vregOdd, maskLowHalf);
        // Part2 high = cos * odds,          Part2 low  preserved
        MicroAPI::Mul(vregOdd, vregCos, vregOdd, maskHighHalf);
        // Part1 high = sin * evens,         Part1 low  preserved
        MicroAPI::Mul(vregEven, vregSin, vregEven, maskHighHalf);

        // Part1 = [cos_l*even + sin_l*odd, cos_u*odd + sin_u*even] = [y_lower, y_upper]
        MicroAPI::Add(vregEven, vregEven, vregOdd, maskAll);

        MicroAPI::DataCopy(curResUb, vregEven);
    }
}

template <typename C>
__aicore__ inline void RotaryPosEmbVF(const LocalTensor<C> &outputLocal, const LocalTensor<C> &inputLocal,
                                      const LocalTensor<C> &cosLocal, const LocalTensor<C> &sinLocal,
                                      uint64_t row, uint64_t col, uint64_t srcStride, uint64_t dstStride, uint64_t sinCosStride)
{
    static_assert(std::is_same<C, float>::value, "RotaryPosEmbVF only supports float compute type.");
    __ubuf__ float *inputUb = (__ubuf__ float *)inputLocal.GetPhyAddr();
    __ubuf__ float *sinUb = (__ubuf__ float *)sinLocal.GetPhyAddr();
    __ubuf__ float *cosUb = (__ubuf__ float *)cosLocal.GetPhyAddr();
    __ubuf__ float *outputUb = (__ubuf__ float *)outputLocal.GetPhyAddr();

    RopeVFImpl(outputUb, inputUb, sinUb, cosUb, static_cast<uint16_t>(row), srcStride, dstStride, sinCosStride);
}
} // namespace MlaProlog

#endif // VF_ROPE_MOD_H
