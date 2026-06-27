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
 * \file vf_rope.h
 * \brief VF implementation for mla_prolog rope path (interleave-half mode).
 */

#ifndef VF_ROPE_H
#define VF_ROPE_H

#include "kernel_tensor.h"
#include "vf_comm.h"

namespace MlaProlog {

template <typename O, typename C>
__simd_vf__ inline void RopeVFImpl(__ubuf__ O *outputUb, __ubuf__ C *inputUb, __ubuf__ C *sinUb, __ubuf__ C *cosUb,
                                   uint32_t row, uint64_t srcStride, uint64_t dstStride, uint64_t sinCosStride)
{
    MicroAPI::RegTensor<C> vregX;
    MicroAPI::RegTensor<C> vregSin;
    MicroAPI::RegTensor<C> vregCos;
    MicroAPI::RegTensor<C> vregEven;
    MicroAPI::RegTensor<C> vregOdd;
    MicroAPI::RegTensor<C> vregResLow;
    MicroAPI::RegTensor<C> vregResHigh;
    MicroAPI::RegTensor<C> vregRes;
    MicroAPI::RegTensor<O> vregResCast;

    uint32_t halfReg = ROPE_VF_COL / 2;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<C, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllCast = MicroAPI::CreateMask<O, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskLowHalf = MicroAPI::CreateMask<C, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg maskHighHalf;
    MicroAPI::Not(maskHighHalf, maskLowHalf, maskAll);

    for (uint16_t i = 0; i < row; ++i) {
        __ubuf__ C *curXUb = inputUb + i * srcStride;
        __ubuf__ O *curResUb = outputUb + i * dstStride;
        __ubuf__ C *curSinUb = sinUb + i * sinCosStride;
        __ubuf__ C *curCosUb = cosUb + i * sinCosStride;

        MicroAPI::LoadAlign(vregX, curXUb);
        MicroAPI::LoadAlign(vregSin, curSinUb);
        MicroAPI::LoadAlign(vregCos, curCosUb);

        // vregEven = [evens(0..31), evens(32..63)]
        // vregOdd = [odds(0..31),  odds(32..63)]
        MicroAPI::DeInterleave<C>(vregEven, vregOdd, vregX, vregX);

        // Part1 low  = cos * evens,        Part1 high preserved
        MicroAPI::Mul(vregResLow, vregCos, vregEven, maskLowHalf);
        // Part1 high = sin * evens,         Part1 low  preserved
        MicroAPI::Mul(vregResHigh, vregSin, vregEven, maskHighHalf);
        MicroAPI::Add(vregRes, vregResLow, vregResHigh, maskAll);
        // Part2 low  = sin(-) * odds,      Part2 high preserved
        MicroAPI::Mul(vregResLow, vregSin, vregOdd, maskLowHalf);
        // Part2 high = cos * odds,          Part2 low  preserved
        MicroAPI::Mul(vregResHigh, vregCos, vregOdd, maskHighHalf);

        // Part1 = [cos_l*even + sin_l*odd, cos_u*odd + sin_u*even] = [y_lower, y_upper]
        MicroAPI::Add(vregRes, vregRes, vregResHigh, maskAll);
        MicroAPI::Add(vregRes, vregRes, vregResLow, maskAll);

        if constexpr (std::is_same<C, O>::value) {
            MicroAPI::StoreAlign(curResUb, vregRes, maskAll);
        } else {
            MicroAPI::Cast<O, C, castTraitB322B16>(vregResCast, vregRes, maskAllCast);
            MicroAPI::DataCopy<O, MicroAPI::StoreDist::DIST_PACK_B32>(curResUb, vregResCast, maskAllCast);
        }
    }
}


// col == 64
template <typename O, typename C>
__aicore__ inline void RotaryPosEmbVF(const LocalTensor<O> &outputLocal, const LocalTensor<C> &inputLocal,
                                      const LocalTensor<C> &cosLocal, const LocalTensor<C> &sinLocal, uint32_t row,
                                      uint64_t srcStride, uint64_t dstStride, uint64_t sinCosStride)
{
    __ubuf__ C *inputUb = (__ubuf__ C *)inputLocal.GetPhyAddr();
    __ubuf__ C *sinUb = (__ubuf__ C *)sinLocal.GetPhyAddr();
    __ubuf__ C *cosUb = (__ubuf__ C *)cosLocal.GetPhyAddr();
    __ubuf__ O *outputUb = (__ubuf__ O *)outputLocal.GetPhyAddr();

    RopeVFImpl(outputUb, inputUb, sinUb, cosUb, row, srcStride, dstStride, sinCosStride);
}
} // namespace MlaProlog

#endif // VF_ROPE_H
