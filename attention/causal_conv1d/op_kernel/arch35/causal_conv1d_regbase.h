/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_regbase.h
 * \brief
 */
#ifndef CAUSAL_CONV1D_REGBASE_H
#define CAUSAL_CONV1D_REGBASE_H

namespace NsCausalConv1d {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);

constexpr CastTrait CastTraitB16ToB32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr CastTrait CastTraitB32ToB16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                         RoundMode::CAST_RINT};

template <typename T, bool kHasBias>
__aicore__ inline void WeightBiasCast(LocalTensor<T> weightT, LocalTensor<T> biasT, LocalTensor<float> weightF,
                                      LocalTensor<float> biasF, int32_t width, int32_t baseDim)
{
    uint32_t origDataCount = static_cast<uint32_t>(baseDim);
    uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(baseDim, V_LENGTH));
    uint16_t rowCount = static_cast<uint16_t>(width);
    __ubuf__ T *weightTAddr = (__ubuf__ T *)weightT.GetPhyAddr();
    __ubuf__ float *weightFAddr = (__ubuf__ float *)weightF.GetPhyAddr();
    __ubuf__ T *biasTAddr = nullptr;
    __ubuf__ float *biasFAddr = nullptr;
    if constexpr (kHasBias) {
        biasTAddr = (__ubuf__ T *)biasT.GetPhyAddr();
        biasFAddr = (__ubuf__ float *)biasF.GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        RegTensor<T> src;
        RegTensor<float> dst;
        MaskReg pregLoop;
        uint32_t curCount;
        for (uint16_t j = 0; j < rowCount; ++j) {
            curCount = origDataCount;
            for (uint16_t k = 0; k < colLoopTimes; ++k) {
                pregLoop = UpdateMask<float>(curCount);
                DataCopy<T, LoadDist::DIST_UNPACK_B16>(src, weightTAddr + j * baseDim * 2 + baseDim + k * V_LENGTH);
                Cast<float, T, CastTraitB16ToB32>(dst, src, pregLoop);
                DataCopy(weightFAddr + j * baseDim + k * V_LENGTH, dst, pregLoop);
            }
        }
        if constexpr (kHasBias) {
            curCount = origDataCount;
            for (uint16_t k = 0; k < colLoopTimes; ++k) {
                pregLoop = UpdateMask<float>(curCount);
                DataCopy<T, LoadDist::DIST_UNPACK_B16>(src, biasTAddr + baseDim + k * V_LENGTH);
                Cast<float, T, CastTraitB16ToB32>(dst, src, pregLoop);
                DataCopy(biasFAddr + k * V_LENGTH, dst, pregLoop);
            }
        }
    }
}

} // namespace NsCausalConv1d

#endif // CAUSAL_CONV1D_REGBASE_H
