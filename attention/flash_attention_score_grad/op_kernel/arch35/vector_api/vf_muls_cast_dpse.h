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
 * \file vf_muls_cast_dpse.h
 */
#ifndef VF_MULS_CAST_DPSE_H
#define VF_MULS_CAST_DPSE_H

#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
constexpr static MicroAPI::CastTrait DPSE_CAST_TRAIT_B32_TO_B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename T, uint32_t dataSize>
__simd_vf__ inline void MulsCastDpseVF(uint64_t dstLocalInt, uint64_t srcLocalInt, float scaleValue)
{
    using namespace MicroAPI;
    RegTensor<float> srcReg;
    RegTensor<float> scaledReg;
    RegTensor<T> dstReg;
    uint32_t sreg = dataSize;
    constexpr uint32_t regSize = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t repeatTimes = (dataSize + regSize - 1) / regSize;

#pragma unroll
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        MaskReg mask = UpdateMask<float>(sreg);
        LoadAlign(srcReg, (__ubuf__ float *&)srcLocalInt + i * regSize);
        Muls(scaledReg, srcReg, scaleValue, mask);
        Cast<T, float, DPSE_CAST_TRAIT_B32_TO_B16>(dstReg, scaledReg, mask);
        StoreAlign<T, StoreDist::DIST_PACK_B32>((__ubuf__ T *&)dstLocalInt + i * regSize, dstReg, mask);
    }
}
#else
template <typename T, uint32_t dataSize>
__aicore__ inline void MulsCastDpseVF(uint64_t dstLocalInt, uint64_t srcLocalInt, float scaleValue)
{}
#endif
} // namespace AscendC

#endif // VF_MULS_CAST_DPSE_H
