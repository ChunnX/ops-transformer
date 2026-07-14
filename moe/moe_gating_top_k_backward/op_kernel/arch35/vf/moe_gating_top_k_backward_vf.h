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
 * \file moe_gating_top_k_backward_vf.h
 * \brief VF functions for MoeGatingTopKBackward on Arch35
 */

#ifndef MOE_GATING_TOP_K_BACKWARD_VF_H
#define MOE_GATING_TOP_K_BACKWARD_VF_H

#include "kernel_operator.h"

namespace MoeGatingTopKBackwardNs {
using namespace AscendC;

constexpr uint32_t VL_FP32_VF = VECTOR_REG_WIDTH / sizeof(float);

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

__simd_vf__ void SigmoidGradFP32VF(__ubuf__ float *xNormAddr, __ubuf__ float *gradNormXAddr, __ubuf__ float *gradXAddr,
                                   uint32_t totalElements, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<float> xReg, gNormReg, tmpReg, dstReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float>();

    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t remaining = totalElements - i * oneRepeatSize;
        uint32_t chunkElems = (remaining > oneRepeatSize) ? oneRepeatSize : remaining;
        preg = MicroAPI::UpdateMask<float>(chunkElems);
        MicroAPI::DataCopy(xReg, xNormAddr + i * oneRepeatSize);
        MicroAPI::DataCopy(gNormReg, gradNormXAddr + i * oneRepeatSize);
        MicroAPI::Muls(tmpReg, xReg, static_cast<float>(-1), preg);
        MicroAPI::Adds(tmpReg, tmpReg, static_cast<float>(1), preg);
        MicroAPI::Mul(tmpReg, tmpReg, xReg, preg);
        MicroAPI::Mul(dstReg, tmpReg, gNormReg, preg);
        MicroAPI::DataCopy(gradXAddr + i * oneRepeatSize, dstReg, preg);
    }
}

template <typename T>
__simd_vf__ void SigmoidGradHalfVF(__ubuf__ float *xNormAddr, __ubuf__ float *gradNormXAddr, __ubuf__ T *gradXAddr,
                                   uint32_t totalElements, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<float> xReg, gNormReg, tmpReg, dstReg;
    MicroAPI::RegTensor<T> outReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float>();

    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t remaining = totalElements - i * oneRepeatSize;
        uint32_t chunkElems = (remaining > oneRepeatSize) ? oneRepeatSize : remaining;
        preg = MicroAPI::UpdateMask<float>(chunkElems);
        MicroAPI::DataCopy(xReg, xNormAddr + i * oneRepeatSize);
        MicroAPI::DataCopy(gNormReg, gradNormXAddr + i * oneRepeatSize);
        MicroAPI::Muls(tmpReg, xReg, static_cast<float>(-1), preg);
        MicroAPI::Adds(tmpReg, tmpReg, static_cast<float>(1), preg);
        MicroAPI::Mul(tmpReg, tmpReg, xReg, preg);
        MicroAPI::Mul(dstReg, tmpReg, gNormReg, preg);
        MicroAPI::Cast<T, float, castTraitB322B16>(outReg, dstReg, preg);
        MicroAPI::DataCopy(gradXAddr + i * oneRepeatSize, outReg, preg);
    }
}

} // namespace MoeGatingTopKBackwardNs
#endif // MOE_GATING_TOP_K_BACKWARD_VF_H
