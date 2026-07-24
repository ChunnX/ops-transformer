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
 * \file dense_lightning_indexer_grad_kl_loss_sy_vf.h
 * \brief DAV_3510 RegBase Sy weight multiply helper.
 */
#ifndef DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SY_VF_H
#define DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SY_VF_H

#include "kernel_tensor.h"

namespace DenseDliGradKLLossArch35 {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t SY_FP32_REG_ELEMENTS = 64;

__simd_vf__ inline void ComputeSyWeightedHeadVF(
    __ubuf__ float *scoreUb, __ubuf__ float *weightUb, uint32_t rowCount,
    uint32_t rowSize, int32_t actualLenFirstRow, uint32_t n1IndexSize,
    uint32_t n1IndexIdx)
{
    RegTensor<float> scoreReg;
    RegTensor<float> weightReg;
    RegTensor<float> weightedReg;

    for (uint32_t rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
        int32_t validRowCount = actualLenFirstRow + static_cast<int32_t>(rowIdx);
        validRowCount = validRowCount < 0 ? 0 : validRowCount;
        validRowCount = validRowCount > static_cast<int32_t>(S2_BASE_STEP) ?
                            static_cast<int32_t>(S2_BASE_STEP) :
                            validRowCount;
        if (validRowCount == 0) {
            continue;
        }

        LoadAlign<float, LoadDist::DIST_BRC_B32>(
            weightReg, weightUb + rowIdx * n1IndexSize + n1IndexIdx);
        uint32_t rowOffset = rowIdx * rowSize;
        for (uint32_t columnOffset = 0;
             columnOffset < static_cast<uint32_t>(validRowCount);
             columnOffset += SY_FP32_REG_ELEMENTS) {
            uint32_t validCount =
                static_cast<uint32_t>(validRowCount) - columnOffset < SY_FP32_REG_ELEMENTS ?
                    static_cast<uint32_t>(validRowCount) - columnOffset :
                    SY_FP32_REG_ELEMENTS;
            MaskReg validMask = UpdateMask<float>(validCount);
            uint32_t offset = rowOffset + columnOffset;
            LoadAlign(scoreReg, scoreUb + offset);
            Mul(weightedReg, scoreReg, weightReg, validMask);
            StoreAlign(scoreUb + offset, weightedReg, validMask);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__aicore__ inline void ComputeSyWeightedHead(
    const LocalTensor<float> &scoreTensor, const LocalTensor<float> &weightTensor,
    uint32_t rowCount, uint32_t rowSize, int32_t actualLenFirstRow,
    uint32_t n1IndexSize, uint32_t n1IndexIdx)
{
    __ubuf__ float *scoreUb =
        reinterpret_cast<__ubuf__ float *>(scoreTensor.GetPhyAddr());
    __ubuf__ float *weightUb =
        reinterpret_cast<__ubuf__ float *>(weightTensor.GetPhyAddr());
    ComputeSyWeightedHeadVF(
        scoreUb, weightUb, rowCount, rowSize, actualLenFirstRow, n1IndexSize,
        n1IndexIdx);
}
} // namespace DenseDliGradKLLossArch35

#endif // DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SY_VF_H
