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
 * \file vf_add.h
 * \brief
 */

#ifndef VF_ADD_H
#define VF_ADD_H

#include "kernel_operator.h"
#include <cstdint>
using namespace AscendC;
constexpr uint32_t FLOAT_REP_SIZE = 64;
constexpr uint32_t BTYEALIGNSIZE = 32;
constexpr uint32_t REGSIZE = 256;
constexpr uint32_t HALFCORED = 128;

template <typename T>
__simd_vf__ void AddContiguousVFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, uint32_t row, uint32_t col)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vregape0;
    uint32_t totalCnt = row * col;
    for (uint32_t offset = 0; offset < totalCnt; offset += FLOAT_REP_SIZE) {
        MicroAPI::MaskReg mask = MicroAPI::UpdatePredicate<T>(totalCnt);
        MicroAPI::LoadAlign(vreg0, inputAddr + offset);
        MicroAPI::LoadAlign(vregape0, apeAddr + offset);
        MicroAPI::Add(vreg0, vreg0, vregape0, mask);
        MicroAPI::StoreAlign(inputAddr + offset, vreg0, mask);
    }
}

template <typename T>
__simd_vf__ void AddFragmentedVFImpl(__ubuf__ T *inputAddr, __ubuf__ T *apeAddr, uint32_t row, uint32_t col, uint32_t actualCol)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vregape0;
    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(col);
    for (uint32_t rowIdx = 0; rowIdx < row; rowIdx++) {
        MicroAPI::LoadAlign(vreg0, inputAddr + rowIdx * actualCol);
        MicroAPI::LoadAlign(vregape0, apeAddr + rowIdx * actualCol);
        MicroAPI::Add(vreg0, vreg0, vregape0, mask);
        MicroAPI::StoreAlign(inputAddr + rowIdx * actualCol, vreg0, mask);
    }
}

/**
 * @brief AddVF 输入与apt相加
 * @param rightLocal 输出tensor []
 * @param leftLocal 输入tensor [row, col]
 * @param aptLocal apt输入tensor [r]
 * @param apeIdx ape起始位置
 * @param d  coff*d为ape的D轴大小
 * @param coreSplitD scoreleft大小，coff*coreSplitD为总大小
 * @param coreSplitS 核间d轴切分大小
 */
template <typename T>
__aicore__ inline void AddVF(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeLocal,
                             uint32_t row, uint32_t col, uint32_t actualCol)
{
    __ubuf__ T *scoreAddr = (__ubuf__ T *)scoreLocal.GetPhyAddr();
    __ubuf__ T *apeAddr = (__ubuf__ T *)apeLocal.GetPhyAddr();

    if (col == actualCol) {
        AddContiguousVFImpl<T>(scoreAddr, apeAddr, row, col);
    } else {
        AddFragmentedVFImpl<T>(scoreAddr, apeAddr, row, col, actualCol);
    }
}

#endif
