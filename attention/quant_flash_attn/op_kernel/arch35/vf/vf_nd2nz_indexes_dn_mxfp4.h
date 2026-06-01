/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
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
 * \file vf_nd2nz_indexes_dn_mxfp4.h
 * \brief
 */

#ifndef ND2NZ_INDEXES_DN_MXFP4_H_
#define ND2NZ_INDEXES_DN_MXFP4_H_
#include "kernel_tensor.h"
namespace Mxfp4Api {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

constexpr uint8_t NUM_128 = static_cast<uint8_t>(128);
constexpr int16_t NUM_2 = static_cast<int16_t>(2);
constexpr int8_t indexSubLength = static_cast<int8_t>(32);

__simd_vf__ void  inline  ProcessIndexesVF(__ubuf__ uint8_t *index_nd2xz)
{
    // 寄存器定义
    RegTensor<uint8_t> index_reg, index_reg_1, index_reg_2;
    MaskReg mask_32 = CreateMask<int8_t, MaskPattern::VL32>();

    Arange((RegTensor<int8_t> &)index_reg, 0);

    ShiftLefts(index_reg, index_reg, NUM_2, mask_32);
    for (uint16_t i = 0; i < 4; ++i) {
        Adds(index_reg_1, index_reg, i, mask_32);
        Adds(index_reg_2, index_reg_1, NUM_128, mask_32);
        StoreAlign(index_nd2xz + indexSubLength * i, (RegTensor<uint8_t> &)index_reg_1, mask_32);
        StoreAlign(index_nd2xz + indexSubLength * i + NUM_128, (RegTensor<uint8_t> &)index_reg_2, mask_32);
    }
}

__aicore__ inline void InitIndexes(LocalTensor<uint8_t> &nd2nzIndexes)
{
    __ubuf__ uint8_t * index_nd2xz = (__ubuf__ uint8_t*)nd2nzIndexes.GetPhyAddr();
    ProcessIndexesVF(index_nd2xz);
}

}
#endif  // ND2NZ_INDEXES_DN_MXFP4_H_