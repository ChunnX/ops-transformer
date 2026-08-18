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
 * \file vf_softmax_dn_cast_nz_mxfp8.h
 * \brief
 */

#ifndef VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_QS64_MXFP8_H
#define VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_QS64_MXFP8_H
// #include "kernel_tensor.h"

#include "vf_common_def.h"

namespace NpuArch::Epilogue::Block::Mxfp8VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

template <typename T, uint16_t QsBase = 128>
__simd_vf__ inline void mm1_res_pre_padding_align_kvs32_multi_qs64_vf(__ubuf__ T *s, uint16_t actSingleLoopS2Size, uint16_t actSingleLoopS2SizeAlign32)
{
    // ====================== 寄存器定义 ======================
    MaskReg mask_VL64_16b = CreateMask<uint16_t, MaskPattern::VL64>();
    uint16_t s2Idx = 0;
    RegTensor<T> padding_tensor1;
    Duplicate(padding_tensor1, MIN_VALUE, mask_VL64_16b);
    Muls(padding_tensor1, padding_tensor1, TWO_VALE, mask_VL64_16b);
    for (s2Idx = actSingleLoopS2Size; s2Idx < actSingleLoopS2SizeAlign32 - 1; s2Idx += 2) {
        StoreAlign(s + (s2Idx * QsBase)*2, padding_tensor1, mask_VL64_16b);
        StoreAlign(s + (s2Idx * QsBase + 1 * QsBase)*2, padding_tensor1, mask_VL64_16b);
    }

    for (uint16_t idx = s2Idx; idx < actSingleLoopS2SizeAlign32; ++idx) {
        StoreAlign(s + (idx * QsBase )*2, padding_tensor1, mask_VL64_16b);
    }

}

template <typename T>
__aicore__ inline void Mm1ResPrePaddingAlignKvs32MultiQs64CallVF(const LocalTensor<T> &srcTensor, uint16_t actSingleLoopS2Size, uint16_t actSingleLoopS2SizeAlign32)
{
    __ubuf__ T *input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();

    mm1_res_pre_padding_align_kvs32_multi_qs64_vf<T>(
        input_x_local_UB, actSingleLoopS2Size, actSingleLoopS2SizeAlign32);
}

}
#endif  // VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_QS64_H