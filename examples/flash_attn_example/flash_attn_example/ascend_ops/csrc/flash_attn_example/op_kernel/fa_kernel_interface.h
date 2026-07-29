/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fa_kernel_interface.h
 * \brief FA kernel interface — direct Cube/Vector dispatch, no FlashAttentionKernel class.
 *        BN1→S1 split, no combine.
 */

#ifndef FA_KERNEL_INTERFACE_H
#define FA_KERNEL_INTERFACE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel/fa_block_cube.h"
#include "kernel/fa_block_vec.h"
#include "../fia_tiling_data.h"

using optiling::FlashAttnTilingData;

__aicore__ inline void CopyTiling(FlashAttnTilingData *tilingData, __gm__ uint8_t *tilingGM)
{
    constexpr int64_t wordCount = sizeof(FlashAttnTilingData) / sizeof(int64_t);
    constexpr int64_t tailBytes = sizeof(FlashAttnTilingData) % sizeof(int64_t);

    auto ptr = reinterpret_cast<int64_t *>(tilingData);
    auto tiling64 = reinterpret_cast<__gm__ int64_t *>(tilingGM);
    for (int64_t i = 0; i < wordCount; i++, ptr++) {
        *ptr = *(tiling64 + i);
    }
    if constexpr (tailBytes != 0) {
        auto ptr8 = reinterpret_cast<uint8_t *>(tilingData) + wordCount * sizeof(int64_t);
        auto tiling8 = tilingGM + wordCount * sizeof(int64_t);
        for (int64_t i = 0; i < tailBytes; i++) {
            ptr8[i] = tiling8[i];
        }
    }
    return;
}

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__global__ __aicore__ void FaKernel(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                    GM_ADDR attnMask, GM_ADDR attentionOut,
                                    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();

    FlashAttnTilingData tilingData;
    if (tiling != nullptr) {
        CopyTiling(&tilingData, (__gm__ uint8_t *)tiling);
    }

    auto faBaseParams = tilingData.baseTiling.faBaseParams;
    auto faAttnMaskParams = tilingData.baseTiling.faAttnMaskParams;

    uint32_t aicIdx = GetBlockIdx();

#ifdef __DAV_C310_CUBE__
    FaCube::CubeFunc<M_BASE, N_BASE, D_SIZE>(query, key, value, attnMask, tiling,
        faBaseParams, faAttnMaskParams,
        tilingData.faMetaData, aicIdx);
#else
    FaVector::VectorFunc<M_BASE, N_BASE, D_SIZE>(query, key, value, attnMask, attentionOut, tiling,
        faBaseParams, faAttnMaskParams,
        tilingData.faMetaData, aicIdx);
#endif
}

#endif  // FA_KERNEL_INTERFACE_H
