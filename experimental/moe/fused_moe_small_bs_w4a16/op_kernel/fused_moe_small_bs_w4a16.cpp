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
 * \file fused_moe_small_bs_w4a16.cpp
 * \brief fused moe w4a16 (small batch size) kernel entry
 */

#define K_MAX_SHAPE_DIM 0
#include "fused_moe_small_bs_w4a16_kernel.h"
#include "fused_moe_small_bs_w4a16_tiling_data.h"
#include "fused_moe_small_bs_w4a16_tiling_key.h"

using namespace NsFusedMoeSmallBsW4a16;

template <uint32_t schMode>
__global__ __aicore__ void fused_moe_small_bs_w4a16(
    GM_ADDR x, GM_ADDR w13Weight, GM_ADDR w13Scales,
    GM_ADDR w2Weight, GM_ADDR w2Scales,
    GM_ADDR expertIds, GM_ADDR topkWeights,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FusedMoeSmallBsW4a16TilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedMoeSmallBsW4a16TilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::InitSocState();
    AscendC::ICachePreLoad(5);

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if constexpr (schMode == FUSED_MOE_SMALL_BS_W4A16_SCH_MODE_FP16) {
        FusedMoeSmallBsImpl<half>(x, w13Weight, w13Scales, w2Weight, w2Scales,
                                  expertIds, topkWeights, y, userWS,
                                  tilingData.batchSize, tilingData.hiddenSize, tilingData.interSize,
                                  tilingData.numExperts, tilingData.topK);
    }
    if constexpr (schMode == FUSED_MOE_SMALL_BS_W4A16_SCH_MODE_BF16) {
        FusedMoeSmallBsImpl<bfloat16_t>(x, w13Weight, w13Scales, w2Weight, w2Scales,
                                        expertIds, topkWeights, y, userWS,
                                        tilingData.batchSize, tilingData.hiddenSize, tilingData.interSize,
                                        tilingData.numExperts, tilingData.topK);
    }
}
