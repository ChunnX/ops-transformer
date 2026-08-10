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
 * \file block_attn_res_prepare_apt.cpp
 * \brief Kernel entry for BlockAttnResPrepare on Ascend 950.
 */

#include "arch35/block_attn_res_prepare_cube.h"
#include "arch35/block_attn_res_prepare_vector.h"

extern "C" __global__ __aicore__ void block_attn_res_prepare(
    GM_ADDR blockResidual, GM_ADDR effectiveQuery, GM_ADDR validBlocks, GM_ADDR softmaxMax, GM_ADDR weightedOutput,
    GM_ADDR softmaxSum, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(optiling::BLOCK_ATTN_RES_PREPARE_VECTOR_KEY, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(optiling::BLOCK_ATTN_RES_PREPARE_MIX_CUBE_KEY, KERNEL_TYPE_MIX_AIC_1_2);
    if (blockResidual == nullptr || effectiveQuery == nullptr || validBlocks == nullptr || softmaxMax == nullptr ||
        weightedOutput == nullptr || softmaxSum == nullptr || workspace == nullptr || tiling == nullptr) {
        return;
    }
    // Blaze MMAD may use the platform-reserved system workspace.  Keep the user Cube
    // workspace offset in TilingData while registering the complete workspace here.
    AscendC::SetSysWorkspace(workspace);
    if (TILING_KEY_IS(optiling::BLOCK_ATTN_RES_PREPARE_VECTOR_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(optiling::BlockAttnResPrepareTilingData, tilingData, tiling);
        BlockAttnResPrepare::BlockAttnResPrepareVector op(&tilingData);
        if (!op.Init(blockResidual, effectiveQuery, validBlocks, softmaxMax, weightedOutput, softmaxSum)) {
            return;
        }
        op.Process();
    } else if (TILING_KEY_IS(optiling::BLOCK_ATTN_RES_PREPARE_MIX_CUBE_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(optiling::BlockAttnResPrepareCubeTilingData, cubeTilingData, tiling);
        BlockAttnResPrepare::BlockAttnResPrepareCubeAicService aicService;
        BlockAttnResPrepare::BlockAttnResPrepareCubeAivService aivService;
        BlockAttnResPrepare::BlockAttnResPrepareCubeTemplate<
            BlockAttnResPrepare::BlockAttnResPrepareCubeAicService,
            BlockAttnResPrepare::BlockAttnResPrepareCubeAivService>
            op(&cubeTilingData, aicService, aivService);
        if (!op.Init(blockResidual, effectiveQuery, validBlocks, softmaxMax, weightedOutput, softmaxSum, workspace)) {
            return;
        }
        op.Process();
    }
}
