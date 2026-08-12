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

static_assert(optiling::BLOCK_ATTN_RES_PREPARE_VECTOR_KEY == 1000UL,
              "Vector tiling key must match the kernel entry");
static_assert(optiling::BLOCK_ATTN_RES_PREPARE_MIX_CUBE_KEY == 2000UL,
              "Cube tiling key must match the kernel entry");

extern "C" __global__ __aicore__ void block_attn_res_prepare(
    GM_ADDR blockResidual, GM_ADDR effectiveQuery, GM_ADDR validBlocks, GM_ADDR softmaxMax, GM_ADDR weightedOutput,
    GM_ADDR softmaxSum, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::BlockAttnResPrepareTilingData);
    REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR == 2000)", optiling::BlockAttnResPrepareCubeTilingData);
    if (blockResidual == nullptr || effectiveQuery == nullptr || validBlocks == nullptr || softmaxMax == nullptr ||
        weightedOutput == nullptr || softmaxSum == nullptr || workspace == nullptr || tiling == nullptr) {
        return;
    }
    // Blaze MMAD may use the platform-reserved system workspace.  Keep the user Cube
    // workspace offset in TilingData while registering the complete workspace here.
    AscendC::SetSysWorkspace(workspace);
    // TILING_KEY_IS and KERNEL_TASK_TYPE are parsed by the AscendC compiler to
    // generate one binary entry per key.  Their key arguments must be integer
    // literals rather than C++ constexpr expressions.
    if (TILING_KEY_IS(1000)) {
        KERNEL_TASK_TYPE(1000, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(optiling::BlockAttnResPrepareTilingData, tilingData, tiling);
        BlockAttnResPrepare::BlockAttnResPrepareVector op(&tilingData);
        if (!op.Init(blockResidual, effectiveQuery, validBlocks, softmaxMax, weightedOutput, softmaxSum)) {
            return;
        }
        op.Process();
    } else if (TILING_KEY_IS(2000)) {
        KERNEL_TASK_TYPE(2000, KERNEL_TYPE_MIX_AIC_1_2);
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
