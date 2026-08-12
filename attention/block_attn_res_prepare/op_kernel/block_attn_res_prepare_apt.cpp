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

#include "arch35/block_attn_res_prepare_vector.h"
#include "blaze/gemm/kernel/kernel_attention_residuals_phase1.h"

#define BLOCK_ATTN_RES_PREPARE_VECTOR_TILING_KEY 1000UL
#define BLOCK_ATTN_RES_PREPARE_MIX_CUBE_TILING_KEY 2000UL

namespace {
__aicore__ inline Blaze::Gemm::AttentionResidualsPhase1Params MakePhase1Params(
    GM_ADDR blockResidual, GM_ADDR effectiveQuery, GM_ADDR validBlocks, GM_ADDR softmaxMax,
    GM_ADDR weightedOutput, GM_ADDR softmaxSum, GM_ADDR workspace,
    const optiling::BlockAttnResPrepareCubeTilingData &tiling)
{
    Blaze::Gemm::AttentionResidualsPhase1Params params{};
    params.gmParams.blockResidualGmAddr = blockResidual;
    params.gmParams.effectiveQueryGmAddr = effectiveQuery;
    params.gmParams.validBlocksGmAddr = validBlocks;
    params.gmParams.softmaxMaxGmAddr = softmaxMax;
    params.gmParams.weightedOutputGmAddr = weightedOutput;
    params.gmParams.softmaxSumGmAddr = softmaxSum;
    params.gmParams.workspaceGmAddr = workspace;

    params.tilingParams.totalT = tiling.totalT;
    params.tilingParams.totalN = tiling.totalN;
    params.tilingParams.totalS = tiling.totalS;
    params.tilingParams.totalWorkUnits = tiling.totalWorkUnits;
    params.tilingParams.totalD = tiling.totalD;
    params.tilingParams.usedCoreNum = tiling.usedCoreNum;
    params.tilingParams.bigCoreNum = tiling.bigCoreNum;
    params.tilingParams.blockFactor = tiling.blockFactor;
    params.tilingParams.baseS = tiling.baseS;
    params.tilingParams.baseD = tiling.baseD;
    params.tilingParams.baseDAlign = tiling.baseDAlign;
    params.tilingParams.sTileNum = tiling.sTileNum;
    params.tilingParams.dTileNum = tiling.dTileNum;
    params.tilingParams.sAlign = tiling.sAlign;
    params.tilingParams.nAlign = tiling.nAlign;
    params.tilingParams.vUbBufferNum = tiling.vUbBufferNum;
    params.tilingParams.eWorkspaceElems = tiling.eL1Elems;
    params.tilingParams.vUbElems = tiling.vUbElems;
    params.tilingParams.dotUbElems = tiling.dotUbElems;
    params.tilingParams.reduceUbElems = tiling.reduceUbElems;
    params.tilingParams.softmaxUbElems = tiling.softmaxUbElems;
    params.tilingParams.workspacePerCoreElems = tiling.workspacePerCoreElems;
    params.tilingParams.epsilon = tiling.epsilon;
    return params;
}
} // namespace

extern "C" __global__ __aicore__ void block_attn_res_prepare(
    GM_ADDR blockResidual, GM_ADDR effectiveQuery, GM_ADDR validBlocks, GM_ADDR softmaxMax, GM_ADDR weightedOutput,
    GM_ADDR softmaxSum, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::BlockAttnResPrepareCubeTilingData);
    KERNEL_TASK_TYPE(BLOCK_ATTN_RES_PREPARE_VECTOR_TILING_KEY, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(BLOCK_ATTN_RES_PREPARE_MIX_CUBE_TILING_KEY, KERNEL_TYPE_MIX_AIC_1_2);
    if (blockResidual == nullptr || effectiveQuery == nullptr || validBlocks == nullptr || softmaxMax == nullptr ||
        weightedOutput == nullptr || softmaxSum == nullptr || workspace == nullptr || tiling == nullptr) {
        return;
    }
    if (TILING_KEY_IS(BLOCK_ATTN_RES_PREPARE_VECTOR_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(optiling::BlockAttnResPrepareTilingData, tilingData, tiling);
        BlockAttnResPrepare::BlockAttnResPrepareVector op(&tilingData);
        if (!op.Init(blockResidual, effectiveQuery, validBlocks, softmaxMax, weightedOutput, softmaxSum)) {
            return;
        }
        op.Process();
    } else if (TILING_KEY_IS(BLOCK_ATTN_RES_PREPARE_MIX_CUBE_TILING_KEY)) {
        AscendC::SetSysWorkspace(workspace);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        GET_TILING_DATA_WITH_STRUCT(optiling::BlockAttnResPrepareCubeTilingData, cubeTilingData, tiling);
        const auto params = MakePhase1Params(blockResidual, effectiveQuery, validBlocks, softmaxMax,
                                             weightedOutput, softmaxSum, userWorkspace, cubeTilingData);
        Blaze::Gemm::Kernel::KernelAttentionResidualsPhase1<> kernel;
        kernel(params);
    }
}
