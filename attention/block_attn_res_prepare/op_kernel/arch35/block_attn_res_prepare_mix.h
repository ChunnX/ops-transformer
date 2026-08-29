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
 * \file block_attn_res_prepare_mix.h
 * \brief Blaze tensor API entry for BlockAttnResPrepare.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_MIX_H
#define BLOCK_ATTN_RES_PREPARE_MIX_H

#include "block_attn_res_prepare_tiling_data.h"
#include "blaze/attention/kernel/kernel_block_attn_res_prepare.h"
#include "kernel_operator.h"

namespace BlockAttnResPrepare {

namespace BlockAttnResPrepareMixDetail {
constexpr uint32_t MM1_L0_K_MAX = 64U;
constexpr uint32_t SINGLE_STAGE = 1U;

using ProblemShape = Blaze::Attention::Kernel::BlockAttnResPrepareProblemShape;
using BlockMmadTuple = Blaze::Attention::Kernel::BlockAttnResPrepareMmadTuple;
using BlockEpilogue = Blaze::Attention::Kernel::BlockAttnResPrepareBlockEpilogue;
using BlockScheduler = Blaze::Attention::Kernel::BlockAttnResPrepareBlockScheduler;
using TensorApiKernel =
    Blaze::Attention::Kernel::AttentionUniversal<ProblemShape, BlockMmadTuple, BlockEpilogue, BlockScheduler>;
using Params = typename TensorApiKernel::Params;

__aicore__ inline void AssembleProblemShape(Params &params,
                                            const optiling::BlockAttnResPrepareCubeTilingData &tiling)
{
    params.problemShape = {static_cast<int64_t>(tiling.totalS), static_cast<int64_t>(tiling.totalN),
                           static_cast<int64_t>(tiling.totalD), static_cast<int64_t>(tiling.totalT)};
}

__aicore__ inline void AssembleMm1Params(Params &params, GM_ADDR pseudoQuery, GM_ADDR blockRes, GM_ADDR workspace,
                                         const optiling::BlockAttnResPrepareCubeTilingData &tiling)
{
    params.mm1Params.aGmAddr = pseudoQuery;
    params.mm1Params.bGmAddr = blockRes;
    params.mm1Params.cGmAddr = workspace;
    params.mm1Params.workspaceGmAddr = workspace;
    params.mm1Params.mL1 = tiling.sAlign;
    params.mm1Params.nL1 = tiling.mm1NAlign;
    params.mm1Params.kL1 = tiling.baseDAlign;
    params.mm1Params.mL0 = tiling.sAlign;
    params.mm1Params.nL0 = tiling.mm1NAlign;
    params.mm1Params.kL0 = tiling.baseD < MM1_L0_K_MAX ? tiling.baseD : MM1_L0_K_MAX;
    params.mm1Params.l1Stages = tiling.qL1BufferNum < tiling.vL1BufferNum ? tiling.qL1BufferNum : tiling.vL1BufferNum;
    params.mm1Params.l0cStages = SINGLE_STAGE;
}

__aicore__ inline void AssembleMm2Params(Params &params, GM_ADDR blockRes, GM_ADDR numerator, GM_ADDR workspace,
                                         const optiling::BlockAttnResPrepareCubeTilingData &tiling)
{
    params.mm2Params.aGmAddr = workspace;
    params.mm2Params.bGmAddr = blockRes;
    params.mm2Params.cGmAddr = numerator;
    params.mm2Params.workspaceGmAddr = workspace;
    params.mm2Params.mL1 = tiling.sAlign;
    params.mm2Params.nL1 = tiling.baseDAlign;
    params.mm2Params.kL1 = tiling.nAlign;
    params.mm2Params.mL0 = tiling.sAlign;
    params.mm2Params.nL0 = tiling.baseDAlign;
    params.mm2Params.kL0 = tiling.nAlign;
    params.mm2Params.l1Stages = SINGLE_STAGE;
    params.mm2Params.l0cStages = SINGLE_STAGE;
}

__aicore__ inline void AssembleEpilogueParams(Params &params, GM_ADDR blockRes, GM_ADDR validBlocks,
                                              GM_ADDR numerator, GM_ADDR logitMax, GM_ADDR expSum, GM_ADDR workspace,
                                              const optiling::BlockAttnResPrepareCubeTilingData &tiling)
{
    params.epilogueParams.blockResidualGmAddr = blockRes;
    params.epilogueParams.validBlocksGmAddr = validBlocks;
    params.epilogueParams.softmaxMaxGmAddr = logitMax;
    params.epilogueParams.weightedOutputGmAddr = numerator;
    params.epilogueParams.softmaxSumGmAddr = expSum;
    params.epilogueParams.workspaceGmAddr = workspace;
    params.epilogueParams.totalD = tiling.totalD;
    params.epilogueParams.baseD = tiling.baseD;
    params.epilogueParams.baseDAlign = tiling.baseDAlign;
    params.epilogueParams.dTileNum = tiling.dTileNum;
    params.epilogueParams.sAlign = tiling.sAlign;
    params.epilogueParams.vUbBufferNum = tiling.vUbBufferNum;
    params.epilogueParams.eWorkspaceElems = tiling.eL1Elems;
    params.epilogueParams.vUbElems = tiling.vUbElems;
    params.epilogueParams.dotUbElems = tiling.dotUbElems;
    params.epilogueParams.reduceUbElems = tiling.reduceUbElems;
    params.epilogueParams.softmaxUbElems = tiling.softmaxUbElems;
    params.epilogueParams.workspacePerCoreElems = tiling.workspacePerCoreElems;
    params.epilogueParams.epsilon = tiling.eps;
}

__aicore__ inline void AssembleSchedulerParams(Params &params,
                                               const optiling::BlockAttnResPrepareCubeTilingData &tiling)
{
    params.schedulerParams.totalWorkUnits = tiling.totalWorkUnits;
    params.schedulerParams.usedCoreNum = tiling.usedCoreNum;
    params.schedulerParams.baseT = tiling.baseT;
    params.schedulerParams.baseS = tiling.baseS;
    params.schedulerParams.sTileNum = tiling.sTileNum;
    params.schedulerParams.mm1NAlign = tiling.mm1NAlign;
}
} // namespace BlockAttnResPrepareMixDetail

__aicore__ inline void BlockAttnResPrepareTensorApiBlazeKernel(
    GM_ADDR blockRes, GM_ADDR validBlocks, GM_ADDR pseudoQuery, GM_ADDR numerator, GM_ADDR logitMax, GM_ADDR expSum,
    GM_ADDR workspace, const optiling::BlockAttnResPrepareCubeTilingData &tiling)
{
    BlockAttnResPrepareMixDetail::Params params{};
    BlockAttnResPrepareMixDetail::AssembleProblemShape(params, tiling);
    BlockAttnResPrepareMixDetail::AssembleMm1Params(params, pseudoQuery, blockRes, workspace, tiling);
    BlockAttnResPrepareMixDetail::AssembleMm2Params(params, blockRes, numerator, workspace, tiling);
    BlockAttnResPrepareMixDetail::AssembleEpilogueParams(params, blockRes, validBlocks, numerator, logitMax, expSum,
                                                         workspace, tiling);
    BlockAttnResPrepareMixDetail::AssembleSchedulerParams(params, tiling);

    BlockAttnResPrepareMixDetail::TensorApiKernel kernel;
    kernel(params);
}

} // namespace BlockAttnResPrepare

#endif // BLOCK_ATTN_RES_PREPARE_MIX_H
