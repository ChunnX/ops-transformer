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
using TensorApiKernel = Blaze::Attention::Kernel::KernelBlockAttnResPrepare;
using Params = typename TensorApiKernel::Params;

__aicore__ inline void AssembleShapeParams(Params &params,
                                           const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    auto &apiTiling = params.tilingParams;
    apiTiling.totalT = tiling.totalT;
    apiTiling.totalN = tiling.totalN;
    apiTiling.totalS = tiling.totalS;
    apiTiling.totalWorkUnits = tiling.totalWorkUnits;
    apiTiling.totalD = tiling.totalD;
}

__aicore__ inline void AssembleTileParams(Params &params,
                                          const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    auto &apiTiling = params.tilingParams;
    apiTiling.baseS = tiling.baseS;
    apiTiling.baseT = tiling.baseT;
    apiTiling.baseD = tiling.baseD;
    apiTiling.baseDAlign = tiling.baseDAlign;
    apiTiling.sTileNum = tiling.sTileNum;
    apiTiling.dTileNum = tiling.dTileNum;
    apiTiling.sAlign = tiling.sAlign;
    apiTiling.nAlign = tiling.nAlign;
    apiTiling.mm1NAlign = tiling.mm1NAlign;
    apiTiling.mm1L1Stages =
        tiling.qL1BufferNum < tiling.vL1BufferNum ? tiling.qL1BufferNum : tiling.vL1BufferNum;
}

__aicore__ inline void AssembleBufferParams(Params &params,
                                            const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    auto &apiTiling = params.tilingParams;
    apiTiling.vUbBufferNum = tiling.vUbBufferNum;
    apiTiling.eWorkspaceElems = tiling.eL1Elems;
    apiTiling.vUbElems = tiling.vUbElems;
    apiTiling.dotUbElems = tiling.dotUbElems;
    apiTiling.reduceUbElems = tiling.reduceUbElems;
    apiTiling.softmaxUbElems = tiling.softmaxUbElems;
    apiTiling.workspacePerCoreElems = tiling.workspacePerCoreElems;
    apiTiling.epsilon = tiling.eps;
}

__aicore__ inline void AssembleGmParams(Params &params, GM_ADDR blockRes, GM_ADDR validBlocks,
                                        GM_ADDR pseudoQuery, GM_ADDR numerator, GM_ADDR logitMax, GM_ADDR expSum,
                                        GM_ADDR workspace)
{
    params.gmParams = {blockRes, pseudoQuery, validBlocks, logitMax, numerator, expSum, workspace};
}

__aicore__ inline void AssembleSchedulerParams(Params &params, const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    auto &apiTiling = params.tilingParams;
    apiTiling.usedCoreNum = tiling.usedCoreNum;
    if (apiTiling.usedCoreNum == 0U) {
        return;
    }
    apiTiling.bigCoreNum = tiling.totalWorkUnits % apiTiling.usedCoreNum;
    apiTiling.blockFactor = tiling.totalWorkUnits / apiTiling.usedCoreNum;
}
} // namespace BlockAttnResPrepareMixDetail

__aicore__ inline void BlockAttnResPrepareTensorApiBlazeKernel(GM_ADDR blockRes, GM_ADDR validBlocks,
                                                               GM_ADDR pseudoQuery, GM_ADDR numerator, GM_ADDR logitMax,
                                                               GM_ADDR expSum, GM_ADDR workspace,
                                                               const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    BlockAttnResPrepareMixDetail::Params params{};
    BlockAttnResPrepareMixDetail::AssembleGmParams(params, blockRes, validBlocks, pseudoQuery, numerator, logitMax,
                                                   expSum, workspace);
    BlockAttnResPrepareMixDetail::AssembleShapeParams(params, tiling);
    BlockAttnResPrepareMixDetail::AssembleTileParams(params, tiling);
    BlockAttnResPrepareMixDetail::AssembleBufferParams(params, tiling);
    BlockAttnResPrepareMixDetail::AssembleSchedulerParams(params, tiling);

    BlockAttnResPrepareMixDetail::TensorApiKernel kernel;
    kernel(params);
}

} // namespace BlockAttnResPrepare

#endif // BLOCK_ATTN_RES_PREPARE_MIX_H
