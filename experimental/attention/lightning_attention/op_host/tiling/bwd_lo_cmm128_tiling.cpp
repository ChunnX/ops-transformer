/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "bwd_lo_cmm128_tiling.h"

#include "utils.h"

#include <tiling/tiling_api.h>

namespace LightningAttn::Tiling
{

bool BwdLoCmm128Tiling::DoTiling(uint64_t inputTypeSize)
{
    tilingData.batchSize = batchSize;
    tilingData.seqLen = seqLen;
    tilingData.numHeads = numHeads;
    tilingData.dimSize = dimSize;
    tilingData.blockSize = BlockSize;
    tilingData.isTraining = isTraining;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    tilingData.coreNum = platform->GetCoreNumAic();
    int64_t totalBatches = batchSize * numHeads;

    workspaceSize = platform->GetLibApiWorkSpaceSize();

    const uint64_t alignSize = 512ul;
    const uint64_t blockMatrixSize = numHeads * BlockSize * BlockSize;
    const uint64_t headMatrixSize = numHeads * dimSize * dimSize;

    // size of Q, K, KV decay matrices
    const uint64_t qkDecaySize = Align(blockMatrixSize * inputTypeSize, alignSize);
    const uint64_t kvDecaySize = Align(headMatrixSize * inputTypeSize, alignSize);
    const uint64_t diagDecaySize = Align(blockMatrixSize * sizeof(float), alignSize);

    workspaceSize += (qkDecaySize * 2 + kvDecaySize + diagDecaySize);

    // size of temporary matrices 
    const uint64_t tmpBuffSize = blockMatrixSize * tilingData.coreNum;
    const uint64_t tmpInputBuffSize = Align(tmpBuffSize * inputTypeSize, alignSize);
    const uint64_t tmpFloatBuffSize = Align(tmpBuffSize * sizeof(float), alignSize);

    workspaceSize += (tmpInputBuffSize * 2 + tmpFloatBuffSize * 2);

    return true;
}

LoCmm128TilingData* BwdLoCmm128Tiling::GetTilingData()
{
    return &tilingData;
}

uint64_t BwdLoCmm128Tiling::GetWorkspaceSize()
{
    return workspaceSize;
}

} // namespace LightningAttn::Tiling
