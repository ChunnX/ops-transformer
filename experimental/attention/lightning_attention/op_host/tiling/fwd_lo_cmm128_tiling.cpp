/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fwd_lo_cmm128_tiling.h"

#include "utils.h"

#include <tiling/tiling_api.h>

namespace LightningAttn::Tiling
{

bool FwdLoCmm128Tiling::DoTiling(uint64_t inputTypeSize)
{
    tilingData.batchSize = batchSize;
    tilingData.seqLen = seqLen;
    tilingData.numHeads = numHeads;
    tilingData.dimSize = dimSize;
    tilingData.blockSize = BlockSize;
    tilingData.isTraining = false;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    tilingData.coreNum = platform->GetCoreNumAic();
    int64_t totalBatches = batchSize * numHeads;
    if (totalBatches < tilingData.coreNum)
    {
        tilingData.coreNum = totalBatches;
    }

    workspaceSize = platform->GetLibApiWorkSpaceSize();

    const uint64_t alignSize = 512ul;

    // Q, K, KV decay matrices size
    uint64_t qkDecaySize = numHeads * BlockSize * BlockSize * inputTypeSize;
    qkDecaySize = Align(qkDecaySize, alignSize);

    uint64_t kvDecaySize = numHeads * dimSize * dimSize * inputTypeSize;
    kvDecaySize = Align(kvDecaySize, alignSize);

    uint64_t diagDecaySize = numHeads * BlockSize * BlockSize * sizeof(float);
    diagDecaySize = Align(diagDecaySize, alignSize);

    uint64_t qkGMSize = batchSize * numHeads * BlockSize * BlockSize * sizeof(float);
    qkGMSize = Align(qkGMSize, alignSize);

    uint64_t qkmGMSize = batchSize * numHeads * BlockSize * BlockSize * inputTypeSize;
    qkmGMSize = Align(qkmGMSize, alignSize);

    uint64_t dumpKVGMSize = batchSize * numHeads * BlockSize * BlockSize * inputTypeSize;
    dumpKVGMSize = Align(dumpKVGMSize, alignSize);

    workspaceSize += qkDecaySize * 2 + kvDecaySize + diagDecaySize + qkGMSize + qkmGMSize + dumpKVGMSize;

    return true;
}

LoCmm128TilingData* FwdLoCmm128Tiling::GetTilingData()
{
    return &tilingData;
}

uint64_t FwdLoCmm128Tiling::GetWorkspaceSize()
{
    return workspaceSize;
}

} // namespace LightningAttn::Tiling
