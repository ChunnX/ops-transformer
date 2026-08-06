/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "common/lo_cmm128_tiling_data.h"

#include <string>

using namespace LightningAttn::Common;

namespace LightningAttn::Tiling
{

struct BwdLoCmm128Tiling
{
    static constexpr int64_t BlockSize = 128;

    BwdLoCmm128Tiling(int64_t batchSize, int64_t seqLen, int64_t numHeads, int64_t dimSize, bool isTraining) :
        batchSize(batchSize), seqLen(seqLen), numHeads(numHeads), dimSize(dimSize), isTraining(isTraining)
    {
    }

    bool DoTiling(uint64_t inputTypeSize);

    LoCmm128TilingData* GetTilingData();
    uint64_t GetWorkspaceSize();

private:
    int64_t batchSize;
    int64_t seqLen;
    int64_t numHeads;
    int64_t dimSize;
    bool isTraining;
    uint64_t workspaceSize;
    LoCmm128TilingData tilingData;
};

} // namespace LightningAttn::Tiling
