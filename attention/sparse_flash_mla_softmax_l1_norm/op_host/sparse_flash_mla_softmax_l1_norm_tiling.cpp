/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_mla_softmax_l1_norm_tiling.cpp
 * \brief
 */
#include "sparse_flash_mla_softmax_l1_norm_tiling.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

static ge::graphStatus SparseFlashMlaSoftmaxL1NormTilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // query: [B, Sq, G, D], key: [B, Sk, 1, D]
    auto queryShape = context->GetInputShape(0);
    auto keyShape = context->GetInputShape(1);
    int64_t bSize = queryShape->GetStorageShape().GetDim(0);
    int64_t sqSize = queryShape->GetStorageShape().GetDim(1);
    int64_t gSize = queryShape->GetStorageShape().GetDim(2);
    int64_t dSize = queryShape->GetStorageShape().GetDim(3);
    int64_t skSize = keyShape->GetStorageShape().GetDim(1);

    // OpTiling comes from the pypto kernel (.py dataclass) via force-include.
    OpTiling *tilingData = context->GetTilingData<OpTiling>();
    tilingData->b = bSize;
    tilingData->sq = sqSize;
    tilingData->sk = skSize;
    tilingData->g = gSize;
    tilingData->d = dSize;
    context->SetTilingKey(0);

    // Workspace: system reserve + qk buffer [G, Sk] FP32.
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += static_cast<uint32_t>(gSize * skSize * sizeof(float));
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = static_cast<size_t>(workspaceSize);

    OP_LOGI(context->GetNodeName(), "SparseFlashMlaSoftmaxL1Norm tiling completed: blockDim=%u, aivNum=%u, aicNum=%u, B=%ld, Sq=%ld, Sk=%ld, G=%ld, D=%ld.",
            blockDim, aivNum, aicNum, bSize, sqSize, skSize, gSize, dSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSparseFlashMlaSoftmaxL1Norm([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseFlashMlaSoftmaxL1Norm)
    .Tiling(SparseFlashMlaSoftmaxL1NormTilingFunc)
    .TilingParse<SparseFlashMlaSoftmaxL1NormCompileInfo>(TilingParseForSparseFlashMlaSoftmaxL1Norm);
} // namespace optiling
