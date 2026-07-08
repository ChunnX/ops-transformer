/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_v2_tiling.cpp
 * \brief
 */

#include "dense_lightning_indexer_softmax_lse_v2_tiling.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

static ge::graphStatus DenseLightningIndexerSoftmaxLseV2TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(aivNum);

    // Extract runtime dimensions from input shapes (BSND layout).
    // query_index: [B, S1, N1, D], key_index: [B, S2, 1, D]
    auto queryShape = context->GetInputShape(0);
    auto keyShape = context->GetInputShape(1);
    int64_t bSize = queryShape->GetStorageShape().GetDim(0);
    int64_t s1Size = queryShape->GetStorageShape().GetDim(1);
    int64_t n1Size = queryShape->GetStorageShape().GetDim(2);
    int64_t dSize = queryShape->GetStorageShape().GetDim(3);
    int64_t s2Size = keyShape->GetStorageShape().GetDim(1);

    // OpTiling comes from the pypto kernel (.py dataclass) via force-include.
    OpTiling *tilingData = context->GetTilingData<OpTiling>();
    tilingData->b = bSize;
    tilingData->s1 = s1Size;
    tilingData->s2 = s2Size;
    tilingData->n1 = n1Size;
    tilingData->d = dSize;
    context->SetTilingKey(0);

    // Workspace: system reserve + workspace_max [1,72] FP32 (padded to 512 bytes) + workspace [1,S2] FP32.
    constexpr size_t WS_MAX_BYTES = 512;
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += WS_MAX_BYTES + static_cast<size_t>(s2Size) * sizeof(float);
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(), "DenseLightningIndexerSoftmaxLseV2 tiling completed: blockDim=%u, B=%ld, S1=%ld, S2=%ld, N1=%ld, D=%ld, workspaceSize=%zu.",
            aivNum, bSize, s1Size, s2Size, n1Size, dSize, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDenseLightningIndexerSoftmaxLseV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DenseLightningIndexerSoftmaxLseV2)
    .Tiling(DenseLightningIndexerSoftmaxLseV2TilingFunc)
    .TilingParse<DenseLISoftmaxLseV2CompileInfo>(TilingParseForDenseLightningIndexerSoftmaxLseV2);

} // namespace optiling
