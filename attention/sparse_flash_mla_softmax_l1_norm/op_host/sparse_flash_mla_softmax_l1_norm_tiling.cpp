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

constexpr uint32_t METADATA_INDEX = 10;
constexpr int32_t SMLA_MAX_CORE_NUM = 36;

static int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

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

    OpTiling *tilingData = context->GetTilingData<OpTiling>();
    tilingData->b = bSize;
    tilingData->sq = sqSize;
    tilingData->sk = skSize;
    tilingData->g = gSize;
    tilingData->d = dSize;

    // Read metadata from the AICPU metadata op (input index 10).
    // Metadata layout (int32[64]): [totalNum, formerCoreProcessNum,
    //   remainCoreProcessNum, remainCoreNum, totalCoreNum, ...]
    auto metadataTensor = context->GetOptionalInputTensor(METADATA_INDEX);
    if (metadataTensor != nullptr && metadataTensor->GetData<int32_t>() != nullptr) {
        const int32_t *meta = metadataTensor->GetData<int32_t>();
        tilingData->totalNum = static_cast<int64_t>(meta[0]);
        tilingData->formerCoreProcessNum = static_cast<int64_t>(meta[1]);
        tilingData->remainCoreProcessNum = static_cast<int64_t>(meta[2]);
        tilingData->remainCoreNum = static_cast<int64_t>(meta[3]);
        tilingData->totalCoreNum = static_cast<int64_t>(meta[4]);
    } else {
        // Fallback: compute load-balancing parameters on host (UT environment).
        int64_t totalNum = bSize * sqSize;
        int64_t totalCoreNum = std::min<int64_t>(std::min(totalNum, static_cast<int64_t>(blockDim)),
                                                  static_cast<int64_t>(SMLA_MAX_CORE_NUM));
        int64_t formerCoreProcessNum = CeilDiv(totalNum, totalCoreNum);
        int64_t remainCoreProcessNum = totalNum / totalCoreNum;
        int64_t remainder = totalNum % totalCoreNum;
        int64_t remainCoreNum = (remainder == 0) ? 0 : (totalCoreNum - remainder);
        tilingData->totalNum = totalNum;
        tilingData->formerCoreProcessNum = formerCoreProcessNum;
        tilingData->remainCoreProcessNum = remainCoreProcessNum;
        tilingData->remainCoreNum = remainCoreNum;
        tilingData->totalCoreNum = totalCoreNum;
    }

    context->SetTilingKey(0);

    // Workspace: system reserve + qk buffer [G, Sk] FP32.
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += static_cast<size_t>(gSize) * static_cast<size_t>(skSize) * sizeof(float);
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(), "SparseFlashMlaSoftmaxL1Norm tiling completed: blockDim=%u, aivNum=%u, aicNum=%u, B=%ld, Sq=%ld, Sk=%ld, G=%ld, D=%ld, totalNum=%ld, totalCoreNum=%ld.",
            blockDim, aivNum, aicNum, bSize, sqSize, skSize, gSize, dSize,
            tilingData->totalNum, tilingData->totalCoreNum);
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
