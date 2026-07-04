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
    DenseLISoftmaxLseV2TilingData tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(aivNum);

    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = 0;

    tilingData.set_usedCoreNum(aivNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGI(context->GetNodeName(), "DenseLightningIndexerSoftmaxLseV2 tiling completed: blockDim=%u, usedCoreNum=%u.",
            aivNum, aivNum);

    constexpr uint32_t METADATA_INPUT_INDEX = 8;
    auto metadataTensor = context->GetOptionalInputTensor(METADATA_INPUT_INDEX);
    if (metadataTensor != nullptr && metadataTensor->GetData() != nullptr) {
        const int32_t *metadataData = metadataTensor->GetData<int32_t>();
        auto metadataSize = metadataTensor->GetShapeSize();
        OP_LOGI(context->GetNodeName(), "metadata tensor found, shapeSize=%ld.", metadataSize);
        for (int64_t i = 0; i < metadataSize && i < 10; i++) {
            OP_LOGI(context->GetNodeName(), "metadata[%ld] = %d.", i, metadataData[i]);
        }
    } else {
        OP_LOGI(context->GetNodeName(), "metadata tensor is nullptr, not provided.");
    }

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
