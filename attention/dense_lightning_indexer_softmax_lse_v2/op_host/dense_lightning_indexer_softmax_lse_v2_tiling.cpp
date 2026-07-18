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
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // Extract layout attr
    auto attrs = context->GetAttrs();
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t layout = 0;  // 0=BSND, 1=TND
    if (attrs != nullptr) {
        auto maskModePtr = attrs->GetAttrPointer<int64_t>(2);
        auto cmpRatioPtr = attrs->GetAttrPointer<int64_t>(3);
        if (maskModePtr != nullptr) {
            maskMode = *maskModePtr;
        }
        if (cmpRatioPtr != nullptr) {
            cmpRatio = *cmpRatioPtr;
        }
        // Attr index 4 and 5 are layoutQ and layoutK
        auto layoutQPtr = attrs->GetAttrPointer<char*>(4);
        if (layoutQPtr != nullptr && *layoutQPtr != nullptr) {
            std::string layoutQStr(*layoutQPtr);
            if (layoutQStr == "TND") {
                layout = 1;
            }
        }
    }

    // Extract runtime dimensions from input shapes.
    auto queryShape = context->GetInputShape(0);
    auto keyShape = context->GetInputShape(1);
    int64_t bSize, s1Size, n1Size, dSize, s2Size;

    if (layout == 1) {
        // TND: query [T1, N1, D], key [T2, 1, D]
        s1Size = queryShape->GetStorageShape().GetDim(0);
        n1Size = queryShape->GetStorageShape().GetDim(1);
        dSize = queryShape->GetStorageShape().GetDim(2);
        s2Size = keyShape->GetStorageShape().GetDim(0);
        // B from cu_seq_lens_q input (shape [B+1])
        auto cuSeqLensQShape = context->GetInputShape(5);
        if (cuSeqLensQShape != nullptr) {
            bSize = cuSeqLensQShape->GetStorageShape().GetDim(0) - 1;
        } else {
            bSize = 1;
        }
    } else {
        // BSND: query [B, S1, N1, D], key [B, S2, 1, D]
        bSize = queryShape->GetStorageShape().GetDim(0);
        s1Size = queryShape->GetStorageShape().GetDim(1);
        n1Size = queryShape->GetStorageShape().GetDim(2);
        dSize = queryShape->GetStorageShape().GetDim(3);
        s2Size = keyShape->GetStorageShape().GetDim(1);
    }

    // OpTiling comes from the pypto kernel (.py dataclass) via force-include.
    OpTiling *tilingData = context->GetTilingData<OpTiling>();
    tilingData->b = bSize;
    tilingData->s1 = s1Size;
    tilingData->s2 = s2Size;
    tilingData->n1 = n1Size;
    tilingData->d = dSize;

    // Determine if S2 exceeds reduce_sum_vec UB capacity.
    constexpr int64_t TKV = 128;
    constexpr int64_t TKV_HALF = TKV / 2;
    constexpr int64_t KV_CACHE_NUM = 46832;
    int64_t s2Tiles = (s2Size + TKV - 1) / TKV;
    int64_t maxUbS2Tiles = KV_CACHE_NUM / TKV_HALF;
    int64_t isLongS2 = (s2Tiles > maxUbS2Tiles) ? 1 : 0;
    tilingData->isLongS2 = isLongS2;

    tilingData->mask_mode = maskMode;
    tilingData->cmp_ratio = cmpRatio;
    tilingData->layout = layout;

    context->SetTilingKey(static_cast<uint32_t>(isLongS2));

    // Workspace: system reserve + workspace_max [1,72] FP32 (288 bytes) + workspace_sum [1,72] FP32 (288 bytes)
    //            + workspace [36, S2] FP32 (for long S2 reduce_sum spill).
    constexpr size_t WS_MAX_BYTES = 288;
    constexpr int64_t WS_MAX_CORES = 36;
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += WS_MAX_BYTES * 2;
    workspaceSize += static_cast<size_t>(WS_MAX_CORES) * static_cast<size_t>(s2Size) * sizeof(float);
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(), "DenseLightningIndexerSoftmaxLseV2 tiling completed: blockDim=%u, aivNum=%u, aicNum=%u, layout=%ld, B=%ld, S1=%ld, S2=%ld, N1=%ld, D=%ld, isLongS2=%ld, maskMode=%ld, cmpRatio=%ld, workspaceSize=%zu.",
            blockDim, aivNum, aicNum, layout, bSize, s1Size, s2Size, n1Size, dSize, isLongS2, maskMode, cmpRatio, workspaceSize);
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
