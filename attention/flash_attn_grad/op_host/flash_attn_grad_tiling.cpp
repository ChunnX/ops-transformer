/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling.h"
#include "log/log.h"
#include "op_host/tiling_util.h"

namespace optiling {

static bool HasAttenMask(gert::TilingContext *context, int64_t maskMode)
{
    if (maskMode == 0) {
        return false;
    }
    if (maskMode != 3 && maskMode != 4) {
        return false;
    }
    auto attnMaskShape = context->GetOptionalInputShape(10);
    if (attnMaskShape == nullptr || attnMaskShape->GetStorageShape().GetDimNum() == 0) {
        return false;
    }
    return true;
}

static ge::graphStatus FlashAttnGradTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "GetPlatformInfo is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
                OP_LOGE(context->GetNodeName(), "num of core obtained is 0, aicNum=%u, aivNum=%u.", aicNum, aivNum),
                return ge::GRAPH_PARAM_INVALID);

    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGE(context->GetNodeName(), "SOC Version is not support.");
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // Extract attrs
    auto attrs = context->GetAttrs();
    // Attr indices: 0=softmax_scale(Float), 1=mask_mode(Int), 2=win_left(Int), 3=win_right(Int),
    //               4=max_seqlen_q(Int), 5=max_seqlen_kv(Int), 6=layout_q(String), 7=layout_kv(String), 8=layout_out(String)
    std::string layoutQStr = "BSND";
    int64_t maxSeqlenQ = -1;
    int64_t maxSeqlenKv = -1;
    int64_t maskMode = 0;
    int64_t winLeft = -1;
    int64_t winRight = -1;
    float scaleValue = 0.0f;
    if (attrs != nullptr) {
        auto scalePtr = attrs->GetAttrPointer<float>(0);
        if (scalePtr != nullptr) {
            scaleValue = *scalePtr;
        }
        auto maskModePtr = attrs->GetAttrPointer<int64_t>(1);
        if (maskModePtr != nullptr) {
            maskMode = *maskModePtr;
        }
        auto winLeftPtr = attrs->GetAttrPointer<int64_t>(2);
        if (winLeftPtr != nullptr) {
            winLeft = *winLeftPtr;
        }
        auto winRightPtr = attrs->GetAttrPointer<int64_t>(3);
        if (winRightPtr != nullptr) {
            winRight = *winRightPtr;
        }
        auto layoutQPtr = attrs->GetAttrPointer<char>(6);
        if (layoutQPtr != nullptr) {
            layoutQStr = std::string(layoutQPtr);
        }
        auto maxSeqlenQPtr = attrs->GetAttrPointer<int64_t>(4);
        if (maxSeqlenQPtr != nullptr) {
            maxSeqlenQ = *maxSeqlenQPtr;
        }
        auto maxSeqlenKvPtr = attrs->GetAttrPointer<int64_t>(5);
        if (maxSeqlenKvPtr != nullptr) {
            maxSeqlenKv = *maxSeqlenKvPtr;
        }
    }

    int64_t layout = 0; // 0=BSND, 1=TND
    if (layoutQStr == "TND") {
        layout = 1;
    }

    // Extract shape dims
    auto qShape = context->GetInputShape(0);
    auto kShape = context->GetInputShape(1);
    auto vShape = context->GetInputShape(2);
    OP_CHECK_IF(qShape == nullptr || kShape == nullptr || vShape == nullptr,
                OP_LOGE(context->GetNodeName(), "q or k or v shape is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    int64_t bSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t dSize = 0;
    int64_t dvSize = 0;

    if (layout == 1) {
        // TND: q=(T1, N1, D), k=(T2, N2, D)
        s1Size = maxSeqlenQ > 0 ? maxSeqlenQ : qShape->GetStorageShape().GetDim(0);
        s2Size = maxSeqlenKv > 0 ? maxSeqlenKv : kShape->GetStorageShape().GetDim(0);
        n1Size = qShape->GetStorageShape().GetDim(1);
        n2Size = kShape->GetStorageShape().GetDim(1);
        dSize = qShape->GetStorageShape().GetDim(2);
        dvSize = vShape->GetStorageShape().GetDim(2);
        // B from cu_seqlens_q (index 6)
        auto cuSeqShape = context->GetOptionalInputShape(6);
        if (cuSeqShape != nullptr && cuSeqShape->GetStorageShape().GetDimNum() > 0) {
            bSize = cuSeqShape->GetStorageShape().GetDim(0) - 1;
        } else {
            bSize = 1;
        }
    } else {
        // BSND: q=(B, S1, N1, D), k=(B, S2, N2, D)
        bSize = qShape->GetStorageShape().GetDim(0);
        s1Size = qShape->GetStorageShape().GetDim(1);
        n1Size = qShape->GetStorageShape().GetDim(2);
        dSize = qShape->GetStorageShape().GetDim(3);
        dvSize = vShape->GetStorageShape().GetDim(3);
        s2Size = kShape->GetStorageShape().GetDim(1);
        n2Size = kShape->GetStorageShape().GetDim(2);
    }

    // Set tiling data
    FlashAttnGradTilingData *tilingData = context->GetTilingData<FlashAttnGradTilingData>();
    tilingData->b = bSize;
    tilingData->s1 = s1Size;
    tilingData->s2 = s2Size;
    tilingData->n1 = n1Size;
    tilingData->n2 = n2Size;
    tilingData->d = dSize;
    tilingData->dv = dvSize;
    tilingData->scaleValue = scaleValue == 0.0f ? 1.0f / sqrt(static_cast<float>(dSize)) : scaleValue;
    tilingData->maskMode = maskMode;
    tilingData->winLeft = winLeft;
    tilingData->winRight = winRight;
    tilingData->maxSeqlenQ = maxSeqlenQ;
    tilingData->maxSeqlenKv = maxSeqlenKv;

    // TilingKey: bit0=layout (0=BSND/BNSD, 1=TND), bit1=has_atten_mask
    bool hasAttenMask = HasAttenMask(context, maskMode);
    uint64_t tilingKey = static_cast<uint64_t>(layout) | (static_cast<uint64_t>(hasAttenMask) << 1);
    context->SetTilingKey(tilingKey);

    // Workspace
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(),
            "FlashAttnGrad tiling: blockDim=%u, layout=%ld, maskMode=%ld, hasAttenMask=%d, B=%ld, S1=%ld, S2=%ld, N1=%ld, D=%ld, tilingKey=%llu, workspaceSize=%zu.",
            blockDim, layout, maskMode, hasAttenMask, bSize, s1Size, s2Size, n1Size, dSize, tilingKey, workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFlashAttnGrad([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlashAttnGrad)
    .Tiling(FlashAttnGradTilingFunc)
    .TilingParse<FlashAttnGradCompileInfo>(TilingParseForFlashAttnGrad);

} // namespace optiling
