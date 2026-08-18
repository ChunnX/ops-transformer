/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling.h"
#include "flash_attn_grad_check.h"
#include "opdev/op_log.h"
#include "op_host/tiling_util.h"

namespace optiling {

static bool HasAttenMask(gert::TilingContext *context, int64_t maskMode)
{
    if (maskMode == 0 || (maskMode != 3 && maskMode != 4)) {
        return false;
    }
    auto attnMaskShape = context->GetOptionalInputShape(ATTN_MASK_INDEX);
    return (attnMaskShape != nullptr) && (attnMaskShape->GetStorageShape().GetDimNum() > 0);
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

    auto checkRet = FlashAttnGradCheck::CheckParams(context);
    OP_CHECK_IF(checkRet != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckParams failed."),
                return ge::GRAPH_PARAM_INVALID);

    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    auto attrs = context->GetAttrs();
    std::string layoutQStr = "BSND";
    int64_t maxSeqlenQ = -1;
    int64_t maxSeqlenKv = -1;
    int64_t maskMode = 0;
    float scaleValue = 0.0f;
    if (attrs != nullptr) {
        auto scalePtr = attrs->GetAttrPointer<float>(0);
        if (scalePtr != nullptr) { scaleValue = *scalePtr; }
        auto maskModePtr = attrs->GetAttrPointer<int64_t>(1);
        if (maskModePtr != nullptr) { maskMode = *maskModePtr; }
        auto maxSeqlenQPtr = attrs->GetAttrPointer<int64_t>(4);
        if (maxSeqlenQPtr != nullptr) { maxSeqlenQ = *maxSeqlenQPtr; }
        auto maxSeqlenKvPtr = attrs->GetAttrPointer<int64_t>(5);
        if (maxSeqlenKvPtr != nullptr) { maxSeqlenKv = *maxSeqlenKvPtr; }
        auto layoutQPtr = attrs->GetAttrPointer<char>(6);
        if (layoutQPtr != nullptr) { layoutQStr = std::string(layoutQPtr); }
    }

    int64_t layout = (layoutQStr == "TND") ? 1 : 0;

    auto qShape = context->GetInputShape(Q_INDEX);
    auto kShape = context->GetInputShape(K_INDEX);
    auto vShape = context->GetInputShape(V_INDEX);

    int64_t bSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t dSize = 0;
    int64_t dvSize = 0;

    if (layout == 1) {
        s1Size = maxSeqlenQ > 0 ? maxSeqlenQ : qShape->GetStorageShape().GetDim(0);
        s2Size = maxSeqlenKv > 0 ? maxSeqlenKv : kShape->GetStorageShape().GetDim(0);
        n1Size = qShape->GetStorageShape().GetDim(1);
        n2Size = kShape->GetStorageShape().GetDim(1);
        dSize = qShape->GetStorageShape().GetDim(2);
        dvSize = vShape->GetStorageShape().GetDim(2);
        auto cuSeqShape = context->GetOptionalInputShape(CU_SEQLENS_Q_INDEX);
        if (cuSeqShape != nullptr && cuSeqShape->GetStorageShape().GetDimNum() > 0) {
            bSize = cuSeqShape->GetStorageShape().GetDim(0) - 1;
        } else {
            bSize = 1;
        }
    } else {
        bSize = qShape->GetStorageShape().GetDim(0);
        s1Size = qShape->GetStorageShape().GetDim(1);
        n1Size = qShape->GetStorageShape().GetDim(2);
        dSize = qShape->GetStorageShape().GetDim(3);
        dvSize = vShape->GetStorageShape().GetDim(3);
        s2Size = kShape->GetStorageShape().GetDim(1);
        n2Size = kShape->GetStorageShape().GetDim(2);
    }

    FlashAttnGradTilingData *tilingData = context->GetTilingData<FlashAttnGradTilingData>();
    tilingData->b = bSize;
    tilingData->s1 = s1Size;
    tilingData->s2 = s2Size;
    tilingData->n1 = n1Size;
    tilingData->n2 = n2Size;
    tilingData->d = dSize;
    tilingData->dv = dvSize;
    tilingData->scaleValue = scaleValue == 0.0f ? 1.0f / sqrt(static_cast<float>(dSize)) : scaleValue;

    bool hasAttenMask = HasAttenMask(context, maskMode);
    // TilingKey: bit[1:0]=template(0=bn2gs1s2,1=bn2,2=bn2s2,3=bn2multiblk),
    //            bit2=layout(0=non-TND,1=TND), bit3=hasAttenMask
    uint64_t tilingKey = static_cast<uint64_t>(0) |
                         (static_cast<uint64_t>(layout) << 2) |
                         (static_cast<uint64_t>(hasAttenMask) << 3);
    context->SetTilingKey(tilingKey);

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
