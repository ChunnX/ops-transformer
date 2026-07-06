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
#include "log/log.h"
#include "sparse_flash_mla_softmax_l1_norm_tiling.h"
#include "../op_kernel/sparse_flash_mla_softmax_l1_norm_metadata.h"

using namespace ge;

namespace optiling {

static ge::graphStatus SparseFlashMlaSoftmaxL1NormTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("SparseFlashMlaSoftmaxL1Norm", "Tiling context is null."),
                return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE(context->GetNodeName(), "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivNum == 0,
                OP_LOGE(context->GetNodeName(), "aivNum is 0."),
                return ge::GRAPH_FAILED);

    SmlaSoftmaxL1NormTilingData tilingData;
    context->SetBlockDim(aivNum);

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    // 读取 attrs
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "attrs is nullptr."),
                return ge::GRAPH_FAILED);
    const float *scaleValuePtr = attrs->GetAttrPointer<float>(ATTR_SCALE_VALUE_INDEX);
    const int64_t *cmpRatioPtr = attrs->GetAttrPointer<int64_t>(ATTR_CMP_RATIO_INDEX);
    const int64_t *maskModePtr = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);
    const char *layoutQPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_Q_INDEX);

    int64_t maskMode = (maskModePtr != nullptr) ? *maskModePtr : 0;
    int32_t cmpRatio = (cmpRatioPtr != nullptr) ? static_cast<int32_t>(*cmpRatioPtr) : 1;
    float scaleValue = (scaleValuePtr != nullptr) ? *scaleValuePtr : 1.0f;
    std::string layoutQStr = (layoutQPtr != nullptr) ? std::string(layoutQPtr) : "BSND";
    SmlaDataLayout layout = (layoutQStr == "TND") ? SmlaDataLayout::TND : SmlaDataLayout::BSND;

    // 读取 query/key shape
    const auto *queryStorageShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(queryStorageShape == nullptr,
                OP_LOGE(context->GetNodeName(), "query shape is nullptr."),
                return ge::GRAPH_FAILED);
    const auto *keyStorageShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(keyStorageShape == nullptr,
                OP_LOGE(context->GetNodeName(), "key shape is nullptr."),
                return ge::GRAPH_FAILED);
    const auto &queryShape = queryStorageShape->GetStorageShape();
    const auto &keyShape = keyStorageShape->GetStorageShape();

    int64_t bSize = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t dSize = 0;
    if (layout == SmlaDataLayout::BSND) {
        bSize = queryShape.GetDim(0);
        s1Size = queryShape.GetDim(1);
        n1Size = queryShape.GetDim(2);
        dSize = queryShape.GetDim(3);
        s2Size = keyShape.GetDim(1);
        n2Size = keyShape.GetDim(2);
    } else { // TND
        s1Size = queryShape.GetDim(0);
        n1Size = queryShape.GetDim(1);
        dSize = queryShape.GetDim(2);
        s2Size = keyShape.GetDim(0);
        n2Size = keyShape.GetDim(1);
        const gert::Tensor *cuSeqQTensor = context->GetOptionalInputTensor(CUM_SEQ_Q_INDEX);
        if (cuSeqQTensor != nullptr) {
            bSize = static_cast<int64_t>(cuSeqQTensor->GetShapeSize()) - 1;
        }
    }

    int64_t gSize = (n2Size > 0) ? (n1Size / n2Size) : 0;

    // 可选输入存在性判断
    const gert::Tensor *seqUsedQTensor = context->GetOptionalInputTensor(SEQ_USED_Q_INDEX);
    const gert::Tensor *seqUsedKTensor = context->GetOptionalInputTensor(SEQ_USED_K_INDEX);
    const gert::Tensor *topkLengthTensor = context->GetOptionalInputTensor(TOPK_LENGTH_INDEX);
    uint8_t hasUsedSeqQ = (seqUsedQTensor != nullptr) ? 1 : 0;
    uint8_t hasUsedSeqK = (seqUsedKTensor != nullptr) ? 1 : 0;
    uint8_t hasTopK = (topkLengthTensor != nullptr) ? 1 : 0;

    // selectedBlockCount: 暂设为0，后续根据 sparse_indices 决定
    int64_t selectedBlockCount = 0;

    tilingData.set_b(bSize);
    tilingData.set_n2(n2Size);
    tilingData.set_g(gSize);
    tilingData.set_s1(s1Size);
    tilingData.set_s2(s2Size);
    tilingData.set_d(dSize);
    tilingData.set_selectedBlockCount(selectedBlockCount);
    tilingData.set_usedCoreNum(static_cast<int64_t>(aivNum));
    tilingData.set_layout(static_cast<int64_t>(layout));
    tilingData.set_maskMode(static_cast<int32_t>(maskMode));
    tilingData.set_scaleValue(scaleValue);
    tilingData.set_cmpRatio(cmpRatio);
    tilingData.set_hasUsedSeqQ(hasUsedSeqQ);
    tilingData.set_hasUsedSeqK(hasUsedSeqK);
    tilingData.set_hasTopK(hasTopK);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetTilingKey(0);

    OP_LOGI(context->GetNodeName(), "SparseFlashMlaSoftmaxL1Norm tiling completed, blockDim=%u.", aivNum);
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
