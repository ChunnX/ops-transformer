/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arch35_checker_adapter.cpp
 * \brief
 */

#include "arch35_checker_adapter.h"

#include "checker_adapter.h"
#include "liv2_checker.h"

namespace optiling {
namespace {
constexpr uint32_t BLOCK_TABLE_RANK = 2U;
constexpr char OP_NAME[] = "LightningIndexerV2";

ge::graphStatus InitCheckerInfo(gert::TilingContext *context, LIV2TilingInfo &tilingInfo,
                                indexer_v2_checker::IndexerV2CheckerInfo &checkerInfo)
{
    const char *opName = context->GetNodeName();
    opName = opName == nullptr ? OP_NAME : opName;
    tilingInfo.platformInfo = context->GetPlatformInfo();
    if (tilingInfo.platformInfo == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "platform_info", "Platform information must be provided");
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    if (platform.GetCoreNumAic() == 0 || platform.GetCoreNumAiv() == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "core_num", "0",
                                              "AIC and AIV core counts must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (context->GetWorkspaceSizes(1) == nullptr || context->GetRawTilingData() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "tiling_buffer",
                                                 "Workspace sizes and raw tiling data must be provided");
        return ge::GRAPH_FAILED;
    }
    if (platform.GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "npu_arch", "non-DAV_3510", "DAV_3510");
        return ge::GRAPH_FAILED;
    }
    tilingInfo.opName = opName;
    tilingInfo.socVersion = platform.GetSocVersion();
    checkerInfo.opName = opName;
    return ge::GRAPH_SUCCESS;
}

void PopulateCheckerTensors(gert::TilingContext *context, indexer_v2_checker::IndexerV2CheckerInfo &info)
{
    info.query = indexer_v2_checker::MakeRequiredTensor(context, QUERY_INDEX);
    info.key = indexer_v2_checker::MakeRequiredTensor(context, KEY_INDEX);
    info.weights = indexer_v2_checker::MakeRequiredTensor(context, WEIGTHS_INDEX);
    info.cuSeqlensQ = indexer_v2_checker::MakeOptionalTensor(context, CU_SEQLENS_Q_INDEX);
    info.cuSeqlensK = indexer_v2_checker::MakeOptionalTensor(context, CU_SEQLENS_K_INDEX);
    info.sequsedQ = indexer_v2_checker::MakeOptionalTensor(context, SEQUSED_Q_INDEX);
    info.sequsedK = indexer_v2_checker::MakeOptionalTensor(context, SEQUSED_K_INDEX);
    info.cmpResidualK = indexer_v2_checker::MakeOptionalTensor(context, CMP_RESIDUAL_K_INDEX);
    info.blockTable = indexer_v2_checker::MakeOptionalTensor(context, BLOCK_TABLE_INDEX);
    info.outputIdxOffset = indexer_v2_checker::MakeOptionalTensor(context, OUTPUT_IDX_OFFSET_INDEX);
    info.metadata = indexer_v2_checker::MakeOptionalTensor(context, METADATA_INDEX);
    info.sparseIndices = indexer_v2_checker::MakeOutputTensor(context, LIGHTNING_INDEXER);
    info.sparseValues = indexer_v2_checker::MakeOutputTensor(context, LIGHTNING_VALUES);
}

ge::graphStatus ParseCheckerAttrs(gert::TilingContext *context, indexer_v2_checker::IndexerV2CheckerInfo &info)
{
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attrs", "Operator attributes must be provided");
        return ge::GRAPH_FAILED;
    }
    const int32_t *topk = attrs->GetAttrPointer<int32_t>(ATTR_TOPK_INDEX);
    const int32_t *maxSeqlenQ = attrs->GetAttrPointer<int32_t>(ATTR_MAX_SEQLEN_Q_INDEX);
    const char *layoutQ = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    const char *layoutK = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    const int32_t *maskMode = attrs->GetAttrPointer<int32_t>(ATTR_MASK_MODE_INDEX);
    const int64_t *cmpRatio = attrs->GetAttrPointer<int64_t>(ATTR_CMP_RATIO_INDEX);
    const int32_t *returnValue = attrs->GetAttrPointer<int32_t>(ATTR_RETURN_VALUE_INDEX);
    if (topk == nullptr || maxSeqlenQ == nullptr || layoutQ == nullptr || layoutK == nullptr || maskMode == nullptr ||
        cmpRatio == nullptr || returnValue == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attrs", "Required/defaulted attributes are missing");
        return ge::GRAPH_FAILED;
    }
    info.topk = *topk;
    info.maxSeqlenQ = *maxSeqlenQ;
    info.layoutQText = layoutQ;
    info.layoutKText = layoutK;
    info.maskMode = *maskMode;
    info.cmpRatio = *cmpRatio;
    info.returnValue = *returnValue;
    indexer_v2_checker::PopulateDerivedInfo(info);
    return ge::GRAPH_SUCCESS;
}

void SetDataLayouts(const indexer_v2_checker::IndexerV2CheckerInfo &info, LIV2TilingInfo &tilingInfo)
{
    if (info.layoutQ == indexer_v2_checker::CheckerLayout::TND) {
        tilingInfo.inputQLayout = DataLayout::TND;
    } else {
        tilingInfo.inputQLayout = DataLayout::BSND;
    }
    if (info.layoutK == indexer_v2_checker::CheckerLayout::TND) {
        tilingInfo.inputKLayout = DataLayout::TND;
    } else if (info.layoutK == indexer_v2_checker::CheckerLayout::PA_BBND) {
        tilingInfo.inputKLayout = DataLayout::PA_BBND;
    } else {
        tilingInfo.inputKLayout = DataLayout::BSND;
    }
}

void PopulateTilingInfo(const indexer_v2_checker::IndexerV2CheckerInfo &info, LIV2TilingInfo &tilingInfo)
{
    tilingInfo.bSize = static_cast<uint32_t>(info.batch);
    tilingInfo.n1Size = static_cast<uint32_t>(info.qHeads);
    tilingInfo.n2Size = static_cast<uint32_t>(info.kHeads);
    tilingInfo.s1Size = info.layoutQ == indexer_v2_checker::CheckerLayout::BSND ? static_cast<uint32_t>(info.qSeq) : 0U;
    tilingInfo.s2Size = info.kSeq;
    tilingInfo.qkHeadDim = static_cast<uint32_t>(info.headDim);
    tilingInfo.gSize = info.kHeads == 0 ? 0U : static_cast<uint32_t>(info.qHeads / info.kHeads);
    tilingInfo.pageAttentionFlag = info.layoutK == indexer_v2_checker::CheckerLayout::PA_BBND;
    tilingInfo.blockSize = static_cast<int32_t>(info.blockSize);
    const gert::Shape *blockShape = info.blockTable.GetShape();
    if (tilingInfo.pageAttentionFlag && blockShape != nullptr && blockShape->GetDimNum() >= BLOCK_TABLE_RANK) {
        tilingInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape->GetDim(DIM_IDX_ONE));
    }
    tilingInfo.inputQType = info.query.desc->GetDataType();
    tilingInfo.inputKType = info.key.desc->GetDataType();
    tilingInfo.outputType = info.sparseIndices.desc->GetDataType();
    tilingInfo.maskMode = static_cast<int32_t>(info.maskMode);
    tilingInfo.topk = static_cast<uint32_t>(info.topk);
    tilingInfo.maxSeqlenQ = static_cast<int32_t>(info.maxSeqlenQ);
    tilingInfo.cmpRatio = info.cmpRatio;
    tilingInfo.returnValue = static_cast<uint32_t>(info.returnValue);
    if (info.key.stride != nullptr && info.key.stride->GetDimNum() > 0) {
        tilingInfo.keyStride0 = static_cast<uint32_t>(info.key.stride->GetStride(0));
    }
    SetDataLayouts(info, tilingInfo);
}
} // namespace

ge::graphStatus ParseAndCheckLIV2Arch35(gert::TilingContext *context, LIV2TilingInfo &tilingInfo)
{
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "context", "Tiling context must not be null");
        return ge::GRAPH_FAILED;
    }
    indexer_v2_checker::IndexerV2CheckerInfo checkerInfo;
    if (InitCheckerInfo(context, tilingInfo, checkerInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PopulateCheckerTensors(context, checkerInfo);
    if (ParseCheckerAttrs(context, checkerInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const indexer_v2_checker::LIV2Checker checker(checkerInfo);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PopulateTilingInfo(checkerInfo, tilingInfo);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
