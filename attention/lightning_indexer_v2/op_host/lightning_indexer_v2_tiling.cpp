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
 * \file lightning_indexer_v2_tiling.cpp
 * \brief
 */

#include "lightning_indexer_v2_tiling.h"
#include "../op_kernel/lightning_indexer_v2_template_tiling_key.h"
#include "checkers/checker_adapter.h"
#include "checkers/liv2_checker.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace {
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
constexpr uint32_t LI_WEIGHTS_INDEX = 2;
constexpr uint32_t LI_CU_SEQLENS_K_INDEX = 4;
constexpr uint32_t LI_SEQUSED_Q_INDEX = 5;
constexpr uint32_t LI_SEQUSED_K_INDEX = 6;
constexpr uint32_t LI_CMP_RESIDUAL_K_INDEX = 7;
constexpr uint32_t LI_OUTPUT_IDX_OFFSET_INDEX = 9;
constexpr uint32_t LI_METADATA_INDEX = 10;
constexpr uint32_t LI_SPARSE_VALUES_INDEX = 1;
constexpr uint32_t LI_ATTR_MAX_SEQLEN_Q_INDEX = 1;
constexpr uint32_t BLOCK_TABLE_RANK = 2U;
constexpr uint32_t MM1_RES_ELEM_SIZE = 4;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t M_BASE_SIZE = 512;
constexpr uint32_t S2_BASE_SIZE = 512;
constexpr uint32_t V1_RES_ELEM_SIZE = 4;
constexpr uint32_t V1_RES_ELEM_TYPE = 2;
constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8;
constexpr uint32_t V1_DECODE_PARAM_NUM = 16;
constexpr uint32_t V1_DECODE_DATA_NUM = 2;
constexpr uint32_t S1_BASE_SIZE = 8;
constexpr uint32_t TOPK_MAX_SIZE = 8192;
constexpr uint32_t LI_3510_S1_BASE = 4;
constexpr uint32_t LI_3510_S2_BASE = 128;
constexpr char OP_NAME[] = "LightningIndexerV2";

ge::graphStatus InitParserContext(gert::TilingContext *context, LIV2TilingInfo &liInfo)
{
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "context", "Tiling context must not be null");
        return ge::GRAPH_FAILED;
    }
    const char *opName = context->GetNodeName();
    opName = opName == nullptr ? OP_NAME : opName;
    liInfo.platformInfo = context->GetPlatformInfo();
    if (liInfo.platformInfo == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "platform_info", "Platform information must be provided");
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(liInfo.platformInfo);
    if (platform.GetCoreNumAic() == 0 || platform.GetCoreNumAiv() == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "core_num", "0",
                                              "AIC and AIV core counts must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (context->GetWorkspaceSizes(1) == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "workspace_sizes",
                                                 "Workspace sizes must be provided by the tiling context");
        return ge::GRAPH_FAILED;
    }
    if (context->GetRawTilingData() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "raw_tiling_data",
                                                 "Raw tiling data must be provided by the tiling context");
        return ge::GRAPH_FAILED;
    }
    const NpuArch npuArch = platform.GetCurNpuArch();
    if (npuArch != NpuArch::DAV_2201 && npuArch != NpuArch::DAV_3510) {
        const std::string actual = std::to_string(static_cast<int32_t>(npuArch));
        OP_LOGE_FOR_INVALID_VALUE(opName, "npu_arch", actual.c_str(), "DAV_2201 or DAV_3510");
        return ge::GRAPH_FAILED;
    }
    liInfo.checkerInfo.opName = opName;
    liInfo.checkerInfo.arch35 = npuArch == NpuArch::DAV_3510;
    return ge::GRAPH_SUCCESS;
}

void PopulateCheckerTensors(gert::TilingContext *context, indexer_v2_checker::IndexerV2CheckerInfo &info)
{
    info.query = indexer_v2_checker::MakeRequiredTensor(context, QUERY_INDEX);
    info.key = indexer_v2_checker::MakeRequiredTensor(context, KEY_INDEX);
    info.weights = indexer_v2_checker::MakeRequiredTensor(context, LI_WEIGHTS_INDEX);
    info.cuSeqlensQ = indexer_v2_checker::MakeOptionalTensor(context, CU_SEQLENS_Q_INDEX);
    info.cuSeqlensK = indexer_v2_checker::MakeOptionalTensor(context, LI_CU_SEQLENS_K_INDEX);
    info.sequsedQ = indexer_v2_checker::MakeOptionalTensor(context, LI_SEQUSED_Q_INDEX);
    info.sequsedK = indexer_v2_checker::MakeOptionalTensor(context, LI_SEQUSED_K_INDEX);
    info.cmpResidualK = indexer_v2_checker::MakeOptionalTensor(context, LI_CMP_RESIDUAL_K_INDEX);
    info.blockTable = indexer_v2_checker::MakeOptionalTensor(context, BLOCK_TABLE_INDEX);
    info.outputIdxOffset = indexer_v2_checker::MakeOptionalTensor(context, LI_OUTPUT_IDX_OFFSET_INDEX);
    info.metadata = indexer_v2_checker::MakeOptionalTensor(context, LI_METADATA_INDEX);
    info.sparseIndices = indexer_v2_checker::MakeOutputTensor(context, LIGHTNING_INDEXER);
    info.sparseValues = indexer_v2_checker::MakeOutputTensor(context, LI_SPARSE_VALUES_INDEX);
}

ge::graphStatus ParseCheckerAttrs(gert::TilingContext *context, indexer_v2_checker::IndexerV2CheckerInfo &info)
{
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attrs", "Operator attributes must be provided");
        return ge::GRAPH_FAILED;
    }
    const int32_t *topk = attrs->GetAttrPointer<int32_t>(ATTR_TOPK_INDEX);
    const int32_t *maxSeqlenQ = attrs->GetAttrPointer<int32_t>(LI_ATTR_MAX_SEQLEN_Q_INDEX);
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
    info.returnValue = static_cast<int64_t>(*returnValue);
    indexer_v2_checker::PopulateDerivedInfo(info);
    return ge::GRAPH_SUCCESS;
}

void SetDataLayouts(const indexer_v2_checker::IndexerV2CheckerInfo &info, LIV2TilingInfo &liInfo)
{
    if (info.layoutQ == indexer_v2_checker::CheckerLayout::BSND) {
        liInfo.inputQLayout = DataLayout::BSND;
    } else if (info.layoutQ == indexer_v2_checker::CheckerLayout::TND) {
        liInfo.inputQLayout = DataLayout::TND;
    }
    if (info.layoutK == indexer_v2_checker::CheckerLayout::BSND) {
        liInfo.inputKLayout = DataLayout::BSND;
    } else if (info.layoutK == indexer_v2_checker::CheckerLayout::TND) {
        liInfo.inputKLayout = DataLayout::TND;
    } else if (info.layoutK == indexer_v2_checker::CheckerLayout::PA_BBND) {
        liInfo.inputKLayout = DataLayout::PA_BBND;
    }
}

void PopulateTilingInfo(const indexer_v2_checker::IndexerV2CheckerInfo &info, LIV2TilingInfo &liInfo)
{
    liInfo.bSize = static_cast<uint32_t>(info.batch);
    liInfo.s1Size = info.layoutQ == indexer_v2_checker::CheckerLayout::BSND ? static_cast<uint32_t>(info.qSeq) : 0U;
    liInfo.s2Size = info.kSeq;
    liInfo.gSize = info.kHeads == 0 ? 0U : static_cast<uint32_t>(info.qHeads / info.kHeads);
    liInfo.pageAttentionFlag = info.layoutK == indexer_v2_checker::CheckerLayout::PA_BBND;
    liInfo.blockSize = static_cast<int32_t>(info.blockSize);
    const gert::Shape *blockShape = info.blockTable.GetShape();
    if (liInfo.pageAttentionFlag && blockShape != nullptr && blockShape->GetDimNum() >= BLOCK_TABLE_RANK) {
        liInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape->GetDim(DIM_IDX_ONE));
    }
    if (info.query.desc != nullptr) {
        liInfo.inputQType = info.query.desc->GetDataType();
    }
    if (info.key.desc != nullptr) {
        liInfo.inputKType = info.key.desc->GetDataType();
    }
    if (info.sparseIndices.desc != nullptr) {
        liInfo.outputType = info.sparseIndices.desc->GetDataType();
    }
    liInfo.maskMode = static_cast<int32_t>(info.maskMode);
    liInfo.topk = static_cast<uint32_t>(info.topk);
    liInfo.cmpRatio = info.cmpRatio;
    liInfo.returnValue = static_cast<uint32_t>(info.returnValue);
    if (info.key.stride != nullptr && info.key.stride->GetDimNum() > 0) {
        liInfo.keyStride0 = static_cast<uint32_t>(info.key.stride->GetStride(0));
    }
}

uint64_t CalculateWorkspaceSize(const LIV2TilingInfo &tilingInfo, platform_ascendc::PlatformAscendC &platform,
                                uint32_t aicNum)
{
    uint64_t workspaceSize = platform.GetLibApiWorkSpaceSize();
    if (platform.GetCurNpuArch() == NpuArch::DAV_3510) {
        workspaceSize += static_cast<uint64_t>(LI_3510_S1_BASE) *
                         ((tilingInfo.s2Size + LI_3510_S2_BASE - 1) / LI_3510_S2_BASE) * LI_3510_S2_BASE *
                         sizeof(uint32_t) * aicNum;
    } else {
        const uint64_t mm1ResSize = static_cast<uint64_t>(M_BASE_SIZE) * S2_BASE_SIZE;
        workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
    }
    workspaceSize += static_cast<uint64_t>(V1_DECODE_DATA_NUM) * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE *
                     V1_RES_ELEM_SIZE * aicNum;
    workspaceSize += static_cast<uint64_t>(V1_DECODE_DATA_NUM) * S1_BASE_SIZE * V1_DECODE_PARAM_NUM *
                     V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    return workspaceSize;
}

void FillTilingData(const LIV2TilingInfo &info, uint32_t blockDim, LIV2TilingData &data)
{
    data.set_bSize(info.bSize);
    data.set_s2Size(info.s2Size);
    data.set_s1Size(info.s1Size);
    data.set_topk(info.topk);
    data.set_gSize(info.gSize);
    data.set_blockSize(info.blockSize);
    data.set_maxBlockNumPerBatch(info.maxBlockNumPerBatch);
    data.set_maskMode(info.maskMode);
    data.set_preTokens(info.preTokens);
    data.set_nextTokens(info.nextTokens);
    data.set_cmpRatio(info.cmpRatio);
    data.set_keyStride0(info.keyStride0);
    data.set_returnValue(info.returnValue);
    data.set_usedCoreNum(blockDim);
    data.set_batchSupperFlag(info.batchSupperFlag);
}

ge::graphStatus SaveTilingData(gert::TilingContext *context, LIV2TilingData &data)
{
    auto *rawTilingData = context->GetRawTilingData();
    auto *rawTilingBuffer = rawTilingData == nullptr ? nullptr : rawTilingData->GetData();
    if (rawTilingBuffer == nullptr || rawTilingData->GetCapacity() < data.GetDataSize()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "raw_tiling_data",
                                                 "Raw tiling data buffer is missing or insufficient");
        return ge::GRAPH_FAILED;
    }
    data.SaveToBuffer(rawTilingBuffer, rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void SetTilingKey(gert::TilingContext *context, const LIV2TilingInfo &info)
{
    const uint32_t inputQType = static_cast<uint32_t>(info.inputQType);
    const uint32_t inputKType = static_cast<uint32_t>(info.inputKType);
    const uint32_t outputType = static_cast<uint32_t>(info.outputType);
    const uint32_t pageAttentionFlag = static_cast<uint32_t>(info.pageAttentionFlag);
    const uint32_t inputQLayout = static_cast<uint32_t>(info.inputQLayout);
    const uint32_t inputKLayout = static_cast<uint32_t>(info.inputKLayout);
    const uint32_t weightTypeFlag = 0;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout,
                                                  inputKLayout, weightTypeFlag);
    context->SetTilingKey(tilingKey);
}
} // namespace

ge::graphStatus LIV2InfoParser::Parse(LIV2TilingInfo &liInfo) const
{
    if (InitParserContext(context_, liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto &checkerInfo = liInfo.checkerInfo;
    PopulateCheckerTensors(context_, checkerInfo);
    if (ParseCheckerAttrs(context_, checkerInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetDataLayouts(checkerInfo, liInfo);
    PopulateTilingInfo(checkerInfo, liInfo);
    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForLightningIndexerV2(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------LightningIndexerV2Tiling类成员函数定义-----------------------
ge::graphStatus LightningIndexerV2Tiling::DoTiling(const LIV2TilingInfo &tilingInfo)
{
    if (context_ == nullptr || tilingInfo.platformInfo == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("LightningIndexerV2", "context and platform_info",
                                                 "Tiling context and platform information must not be null");
        return ge::GRAPH_FAILED;
    }
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    if (workSpaces == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("LightningIndexerV2", "workspace_sizes and raw_tiling_data",
                                                 "Workspace sizes and raw tiling data must be provided");
        return ge::GRAPH_FAILED;
    }
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    workSpaces[0] = CalculateWorkspaceSize(tilingInfo, ascendcPlatform, aicNum);

    FillTilingData(tilingInfo, blockDim, tilingData_);
    if (SaveTilingData(context_, tilingData_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetTilingKey(context_, tilingInfo);
    context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForLightningIndexerV2(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("LightningIndexerV2", "context", "Tiling context must not be null");
        return ge::GRAPH_FAILED;
    }
    LIV2TilingInfo liV2Info;
    LIV2InfoParser LIV2InfoParser(context);
    if (LIV2InfoParser.Parse(liV2Info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    indexer_v2_checker::LIV2Checker checker(liV2Info.checkerInfo);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    LightningIndexerV2Tiling liTiling(context);
    return liTiling.DoTiling(liV2Info);
}
// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(LightningIndexerV2)
    .Tiling(TilingForLightningIndexerV2)
    .TilingParse<LIV2CompileInfo>(TilingPrepareForLightningIndexerV2);
} // namespace optiling
