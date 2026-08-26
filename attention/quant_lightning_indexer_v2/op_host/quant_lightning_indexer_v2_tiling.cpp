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
 * \file quant_lightning_indexer_v2_tiling.cpp
 * \brief
 */

#include "quant_lightning_indexer_v2_tiling.h"

#include "../op_kernel/quant_lightning_indexer_v2_template_tiling_key.h"
#include "../../lightning_indexer_v2/op_host/checkers/checker_adapter.h"
#include "checkers/qliv2_checker.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace {
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
constexpr uint32_t QLI_WEIGHTS_INDEX = 2;
constexpr uint32_t QLI_QUERY_DESCALE_INDEX = 3;
constexpr uint32_t QLI_CU_SEQLENS_K_INDEX = 6;
constexpr uint32_t QLI_SEQUSED_Q_INDEX = 7;
constexpr uint32_t QLI_SEQUSED_K_INDEX = 8;
constexpr uint32_t QLI_CMP_RESIDUAL_K_INDEX = 9;
constexpr uint32_t QLI_OUTPUT_IDX_OFFSET_INDEX = 11;
constexpr uint32_t QLI_METADATA_INDEX = 12;
constexpr uint32_t QLI_SPARSE_VALUES_INDEX = 1;
constexpr uint32_t BLOCK_TABLE_RANK = 2U;
constexpr uint32_t MM1_RES_ELEM_SIZE = 4;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t M_BASE_SIZE = 256;
constexpr uint32_t S2_BASE_SIZE = 2048;
constexpr uint32_t V1_RES_ELEM_SIZE = 4;
constexpr uint32_t V1_RES_ELEM_TYPE = 2;
constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8;
constexpr uint32_t V1_DECODE_PARAM_NUM = 16;
constexpr uint32_t V1_DECODE_DATA_NUM = 2;
constexpr uint32_t S1_BASE_SIZE = 4;
constexpr uint32_t TOPK_MAX_SIZE = 2048;
constexpr uint32_t TOPK_MAX_SIZE_950 = 8192;
constexpr uint32_t S1_BASE_SIZE_950 = 4;
constexpr uint32_t S2_BASE_SIZE_950 = 128;
constexpr char OP_NAME[] = "QuantLightningIndexerV2";

ge::graphStatus InitParserContext(gert::TilingContext *context, QLIV2TilingInfo &qliv2Info)
{
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "context", "Tiling context must not be null");
        return ge::GRAPH_FAILED;
    }
    const char *opName = context->GetNodeName();
    opName = opName == nullptr ? OP_NAME : opName;
    qliv2Info.platformInfo = context->GetPlatformInfo();
    if (qliv2Info.platformInfo == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "platform_info", "Platform information must be provided");
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(qliv2Info.platformInfo);
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
    qliv2Info.checkerInfo.opName = opName;
    qliv2Info.checkerInfo.quantized = true;
    qliv2Info.checkerInfo.arch35 = npuArch == NpuArch::DAV_3510;
    return ge::GRAPH_SUCCESS;
}

void PopulateCheckerTensors(gert::TilingContext *context, indexer_v2_checker::IndexerV2CheckerInfo &info)
{
    info.query = indexer_v2_checker::MakeRequiredTensor(context, QUERY_INDEX);
    info.key = indexer_v2_checker::MakeRequiredTensor(context, KEY_INDEX);
    info.weights = indexer_v2_checker::MakeRequiredTensor(context, QLI_WEIGHTS_INDEX);
    info.queryDescale = indexer_v2_checker::MakeRequiredTensor(context, QLI_QUERY_DESCALE_INDEX);
    info.keyDescale = indexer_v2_checker::MakeRequiredTensor(context, KEY_DEQUANT_SCALE_INDEX);
    info.cuSeqlensQ = indexer_v2_checker::MakeOptionalTensor(context, CU_SEQLENS_Q_INDEX);
    info.cuSeqlensK = indexer_v2_checker::MakeOptionalTensor(context, QLI_CU_SEQLENS_K_INDEX);
    info.sequsedQ = indexer_v2_checker::MakeOptionalTensor(context, QLI_SEQUSED_Q_INDEX);
    info.sequsedK = indexer_v2_checker::MakeOptionalTensor(context, QLI_SEQUSED_K_INDEX);
    info.cmpResidualK = indexer_v2_checker::MakeOptionalTensor(context, QLI_CMP_RESIDUAL_K_INDEX);
    info.blockTable = indexer_v2_checker::MakeOptionalTensor(context, BLOCK_TABLE_INDEX);
    info.outputIdxOffset = indexer_v2_checker::MakeOptionalTensor(context, QLI_OUTPUT_IDX_OFFSET_INDEX);
    info.metadata = indexer_v2_checker::MakeOptionalTensor(context, QLI_METADATA_INDEX);
    info.sparseIndices = indexer_v2_checker::MakeOutputTensor(context, SPARSE_INDICES_INDEX);
    info.sparseValues = indexer_v2_checker::MakeOutputTensor(context, QLI_SPARSE_VALUES_INDEX);
}

ge::graphStatus ParseCheckerAttrs(gert::TilingContext *context, indexer_v2_checker::IndexerV2CheckerInfo &info)
{
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attrs", "Operator attributes must be provided");
        return ge::GRAPH_FAILED;
    }
    const int64_t *topk = attrs->GetInt(ATTR_TOPK_INDEX);
    const int64_t *quantMode = attrs->GetInt(ATTR_QUANT_MODE_INDEX);
    const int64_t *maxSeqlenQ = attrs->GetInt(ATTR_MAX_SEQLEN_Q_INDEX);
    const char *layoutQ = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    const char *layoutK = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    const int64_t *maskMode = attrs->GetInt(ATTR_MASK_MODE_INDEX);
    const int64_t *cmpRatio = attrs->GetInt(ATTR_CMP_RATIO_INDEX);
    const int64_t *returnValue = attrs->GetInt(ATTR_RETURN_VALUE_INDEX);
    if (topk == nullptr || quantMode == nullptr || maxSeqlenQ == nullptr || layoutQ == nullptr || layoutK == nullptr ||
        maskMode == nullptr || cmpRatio == nullptr || returnValue == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attrs", "Required/defaulted attributes are missing");
        return ge::GRAPH_FAILED;
    }
    info.topk = *topk;
    info.quantMode = *quantMode;
    info.maxSeqlenQ = *maxSeqlenQ;
    info.layoutQText = layoutQ;
    info.layoutKText = layoutK;
    info.maskMode = *maskMode;
    info.cmpRatio = *cmpRatio;
    info.returnValue = *returnValue;
    indexer_v2_checker::PopulateDerivedInfo(info);
    return ge::GRAPH_SUCCESS;
}

void SetDataLayouts(const indexer_v2_checker::IndexerV2CheckerInfo &info, QLIV2TilingInfo &qliv2Info)
{
    if (info.layoutQ == indexer_v2_checker::CheckerLayout::BSND) {
        qliv2Info.inputQLayout = DataLayout::BSND;
    } else if (info.layoutQ == indexer_v2_checker::CheckerLayout::TND) {
        qliv2Info.inputQLayout = DataLayout::TND;
    }
    if (info.layoutK == indexer_v2_checker::CheckerLayout::BSND) {
        qliv2Info.inputKLayout = DataLayout::BSND;
    } else if (info.layoutK == indexer_v2_checker::CheckerLayout::TND) {
        qliv2Info.inputKLayout = DataLayout::TND;
    } else if (info.layoutK == indexer_v2_checker::CheckerLayout::PA_BBND) {
        qliv2Info.inputKLayout = DataLayout::PA_BBND;
    }
}

void PopulateTilingInfo(const indexer_v2_checker::IndexerV2CheckerInfo &info, QLIV2TilingInfo &qliv2Info)
{
    qliv2Info.bSize = static_cast<uint32_t>(info.batch);
    qliv2Info.s1Size = info.layoutQ == indexer_v2_checker::CheckerLayout::BSND ? static_cast<uint32_t>(info.qSeq) : 0U;
    qliv2Info.s2Size = info.kSeq;
    qliv2Info.qkHeadDim = static_cast<uint32_t>(info.headDim);
    qliv2Info.gSize = info.kHeads == 0 ? 0U : static_cast<uint32_t>(info.qHeads / info.kHeads);
    qliv2Info.pageAttentionFlag = info.layoutK == indexer_v2_checker::CheckerLayout::PA_BBND;
    qliv2Info.blockSize = static_cast<int32_t>(info.blockSize);
    const gert::Shape *blockShape = info.blockTable.GetShape();
    if (qliv2Info.pageAttentionFlag && blockShape != nullptr && blockShape->GetDimNum() >= BLOCK_TABLE_RANK) {
        qliv2Info.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape->GetDim(DIM_IDX_ONE));
    }
    if (info.query.desc != nullptr) {
        qliv2Info.inputQType = info.query.desc->GetDataType();
    }
    if (info.key.desc != nullptr) {
        qliv2Info.inputKType = info.key.desc->GetDataType();
    }
    if (info.sparseIndices.desc != nullptr) {
        qliv2Info.outputType = info.sparseIndices.desc->GetDataType();
    }
    qliv2Info.sparseMode = static_cast<int32_t>(info.maskMode);
    qliv2Info.sparseCount = static_cast<uint32_t>(info.topk);
    qliv2Info.quantMode = static_cast<uint32_t>(info.quantMode);
    qliv2Info.cmpRatio = static_cast<uint32_t>(info.cmpRatio);
    qliv2Info.returnValue = info.returnValue != 0;
    qliv2Info.maxSeqlenQ = static_cast<int32_t>(info.maxSeqlenQ);
}

void PopulateStrideInfo(const indexer_v2_checker::IndexerV2CheckerInfo &info, QLIV2TilingInfo &qliv2Info)
{
    if (info.key.stride != nullptr && info.key.stride->GetDimNum() > 0) {
        qliv2Info.keyStride0 = static_cast<uint32_t>(info.key.stride->GetStride(0));
        if (info.quantMode == QUANT_MODE_MXFP4) {
            qliv2Info.keyStride0 /= MXFP4_PACK_NUM;
        }
    }
    if (info.keyDescale.stride != nullptr && info.keyDescale.stride->GetDimNum() > 0) {
        qliv2Info.keyDequantScaleStride0 = static_cast<uint32_t>(info.keyDescale.stride->GetStride(0));
    } else if (info.quantMode == QUANT_MODE_MXFP8 || info.quantMode == QUANT_MODE_MXFP4) {
        qliv2Info.keyDequantScaleStride0 =
            static_cast<uint32_t>(qliv2Info.blockSize) * (qliv2Info.qkHeadDim / MX_SCALE_GROUP_SIZE);
    }
}

uint64_t CalculateWorkspaceSize(const QLIV2TilingInfo &info, platform_ascendc::PlatformAscendC &platform,
                                uint32_t aicNum)
{
    uint64_t workspaceSize = platform.GetLibApiWorkSpaceSize();
    if (platform.GetCurNpuArch() == NpuArch::DAV_3510) {
        workspaceSize += static_cast<uint64_t>(S1_BASE_SIZE_950) *
                         ((info.s2Size + S2_BASE_SIZE_950 - 1) / S2_BASE_SIZE_950) * S2_BASE_SIZE_950 *
                         sizeof(uint16_t) * aicNum;
        workspaceSize += static_cast<uint64_t>(V1_DECODE_DATA_NUM) * S1_BASE_SIZE * V1_RES_ELEM_TYPE *
                         TOPK_MAX_SIZE_950 * V1_RES_ELEM_SIZE * aicNum;
    } else {
        const uint64_t mm1ResSize = static_cast<uint64_t>(M_BASE_SIZE) * S2_BASE_SIZE;
        workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
        workspaceSize += static_cast<uint64_t>(V1_DECODE_DATA_NUM) * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE *
                         V1_RES_ELEM_SIZE * aicNum;
    }
    workspaceSize += static_cast<uint64_t>(V1_DECODE_DATA_NUM) * S1_BASE_SIZE * V1_DECODE_PARAM_NUM *
                     V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    return workspaceSize;
}

void FillTilingData(const QLIV2TilingInfo &info, uint32_t blockDim, QLIV2TilingData &data)
{
    data.set_bSize(info.bSize);
    data.set_s2Size(info.s2Size);
    data.set_s1Size(info.s1Size);
    data.set_sparseCount(info.sparseCount);
    data.set_gSize(info.gSize);
    data.set_blockSize(info.blockSize);
    data.set_maxBlockNumPerBatch(info.maxBlockNumPerBatch);
    data.set_sparseMode(info.sparseMode);
    data.set_cmpRatio(info.cmpRatio);
    data.set_returnValue(info.returnValue);
    data.set_maxSeqlenQ(info.maxSeqlenQ);
    data.set_keyStride0(info.keyStride0);
    data.set_keyDequantScaleStride0(info.keyDequantScaleStride0);
    data.set_quantMode(info.quantMode);
    data.set_usedCoreNum(blockDim);
}

ge::graphStatus SaveTilingData(gert::TilingContext *context, QLIV2TilingData &data)
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

void SetTilingKey(gert::TilingContext *context, const QLIV2TilingInfo &info)
{
    const uint32_t inputQType = static_cast<uint32_t>(info.inputQType);
    const uint32_t inputKType = static_cast<uint32_t>(info.inputKType);
    const uint32_t outputType = static_cast<uint32_t>(info.outputType);
    const uint32_t pageAttentionFlag = static_cast<uint32_t>(info.pageAttentionFlag);
    const uint32_t inputQLayout = static_cast<uint32_t>(info.inputQLayout);
    const uint32_t inputKLayout = static_cast<uint32_t>(info.inputKLayout);
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout, inputKLayout);
    context->SetTilingKey(tilingKey);
}
} // namespace

ge::graphStatus QLIV2InfoParser::Parse(QLIV2TilingInfo &qliv2Info) const
{
    if (InitParserContext(context_, qliv2Info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto &checkerInfo = qliv2Info.checkerInfo;
    PopulateCheckerTensors(context_, checkerInfo);
    if (ParseCheckerAttrs(context_, checkerInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetDataLayouts(checkerInfo, qliv2Info);
    PopulateTilingInfo(checkerInfo, qliv2Info);
    PopulateStrideInfo(checkerInfo, qliv2Info);
    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForQuantLightningIndexerV2(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------QuantLightningIndexerV2Tiling类成员函数定义-----------------------
ge::graphStatus QuantLightningIndexerV2Tiling::DoTiling(const QLIV2TilingInfo &tilingInfo)
{
    if (context_ == nullptr || tilingInfo.platformInfo == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantLightningIndexerV2", "context and platform_info",
                                                 "Tiling context and platform information must not be null");
        return ge::GRAPH_FAILED;
    }
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    if (workSpaces == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantLightningIndexerV2", "workspace_sizes and raw_tiling_data",
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
ge::graphStatus TilingForQuantLightningIndexerV2(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantLightningIndexerV2", "context",
                                                 "Tiling context must not be null");
        return ge::GRAPH_FAILED;
    }
    QLIV2TilingInfo QLIV2Info;
    QLIV2InfoParser QLIV2InfoParser(context);
    if (QLIV2InfoParser.Parse(QLIV2Info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    indexer_v2_checker::QLIV2Checker checker(QLIV2Info.checkerInfo);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QuantLightningIndexerV2Tiling QLIV2Tiling(context);
    return QLIV2Tiling.DoTiling(QLIV2Info);
}

// --------------------------Tiling及函数TilingPrepare函数注册--------
IMPL_OP_OPTILING(QuantLightningIndexerV2)
    .Tiling(TilingForQuantLightningIndexerV2)
    .TilingParse<QLIV2CompileInfo>(TilingPrepareForQuantLightningIndexerV2);

} // namespace optiling
