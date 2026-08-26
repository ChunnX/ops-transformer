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
#include "checkers/liv2_checker.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
namespace {
constexpr uint32_t LI_WEIGHTS_INDEX = 2;
constexpr uint32_t LI_CU_SEQLENS_K_INDEX = 4;
constexpr uint32_t LI_SEQUSED_Q_INDEX = 5;
constexpr uint32_t LI_SEQUSED_K_INDEX = 6;
constexpr uint32_t LI_CMP_RESIDUAL_K_INDEX = 7;
constexpr uint32_t LI_OUTPUT_IDX_OFFSET_INDEX = 9;
constexpr uint32_t LI_METADATA_INDEX = 10;
constexpr uint32_t LI_SPARSE_VALUES_INDEX = 1;
constexpr uint32_t LI_ATTR_MAX_SEQLEN_Q_INDEX = 1;

indexer_v2_checker::CheckerTensor RequiredInput(gert::TilingContext *context, uint32_t index)
{
    return {context->GetInputDesc(index), context->GetInputShape(index), nullptr,
            context->GetDynamicInputStride(index, 0)};
}

indexer_v2_checker::CheckerTensor OptionalInput(gert::TilingContext *context, uint32_t index)
{
    return {context->GetOptionalInputDesc(index), nullptr, context->GetOptionalInputTensor(index),
            context->GetDynamicInputStride(index, 0)};
}

indexer_v2_checker::CheckerTensor Output(gert::TilingContext *context, uint32_t index)
{
    return {context->GetOutputDesc(index), context->GetOutputShape(index), nullptr, nullptr};
}
} // namespace

ge::graphStatus LIV2InfoParser::Parse(LIV2TilingInfo &liInfo) const
{
    const char *opName = context_->GetNodeName() == nullptr ? "LightningIndexerV2" : context_->GetNodeName();
    liInfo.platformInfo = context_->GetPlatformInfo();
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
    if (context_->GetWorkspaceSizes(1) == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "workspace_sizes",
                                                 "Workspace sizes must be provided by the tiling context");
        return ge::GRAPH_FAILED;
    }
    if (context_->GetRawTilingData() == nullptr) {
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

    auto &checkerInfo = liInfo.checkerInfo;
    checkerInfo.opName = opName;
    checkerInfo.arch35 = npuArch == NpuArch::DAV_3510;
    checkerInfo.query = RequiredInput(context_, QUERY_INDEX);
    checkerInfo.key = RequiredInput(context_, KEY_INDEX);
    checkerInfo.weights = RequiredInput(context_, LI_WEIGHTS_INDEX);
    checkerInfo.cuSeqlensQ = OptionalInput(context_, CU_SEQLENS_Q_INDEX);
    checkerInfo.cuSeqlensK = OptionalInput(context_, LI_CU_SEQLENS_K_INDEX);
    checkerInfo.sequsedQ = OptionalInput(context_, LI_SEQUSED_Q_INDEX);
    checkerInfo.sequsedK = OptionalInput(context_, LI_SEQUSED_K_INDEX);
    checkerInfo.cmpResidualK = OptionalInput(context_, LI_CMP_RESIDUAL_K_INDEX);
    checkerInfo.blockTable = OptionalInput(context_, BLOCK_TABLE_INDEX);
    checkerInfo.outputIdxOffset = OptionalInput(context_, LI_OUTPUT_IDX_OFFSET_INDEX);
    checkerInfo.metadata = OptionalInput(context_, LI_METADATA_INDEX);
    checkerInfo.sparseIndices = Output(context_, LIGHTNING_INDEXER);
    checkerInfo.sparseValues = Output(context_, LI_SPARSE_VALUES_INDEX);

    const auto *attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "attrs", "Operator attributes must be provided");
        return ge::GRAPH_FAILED;
    }
    const int32_t *topk = attrs->GetAttrPointer<int32_t>(ATTR_TOPK_INDEX);
    const int32_t *maxSeqlenQ = attrs->GetAttrPointer<int32_t>(LI_ATTR_MAX_SEQLEN_Q_INDEX);
    const char *layoutQAttr = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    const char *layoutKAttr = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    const int32_t *maskMode = attrs->GetAttrPointer<int32_t>(ATTR_MASK_MODE_INDEX);
    const int64_t *cmpRatio = attrs->GetAttrPointer<int64_t>(ATTR_CMP_RATIO_INDEX);
    const int32_t *returnValue = attrs->GetAttrPointer<int32_t>(ATTR_RETURN_VALUE_INDEX);
    if (topk == nullptr || maxSeqlenQ == nullptr || layoutQAttr == nullptr || layoutKAttr == nullptr ||
        maskMode == nullptr || cmpRatio == nullptr || returnValue == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "attrs", "Required/defaulted attributes are missing");
        return ge::GRAPH_FAILED;
    }
    checkerInfo.topk = *topk;
    checkerInfo.maxSeqlenQ = *maxSeqlenQ;
    checkerInfo.layoutQText = layoutQAttr;
    checkerInfo.layoutKText = layoutKAttr;
    checkerInfo.maskMode = *maskMode;
    checkerInfo.cmpRatio = *cmpRatio;
    checkerInfo.returnValue = *returnValue;
    indexer_v2_checker::FinalizeCommonInfo(checkerInfo);

    if (checkerInfo.layoutQ == indexer_v2_checker::CheckerLayout::BSND) {
        liInfo.inputQLayout = DataLayout::BSND;
    } else if (checkerInfo.layoutQ == indexer_v2_checker::CheckerLayout::TND) {
        liInfo.inputQLayout = DataLayout::TND;
    }
    if (checkerInfo.layoutK == indexer_v2_checker::CheckerLayout::BSND) {
        liInfo.inputKLayout = DataLayout::BSND;
    } else if (checkerInfo.layoutK == indexer_v2_checker::CheckerLayout::TND) {
        liInfo.inputKLayout = DataLayout::TND;
    } else if (checkerInfo.layoutK == indexer_v2_checker::CheckerLayout::PA_BBND) {
        liInfo.inputKLayout = DataLayout::PA_BBND;
    }

    liInfo.bSize = static_cast<uint32_t>(checkerInfo.batch);
    liInfo.s1Size = checkerInfo.layoutQ == indexer_v2_checker::CheckerLayout::BSND ?
                        static_cast<uint32_t>(checkerInfo.qSeq) : 0U;
    liInfo.s2Size = checkerInfo.kSeq;
    liInfo.gSize = checkerInfo.kHeads == 0 ? 0U : static_cast<uint32_t>(checkerInfo.qHeads / checkerInfo.kHeads);
    liInfo.pageAttentionFlag = checkerInfo.layoutK == indexer_v2_checker::CheckerLayout::PA_BBND;
    liInfo.blockSize = static_cast<int32_t>(checkerInfo.blockSize);
    const gert::Shape *blockShape = checkerInfo.blockTable.GetShape();
    if (liInfo.pageAttentionFlag && blockShape != nullptr && blockShape->GetDimNum() >= 2) {
        liInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape->GetDim(DIM_IDX_ONE));
    }
    if (checkerInfo.query.desc != nullptr) {
        liInfo.inputQType = checkerInfo.query.desc->GetDataType();
    }
    if (checkerInfo.key.desc != nullptr) {
        liInfo.inputKType = checkerInfo.key.desc->GetDataType();
    }
    if (checkerInfo.sparseIndices.desc != nullptr) {
        liInfo.outputType = checkerInfo.sparseIndices.desc->GetDataType();
    }
    liInfo.maskMode = static_cast<int32_t>(checkerInfo.maskMode);
    liInfo.topk = static_cast<uint32_t>(checkerInfo.topk);
    liInfo.cmpRatio = checkerInfo.cmpRatio;
    liInfo.returnValue = static_cast<uint32_t>(checkerInfo.returnValue);
    if (checkerInfo.key.stride != nullptr && checkerInfo.key.stride->GetDimNum() > 0) {
        liInfo.keyStride0 = static_cast<uint32_t>(checkerInfo.key.stride->GetStride(0));
    }
    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForLightningIndexerV2(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------LightningIndexerV2Tiling类成员函数定义-----------------------
ge::graphStatus LightningIndexerV2Tiling::DoTiling(LIV2TilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;             // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 512;             // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 512;            // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;          // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;          // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8; // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;      // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;        // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 8;              // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 8192;          // TopK选取个数
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        constexpr uint32_t li3510S1Base = 4;
        constexpr uint32_t li3510S2Base = 128;
        workspaceSize += li3510S1Base * ((tilingInfo->s2Size + li3510S2Base - 1) / li3510S2Base) * li3510S2Base *
                         sizeof(uint32_t) * aicNum;
        workspaceSize +=
            V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
        workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    } else {
        // 主流程需Workspace大小
        uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
        workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
        // Decode流程(LD)需要Workspace大小
        // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
        workspaceSize +=
            V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
        // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
        workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    }

    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_topk(tilingInfo->topk);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_maskMode(tilingInfo->maskMode);
    tilingData_.set_preTokens(tilingInfo->preTokens);
    tilingData_.set_nextTokens(tilingInfo->nextTokens);
    tilingData_.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.set_keyStride0(tilingInfo->keyStride0);
    tilingData_.set_returnValue(tilingInfo->returnValue);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.set_batchSupperFlag(tilingInfo->batchSupperFlag);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t weightTypeFlag = 0;
    uint64_t tilingKey = GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout,
                                            inputKLayout, weightTypeFlag);
    context_->SetTilingKey(tilingKey);

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
    indexer_v2_checker::LIV2Checker checker;
    if (checker.Init(liV2Info.checkerInfo) != ge::GRAPH_SUCCESS ||
        checker.Process(liV2Info.checkerInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    LightningIndexerV2Tiling liTiling(context);
    return liTiling.DoTiling(&liV2Info);
}
// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(LightningIndexerV2)
    .Tiling(TilingForLightningIndexerV2)
    .TilingParse<LIV2CompileInfo>(TilingPrepareForLightningIndexerV2);
} // namespace optiling
