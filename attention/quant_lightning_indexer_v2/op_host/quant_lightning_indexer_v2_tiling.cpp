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
constexpr uint32_t QLI_WEIGHTS_INDEX = 2;
constexpr uint32_t QLI_QUERY_DESCALE_INDEX = 3;
constexpr uint32_t QLI_CU_SEQLENS_K_INDEX = 6;
constexpr uint32_t QLI_SEQUSED_Q_INDEX = 7;
constexpr uint32_t QLI_SEQUSED_K_INDEX = 8;
constexpr uint32_t QLI_CMP_RESIDUAL_K_INDEX = 9;
constexpr uint32_t QLI_OUTPUT_IDX_OFFSET_INDEX = 11;
constexpr uint32_t QLI_METADATA_INDEX = 12;
constexpr uint32_t QLI_SPARSE_VALUES_INDEX = 1;
} // namespace

ge::graphStatus QLIV2InfoParser::Parse(QLIV2TilingInfo &qliv2Info) const
{
    if (context_ == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantLightningIndexerV2", "context",
                                                 "Tiling context must not be null");
        return ge::GRAPH_FAILED;
    }
    const char *opName = context_->GetNodeName();
    opName = opName == nullptr ? "QuantLightningIndexerV2" : opName;
    qliv2Info.platformInfo = context_->GetPlatformInfo();
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

    auto &checkerInfo = qliv2Info.checkerInfo;
    checkerInfo.opName = opName;
    checkerInfo.quantized = true;
    checkerInfo.arch35 = npuArch == NpuArch::DAV_3510;
    checkerInfo.query = indexer_v2_checker::MakeRequiredTensor(context_, QUERY_INDEX);
    checkerInfo.key = indexer_v2_checker::MakeRequiredTensor(context_, KEY_INDEX);
    checkerInfo.weights = indexer_v2_checker::MakeRequiredTensor(context_, QLI_WEIGHTS_INDEX);
    checkerInfo.queryDescale = indexer_v2_checker::MakeRequiredTensor(context_, QLI_QUERY_DESCALE_INDEX);
    checkerInfo.keyDescale = indexer_v2_checker::MakeRequiredTensor(context_, KEY_DEQUANT_SCALE_INDEX);
    checkerInfo.cuSeqlensQ = indexer_v2_checker::MakeOptionalTensor(context_, CU_SEQLENS_Q_INDEX);
    checkerInfo.cuSeqlensK = indexer_v2_checker::MakeOptionalTensor(context_, QLI_CU_SEQLENS_K_INDEX);
    checkerInfo.sequsedQ = indexer_v2_checker::MakeOptionalTensor(context_, QLI_SEQUSED_Q_INDEX);
    checkerInfo.sequsedK = indexer_v2_checker::MakeOptionalTensor(context_, QLI_SEQUSED_K_INDEX);
    checkerInfo.cmpResidualK = indexer_v2_checker::MakeOptionalTensor(context_, QLI_CMP_RESIDUAL_K_INDEX);
    checkerInfo.blockTable = indexer_v2_checker::MakeOptionalTensor(context_, BLOCK_TABLE_INDEX);
    checkerInfo.outputIdxOffset = indexer_v2_checker::MakeOptionalTensor(context_, QLI_OUTPUT_IDX_OFFSET_INDEX);
    checkerInfo.metadata = indexer_v2_checker::MakeOptionalTensor(context_, QLI_METADATA_INDEX);
    checkerInfo.sparseIndices = indexer_v2_checker::MakeOutputTensor(context_, SPARSE_INDICES_INDEX);
    checkerInfo.sparseValues = indexer_v2_checker::MakeOutputTensor(context_, QLI_SPARSE_VALUES_INDEX);

    const auto *attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "attrs", "Operator attributes must be provided");
        return ge::GRAPH_FAILED;
    }
    const int64_t *sparseCount = attrs->GetInt(ATTR_TOPK_INDEX);
    const int64_t *quantMode = attrs->GetInt(ATTR_QUANT_MODE_INDEX);
    const int64_t *maxSeqlenQ = attrs->GetInt(ATTR_MAX_SEQLEN_Q_INDEX);
    const char *layoutQAttr = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    const char *layoutKAttr = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    const int64_t *sparseMode = attrs->GetInt(ATTR_MASK_MODE_INDEX);
    const int64_t *cmpRatio = attrs->GetInt(ATTR_CMP_RATIO_INDEX);
    const int64_t *returnValue = attrs->GetInt(ATTR_RETURN_VALUE_INDEX);
    if (sparseCount == nullptr || quantMode == nullptr || maxSeqlenQ == nullptr || layoutQAttr == nullptr ||
        layoutKAttr == nullptr || sparseMode == nullptr || cmpRatio == nullptr || returnValue == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "attrs", "Required/defaulted attributes are missing");
        return ge::GRAPH_FAILED;
    }
    checkerInfo.topk = *sparseCount;
    checkerInfo.quantMode = *quantMode;
    checkerInfo.maxSeqlenQ = *maxSeqlenQ;
    checkerInfo.layoutQText = layoutQAttr;
    checkerInfo.layoutKText = layoutKAttr;
    checkerInfo.maskMode = *sparseMode;
    checkerInfo.cmpRatio = *cmpRatio;
    checkerInfo.returnValue = *returnValue;
    indexer_v2_checker::PopulateDerivedInfo(checkerInfo);

    if (checkerInfo.layoutQ == indexer_v2_checker::CheckerLayout::BSND) {
        qliv2Info.inputQLayout = DataLayout::BSND;
    } else if (checkerInfo.layoutQ == indexer_v2_checker::CheckerLayout::TND) {
        qliv2Info.inputQLayout = DataLayout::TND;
    }
    if (checkerInfo.layoutK == indexer_v2_checker::CheckerLayout::BSND) {
        qliv2Info.inputKLayout = DataLayout::BSND;
    } else if (checkerInfo.layoutK == indexer_v2_checker::CheckerLayout::TND) {
        qliv2Info.inputKLayout = DataLayout::TND;
    } else if (checkerInfo.layoutK == indexer_v2_checker::CheckerLayout::PA_BBND) {
        qliv2Info.inputKLayout = DataLayout::PA_BBND;
    }

    qliv2Info.bSize = static_cast<uint32_t>(checkerInfo.batch);
    qliv2Info.s1Size =
        checkerInfo.layoutQ == indexer_v2_checker::CheckerLayout::BSND ? static_cast<uint32_t>(checkerInfo.qSeq) : 0U;
    qliv2Info.s2Size = checkerInfo.kSeq;
    qliv2Info.qkHeadDim = static_cast<uint32_t>(checkerInfo.headDim);
    qliv2Info.gSize = checkerInfo.kHeads == 0 ? 0U : static_cast<uint32_t>(checkerInfo.qHeads / checkerInfo.kHeads);
    qliv2Info.pageAttentionFlag = checkerInfo.layoutK == indexer_v2_checker::CheckerLayout::PA_BBND;
    qliv2Info.blockSize = static_cast<int32_t>(checkerInfo.blockSize);
    const gert::Shape *blockShape = checkerInfo.blockTable.GetShape();
    if (qliv2Info.pageAttentionFlag && blockShape != nullptr && blockShape->GetDimNum() >= 2) {
        qliv2Info.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape->GetDim(DIM_IDX_ONE));
    }
    if (checkerInfo.query.desc != nullptr) {
        qliv2Info.inputQType = checkerInfo.query.desc->GetDataType();
    }
    if (checkerInfo.key.desc != nullptr) {
        qliv2Info.inputKType = checkerInfo.key.desc->GetDataType();
    }
    if (checkerInfo.sparseIndices.desc != nullptr) {
        qliv2Info.outputType = checkerInfo.sparseIndices.desc->GetDataType();
    }
    qliv2Info.sparseMode = static_cast<int32_t>(checkerInfo.maskMode);
    qliv2Info.sparseCount = static_cast<uint32_t>(checkerInfo.topk);
    qliv2Info.quantMode = static_cast<uint32_t>(checkerInfo.quantMode);
    qliv2Info.cmpRatio = static_cast<uint32_t>(checkerInfo.cmpRatio);
    qliv2Info.returnValue = checkerInfo.returnValue != 0;
    qliv2Info.maxSeqlenQ = static_cast<int32_t>(checkerInfo.maxSeqlenQ);

    if (checkerInfo.key.stride != nullptr && checkerInfo.key.stride->GetDimNum() > 0) {
        qliv2Info.keyStride0 = static_cast<uint32_t>(checkerInfo.key.stride->GetStride(0));
        if (checkerInfo.quantMode == QUANT_MODE_MXFP4) {
            qliv2Info.keyStride0 /= MXFP4_PACK_NUM;
        }
    }
    if (checkerInfo.keyDescale.stride != nullptr && checkerInfo.keyDescale.stride->GetDimNum() > 0) {
        qliv2Info.keyDequantScaleStride0 = static_cast<uint32_t>(checkerInfo.keyDescale.stride->GetStride(0));
    } else if (checkerInfo.quantMode == QUANT_MODE_MXFP8 || checkerInfo.quantMode == QUANT_MODE_MXFP4) {
        qliv2Info.keyDequantScaleStride0 =
            static_cast<uint32_t>(qliv2Info.blockSize) * (qliv2Info.qkHeadDim / MX_SCALE_GROUP_SIZE);
    }
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
    auto *rawTilingData = context_->GetRawTilingData();
    auto *rawTilingBuffer = rawTilingData == nullptr ? nullptr : rawTilingData->GetData();
    if (workSpaces == nullptr || rawTilingBuffer == nullptr) {
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

    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;             // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 256;             // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 2048;           // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;          // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;          // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8; // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;      // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;        // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 4;              // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 2048;          // TopK选取个数
    constexpr uint32_t TOPK_MAX_SIZE_950 = 8192;      // A5 TopK最大选取个数
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 主流程需Workspace大小
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        constexpr uint32_t S1_BASE_SIZE_950 = 4;
        constexpr uint32_t S2_BASE_SIZE_950 = 128;
        workspaceSize += static_cast<uint64_t>(S1_BASE_SIZE_950) *
                         ((tilingInfo.s2Size + S2_BASE_SIZE_950 - 1) / S2_BASE_SIZE_950) * S2_BASE_SIZE_950 *
                         sizeof(uint16_t) * aicNum;
        // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
        workspaceSize += static_cast<uint64_t>(V1_DECODE_DATA_NUM) * S1_BASE_SIZE * V1_RES_ELEM_TYPE *
                         TOPK_MAX_SIZE_950 * V1_RES_ELEM_SIZE * aicNum;
        // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
        workspaceSize += static_cast<uint64_t>(V1_DECODE_DATA_NUM) * S1_BASE_SIZE * V1_DECODE_PARAM_NUM *
                         V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    } else {
        const uint64_t mm1ResSize = static_cast<uint64_t>(M_BASE_SIZE) * S2_BASE_SIZE;
        workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
        // Decode流程(LD)需要Workspace大小
        // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
        workspaceSize += static_cast<uint64_t>(V1_DECODE_DATA_NUM) * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE *
                         V1_RES_ELEM_SIZE * aicNum;
        // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
        workspaceSize += static_cast<uint64_t>(V1_DECODE_DATA_NUM) * S1_BASE_SIZE * V1_DECODE_PARAM_NUM *
                         V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    }
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo.bSize);
    tilingData_.set_s2Size(tilingInfo.s2Size);
    tilingData_.set_s1Size(tilingInfo.s1Size);
    tilingData_.set_sparseCount(tilingInfo.sparseCount);
    tilingData_.set_gSize(tilingInfo.gSize);
    tilingData_.set_blockSize(tilingInfo.blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo.maxBlockNumPerBatch);
    tilingData_.set_sparseMode(tilingInfo.sparseMode);
    tilingData_.set_cmpRatio(tilingInfo.cmpRatio);
    tilingData_.set_returnValue(tilingInfo.returnValue);
    tilingData_.set_maxSeqlenQ(tilingInfo.maxSeqlenQ);
    tilingData_.set_keyStride0(tilingInfo.keyStride0);
    tilingData_.set_keyDequantScaleStride0(tilingInfo.keyDequantScaleStride0);
    tilingData_.set_quantMode(tilingInfo.quantMode);
    tilingData_.set_usedCoreNum(blockDim);
    if (rawTilingData->GetCapacity() < tilingData_.GetDataSize()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantLightningIndexerV2", "raw_tiling_data",
                                                 "Raw tiling data capacity is insufficient");
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(rawTilingBuffer, rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo.inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo.inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo.outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo.pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo.inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo.inputKLayout);
    uint64_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout, inputKLayout);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);

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
