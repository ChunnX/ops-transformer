/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_grad_tiling.cpp
 * \brief
 */
#include "../op_kernel/lightning_indexer_grad_tiling.h"
#include "lightning_indexer_grad_tiling_data.h"
#include "../op_kernel/lightning_indexer_grad_template_tiling_key.h"

#define CHECK_NULLPTR(ptr)       \
    if (ptr == nullptr) {        \
        return ge::GRAPH_FAILED; \
    }

#define ADD_TILING_DATA(context, tiling)                                                                     \
    CHECK_NULLPTR(context->GetRawTilingData())                                                               \
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity()); \
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {

constexpr size_t GM_ALIGN = 512;
constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
constexpr uint32_t LAYOUT_BSND = 0;
constexpr uint32_t LAYOUT_TND = 1;
constexpr uint64_t MAX_GROUPNUM = 64;
constexpr uint64_t MAX_HEADIM = 128;
constexpr uint64_t MAX_TOPK = 2048;
constexpr uint64_t LIMIT_HEADNUMK = 1;
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint32_t SOC_VER_950_CODE = 4;

ge::graphStatus LightningIndexerGradTiling::DoTiling()
{
    opName_ = context_->GetNodeName();
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    uint64_t workspaceOffset = ascendcPlatform.GetLibApiWorkSpaceSize();;
    size_t *workSpaces = context_->GetWorkspaceSizes(1);

    // input tensor
    opParamInfo.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo.query.shape = context_->GetInputShape(QUERY_INDEX);
    queryDataType = opParamInfo.query.desc->GetDataType();

    opParamInfo.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo.key.shape = context_->GetInputShape(KEY_INDEX);

    opParamInfo.dy.desc = context_->GetInputDesc(DY_INDEX);
    opParamInfo.dy.shape = context_->GetInputShape(DY_INDEX);

    opParamInfo.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INDEX);
    opParamInfo.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INDEX);

    opParamInfo.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);

    // input attr
    auto attrs = context_->GetAttrs();
    opParamInfo.headNum = *attrs->GetInt(ATTR_HEADNUM_INDEX);
    opParamInfo.layout = attrs->GetStr(ATTR_LAYOUT_INDEX);
    opParamInfo.sparseMode = *attrs->GetInt(ATTR_SPARSEMODE_INDEX);
    opParamInfo.preTokens = *attrs->GetInt(ATTR_PRETOKENS_INDEX);
    opParamInfo.nextTokens = *attrs->GetInt(ATTR_NEXTTOKENS_INDEX);
    bool deterministic =  *attrs->GetBool(ATTR_DETERMINSTIC_INDEX) || (context_->GetDeterministic() == 1);
    opParamInfo.deterministic = deterministic;

    uint32_t dyShapeDim = opParamInfo.dy.shape->GetStorageShape().GetDimNum();
    uint32_t dataType = static_cast<uint32_t>(queryDataType);
    uint32_t inputLayout = -1;

    if (std::string(opParamInfo.layout) == "BSND") {
        batch = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
        seqlenQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
        headNumQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        headDim = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_FOUR));
        seqlenK = static_cast<uint32_t>(opParamInfo.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
        headNumK = static_cast<uint32_t>(opParamInfo.key.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        groupNum = headNumQ / headNumK;
        topK = static_cast<uint32_t>(opParamInfo.dy.shape->GetStorageShape().GetDim(dyShapeDim - 1));
        dkSize = batch * seqlenK * headNumK * headDim;
        inputLayout = LAYOUT_BSND;
    } else if (std::string(opParamInfo.layout) == "TND") {
        opParamInfo.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
        opParamInfo.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
        opParamInfo.actualSeqLengthsK.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
        opParamInfo.actualSeqLengthsK.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
        OP_CHECK_IF(opParamInfo.actualSeqLengthsQ.tensor == nullptr || opParamInfo.actualSeqLengthsK.tensor == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                "TND layout requires actual_seq_lengths_query and actual_seq_lengths_key to be provided, but got null."),
            return ge::GRAPH_FAILED);

        batch = static_cast<uint32_t>(opParamInfo.actualSeqLengthsQ.tensor->GetShapeSize());
        seqlenQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
        headNumQ = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
        headDim = static_cast<uint32_t>(opParamInfo.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        seqlenK = static_cast<uint32_t>(opParamInfo.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
        headNumK = static_cast<uint32_t>(opParamInfo.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
        groupNum = headNumQ / headNumK;
        topK = static_cast<uint32_t>(opParamInfo.dy.shape->GetStorageShape().GetDim(dyShapeDim - 1));
        dkSize = seqlenK * headNumK * headDim;
        inputLayout = LAYOUT_TND;
    } else {
        OP_LOGE(context_, "only support layout is BSND and TND.\n", opParamInfo.layout);
        return ge::GRAPH_FAILED; 
    }

    uint32_t dkCoreSize = seqlenK * headDim;
    // check headDim, groupNum, headNumK
    OP_CHECK_IF((headDim != MAX_HEADIM) || (groupNum > MAX_GROUPNUM) || (headNumK != LIMIT_HEADNUMK),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), 
            "only support headDim is %lu, groupNum is %lu, headNumK is %lu, but current headDim is %lu, groupNum is %lu, headNumK is %lu",
                MAX_HEADIM, MAX_GROUPNUM, LIMIT_HEADNUMK, headDim, groupNum, headNumK),
            return ge::GRAPH_FAILED);
    
    // check sparseMode
    OP_CHECK_IF((opParamInfo.sparseMode != 0) && (opParamInfo.sparseMode  != 3),
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), 
            "sparseMode only support value is 0 or 3"),
            return ge::GRAPH_FAILED);

    // set tilingData
    tilingData_->set_batch(batch);
    tilingData_->set_seqlenQ(seqlenQ);
    tilingData_->set_headNumQ(headNumQ);
    tilingData_->set_headDim(headDim);
    tilingData_->set_seqlenK(seqlenK);
    tilingData_->set_headNumK(headNumK);
    tilingData_->set_groupNum(groupNum);
    tilingData_->set_topK(topK);
    tilingData_->set_usedCoreNum(blockDim * 2);
    tilingData_->set_dkSize(dkSize);
    tilingData_->set_dkCoreSize(dkCoreSize);
    tilingData_->set_sparseMode(static_cast<uint64_t>(opParamInfo.sparseMode));
    tilingData_->set_deterministic(static_cast<uint64_t>(opParamInfo.deterministic));

    // set workspace 
    tilingData_->set_dkWorkSpaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + dkSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tilingData_->set_dkCoreWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + aicNum * dkCoreSize * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    uint64_t keyGatherWorkspaceSize = MAX_HEADIM * MAX_TOPK * sizeof(uint16_t) * DOUBLE_BUFFER;
    tilingData_->set_keyGatherWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + aicNum * keyGatherWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    uint64_t reluInWorkspaceSize = MAX_GROUPNUM * MAX_TOPK * sizeof(float) * DOUBLE_BUFFER;
    tilingData_->set_reluInWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + aicNum * reluInWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    uint64_t reluGradWorkspaceSize = MAX_GROUPNUM * MAX_TOPK * sizeof(uint16_t) * DOUBLE_BUFFER;
    tilingData_->set_reluGradWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + aicNum * reluGradWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    uint64_t scatterAddWorkspaceSize = MAX_HEADIM * MAX_TOPK * sizeof(float) * DOUBLE_BUFFER;
    tilingData_->set_scatterAddWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + aicNum * scatterAddWorkspaceSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    workspaceOffset += WORKSPACE_RSV_BYTE;
    workSpaces[0] = (workspaceOffset - 0);

    // -------------set tilingkey-----------------
    uint32_t tilingKey = GET_TPL_TILING_KEY(dataType, inputLayout);
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForLightningIndexerGrad(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4LightningIndexerGradArch35(gert::TilingContext* context)
{
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    ascendPlatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto aicNum = ascendPlatformInfo.GetCoreNumAic();
    auto aivNum = ascendPlatformInfo.GetCoreNumAiv();

    ubSize = ubSize - 40 * 1024;
    context->SetDynUBufSize(ubSize);

    if (aicNum == 0 || aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(aicNum);
    LightningIndexerGradArch35TilingData tilingData;
    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    auto layout = std::string(attrsPtr->GetStr(ATTR_LAYOUT_INDEX));
    auto sparseMode = *attrsPtr->GetInt(ATTR_SPARSEMODE_INDEX);

    int64_t batchSize = 0, headNumQ = 0, headDim = 128;
    int64_t querySeqLen = 0;
    int64_t keySeqLen = 0;
    int64_t tileSeqQ = 1;
    int64_t tileTopK = 128;
    int64_t totalQuerySeqLength = 0;
    int64_t totalKeySeqLength = 0;
    int64_t inputLayout = 0;
    const auto sparseIndicesPtr = context->GetInputShape(SPARSE_INDICES_INDEX);
    CHECK_NULLPTR(sparseIndicesPtr);
    auto sparseIndicesShape = sparseIndicesPtr->GetStorageShape();
    int64_t topK = sparseIndicesShape.GetDim(sparseIndicesShape.GetDimNum() - 1);
    
    if (layout == "BSND") {
        inputLayout = 0;
        const auto queryShapePtr = context->GetInputShape(QUERY_INDEX);
        CHECK_NULLPTR(queryShapePtr);
        auto queryShape = queryShapePtr->GetStorageShape();

        batchSize = queryShape.GetDim(0);
        querySeqLen = queryShape.GetDim(1);
        headNumQ = queryShape.GetDim(2);
        headDim = queryShape.GetDim(3);
        totalQuerySeqLength = batchSize * querySeqLen;

        const auto keyShapePtr = context->GetInputShape(KEY_INDEX);
        CHECK_NULLPTR(keyShapePtr);
        auto keyShape = keyShapePtr->GetStorageShape();
        keySeqLen = keyShape.GetDim(1);
        totalKeySeqLength = batchSize * keySeqLen;
    } else if (layout == "TND") {
        inputLayout = 1;
        const auto queryShapePtr = context->GetInputShape(QUERY_INDEX);
        CHECK_NULLPTR(queryShapePtr);
        auto queryShape = queryShapePtr->GetStorageShape();

        const auto keyShapePtr = context->GetInputShape(KEY_INDEX);
        CHECK_NULLPTR(keyShapePtr);
        auto keyShape = keyShapePtr->GetStorageShape();

        const auto actualSeqLengthsQueryPtr = context->GetInputShape(ACTUAL_SEQ_Q_INDEX);
        CHECK_NULLPTR(actualSeqLengthsQueryPtr);
        auto actualSeqLengthsQueryShape = actualSeqLengthsQueryPtr->GetStorageShape();

        batchSize = actualSeqLengthsQueryShape.GetDim(0);
        totalQuerySeqLength = queryShape.GetDim(0);
        headNumQ = queryShape.GetDim(1);
        headDim = queryShape.GetDim(2);
        totalKeySeqLength = keyShape.GetDim(0);
    }  else {
        OP_LOGE(context, "only support layout is BSND and TND.\n", layout);
        return ge::GRAPH_FAILED;
    }

    const auto queryTensorPtr = context->GetInputTensor(QUERY_INDEX);
    CHECK_NULLPTR(queryTensorPtr);
    matmul_tiling::DataType queryDataType;

    if (queryTensorPtr->GetDataType() == ge::DT_FLOAT16) {
        queryDataType = matmul_tiling::DataType::DT_FLOAT16;
    }
    else if (queryTensorPtr->GetDataType() == ge::DT_BF16)
        queryDataType = matmul_tiling::DataType::DT_BF16;
    else {
        OP_LOGE(context, "input only support FP16 and BF16.\n");
        return ge::GRAPH_FAILED;
    }

    tilingData.set_batchSize(batchSize);
    tilingData.set_headNumQ(headNumQ);
    tilingData.set_headDim(headDim);
    tilingData.set_querySeqLen(querySeqLen);
    tilingData.set_keySeqLen(keySeqLen);
    tilingData.set_tileTopK(tileTopK);
    tilingData.set_tileSeqQ(tileSeqQ);
    tilingData.set_totalQuerySeqLength(totalQuerySeqLength);
    tilingData.set_totalKeySeqLength(totalKeySeqLength);
    tilingData.set_inputLayout(inputLayout);
    tilingData.set_topK(topK);
    tilingData.set_aivNum(aivNum);
    tilingData.set_sparseMode(sparseMode);

    ADD_TILING_DATA(context, tilingData);

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = 0;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;

    return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus TilingForLightningIndexerGrad(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("LightningIndexerGrad", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t socVer = static_cast<uint32_t>(ascendcPlatform.GetSocVersion());
    if (socVer == SOC_VER_950_CODE) {
         return Tiling4LightningIndexerGradArch35(context);
     } else {
        LightningIndexerGradTiling LIGTiling(context);
        return LIGTiling.DoTiling();
     }
}

IMPL_OP_OPTILING(LightningIndexerGrad)
    .Tiling(TilingForLightningIndexerGrad)
    .TilingParse<LIGCompileInfo>(TilingPrepareForLightningIndexerGrad);

} // namespace optiling
