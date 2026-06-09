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
 * \file hc_post_tiling_arch35.cpp
 * \brief
 */

#include "hc_post_tiling.h"

namespace optiling {

ge::graphStatus HcPostTilingRegbase::GetPlatformInfoRegbase()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNum_ <= 0, OP_LOGE(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        ubSize_ <= 0, OP_LOGE(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTilingRegbase::GetShapeInfoRegbase()
{
    OP_CHECK_IF(
        context_ == nullptr, OP_LOGE("HcPostTilingRegBase", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    if (GetInputShapeInfoRegbase() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // dtype校验
    if (GetInputDtypeInfoRegbase() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTilingRegbase::GetInputShapeInfoRegbase()
{
    auto xInput = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_IF(xInput == nullptr, OP_LOGE(context_->GetNodeName(), "get xInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape xShape = xInput->GetStorageShape();
    size_t xDimsN = xShape.GetDimNum();
    OP_CHECK_IF((xDimsN != CONST2 && xDimsN != CONST3),
        OP_LOGE(context_->GetNodeName(), "xInput dim:%lu should be 2 or 3.", xDimsN),
        return ge::GRAPH_FAILED);

    auto residualInput = context_->GetInputShape(INPUT_IDX_RESIDUAL);
    OP_CHECK_IF(residualInput == nullptr, OP_LOGE(context_->GetNodeName(), "get residualInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape residualShape = residualInput->GetStorageShape();
    size_t residualDimsN = residualShape.GetDimNum();
    OP_CHECK_IF((residualDimsN != xDimsN + 1),
        OP_LOGE(context_->GetNodeName(), "residualInput dim:%lu should be %lu.", residualDimsN, xDimsN + 1),
        return ge::GRAPH_FAILED);

    auto postInput = context_->GetInputShape(INPUT_IDX_POST);
    OP_CHECK_IF(postInput == nullptr, OP_LOGE(context_->GetNodeName(), "get residualInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape postShape = postInput->GetStorageShape();
    size_t postDimsN = postShape.GetDimNum();
    OP_CHECK_IF((postDimsN != xDimsN),
        OP_LOGE(context_->GetNodeName(), "postInput dim:%lu should be %lu.", postDimsN, xDimsN),
        return ge::GRAPH_FAILED);

    auto combInput = context_->GetInputShape(INPUT_IDX_COMB);
    OP_CHECK_IF(combInput == nullptr, OP_LOGE(context_->GetNodeName(), "get residualInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape combShape = combInput->GetStorageShape();
    size_t combDimsN = combShape.GetDimNum();
    OP_CHECK_IF((combDimsN != xDimsN + 1),
        OP_LOGE(context_->GetNodeName(), "combInput dim:%lu should be %lu.", combDimsN, xDimsN + 1),
        return ge::GRAPH_FAILED);
    if (xDimsN == CONST2) {
        bsParam_ = xShape.GetDim(DIM_INDEX_0);
        dParam_ = xShape.GetDim(DIM_INDEX_1);
        hcParam_ = residualShape.GetDim(DIM_INDEX_1);
    } else {
        bsParam_ = xShape.GetDim(DIM_INDEX_0) * xShape.GetDim(DIM_INDEX_1);
        dParam_ = xShape.GetDim(DIM_INDEX_2);
        hcParam_ = residualShape.GetDim(DIM_INDEX_2);
    }
    tilingRegbaseData_.set_bsParam(bsParam_);
    tilingRegbaseData_.set_dParam(dParam_);
    tilingRegbaseData_.set_hcParam(hcParam_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTilingRegbase::GetInputDtypeInfoRegbase()
{
    auto xDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get xDesc nullptr."),
        return ge::GRAPH_FAILED);
    auto xDtype = xDesc->GetDataType();
    OP_CHECK_IF(
        (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT),
        OP_LOGE(context_->GetNodeName(), "xDtype is not supported."),
        return ge::GRAPH_FAILED);

    auto residualDesc = context_->GetInputDesc(INPUT_IDX_RESIDUAL);
    OP_CHECK_IF(residualDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get residualDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType residualDtype = residualDesc->GetDataType();
    OP_CHECK_IF(
        (residualDtype != xDtype),
        OP_LOGE(context_->GetNodeName(), "residualDtype is not equal to xDtype."),
        return ge::GRAPH_FAILED);

    auto postDesc = context_->GetInputDesc(INPUT_IDX_POST);
    OP_CHECK_IF(postDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get postDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType postDtype = postDesc->GetDataType();
    OP_CHECK_IF(
        (postDtype != ge::DT_FLOAT16 && postDtype != ge::DT_BF16 && postDtype != ge::DT_FLOAT),
        OP_LOGE(context_->GetNodeName(), "postDtype is not supported."),
        return ge::GRAPH_FAILED);

    auto combDesc = context_->GetInputDesc(INPUT_IDX_COMB);
    OP_CHECK_IF(combDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get combDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType combDtype = combDesc->GetDataType();
    OP_CHECK_IF(
        (combDtype != postDtype),
        OP_LOGE(context_->GetNodeName(), "combDtype is not equal to postDtype."),
        return ge::GRAPH_FAILED);

    if (xDtype == ge::DT_FLOAT) {
        tilingKey_ = 0;
    } else {
        tilingKey_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTilingRegbase::DoOpTilingRegbase()
{
    int64_t batchSize = bsParam_;
    context_->SetTilingKey(tilingKey_);
    int64_t useCoreNum = batchSize < coreNum_ ? batchSize : coreNum_;
    int64_t batchOneCore = CeilDiv(batchSize, static_cast<int64_t>(useCoreNum));
    int64_t batchOneCoreTail = batchOneCore - 1;
    int64_t frontCore = batchSize - batchOneCoreTail * useCoreNum;
    tilingRegbaseData_.set_usedCoreNum(useCoreNum);
    tilingRegbaseData_.set_batchOneCore(batchOneCore);
    tilingRegbaseData_.set_batchOneCoreTail(batchOneCoreTail);
    tilingRegbaseData_.set_frontCore(frontCore);
    context_->SetBlockDim(useCoreNum);

    int64_t dSplitTime = dParam_ / ONCE_DEAL_DPARAM;
    tilingRegbaseData_.set_dSplitTime(dSplitTime);
    tilingRegbaseData_.set_dOnceDealing(ONCE_DEAL_DPARAM);
    tilingRegbaseData_.set_dLastDealing(dParam_ - dSplitTime * ONCE_DEAL_DPARAM);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTilingRegbase::PostTilingRegbase()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = WORKSPACE_SIZE;
    tilingRegbaseData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingRegbaseData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTilingRegbase::RunTilingRegbase()
{
    ge::graphStatus ret = GetShapeInfoRegbase();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetPlatformInfoRegbase();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = DoOpTilingRegbase();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return PostTilingRegbase();
}
}
// namespace optiling