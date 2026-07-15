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
 * \file compressor_grad_tiling.cpp
 * \brief
 */

#include <functional>
#include <algorithm>
#include <unordered_map>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "compressor_grad_tiling.h"

using namespace ge;
using namespace AscendC;
namespace optiling {


void CompressorGradTiling::ConvertRequiredParams(gert::TilingContext &context,
                                                 CompressorGradContext &compressorGradContext)
{
    compressorGradContext.x.desc = context.GetRequiredInputDesc(CMPG_TOKEN_X_INPUT_INDEX);
    compressorGradContext.x.shape = context.GetRequiredInputShape(CMPG_TOKEN_X_INPUT_INDEX);
    compressorGradContext.wkv.desc = context.GetRequiredInputDesc(CMPG_WEIGHT_KV_INPUT_INDEX);
    compressorGradContext.wkv.shape = context.GetRequiredInputShape(CMPG_WEIGHT_KV_INPUT_INDEX);
    compressorGradContext.wgate.desc = context.GetRequiredInputDesc(CMPG_WEIGHT_WGATE_INPUT_INDEX);
    compressorGradContext.wgate.shape = context.GetRequiredInputShape(CMPG_WEIGHT_WGATE_INPUT_INDEX);
    compressorGradContext.dCpmKv.desc = context.GetRequiredInputDesc(CMPG_D_CPM_KV_INPUT_INDEX);
    compressorGradContext.dCpmKv.shape = context.GetRequiredInputShape(CMPG_D_CPM_KV_INPUT_INDEX);
    compressorGradContext.softmaxScore.desc = context.GetRequiredInputDesc(CMPG_SOFTMAX_SCORE_INPUT_INDEX);
    compressorGradContext.softmaxScore.shape = context.GetRequiredInputShape(CMPG_SOFTMAX_SCORE_INPUT_INDEX);
    compressorGradContext.kv.desc = context.GetRequiredInputDesc(CMPG_KV_INPUT_INDEX);
    compressorGradContext.kv.shape = context.GetRequiredInputShape(CMPG_KV_INPUT_INDEX);

    compressorGradContext.dX.desc = context.GetOutputDesc(CMPG_D_X_OUTPUT_INDEX);
    compressorGradContext.dX.shape = context.GetOutputShape(CMPG_D_X_OUTPUT_INDEX);
    compressorGradContext.dWkv.desc = context.GetOutputDesc(CMPG_D_WKV_OUTPUT_INDEX);
    compressorGradContext.dWkv.shape = context.GetOutputShape(CMPG_D_WKV_OUTPUT_INDEX);
    compressorGradContext.dWgate.desc = context.GetOutputDesc(CMPG_D_WGATE_OUTPUT_INDEX);
    compressorGradContext.dWgate.shape = context.GetOutputShape(CMPG_D_WGATE_OUTPUT_INDEX);
    compressorGradContext.dApe.desc = context.GetOutputDesc(CMPG_D_APE_OUTPUT_INDEX);
    compressorGradContext.dApe.shape = context.GetOutputShape(CMPG_D_APE_OUTPUT_INDEX);

    compressorGradContext.dtype = compressorGradContext.x.desc->GetDataType();
    auto xDimNum = compressorGradContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == CMPG_DIM_NUM_3) {
        compressorGradContext.layout = CmpGradLayoutType::LAYOUT_BSH;
    } else if (xDimNum == CMPG_DIM_NUM_2) {
        compressorGradContext.layout = CmpGradLayoutType::LAYOUT_TH;
    }
}

void CompressorGradTiling::ConvertOptionalParams(gert::TilingContext &context,
                                                 CompressorGradContext &compressorGradContext)
{
    compressorGradContext.cuSeqlens.desc = context.GetOptionalInputDesc(CMPG_CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.cuSeqlens.shape = context.GetOptionalInputShape(CMPG_CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.seqused.desc = context.GetOptionalInputDesc(CMPG_SEQ_USED_INPUT_INDEX);
    compressorGradContext.seqused.shape = context.GetOptionalInputShape(CMPG_SEQ_USED_INPUT_INDEX);
    compressorGradContext.startPos.desc = context.GetOptionalInputDesc(CMPG_START_POS_INPUT_INDEX);
    compressorGradContext.startPos.shape = context.GetOptionalInputShape(CMPG_START_POS_INPUT_INDEX);
}

ge::graphStatus CompressorGradTiling::ConvertContext(gert::TilingContext &context,
                                                     CompressorGradContext &compressorGradContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("CompressorGrad", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI("Getting Context");

    compressorGradContext.opName = context.GetNodeName();
    compressorGradContext.opType = context.GetNodeType();
    compressorGradContext.platformInfo = context.GetPlatformInfo();
    ConvertRequiredParams(context, compressorGradContext);
    ConvertOptionalParams(context, compressorGradContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    compressorGradContext.coff = attrs->GetAttrPointer<int>(CMPG_COFF_ATTR_INDEX);
    compressorGradContext.cmpRatio = attrs->GetAttrPointer<int>(CMPG_CMP_RATIO_ATTR_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    compressorGradContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();

    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::SetBaseInfo()
{
    if (context_->x.shape->GetStorageShape().GetDimNum() == CMPG_DIM_NUM_3) {
        baseParams_->batchSize = context_->x.shape->GetStorageShape().GetDim(CMPG_DIM_INDEX_0);
        baseParams_->seqSize = context_->x.shape->GetStorageShape().GetDim(CMPG_DIM_INDEX_1);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(CMPG_DIM_INDEX_2);
        baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seqSize;
    } else {
        baseParams_->batchSize = context_->cuSeqlens.shape->GetStorageShape().GetDim(CMPG_DIM_INDEX_0) - 1;
        baseParams_->tokenSize = context_->x.shape->GetStorageShape().GetDim(CMPG_DIM_INDEX_0);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(CMPG_DIM_INDEX_1);
    }

    coff = static_cast<uint8_t>(*context_->coff);
    baseParams_->headDim = context_->wkv.shape->GetStorageShape().GetDim(CMPG_DIM_INDEX_0) / coff;
    baseParams_->cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_->usedCoreNum = aicNum_;

    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u",
            baseParams_->batchSize, baseParams_->tokenSize, baseParams_->cmpRatio, coff);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::SetInnerSplitInfo()
{
    constexpr uint32_t M_BASE_COFF1 = 256;
    constexpr uint32_t M_BASE_COFF2 = 128;
    constexpr uint32_t D_BASE_SIZE = 128;
    constexpr uint32_t H_BASE_SIZE = 128;

    mBaseSize = (coff == 1) ? M_BASE_COFF1 : M_BASE_COFF2;
    dBaseSize = D_BASE_SIZE;

    innerSplitParams_->mBaseSize = mBaseSize;
    innerSplitParams_->dBaseSize = dBaseSize;
    innerSplitParams_->dGroupNum = baseParams_->headDim / dBaseSize;
    innerSplitParams_->mBlockNum = (baseParams_->tokenSize + mBaseSize - 1) / mBaseSize;
    innerSplitParams_->hBaseNum = (baseParams_->hiddenSize + H_BASE_SIZE - 1) / H_BASE_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::SetWorkSpaceInfo()
{
    workspaceParams_->dxPartialSize = mBaseSize * baseParams_->hiddenSize;
    workspaceParams_->dwkvPartialSize = coff * dBaseSize * baseParams_->hiddenSize;
    workspaceParams_->dwgatePartialSize = coff * dBaseSize * baseParams_->hiddenSize;
    workspaceParams_->dapePartialSize = baseParams_->cmpRatio * coff * dBaseSize;
    workspaceParams_->dbWorkspaceRatio = 2;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CalcWorkSpace()
{
    constexpr uint32_t ELEM_SIZE_FP32 = 4;
    uint32_t dwkvPartialSize = coff * dBaseSize * baseParams_->hiddenSize;
    uint32_t dwgatePartialSize = coff * dBaseSize * baseParams_->hiddenSize;
    uint32_t dxPartialSize = mBaseSize * baseParams_->hiddenSize;
    uint32_t dapePartialSize = baseParams_->cmpRatio * coff * dBaseSize;
    uint32_t groupNum = innerSplitParams_->dGroupNum;

    workspaceSize_ = libapiSize_;
    workspaceSize_ += static_cast<size_t>(dxPartialSize) * ELEM_SIZE_FP32 * groupNum * workspaceParams_->dbWorkspaceRatio;
    workspaceSize_ += static_cast<size_t>(dwkvPartialSize) * ELEM_SIZE_FP32 * groupNum *
                      workspaceParams_->dbWorkspaceRatio;
    workspaceSize_ += static_cast<size_t>(dwgatePartialSize) * ELEM_SIZE_FP32 * groupNum *
                      workspaceParams_->dbWorkspaceRatio;
    workspaceSize_ += static_cast<size_t>(dapePartialSize) * ELEM_SIZE_FP32 * groupNum *
                      workspaceParams_->dbWorkspaceRatio;

    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }

    OP_LOGI(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckEmptyTensor() const
{
    if (context_->layout == CmpGradLayoutType::LAYOUT_BSH &&
            context_->x.shape->GetStorageShape().GetDim(CMPG_DIM_INDEX_0) == 0 ||
        context_->layout == CmpGradLayoutType::LAYOUT_BSH &&
            context_->x.shape->GetStorageShape().GetDim(CMPG_DIM_INDEX_1) == 0 ||
        context_->layout == CmpGradLayoutType::LAYOUT_TH &&
            context_->x.shape->GetStorageShape().GetDim(CMPG_DIM_INDEX_0) == 0) {
        context_->templateId = CmpGradTemplateId::EMPTY_X;
    } else {
        if (context_->x.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wkv.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wgate.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->dCpmKv.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->kv.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->softmaxScore.shape->GetStorageShape().GetShapeSize() == 0) {
            OP_LOGE(context_->opName, "Only input tensor x dim B or S or T supports to be 0");
            return ge::GRAPH_FAILED;
        }
        context_->templateId = CmpGradTemplateId::NORMAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::RunBigKernelTiling(CompressorGradTilingData *tilingData)
{
    this->baseParams_ = &tilingData->baseParams;
    this->innerSplitParams_ = &tilingData->innerSplitParams;
    this->workspaceParams_ = &tilingData->workspaceParams;
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs {
        std::bind(&CompressorGradTiling::GetNpuInfo, this),
        std::bind(&CompressorGradTiling::CheckRequiredParaExistence, this),
        std::bind(&CompressorGradTiling::CheckEmptyTensor, this),
        std::bind(&CompressorGradTiling::CheckSinglePara, this),
        std::bind(&CompressorGradTiling::SetBaseInfo, this),
        std::bind(&CompressorGradTiling::CheckFeature, this),
        std::bind(&CompressorGradTiling::CheckMultiParaConsistency, this),
        std::bind(&CompressorGradTiling::CheckBlockDimConstrain, this),
        std::bind(&CompressorGradTiling::SetInnerSplitInfo, this),
        std::bind(&CompressorGradTiling::SetWorkSpaceInfo, this)
    };
    for (const auto &func: requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->templateId == CmpGradTemplateId::EMPTY_X) {
        workspaceSize_ = libapiSize_;
        if (context_->workSpaces) {
            context_->workSpaces[0] = workspaceSize_;
        }
        GenTilingKey();
        context_->blockDim = 1U;
        return ge::GRAPH_SUCCESS;
    }
    std::vector<StatusFunction> optionalTilingFuncs {
        std::bind(&CompressorGradTiling::CalcWorkSpace, this),
        std::bind(&CompressorGradTiling::GenTilingKey, this)
    };
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    OP_LOGI("Run big kernel");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::GenTilingKey() const
{
    uint8_t dtype = 0;
    uint8_t layout = 0;
    uint8_t templateId = static_cast<uint8_t>(context_->templateId);

    auto xDtype = context_->x.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == CMPG_DIM_NUM_3) {
        layout = 0;
    } else {
        layout = 1;
    }

    context_->tilingKey = GET_TPL_TILING_KEY(
        layout,
        dtype,
        coff,
        templateId
    );
    OP_LOGI(context_->opName,
            "CompressorGrad dtype:%hhu layout:%hhu  coff:%hhu, template_id:%hhu", dtype,
            layout, coff, templateId);
    OP_LOGI(context_->opName, "CompressorGrad tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaX() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWgate() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDCpmKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaScKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSoftmaxOut() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqused() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDX() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWgate() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDApe() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCoff()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CompressorGradTiling::CheckFeatureValueSupport(const T *featureValue,
                                                               const std::vector<T> &expectFeatureValList,
                                                               const std::string &name) const
{
    if (std::find(expectFeatureValList.begin(), expectFeatureValList.end(), *featureValue) ==
        expectFeatureValList.end()) {
        LogErrorNumberSupport(expectFeatureValList, *featureValue, name, "feature value");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CompressorGradTiling::CheckAttrValueSupport(const T *attrValue,
                                                            const std::vector<T> &expectAttrValList,
                                                            const std::string &name) const
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectAttrValList.begin(), expectAttrValList.end(), *attrValue) == expectAttrValList.end()) {
        LogErrorNumberSupport(expectAttrValList, *attrValue, name, "attr value");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
std::string to_string(const T &value)
{
    if (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        return std::to_string(value);
    }
}

template <typename T>
void CompressorGradTiling::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                                 const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE(context_->opName, "%s %s only supports %s, but got %s", name.c_str(), subName.c_str(), oss.str().c_str(),
            to_string(actualValue).c_str());
}

std::string LayoutTypeToStr(CmpGradLayoutType layout)
{
    switch (layout) {
        case CmpGradLayoutType::LAYOUT_BSH:
            return "BSH";
        case CmpGradLayoutType::LAYOUT_TH:
            return "TH";
        default:
            return "UNKNOWN_LAYOUT";
    }
}

ge::graphStatus CompressorGradTiling::CheckDimNumInLayoutSupport(const std::string &layout,
                                                                 const gert::StorageShape *shape,
                                                                 const std::string &name) const
{
    const auto &dimIt = CMPG_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE(context_->opName, "When layout is %s, %s dimension should be %zu, but it's %zu", layout.c_str(),
                        name.c_str(), dimIt->second, shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                        const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = CMPG_DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(
            it == CMPG_DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(context_->opName, "%s datatype support list should be specify in CMPG_DTYPE_SUPPORT_MAP",
                    name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) ==
                        expectDtypeList.end(),
                    LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void CompressorGradTiling::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                                                const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << DataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(context_->opName, "Tensor %s only supports dtype %s, but got %s", name.c_str(), oss.str().c_str(),
            DataTypeToSerialString(actualDtype).c_str());
}

static std::string DataTypeToSerialString(ge::DataType type)
{
    const auto it = CMPG_DATATYPE_TO_STRING_MAP.find(type);
    if (it != CMPG_DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("CompressorGrad", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

ge::graphStatus CompressorGradTiling::CheckDimNumSupport(const gert::StorageShape *shape,
                                                         const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto &it = CMPG_DIM_NUM_MAP.find(name);
    OP_CHECK_IF(it == CMPG_DIM_NUM_MAP.end(),
                OP_LOGE(context_->opName, "%s dim number support list should be specify in CMPG_DIM_NUM_MAP",
                        name.c_str()),
                return ge::GRAPH_FAILED);
    auto &expectDimNumList = it->second;
    OP_CHECK_IF(std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
                    expectDimNumList.end(),
                LogErrorNumberSupport(expectDimNumList, static_cast<uint32_t>(shape->GetStorageShape().GetDimNum()),
                                      name, "dimension"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaX() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->x.desc, CMPG_X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->x.shape, CMPG_X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout), context_->x.shape,
                                                        CMPG_X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaWkv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wkv.desc, CMPG_WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wkv.shape, CMPG_WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaWgate() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wgate.desc, CMPG_WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wgate.shape, CMPG_WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaDCpmKv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dCpmKv.desc, CMPG_D_CPM_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dCpmKv.shape, CMPG_D_CPM_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaScKv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->kv.desc, CMPG_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->kv.shape, CMPG_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaSoftmaxOut() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->softmaxScore.desc, CMPG_SOFTMAX_SCORE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->softmaxScore.shape, CMPG_SOFTMAX_SCORE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaStartPos() const
{
    if (context_->startPos.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->startPos.desc, CMPG_START_POS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->startPos.shape, CMPG_START_POS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaCuSeqlens() const
{
    if (context_->cuSeqlens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cuSeqlens.desc, CMPG_CU_SEQLENS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cuSeqlens.shape, CMPG_CU_SEQLENS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaSeqused() const
{
    if (context_->seqused.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->seqused.desc, CMPG_SEQUSED_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->seqused.shape, CMPG_SEQUSED_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaDX() const
{
    if (context_->dX.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dX.desc, CMPG_D_X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dX.shape, CMPG_D_X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaDWkv() const
{
    if (context_->dWkv.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dWkv.desc, CMPG_D_WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dWkv.shape, CMPG_D_WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaDWgate() const
{
    if (context_->dWgate.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dWgate.desc, CMPG_D_WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dWgate.shape, CMPG_D_WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaDApe() const
{
    if (context_->dApe.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dApe.desc, CMPG_D_APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dApe.shape, CMPG_D_APE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaCmpRatio() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->cmpRatio, CMPG_CMP_RATIO, CMPG_CMP_RATIO_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaCoff() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->coff, CMPG_COFF, CMPG_COFF_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(context_->x.shape == nullptr, OP_LOGE(context_->opName, "tensor x is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->x.desc == nullptr, OP_LOGE(context_->opName, "tensor x is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wkv.shape == nullptr, OP_LOGE(context_->opName, "tensor wkv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wkv.desc == nullptr, OP_LOGE(context_->opName, "tensor wkv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wgate.shape == nullptr, OP_LOGE(context_->opName, "tensor wgate is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wgate.desc == nullptr, OP_LOGE(context_->opName, "tensor wgate is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dCpmKv.shape == nullptr, OP_LOGE(context_->opName, "tensor d_cpm_kv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dCpmKv.desc == nullptr, OP_LOGE(context_->opName, "tensor d_cpm_kv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kv.shape == nullptr, OP_LOGE(context_->opName, "tensor kv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kv.desc == nullptr, OP_LOGE(context_->opName, "tensor kv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->softmaxScore.shape == nullptr, OP_LOGE(context_->opName, "tensor softmax_score is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->softmaxScore.desc == nullptr, OP_LOGE(context_->opName, "tensor softmax_score is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dX.shape == nullptr, OP_LOGE(context_->opName, "tensor d_x is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dX.desc == nullptr, OP_LOGE(context_->opName, "tensor d_x is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dWkv.shape == nullptr, OP_LOGE(context_->opName, "tensor d_wkv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dWkv.desc == nullptr, OP_LOGE(context_->opName, "tensor d_wkv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dWgate.shape == nullptr, OP_LOGE(context_->opName, "tensor d_wgate is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dWgate.desc == nullptr, OP_LOGE(context_->opName, "tensor d_wgate is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dApe.shape == nullptr, OP_LOGE(context_->opName, "tensor d_ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dApe.desc == nullptr, OP_LOGE(context_->opName, "tensor d_ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->startPos.shape == nullptr,
                OP_LOGE(context_->opName, "tensor startPos is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->startPos.desc == nullptr,
                OP_LOGE(context_->opName, "tensor startPos is nullptr"), return ge::GRAPH_FAILED);
    if (context_->layout == CmpGradLayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context_->cuSeqlens.desc == nullptr,
                    OP_LOGE(context_->opName, "In TH layout, tensor cuSeqlens should not be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape == nullptr,
                    OP_LOGE(context_->opName, "In TH layout, tensor cuSeqlens should not be nullptr"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context_->cuSeqlens.desc != nullptr,
                    OP_LOGE(context_->opName, "In BSH layout, tensor cuSeqlens must be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape != nullptr,
                    OP_LOGE(context_->opName, "In BSH layout, tensor cuSeqlens must be nullptr"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->cmpRatio == nullptr, OP_LOGE(context_->opName, "attr cmpRatio is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&baseParams_->headDim, CMPG_HEAD_DIM, "headDim")) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseParams_->hiddenSize > CMPG_MAX_HIDDEN_SIZE ||
                baseParams_->hiddenSize < CMPG_MIN_HIDDEN_SIZE ||
                baseParams_->hiddenSize % CMPG_ALIGN_FACTOR_HIDDEN_SIZE != 0,
                OP_LOGE(context_->opName, "hiddenSize should be whthin [1k, 10k] and be 512-aligned, but got %u",
                        baseParams_->hiddenSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::LogErrorShapeConsistency(const std::string &name,
                                                               const gert::StorageShape *shape,
                                                               const uint32_t &dimNum, const std::string &subName,
                                                               const uint32_t &expectNum) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE(context_->opName, "%s shape dim %u, should be equal to %s: %u, but got %u", name.c_str(),
                        dimNum, subName.c_str(), expectNum, actualNum),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckShapeConsistency() const
{
    auto coffD = coff * baseParams_->headDim;
    uint32_t C = coffD;
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wkv", context_->wkv.shape,
                                                      CMPG_DIM_INDEX_1, "hiddenSize", baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wgate", context_->wgate.shape,
                                                      CMPG_DIM_INDEX_1, "hiddenSize", baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wkv", context_->wkv.shape,
                                                      CMPG_DIM_INDEX_0, "coff*headDim", static_cast<uint32_t>(C)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wgate", context_->wgate.shape,
                                                      CMPG_DIM_INDEX_0, "coff*headDim", static_cast<uint32_t>(C)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("d_wkv", context_->dWkv.shape,
                                                      CMPG_DIM_INDEX_0, "coff*headDim", static_cast<uint32_t>(C)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("d_wgate", context_->dWgate.shape,
                                                      CMPG_DIM_INDEX_0, "coff*headDim", static_cast<uint32_t>(C)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("d_ape", context_->dApe.shape,
                                                      CMPG_DIM_INDEX_0, "cmpRatio", baseParams_->cmpRatio) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("d_ape", context_->dApe.shape,
                                                      CMPG_DIM_INDEX_1, "coff*headDim", static_cast<uint32_t>(C)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("d_cpm_kv", context_->dCpmKv.shape,
                                                      CMPG_DIM_INDEX_1, "headDim", baseParams_->headDim) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("d_cpm_kv", context_->dCpmKv.shape,
                                                      CMPG_DIM_INDEX_1, "headDim", baseParams_->headDim) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("d_x", context_->dX.shape,
                                                      CMPG_DIM_INDEX_1, "hiddenSize", baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("startPos", context_->startPos.shape,
                                                      CMPG_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("seqused", context_->seqused.shape,
                                                      CMPG_DIM_INDEX_0, "batchSize", baseParams_->batchSize)) {
        return ge::GRAPH_FAILED;
    }
    if (context_->layout == CmpGradLayoutType::LAYOUT_TH &&
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("cuSeqlens", context_->cuSeqlens.shape, CMPG_DIM_INDEX_0,
                                                      "batchSize+1", baseParams_->batchSize + 1)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc *desc,
                                                             const std::string &name) const
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(actualDtype != context_->dtype,
                OP_LOGE(context_->opName, "%s datatype should be same with x: %s, but got %s", name.c_str(),
                        DataTypeToSerialString(actualDtype).c_str(), DataTypeToSerialString(context_->dtype).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckDtypeConsistency() const
{
    if (CheckDtypeConsistencyX(context_->wkv.desc, CMPG_WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->wgate.desc, CMPG_WGATE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dCpmKv.desc, CMPG_D_CPM_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dX.desc, CMPG_D_X_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dWkv.desc, CMPG_D_WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dWgate.desc, CMPG_D_WGATE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckDimNumConsistency() const
{
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != context_->dX.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(context_->opName, "d_x dim num should be equal to x: %u, but got %u", xDimNum,
                        context_->dX.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckBlockDimConstrain() const
{
    uint32_t minBlockNum = baseParams_->headDim / 64;
    OP_CHECK_IF(aicNum_ < minBlockNum,
                OP_LOGE(context_->opName, "aicNum is %d, which should not be less than %d", aicNum_, minBlockNum),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckMultiParaConsistency() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS || CheckDtypeConsistency() != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

CMPG_EXTERN_C ge::graphStatus TilingCompressorGrad(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("CompressorGrad", "Context is nullptr."),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    CompressorGradContext compressorGradContext{};
    if (CompressorGradTiling::ConvertContext(*context, compressorGradContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to CompressorGrad context");
        return ge::GRAPH_FAILED;
    }
    CompressorGradTiling compressorGradTiling(&compressorGradContext);
    CompressorGradTilingData *tilingData = context->GetTilingData<CompressorGradTilingData>();
    OP_CHECK_IF(tilingData == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(compressorGradContext.opName, "TilingData is nullptr."),
                return ge::GRAPH_FAILED);
    context->SetScheduleMode(CMPG_BATCH_MODE_SCHEDULE);
    if (compressorGradTiling.RunBigKernelTiling(tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(compressorGradContext.tilingKey);
    context->SetBlockDim(compressorGradContext.blockDim);
    OP_LOGI(compressorGradContext.opName, "block dim: %u.", compressorGradContext.blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForCompressorGrad(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CompressorGrad)
    .Tiling(TilingCompressorGrad)
    .TilingParse<CompressorGradCompileInfo>(TilingPrepareForCompressorGrad);
} // namespace optiling
