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


void CompressorGradTiling::ConvertRequiredParams(gert::TilingContext &context, CompressorGradContext &compressorGradContext)
{
    compressorGradContext.x.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    compressorGradContext.x.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    compressorGradContext.wkv.desc = context.GetRequiredInputDesc(WEIGHT_KV_INPUT_INDEX);
    compressorGradContext.wkv.shape = context.GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    compressorGradContext.wgate.desc = context.GetRequiredInputDesc(WEIGHT_WGATE_INPUT_INDEX);
    compressorGradContext.wgate.shape = context.GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);
    compressorGradContext.dCmpKv.desc = context.GetRequiredInputDesc(D_CMP_KV_INPUT_INDEX);
    compressorGradContext.dCmpKv.shape = context.GetRequiredInputShape(D_CMP_KV_INPUT_INDEX);
    compressorGradContext.softmaxScore.desc = context.GetRequiredInputDesc(SOFTMAX_SCORE_INPUT_INDEX);
    compressorGradContext.softmaxScore.shape = context.GetRequiredInputShape(SOFTMAX_SCORE_INPUT_INDEX);
    compressorGradContext.kv.desc = context.GetRequiredInputDesc(KV_INPUT_INDEX);
    compressorGradContext.kv.shape = context.GetRequiredInputShape(KV_INPUT_INDEX);

    compressorGradContext.dtype = compressorGradContext.x.desc->GetDataType();
    auto xDimNum = compressorGradContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_3) {
        compressorGradContext.layout = LayoutType::LAYOUT_BSH;
    } else if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_2) {
        compressorGradContext.layout = LayoutType::LAYOUT_TH;
    }
}

void CompressorGradTiling::ConvertOptionalParams(gert::TilingContext &context, CompressorGradContext &compressorGradContext)
{
    compressorGradContext.cuSeqlens.desc = context.GetOptionalInputDesc(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.cuSeqlens.shape = context.GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    compressorGradContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    compressorGradContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    compressorGradContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus CompressorGradTiling::ConvertContext(gert::TilingContext &context, CompressorGradContext &compressorGradContext)
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
    compressorGradContext.coff = attrs->GetAttrPointer<int>(COFF_ATTR_INDEX);
    compressorGradContext.cmpRatio = attrs->GetAttrPointer<int>(CMP_RATIO_ATTR_INDEX);

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
    if (context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3) {
        baseParams_->batchSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
        baseParams_->seqSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_2);
        baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seqSize;
    } else {
        baseParams_->batchSize = context_->cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) - 1;
        baseParams_->tokenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    }

    coff = static_cast<uint8_t>(*context_->coff);
    baseParams_->headDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) / coff;
    baseParams_->cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_->nSize = 2; // 2:每个核处理两个基本块后做全核同步
    baseParams_->usedCoreNum = aicNum_;
    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u",
            baseParams_->batchSize, baseParams_->tokenSize, baseParams_->cmpRatio, coff);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::SetWorkSpaceInfo()
{
    workspaceParams_->dbWorkspaceRatio = 2;
    workspaceParams_->mm1KvResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * coff;
    workspaceParams_->mm1ScoreResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * coff;
    if (coff == 2) {
        workspaceParams_->vec1TailCacheSize = baseParams_->cmpRatio * baseParams_->headDim;
    }
    workspaceParams_->vec1ResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * baseParams_->nSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::SetScenarioInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::SetTemplateId()
{
    if (context_->templateId == TemplateId::EMPTY_X) {
        return ge::GRAPH_SUCCESS;
    }
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND950) {
        // 设置高性能模板
        if (context_->layout == LayoutType::LAYOUT_BSH && baseParams_->seqSize <= 4 && baseParams_->tokenSize <= 256) {
            context_->templateId = TemplateId::FULL_LOAD;
        }
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::SetInnerSplitInfo()
{
    if (context_->templateId == TemplateId::FULL_LOAD) {
        innerSplitParams_->mBaseSize = 256;              // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 256 / (coff * 2); // nBase = dBase * coff * 2
        uint32_t dBaseNum = baseParams_->headDim / innerSplitParams_->dBaseSize;
        uint32_t mBaseNum = (baseParams_->tokenSize + innerSplitParams_->mBaseSize - 1) / innerSplitParams_->mBaseSize;
        baseParams_->coreGroupNum = baseParams_->usedCoreNum / dBaseNum;
        baseParams_->kBaseNum = 1;
        baseParams_->kBaseSize = baseParams_->hiddenSize;
        if ((dBaseNum * mBaseNum) < baseParams_->usedCoreNum) {
            baseParams_->kBaseNum = baseParams_->usedCoreNum / dBaseNum;
            uint32_t kAlignSize = (baseParams_->hiddenSize + baseParams_->kBaseNum - 1) / baseParams_->kBaseNum;
            baseParams_->kBaseSize = kAlignSize / 16 * 16; // 切k的size需要16对齐
        }
        for (uint32_t i = 0; i < baseParams_->usedCoreNum; i++) {
            baseParams_->splitCoreParam[i].nStart = (i % dBaseNum) * innerSplitParams_->dBaseSize;
            baseParams_->splitCoreParam[i].nEnd = baseParams_->splitCoreParam[i].nStart + innerSplitParams_->dBaseSize;
            if (baseParams_->kBaseNum > 1) {
                uint32_t kStartIdx = i / dBaseNum;
                if (kStartIdx + 1 < baseParams_->coreGroupNum) {
                    uint32_t dealKSize = baseParams_->kBaseSize;
                    baseParams_->splitCoreParam[i].kStart = kStartIdx * baseParams_->kBaseSize;
                    baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + dealKSize;
                } else {
                    uint32_t dealKSize = kStartIdx < baseParams_->coreGroupNum ?
                                        baseParams_->hiddenSize - kStartIdx * baseParams_->kBaseSize : 0;
                    baseParams_->splitCoreParam[i].kStart = kStartIdx * baseParams_->kBaseSize;
                    baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + dealKSize;
                }
                baseParams_->splitCoreParam[i].mStart = 0;
                baseParams_->splitCoreParam[i].mEnd = baseParams_->tokenSize;
                baseParams_->mLoopNum = 1;
            } else {
                baseParams_->splitCoreParam[i].kStart = 0;
                baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + baseParams_->kBaseSize;
                baseParams_->splitCoreParam[i].mStart = (i / dBaseNum) * innerSplitParams_->mBaseSize;
                baseParams_->splitCoreParam[i].mEnd =
                    baseParams_->splitCoreParam[i].mStart + innerSplitParams_->mBaseSize;
                baseParams_->mLoopNum = mBaseNum / baseParams_->coreGroupNum;
            }
        }
    } else {
        innerSplitParams_->mBaseSize = 256;        // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 128 / coff; // 128：核间切分，D轴基本块大小
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CalcWorkSpace()
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4; // 4: fp32
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;  // 4: fp32
    uint32_t maxGroupNum = aicNum_ / (baseParams_->headDim / innerSplitParams_->dBaseSize);
    workspaceSize_ = libapiSize_;
    workspaceSize_ +=
        workspaceParams_->mm1KvResSize * maxGroupNum * MM1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio;
    workspaceSize_ +=
        workspaceParams_->mm1ScoreResSize * maxGroupNum * MM1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio;
    workspaceSize_ +=
        workspaceParams_->vec1TailCacheSize * MM1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio * 2; // 2 kv和score
    workspaceSize_ +=
        workspaceParams_->vec1ResSize * maxGroupNum * V1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio;

    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }

    OP_LOGI(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckEmptyTensor() const
{
    if ((context_->layout == LayoutType::LAYOUT_BSH &&
            context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) == 0) ||
        (context_->layout == LayoutType::LAYOUT_BSH &&
            context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1) == 0) ||
        (context_->layout == LayoutType::LAYOUT_TH &&
            context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) == 0)) {
        context_->templateId = TemplateId::EMPTY_X;
    } else {
        if (context_->x.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wkv.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wgate.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->dCmpKv.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->softmaxScore.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->kv.shape->GetStorageShape().GetShapeSize() == 0) {
            OP_LOGE(context_->opName, "Only input tensor x dim B or S or T supports to be 0");
            return ge::GRAPH_FAILED;
        }
        context_->templateId = TemplateId::NORMAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::RunBigKernelTiling(CompressorGradTilingData *tilingData)
{
    this->baseParams_ = &tilingData->baseParams;
    this->pageAttentionParams_ = &tilingData->pageAttentionParams;
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
        std::bind(&CompressorGradTiling::SetTemplateId, this),
        std::bind(&CompressorGradTiling::SetInnerSplitInfo, this),
        std::bind(&CompressorGradTiling::SetWorkSpaceInfo, this),
        std::bind(&CompressorGradTiling::SetScenarioInfo, this)
    };
    for (const auto &func: requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->templateId == TemplateId::EMPTY_X) {
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
    // 0:BF16, 1:FP16
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;

    auto xDtype = context_->x.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_3) {
        layout = 0;
    } else {
        layout = 1;
    }

    context_->tilingKey = GET_TPL_TILING_KEY(
        layout,
        dtype,
        coff
    );
    OP_LOGI(context_->opName,
            "CompressorGrad dtype:%hhu layout:%hhu  coff:%hhu", dtype, layout, coff);
    OP_LOGI(context_->opName, "CompressorGrad tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaX() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWgate() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSoftmaxScore() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqused() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDX() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWgate() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWApe() ||
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
ge::graphStatus CompressorGradTiling::CheckAttrValueSupportInterVal(const T *attrValue, const T minVal, const T maxVal,
                                                        const std::string &name) const
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (*attrValue < minVal ||  *attrValue > maxVal) {
        std::ostringstream oss;
        oss << "[" << minVal << ", " << maxVal << "]";
        OP_LOGE(context_->opName, "%s %s only supports value in range %s, but got %s",
                name.c_str(),
                "attr value".c_str(),
                oss.str().c_str(),
                to_string(*attrValue).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CompressorGradTiling::CheckAttrValueSupportEnum(const T *attrValue, const std::vector<T> &expectAttrValList,
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

std::string LayoutTypeToStr(LayoutType layout)
{
    switch (layout) {
        case LayoutType::LAYOUT_BSH:
            return "BSH";
        case LayoutType::LAYOUT_TH:
            return "TH";
        default:
            return "UNKNOWN_LAYOUT";
    }
}

ge::graphStatus CompressorGradTiling::CheckDimNumInLayoutSupport(const std::string &layout, const gert::StorageShape *shape,
                                                             const std::string &name) const
{
    const auto &dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE(context_->opName, "When layout is %s, %s dimension should be %u, but it's %zu", layout.c_str(),
                        name.c_str(), dimIt->second, shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                    const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(
            it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(context_->opName, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
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
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("CompressorGrad", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

ge::graphStatus CompressorGradTiling::CheckDimNumSupport(const gert::StorageShape *shape, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto &it = DIM_NUM_MAP.find(name);
    OP_CHECK_IF(it == DIM_NUM_MAP.end(),
                OP_LOGE(context_->opName, "%s dim number support list should be specify in DIM_NUM_MAP", name.c_str()),
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
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->x.desc, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->x.shape, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout), context_->x.shape, X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaWkv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wkv.desc, WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wkv.shape, WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaWgate() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wgate.desc, WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wgate.shape, WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaDCmpKv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dCmpKv.desc, D_CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dCmpKv.shape, D_CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaSoftmaxScore() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->softmaxScore.desc, SOFTMAX_SCORE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->softmaxScore.shape, SOFTMAX_SCORE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static const std::string CMP_RATIO_NAME = "cmp_ratio";
static const std::string COFF_NAME = "coff";
static const std::string D_X_NAME = "d_x";
static const std::string D_WKV_NAME = "d_wkv";
static const std::string D_WGATE_NAME = "d_wgate";
static const std::string D_APE_NAME = "d_ape";

ge::graphStatus CompressorGradTiling::CheckSingleParaKV() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->kv.desc, KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->kv.shape, KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaCuSeqlens() const
{
    if (context_->cuSeqlens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cuSeqlens.desc, CU_SEQLENS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cuSeqlens.shape, CU_SEQLENS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaSeqused() const
{
    if (context_->seqUsed.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->seqUsed.desc, SEQUSED_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->seqUsed.shape, SEQUSED_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaStartPos() const
{
    if (context_->startPos.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->startPos.desc, START_POS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->startPos.shape, START_POS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaDX() const
{
    if (context_->dX.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dX.desc, D_X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dX.shape, D_X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaDWkv() const
{
    if (context_->dWkv.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dWkv.desc, D_WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dWkv.shape, D_WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaDWgate() const
{
    if (context_->dWgate.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dWgate.desc, D_WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dWgate.shape, D_WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaDApe() const
{
    if (context_->dApe.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dApe.desc, D_APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dApe.shape, D_APE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaCmpRatio()const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->cmpRatio, CMP_RATIO, CMP_RATIO_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckSingleParaCoff() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->coff, COFF, COFF_NAME)) {
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
    OP_CHECK_IF(context_->dCmpKv.shape == nullptr, OP_LOGE(context_->opName, "tensor d_cmp_kv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->dCmpKv.desc == nullptr, OP_LOGE(context_->opName, "tensor d_cmp_kv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->softmaxScore.shape == nullptr, OP_LOGE(context_->opName, "tensor softmax_score is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->softmaxScore.desc == nullptr, OP_LOGE(context_->opName, "tensor softmax_score is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kv.shape == nullptr,
                OP_LOGE(context_->opName, "tensor kv is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kv.desc == nullptr,
                OP_LOGE(context_->opName, "tensor kv is nullptr"), return ge::GRAPH_FAILED);

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
    
    if (context_->layout == LayoutType::LAYOUT_TH) {
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
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&baseParams_->headDim, HEAD_DIM, "headDim")) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseParams_->hiddenSize > MAX_HIDDEN_SIZE || baseParams_->hiddenSize < MIN_HIDDEN_SIZE ||
                    baseParams_->hiddenSize % ALIGN_FACTOR_HIDDEN_SIZE != 0,
                OP_LOGE(context_->opName, "hiddenSize should be whthin [1k, 10k] and be 512-aligned, but got %u",
                        baseParams_->hiddenSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        pageAttentionParams_->blockSize < MIN_BLOCK_SIZE,
        OP_LOGE(context_->opName, "blockSize should not be less than 1, but got %u", pageAttentionParams_->blockSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::LogErrorShapeConsistency(const std::string &name, const gert::StorageShape *shape,
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
    coff = static_cast<uint8_t>(*context_->coff);
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("cuSeqlens", context_->cuSeqlens.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                                      "batchSize+1", baseParams_->batchSize + 1) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("seqUsed", context_->seqUsed.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("startPos", context_->startPos.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wkv", context_->wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "hiddenSize",
                                                      baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wgate", context_->wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_1,
                                                      "hiddenSize", baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wkv", context_->wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wgate", context_->wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                                      "coff*headDim", static_cast<uint32_t>(coffD))) {
        return ge::GRAPH_FAILED;
    }

    if (context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_2 &&
        context_->dCmpKv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_2 &&
        context_->softmaxScore.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        context_->kv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3) {
        if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("dCmpKv", context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "headDim",
                                                      baseParams_->headDim) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "coff*cmp_ratio",
                                                      coff * baseParams_->cmpRatio) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim",
                                                      baseParams_->headDim) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "coff*cmp_ratio",
                                                      coff * baseParams_->cmpRatio) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim",
                                                      baseParams_->headDim)) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        context_->dCmpKv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        context_->softmaxScore.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_4 &&
        context_->kv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_4) {
        if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("dCmpKv", context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize",
                                                      baseParams_->batchSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("dCmpKv", context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim",
                                                      baseParams_->headDim) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_0 "batchSize",
                                                      baseParams_->batchSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio",
                                                      coff * baseParams_->cmpRatio) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "headDim",
                                                      baseParams_->headDim) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize",
                                                      baseParams_->batchSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio",
                                                      coff * baseParams_->cmpRatio) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "headDim",
                                                      baseParams_->headDim)) {
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
    if (CheckDtypeConsistencyX(context_->wkv.desc, WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->wgate.desc, WGATE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->cmpKv.desc, CMP_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckDimNumConsistency() const
{
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != context_->cmpKv.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(context_->opName, "cmpKv dim num should be equal to x: %lu, but got %zu", xDimNum,
                        context_->cmpKv.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckScenarioConsistency() const
{
    auto curCmpratio = baseParams_->cmpRatio;
    auto curHeaddim = baseParams_->headDim;
    auto curCoff = static_cast<uint8_t>(*context_->coff);
    std::vector<uint32_t> curScenario{curCmpratio, curCoff, curHeaddim};
    const std::vector<std::vector<uint32_t>> allowdScenarios = {{4, 2, 512}, {4, 2, 128}, {128, 1, 512}};

    OP_CHECK_IF(std::find(allowdScenarios.begin(), allowdScenarios.end(), curScenario) == allowdScenarios.end(),
                OP_LOGE(context_->opName,
                        "Cmpratio Coff Headdim should be equal to {4, 2, 512}, {4, 2, 128}, {128, 1, 512}, \
                        but now cmpratio=%u, coff=%u, headdim=%u", curCmpratio, curCoff, curHeaddim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckBlockDimConstrain() const
{
    uint32_t minBlockNum = baseParams_->headDim / 64; // 64 is the largest dBaseSize
    OP_CHECK_IF(aicNum_ < minBlockNum,
                OP_LOGE(context_->opName, "aicNum is %u, which should not be less than %u", aicNum_, minBlockNum),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTiling::CheckMultiParaConsistency() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS || CheckDtypeConsistency() != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
#ifdef DAY0_SCOPE
    if (CheckScenarioConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
#endif
    return ge::GRAPH_SUCCESS;
}

CMP_EXTERN_C ge::graphStatus TilingCompressorGrad(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("CompressorGrad", "Context is nullptr."),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    CompressorGradContext compressorGradContext{};
    if (CompressorGradTiling::ConvertContext(*context, compressorGradContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to CompressorGrad context");
        return ge::GRAPH_FAILED;
    }
    CompressorGradTiling compressorTiling(&compressorGradContext);
    CompressorGradTilingData *tilingData = context->GetTilingData<CompressorGradTilingData>();
    OP_CHECK_IF(tilingData == nullptr, OPS_REPORT_VECTOR_INNER_ERR(compressorGradContext.opName, "TilingData is nullptr."),
                return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    if (compressorTiling.RunBigKernelTiling(tilingData) != ge::GRAPH_SUCCESS) {
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


