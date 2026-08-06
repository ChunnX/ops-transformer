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
 * \file k_pool_tiling.cpp
 * \file k_pool_tiling.cpp
 * \brief
 */

#include <functional>
#include <algorithm>
#include <unordered_map>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "k_pool_tiling.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

void KPoolTiling::ConvertRequiredParams(gert::TilingContext &context, KPoolContext &k_poolContext)
{
    k_poolContext.hidden_states.desc = context.GetRequiredInputDesc(HIDDEN_STATES_INPUT_INDEX);
    k_poolContext.hidden_states.shape = context.GetRequiredInputShape(HIDDEN_STATES_INPUT_INDEX);
    k_poolContext.wk.desc = context.GetRequiredInputDesc(WK_INPUT_INDEX);
    k_poolContext.wk.shape = context.GetRequiredInputShape(WK_INPUT_INDEX);
    k_poolContext.gate_weight.desc = context.GetRequiredInputDesc(GATE_WEIGHT_INPUT_INDEX);
    k_poolContext.gate_weight.shape = context.GetRequiredInputShape(GATE_WEIGHT_INPUT_INDEX);
    k_poolContext.stateCache.desc = context.GetRequiredInputDesc(STATE_CACHE_INPUT_INDEX);
    k_poolContext.stateCache.shape = context.GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);
    k_poolContext.ape.desc = context.GetRequiredInputDesc(APE_INPUT_INDEX);
    k_poolContext.ape.shape = context.GetRequiredInputShape(APE_INPUT_INDEX);

    k_poolContext.pooledKey.desc = context.GetOutputDesc(POOLED_KEY_OUTPUT_INDEX);
    k_poolContext.pooledKey.shape = context.GetOutputShape(POOLED_KEY_OUTPUT_INDEX);

    k_poolContext.dtype = k_poolContext.hidden_states.desc->GetDataType();
    auto hiddenStatesDimNum = k_poolContext.hidden_states.shape->GetStorageShape().GetDimNum();
    if (hiddenStatesDimNum == K_POOL_DIM_NUM_3) {
        k_poolContext.layout = LayoutType::LAYOUT_BSH;
    } else if (hiddenStatesDimNum == K_POOL_DIM_NUM_2) {
        k_poolContext.layout = LayoutType::LAYOUT_TH;
    }
}

void KPoolTiling::ConvertOptionalParams(gert::TilingContext &context, KPoolContext &k_poolContext)
{
    k_poolContext.cacheBlockTable.desc = context.GetOptionalInputDesc(CACHE_BLOCK_TABLE_INPUT_INDEX);
    k_poolContext.cacheBlockTable.shape = context.GetOptionalInputShape(CACHE_BLOCK_TABLE_INPUT_INDEX);
    k_poolContext.seqLens.desc = context.GetOptionalInputDesc(SEQ_LENS_INPUT_INDEX);
    k_poolContext.seqLens.shape = context.GetOptionalInputShape(SEQ_LENS_INPUT_INDEX);
    k_poolContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    k_poolContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    k_poolContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    k_poolContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus KPoolTiling::ConvertContext(gert::TilingContext &context, KPoolContext &k_poolContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("KPool", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI("Getting Context");

    k_poolContext.opName = context.GetNodeName();
    k_poolContext.opType = context.GetNodeType();
    k_poolContext.platformInfo = context.GetPlatformInfo();
    ConvertRequiredParams(context, k_poolContext);
    ConvertOptionalParams(context, k_poolContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    k_poolContext.coff = attrs->GetAttrPointer<int>(COFF_ATTR_INDEX);
    k_poolContext.cmpRatio = attrs->GetAttrPointer<int>(CMP_RATIO_ATTR_INDEX);
    k_poolContext.cacheMode = attrs->GetAttrPointer<int>(CACHE_MODE_ATTR_INDEX);
    k_poolContext.stateCacheStrideDim0 = attrs->GetAttrPointer<int>(STATE_CACHE_STRIDE_DIM0_ATTR_INDEX);
    k_poolContext.batchConsistency = context.GetDeterministicLevel();
    OP_LOGD(context.GetNodeName(), "deterministic_level=%d", context.GetDeterministicLevel());
    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    k_poolContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::GetNpuInfo()
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

ge::graphStatus KPoolTiling::SetBaseInfo()
{
    if (context_->hidden_states.shape->GetStorageShape().GetDimNum() == K_POOL_DIM_NUM_3) {
        baseParams_->batchSize = context_->hidden_states.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_0);
        baseParams_->seqSize = context_->hidden_states.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_1);
        baseParams_->hiddenSize = context_->hidden_states.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_2);
        baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seqSize;
    } else {
        baseParams_->batchSize = context_->seqLens.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_0) - 1;
        baseParams_->tokenSize = context_->hidden_states.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_0);
        baseParams_->hiddenSize = context_->hidden_states.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_1);
    }

    coff = static_cast<uint8_t>(*context_->coff);
    baseParams_->headDim = context_->wk.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_0) / coff;
    baseParams_->cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_->csSize = baseParams_->seqSize - (baseParams_->seqSize % baseParams_->cmpRatio);
    baseParams_->stateCacheStrideDim0 = static_cast<uint64_t>(*context_->stateCacheStrideDim0);
    baseParams_->nSize = 2; // 2:每个核处理两个基本块后做全核同步
    baseParams_->usedCoreNum = aicNum_;
    baseParams_->batchConsistency = static_cast<uint32_t>(context_->batchConsistency);
    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u, stateCacheStrideDim0:%u",
            baseParams_->batchSize, baseParams_->tokenSize, baseParams_->cmpRatio, coff,
            baseParams_->stateCacheStrideDim0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::SetPageAttentionInfo()
{
    pageAttentionParams_->blockNum = context_->stateCache.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_0);
    pageAttentionParams_->blockSize = context_->stateCache.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_1);
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::LINEAR_BUFFER)) {
        pageAttentionParams_->maxBlockNumPerBatch =
            context_->cacheBlockTable.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::SetWorkSpaceInfo()
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

ge::graphStatus KPoolTiling::SetScenarioInfo() { return ge::GRAPH_SUCCESS; }

ge::graphStatus KPoolTiling::SetTemplateId()
{
    if (context_->templateId == TemplateId::EMPTY_HIDDEN_STATES) {
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

ge::graphStatus KPoolTiling::SetInnerSplitInfo()
{
    if (context_->templateId == TemplateId::FULL_LOAD) {
        innerSplitParams_->mBaseSize = 256;              // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 256 / (coff * 2); // nBase = dBase * coff * 2
        uint32_t dBaseNum = baseParams_->headDim / innerSplitParams_->dBaseSize;
        uint32_t mBaseNum = (baseParams_->tokenSize + innerSplitParams_->mBaseSize - 1) / innerSplitParams_->mBaseSize;
        baseParams_->coreGroupNum = baseParams_->usedCoreNum / dBaseNum;
        baseParams_->kBaseNum = 1;
        baseParams_->kBaseSize = baseParams_->hiddenSize;
        if ((dBaseNum * mBaseNum) < baseParams_->usedCoreNum && baseParams_->batchConsistency != BATCH_CONSISTENCY) {
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
                                             baseParams_->hiddenSize - kStartIdx * baseParams_->kBaseSize :
                                             0;
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

ge::graphStatus KPoolTiling::CalcWorkSpace()
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

ge::graphStatus KPoolTiling::CheckEmptyTensor() const
{
    if ((context_->layout == LayoutType::LAYOUT_BSH &&
         context_->hidden_states.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_0) == 0) ||
        (context_->layout == LayoutType::LAYOUT_BSH &&
         context_->hidden_states.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_1) == 0) ||
        (context_->layout == LayoutType::LAYOUT_TH &&
         context_->hidden_states.shape->GetStorageShape().GetDim(K_POOL_DIM_INDEX_0) == 0)) {
        context_->templateId = TemplateId::EMPTY_HIDDEN_STATES;
    } else {
        if (context_->hidden_states.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wk.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->gate_weight.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->stateCache.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ape.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->cacheBlockTable.shape->GetStorageShape().GetShapeSize() == 0) {
            OP_LOGE(context_->opName, "Only input tensor hidden_states dim B or S or T supports to be 0");
            return ge::GRAPH_FAILED;
        }
        context_->templateId = TemplateId::NORMAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::RunBigKernelTiling(KPoolTilingData *tilingData)
{
    this->baseParams_ = &tilingData->baseParams;
    this->pageAttentionParams_ = &tilingData->pageAttentionParams;
    this->innerSplitParams_ = &tilingData->innerSplitParams;
    this->workspaceParams_ = &tilingData->workspaceParams;
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs{std::bind(&KPoolTiling::GetNpuInfo, this),
                                                    std::bind(&KPoolTiling::CheckRequiredParaExistence, this),
                                                    std::bind(&KPoolTiling::CheckEmptyTensor, this),
                                                    std::bind(&KPoolTiling::CheckSinglePara, this),
                                                    std::bind(&KPoolTiling::SetBaseInfo, this),
                                                    std::bind(&KPoolTiling::SetPageAttentionInfo, this),
                                                    std::bind(&KPoolTiling::CheckFeature, this),
                                                    std::bind(&KPoolTiling::CheckMultiParaConsistency, this),
                                                    std::bind(&KPoolTiling::CheckBlockDimConstrain, this),
                                                    std::bind(&KPoolTiling::SetTemplateId, this),
                                                    std::bind(&KPoolTiling::SetInnerSplitInfo, this),
                                                    std::bind(&KPoolTiling::SetWorkSpaceInfo, this),
                                                    std::bind(&KPoolTiling::SetScenarioInfo, this)};
    for (const auto &func : requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->templateId == TemplateId::EMPTY_HIDDEN_STATES) {
        workspaceSize_ = libapiSize_;
        if (context_->workSpaces) {
            context_->workSpaces[0] = workspaceSize_;
        }
        GenTilingKey();
        context_->blockDim = 1U;
        return ge::GRAPH_SUCCESS;
    }
    std::vector<StatusFunction> optionalTilingFuncs{std::bind(&KPoolTiling::CalcWorkSpace, this),
                                                    std::bind(&KPoolTiling::GenTilingKey, this)};
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    OP_LOGI("Run big kernel");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::GenTilingKey() const
{
    // 0:BF16, 1:FP16
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;
    uint8_t templateId = static_cast<uint8_t>(context_->templateId);
    uint8_t cacheMode = static_cast<uint8_t>(*context_->cacheMode);

    auto hiddenStatesDtype = context_->hidden_states.desc->GetDataType();
    if (hiddenStatesDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (hiddenStatesDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto hiddenStatesDimNum = context_->hidden_states.shape->GetStorageShape().GetDimNum();
    if (hiddenStatesDimNum == K_POOL_DIM_NUM_3) {
        layout = 0;
    } else {
        layout = 1;
    }

    context_->tilingKey = GET_TPL_TILING_KEY(layout, dtype, coff, cacheMode, templateId);
    OP_LOGI(context_->opName, "KPool dtype:%hhu layout:%hhu  coff:%hhu, cacheMode: %u, template_id:%hhu", dtype,
            layout, coff, cacheMode, templateId);
    OP_LOGI(context_->opName, "KPool tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaHiddenStates() || ge::GRAPH_SUCCESS != CheckSingleParaWk() ||
        ge::GRAPH_SUCCESS != CheckSingleParaGateWeight() || ge::GRAPH_SUCCESS != CheckSingleParaStateCache() ||
        ge::GRAPH_SUCCESS != CheckSingleParaApe() || ge::GRAPH_SUCCESS != CheckSingleParaCacheBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqLens() || ge::GRAPH_SUCCESS != CheckSingleParaSeqUsed() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos() || ge::GRAPH_SUCCESS != CheckSingleParaPooledKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() || ge::GRAPH_SUCCESS != CheckSingleParaCoff() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCacheMode()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus KPoolTiling::CheckFeatureValueSupport(const T *featureValue,
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
ge::graphStatus KPoolTiling::CheckAttrValueSupport(const T *attrValue, const std::vector<T> &expectAttrValList,
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
void KPoolTiling::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
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

static std::string LayoutTypeToStr(LayoutType layout)
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

ge::graphStatus KPoolTiling::CheckDimNumInLayoutSupport(const std::string &layout, const gert::StorageShape *shape,
                                                             const std::string &name) const
{
    const auto &dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE(context_->opName, "When layout is %s, %s dimension should be %u, but it's %zu", layout.c_str(),
                        name.c_str(), dimIt->second, shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                    const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(
            it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(context_->opName, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(
            std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void KPoolTiling::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
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
        OP_LOGE("KPool", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

ge::graphStatus KPoolTiling::CheckDimNumSupport(const gert::StorageShape *shape, const std::string &name) const
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

ge::graphStatus KPoolTiling::CheckSingleParaHiddenStates() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->hidden_states.desc, HIDDEN_STATES_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->hidden_states.shape, HIDDEN_STATES_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout), context_->hidden_states.shape, HIDDEN_STATES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaWk() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wk.desc, WK_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wk.shape, WK_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaGateWeight() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->gate_weight.desc, GATE_WEIGHT_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->gate_weight.shape, GATE_WEIGHT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaStateCache() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->stateCache.desc, STATE_CACHE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->stateCache.shape, STATE_CACHE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaApe() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->ape.desc, APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->ape.shape, APE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaCacheBlockTable() const
{
    if (context_->cacheBlockTable.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cacheBlockTable.desc, CACHE_BLOCK_TABLE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cacheBlockTable.shape, CACHE_BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaSeqLens() const
{
    if (context_->seqLens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->seqLens.desc, SEQ_LENS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->seqLens.shape, SEQ_LENS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaSeqUsed() const
{
    if (context_->seqUsed.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->seqUsed.desc, SEQ_USED_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->seqUsed.shape, SEQ_USED_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaStartPos() const
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

ge::graphStatus KPoolTiling::CheckSingleParaPooledKey() const
{
    if (context_->pooledKey.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->pooledKey.desc, POOLED_KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->pooledKey.shape, POOLED_KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaCmpRatio() const
{
    uint32_t cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    OP_CHECK_IF(cmpRatio > MAX_CMPRATIO_SIZE || cmpRatio < MIN_CMPRATIO_SIZE,
                OP_LOGE(context_->opName, "cmpRatio should be within [%u, %u], but got %u", MIN_CMPRATIO_SIZE,
                        MAX_CMPRATIO_SIZE, cmpRatio),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaCoff() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->coff, COFF, COFF_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckSingleParaCacheMode() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->cacheMode, CACHE_MODE, CACHE_MODE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(context_->hidden_states.shape == nullptr, OP_LOGE(context_->opName, "tensor hidden_states is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->hidden_states.desc == nullptr, OP_LOGE(context_->opName, "tensor hidden_states is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wk.shape == nullptr, OP_LOGE(context_->opName, "tensor wk is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wk.desc == nullptr, OP_LOGE(context_->opName, "tensor wk is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->gate_weight.shape == nullptr, OP_LOGE(context_->opName, "tensor gate_weight is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->gate_weight.desc == nullptr, OP_LOGE(context_->opName, "tensor gate_weight is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.shape == nullptr, OP_LOGE(context_->opName, "tensor stateCache is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.desc == nullptr, OP_LOGE(context_->opName, "tensor stateCache is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.shape == nullptr, OP_LOGE(context_->opName, "tensor ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.desc == nullptr, OP_LOGE(context_->opName, "tensor ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cacheBlockTable.shape == nullptr,
                OP_LOGE(context_->opName, "tensor cacheBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cacheBlockTable.desc == nullptr,
                OP_LOGE(context_->opName, "tensor cacheBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->pooledKey.shape == nullptr, OP_LOGE(context_->opName, "tensor pooledKey is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->pooledKey.desc == nullptr, OP_LOGE(context_->opName, "tensor pooledKey is nullptr"),
                return ge::GRAPH_FAILED);
    if (context_->layout == LayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context_->seqLens.desc == nullptr,
                    OP_LOGE(context_->opName, "In TH layout, tensor seqLens should not be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->seqLens.shape == nullptr,
                    OP_LOGE(context_->opName, "In TH layout, tensor seqLens should not be nullptr"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context_->seqLens.desc != nullptr,
                    OP_LOGE(context_->opName, "In BSH layout, tensor seqLens must be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->seqLens.shape != nullptr,
                    OP_LOGE(context_->opName, "In BSH layout, tensor seqLens must be nullptr"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->cmpRatio == nullptr, OP_LOGE(context_->opName, "attr cmpRatio is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&baseParams_->headDim, HEAD_DIM, "headDim")) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseParams_->hiddenSize > MAX_HIDDEN_SIZE || baseParams_->hiddenSize < MIN_HIDDEN_SIZE ||
                    baseParams_->hiddenSize % ALIGN_FACTOR_HIDDEN_SIZE != 0,
                OP_LOGE(context_->opName, "hiddenSize should be within [%u, %u] and be 512-aligned, but got %u",
                        MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE, baseParams_->hiddenSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pageAttentionParams_->blockSize > MAX_BLOCK_SIZE || pageAttentionParams_->blockSize < MIN_BLOCK_SIZE,
                OP_LOGE(context_->opName, "blockSize should be within [%u, %u], but got %u", MIN_BLOCK_SIZE,
                        MAX_BLOCK_SIZE, pageAttentionParams_->blockSize),
                return ge::GRAPH_FAILED);
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::RING_BUFFER)) {
        OP_CHECK_IF(pageAttentionParams_->blockNum < baseParams_->batchSize,
                    OP_LOGE(context_->opName,
                            "when cacheMode is %u, blockNum should not be less than batchSize(%u), but got %u",
                            static_cast<uint8_t>(CACHE_MODE::RING_BUFFER), baseParams_->batchSize,
                            pageAttentionParams_->blockNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            context_->cacheBlockTable.shape->GetStorageShape().GetDimNum() != K_POOL_DIM_NUM_1,
            OP_LOGE(context_->opName, "when cacheMode is %u, cacheBlockTable dim num should be equal to %u, but got %u",
                    static_cast<uint8_t>(CACHE_MODE::RING_BUFFER), K_POOL_DIM_NUM_1,
                    context_->cacheBlockTable.shape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            context_->cacheBlockTable.shape->GetStorageShape().GetDimNum() != K_POOL_DIM_NUM_2,
            OP_LOGE(context_->opName, "when cacheMode is %u, cacheBlockTable dim num should be equal to %u, but got %u",
                    static_cast<uint8_t>(CACHE_MODE::LINEAR_BUFFER), K_POOL_DIM_NUM_2,
                    context_->cacheBlockTable.shape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::LogErrorShapeConsistency(const std::string &name, const gert::StorageShape *shape,
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

ge::graphStatus KPoolTiling::CheckShapeConsistency() const
{
    auto coffD = coff * baseParams_->headDim;
    uint32_t stateNum = 2;
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("cacheBlockTable", context_->cacheBlockTable.shape,
                                                      K_POOL_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("seqLens", context_->seqLens.shape, K_POOL_DIM_INDEX_0,
                                                      "batchSize+1", baseParams_->batchSize + 1) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("seqUsed", context_->seqUsed.shape, K_POOL_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("startPos", context_->startPos.shape, K_POOL_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wk", context_->wk.shape, K_POOL_DIM_INDEX_1, "hiddenSize",
                                                      baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("gate_weight", context_->gate_weight.shape, K_POOL_DIM_INDEX_1,
                                                      "hiddenSize", baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wk", context_->wk.shape, K_POOL_DIM_INDEX_0,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("gate_weight", context_->gate_weight.shape, K_POOL_DIM_INDEX_0,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, K_POOL_DIM_INDEX_2,
                                                      "2*coff*headDim", stateNum * static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ape", context_->ape.shape, K_POOL_DIM_INDEX_1,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ape", context_->ape.shape, K_POOL_DIM_INDEX_0, "cmpRatio",
                                                      baseParams_->cmpRatio)) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::LINEAR_BUFFER) &&
        (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, K_POOL_DIM_INDEX_0,
                                                       "blockNum", pageAttentionParams_->blockNum) ||
         ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, K_POOL_DIM_INDEX_1,
                                                       "blockSize", pageAttentionParams_->blockSize))) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckDtypeConsistencyHiddenStates(const gert::CompileTimeTensorDesc *desc,
                                                         const std::string &name) const
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(actualDtype != context_->dtype,
                OP_LOGE(context_->opName, "%s datatype should be same with hidden_states: %s, but got %s", name.c_str(),
                        DataTypeToSerialString(actualDtype).c_str(), DataTypeToSerialString(context_->dtype).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckDtypeConsistency() const
{
    if (CheckDtypeConsistencyHiddenStates(context_->wk.desc, WK_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyHiddenStates(context_->gate_weight.desc, GATE_WEIGHT_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyHiddenStates(context_->pooledKey.desc, POOLED_KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckDimNumConsistency() const
{
    auto hiddenStatesDimNum = context_->hidden_states.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(hiddenStatesDimNum != context_->pooledKey.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(context_->opName, "pooledKey dim num should be equal to hidden_states: %lu, but got %zu", hiddenStatesDimNum,
                        context_->pooledKey.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckBlockDimConstrain() const
{
    uint32_t minBlockNum = baseParams_->headDim / 64; // 64 is the largest dBaseSize
    OP_CHECK_IF(aicNum_ < minBlockNum,
                OP_LOGE(context_->opName, "aicNum is %u, which should not be less than %u", aicNum_, minBlockNum),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KPoolTiling::CheckMultiParaConsistency() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS || CheckDtypeConsistency() != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

CMP_EXTERN_C ge::graphStatus TilingKPool(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KPool", "Context is nullptr."),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    KPoolContext k_poolContext{};
    if (KPoolTiling::ConvertContext(*context, k_poolContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to KPool context");
        return ge::GRAPH_FAILED;
    }
    KPoolTiling k_poolTiling(&k_poolContext);
    KPoolTilingData *tilingData = context->GetTilingData<KPoolTilingData>();
    OP_CHECK_IF(tilingData == nullptr, OPS_REPORT_VECTOR_INNER_ERR(k_poolContext.opName, "TilingData is nullptr."),
                return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    if (k_poolTiling.RunBigKernelTiling(tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(k_poolContext.tilingKey);
    context->SetBlockDim(k_poolContext.blockDim);
    OP_LOGI(k_poolContext.opName, "block dim: %u.", k_poolContext.blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForKPool(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KPool).Tiling(TilingKPool).TilingParse<KPoolCompileInfo>(TilingPrepareForKPool);
} // namespace optiling
