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

void ConvertRequiredParams(gert::TilingContext &context, CompressorGradContext &compressorGradContext)
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

    compressorGradContext.dX.desc = context.GetOutputDesc(D_X_OUTPUT_INDEX);
    compressorGradContext.dX.shape = context.GetOutputShape(D_X_OUTPUT_INDEX);
    compressorGradContext.dWkv.desc = context.GetOutputDesc(D_WKV_OUTPUT_INDEX);
    compressorGradContext.dWkv.shape = context.GetOutputShape(D_WKV_OUTPUT_INDEX);
    compressorGradContext.dWgate.desc = context.GetOutputDesc(D_WGATE_OUTPUT_INDEX);
    compressorGradContext.dWgate.shape = context.GetOutputShape(D_WGATE_OUTPUT_INDEX);
    compressorGradContext.dApe.desc = context.GetOutputDesc(D_APE_OUTPUT_INDEX);
    compressorGradContext.dApe.shape = context.GetOutputShape(D_APE_OUTPUT_INDEX);

    compressorGradContext.dtype = compressorGradContext.x.desc->GetDataType();
    auto xDimNum = compressorGradContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_3) {
        compressorGradContext.layout = LayoutType::LAYOUT_BSH;
    } else if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_2) {
        compressorGradContext.layout = LayoutType::LAYOUT_TH;
    }
}

void ConvertOptionalParams(gert::TilingContext &context, CompressorGradContext &compressorGradContext)
{
    compressorGradContext.cuSeqlens.desc = context.GetOptionalInputDesc(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.cuSeqlens.shape = context.GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    compressorGradContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    compressorGradContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    compressorGradContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus ConvertContext(gert::TilingContext &context, CompressorGradContext &compressorGradContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("CompressorGrad", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    compressorGradContext.opName = context.GetNodeName();
    compressorGradContext.opType = context.GetNodeType();
    // compressorGradContext.platformInfo = context.GetPlatformInfo();
    ConvertRequiredParams(context, compressorGradContext);
    ConvertOptionalParams(context, compressorGradContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    compressorGradContext.coff = attrs->GetAttrPointer<uint32_t>(COFF_ATTR_INDEX);
    compressorGradContext.cmpRatio = attrs->GetAttrPointer<uint32_t>(CMP_RATIO_ATTR_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    compressorGradContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetBaseInfo(CompressorGradContext &compressorGradContext,
                            CompressorGradBaseParams &compressorGradBaseParams, uint32_t aicNum)
{
    if (compressorGradContext.x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3) {
        compressorGradBaseParams.batchSize =
            compressorGradContext.x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
        compressorGradBaseParams.seqSize =
            compressorGradContext.x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
        compressorGradBaseParams.hiddenSize =
            compressorGradContext.x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_2);
        compressorGradBaseParams.tokenSize =
            compressorGradBaseParams.batchSize * compressorGradBaseParams.seqSize;
    } else {
        compressorGradBaseParams.batchSize =
            compressorGradContext.cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) - 1;
        compressorGradBaseParams.tokenSize =
            compressorGradContext.x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
        compressorGradBaseParams.hiddenSize =
            compressorGradContext.x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    }

    uint8_t coff = compressorGradContext.coff == nullptr ? 1 : static_cast<uint8_t>(*compressorGradContext.coff);
    compressorGradBaseParams.headDim =
        compressorGradContext.wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) / coff;
    compressorGradBaseParams.featureDim =
        compressorGradContext.wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
    compressorGradBaseParams.cmpRatio = static_cast<uint32_t>(*compressorGradContext.cmpRatio);
    compressorGradBaseParams.nSize = 2; // 2:每个核处理两个基本块后做全核同步
    compressorGradBaseParams.usedCoreNum = aicNum;
    OP_LOGI(compressorGradContext.opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u",
            compressorGradBaseParams.batchSize, compressorGradBaseParams.tokenSize,
            compressorGradBaseParams.cmpRatio, coff);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcWorkSpace(CompressorGradContext &compressorGradContext,
    CompressorGradBaseParams &compressorGradBaseParams, CompressorGradWorkspaceParams &compressorGradWorkspaceParams,
    size_t &workspaceSize, size_t libapiSize, uint32_t aicNum)
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4; // 4: fp32
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;  // 4: fp32
    constexpr uint32_t M_BASE_SIZE = 128;
    uint8_t coff = compressorGradContext.coff == nullptr ? 1 : static_cast<uint8_t>(*compressorGradContext.coff);
    auto dBaseSize = 128 / coff;
    uint32_t cmpRatio = compressorGradBaseParams.cmpRatio;
    uint32_t headDim = compressorGradBaseParams.headDim;
    uint32_t hiddenSize = compressorGradBaseParams.hiddenSize;
    uint32_t cubeCoreNum = aicNum;
    uint32_t groupSize = headDim / 128;  // 与 kernel groupSize = headDim // D_BASE_SIZE 一致
    uint32_t groupNum = cubeCoreNum / groupSize;
    uint32_t cmpSize = coff * cmpRatio * headDim;
    uint32_t totalHeadDim = coff * headDim;
    uint32_t coffCoef = 2 / coff;
    uint32_t dealScNum = 128 / cmpRatio;
    uint32_t groupDealScNum = dealScNum * coffCoef;
    uint32_t groupRowStride = groupDealScNum * cmpRatio + (coff - 1) * cmpRatio;  // 与 kernel 一致
    uint32_t dbRatio = 2;

    // 与 kernel 的 workspace 指针链逐分区对齐（元素数，FP32）:
    //   ape / dWkv / dWgate 单缓冲（v3 kernel 无 dbIdx）；dX / x / dXCache 按 dbRatio=2 双缓冲
    uint64_t apeWorkSpaceSize = static_cast<uint64_t>(groupNum) * cmpSize * coffCoef;
    uint64_t dXWorkSpaceSize = static_cast<uint64_t>(dbRatio) * cubeCoreNum * (M_BASE_SIZE * 2) * hiddenSize;
    uint64_t dWeightWorkSpaceSize = static_cast<uint64_t>(groupNum) * totalHeadDim * hiddenSize;
    // dWkv / dWGate 各占一份 dWeightWorkSpaceSize
    uint64_t xWorkSpaceSize = static_cast<uint64_t>(dbRatio) * groupNum * groupRowStride * hiddenSize;
    uint64_t dXCacheWorkSpaceSize = static_cast<uint64_t>(dbRatio) * cmpRatio * hiddenSize;

    workspaceSize = libapiSize;
    workspaceSize += (apeWorkSpaceSize + dXWorkSpaceSize + dWeightWorkSpaceSize * 2 +
                     xWorkSpaceSize + dXCacheWorkSpaceSize) * MM1_RES_ELEM_SIZE;

    if (compressorGradContext.workSpaces) {
        compressorGradContext.workSpaces[0] = workspaceSize;
    }

    OP_LOGI(compressorGradContext.opName,
            "Tiling info: workspaceSize = %zu (ape=%llu dx=%llu dw=%llu x=%llu dxcache=%llu)",
            workspaceSize, apeWorkSpaceSize, dXWorkSpaceSize,
            dWeightWorkSpaceSize * 2, xWorkSpaceSize, dXCacheWorkSpaceSize);
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

ge::graphStatus GenTilingKey(CompressorGradContext &compressorGradContext)
{
    // 0:BF16, 1:FP16
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;

    auto xDtype = compressorGradContext.x.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto xDimNum = compressorGradContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_3) {
        layout = 0;
    } else {
        layout = 1;
    }

    uint8_t coff = compressorGradContext.coff == nullptr ? 1 : static_cast<uint8_t>(*compressorGradContext.coff);
    compressorGradContext.tilingKey = static_cast<uint64_t>(coff)
                                        | (static_cast<uint64_t>(layout) << 2)
                                        | (static_cast<uint64_t>(dtype) << 3);
    
    OP_LOGI(compressorGradContext.opName,
            "CompressorGrad dtype:%hhu layout:%hhu  coff:%hhu, CompressorGrad tilingKey:0x%lx",
            dtype, layout, coff, compressorGradContext.tilingKey);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                           const std::string &name, const std::string subName, const char *opName)
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE(opName, "%s %s only supports %s, but got %s", name.c_str(), subName.c_str(), oss.str().c_str(),
            to_string(actualValue).c_str());
}

template <typename T>
ge::graphStatus CheckFeatureValueSupport(const T *featureValue,
                                         const std::vector<T> &expectFeatureValList,
                                         const std::string &name, const char* opName)
{
    if (std::find(expectFeatureValList.begin(), expectFeatureValList.end(), *featureValue) ==
        expectFeatureValList.end()) {
        LogErrorNumberSupport(expectFeatureValList, *featureValue, name, "feature value", opName);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CheckAttrValueSupportInterval(const T *attrValue, const uint32_t minVal, const uint32_t maxVal,
                                              const std::string &name, const char *opName)
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    
    std::string attr_value = "attr value";
    if (*attrValue < minVal ||  *attrValue > maxVal) {
        std::ostringstream oss;
        oss << "[" << minVal << ", " << maxVal << "]";
        OP_LOGE(opName, "%s %s only supports value in range %s, but got %s",
                name.c_str(),
                attr_value.c_str(),
                oss.str().c_str(),
                std::to_string(*attrValue).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CheckAttrValueSupportList(const T *attrValue, const std::vector<T> &expectAttrValList,
                                          const std::string &name, const char *opName)
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectAttrValList.begin(), expectAttrValList.end(), *attrValue) == expectAttrValList.end()) {
        LogErrorNumberSupport(expectAttrValList, *attrValue, name, "attr value", opName);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

namespace compressor_grad_tiling {
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
} // namespace compressor_grad_tiling

ge::graphStatus CheckDimNumInLayoutSupport(CompressorGradContext &compressorGradContext, const std::string &layout,
                                           const gert::StorageShape *shape, const std::string &name)
{
    const auto &dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE(compressorGradContext.opName, "When layout is %s, %s dimension should be %u, but it's %zu",
                        layout.c_str(), name.c_str(), dimIt->second, shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                          const ge::DataType &actualDtype, const std::string &name, const char *opName)
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << DataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName, "Tensor %s only supports dtype %s, but got %s", name.c_str(), oss.str().c_str(),
            DataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus CheckDtypeSupport(CompressorGradContext &compressorGradContext, const gert::CompileTimeTensorDesc *desc,
                                  const std::string &name)
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(
            it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(compressorGradContext.opName, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP",
                    name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) ==
                        expectDtypeList.end(),
                    LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(),
                                         name, compressorGradContext.opName), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
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

ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape, const std::string &name, const char *opName)
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto &it = DIM_NUM_MAP.find(name);
    OP_CHECK_IF(it == DIM_NUM_MAP.end(),
                OP_LOGE(opName, "%s dim number support list should be specify in DIM_NUM_MAP", name.c_str()),
                return ge::GRAPH_FAILED);
    auto &expectDimNumList = it->second;
    OP_CHECK_IF(std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
                    expectDimNumList.end(),
                LogErrorNumberSupport(expectDimNumList, static_cast<uint32_t>(shape->GetStorageShape().GetDimNum()),
                                      name, "dimension", opName),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaX(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.x.desc, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.x.shape, X_NAME, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(compressorGradContext,
            compressor_grad_tiling::LayoutTypeToStr(compressorGradContext.layout),
                compressorGradContext.x.shape, X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaWkv(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.wkv.desc, WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.wkv.shape,
                                                WKV_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaWgate(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.wgate.desc, WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.wgate.shape,
                                                WGATE_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaDCmpKv(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext,
                                               compressorGradContext.dCmpKv.desc, D_CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.dCmpKv.shape,
                                                D_CMP_KV_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaSoftmaxScore(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext,
            compressorGradContext.softmaxScore.desc, SOFTMAX_SCORE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.softmaxScore.shape,
            SOFTMAX_SCORE_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus CheckSingleParaKV(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.kv.desc, KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.kv.shape,
                                                KV_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaCuSeqlens(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.cuSeqlens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext,
            compressorGradContext.cuSeqlens.desc, CU_SEQLENS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.cuSeqlens.shape,
             CU_SEQLENS_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaSeqused(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.seqUsed.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext,
            compressorGradContext.seqUsed.desc, SEQUSED_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.seqUsed.shape,
            SEQUSED_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaStartPos(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.startPos.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext,
            compressorGradContext.startPos.desc, START_POS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.startPos.shape,
            START_POS_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaDX(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.dX.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext,
            compressorGradContext.dX.desc, D_X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.dX.shape,
            D_X_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaDWkv(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.dWkv.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext,
            compressorGradContext.dWkv.desc, D_WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.dWkv.shape,
            D_WKV_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaDWgate(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.dWgate.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext,
            compressorGradContext.dWgate.desc, D_WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.dWgate.shape,
            D_WGATE_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaDApe(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.dApe.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext,
            compressorGradContext.dApe.desc, D_APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.dApe.shape,
            D_APE_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaCmpRatio(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupportInterval(compressorGradContext.cmpRatio,
            MIN_CMP_RATIO, MAX_CMP_RATIO, CMP_RATIO_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaCoff(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupportList(compressorGradContext.coff,
            COFF, COFF_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFeature(CompressorGradContext &compressorGradContext,
    CompressorGradBaseParams &compressorGradBaseParams,
    CompressorGradPageAttentionParams &compressorGradPageAttentionParams)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&compressorGradBaseParams.headDim,
            HEAD_DIM, "headDim", compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(compressorGradBaseParams.hiddenSize > MAX_HIDDEN_SIZE ||
                compressorGradBaseParams.hiddenSize < MIN_HIDDEN_SIZE ||
                    compressorGradBaseParams.hiddenSize % ALIGN_FACTOR_HIDDEN_SIZE != 0,
                OP_LOGE(compressorGradContext.opName,
                    "hiddenSize should be whthin [1k, 10k] and be 512-aligned, but got %u",
                    compressorGradBaseParams.hiddenSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        compressorGradPageAttentionParams.blockSize < MIN_BLOCK_SIZE,
        OP_LOGE(compressorGradContext.opName, "blockSize should not be less than 1, but got %u",
            compressorGradPageAttentionParams.blockSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogErrorShapeConsistency(const std::string &name, const gert::StorageShape *shape,
                                         const uint32_t &dimNum, const std::string &subName,
                                         const uint32_t &expectNum, const char *opName)
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE(opName, "%s shape dim %u, should be equal to %s: %u, but got %u", name.c_str(),
                        dimNum, subName.c_str(), expectNum, actualNum),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeConsistency(CompressorGradContext &compressorGradContext,
    CompressorGradBaseParams &compressorGradBaseParams)
{
    uint8_t coff = compressorGradContext.coff == nullptr ? 1 : static_cast<uint8_t>(*compressorGradContext.coff);
    auto coffD = compressorGradBaseParams.headDim * coff;
    if (ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("cuSeqlens", compressorGradContext.cuSeqlens.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                     "batchSize+1", compressorGradBaseParams.batchSize + 1, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("seqUsed", compressorGradContext.seqUsed.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                     "batchSize", compressorGradBaseParams.batchSize, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("startPos", compressorGradContext.startPos.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                     "batchSize", compressorGradBaseParams.batchSize, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("wkv", compressorGradContext.wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "hiddenSize",
                                     compressorGradBaseParams.hiddenSize, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("wgate", compressorGradContext.wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_1,
                                     "hiddenSize", compressorGradBaseParams.hiddenSize, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("wkv", compressorGradContext.wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                     "coff*headDim", static_cast<uint32_t>(coffD), compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("wgate", compressorGradContext.wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                     "coff*headDim", static_cast<uint32_t>(coffD), compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }

    if (compressorGradContext.x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_2 &&
        compressorGradContext.dCmpKv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_2 &&
        compressorGradContext.softmaxScore.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        compressorGradContext.kv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3) {
        if (ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("dCmpKv", compressorGradContext.dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "headDim",
                                         compressorGradBaseParams.headDim, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("softmaxScore", compressorGradContext.softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "coff*cmp_ratio",
                                         coff * compressorGradBaseParams.cmpRatio, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("softmaxScore", compressorGradContext.softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim",
                                         compressorGradBaseParams.headDim, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("kv", compressorGradContext.kv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "coff*cmp_ratio",
                                         coff * compressorGradBaseParams.cmpRatio, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("kv", compressorGradContext.kv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim",
                                         compressorGradBaseParams.headDim, compressorGradContext.opName)) {
            return ge::GRAPH_FAILED;
        }
    }

    if (compressorGradContext.x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        compressorGradContext.dCmpKv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        compressorGradContext.softmaxScore.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_4 &&
        compressorGradContext.kv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_4) {
        if (ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("dCmpKv", compressorGradContext.dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize",
                                         compressorGradBaseParams.batchSize, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("dCmpKv", compressorGradContext.dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim",
                                         compressorGradBaseParams.headDim, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("softmaxScore", compressorGradContext.softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize",
                                         compressorGradBaseParams.batchSize, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("softmaxScore", compressorGradContext.softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio",
                                         coff * compressorGradBaseParams.cmpRatio, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("softmaxScore", compressorGradContext.softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "headDim",
                                         compressorGradBaseParams.headDim, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("kv", compressorGradContext.kv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize",
                                         compressorGradBaseParams.batchSize, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("kv", compressorGradContext.kv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio",
                                         coff * compressorGradBaseParams.cmpRatio, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("kv", compressorGradContext.kv.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "headDim",
                                         compressorGradBaseParams.headDim, compressorGradContext.opName)) {
                return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc *desc,
                                       const std::string &name, CompressorGradContext &compressorGradContext)
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(actualDtype != compressorGradContext.dtype,
                OP_LOGE(compressorGradContext.opName, "%s datatype should be same with x: %s, but got %s", name.c_str(),
                        DataTypeToSerialString(actualDtype).c_str(),
                        DataTypeToSerialString(compressorGradContext.dtype).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeConsistency(CompressorGradContext &compressorGradContext)
{
    if (CheckDtypeConsistencyX(compressorGradContext.wkv.desc,
            WKV_NAME, compressorGradContext) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(compressorGradContext.wgate.desc,
            WGATE_NAME, compressorGradContext) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(compressorGradContext.dCmpKv.desc,
            D_CMP_KV_NAME, compressorGradContext) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(compressorGradContext.dX.desc,
            D_X_NAME, compressorGradContext) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(compressorGradContext.dWkv.desc,
            D_WKV_NAME, compressorGradContext) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(compressorGradContext.dWgate.desc,
            D_WGATE_NAME, compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDimNumConsistency(CompressorGradContext &compressorGradContext)
{
    auto xDimNum = compressorGradContext.x.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != compressorGradContext.dX.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(compressorGradContext.opName, "dX dim num should be equal to x: %lu, but got %zu", xDimNum,
                        compressorGradContext.dX.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(xDimNum != compressorGradContext.dCmpKv.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(compressorGradContext.opName, "dCmpKv dim num should be equal to x: %lu, but got %zu", xDimNum,
                        compressorGradContext.dCmpKv.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(xDimNum != compressorGradContext.softmaxScore.shape->GetStorageShape().GetDimNum() - 1,
                OP_LOGE(compressorGradContext.opName, "softmaxScore dim num should be x dim + 1: %lu, but got %zu",
                        xDimNum + 1, compressorGradContext.softmaxScore.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(xDimNum != compressorGradContext.kv.shape->GetStorageShape().GetDimNum() - 1,
                OP_LOGE(compressorGradContext.opName, "kv dim num should be x dim + 1: %lu, but got %zu", xDimNum + 1,
                        compressorGradContext.kv.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckScenarioConsistency(CompressorGradContext &compressorGradContext,
                                         CompressorGradBaseParams &compressorGradBaseParams)
{
    auto curCmpratio = compressorGradBaseParams.cmpRatio;
    auto curHeaddim = compressorGradBaseParams.headDim;
    auto curCoff = compressorGradContext.coff == nullptr ? 1 : static_cast<uint8_t>(*compressorGradContext.coff);
    std::vector<uint32_t> curScenario{curCmpratio, curCoff, curHeaddim};
    const std::vector<std::vector<uint32_t>> allowdScenarios = {{4, 2, 512}, {4, 2, 128}, {128, 1, 512}};

    OP_CHECK_IF(std::find(allowdScenarios.begin(), allowdScenarios.end(), curScenario) == allowdScenarios.end(),
                OP_LOGE(compressorGradContext.opName,
                        "Cmpratio Coff Headdim should be equal to {4, 2, 512}, {4, 2, 128}, {128, 1, 512}, \
                        but now cmpratio=%u, coff=%u, headdim=%u", curCmpratio, curCoff, curHeaddim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckBlockDimConstrain(CompressorGradContext &compressorGradContext,
                                       CompressorGradBaseParams &compressorGradBaseParams,
                                       uint32_t &aicNum)
{
    uint32_t minBlockNum = compressorGradBaseParams.headDim / 64; // 64 is the largest dBaseSize
    OP_CHECK_IF(aicNum < minBlockNum,
                OP_LOGE(compressorGradContext.opName, "aicNum is %u, which should not be less than %u", aicNum, minBlockNum),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMultiParaConsistency(CompressorGradContext &compressorGradContext,
                                          CompressorGradBaseParams &compressorGradBaseParams)
{
    if (CheckShapeConsistency(compressorGradContext, compressorGradBaseParams) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistency(compressorGradContext) != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency(compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
#ifdef DAY0_SCOPE
    if (CheckScenarioConsistency(compressorGradContext, compressorGradBaseParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
#endif
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSinglePara(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaX(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaWkv(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaWgate(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaDCmpKv(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaSoftmaxScore(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqused(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaDX(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWkv(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWgate(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaDApe(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaCoff(compressorGradContext)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckRequiredInOutExistence(CompressorGradContext &context)
{
    OP_CHECK_IF(context.x.shape == nullptr, OP_LOGE(context.opName, "tensor x is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.x.desc == nullptr, OP_LOGE(context.opName, "tensor x is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.wkv.shape == nullptr, OP_LOGE(context.opName, "tensor wkv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.wkv.desc == nullptr, OP_LOGE(context.opName, "tensor wkv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.wgate.shape == nullptr, OP_LOGE(context.opName, "tensor wgate is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.wgate.desc == nullptr, OP_LOGE(context.opName, "tensor wgate is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dCmpKv.shape == nullptr, OP_LOGE(context.opName, "tensor d_cmp_kv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dCmpKv.desc == nullptr, OP_LOGE(context.opName, "tensor d_cmp_kv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.softmaxScore.shape == nullptr, OP_LOGE(context.opName, "tensor softmax_score is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.softmaxScore.desc == nullptr, OP_LOGE(context.opName, "tensor softmax_score is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.kv.shape == nullptr,
                OP_LOGE(context.opName, "tensor kv is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.kv.desc == nullptr,
                OP_LOGE(context.opName, "tensor kv is nullptr"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.dX.shape == nullptr, OP_LOGE(context.opName, "tensor d_x is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dX.desc == nullptr, OP_LOGE(context.opName, "tensor d_x is nullptr"),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(context.dWkv.shape == nullptr, OP_LOGE(context.opName, "tensor d_wkv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dWkv.desc == nullptr, OP_LOGE(context.opName, "tensor d_wkv is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.dWgate.shape == nullptr, OP_LOGE(context.opName, "tensor d_wgate is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dWgate.desc == nullptr, OP_LOGE(context.opName, "tensor d_wgate is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.dApe.shape == nullptr, OP_LOGE(context.opName, "tensor d_ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dApe.desc == nullptr, OP_LOGE(context.opName, "tensor d_ape is nullptr"),
                return ge::GRAPH_FAILED);
    
    if (context.layout == LayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context.cuSeqlens.desc == nullptr,
                    OP_LOGE(context.opName, "In TH layout, tensor cuSeqlens should not be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cuSeqlens.shape == nullptr,
                    OP_LOGE(context.opName, "In TH layout, tensor cuSeqlens should not be nullptr"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context.cuSeqlens.desc != nullptr,
                    OP_LOGE(context.opName, "In BSH layout, tensor cuSeqlens must be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cuSeqlens.shape != nullptr,
                    OP_LOGE(context.opName, "In BSH layout, tensor cuSeqlens must be nullptr"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckRequiredAttrExistence(CompressorGradContext &context)
{
    OP_CHECK_IF(context.cmpRatio == nullptr, OP_LOGE(context.opName, "attr cmpRatio is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckRequiredParaExistence(CompressorGradContext &compressorGradcontext)
{
    if (CheckRequiredInOutExistence(compressorGradcontext) != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence(compressorGradcontext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckEmptyTensor(CompressorGradContext &compressorGradcontext)
{
    if (compressorGradcontext.x.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.wkv.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.wgate.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.dCmpKv.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.softmaxScore.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.kv.shape->GetStorageShape().GetShapeSize() == 0) {
        OP_LOGE(compressorGradcontext.opName, "Only input tensor x dim B or S or T supports to be 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CompressorGradTilingFunc(gert::TilingContext *context)
{
    CompressorGradContext compressorGradContext{};
    CompressorGradBaseParams compressorGradBaseParams{};
    CompressorGradPageAttentionParams compressorGradPageAttentionParams{};
    CompressorGradWorkspaceParams compressorGradWorkspaceParams{};
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("CompressorGrad", "Context is nullptr."),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    OP_CHECK_IF(context->GetPlatformInfo() == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "num of core obtained is 0."), return GRAPH_FAILED);
    
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);
    
    if (ConvertContext(*context, compressorGradContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to CompressorGrad context");
        return ge::GRAPH_FAILED;
    }

    compressorGradContext.blockDim = aicNum;
    if (CheckRequiredParaExistence(compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckEmptyTensor(compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckSinglePara(compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (SetBaseInfo(compressorGradContext, compressorGradBaseParams, aicNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckFeature(compressorGradContext, compressorGradBaseParams, compressorGradPageAttentionParams) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckMultiParaConsistency(compressorGradContext, compressorGradBaseParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckBlockDimConstrain(compressorGradContext, compressorGradBaseParams, aicNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    size_t libapiSize = 0;
    size_t workspaceSize = 0;

    if (CalcWorkSpace(compressorGradContext, compressorGradBaseParams, compressorGradWorkspaceParams, workspaceSize,
                      libapiSize, aicNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GenTilingKey(compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint8_t coff = compressorGradContext.coff == nullptr ? 1 : static_cast<uint8_t>(*compressorGradContext.coff);
    CompressorGradTiling *tilingData = context->GetTilingData<CompressorGradTiling>();
    tilingData->batchSize = compressorGradBaseParams.batchSize;
    tilingData->tokenSize = compressorGradBaseParams.tokenSize;
    tilingData->seqSize = compressorGradBaseParams.seqSize;
    tilingData->cmpRatio = compressorGradBaseParams.cmpRatio;
    tilingData->hiddenSize = compressorGradBaseParams.hiddenSize;
    tilingData->headDim = compressorGradBaseParams.headDim;
    // ── 核数（与 launch block_dim 一致；vec 每核 2 子核）──
    tilingData->cubeCoreNum = compressorGradContext.blockDim;
    tilingData->coreNum = compressorGradContext.blockDim * 2;
    // ── shape 派生 ──
    tilingData->totalHeadDim = coff * compressorGradBaseParams.headDim;
    tilingData->cmpRowCnt = coff * compressorGradBaseParams.cmpRatio;
    tilingData->cmpSize = coff * compressorGradBaseParams.cmpRatio * compressorGradBaseParams.headDim;
    tilingData->cmpKvBatchStride =
        (compressorGradBaseParams.seqSize + compressorGradBaseParams.cmpRatio - 1) / compressorGradBaseParams.cmpRatio;
    if (compressorGradContext.layout == LayoutType::LAYOUT_BSH) {
        tilingData->xRows = compressorGradBaseParams.batchSize * compressorGradBaseParams.seqSize;
        tilingData->cmpKvRows = compressorGradBaseParams.batchSize * tilingData->cmpKvBatchStride;
    } else {
        tilingData->xRows = compressorGradBaseParams.tokenSize;
        tilingData->cmpKvRows = std::min(compressorGradBaseParams.tokenSize,
            compressorGradBaseParams.tokenSize /
            compressorGradBaseParams.cmpRatio +compressorGradBaseParams.batchSize);
    }
    // ── 分核派生 ──
    tilingData->groupSize = compressorGradBaseParams.headDim / 128;
    tilingData->groupNum = compressorGradContext.blockDim / tilingData->groupSize;
    tilingData->cubeMBaseSize = 128 * (2 / coff);
    tilingData->coffCoef = 2 / coff;
    tilingData->dDealSize = 128 / coff;
    // ── workspace 分区（FP32 元素数；ape/dW 单缓冲，dX/x/dXCache 双缓冲 dbRatio=2）──
    tilingData->dapeWsSize = tilingData->groupNum * tilingData->cmpSize * tilingData->coffCoef;
    tilingData->dXWsSize = 2 * tilingData->cubeCoreNum * 256 * compressorGradBaseParams.hiddenSize;
    tilingData->dWWeightWsSize = tilingData->groupNum * tilingData->totalHeadDim * compressorGradBaseParams.hiddenSize;
    tilingData->xWsSize = 2 * tilingData->groupNum * tilingData->groupRowStride *
        compressorGradBaseParams.hiddenSize;
    tilingData->dXCacheWsSize = 2 * compressorGradBaseParams.cmpRatio * compressorGradBaseParams.hiddenSize;

    context->SetTilingKey(compressorGradContext.tilingKey);
    context->SetBlockDim(compressorGradContext.blockDim);
    OP_LOGI(compressorGradContext.opName, "block dim: %u.", compressorGradContext.blockDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCompressorGrad(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CompressorGrad)
    .Tiling(CompressorGradTilingFunc)
    .TilingParse<CompressorGradCompileInfo>(TilingParseForCompressorGrad);
} // namespace optiling