/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_attn_res_update_tiling_base.h"

#include <array>
#include <cmath>
#include <graph/utils/type_utils.h>
#include <limits>
#include <string>
#include "log/log.h"

namespace optiling {
namespace block_attn_res_update {
namespace {

constexpr const char *OP_NAME = "BlockAttnResUpdate";
constexpr uint32_t INPUT_PARTIAL_BLOCK = 0;
constexpr uint32_t INPUT_DELTA = 1;
constexpr uint32_t INPUT_PSEUDO_QUERY = 2;
constexpr uint32_t INPUT_NUMERATOR = 3;
constexpr uint32_t INPUT_LOGIT_MAX = 4;
constexpr uint32_t INPUT_EXP_SUM = 5;
constexpr uint32_t OUTPUT_PARTIAL_BLOCK = 0;
constexpr uint32_t OUTPUT_H = 1;
constexpr uint32_t ATTR_EPS = 0;
constexpr float DEFAULT_EPS = 1e-6F;
constexpr int64_t MAX_D_SIZE = 8192;

constexpr uint32_t MATRIX_RANK = 2;
constexpr uint32_t VECTOR_RANK = 1;
constexpr size_t INPUT_COUNT = static_cast<size_t>(INPUT_EXP_SUM) + 1U;

constexpr std::array<const char *, INPUT_COUNT> INPUT_NAMES = {
    "partial_block", "delta", "pseudo_query", "numerator", "logit_max", "exp_sum",
};
constexpr std::array<ge::DataType, INPUT_COUNT> EXPECTED_DTYPES = {
    ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
};

struct TensorContextInfo {
    std::array<const gert::StorageShape *, INPUT_COUNT> inputShapes{};
    const gert::StorageShape *partialRefStorageShape = nullptr;
    const gert::StorageShape *hStorageShape = nullptr;
};

bool SameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

std::string ShapeToString(const gert::Shape &shape)
{
    std::string shapeStr = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0U) {
            shapeStr += ", ";
        }
        shapeStr += std::to_string(shape.GetDim(i));
    }
    shapeStr += "]";
    return shapeStr;
}

bool CheckShapeRank(const char *opName, const char *tensorName, const gert::Shape &shape, uint32_t expectedRank)
{
    const size_t actualRank = shape.GetDimNum();
    if (actualRank == expectedRank) {
        return true;
    }

    const std::string actualRankStr = std::to_string(actualRank) + "D";
    const std::string reason =
        "the shape dim of " + std::string(tensorName) + " must be " + std::to_string(expectedRank) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, tensorName, actualRankStr.c_str(), reason.c_str());
    return false;
}

bool CheckShapeMatches(const char *opName, const char *tensorName, const gert::Shape &shape, const char *referenceName,
                       const gert::Shape &referenceShape)
{
    if (SameShape(shape, referenceShape)) {
        return true;
    }

    const std::string shapeStr = ShapeToString(shape);
    const std::string reason = "the shape of " + std::string(tensorName) + " must be the same as " + referenceName +
                               ", whose shape is " + ShapeToString(referenceShape);
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, tensorName, shapeStr.c_str(), reason.c_str());
    return false;
}

bool CheckVectorLength(const char *opName, const char *tensorName, const gert::Shape &shape, int64_t expectedLength,
                       const char *axisName)
{
    if (shape.GetDim(0) == expectedLength) {
        return true;
    }

    const std::string shapeStr = ShapeToString(shape);
    const std::string reason = "the shape of " + std::string(tensorName) + " must be [" +
                               std::to_string(expectedLength) + "] to match the " + axisName +
                               " dimension of partial_block";
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, tensorName, shapeStr.c_str(), reason.c_str());
    return false;
}

bool CheckTensorDtype(const char *opName, const char *tensorName, ge::DataType actualDtype, ge::DataType expectedDtype)
{
    if (actualDtype == expectedDtype) {
        return true;
    }

    const std::string actualDtypeStr = ge::TypeUtils::DataTypeToSerialString(actualDtype);
    const std::string expectedDtypeStr = ge::TypeUtils::DataTypeToSerialString(expectedDtype);
    const std::string reason = "the dtype of " + std::string(tensorName) + " must be " + expectedDtypeStr;
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, tensorName, actualDtypeStr.c_str(), reason.c_str());
    return false;
}

bool CheckInputMetadata(gert::TilingContext *context, const char *opName, TensorContextInfo &tensorInfo)
{
    for (uint32_t i = INPUT_PARTIAL_BLOCK; i <= INPUT_EXP_SUM; ++i) {
        tensorInfo.inputShapes[i] = context->GetInputShape(i);
        if (tensorInfo.inputShapes[i] == nullptr) {
            OP_LOGE(opName, "Input %s shape is nullptr.", INPUT_NAMES[i]);
            return false;
        }

        const auto *inputDesc = context->GetInputDesc(i);
        if (inputDesc == nullptr) {
            OP_LOGE(opName, "Input %s desc is nullptr.", INPUT_NAMES[i]);
            return false;
        }
        if (!CheckTensorDtype(opName, INPUT_NAMES[i], inputDesc->GetDataType(), EXPECTED_DTYPES[i])) {
            return false;
        }
    }
    return true;
}

bool CheckOutputMetadata(gert::TilingContext *context, const char *opName, TensorContextInfo &tensorInfo)
{
    tensorInfo.partialRefStorageShape = context->GetOutputShape(OUTPUT_PARTIAL_BLOCK);
    if (tensorInfo.partialRefStorageShape == nullptr) {
        OP_LOGE(opName, "Output partial_block_ref shape is nullptr.");
        return false;
    }
    const auto *partialRefDesc = context->GetOutputDesc(OUTPUT_PARTIAL_BLOCK);
    if (partialRefDesc == nullptr) {
        OP_LOGE(opName, "Output partial_block_ref desc is nullptr.");
        return false;
    }
    if (!CheckTensorDtype(opName, "partial_block_ref", partialRefDesc->GetDataType(), ge::DT_FLOAT)) {
        return false;
    }

    tensorInfo.hStorageShape = context->GetOutputShape(OUTPUT_H);
    if (tensorInfo.hStorageShape == nullptr) {
        OP_LOGE(opName, "Output h shape is nullptr.");
        return false;
    }
    const auto *hDesc = context->GetOutputDesc(OUTPUT_H);
    if (hDesc == nullptr) {
        OP_LOGE(opName, "Output h desc is nullptr.");
        return false;
    }
    if (!CheckTensorDtype(opName, "h", hDesc->GetDataType(), ge::DT_BF16)) {
        return false;
    }
    return true;
}

bool CheckTensorRanks(const char *opName, const TensorContextInfo &tensorInfo)
{
    return CheckShapeRank(opName, "partial_block", tensorInfo.inputShapes[INPUT_PARTIAL_BLOCK]->GetOriginShape(),
                          MATRIX_RANK) &&
           CheckShapeRank(opName, "delta", tensorInfo.inputShapes[INPUT_DELTA]->GetOriginShape(), MATRIX_RANK) &&
           CheckShapeRank(opName, "pseudo_query", tensorInfo.inputShapes[INPUT_PSEUDO_QUERY]->GetOriginShape(),
                          VECTOR_RANK) &&
           CheckShapeRank(opName, "numerator", tensorInfo.inputShapes[INPUT_NUMERATOR]->GetOriginShape(),
                          MATRIX_RANK) &&
           CheckShapeRank(opName, "logit_max", tensorInfo.inputShapes[INPUT_LOGIT_MAX]->GetOriginShape(),
                          VECTOR_RANK) &&
           CheckShapeRank(opName, "exp_sum", tensorInfo.inputShapes[INPUT_EXP_SUM]->GetOriginShape(), VECTOR_RANK) &&
           CheckShapeRank(opName, "partial_block_ref", tensorInfo.partialRefStorageShape->GetOriginShape(),
                          MATRIX_RANK) &&
           CheckShapeRank(opName, "h", tensorInfo.hStorageShape->GetOriginShape(), MATRIX_RANK);
}

bool CheckRelatedShapes(const char *opName, const TensorContextInfo &tensorInfo)
{
    const gert::Shape &partialShape = tensorInfo.inputShapes[INPUT_PARTIAL_BLOCK]->GetOriginShape();
    return CheckShapeMatches(opName, "delta", tensorInfo.inputShapes[INPUT_DELTA]->GetOriginShape(), "partial_block",
                             partialShape) &&
           CheckShapeMatches(opName, "numerator", tensorInfo.inputShapes[INPUT_NUMERATOR]->GetOriginShape(),
                             "partial_block", partialShape) &&
           CheckShapeMatches(opName, "partial_block_ref", tensorInfo.partialRefStorageShape->GetOriginShape(),
                             "partial_block", partialShape) &&
           CheckShapeMatches(opName, "h", tensorInfo.hStorageShape->GetOriginShape(), "partial_block", partialShape);
}

bool CheckDimensions(const char *opName, const TensorContextInfo &tensorInfo, uint64_t &tSize, uint32_t &dSize)
{
    const gert::Shape &partialShape = tensorInfo.inputShapes[INPUT_PARTIAL_BLOCK]->GetOriginShape();
    const int64_t tDim = partialShape.GetDim(0);
    const int64_t dDim = partialShape.GetDim(1);
    if (tDim <= 0) {
        const std::string tDimStr = std::to_string(tDim);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "partial_block dimension T", tDimStr.c_str(),
                                              "the T dimension of partial_block must be greater than 0");
        return false;
    }
    if (dDim <= 0) {
        const std::string dDimStr = std::to_string(dDim);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "partial_block dimension D", dDimStr.c_str(),
                                              "the D dimension of partial_block must be greater than 0");
        return false;
    }
    if (dDim > MAX_D_SIZE) {
        const std::string dDimStr = std::to_string(dDim);
        const std::string reason =
            "the D dimension of partial_block must be in the range [1, " + std::to_string(MAX_D_SIZE) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "partial_block dimension D", dDimStr.c_str(), reason.c_str());
        return false;
    }
    if (tDim > std::numeric_limits<int64_t>::max() / dDim) {
        OP_LOGE(opName, "T * D exceeds the signed 64-bit kernel GM offset range, T=%ld, D=%ld.", tDim, dDim);
        return false;
    }

    const gert::Shape &logitMaxShape = tensorInfo.inputShapes[INPUT_LOGIT_MAX]->GetOriginShape();
    const gert::Shape &expSumShape = tensorInfo.inputShapes[INPUT_EXP_SUM]->GetOriginShape();
    const gert::Shape &pseudoQueryShape = tensorInfo.inputShapes[INPUT_PSEUDO_QUERY]->GetOriginShape();
    if (!CheckVectorLength(opName, "logit_max", logitMaxShape, tDim, "T") ||
        !CheckVectorLength(opName, "exp_sum", expSumShape, tDim, "T") ||
        !CheckVectorLength(opName, "pseudo_query", pseudoQueryShape, dDim, "D")) {
        return false;
    }

    tSize = static_cast<uint64_t>(tDim);
    dSize = static_cast<uint32_t>(dDim);
    return true;
}

bool CheckRawTilingData(gert::TilingContext *context, const char *opName)
{
    auto *rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr) {
        OP_LOGE(opName, "Raw tiling data is nullptr.");
        return false;
    }
    if (rawTilingData->GetData() == nullptr) {
        OP_LOGE(opName, "Raw tiling data buffer is nullptr.");
        return false;
    }
    return true;
}

} // namespace

ge::graphStatus BlockAttnResUpdateTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus BlockAttnResUpdateTilingBase::GetWorkspaceSize()
{
    // Reserve the workspace publishing flow. The current kernel does not use workspace storage.
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE(OP_NAME, "Context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    if (opName_ == nullptr) {
        OP_LOGE(OP_NAME, "Node name is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto ret = CheckContext();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const auto *attrs = context_->GetAttrs();
    const float *eps = attrs == nullptr ? nullptr : attrs->GetAttrPointer<float>(ATTR_EPS);
    const float epsValue = eps == nullptr ? DEFAULT_EPS : *eps;
    if (!std::isfinite(epsValue) || epsValue <= 0.0F) {
        const std::string epsValueStr = std::to_string(epsValue);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "eps", epsValueStr.c_str(),
                                              "the value of eps must be finite and greater than 0");
        return ge::GRAPH_FAILED;
    }
    eps_ = epsValue;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::CheckContext()
{
    if (context_ == nullptr) {
        OP_LOGE(OP_NAME, "Context is nullptr.");
        return ge::GRAPH_FAILED;
    }

    TensorContextInfo tensorInfo;
    if (!CheckInputMetadata(context_, opName_, tensorInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckOutputMetadata(context_, opName_, tensorInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckTensorRanks(opName_, tensorInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckRelatedShapes(opName_, tensorInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckDimensions(opName_, tensorInfo, tSize_, dSize_)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckRawTilingData(context_, opName_)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::GetPlatformInfo()
{
    if (context_ == nullptr) {
        OP_LOGE(OP_NAME, "Context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const auto *compileInfo = static_cast<const BlockAttnResUpdateCompileInfo *>(context_->GetCompileInfo());
    if (compileInfo == nullptr) {
        OP_LOGE(opName_, "CompileInfo is nullptr.");
        return ge::GRAPH_FAILED;
    }
    compileInfo_ = *compileInfo;
    if (compileInfo_.aivNum == 0 || compileInfo_.ubSize == 0) {
        OP_LOGE(opName_, "Invalid CompileInfo: aivNum=%u, ubSize=%lu.", compileInfo_.aivNum, compileInfo_.ubSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace block_attn_res_update
} // namespace optiling
