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

    constexpr const char *INPUT_NAMES[] = {
        "partial_block", "delta", "pseudo_query", "numerator", "logit_max", "exp_sum",
    };
    constexpr ge::DataType EXPECTED_DTYPES[] = {
        ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    };
    const gert::StorageShape *inputShapes[INPUT_EXP_SUM + 1] = {};
    const gert::CompileTimeTensorDesc *inputDescs[INPUT_EXP_SUM + 1] = {};

    for (uint32_t i = INPUT_PARTIAL_BLOCK; i <= INPUT_EXP_SUM; ++i) {
        inputShapes[i] = context_->GetInputShape(i);
        if (inputShapes[i] == nullptr) {
            OP_LOGE(opName_, "Input %s shape is nullptr.", INPUT_NAMES[i]);
            return ge::GRAPH_FAILED;
        }
        inputDescs[i] = context_->GetInputDesc(i);
        if (inputDescs[i] == nullptr) {
            OP_LOGE(opName_, "Input %s desc is nullptr.", INPUT_NAMES[i]);
            return ge::GRAPH_FAILED;
        }
        const ge::DataType actualDtype = inputDescs[i]->GetDataType();
        if (!CheckTensorDtype(opName_, INPUT_NAMES[i], actualDtype, EXPECTED_DTYPES[i])) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *partialRefStorageShape = context_->GetOutputShape(OUTPUT_PARTIAL_BLOCK);
    if (partialRefStorageShape == nullptr) {
        OP_LOGE(opName_, "Output partial_block_ref shape is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const auto *partialRefDesc = context_->GetOutputDesc(OUTPUT_PARTIAL_BLOCK);
    if (partialRefDesc == nullptr) {
        OP_LOGE(opName_, "Output partial_block_ref desc is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType partialRefDtype = partialRefDesc->GetDataType();
    if (!CheckTensorDtype(opName_, "partial_block_ref", partialRefDtype, ge::DT_FLOAT)) {
        return ge::GRAPH_FAILED;
    }
    const auto *hStorageShape = context_->GetOutputShape(OUTPUT_H);
    if (hStorageShape == nullptr) {
        OP_LOGE(opName_, "Output h shape is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const auto *hDesc = context_->GetOutputDesc(OUTPUT_H);
    if (hDesc == nullptr) {
        OP_LOGE(opName_, "Output h desc is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType hDtype = hDesc->GetDataType();
    if (!CheckTensorDtype(opName_, "h", hDtype, ge::DT_BF16)) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &partialShape = inputShapes[INPUT_PARTIAL_BLOCK]->GetOriginShape();
    const gert::Shape &deltaShape = inputShapes[INPUT_DELTA]->GetOriginShape();
    const gert::Shape &pseudoQueryShape = inputShapes[INPUT_PSEUDO_QUERY]->GetOriginShape();
    const gert::Shape &numeratorShape = inputShapes[INPUT_NUMERATOR]->GetOriginShape();
    const gert::Shape &logitMaxShape = inputShapes[INPUT_LOGIT_MAX]->GetOriginShape();
    const gert::Shape &expSumShape = inputShapes[INPUT_EXP_SUM]->GetOriginShape();
    const gert::Shape &partialRefShape = partialRefStorageShape->GetOriginShape();
    const gert::Shape &hShape = hStorageShape->GetOriginShape();

    if (!CheckShapeRank(opName_, "partial_block", partialShape, MATRIX_RANK) ||
        !CheckShapeRank(opName_, "delta", deltaShape, MATRIX_RANK) ||
        !CheckShapeRank(opName_, "pseudo_query", pseudoQueryShape, VECTOR_RANK) ||
        !CheckShapeRank(opName_, "numerator", numeratorShape, MATRIX_RANK) ||
        !CheckShapeRank(opName_, "logit_max", logitMaxShape, VECTOR_RANK) ||
        !CheckShapeRank(opName_, "exp_sum", expSumShape, VECTOR_RANK) ||
        !CheckShapeRank(opName_, "partial_block_ref", partialRefShape, MATRIX_RANK) ||
        !CheckShapeRank(opName_, "h", hShape, MATRIX_RANK)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckShapeMatches(opName_, "delta", deltaShape, "partial_block", partialShape) ||
        !CheckShapeMatches(opName_, "numerator", numeratorShape, "partial_block", partialShape) ||
        !CheckShapeMatches(opName_, "partial_block_ref", partialRefShape, "partial_block", partialShape) ||
        !CheckShapeMatches(opName_, "h", hShape, "partial_block", partialShape)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t tSize = partialShape.GetDim(0);
    const int64_t dSize = partialShape.GetDim(1);
    if (tSize <= 0) {
        const std::string tSizeStr = std::to_string(tSize);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "partial_block dimension T", tSizeStr.c_str(),
                                              "the T dimension of partial_block must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (dSize <= 0) {
        const std::string dSizeStr = std::to_string(dSize);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "partial_block dimension D", dSizeStr.c_str(),
                                              "the D dimension of partial_block must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (dSize > MAX_D_SIZE) {
        const std::string dSizeStr = std::to_string(dSize);
        const std::string reason =
            "the D dimension of partial_block must be in the range [1, " + std::to_string(MAX_D_SIZE) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "partial_block dimension D", dSizeStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    const uint64_t tSizeValue = static_cast<uint64_t>(tSize);
    const uint64_t dSizeValue = static_cast<uint64_t>(dSize);
    if (tSize > std::numeric_limits<int64_t>::max() / dSize) {
        OP_LOGE(opName_, "T * D exceeds the signed 64-bit kernel GM offset range, T=%ld, D=%ld.", tSize, dSize);
        return ge::GRAPH_FAILED;
    }
    if (!CheckVectorLength(opName_, "logit_max", logitMaxShape, tSize, "T") ||
        !CheckVectorLength(opName_, "exp_sum", expSumShape, tSize, "T") ||
        !CheckVectorLength(opName_, "pseudo_query", pseudoQueryShape, dSize, "D")) {
        return ge::GRAPH_FAILED;
    }

    tSize_ = tSizeValue;
    dSize_ = static_cast<uint32_t>(dSizeValue);

    auto *rawTilingData = context_->GetRawTilingData();
    if (rawTilingData == nullptr) {
        OP_LOGE(opName_, "Raw tiling data is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (rawTilingData->GetData() == nullptr) {
        OP_LOGE(opName_, "Raw tiling data buffer is nullptr.");
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
