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
#include <limits>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace block_attn_res_update {
namespace {

constexpr uint32_t INPUT_PARTIAL_BLOCK = 0;
constexpr uint32_t INPUT_DELTA = 1;
constexpr uint32_t INPUT_INTER_NUMERATOR = 2;
constexpr uint32_t INPUT_INTER_MAX = 3;
constexpr uint32_t INPUT_INTER_EXP_SUM = 4;
constexpr uint32_t INPUT_EFFECTIVE_QUERY = 5;
constexpr uint32_t OUTPUT_PARTIAL_BLOCK = 0;
constexpr uint32_t OUTPUT_H = 1;
constexpr uint32_t ATTR_EPSILON = 0;
constexpr float DEFAULT_EPSILON = 1e-6F;

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

} // namespace

ge::graphStatus BlockAttnResUpdateTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::GetWorkspaceSize()
{
    // Keep the framework workspace ABI slot but request no user workspace storage.
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    auto ret = ValidateShapesAndDtypes();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const auto *attrs = context_->GetAttrs();
    const float *epsilon = attrs == nullptr ? nullptr : attrs->GetAttrPointer<float>(ATTR_EPSILON);
    const float epsilonValue = epsilon == nullptr ? DEFAULT_EPSILON : *epsilon;
    if (!std::isfinite(epsilonValue) || epsilonValue < 0.0F) {
        OP_LOGE(opName_, "epsilon must be a finite non-negative scalar");
        return ge::GRAPH_FAILED;
    }
    epsilon_ = epsilonValue;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::ValidateShapesAndDtypes()
{
    constexpr const char *INPUT_NAMES[] = {
        "partial_block",
        "delta",
        "inter_numerator",
        "inter_max",
        "inter_exp_sum",
        "effective_query",
    };
    constexpr ge::DataType EXPECTED_DTYPES[] = {
        ge::DT_FLOAT,
        ge::DT_BF16,
        ge::DT_FLOAT,
        ge::DT_FLOAT,
        ge::DT_FLOAT,
        ge::DT_FLOAT,
    };

    for (uint32_t i = INPUT_PARTIAL_BLOCK; i <= INPUT_EFFECTIVE_QUERY; ++i) {
        if (context_->GetInputShape(i) == nullptr || context_->GetInputDesc(i) == nullptr) {
            OP_LOGE(opName_, "input %s shape or desc is null", INPUT_NAMES[i]);
            return ge::GRAPH_FAILED;
        }
        const ge::DataType actualDtype = context_->GetInputDesc(i)->GetDataType();
        if (actualDtype != EXPECTED_DTYPES[i]) {
            OP_LOGE(opName_, "input %s has invalid dtype %d, expected %d", INPUT_NAMES[i],
                    static_cast<int32_t>(actualDtype), static_cast<int32_t>(EXPECTED_DTYPES[i]));
            return ge::GRAPH_FAILED;
        }
    }
    if (context_->GetOutputShape(OUTPUT_PARTIAL_BLOCK) == nullptr ||
        context_->GetOutputDesc(OUTPUT_PARTIAL_BLOCK) == nullptr) {
        OP_LOGE(opName_, "output partial_block shape or desc is null");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType partialRefDtype = context_->GetOutputDesc(OUTPUT_PARTIAL_BLOCK)->GetDataType();
    if (partialRefDtype != ge::DT_FLOAT) {
        OP_LOGE(opName_, "output partial_block has invalid dtype %d, expected %d",
                static_cast<int32_t>(partialRefDtype), static_cast<int32_t>(ge::DT_FLOAT));
        return ge::GRAPH_FAILED;
    }
    if (context_->GetOutputShape(OUTPUT_H) == nullptr || context_->GetOutputDesc(OUTPUT_H) == nullptr) {
        OP_LOGE(opName_, "output h shape or desc is null");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType hDtype = context_->GetOutputDesc(OUTPUT_H)->GetDataType();
    if (hDtype != ge::DT_BF16) {
        OP_LOGE(opName_, "output h has invalid dtype %d, expected %d", static_cast<int32_t>(hDtype),
                static_cast<int32_t>(ge::DT_BF16));
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &partialShape = context_->GetInputShape(INPUT_PARTIAL_BLOCK)->GetOriginShape();
    const gert::Shape &deltaShape = context_->GetInputShape(INPUT_DELTA)->GetOriginShape();
    const gert::Shape &numeratorShape = context_->GetInputShape(INPUT_INTER_NUMERATOR)->GetOriginShape();
    const gert::Shape &maxShape = context_->GetInputShape(INPUT_INTER_MAX)->GetOriginShape();
    const gert::Shape &sumShape = context_->GetInputShape(INPUT_INTER_EXP_SUM)->GetOriginShape();
    const gert::Shape &queryShape = context_->GetInputShape(INPUT_EFFECTIVE_QUERY)->GetOriginShape();
    const gert::Shape &partialRefShape = context_->GetOutputShape(OUTPUT_PARTIAL_BLOCK)->GetOriginShape();
    const gert::Shape &hShape = context_->GetOutputShape(OUTPUT_H)->GetOriginShape();

    if (partialShape.GetDimNum() != MATRIX_RANK || deltaShape.GetDimNum() != MATRIX_RANK ||
        numeratorShape.GetDimNum() != MATRIX_RANK || partialRefShape.GetDimNum() != MATRIX_RANK ||
        hShape.GetDimNum() != MATRIX_RANK ||
        maxShape.GetDimNum() != VECTOR_RANK || sumShape.GetDimNum() != VECTOR_RANK ||
        queryShape.GetDimNum() != VECTOR_RANK) {
        OP_LOGE(opName_, "expected partial/delta/numerator/partial_ref/h rank 2 and max/sum/query rank 1");
        return ge::GRAPH_FAILED;
    }
    if (!SameShape(partialShape, deltaShape) || !SameShape(partialShape, numeratorShape) ||
        !SameShape(partialShape, partialRefShape) || !SameShape(partialShape, hShape)) {
        OP_LOGE(opName_,
                "partial_block input/output, delta, inter_numerator and h must have the same [T, D] shape");
        return ge::GRAPH_FAILED;
    }

    const int64_t t = partialShape.GetDim(0);
    const int64_t cols = partialShape.GetDim(1);
    if (t <= 0 || cols <= 0) {
        OP_LOGE(opName_, "T and D must be positive, but got T=%ld, D=%ld", t, cols);
        return ge::GRAPH_FAILED;
    }
    const uint64_t tValue = static_cast<uint64_t>(t);
    const uint64_t colsValue = static_cast<uint64_t>(cols);
    if (colsValue > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(opName_, "D exceeds the uint32_t tiling range, D=%ld", cols);
        return ge::GRAPH_FAILED;
    }
    if (tValue > std::numeric_limits<uint64_t>::max() / colsValue) {
        OP_LOGE(opName_, "T * D exceeds the kernel GM index range, T=%ld, D=%ld", t, cols);
        return ge::GRAPH_FAILED;
    }
    if (maxShape.GetDim(0) != t || sumShape.GetDim(0) != t || queryShape.GetDim(0) != cols) {
        OP_LOGE(opName_, "inter_max/inter_exp_sum must be [T] and effective_query must be [D]");
        return ge::GRAPH_FAILED;
    }

    totalRows_ = tValue;
    cols_ = static_cast<uint32_t>(colsValue);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::GetPlatformInfo()
{
    const auto *compileInfo = static_cast<const BlockAttnResUpdateCompileInfo *>(context_->GetCompileInfo());
    if (compileInfo != nullptr && compileInfo->coreNum != 0 && compileInfo->ubSize != 0) {
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
        return ge::GRAPH_SUCCESS;
    }

    const auto *platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(opName_, "platform and compile info are both unavailable");
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    if (coreNum_ == 0 || ubSize_ == 0) {
        OP_LOGE(opName_, "Invalid platform data: coreNum=%u, ubSize=%lu", coreNum_, ubSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace block_attn_res_update
} // namespace optiling
