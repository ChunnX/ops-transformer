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
 * \file fused_causal_conv1d_cut_bh_tiling_arch35.cpp
 * \brief FusedCausalConv1dCutBH tiling implementation
 */

#include "fused_causal_conv1d_cut_bh_tiling_arch35.h"
#include <algorithm>
#include "securec.h"

namespace optiling {

bool FusedCausalConv1dCutBHTiling::IsCapable()
{
    return true;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::GetPlatformInfo()
{
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FusedCausalConv1dCutBHCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        totalCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
        if (totalCoreNum_ == 0UL) {
            OP_LOGE(context_->GetNodeName(), "coreNum is 0");
            return ge::GRAPH_FAILED;
        }
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (ubSize == static_cast<uint64_t>(0)) {
            OP_LOGE(context_->GetNodeName(), "ubSize is 0");
            return ge::GRAPH_FAILED;
        }
        ubSize_ = static_cast<uint64_t>(ubSize);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("FusedCausalConv1dCutBH", "context is null"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetShapeInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "GetShapeInfo FAILED"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetTypeInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "GetTypeInfo FAILED"),
        return ge::GRAPH_FAILED);

    // Perform all validations (depends on shapes and types)
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "FusedCausalConv1dCutBH CheckInputParams FAILED."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttrInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "GetAttrInfo FAILED"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetStrideInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "GetStrideInfo FAILED"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Shape related info: x/weight/conv_states shapes and derived fields
ge::graphStatus FusedCausalConv1dCutBHTiling::GetShapeInfo()
{
    // Get x shape
    auto xShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xOriginShape = xShape->GetOriginShape();

    // Support both 3D [batch, seq_len, dim] and 2D [cu_seq_len, dim] input
    if (xOriginShape.GetDimNum() == DIM_3) {
        xInputMode_ = X_INPUT_3D;  // 3D input mode
        batchSize_ = xOriginShape.GetDim(DIM_0);
        seqLen_ = xOriginShape.GetDim(DIM_1);
        dim_ = xOriginShape.GetDim(DIM_2);
    } else if (xOriginShape.GetDimNum() == DIM_2) {
        xInputMode_ = X_INPUT_2D;                // 2D input mode
        cuSeqLen_ = xOriginShape.GetDim(DIM_0);  // cu_seq_len = batch * seq_len
        dim_ = xOriginShape.GetDim(DIM_1);

        // For 2D input, query_start_loc must be specified to get batch
        auto queryStartLocShape = context_->GetOptionalInputShape(QUERY_START_LOC_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, queryStartLocShape);
        auto queryStartLocOriginShape = queryStartLocShape->GetOriginShape();

        // query_start_loc shape is (batch + 1,), so batch = dim0 - 1
        batchSize_ = queryStartLocOriginShape.GetDim(DIM_0) - 1;
        seqLen_ = 0;
    } else {
        OP_LOGE(context_->GetNodeName(), "X dimension number must be 2 or 3, but got %lu", xOriginShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    // Get weight shape
    auto weightShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    auto weightOriginShape = weightShape->GetOriginShape();
    if (weightOriginShape.GetDimNum() != DIM_2) {
        OP_LOGE(
            context_->GetNodeName(), "Weight dimension number must be 2, but got %lu", weightOriginShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    kernelSize_ = weightOriginShape.GetDim(DIM_0);

    // Get convStates shape to retrieve stateLen
    auto convStatesShape = context_->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesShape);
    auto convStatesOriginShape = convStatesShape->GetOriginShape();
    // stateLen is the second dimension of convStates [-1, stateLen, dim]
    stateLen_ = convStatesOriginShape.GetDim(DIM_1);

    return ge::GRAPH_SUCCESS;
}

// Type related info: dtypes and dtype size
ge::graphStatus FusedCausalConv1dCutBHTiling::GetTypeInfo()
{
    // Get data types
    xDtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    weightDtype_ = context_->GetInputDesc(WEIGHT_INDEX)->GetDataType();
    convStatesDtype_ = context_->GetInputDesc(CONV_STATES_INDEX)->GetDataType();

    // Get queryStartLoc dtype if available (optional input)
    auto queryStartLocDesc = context_->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
    if (queryStartLocDesc != nullptr) {
        queryStartLocDtype_ = queryStartLocDesc->GetDataType();
    }

    // Get cacheIndices dtype if available (optional input)
    auto cacheIndicesDesc = context_->GetOptionalInputDesc(CACHE_INDICES_INDEX);
    if (cacheIndicesDesc != nullptr) {
        cacheIndicesDtype_ = cacheIndicesDesc->GetDataType();
    }

    // Get numAcceptedToken dtype if available (optional input)
    auto numAcceptedTokenDesc = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKEN_INDEX);
    if (numAcceptedTokenDesc != nullptr) {
        numAcceptedTokenDtype_ = numAcceptedTokenDesc->GetDataType();
        hasAcceptTokenNum_ = 1;  // true
    } else {
        hasAcceptTokenNum_ = 0;  // false
    }

    // Get dtype size
    xDtypeSize_ = GetSizeByDataType(xDtype_);
    OP_CHECK_IF(xDtypeSize_ == 0,
        OP_LOGE(context_->GetNodeName(),
            "FusedCausalConv1dCutBH get x dtype[%s] size is 0.",
            Ops::Base::ToString(xDtype_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Attr related info: read op attributes
ge::graphStatus FusedCausalConv1dCutBHTiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *activationModePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_MODE_INDEX);
    if (activationModePtr != nullptr) {
        activationMode_ = *activationModePtr;
    }

    const int64_t *padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
    if (padSlotIdPtr != nullptr) {
        padSlotId_ = *padSlotIdPtr;
    }

    const int64_t *runModePtr = attrs->GetAttrPointer<int64_t>(ATTR_RUN_MODE_INDEX);
    if (runModePtr != nullptr) {
        runMode_ = *runModePtr;
    }

    const int64_t *residualConnectionPtr = attrs->GetAttrPointer<int64_t>(ATTR_RESIDUAL_CONNECTION_INDEX);
    if (residualConnectionPtr != nullptr) {
        residualConnection_ = *residualConnectionPtr;
    }

    return ge::GRAPH_SUCCESS;
}

// Stride related info: input strides when input is view
ge::graphStatus FusedCausalConv1dCutBHTiling::GetStrideInfo()
{
    // 获取 x 的 stride
    bool xIsView = context_->InputIsView(X_INDEX);
    if (xIsView) {
        auto *xStride = context_->GetInputStride(X_INDEX);
        OP_CHECK_IF(
            xStride == nullptr, OP_LOGE(context_->GetNodeName(), "x stride is invalid."), return ge::GRAPH_FAILED);

        if (xInputMode_ == X_INPUT_3D) {
            OP_CHECK_IF(xStride->GetDimNum() != DIM_3,
                OP_LOGE(context_->GetNodeName(), "The number of dimensions in x stride must be 3."),
                return ge::GRAPH_FAILED);
            xStride_ = xStride->GetStride(DIM_1);  // batch, seq_len, dim
        } else {
            OP_CHECK_IF(xStride->GetDimNum() != DIM_2,
                OP_LOGE(context_->GetNodeName(), "The number of dimensions in x stride must be 2."),
                return ge::GRAPH_FAILED);
            xStride_ = xStride->GetStride(DIM_0);  // cu_seq_len, dim
        }
    } else {
        xStride_ = dim_;
    }

    // 获取 cacheStates 的 stride
    bool cacheIsView = context_->InputIsView(CONV_STATES_INDEX);
    if (cacheIsView) {
        auto *cacheStride = context_->GetInputStride(CONV_STATES_INDEX);
        OP_CHECK_IF(cacheStride == nullptr,
            OP_LOGE(context_->GetNodeName(), "conv_states stride is invalid."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(cacheStride->GetDimNum() != DIM_3,
            OP_LOGE(context_->GetNodeName(), "The number of dimensions in conv_states stride must be 3."),
            return ge::GRAPH_FAILED);
        cacheStride0_ = cacheStride->GetStride(DIM_0);
        cacheStride1_ = cacheStride->GetStride(DIM_1);
    } else {
        cacheStride0_ = dim_ * stateLen_;
        cacheStride1_ = dim_;
    }

    return ge::GRAPH_SUCCESS;
}

// Validate X tensor shape
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateXShape()
{
    // Validate batch size: [1, 256]
    OP_CHECK_IF(batchSize_ < MIN_BATCH || batchSize_ > MAX_BATCH,
        OP_LOGE(context_->GetNodeName(),
            "X batch size must be in [%ld, %ld], but got %ld",
            MIN_BATCH,
            MAX_BATCH,
            batchSize_),
        return ge::GRAPH_FAILED);

    // For 3D input, validate sequence length
    if (xInputMode_ == X_INPUT_3D) {
        // Validate sequence length: m+1 where m in [0, 5], so seqLen in [1, 6]
        int64_t m = seqLen_ - 1;
        OP_CHECK_IF(m < MIN_M || m > MAX_M,
            OP_LOGE(context_->GetNodeName(),
                "X sequence length must be m+1 where m in [%ld, %ld], but got seqLen=%ld (m=%ld)",
                MIN_M,
                MAX_M,
                seqLen_,
                m),
            return ge::GRAPH_FAILED);
    }
    // For 2D input, cuSeqLen_ is used instead of seqLen_

    // Validate dimension: [128, 16384]
    OP_CHECK_IF(dim_ < MIN_DIM || dim_ > MAX_DIM,
        OP_LOGE(context_->GetNodeName(), "X dimension must be in [%ld, %ld], but got %ld", MIN_DIM, MAX_DIM, dim_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(dim_ % DIM_ALIGN_ELEMENT != 0,
        OP_LOGE(context_->GetNodeName(), "The dimension of x must be a multiple of 128, but got %ld", dim_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate weight tensor shape
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateWeightShape()
{
    auto weightShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    auto weightOriginShape = weightShape->GetOriginShape();

    // Validate weight dim matches x dim
    int64_t weightDim = weightOriginShape.GetDim(1);
    OP_CHECK_IF(weightDim != dim_,
        OP_LOGE(context_->GetNodeName(), "Weight dimension must match X dimension %ld, but got %ld", dim_, weightDim),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate conv states tensor shape
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateConvStatesShape()
{
    auto convStatesShape = context_->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesShape);
    auto convStatesOriginShape = convStatesShape->GetOriginShape();

    // Validate dimension number
    OP_CHECK_IF(convStatesOriginShape.GetDimNum() != DIM_3,
        OP_LOGE(context_->GetNodeName(),
            "ConvStates dimension number must be 3, but got %lu",
            convStatesOriginShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    // conv states shape: [-1, K-1+m, dim]
    // The second dimension should be K-1 + m = K-1 + (seqLen-1) = K + seqLen - 2
    // state_len must be greater than the maximum of width-1+seq_len-1 for all batches
    if (xInputMode_ == X_INPUT_3D) {
        int64_t expectedCacheLen = kernelSize_ + seqLen_ - 2;
        int64_t state_len = convStatesOriginShape.GetDim(DIM_1);
        OP_CHECK_IF(state_len < expectedCacheLen,
            OP_LOGE(context_->GetNodeName(),
                "state_len must be greater than width-1+seq_len-1 = %ld, but got %ld",
                expectedCacheLen,
                state_len),
            return ge::GRAPH_FAILED);
    }

    // Validate conv states dim matches x dim
    int64_t convStatesDim = convStatesOriginShape.GetDim(DIM_2);
    OP_CHECK_IF(convStatesDim != dim_,
        OP_LOGE(context_->GetNodeName(),
            "ConvStates dimension must match X dimension %ld, but got %ld",
            dim_,
            convStatesDim),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate cache indices tensor shape
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateCacheIndicesShape()
{
    // This is an optional input
    auto indicesShape = context_->GetOptionalInputShape(CACHE_INDICES_INDEX);
    if (indicesShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto indicesOriginShape = indicesShape->GetOriginShape();

    // Validate dimension number: must be 1
    OP_CHECK_IF(indicesOriginShape.GetDimNum() != DIM_1,
        OP_LOGE(context_->GetNodeName(),
            "CacheIndices dimension number must be 1, but got %lu",
            indicesOriginShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    // Validate shape matches batch size
    int64_t indicesLen = indicesOriginShape.GetDim(DIM_0);
    OP_CHECK_IF(indicesLen != batchSize_,
        OP_LOGE(context_->GetNodeName(),
            "CacheIndices length must match batch size %ld, but got %ld",
            batchSize_,
            indicesLen),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate accept token num tensor shape
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateNumAcceptedTokenShape()
{
    // This is an optional input
    if (context_->GetOptionalInputTensor(NUM_ACCEPTED_TOKEN_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    auto acceptShape = context_->GetOptionalInputShape(NUM_ACCEPTED_TOKEN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, acceptShape);
    auto acceptOriginShape = acceptShape->GetOriginShape();

    // Validate dimension number: must be 1
    OP_CHECK_IF(acceptOriginShape.GetDimNum() != DIM_1,
        OP_LOGE(context_->GetNodeName(),
            "NumAcceptedToken dimension number must be 1, but got %lu",
            acceptOriginShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    // Validate shape matches batch size
    int64_t acceptLen = acceptOriginShape.GetDim(DIM_0);
    OP_CHECK_IF(acceptLen != batchSize_,
        OP_LOGE(context_->GetNodeName(),
            "NumAcceptedToken length must match batch size %ld, but got %ld",
            batchSize_,
            acceptLen),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate query start loc tensor shape
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateQueryStartLocShape()
{
    if (xInputMode_ == X_INPUT_3D) {
        return ge::GRAPH_SUCCESS;
    }
    // This is an optional input
    auto queryStartLocShape = context_->GetOptionalInputShape(QUERY_START_LOC_INDEX);
    OP_CHECK_IF(queryStartLocShape == nullptr,
        OP_LOGE(context_->GetNodeName(), "queryStartLoc must be provided when input X is 2D"),
        return ge::GRAPH_FAILED);

    auto queryStartLocOriginShape = queryStartLocShape->GetOriginShape();

    // Validate dimension number: must be 1
    OP_CHECK_IF(queryStartLocOriginShape.GetDimNum() != DIM_1,
        OP_LOGE(context_->GetNodeName(),
            "QueryStartLoc dimension number must be 1, but got %lu",
            queryStartLocOriginShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    // Validate shape: should be (batch + 1,)
    int64_t queryStartLocLen = queryStartLocOriginShape.GetDim(DIM_0);
    OP_CHECK_IF(queryStartLocLen != batchSize_ + 1,
        OP_LOGE(context_->GetNodeName(),
            "QueryStartLoc length must be batch_size + 1 = %ld, but got %ld",
            batchSize_ + 1,
            queryStartLocLen),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate X tensor type
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateXType()
{
    OP_CHECK_IF(xDtype_ != ge::DataType::DT_FLOAT16 && xDtype_ != ge::DataType::DT_BF16,
        OP_LOGE(context_->GetNodeName(),
            "X data type must be FLOAT16 or BFLOAT16, but got %s",
            Ops::Base::ToString(xDtype_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate weight tensor type
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateWeightType()
{
    OP_CHECK_IF(weightDtype_ != xDtype_,
        OP_LOGE(context_->GetNodeName(),
            "Weight data type must match X data type %s, but got %s",
            Ops::Base::ToString(xDtype_).c_str(),
            Ops::Base::ToString(weightDtype_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate conv states tensor type
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateConvStatesType()
{
    OP_CHECK_IF(convStatesDtype_ != xDtype_,
        OP_LOGE(context_->GetNodeName(),
            "ConvStates data type must match X data type %s, but got %s",
            Ops::Base::ToString(xDtype_).c_str(),
            Ops::Base::ToString(convStatesDtype_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate cache indices tensor type
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateCacheIndicesType()
{
    // This is an optional input
    auto cacheIndicesDesc = context_->GetOptionalInputDesc(CACHE_INDICES_INDEX);
    if (cacheIndicesDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(cacheIndicesDtype_ != ge::DataType::DT_INT32,
        OP_LOGE(context_->GetNodeName(),
            "CacheIndices data type must be INT32, but got %s",
            Ops::Base::ToString(cacheIndicesDtype_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate query start loc tensor type
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateQueryStartLocType()
{
    if (xInputMode_ == X_INPUT_3D) {
        return ge::GRAPH_SUCCESS;
    }
    // This is an optional input
    auto queryStartLocDesc = context_->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
    OP_CHECK_IF(queryStartLocDesc == nullptr,
        OP_LOGE(context_->GetNodeName(), "queryStartLoc must be provided when input X is 2D"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(queryStartLocDtype_ != ge::DataType::DT_INT32,
        OP_LOGE(context_->GetNodeName(),
            "QueryStartLoc data type must be INT32, but got %s",
            Ops::Base::ToString(queryStartLocDtype_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate num accepted tokens tensor type
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateNumAcceptedTokenType()
{
    // This is an optional input
    auto numAcceptedTokenDesc = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKEN_INDEX);
    if (numAcceptedTokenDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(numAcceptedTokenDtype_ != ge::DataType::DT_INT32,
        OP_LOGE(context_->GetNodeName(),
            "NumAcceptedToken data type must be INT32, but got %s",
            Ops::Base::ToString(numAcceptedTokenDtype_).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Overall input parameters validation
ge::graphStatus FusedCausalConv1dCutBHTiling::CheckInputParams()
{
    // Validate all shapes
    OP_CHECK_IF(ValidateXShape() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "X shape validation failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateWeightShape() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Weight shape validation failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateConvStatesShape() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "ConvStates shape validation failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateQueryStartLocShape() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "QueryStartLoc shape validation failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateCacheIndicesShape() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CacheIndices shape validation failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateNumAcceptedTokenShape() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "NumAcceptedToken shape validation failed"),
        return ge::GRAPH_FAILED);

    // Validate all types
    OP_CHECK_IF(ValidateXType() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "X type validation failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateWeightType() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "Weight type validation failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateConvStatesType() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "ConvStates type validation failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateQueryStartLocType() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "QueryStartLoc type validation failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateCacheIndicesType() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "CacheIndices type validation failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateNumAcceptedTokenType() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "NumAcceptedToken type validation failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::DoOpTiling()
{
    if (dim_ >= COARSE_GRAIN_THRESHOLD) {
        // Coarse-grain tiling for large dim (>= 4096)
        OP_CHECK_IF(ComputeInterCoreSplitCoarse() != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "ComputeInterCoreSplitCoarse failed"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(ComputeIntraCoreUbTilingCoarse() != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "ComputeIntraCoreUbTilingCoarse failed"),
            return ge::GRAPH_FAILED);
    } else {
        // Fine-grain tiling for small dim (< 4096)
        OP_CHECK_IF(ComputeInterCoreSplit() != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "ComputeInterCoreSplit failed"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(ComputeIntraCoreUbTiling() != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "ComputeIntraCoreUbTiling failed"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ComputeInterCoreSplit()
{
    // 2D tiling (non-uniform): first split dim by 128, then split batch to maximize cores
    int64_t N = dim_ / DIM_ALIGN_ELEMENT;
    if (N <= 0) {
        OP_LOGE(context_->GetNodeName(), "dim %ld is smaller than DIM_ALIGN_ELEMENT %ld", dim_, DIM_ALIGN_ELEMENT);
        return ge::GRAPH_FAILED;
    }

    // Compute max available cores considering data-size limit
    limitedCoreNum_ = CalculateLimitedCoreNum();
    int64_t maxCoresAvailable = std::min<int64_t>(totalCoreNum_, limitedCoreNum_);

    // Greedy search best (dimCores, bsCores), prioritize more dim splits
    int64_t bestDimCores = 1;
    int64_t bestBSCores = 1;
    int64_t bestUsed = 1;  // dc=1 initially
    for (int64_t dc = N; dc >= 1; --dc) {
        int64_t maxAllowedBSByCore = maxCoresAvailable / dc;
        if (maxAllowedBSByCore == 0)
            continue;
        int64_t actualBS = std::min<int64_t>(batchSize_, maxAllowedBSByCore);
        if (actualBS <= 0)
            continue;
        int64_t usedCores = dc * actualBS;
        if (usedCores > bestUsed || (usedCores == bestUsed && dc > bestDimCores)) {
            bestDimCores = dc;
            bestBSCores = actualBS;
            bestUsed = usedCores;
        }
        if (bestUsed == maxCoresAvailable) {
            break;  // fully utilized
        }
    }

    // Derive dim non-uniform parameters
    int64_t base = N / bestDimCores;
    int64_t remainder = N % bestDimCores;  // big core count

    dimCoreCnt_ = bestDimCores;
    if (remainder == 0) {
        // Even split on dim: treat as only main cores, no tail cores
        dimMainCoreCnt_ = bestDimCores;
        dimTailCoreCnt_ = 0;
        // Both main/tail lengths are the same tile size; keep tail length equal for UB calc
        mainCoredimLen_ = base * DIM_ALIGN_ELEMENT;
        tailCoredimLen_ = base * DIM_ALIGN_ELEMENT;
    } else {
        // Non-even split: first 'remainder' big cores take (base+1) tiles, rest take base tiles
        dimMainCoreCnt_ = remainder;
        dimTailCoreCnt_ = bestDimCores - remainder;
        mainCoredimLen_ = (base + 1) * DIM_ALIGN_ELEMENT;  // big core size
        tailCoredimLen_ = base * DIM_ALIGN_ELEMENT;        // small core size
    }

    // Derive batch non-uniform parameters
    batchCoreCnt_ = bestBSCores;
    int64_t bsBase = batchSize_ / batchCoreCnt_;
    int64_t bsRemainder = batchSize_ % batchCoreCnt_;
    if (bsRemainder == 0) {
        // Even split on batch: treat as only main cores, no tail cores
        batchMainCoreCnt_ = batchCoreCnt_;
        batchTailCoreCnt_ = 0;
        mainCoreBatchNum_ = bsBase;
        tailCoreBatchNum_ = bsBase;
    } else {
        // Non-even split: first 'bsRemainder' big cores take (bsBase+1) batches, rest take bsBase batches
        batchMainCoreCnt_ = bsRemainder;                  // 前remainder个核是大核
        batchTailCoreCnt_ = batchCoreCnt_ - bsRemainder;  // 其余是小核
        mainCoreBatchNum_ = bsBase + 1;                   // 大核批大小
        tailCoreBatchNum_ = bsBase;                       // 小核批大小
    }

    usedCoreNum_ = dimCoreCnt_ * batchCoreCnt_;

    return ge::GRAPH_SUCCESS;
}

void FusedCausalConv1dCutBHTiling::ComputeUbFor(int64_t coreDimElems, int64_t coreBS, int64_t availableUbSize,
    int64_t &outUbDim, int64_t &outUbBS, int64_t &outLoopDim, int64_t &outLoopBS, int64_t &outUbTailDim,
    int64_t &outUbTailBS)
{
    // 如果x为2维输入，seqLen_的值为m的最大值 + 1
    if (xInputMode_ == X_INPUT_2D) {
        seqLen_ = MAX_M + 1;
    }
    // int64_t weightConvStatesCoeffPerDim = (BUFFER_NUM * kernelSize_ + 2 * BUFFER_NUM * stateLen_ * 2) * DTYPE_SIZE;
    int64_t weightConvStatesCoeffPerDim = (BUFFER_NUM * kernelSize_ + BUFFER_NUM * (kernelSize_ - 1) * 2) * DTYPE_SIZE;
    int64_t xCoeffPerDimFullBS = BUFFER_NUM * coreBS * seqLen_ * DTYPE_SIZE * 2;
    int64_t totalCoeffPerDim = weightConvStatesCoeffPerDim + xCoeffPerDimFullBS;
    int64_t maxUbDim = (totalCoeffPerDim > 0) ? (availableUbSize / totalCoeffPerDim) : 0;
    maxUbDim = (maxUbDim / DIM_ALIGN_ELEMENT) * DIM_ALIGN_ELEMENT;
    if (maxUbDim >= DIM_ALIGN_ELEMENT) {
        outUbBS = coreBS;
        outUbDim = std::min(maxUbDim, coreDimElems);
        outUbDim = (outUbDim / DIM_ALIGN_ELEMENT) * DIM_ALIGN_ELEMENT;
        if (outUbDim == 0)
            outUbDim = DIM_ALIGN_ELEMENT;
    } else {
        outUbDim = DIM_ALIGN_ELEMENT;
        int64_t weightConvStatesSize = weightConvStatesCoeffPerDim * outUbDim;
        int64_t availableForX = availableUbSize - weightConvStatesSize;
        int64_t xSizePerBatch = BUFFER_NUM * seqLen_ * outUbDim * DTYPE_SIZE * 2;
        outUbBS = (xSizePerBatch > 0) ? (availableForX / xSizePerBatch) : 0;
        outUbBS = std::max<int64_t>(outUbBS, 1);
        outUbBS = std::min(outUbBS, coreBS);
    }
    outLoopDim = (coreDimElems + outUbDim - 1) / outUbDim;
    outUbTailDim = (outLoopDim == 1) ? outUbDim : (coreDimElems - (outLoopDim - 1) * outUbDim);
    outLoopBS = (coreBS + outUbBS - 1) / outUbBS;
    outUbTailBS = (outLoopBS == 1) ? outUbBS : (coreBS - (outLoopBS - 1) * outUbBS);
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ComputeIntraCoreUbTiling()
{
    // Intra-core UB tiling (big and tail blocks separately)
    int64_t cacheIndicesUBSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    int64_t numAcceptedTokensUBSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;

    fixedUBSize = cacheIndicesUBSize + numAcceptedTokensUBSize;
    if (xInputMode_ == X_INPUT_2D) {
        int64_t queryStartLocUBSize =
            ((batchSize_ + 1) * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        fixedUBSize += queryStartLocUBSize;
    }

    int64_t availableUbSize = static_cast<int64_t>(ubSize_ - SYSTEM_RESERVED_UB_SIZE) - fixedUBSize;

    // Big cores UB params
    ComputeUbFor(mainCoredimLen_,
        mainCoreBatchNum_,
        availableUbSize,
        ubMainFactorDim_,
        ubMainFactorBS_,
        loopNumDim_,
        loopNumBS_,
        ubTailFactorDim_,
        ubTailFactorBS_);

    // Tail cores UB params
    if (dimTailCoreCnt_ > 0 || batchTailCoreCnt_ > 0) {
        int64_t tailCoreDim = (dimMainCoreCnt_ > 0 ? tailCoredimLen_ : mainCoredimLen_);
        ComputeUbFor(tailCoreDim,
            tailCoreBatchNum_,
            availableUbSize,
            tailBlockubFactorDim_,
            tailBlockubFactorBS_,
            tailBlockloopNumDim_,
            tailBlockloopNumBS_,
            tailBlockubTailFactorDim_,
            tailBlockubTailFactorBS_);
    } else {
        // No tail cores: mirror main-core UB params to keep values consistent
        tailBlockubFactorDim_ = ubMainFactorDim_;
        tailBlockubFactorBS_ = ubMainFactorBS_;
        tailBlockloopNumDim_ = loopNumDim_;
        tailBlockloopNumBS_ = loopNumBS_;
        tailBlockubTailFactorDim_ = ubTailFactorDim_;
        tailBlockubTailFactorBS_ = ubTailFactorBS_;
    }

    return ge::GRAPH_SUCCESS;
}

int64_t FusedCausalConv1dCutBHTiling::CalculateLimitedCoreNum()
{
    // Calculate input x size in bytes (only consider batch * dim_)
    int64_t xSizeBytes;
    xSizeBytes = batchSize_ * dim_ * DTYPE_SIZE;

    // Limit core number based on data size
    // increase 1 core per additional 256 bytes of x data
    // effectiveCoreNum = ceil(xSizeBytes / 256)
    int64_t effectiveCoreNum = (xSizeBytes + DIM_ALIGN_SIZE - 1) / DIM_ALIGN_SIZE;
    effectiveCoreNum = std::max(effectiveCoreNum, static_cast<int64_t>(1));

    // Actual core number is min(effectiveCoreNum, totalCoreNum_)
    return std::min(effectiveCoreNum, static_cast<int64_t>(totalCoreNum_));
}

int64_t FusedCausalConv1dCutBHTiling::CalculateLimitedCoreNumCoarse()
{
    // Calculate input x size in bytes (only consider batch * dim_)
    int64_t xSizeBytes = batchSize_ * dim_ * DTYPE_SIZE;

    // Limit core number based on data size for coarse-grain tiling
    // increase 1 core per additional 2048 bytes (1024 elements) of x data
    // effectiveCoreNum = ceil(xSizeBytes / 2048)
    int64_t effectiveCoreNum = (xSizeBytes + COARSE_GRAIN_SIZE - 1) / COARSE_GRAIN_SIZE;
    effectiveCoreNum = std::max(effectiveCoreNum, static_cast<int64_t>(1));

    // Actual core number is min(effectiveCoreNum, totalCoreNum_)
    return std::min(effectiveCoreNum, static_cast<int64_t>(totalCoreNum_));
}

uint64_t FusedCausalConv1dCutBHTiling::GetTilingKey() const
{
    if (xDtype_ == ge::DataType::DT_BF16) {
        return TILING_KEY_BH_BF16;
    } else if (xDtype_ == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_BH_FP16;
    }
}

ge::graphStatus FusedCausalConv1dCutBHTiling::PostTiling()
{
    // Set block dimension (number of cores to use)
    context_->SetBlockDim(usedCoreNum_);

    // Populate tiling data - core distribution
    tilingData_.usedCoreNum = usedCoreNum_;
    tilingData_.dimCoreCnt = dimCoreCnt_;
    tilingData_.batchCoreCnt = batchCoreCnt_;

    // Dim tiling parameters
    tilingData_.dimMainCoreCnt = dimMainCoreCnt_;
    tilingData_.dimTailCoreCnt = dimTailCoreCnt_;
    tilingData_.mainCoredimLen = mainCoredimLen_;
    tilingData_.tailCoredimLen = tailCoredimLen_;

    // Batch tiling parameters
    tilingData_.batchMainCoreCnt = batchMainCoreCnt_;
    tilingData_.batchTailCoreCnt = batchTailCoreCnt_;
    tilingData_.mainCoreBatchNum = mainCoreBatchNum_;
    tilingData_.tailCoreBatchNum = tailCoreBatchNum_;

    // Intra-core tiling parameters (UB loop, big/tail blocks)
    tilingData_.loopNumBS = loopNumBS_;
    tilingData_.loopNumDim = loopNumDim_;
    tilingData_.ubMainFactorBS = ubMainFactorBS_;
    tilingData_.ubTailFactorBS = ubTailFactorBS_;
    tilingData_.ubMainFactorDim = ubMainFactorDim_;
    tilingData_.ubTailFactorDim = ubTailFactorDim_;
    tilingData_.tailBlockloopNumBS = tailBlockloopNumBS_;
    tilingData_.tailBlockloopNumDim = tailBlockloopNumDim_;
    tilingData_.tailBlockubFactorBS = tailBlockubFactorBS_;
    tilingData_.tailBlockubTailFactorBS = tailBlockubTailFactorBS_;
    tilingData_.tailBlockubFactorDim = tailBlockubFactorDim_;
    tilingData_.tailBlockubTailFactorDim = tailBlockubTailFactorDim_;

    // Shape information for kernel use
    tilingData_.batchSize = batchSize_;
    tilingData_.seqLen = seqLen_;
    tilingData_.cuSeqLen = cuSeqLen_;
    tilingData_.dim = dim_;
    tilingData_.kernelSize = kernelSize_;
    tilingData_.stateLen = stateLen_;
    tilingData_.xStride = xStride_;
    tilingData_.cacheStride0 = cacheStride0_;
    tilingData_.cacheStride1 = cacheStride1_;
    tilingData_.padSlotId = padSlotId_;
    tilingData_.xInputMode = xInputMode_;
    tilingData_.hasAcceptTokenNum = hasAcceptTokenNum_;
    tilingData_.residualConnection = residualConnection_;

    // Save tiling data to buffer
    auto tilingDataSize = sizeof(FusedCausalConv1dCutBHTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(),
        context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void *>(&tilingData_),
        tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

void FusedCausalConv1dCutBHTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "=== FusedCausalConv1dCutBH DumpTilingInfo ===");

    // Core distribution parameters
    OP_LOGI(context_->GetNodeName(), "usedCoreNum: %ld", tilingData_.usedCoreNum);
    OP_LOGI(context_->GetNodeName(), "dimCoreCnt: %ld", tilingData_.dimCoreCnt);
    OP_LOGI(context_->GetNodeName(), "batchCoreCnt: %ld", tilingData_.batchCoreCnt);

    // Dim tiling parameters
    OP_LOGI(context_->GetNodeName(), "dimMainCoreCnt: %ld", tilingData_.dimMainCoreCnt);
    OP_LOGI(context_->GetNodeName(), "dimTailCoreCnt: %ld", tilingData_.dimTailCoreCnt);
    OP_LOGI(context_->GetNodeName(), "mainCoredimLen: %ld", tilingData_.mainCoredimLen);
    OP_LOGI(context_->GetNodeName(), "tailCoredimLen: %ld", tilingData_.tailCoredimLen);

    // Batch tiling parameters
    OP_LOGI(context_->GetNodeName(), "batchMainCoreCnt: %ld", tilingData_.batchMainCoreCnt);
    OP_LOGI(context_->GetNodeName(), "batchTailCoreCnt: %ld", tilingData_.batchTailCoreCnt);
    OP_LOGI(context_->GetNodeName(), "mainCoreBatchNum: %ld", tilingData_.mainCoreBatchNum);
    OP_LOGI(context_->GetNodeName(), "tailCoreBatchNum: %ld", tilingData_.tailCoreBatchNum);

    // Intra-core tiling parameters (UB loop, big/tail blocks)
    OP_LOGI(context_->GetNodeName(), "loopNumBS: %ld", tilingData_.loopNumBS);
    OP_LOGI(context_->GetNodeName(), "loopNumDim: %ld", tilingData_.loopNumDim);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorBS: %ld", tilingData_.ubMainFactorBS);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorBS: %ld", tilingData_.ubTailFactorBS);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorDim: %ld", tilingData_.ubMainFactorDim);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorDim: %ld", tilingData_.ubTailFactorDim);
    OP_LOGI(context_->GetNodeName(), "tailBlockloopNumBS: %ld", tilingData_.tailBlockloopNumBS);
    OP_LOGI(context_->GetNodeName(), "tailBlockloopNumDim: %ld", tilingData_.tailBlockloopNumDim);
    OP_LOGI(context_->GetNodeName(), "tailBlockubFactorBS: %ld", tilingData_.tailBlockubFactorBS);
    OP_LOGI(context_->GetNodeName(), "tailBlockubTailFactorBS: %ld", tilingData_.tailBlockubTailFactorBS);
    OP_LOGI(context_->GetNodeName(), "tailBlockubFactorDim: %ld", tilingData_.tailBlockubFactorDim);
    OP_LOGI(context_->GetNodeName(), "tailBlockubTailFactorDim: %ld", tilingData_.tailBlockubTailFactorDim);

    // Shape information for kernel use
    OP_LOGI(context_->GetNodeName(), "batchSize: %ld", tilingData_.batchSize);
    OP_LOGI(context_->GetNodeName(), "seqLen: %ld", tilingData_.seqLen);
    OP_LOGI(context_->GetNodeName(), "cuSeqLen: %ld", tilingData_.cuSeqLen);
    OP_LOGI(context_->GetNodeName(), "dim: %ld", tilingData_.dim);
    OP_LOGI(context_->GetNodeName(), "kernelSize: %ld", tilingData_.kernelSize);
    OP_LOGI(context_->GetNodeName(), "stateLen: %ld", tilingData_.stateLen);
    OP_LOGI(context_->GetNodeName(), "xStride: %ld", tilingData_.xStride);
    OP_LOGI(context_->GetNodeName(), "cacheStride0: %ld", tilingData_.cacheStride0);
    OP_LOGI(context_->GetNodeName(), "cacheStride1: %ld", tilingData_.cacheStride1);
    OP_LOGI(context_->GetNodeName(), "padSlotId: %ld", tilingData_.padSlotId);
    OP_LOGI(context_->GetNodeName(), "xInputMode: %ld", tilingData_.xInputMode);
    OP_LOGI(context_->GetNodeName(), "hasAcceptTokenNum: %ld", tilingData_.hasAcceptTokenNum);
    OP_LOGI(context_->GetNodeName(), "residualConnection: %ld", tilingData_.residualConnection);
}

ge::graphStatus FusedCausalConv1dCutBHTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ComputeInterCoreSplitCoarse()
{
    // Coarse-grain tiling: split dim by 1024 elements, remainder by 128 elements
    int64_t coarseBlocks = dim_ / COARSE_GRAIN_ELEMENT;  // Number of 1024-element blocks
    int64_t remainder = dim_ % COARSE_GRAIN_ELEMENT;     // Remainder elements
    int64_t fineBlocks = remainder / DIM_ALIGN_ELEMENT;  // Number of 128-element blocks from remainder

    // Compute max available cores considering data-size limit (coarse-grain version)
    limitedCoreNum_ = CalculateLimitedCoreNumCoarse();
    int64_t maxCoresAvailable = std::min<int64_t>(totalCoreNum_, limitedCoreNum_);

    // Dim direction: coarseBlocks cores, first fineBlocks cores get extra 128 elements
    dimCoreCnt_ = std::min(coarseBlocks, maxCoresAvailable);
    if (dimCoreCnt_ <= 0) {
        OP_LOGE(
            context_->GetNodeName(), "dim %ld is smaller than COARSE_GRAIN_ELEMENT %ld", dim_, COARSE_GRAIN_ELEMENT);
        return ge::GRAPH_FAILED;
    }

    // Distribute remainder 128-blocks evenly to all cores
    int64_t fineBase = fineBlocks / dimCoreCnt_;       // Base 128-blocks per core
    int64_t fineRemainder = fineBlocks % dimCoreCnt_;  // Extra 128-blocks for first fineRemainder cores

    if (fineRemainder == 0) {
        // Even distribution: all cores get same number of 128-blocks
        dimMainCoreCnt_ = dimCoreCnt_;
        dimTailCoreCnt_ = 0;
        mainCoredimLen_ = COARSE_GRAIN_ELEMENT + fineBase * DIM_ALIGN_ELEMENT;
        tailCoredimLen_ = mainCoredimLen_;  // Same as main
    } else {
        // Non-even distribution: first fineRemainder cores get (fineBase+1) 128-blocks
        dimMainCoreCnt_ = fineRemainder;
        dimTailCoreCnt_ = dimCoreCnt_ - fineRemainder;
        mainCoredimLen_ = COARSE_GRAIN_ELEMENT + (fineBase + 1) * DIM_ALIGN_ELEMENT;
        tailCoredimLen_ = COARSE_GRAIN_ELEMENT + fineBase * DIM_ALIGN_ELEMENT;
    }

    // Batch direction: use remaining cores
    int64_t remainingCores = maxCoresAvailable / dimCoreCnt_;
    batchCoreCnt_ = std::min(batchSize_, remainingCores);
    batchCoreCnt_ = std::max(batchCoreCnt_, static_cast<int64_t>(1));

    // Batch non-uniform split
    int64_t bsBase = batchSize_ / batchCoreCnt_;
    int64_t bsRemainder = batchSize_ % batchCoreCnt_;
    if (bsRemainder == 0) {
        batchMainCoreCnt_ = batchCoreCnt_;
        batchTailCoreCnt_ = 0;
        mainCoreBatchNum_ = bsBase;
        tailCoreBatchNum_ = bsBase;
    } else {
        batchMainCoreCnt_ = bsRemainder;
        batchTailCoreCnt_ = batchCoreCnt_ - bsRemainder;
        mainCoreBatchNum_ = bsBase + 1;
        tailCoreBatchNum_ = bsBase;
    }

    usedCoreNum_ = dimCoreCnt_ * batchCoreCnt_;

    return ge::GRAPH_SUCCESS;
}

void FusedCausalConv1dCutBHTiling::ComputeUbForCoarse(int64_t coreDimElems, int64_t coreBS, int64_t availableUbSize,
    int64_t &outUbDim, int64_t &outUbBS, int64_t &outLoopDim, int64_t &outLoopBS, int64_t &outUbTailDim,
    int64_t &outUbTailBS)
{
    // 如果x为2维输入，seqLen_的值为m的最大值 + 1
    if (xInputMode_ == X_INPUT_2D) {
        seqLen_ = MAX_M + 1;
    }
    // int64_t weightConvStatesCoeffPerDim = (BUFFER_NUM * kernelSize_ + 2 * BUFFER_NUM * stateLen_ * 2) * DTYPE_SIZE;
    int64_t weightConvStatesCoeffPerDim = (BUFFER_NUM * kernelSize_ + BUFFER_NUM * (kernelSize_ - 1) * 2) * DTYPE_SIZE;
    int64_t xCoeffPerDimFullBS = BUFFER_NUM * coreBS * seqLen_ * DTYPE_SIZE * 2;
    int64_t totalCoeffPerDim = weightConvStatesCoeffPerDim + xCoeffPerDimFullBS;
    int64_t maxUbDim = (totalCoeffPerDim > 0) ? (availableUbSize / totalCoeffPerDim) : 0;
    maxUbDim = (maxUbDim / COARSE_GRAIN_ELEMENT) * COARSE_GRAIN_ELEMENT;
    if (maxUbDim >= COARSE_GRAIN_ELEMENT) {
        outUbBS = coreBS;
        outUbDim = std::min(maxUbDim, coreDimElems);
        outUbDim = (outUbDim / COARSE_GRAIN_ELEMENT) * COARSE_GRAIN_ELEMENT;
        if (outUbDim == 0)
            outUbDim = COARSE_GRAIN_ELEMENT;
    } else {
        outUbDim = COARSE_GRAIN_ELEMENT;
        int64_t weightConvStatesSize = weightConvStatesCoeffPerDim * outUbDim;
        int64_t availableForX = availableUbSize - weightConvStatesSize;
        int64_t xSizePerBatch = BUFFER_NUM * seqLen_ * outUbDim * DTYPE_SIZE * 2;
        outUbBS = (xSizePerBatch > 0) ? (availableForX / xSizePerBatch) : 0;
        outUbBS = std::max<int64_t>(outUbBS, 1);
        outUbBS = std::min(outUbBS, coreBS);
    }
    outLoopDim = (coreDimElems + outUbDim - 1) / outUbDim;
    outUbTailDim = (outLoopDim == 1) ? outUbDim : (coreDimElems - (outLoopDim - 1) * outUbDim);
    outLoopBS = (coreBS + outUbBS - 1) / outUbBS;
    outUbTailBS = (outLoopBS == 1) ? outUbBS : (coreBS - (outLoopBS - 1) * outUbBS);
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ComputeIntraCoreUbTilingCoarse()
{
    // Intra-core UB tiling for coarse-grain mode
    int64_t cacheIndicesUBSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    int64_t numAcceptedTokensUBSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;

    fixedUBSize = cacheIndicesUBSize + numAcceptedTokensUBSize;
    if (xInputMode_ == X_INPUT_2D) {
        int64_t queryStartLocUBSize =
            ((batchSize_ + 1) * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        fixedUBSize += queryStartLocUBSize;
    }

    int64_t availableUbSize = static_cast<int64_t>(ubSize_ - SYSTEM_RESERVED_UB_SIZE) - fixedUBSize;

    // Main cores UB params (handle 1152 or 1024 elements depending on remainder)
    ComputeUbForCoarse(mainCoredimLen_,
        mainCoreBatchNum_,
        availableUbSize,
        ubMainFactorDim_,
        ubMainFactorBS_,
        loopNumDim_,
        loopNumBS_,
        ubTailFactorDim_,
        ubTailFactorBS_);

    // Tail cores UB params
    if (dimTailCoreCnt_ > 0 || batchTailCoreCnt_ > 0) {
        int64_t tailCoreDim = (dimMainCoreCnt_ > 0) ? tailCoredimLen_ : mainCoredimLen_;
        ComputeUbForCoarse(tailCoreDim,
            tailCoreBatchNum_,
            availableUbSize,
            tailBlockubFactorDim_,
            tailBlockubFactorBS_,
            tailBlockloopNumDim_,
            tailBlockloopNumBS_,
            tailBlockubTailFactorDim_,
            tailBlockubTailFactorBS_);
    } else {
        // No tail cores: mirror main-core UB params
        tailBlockubFactorDim_ = ubMainFactorDim_;
        tailBlockubFactorBS_ = ubMainFactorBS_;
        tailBlockloopNumDim_ = loopNumDim_;
        tailBlockloopNumBS_ = loopNumBS_;
        tailBlockubTailFactorDim_ = ubTailFactorDim_;
        tailBlockubTailFactorBS_ = ubTailFactorBS_;
    }

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(FusedCausalConv1dCutBH, FusedCausalConv1dCutBHTiling, 1);

}  // namespace optiling
