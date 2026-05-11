/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_backward_tiling.cpp
 * \brief
 */
#include <cmath>
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_base.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "moe_gating_top_k_backward_tiling.h"

namespace optiling {
// Attribute constants
const static int64_t RENORM_TYPE_ZERO = 0; // Re-norm before top K
const static int64_t NORM_TYPE_SOFTMAX = 0;
const static int64_t NORM_TYPE_SIGMOID = 1;

// Tensor dimensions
const static size_t X_NORM_INPUT_DIMS = 2;     // [m,n]
const static size_t GRAD_Y_INPUT_DIMS = 2;     // [m,k]
const static size_t EXPERT_IDX_INPUT_DIMS = 2; // [m,k]
const static size_t GRAD_X_OUTPUT_DIMS = 2;    // [m,n]

// Indices
const static int64_t X_NORM_INPUT_INDEX = 0;
const static int64_t GRAD_Y_INPUT_INDEX = 1;
const static int64_t EXPERT_IDX_INPUT_INDEX = 2;
const static int64_t GRAD_X_OUTPUT_INDEX = 0;
const static int64_t RENORM_ATTR_INDEX = 0;
const static int64_t NORM_TYPE_ATTR_INDEX = 1;
const static int64_t ROUTED_SCALING_FACTOR_ATTR_INDEX = 2;
const static int64_t EPS_ATTR_INDEX = 3;
const static int64_t M_DIM_INDEX = 0;
const static int64_t N_DIM_INDEX = 1;
const static int64_t K_DIM_INDEX = 1;

// Tiling keys
const static uint64_t TILING_KEY_RENORM_SIGMOID = 1;

// Data type sizes
const static size_t FLOAT32_SIZE_T = 4;
const static size_t FLOAT16_SIZE_T = 2;
const static size_t BFLOAT16_SIZE_T = 2;

// Miscellaneous
const static int64_t MAX_EXPERT_COUNT = 2048;           // max value for n
const static int64_t MIN_EXPERT_COUNT = 2;              // min value for n
const static int64_t MIN_K = 1;                         // min value for k
const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216; // 预留16M空间


class MoeGatingTopKBackwardTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeGatingTopKBackwardTilingBase(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeGatingTopKBackwardTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    void SplitRows();

    // Shapes
    const gert::Shape *xNormShape_ = nullptr;
    const gert::Shape *gradYShape_ = nullptr;
    const gert::Shape *expertIdxShape_ = nullptr;
    const gert::Shape *outputGradXShape_ = nullptr;

    int64_t inputDtypeSize_;
    const char *opName_ = "";
    MoeGatingTopKBackwardTilingData MoeGatingTopKBackwardTilingData_;

    // Tiling data parameters
    int64_t needCoreNum_ = 0;
    int64_t perCoreRows_ = 0;
    int64_t lastCoreRows_ = 0;
    int64_t baseRows_ = 0;
    int64_t perLoopTimes_ = 0;
    int64_t perTailRows_ = 0;
    int64_t lastLoopTimes_ = 0;
    int64_t lastTailRows_ = 0;
    int64_t tokenCount_ = 0;
    int64_t expertCount_ = 0;
    int64_t k_ = 0;
    int64_t gradYDtypeSize_ = 0;
    int64_t renorm_ = RENORM_TYPE_ZERO;
    int64_t normType_ = NORM_TYPE_SOFTMAX;
    int64_t routedScalingFactor_ = 1.0f;
    int64_t eps_ = 1e-20f;
};

// todo: add opName_ for OP_CHECK_IF and OP_LOGE
ge::graphStatus MoeGatingTopKBackwardTilingBase::CheckInputShape()   // fqy
{
    // Check x_norm dim num
    size_t xNormDimNum = xNormShape_->GetDimNum();
    OP_CHECK_IF(
        xNormDimNum != X_NORM_INPUT_DIMS,
        OP_LOGE(context_, "The dim number of x_norm is: %zu, but should be %zu.", xNormDimNum, X_NORM_INPUT_DIMS),
        return ge::GRAPH_FAILED);

    // Parse m and n from input x_norm
    tokenCount_ = xNormShape_->GetDim(M_DIM_INDEX);
    expertCount_ = xShape_->GetDim(N_DIM_INDEX);
    MoeGatingTopKBackwardTilingData_.set_tokenCount(tokenCount_);
    MoeGatingTopKBackwardTilingData_.set_expertCount(expertCount_);

    // Check grad_y dim num
    size_t gradYDimNum = gradYShape_->GetDimNum();
    OP_CHECK_IF(
        gradYDimNum != GRAD_Y_INPUT_DIMS,
        OP_LOGE(context_, "The dim number of grad_y is: %zu, but should be %zu.", gradYDimNum, GRAD_Y_INPUT_DIMS),
        return ge::GRAPH_FAILED);

    // Check if grad_y matches x_norm in dim 0
    int64_t gradYTokenCount = gradYShape_->GetDim(M_DIM_INDEX);
    OP_CHECK_IF(
        gradYTokenCount != tokenCount_,
        OP_LOGE(context_, "Dim 0 of grad_y is: %zu, but should be %zu.", gradYTokenCount, tokenCount_),
        return ge::GRAPH_FAILED);

    // Parse k from input grad_y
    k_ = gradYShape_->GetDim(K_DIM_INDEX);
    MoeGatingTopKBackwardTilingData_.set_k(k_);

    // Check if k is less than or equal to n
    OP_CHECK_IF(
        k_ > expertCount_,
        OP_LOGE(context_, "k is: %ld, expert num is: %ld, k cannot be greater than expert num.", k_, expertCount_),
        return ge::GRAPH_FAILED);

    // Check if expert_idx has the same shape as grad_y
    int64_t expertIdxTokenCount = expertIdxShape_->GetDim(M_DIM_INDEX);
    int64_t expertIdxK= expertIdxShape_->GetDim(K_DIM_INDEX);
    OP_CHECK_IF(
        expertIdxTokenCount != tokenCount_,
        OP_LOGE(context_, "Dim 0 of expertIdx is: %zu, but should be %zu.", gradYTokenCount, tokenCount_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        expertIdxK != k_,
        OP_LOGE(context_, "Dim 1 of expertIdx is: %zu, but should be %zu.", expertIdxK, k_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::CheckAttr()   // lc
{
    OP_CHECK_IF(
        expertCount_ > MAX_EXPERT_COUNT,
        OP_LOGE(context_, "expert count is: %ld, but should not be greater than %ld.", expertCount_, MAX_EXPERT_COUNT),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        expertCount_ < MIN_EXPERT_COUNT,
        OP_LOGE(context_, "expert count is: %ld, but should not be less than %ld.", expertCount_, MIN_EXPERT_COUNT),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        k_ < MIN_K,
        OP_LOGE(context_, "k is: %ld, but should not be less than %ld.", k_, MIN_K), 
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        k_ > expertCount_,
        OP_LOGE(context_, "k is: %ld, but should not be greater than %ld.", k_, expertCount_), 
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        renorm_ != RENORM_TYPE_ZERO,
        OP_LOGE(context_, "renorm is: %ld, but currently only %ld is supported.", renorm_, RENORM_TYPE_ZERO),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        normType_ != NORM_TYPE_SIGMOID,
        OP_LOGE(context_, "norm type is: %ld, but currently only %ld is supported.", normType_, NORM_TYPE_SIGMOID),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::CheckDataType()
{
    // Parse input shape info
    auto xNormShapePtr = context_->GetInputShape(X_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xNormShapePtr);
    xNormShape_ = &xNormShapePtr->GetOriginShape();

    auto gradYShapePtr = context_->GetInputShape(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYShapePtr);
    gradYShape_ = &gradYShapePtr->GetOriginShape();

    auto expertIdxShapePtr = context_->GetInputShape(EXPERT_IDX_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr);
    expertIdxShape_ = &expertIdxShapePtr->GetOriginShape();

    // Parse output shape info
    auto gradXShapePtr = context_->GetOutputShape(GRAD_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXShapePtr);
    gradXShape_ = &gradXShapePtr->GetStorageShape();

    // Check data types
    auto x_norm = context_->GetInputDesc(X_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x_norm);
    auto xNormDtype = x_norm->GetDataType();
    OP_CHECK_IF(
        xNormDtype != ge::DataType::DT_FLOAT,
        OP_LOGE(context_, "x_norm dtype is %s, but only float32 is supported.",
            ge::TypeUtils::DataTypeToSerialString(xNormDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto grad_y = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, grad_y);
    auto gradYDtype = grad_y->GetDataType();
    OP_CHECK_IF(
        (gradYDtype != ge::DataType::DT_FLOAT &&
            gradYDtype != ge::DataType::DT_FLOAT16 &&
            gradYDtype != ge::DataType::DT_BF16),
        OP_LOGE(context_, "grad_y dtype is %s, but only supports float32, half and bf16 are supported.",
            ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    // todo: fill the rest
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::GetPlatformInfo()   // fqy
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == null) {

    } else {

    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::CheckOutShape()   // lc
{
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKBackwardTilingBase::SplitRows()   // lc
{

}

ge::graphStatus MoeGatingTopKBackwardTilingBase::DoOpTiling()
{
    auto ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    SplitRows();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKBackwardTilingBase::GetTilingKey() const
{
    return TILING_KEY_RENORM_SIGMOID;
}

void MoeGatingTopKBackwardTilingBase::Reset()
{
    opName_ = nullptr;
    return;
}

void MoeGatingTopKBackwardTilingBase::DumpTilingInfo()   // fqy
{
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKBackward, MoeGatingTopKBackwardTilingBase, 2000);
} // namespace optiling
