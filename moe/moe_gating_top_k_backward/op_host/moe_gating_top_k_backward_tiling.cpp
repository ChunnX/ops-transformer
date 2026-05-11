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
const static int64_t NUM_BYTES_FOUR = 4;
const static int64_t NUM_BYTES_TWO = 2;

// Tiling keys
const static uint64_t TILING_KEY_RENORM_SIGMOID = 1;

// Data type sizes
const static size_t SIZE_OF_FLOAT32 = 4;
const static size_t FLOAT16_SIZE_T = 2;
const static size_t BFLOAT16_SIZE_T = 2;

// Miscellaneous
const static int64_t MAX_EXPERT_COUNT = 2048;           // max value for n
const static int64_t MIN_EXPERT_COUNT = 2;              // min value for n
const static int64_t MIN_K = 1;                         // min value for k
const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216; // 预留16M空间
const static int64_t DOUBLE_BUFFER_NUM = 2;
const static int64_t UB_ALIGN_BYTES = 31;


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
    ge::graphStatus CheckXNorm();
    ge::graphStatus CheckGradY();
    ge::graphStatus CheckExpertIdx();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    int64_t CalcMaxRows();
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
    float routedScalingFactor_ = 1.0f;
    float eps_ = 1e-20f;
};

inline int64_t AlignBytes_(int64_t x) {
    return (x + UB_ALIGN_BYTES) & ~UB_ALIGN_BYTES;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::CheckInputShape()   // fqy
{
    OP_CHECK_IF(CheckXNorm() != ge::GRAPH_SUCCESS,OP_LOGE(opName_, "Check XNorm failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGradY() != ge::GRAPH_SUCCESS,OP_LOGE(opName_, "Check GradY failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckExpertIdx() != ge::GRAPH_SUCCESS,OP_LOGE(opName_, "Check ExpertIdx failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKBackwardTilingBase::CheckXNorm()
{
    const gert::StorageShape* xNormShapePtr = context_->GetInputShape(X_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xNormShapePtr);
    auto xNormDimSize = xNormShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(xNormDimSize != X_NORM_INPUT_DIMS, OP_LOGE(opName_, " xNorm : the number of dimensions must be 2, but got %ld.", xNormDimSize),
        return ge::GRAPH_FAILED);
    
    tokenCount_ = xNormShapePtr->GetOriginShape().GetDim(DIM_ZERO);
    expertCount_ = xNormShapePtr->GetOriginShape().GetDim(DIM_ONE);
    OP_CHECK_IF(expertCount_ > MAX_EXPERT_COUNT, OP_LOGE(opName_, " xNorm : The shape of xNorm is [M, N], where N must be less than or equal to 2048, but got %ld.", expertCount_),
        return ge::GRAPH_FAILED);
    
    auto xNormDesc = context_->GetInputDesc(X_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xNormDesc);
    auto xNormDtype = xNormDesc->GetDataType();
    OP_CHECK_IF(xNormDtype != ge::DataType::DT_FLOAT,OP_LOGE(opName_, "xNorm only support float32 now"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKBackwardTilingBase::CheckGradY()
{
    const gert::StorageShape* gradYShapePtr = context_->GetInputShape(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYShapePtr);
    auto gradYDimSize = gradYShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(gradYDimSize != GRAD_Y_INPUT_DIMS, OP_LOGE(opName_, "gradY : the number of dimensions must be 2, but got %ld.", gradYDimSize),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(gradYShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_, OP_LOGE(opName_, "gradY : The size of the first dimension of xNorm and gradY should be the same."),
        return ge::GRAPH_FAILED);
    k_ = gradYShapePtr->GetOriginShape().GetDim(DIM_ONE);
    OP_CHECK_IF(k_ > expertCount_ || k_ <= 0, OP_LOGE(opName_, "gradY : The size of the second dimension of gradY must be greater than 0 and less than or equal to the size of the second dimension of xNorm.."),
        return ge::GRAPH_FAILED);
    
    auto gradYDesc = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYDesc);
    auto gradYDtype = gradYDesc->GetDataType();
    OP_CHECK_IF(gradYDtype != ge::DataType::DT_FLOAT && gradYDtype != ge::DataType::DT_FLOAT16 && gradYDtype != ge::DataType::DT_BF16,
        OP_LOGE(opName_, "gradY only support float32, float16 and bfloat16 now."), return ge::GRAPH_FAILED);
    gradYDtypeSize_ = gradYDtype == ge::DataType::DT_FLOAT ? NUM_BYTES_FOUR : NUM_BYTES_TWO;
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKBackwardTilingBase::CheckExpertIdx()
{
    const gert::StorageShape* expertIdxShapePtr = context_->GetInputShape(EXPERT_IDX_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr);
    auto expertIdxDimSize = expertIdxShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(expertIdxDimSize != EXPERT_IDX_INPUT_DIMS, OP_LOGE(opName_, "expertIdx : the number of dimensions must be 2, but got %ld.", expertIdxDimSize),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(expertIdxShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_, OP_LOGE(opName_, "expertIdx : The size of the first dimension of xNorm and expertIdx should be the same."),
        return ge::GRAPH_FAILED);
    k_ = expertIdxShapePtr->GetOriginShape().GetDim(DIM_ONE);
    OP_CHECK_IF(expertIdxShapePtr->GetOriginShape().GetDim(DIM_ONE) != k_, OP_LOGE(opName_, "expertIdx : The size of the second dimension of xNorm and expertIdx should be the same."),
        return ge::GRAPH_FAILED);
    
    auto expertIdxDesc = context_->GetInputDesc(EXPERT_IDX_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxDesc);
    auto expertIdxDtype = expertIdxDesc->GetDataType();
    OP_CHECK_IF(expertIdxDtype != ge::DataType::DT_INT32, OP_LOGE(opName_, "expertIdx only support int32 now."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus MoeGatingTopKBackwardTilingBase::CheckAttr()   // lc
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *renormPtr = attrs->GetAttrPointer<int64_t>(RENORM_ATTR_INDEX);
    if (renormPtr != nullptr) {
        renorm_ = *renormPtr;
    }
    OP_LOGI(context_, "Attr renorm is: %ld.", renorm_);

    OP_CHECK_IF(
        renorm_ != RENORM_TYPE_ZERO,
        OP_LOGE(context_, "renorm is: %ld, but currently only %ld is supported.", renorm_, RENORM_TYPE_ZERO),
        return ge::GRAPH_FAILED);

    const int64_t *normTypePtr = attrs->GetAttrPointer<int64_t>(NORM_TYPE_ATTR_INDEX);
    if (normTypePtr != nullptr) {
        normType_ = *normTypePtr;
    }
    OP_LOGI(context_, "Attr normType is: %ld.", normType_);

    OP_CHECK_IF(
        normType_ != NORM_TYPE_SIGMOID,
        OP_LOGE(context_, "norm type is: %ld, but currently only %ld is supported.", normType_, NORM_TYPE_SIGMOID),
        return ge::GRAPH_FAILED);

    const float *routedScalingFactorPtr = attrs->GetAttrPointer<float>(ROUTED_SCALING_FACTOR_ATTR_INDEX);
    if (routedScalingFactorPtr != nullptr) {
        routedScalingFactor_ = *routedScalingFactorPtr;
    }
    OP_LOGI(context_, "Attr routedScalingFactor is: %f.", routedScalingFactor_);

    OP_CHECK_IF(
        routedScalingFactor_ > 0.0f,
        OP_LOGE(context_, "routedScalingFactor is: %f, but should be greater than 0.", routedScalingFactor_),
        return ge::GRAPH_FAILED);

    const float *epsPtr = attrs->GetAttrPointer<float>(EPS_ATTR_INDEX);
    if (epsPtr != nullptr) {
        eps_ = *epsPtr;
    }
    OP_LOGI(context_, "Attr eps is: %f.", eps_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::GetPlatformInfo()   // fqy
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<MoeGatingTopKBackwardCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(opName_, "compileInfoPtr is null."),
                return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->numBlocks;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        OP_CHECK_IF(aicoreParams_.numBlocks == 0, OP_LOGE(opName_, "coreNum is 0"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(aicoreParams_.ubSize == 0, OP_LOGE(opName_, "ubSize is 0"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingBase::CheckOutShape()   // lc
{
    auto gradXShapePtr = context_->GetOutputShape(GRAD_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXShapePtr);
    auto gradXDimSize = gradXShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(gradXDimSize != GRAD_Y_INPUT_DIMS,
                OP_LOGE(opName_, "gradX : the number of dimensions must be 2, but got %ld.", gradXDimSize),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(gradXShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_,
                OP_LOGE(opName_, "gradX : Shape xNorm and gradX should match in dimension 0."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(gradXShapePtr->GetOriginShape().GetDim(DIM_ONE) != expertCount_,
                OP_LOGE(opName_, "gradX : Shape xNorm and gradX should match in dimension 1."),
                return ge::GRAPH_FAILED);

    auto gradYDesc = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYDesc);
    auto gradYDtype = gradYDesc->GetDataType();

    auto gradXDesc = context_->GetOutputDesc(GRAD_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXDesc);
    auto gradXDtype = gradXDesc->GetDataType();

    OP_CHECK_IF(gradXDtype != gradYDtype, OP_LOGE(opName_, "gradX's data type must match that of gradY."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

int64_t MoeGatingTopKBackwardTilingBase::CalcMaxRows()
{
    int64_t gradYQuePerTokenSpace = DOUBLE_BUFFER_NUM * k_ * gradYDtypeSize_;
    int64_t indicesQuePerTokenSpace = DOUBLE_BUFFER_NUM * k_ * SIZE_OF_FLOAT32;
    int64_t xQuePerTokenSpace = DOUBLE_BUFFER_NUM * expertCount_ * SIZE_OF_FLOAT32;
    int64_t outQuePerTokenSpace = expertCount_ * gradYDtypeSize_;
    
    int64_t bufk0PerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufk1PerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufn2PerTokenSpace = expertCount_ * SIZE_OF_FLOAT32;
    int64_t bufn3PerTokenSpace = expertCount_ * SIZE_OF_FLOAT32;
    int64_t bufk4AddPerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufk4IndexPerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufk4RecipSumPerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufsPerTokenSpace = 1 * SIZE_OF_FLOAT32;

    int64_t ubReserve = 1 * 1024;
    int64_t shareTmpBufSpace = 0;

    int64_t quePerTokenSpace =
        gradYQuePerTokenSpace + indicesQuePerTokenSpace + xQuePerTokenSpace + outQuePerTokenSpace;
    int64_t bufPerTokenSpace =
        bufk0PerTokenSpace + bufk1PerTokenSpace + bufn2PerTokenSpace + bufn3PerTokenSpace + bufk4AddPerTokenSpace +
        bufk4IndexPerTokenSpace + bufk4RecipSumPerTokenSpace + bufsPerTokenSpace;
    int64_t maxRows = (aicoreParams_.ubSize - ubReserve - shareTmpBufSpace) / (quePerTokenSpace + bufPerTokenSpace);

    while (maxRows > 0) {
        int64_t queSpace =
            AlignBytes_(maxRows * gradYQuePerTokenSpace) + AlignBytes_(maxRows * indicesQuePerTokenSpace) +
            AlignBytes_(maxRows * xQuePerTokenSpace) + AlignBytes_(maxRows * outQuePerTokenSpace);
        int64_t bufPerTokenSpace =
            AlignBytes_(maxRows * bufk0PerTokenSpace) + AlignBytes_(maxRows * bufk1PerTokenSpace) +
            AlignBytes_(maxRows * bufn2PerTokenSpace) + AlignBytes_(maxRows * bufn3PerTokenSpace) +
            AlignBytes_(maxRows * bufk4AddPerTokenSpace) + AlignBytes_(maxRows * bufk4IndexPerTokenSpace) +
            AlignBytes_(maxRows * bufk4RecipSumPerTokenSpace) + AlignBytes_(maxRows * bufsPerTokenSpace);
        int64_t usedSpace = ubReserve + shareTmpBufSpace + queSpace + bufPerTokenSpace;

        if (usedSpace <= aicoreParams_.ubSize) {
            break;
        }
        maxRows--;
    }

    OP_CHECK_IF(maxRows > 0, OP_LOGE(context_,
        "maxRows should be greater than zero, current value is %ld. Check if N and K are within valid ranges.",
        maxRows), return ge::GRAPH_FAILED);
    
    return maxRows;
}

void MoeGatingTopKBackwardTilingBase::SplitRows()   // lc
{
    perCoreRows_ = Ops::Base::CeilDiv(tokenCount_, static_cast<int64_t>(aicoreParams_.numBlocks));
    needCoreNum_ = Ops::Base::CeilDiv(tokenCount_, perCoreRows_);
    lastCoreRows_ = tokenCount_ % perCoreRows_ == 0 ? perCoreRows_ : tokenCount_ % perCoreRows_;

    baseRows_ = CalcMaxRows();
    perLoopTimes_ = (perCoreRows_ + baseRows_ - 1) / baseRows_;
    perTailRows_ = perCoreRows_ - (perLoopTimes_ - 1) * baseRows_;
    lastLoopTimes_ = (lastCoreRows_ + baseRows_ - 1) / baseRows_;
    lastTailRows_ = lastCoreRows_ - (lastLoopTimes_ - 1) * baseRows_;
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
    MoeGatingTopKBackwardTilingData_.set_needCoreNum(needCoreNum_);
    MoeGatingTopKBackwardTilingData_.set_perCoreRows(perCoreRows_);
    MoeGatingTopKBackwardTilingData_.set_lastCoreRows(lastCoreRows_);
    MoeGatingTopKBackwardTilingData_.set_baseRows(baseRows_);
    MoeGatingTopKBackwardTilingData_.set_perLoopTimes(perLoopTimes_);
    MoeGatingTopKBackwardTilingData_.set_perTailRows(perTailRows_);
    MoeGatingTopKBackwardTilingData_.set_lastLoopTimes(lastLoopTimes_);
    MoeGatingTopKBackwardTilingData_.set_lastTailRows(lastTailRows_);
    MoeGatingTopKBackwardTilingData_.set_tokenCount(tokenCount_);
    MoeGatingTopKBackwardTilingData_.set_expertCount(expertCount_);
    MoeGatingTopKBackwardTilingData_.set_k(k_);
    MoeGatingTopKBackwardTilingData_.set_gradYDtypeSize(gradYDtypeSize_);
    MoeGatingTopKBackwardTilingData_.set_renorm(renorm_);
    MoeGatingTopKBackwardTilingData_.set_normType(normType_);
    MoeGatingTopKBackwardTilingData_.set_routedScalingFactor(routedScalingFactor_);
    MoeGatingTopKBackwardTilingData_.set_eps(eps_);
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
    context_->SetBlockDim(MoeGatingTopKBackwardTilingData_.get_needCoreNum());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    MoeGatingTopKBackwardTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                          context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(MoeGatingTopKBackwardTilingData_.GetDataSize());
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

void MoeGatingTopKBackwardTilingBase::DumpTilingInfo()
{
    OP_LOGD(context_->GetNodeName(), "ubSize:  %ld", aicoreParams_.ubSize);
    OP_LOGD(context_->GetNodeName(), "numBlocks:  %ld", aicoreParams_.numBlocks);
    OP_LOGD(context_->GetNodeName(), "needCoreNum:  %ld", needCoreNum_);
    OP_LOGD(context_->GetNodeName(), "perCoreRows:  %ld", perCoreRows_);
    OP_LOGD(context_->GetNodeName(), "lastCoreRows:  %ld", lastCoreRows_);
    OP_LOGD(context_->GetNodeName(), "baseRows:  %ld", baseRows_);
    OP_LOGD(context_->GetNodeName(), "perLoopTimes:  %ld", perLoopTimes_);
    OP_LOGD(context_->GetNodeName(), "perTailRows:  %ld", perTailRows_);
    OP_LOGD(context_->GetNodeName(), "lastLoopTimes:  %ld", lastLoopTimes_);
    OP_LOGD(context_->GetNodeName(), "lastTailRows:  %ld", lastTailRows_);
    OP_LOGD(context_->GetNodeName(), "tokenCount:  %ld", tokenCount_);
    OP_LOGD(context_->GetNodeName(), "expertCount:  %ld", expertCount_);
    OP_LOGD(context_->GetNodeName(), "k:  %ld", k_);
    OP_LOGD(context_->GetNodeName(), "gradYDtypeSize:  %ld", gradYDtypeSize_);
    OP_LOGD(context_->GetNodeName(), "renorm:  %ld", renorm_);
    OP_LOGD(context_->GetNodeName(), "normType:  %ld", normType_);
    OP_LOGD(context_->GetNodeName(), "routedScalingFactor:  %f", routedScalingFactor_);
    OP_LOGD(context_->GetNodeName(), "eps:  %f", eps_);
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKBackward, MoeGatingTopKBackwardTilingBase, 2000);
} // namespace optiling
