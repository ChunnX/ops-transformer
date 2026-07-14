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
 * \file moe_gating_top_k_backward_tiling_arch35.cpp
 * \brief
 */

#include "moe_gating_top_k_backward_tiling_arch35.h"
#include "op_host/tiling_util.h"

namespace optiling {

const static int64_t NORM_TYPE_SIGMOID = 1;
const static size_t X_NORM_INPUT_DIMS = 2;
const static size_t GRAD_Y_INPUT_DIMS = 2;
const static size_t EXPERT_IDX_INPUT_DIMS = 2;
const static size_t GRAD_X_OUTPUT_DIMS = 2;
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
const static int64_t DOUBLE_BUFFER_NUM = 2;
const static int64_t UB_ALIGN_BYTES_MINUS_ONE = 31;
const static int64_t UB_RESERVE_SPACE = 1024;
const static int64_t NUM_TWO = 2;
const static int64_t MAX_EXPERT_COUNT = 2048;
const static int64_t SIZE_OF_FLOAT32 = 4;
const static int64_t SIZE_OF_INT32 = 4;
const static int64_t NUM_BYTES_FOUR = 4;
const static int64_t NUM_BYTES_TWO = 2;
const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216;
const static uint64_t TILING_KEY_REGBASE = 10000;
const static int64_t DIM_ZERO = 0;
const static int64_t DIM_ONE = 1;

inline int64_t AlignBytes_(int64_t x)
{
    return (x + UB_ALIGN_BYTES_MINUS_ONE) & ~UB_ALIGN_BYTES_MINUS_ONE;
}

MoeGatingTopKBackwardTilingRegbase::MoeGatingTopKBackwardTilingRegbase(gert::TilingContext *context)
    : Ops::Transformer::OpTiling::TilingBaseClass(context)
{
}

bool MoeGatingTopKBackwardTilingRegbase::IsCapable()
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    return true;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platform_info"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    OP_CHECK_IF(aicoreParams_.numBlocks == 0, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(aicoreParams_.ubSize == 0, OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::DoOpTiling()
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
    ret = SplitRows();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    tilingData_.set_needCoreNum(needCoreNum_);
    tilingData_.set_perCoreRows(perCoreRows_);
    tilingData_.set_lastCoreRows(lastCoreRows_);
    tilingData_.set_baseRows(baseRows_);
    tilingData_.set_perLoopTimes(perLoopTimes_);
    tilingData_.set_perTailRows(perTailRows_);
    tilingData_.set_lastLoopTimes(lastLoopTimes_);
    tilingData_.set_lastTailRows(lastTailRows_);
    tilingData_.set_tokenCount(tokenCount_);
    tilingData_.set_expertCount(expertCount_);
    tilingData_.set_k(k_);
    tilingData_.set_gradYDtypeSize(gradYDtypeSize_);
    tilingData_.set_renorm(renorm_);
    tilingData_.set_normType(normType_);
    tilingData_.set_routedScalingFactor(routedScalingFactor_);
    tilingData_.set_eps(eps_);

    DumpTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKBackwardTilingRegbase::GetTilingKey() const
{
    return TILING_KEY_REGBASE;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::PostTiling()
{
    context_->SetBlockDim(tilingData_.get_needCoreNum());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::CheckInputShape()
{
    OP_CHECK_IF(CheckXNorm() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check XNorm failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGradY() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check GradY failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckExpertIdx() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Check ExpertIdx failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::CheckXNorm()
{
    const gert::StorageShape *xNormShapePtr = context_->GetInputShape(X_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xNormShapePtr);
    auto xNormDimSize = xNormShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(xNormDimSize != X_NORM_INPUT_DIMS,
                OP_LOGE(context_, "xNorm: Rank must be 2, but got %lu.", xNormDimSize), return ge::GRAPH_FAILED);

    tokenCount_ = xNormShapePtr->GetOriginShape().GetDim(DIM_ZERO);
    expertCount_ = xNormShapePtr->GetOriginShape().GetDim(DIM_ONE);
    OP_CHECK_IF(
        expertCount_ > MAX_EXPERT_COUNT,
        OP_LOGE(context_, "xNorm: Dimension 1 (N) must be <= 2048, but got %ld.", expertCount_),
        return ge::GRAPH_FAILED);

    auto xNormDesc = context_->GetInputDesc(X_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xNormDesc);
    auto xNormDtype = xNormDesc->GetDataType();
    OP_CHECK_IF(xNormDtype != ge::DataType::DT_FLOAT,
                OP_LOGE(context_, "xNorm: Unsupported dtype. Expected float32, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(xNormDtype).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::CheckGradY()
{
    const gert::StorageShape *gradYShapePtr = context_->GetInputShape(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYShapePtr);
    auto gradYDimSize = gradYShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(gradYDimSize != GRAD_Y_INPUT_DIMS,
                OP_LOGE(context_, "gradY: Rank must be 2, but got %lu.", gradYDimSize), return ge::GRAPH_FAILED);

    OP_CHECK_IF(gradYShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_,
                OP_LOGE(context_, "gradY: Must match xNorm dim 0, but got %ld vs %ld.",
                        gradYShapePtr->GetOriginShape().GetDim(DIM_ZERO), tokenCount_),
                return ge::GRAPH_FAILED);
    k_ = gradYShapePtr->GetOriginShape().GetDim(DIM_ONE);
    OP_CHECK_IF(k_ > expertCount_ || k_ <= 0,
                OP_LOGE(context_, "gradY: Dimension 1 (K) must be in (0, %ld], but got %ld.", expertCount_, k_),
                return ge::GRAPH_FAILED);

    auto gradYDesc = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYDesc);
    gradYDtype_ = gradYDesc->GetDataType();
    OP_CHECK_IF(gradYDtype_ != ge::DataType::DT_FLOAT && gradYDtype_ != ge::DataType::DT_FLOAT16 &&
                    gradYDtype_ != ge::DataType::DT_BF16,
                OP_LOGE(context_, "gradY: Unsupported dtype. Expected float32/float16/bf16, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradYDtype_).c_str()),
                return ge::GRAPH_FAILED);
    gradYDtypeSize_ = gradYDtype_ == ge::DataType::DT_FLOAT ? NUM_BYTES_FOUR : NUM_BYTES_TWO;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::CheckExpertIdx()
{
    const gert::StorageShape *expertIdxShapePtr = context_->GetInputShape(EXPERT_IDX_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr);
    auto expertIdxDimSize = expertIdxShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(expertIdxDimSize != EXPERT_IDX_INPUT_DIMS,
                OP_LOGE(context_, "expertIdx: Rank must be 2, but got %lu.", expertIdxDimSize),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(expertIdxShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_,
                OP_LOGE(context_, "expertIdx: Must match xNorm dim 0, but got %ld vs %ld.",
                        expertIdxShapePtr->GetOriginShape().GetDim(DIM_ZERO), tokenCount_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertIdxShapePtr->GetOriginShape().GetDim(DIM_ONE) != k_,
                OP_LOGE(context_, "expertIdx: Must match gradY dim 1, but got %ld vs %ld.",
                        expertIdxShapePtr->GetOriginShape().GetDim(DIM_ONE), k_),
                return ge::GRAPH_FAILED);

    auto expertIdxDesc = context_->GetInputDesc(EXPERT_IDX_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxDesc);
    auto expertIdxDtype = expertIdxDesc->GetDataType();
    OP_CHECK_IF(expertIdxDtype != ge::DataType::DT_INT32,
                OP_LOGE(context_, "expertIdx: Unsupported dtype. Expected int32, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(expertIdxDtype).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *renormPtr = attrs->GetAttrPointer<int64_t>(RENORM_ATTR_INDEX);
    if (renormPtr != nullptr) {
        renorm_ = *renormPtr;
    }
    OP_LOGI(context_, "Attr renorm: %ld.", renorm_);

    const int64_t *normTypePtr = attrs->GetAttrPointer<int64_t>(NORM_TYPE_ATTR_INDEX);
    if (normTypePtr != nullptr) {
        normType_ = *normTypePtr;
    }
    OP_LOGI(context_, "Attr normType: %ld.", normType_);
    OP_CHECK_IF(normType_ != NORM_TYPE_SIGMOID,
                OP_LOGE(context_, "Attr normType: expected %ld, but got %ld.", NORM_TYPE_SIGMOID, normType_),
                return ge::GRAPH_FAILED);

    const float *routedScalingFactorPtr = attrs->GetAttrPointer<float>(ROUTED_SCALING_FACTOR_ATTR_INDEX);
    if (routedScalingFactorPtr != nullptr) {
        routedScalingFactor_ = *routedScalingFactorPtr;
    }
    OP_LOGI(context_, "Attr routedScalingFactor: %f.", routedScalingFactor_);

    const float *epsPtr = attrs->GetAttrPointer<float>(EPS_ATTR_INDEX);
    if (epsPtr != nullptr) {
        eps_ = *epsPtr;
    }
    OP_LOGI(context_, "Attr eps: %f.", eps_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::CheckOutShape()
{
    auto gradXShapePtr = context_->GetOutputShape(GRAD_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXShapePtr);

    auto gradXDimSize = gradXShapePtr->GetOriginShape().GetDimNum();
    OP_CHECK_IF(gradXDimSize != GRAD_X_OUTPUT_DIMS,
                OP_LOGE(context_, "gradX: Rank must be 2, but got %lu.", gradXDimSize), return ge::GRAPH_FAILED);

    OP_CHECK_IF(gradXShapePtr->GetOriginShape().GetDim(DIM_ZERO) != tokenCount_,
                OP_LOGE(context_, "gradX: Must match xNorm dim 0, but got %ld vs %ld.",
                        gradXShapePtr->GetOriginShape().GetDim(DIM_ZERO), tokenCount_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gradXShapePtr->GetOriginShape().GetDim(DIM_ONE) != expertCount_,
                OP_LOGE(context_, "gradX: Must match xNorm dim 1, but got %ld vs %ld.",
                        gradXShapePtr->GetOriginShape().GetDim(DIM_ONE), expertCount_),
                return ge::GRAPH_FAILED);

    auto gradXDesc = context_->GetOutputDesc(GRAD_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXDesc);
    auto gradXDtype = gradXDesc->GetDataType();
    OP_CHECK_IF(gradXDtype != gradYDtype_,
                OP_LOGE(context_, "gradX: Mismatched dtype. Expected %s, but got %s.",
                        ge::TypeUtils::DataTypeToSerialString(gradYDtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(gradXDtype).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::CalcMaxRows()
{
    int64_t gradYQuePerTokenSpace = k_ * gradYDtypeSize_;
    int64_t indicesQuePerTokenSpace = k_ * SIZE_OF_INT32;
    int64_t xQuePerTokenSpace = expertCount_ * SIZE_OF_FLOAT32;
    int64_t outQuePerTokenSpace = expertCount_ * gradYDtypeSize_;
    int64_t wPrimeCachePerTokenSpace = k_ * SIZE_OF_FLOAT32;
    int64_t bufnPerTokenSpace = expertCount_ * SIZE_OF_FLOAT32;

    int64_t availableSpace = static_cast<int64_t>(aicoreParams_.ubSize) - UB_RESERVE_SPACE;
    int64_t quePerTokenSpace =
        DOUBLE_BUFFER_NUM * (gradYQuePerTokenSpace + indicesQuePerTokenSpace + xQuePerTokenSpace) + outQuePerTokenSpace;
    int64_t bufPerTokenSpace = NUM_TWO * bufnPerTokenSpace + wPrimeCachePerTokenSpace;
    int64_t maxRows = availableSpace / (quePerTokenSpace + bufPerTokenSpace);

    while (maxRows > 0) {
        int64_t queSpace =
            DOUBLE_BUFFER_NUM * maxRows * AlignBytes_(gradYQuePerTokenSpace) +
            DOUBLE_BUFFER_NUM * maxRows * AlignBytes_(indicesQuePerTokenSpace) +
            DOUBLE_BUFFER_NUM * AlignBytes_(maxRows * xQuePerTokenSpace) +
            AlignBytes_(maxRows * outQuePerTokenSpace);
        int64_t bufSpace = NUM_TWO * AlignBytes_(maxRows * bufnPerTokenSpace) +
                           maxRows * AlignBytes_(wPrimeCachePerTokenSpace);

        uint32_t reduceSumTmpSize = 0;
        uint32_t tmpMaxValue = 0;
        int64_t kAligned = AlignBytes_(k_ * SIZE_OF_FLOAT32) >> 1;
        auto shape = ge::Shape({maxRows, kAligned});
        AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::AR, true, false,
                                           tmpMaxValue, reduceSumTmpSize);

        int64_t usedSpace = UB_RESERVE_SPACE + queSpace + bufSpace + static_cast<int64_t>(reduceSumTmpSize);
        if (usedSpace <= static_cast<int64_t>(aicoreParams_.ubSize)) {
            break;
        }
        maxRows--;
    }

    baseRows_ = maxRows;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKBackwardTilingRegbase::SplitRows()
{
    perCoreRows_ = Ops::Base::CeilDiv(tokenCount_, static_cast<int64_t>(aicoreParams_.numBlocks));
    needCoreNum_ = Ops::Base::CeilDiv(tokenCount_, perCoreRows_);
    lastCoreRows_ = tokenCount_ % perCoreRows_ == 0 ? perCoreRows_ : tokenCount_ % perCoreRows_;

    auto ret = CalcMaxRows();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    OP_CHECK_IF(baseRows_ <= 0, OP_LOGE(context_, "baseRows must be > 0, but got %ld.", baseRows_),
                return ge::GRAPH_FAILED);

    perLoopTimes_ = (perCoreRows_ + baseRows_ - 1) / baseRows_;
    perTailRows_ = perCoreRows_ - (perLoopTimes_ - 1) * baseRows_;
    lastLoopTimes_ = (lastCoreRows_ + baseRows_ - 1) / baseRows_;
    lastTailRows_ = lastCoreRows_ - (lastLoopTimes_ - 1) * baseRows_;
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKBackwardTilingRegbase::DumpTiling()
{
    OP_LOGD(context_, "ubSize:  %ld", aicoreParams_.ubSize);
    OP_LOGD(context_, "numBlocks:  %ld", aicoreParams_.numBlocks);
    OP_LOGD(context_, "needCoreNum:  %ld", needCoreNum_);
    OP_LOGD(context_, "perCoreRows:  %ld", perCoreRows_);
    OP_LOGD(context_, "lastCoreRows:  %ld", lastCoreRows_);
    OP_LOGD(context_, "baseRows:  %ld", baseRows_);
    OP_LOGD(context_, "perLoopTimes:  %ld", perLoopTimes_);
    OP_LOGD(context_, "perTailRows:  %ld", perTailRows_);
    OP_LOGD(context_, "lastLoopTimes:  %ld", lastLoopTimes_);
    OP_LOGD(context_, "lastTailRows:  %ld", lastTailRows_);
    OP_LOGD(context_, "tokenCount:  %ld", tokenCount_);
    OP_LOGD(context_, "expertCount:  %ld", expertCount_);
    OP_LOGD(context_, "k:  %ld", k_);
    OP_LOGD(context_, "gradYDtypeSize:  %ld", gradYDtypeSize_);
    OP_LOGD(context_, "renorm:  %ld", renorm_);
    OP_LOGD(context_, "normType:  %ld", normType_);
    OP_LOGD(context_, "routedScalingFactor:  %f", routedScalingFactor_);
    OP_LOGD(context_, "eps:  %f", eps_);
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKBackward, MoeGatingTopKBackwardTilingRegbase, 1000);
} // namespace optiling
