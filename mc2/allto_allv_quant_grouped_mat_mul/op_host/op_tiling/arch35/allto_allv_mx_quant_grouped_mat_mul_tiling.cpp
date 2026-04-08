/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_mx_quant_grouped_mat_mul_tiling.cpp
 * \brief
 */

#include "allto_allv_mx_quant_grouped_mat_mul_tiling.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
bool AlltoAllvMXQuantGmmTiling::IsCapable()
{
    // support fp8_e5m2 or fp8_e4m3
    if (gmmXDataType_ != ge::DT_FLOAT8_E5M2 &&
        gmmXDataType_ != ge::DT_FLOAT8_E4M3FN &&
        gmmXDataType_ != ge::DT_FLOAT4_E2M1) {
        return false;
    }
    if (gmmWeightDataType_ != ge::DT_FLOAT8_E5M2 &&
        gmmWeightDataType_ != ge::DT_FLOAT8_E4M3FN &&
        gmmWeightDataType_ != ge::DT_FLOAT4_E2M1) {
        return false;
    }
    OP_LOGD(context_->GetNodeName(), "AlltoAllvMXQuantGmmTiling is capable.");
    return true;
}

uint64_t AlltoAllvMXQuantGmmTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(ADD_TPL_FP8_FP4, hasSharedExpertFlag_, transGmmWeight_, transMmWeight_);
    return tilingKey;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckInputDtype() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckInputDtype.");
    // gmm dtype
    ge::DataType gXDataType = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    ge::DataType gWtDataType = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    ge::DataType gYDataType = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();
    // check gmm input dtype
    ge::graphStatus status = CheckGmmInputDtype(gXDataType, gWtDataType, gYDataType);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // has mm
    if (hasSharedExpertFlag_) {
        // check mm input dtype(same as gmm)
        status = CheckMmInputDtype(gXDataType, gWtDataType, gYDataType);
        if (status != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    // mxfp4 check
    if (gXDataType == ge::DT_FLOAT4_E2M1) {
        status = CheckFp4Input(gXDataType, gWtDataType);
        if (status != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGD(context_->GetNodeName(), "end CheckInputDtype.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckScaleFormatAndDtype() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckScaleFormatAndDtype.");\
    // check gmm scale
    ge::graphStatus status = CheckGmmScale();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // has mm
    if (hasSharedExpertFlag_) {
        // check mm scale
        status = CheckMmScale();
        if (status != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGD(context_->GetNodeName(), "end CheckScaleFormatAndDtype.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckScaleShape() const
{
    OP_LOGD(context_->GetNodeName(), "Start CheckScaleShape.");
    // check gmm scale
    ge::graphStatus status = CheckGmmScaleShape();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (hasSharedExpertFlag_) {
        // check mm scale
        status = CheckMmScaleShape();
        if (status != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGD(context_->GetNodeName(), "End CheckScaleShape.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::DoGmmTiling(uint64_t gmmxMSzie)
{
    OP_LOGD(context_->GetNodeName(), "start DoGmmTiling.");
    // gmm group matmul tiling
    if (gmmxMSzie != 0) {
        AlltoAllvMXQuantGmmTilingHelper gmmHelper(*this);
        GE_ASSERT_GRAPH_SUCCESS(gmmHelper.SetInputParams(gmmxMSzie, n1_, h1_, transGmmWeight_));
        GE_ASSERT_GRAPH_SUCCESS(gmmHelper.Process());
        tilingData->gmmQuantTilingData = gmmHelper.GetAlltoAllvQuantHelperData();
        PrintGMMQuantTilingData(tilingData->gmmQuantTilingData);
    }
    // mm group matmul tiling
    if (bs_ != 0) {
        AlltoAllvMXQuantGmmTilingHelper mmHelper(*this);
        GE_ASSERT_GRAPH_SUCCESS(mmHelper.SetInputParams(bs_, n2_, h2_, transMmWeight_));
        GE_ASSERT_GRAPH_SUCCESS(mmHelper.Process());
        tilingData->mmQuantTilingData = mmHelper.GetAlltoAllvQuantHelperData();
        PrintGMMQuantTilingData(tilingData->mmQuantTilingData);
    }
    // permute out
    GetPermuteOutSize();
    OP_LOGD(context_->GetNodeName(), "end DoGmmTiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckQuantGroupSize() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckQuantGroupSize.");
    auto groupSizePtr = context_->GetAttrs()->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr,
        OP_LOGE(context_->GetNodeName(), "The groupSize can not be null."),
        return ge::GRAPH_FAILED);
    uint64_t groupSizeK = static_cast<uint64_t>(*groupSizePtr) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    OP_TILING_CHECK(((groupSizeM != MX_GROUP_SIZE_M && groupSizeM != 0) || 
                    (groupSizeN != MX_GROUP_SIZE_N && groupSizeN != 0) || 
                    (groupSizeK != MX_GROUP_SIZE_K && groupSizeK != 0)),
            OP_LOGE(context_->GetNodeName(), "When mx quant mode, GroupSizeM should be 1 or 0, groupSizeN should be 1 or 0 "
            "and groupSizeK should be 32 or 0, but actual is [groupSizeM = %lu, groupSizeN = %lu, groupSizeK = %lu].",
            groupSizeM, groupSizeN, groupSizeK),
            return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckQuantGroupSize.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckQuantMode() const
{
    OP_LOGD(context_->GetNodeName(), "start CheckQuantMode.");
    // check gmmXQuantMode null
    OP_TILING_CHECK(gmmXQuantModePtr_ == nullptr,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmXQuantMode attr can not be null."),
        return ge::GRAPH_FAILED);
    // check gmmXQuantMode
    int64_t gXQuantMode = *gmmXQuantModePtr_;
    OP_TILING_CHECK(gXQuantMode != static_cast<int64_t>(MX_QUANT_MODE),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmXQuantMode should be 6, "
        "but actual is %ld.", gXQuantMode),
        return ge::GRAPH_FAILED);
    // check gmmWeightQuantMode null
    OP_TILING_CHECK(gmmWeightQuantModePtr_ == nullptr,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmWeightQuantMode attr can not be null."),
        return ge::GRAPH_FAILED);
    // check gmmWeightQuantMode
    int64_t gWtQuantMode = *gmmWeightQuantModePtr_;
    OP_TILING_CHECK(gWtQuantMode != static_cast<int64_t>(MX_QUANT_MODE),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmWeightQuantMode should be 6, but actual is %ld.", \
            gWtQuantMode), return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        // check mmXQuantMode null
        OP_TILING_CHECK(mmXQuantModePtr_ == nullptr,
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmXQuantModePtr "
            "attr can not be null."), return ge::GRAPH_FAILED);
        // mmXQuantMode(same as gmmXQuantMode)
        int64_t xQuantMode = *mmXQuantModePtr_;
        OP_TILING_CHECK(xQuantMode != gXQuantMode,
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmXQuantMode "
            "should be same as gmmXQuantMode(6), but actual is %ld.", xQuantMode), return ge::GRAPH_FAILED);
        // check mmWeightQuantMode null
        OP_TILING_CHECK(mmWeightQuantModePtr_ == nullptr,
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmWeightQuantModePtr "
            "attr can not be null."), return ge::GRAPH_FAILED);
        // mmWeightQuantMode(same as gmmWeightQuantMode)
        int64_t wtQuantMode = *mmWeightQuantModePtr_;
        OP_TILING_CHECK(wtQuantMode != gWtQuantMode,
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmWeightQuantMode "
            "should be same as gmmWeightQuantMode(6), but actual is %ld.", wtQuantMode),
            return ge::GRAPH_FAILED);
    }
    ge::graphStatus status = CheckQuantGroupSize();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckQuantMode.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckGmmScale() const
{
    // check gmmXScale null
    auto gXScaleDesc = context_->GetRequiredInputDesc(GMM_X_SCALE_INDEX);
    OP_TILING_CHECK(gXScaleDesc == nullptr,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmXScale should not be null."),
        return ge::GRAPH_FAILED);
    // check gmmXScale format
    OP_TILING_CHECK(gXScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(context_->GetNodeName(), "gmmXScale storage format should be ND, but actual is %s.",
        Ops::Base::ToString(gXScaleDesc->GetStorageFormat()).c_str()),
        return ge::GRAPH_FAILED);
    // check gmmXScale dataType                
    OP_TILING_CHECK(gXScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmXScale should be fp8_e8m0, "
        "but actual is %s.", ge::TypeUtils::DataTypeToSerialString(gXScaleDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    // check gmmWeightScale null
    auto gWtScaleDesc = context_->GetRequiredInputDesc(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(gWtScaleDesc == nullptr,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmWeightScale should not be null."),
        return ge::GRAPH_FAILED);
    // check gmmWeightScale format
    OP_TILING_CHECK(gWtScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(context_->GetNodeName(), "gmmWeightScale storage format should be ND, but actual is %s.",
        Ops::Base::ToString(gWtScaleDesc->GetStorageFormat()).c_str()),
        return ge::GRAPH_FAILED);
    // check gmmWeightScale dataType                
    OP_TILING_CHECK(gWtScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmWeightScale should be fp8_e8m0, but "
        "actual is %s.", ge::TypeUtils::DataTypeToSerialString(gWtScaleDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckMmScale() const
{
    // check mmXScale null
    auto xScaleDesc = context_->GetOptionalInputDesc(MM_X_SCALE_INDEX);
    auto gXScaleDesc = context_->GetRequiredInputDesc(GMM_X_SCALE_INDEX);
    OP_TILING_CHECK(xScaleDesc == nullptr,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmXScale should not be null."),
        return ge::GRAPH_FAILED);
    // check mmXScale format
    OP_TILING_CHECK(xScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(context_->GetNodeName(), "mmXScale storage format should be ND, but actual is %s.",
        Ops::Base::ToString(xScaleDesc->GetStorageFormat()).c_str()),
        return ge::GRAPH_FAILED);
    // check mmXScale dataType(same as gmmXScale)
    OP_TILING_CHECK(xScaleDesc->GetDataType() != gXScaleDesc->GetDataType(),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmXScale should "
        "be same as gmmXScale(fp8_e8m0), but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(xScaleDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    // check mmWeightScale null
    auto wtScaleDesc = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_INDEX);
    auto gWtScaleDesc = context_->GetRequiredInputDesc(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(wtScaleDesc == nullptr,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmWeightScale should not be null."),
        return ge::GRAPH_FAILED);
    // check mmWeightScale format
    OP_TILING_CHECK(wtScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(context_->GetNodeName(), "mmWeightScale storage format should be ND, but actual is %s.",
        Ops::Base::ToString(wtScaleDesc->GetStorageFormat()).c_str()),
        return ge::GRAPH_FAILED);
    // check mmWeightScale dataType(same as gmmWeightScale)
    OP_TILING_CHECK(wtScaleDesc->GetDataType() != gWtScaleDesc->GetDataType(),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, mmWeightScale should "
        "be same as gmmWeightScale(fp8_e8m0), but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(wtScaleDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;   
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckMmInputDtype(ge::DataType gXDataType,
    ge::DataType gWtDataType, ge::DataType gYDataType) const
{
    // // check mmX dataType(same as gmmX)
    ge::DataType xDataType = context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType();
    OP_TILING_CHECK((xDataType != gXDataType), 
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, mmX should be same as gmmX(%s), "
        "but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(gXDataType).c_str(),
        ge::TypeUtils::DataTypeToSerialString(xDataType).c_str()), 
        return ge::GRAPH_FAILED);
    // check mmWeight dataType(same as gmmWeight)
    ge::DataType wtDataType = context_->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK((wtDataType != gWtDataType),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, mmWeight should be same as gmmWeight(%s), "
        "but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(gWtDataType).c_str(),
        ge::TypeUtils::DataTypeToSerialString(wtDataType).c_str()),
        return ge::GRAPH_FAILED);
    // check mmY dataType(same as gmmY)
    ge::DataType yDataType = context_->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(yDataType != gYDataType,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, mmY should be same as gmmY(%s), "
        "but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(gYDataType).c_str(),
        ge::TypeUtils::DataTypeToSerialString(yDataType).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckGmmInputDtype(ge::DataType gXDataType,
    ge::DataType gWtDataType, ge::DataType gYDataType) const
{
    static const std::vector<ge::DataType> legalDtypes = {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
        ge::DT_FLOAT4_E2M1};
    // check gmmX datatype
    OP_TILING_CHECK(std::find(legalDtypes.begin(), legalDtypes.end(), gXDataType) == legalDtypes.end(),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmX should be in {fp8_e5m2, fp8_e4m3, "
            "fp4_e2m1}, but actual is %s.", ge::TypeUtils::DataTypeToSerialString(gXDataType).c_str()), 
            return ge::GRAPH_FAILED);
    // check gmmWeight datatype
    OP_TILING_CHECK(std::find(legalDtypes.begin(), legalDtypes.end(), gWtDataType) == legalDtypes.end(),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmWeight should be in {fp8_e5m2, fp8_e4m3, "
            "fp4_e2m1}, but actual is %s.", ge::TypeUtils::DataTypeToSerialString(gWtDataType).c_str()), 
            return ge::GRAPH_FAILED);
    // check gmmY dataType
    OP_TILING_CHECK(gYDataType != ge::DT_FLOAT16 && gYDataType != ge::DT_BF16,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmY should be float16 or bfloat16, "
        "but actual is %s.", ge::TypeUtils::DataTypeToSerialString(gYDataType).c_str()),
        return ge::GRAPH_FAILED);
    if (permuteOutFlag_) {
        // check permuteOut dtype
        ge::DataType pOutDataType = context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX)->GetDataType();
        OP_TILING_CHECK(pOutDataType != gXDataType,
            OP_LOGE(context_->GetNodeName(), "When mx quant mode, permuteOut should be "
            "same as gmmX dataType(%s), but actual is %s.",
            ge::TypeUtils::DataTypeToSerialString(gXDataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(pOutDataType).c_str()), 
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckFp4Input(ge::DataType gXDataType,
    ge::DataType gWtDataType) const
{
    // mxfp4 gmmweight(same as gmmx)
    OP_TILING_CHECK(gXDataType != gWtDataType,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and gmmx is fp4_e2m1, gmmWeight should "
        "be same as gmmx, but actual is %s.",
        ge::TypeUtils::DataTypeToSerialString(gWtDataType).c_str()), 
        return ge::GRAPH_FAILED);
    // mxfp4 h1 is even and h1 not two
    OP_TILING_CHECK(h1_ % 2 != 0 || h1_ == 2,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and gmmx is fp4_e2m1, h1 should be even "
        "and not be 2, but actual is %lu.", h1_), return ge::GRAPH_FAILED);
    // mxfp4 gmmweight not trans n1 is even
    OP_TILING_CHECK(!transGmmWeight_ && n1_ % 2 != 0,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and gmmx is fp4_e2m1, gmmweight not "
        "trans n1 should be even, but actual is %lu.", n1_), return ge::GRAPH_FAILED);
    // has mm
    if (hasSharedExpertFlag_) {
        // mxfp4 h2 is even and h2 not two
        OP_TILING_CHECK(h2_ % 2 != 0 || h2_ == 2,
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and gmmx is fp4_e2m1, h2 should be even "
            "and not be 2, but actual is %lu.", h2_), return ge::GRAPH_FAILED);
        // mxfp4 mmweight not trans n2 is even
        OP_TILING_CHECK(!transMmWeight_ && n2_ % 2 != 0,
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and gmmx is fp4_e2m1, gmmweight not "
            "trans n2 should be even, but actual is %lu.", n2_), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckGmmScaleShape() const
{
    // check gmmXScale dimNum
    size_t gxSDimNum = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(gxSDimNum != DIM_THREE,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmXScale input shape should be [3], "
        "but actual is %lu", gxSDimNum), return ge::GRAPH_FAILED);
    // check gmmXScale shape
    uint64_t gExpectH = Ops::Base::CeilDiv(h1_, MX_BASIC_FACTOR);
    uint64_t gxSDim0 = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t gxSDim1 = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t gxSDim2 = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    OP_TILING_CHECK((gxSDim0 != bsk_) || (gxSDim1 != gExpectH) || (gxSDim2 != 2),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, the expected shape of gmmxscale is "
        "(%lu, %lu, 2) but the actual is (%lu, %lu, %lu)", bsk_, gExpectH, gxSDim0, gxSDim1, gxSDim2),
        return ge::GRAPH_FAILED);
    // check gmmWeightScale dimNum
    size_t gwtSDimNum = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(gwtSDimNum != DIM_FOUR,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode, gmmWeightScale input shape should be [4], "
        "but actual is %lu", gwtSDimNum), return ge::GRAPH_FAILED);
    // check gmmWeightScale shape
    uint64_t gwtSDim0 = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t gwtSDim1 = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t gwtSDim2 = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    uint64_t gwtSDim3 = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_THREE);
    uint64_t gwtSH = transGmmWeight_ ? gwtSDim2 : gwtSDim1;
    uint64_t gwtSN = transGmmWeight_ ? gwtSDim1 : gwtSDim2;
    if (transGmmWeight_) {
        OP_TILING_CHECK((gwtSN != n1_) || (gwtSH != gExpectH) || (gwtSDim0 != e_) || (gwtSDim3 != 2),
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and trans gmmWeight, the expected "
            "shape of gmmWeightscale is (%lu, %lu, %lu, 2) but the actual is (%lu, %lu, %lu, %lu)",
            e_, n1_, gExpectH, gwtSDim0, gwtSN, gwtSH, gwtSDim3), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((gwtSN != n1_) || (gwtSH != gExpectH) || (gwtSDim0 != e_) || (gwtSDim3 != 2),
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and not trans gmmWeight, the expected "
            "shape of gmmWeightscale is (%lu, %lu, %lu, 2) but the actual is (%lu, %lu, %lu, %lu)",
            e_, gExpectH, n1_, gwtSDim0, gwtSH, gwtSN, gwtSDim3), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTiling::CheckMmScaleShape() const
{
    // check mmXScale dimNum
    size_t xSDimNum = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(xSDimNum != context_->GetRequiredInputShape(GMM_X_SCALE_INDEX)->GetStorageShape().GetDimNum(),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, xSDimNum input dimNum "
        "should be 3, but actual dimNum is %lu.", xSDimNum), return ge::GRAPH_FAILED);
    // check mmXScale shape
    uint64_t expectH = Ops::Base::CeilDiv(h2_, MX_BASIC_FACTOR);
    uint64_t xSDim0 = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xSDim1 = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t xSDim2 = context_->GetOptionalInputShape(MM_X_SCALE_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    OP_TILING_CHECK((xSDim0 != bs_) || (xSDim1 != expectH) || (xSDim2 != 2),
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, the expected shape "
        "of mmxscale is (%lu, %lu, 2) but the actual is (%lu, %lu, %lu)",
        bs_, expectH, xSDim0, xSDim1, xSDim2), return ge::GRAPH_FAILED);
    // check mmWeightScale dimNum
    size_t wtSDimNum = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(wtSDimNum != DIM_THREE,
        OP_LOGE(context_->GetNodeName(), "When mx quant mode and has shared expert, wtSDimNum input dimNum "
        "should be 3, but actual is %lu.", wtSDimNum), return ge::GRAPH_FAILED);
    // check mmWeightScale shape
    uint64_t wtSDim0 = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t wtSDim1 = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t wtSDim2 = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    uint64_t wtSDimH = transMmWeight_ ? wtSDim1 : wtSDim0;
    uint64_t wtSDimN = transMmWeight_ ? wtSDim0 : wtSDim1;
    if (transMmWeight_) {
        OP_TILING_CHECK((wtSDimN != n2_) || (wtSDimH != expectH) || (wtSDim2 != 2),
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and trans mmWeight, the expected shape of "
            "mmWeightscale is (%lu, %lu, 2) but the actual is (%lu, %lu, %lu)",
            n2_, expectH, wtSDimN, wtSDimH, wtSDim2),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((wtSDimN != n2_) || (wtSDimH != expectH) || (wtSDim2 != 2),
            OP_LOGE(context_->GetNodeName(), "When mx quant mode and not trans mmWeight, "
            "the expected shape of mmWeightscale is (%lu, %lu, 2) but the actual is (%lu, %lu, %lu)",
            expectH, n2_, wtSDimH, wtSDimN, wtSDim2),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void AlltoAllvMXQuantGmmTiling::GetPermuteOutSize()
{
    // permuteout size
    permuteOutSize_ = permuteOutFlag_ ? 0 : (a_ * h1_ * GetSizeByDataType(gmmXDataType_));
    // 将 permuteOutSize 对齐到 512 字节
    permuteOutSize_ = Ops::Base::CeilAlign(permuteOutSize_, static_cast<uint64_t>(BASIC_BLOCK_SIZE_512));
    if (gmmXDataType_ == ge::DT_FLOAT4_E2M1) {
        // mxfp4 场景占半字节，shapesize大小除以二
        permuteOutSize_ = Ops::Base::CeilDiv(permuteOutSize_, MXFP4_PACK_FACTOR);
    }
    // permutscaleout size
    uint64_t hSize = Ops::Base::CeilDiv(h1_, MX_BASIC_FACTOR);
    permuteScaleOutSize_ =  Ops::Base::CeilAlign((a_ * hSize * 2 * GetSizeByDataType(gmmWeightDataType_)),
        static_cast<uint64_t>(BASIC_BLOCK_SIZE_512));
}

ge::graphStatus AlltoAllvMXQuantGmmTilingHelper::SetInputParams(uint64_t M, uint64_t N, uint64_t K, bool transB) 
{
    OP_LOGD(context_->GetNodeName(), "start SetInputParams.");
    GetPlatformInfo();
    inputParams_.opName = context_->GetNodeName();
    inputParams_.kernelType = 0UL;
    // 输出是否切分，0/1代表输出多tensor， 2/3代表输出单tensor
    inputParams_.splitItem = 0;
    inputParams_.actType = GMM_ACT_TYPE_NONE;
    // common
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    inputParams_.transA = 0;
    inputParams_.transB = transB;
    inputParams_.hasBias = 0;
    inputParams_.isSingleX = 0;
    inputParams_.isSingleW = 0;
    inputParams_.isSingleY = 0;

    inputParams_.mSize = M;
    inputParams_.kSize = K;
    inputParams_.nSize = N;
    inputParams_.groupNum = SINGLE_GROUP_NUM;
    inputParams_.aQuantMode = static_cast<Mc2GroupedMatmulTiling::QuantMode>(1U << QUANT_MODE_MAP[MX_MODE]);
    inputParams_.bQuantMode = static_cast<Mc2GroupedMatmulTiling::QuantMode>(1U << QUANT_MODE_MAP[MX_MODE]);
    // 是否做切分
    inputParams_.groupType = optiling::Mc2GroupedMatmul::SPLIT_M;
    inputParams_.groupListType = 1;
    inputParams_.aDtype = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    inputParams_.cDtype = context_->GetInputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();
    inputParams_.biasDtype = ge::DT_INT32;
    inputParams_.scaleDtype = ge::DT_FLOAT8_E8M0;
    inputParams_.perTokenScaleDtype = ge::DT_FLOAT8_E8M0;

    mList_[0] = static_cast<int32_t>(M);
    kList_[0] = static_cast<int32_t>(K);
    nList_[0] = static_cast<int32_t>(N);
    SetKernelType();
    
    OP_LOGD(context_->GetNodeName(), "end SetInputParams.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvMXQuantGmmTilingHelper::Process() 
{
    GE_ASSERT_GRAPH_SUCCESS(DoOpTiling());
    GE_ASSERT_GRAPH_SUCCESS(DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvQuantGroupedMatMul, AlltoAllvMXQuantGmmTiling, 2);
} // namespace optiling