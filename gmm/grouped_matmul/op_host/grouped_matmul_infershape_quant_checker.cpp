/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_infershape_quant_checker.cpp
 * \brief
 */
#include "grouped_matmul_infershape_common_util.h"
#include "grouped_matmul_infershape_quant_checker.h"
#include "gmm/common/op_host/log_format_util.h"

namespace ops {

static const std::unordered_set<ge::DataType> X_TYPE_SUPPORT_SET = {
    ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
static const std::unordered_set<ge::DataType> WEIGHT_TYPE_SUPPORT_SET = {ge::DT_INT8, ge::DT_FLOAT8_E4M3FN,
                                                                         ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8,
                                                                         ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
static const std::unordered_set<ge::DataType> BIAS_TYPE_SUPPORT_SET = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                                                                       ge::DT_INT32};
static const std::unordered_set<ge::DataType> SCALE_TYPE_SUPPORT_SET = {ge::DT_UINT64, ge::DT_INT64, ge::DT_FLOAT,
                                                                        ge::DT_BF16, ge::DT_FLOAT8_E8M0};
static const std::unordered_set<ge::DataType> PERTOEKN_SCALE_TYPE_SUPPORT_SET = {ge::DT_FLOAT, ge::DT_FLOAT8_E8M0};
constexpr size_t GROUP_LIST_DIM_NUM = 1UL;
constexpr size_t GROUP_LIST_SPARSE_DIM_NUM = 2UL;
constexpr int64_t GROUP_LIST_TYPE_SPARSE_M = 2L;

static bool inline IsNonEmpty(const gert::Shape *shape)
{
    return (shape != nullptr && !(shape->GetDimNum() == 1 && shape->GetDim(0) == 0));
}

bool GroupedMatmulQuantChecker::LogicXOR(bool cond1, bool cond2) const
{
    uint64_t result = static_cast<uint64_t>(cond1) ^ static_cast<uint64_t>(cond2);
    return static_cast<bool>(result);
}

ge::graphStatus GroupedMatmulQuantChecker::GetXAndWeightDimValue(const gert::InferShapeContext *context,
                                                                 const GMMAttrs &gmmAttrs)
{
    auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_IF(!(IsNonEmpty(xShape) && IsNonEmpty(weightShape)),
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), !IsNonEmpty(xShape) ? "x" : "weight"),
                return ge::GRAPH_FAILED);
    xdimNum_ = xShape->GetDimNum();
    weightdimNum_ = weightShape->GetDimNum();
    OP_CHECK_IF(xdimNum_ < GMM_MIN_FM_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(xdimNum_), Ops::Transformer::Gmm::FormatString("The dim num of x's 1st tensor should be greater than 1, but it is \
[%zu].", xdimNum_).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightdimNum_ < GMM_MIN_WEIGHT_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "weight", std::to_string(weightdimNum_), Ops::Transformer::Gmm::FormatString("The dim num of weight's 1st tensor should be greater than 1, \
but it is [%zu].", weightdimNum_).c_str()),
                return ge::GRAPH_FAILED);
    xMDim_ = gmmAttrs.transposeX ? xShape->GetDim(xdimNum_ - 1) : xShape->GetDim(xdimNum_ - PENULTIMATE_DIM);
    xKDim_ = gmmAttrs.transposeX ? xShape->GetDim(xdimNum_ - PENULTIMATE_DIM) : xShape->GetDim(xdimNum_ - 1);
    weightKDim_ = gmmAttrs.transposeWeight ? weightShape->GetDim(weightdimNum_ - 1) :
                                             weightShape->GetDim(weightdimNum_ - PENULTIMATE_DIM);
    weightNDim_ = gmmAttrs.transposeWeight ? weightShape->GetDim(weightdimNum_ - PENULTIMATE_DIM) :
                                             weightShape->GetDim(weightdimNum_ - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckDtypeValid(const gert::InferDataTypeContext *context) const
{
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto weightDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_WEIGHT, 0);
    // mandory param dtype check
    OP_CHECK_IF(X_TYPE_SUPPORT_SET.find(xDtype) == X_TYPE_SUPPORT_SET.end(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(), Ops::Transformer::Gmm::FormatString("Data type [%s] is not supported for x's 1st tensor; supported types are: INT8, "
                        "FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1.", ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(WEIGHT_TYPE_SUPPORT_SET.find(weightDtype) == WEIGHT_TYPE_SUPPORT_SET.end(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "weight", ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString(), Ops::Transformer::Gmm::FormatString("Data type [%s] is not supported for weight; supported types are: INT8, FLOAT8_E4M3FN, "
                        "FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1.", ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()).c_str()),
                return ge::GRAPH_FAILED);
    if (xDtype == ge::DataType::DT_INT8 || weightDtype == ge::DataType::DT_INT8) {
        OP_CHECK_IF(
            !(xDtype == ge::DataType::DT_INT8 && weightDtype == ge::DataType::DT_INT8),
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, weight", Ops::Transformer::Gmm::FormatString("%s, %s", ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(), ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()).c_str(), Ops::Transformer::Gmm::FormatString("When data type of x is int8, data type of weight should also be int8, vice versa.").c_str()),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        LogicXOR((xDtype == ge::DataType::DT_HIFLOAT8), (weightDtype == ge::DataType::DT_HIFLOAT8)),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, weight", Ops::Transformer::Gmm::FormatString("%s, %s", ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(), ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()).c_str(), Ops::Transformer::Gmm::FormatString("When one input dtype is HIFLOAT8, the other input must also be HIFLOAT8.").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        LogicXOR((xDtype == ge::DataType::DT_FLOAT8_E4M3FN || xDtype == ge::DataType::DT_FLOAT8_E5M2),
                 (weightDtype == ge::DataType::DT_FLOAT8_E4M3FN || weightDtype == ge::DataType::DT_FLOAT8_E5M2)),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, weight", Ops::Transformer::Gmm::FormatString("%s, %s", ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(), ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()).c_str(), Ops::Transformer::Gmm::FormatString("When x input dtype is FLOAT8, the weight input dtype must also be FLOAT8.").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(LogicXOR((xDtype == ge::DataType::DT_FLOAT4_E2M1 || xDtype == ge::DataType::DT_FLOAT4_E1M2),
        (weightDtype == ge::DataType::DT_FLOAT4_E2M1 || weightDtype == ge::DataType::DT_FLOAT4_E1M2)),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, weight", Ops::Transformer::Gmm::FormatString("%s, %s", ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(), ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()).c_str(), Ops::Transformer::Gmm::FormatString("When x input dtype is FLOAT4, the weight input dtype must also be FLOAT4.").c_str()),
                return ge::GRAPH_FAILED);
    auto ScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_SCALE, 0);
    OP_CHECK_IF(SCALE_TYPE_SUPPORT_SET.find(ScaleDtype) == SCALE_TYPE_SUPPORT_SET.end(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "scale", ge::TypeUtils::DataTypeToAscendString(ScaleDtype).GetString(), "Data type is not supported for scale; supported types are: UINT64, INT64, "
                        "FLOAT, BF16, and FLOAT8_E8M0."),
                return ge::GRAPH_FAILED);

    auto biasDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_BIAS, 0);
    OP_CHECK_IF(
        BIAS_TYPE_SUPPORT_SET.find(biasDtype) == BIAS_TYPE_SUPPORT_SET.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "bias", ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString(), "Data type is not supported for bias; supported types are: FLOAT16, BF16, FLOAT, and INT32."),
        return ge::GRAPH_FAILED);

    auto pertokenScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    if (pertokenScaleDtype != ge::DT_UNDEFINED) {
        OP_CHECK_IF(
            PERTOEKN_SCALE_TYPE_SUPPORT_SET.find(pertokenScaleDtype) == PERTOEKN_SCALE_TYPE_SUPPORT_SET.end(),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "perTokenScale", ge::TypeUtils::DataTypeToAscendString(pertokenScaleDtype).GetString(), "Data type is not supported for pertokenScale; supported types are: FLOAT, and FLOAT8_E8M0."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckNotZeroValueForNoneSplitAxis(const gert::InferShapeContext *context,
                                                                             const GMMAttrs &gmmAttrs) const
{
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        auto eDimValue = weightShape->GetDim(0);
        OP_CHECK_IF(xKDim_ == 0 || eDimValue == 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        context->GetNodeName(), "x K, weight E",
                        Ops::Transformer::Gmm::FormatString("%ld, %ld", xKDim_, eDimValue).c_str(),
                        "Non-split K and E dimensions must be non-zero."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckScenarioValidForShape(const gert::InferShapeContext *context,
                                                                      const GMMAttrs &gmmAttrs) const
{
    // only support for single/single/single Scenario
    auto xSecondTensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 1);
    auto weightSecondTensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 1);
    OP_CHECK_IF(IsNonEmpty(xSecondTensorShape),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x", Ops::Transformer::Gmm::FormatString("Only support single/single/single scenario for now, but the second \
tensor of tensor list x is not empty.").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsNonEmpty(weightSecondTensorShape),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight", Ops::Transformer::Gmm::FormatString("Only support single/single/single scenario for now, but the second \
tensor of tensor list weight is not empty.").c_str()),
                return ge::GRAPH_FAILED);
    // check split item value valid
    OP_CHECK_IF(gmmAttrs.splitItem != GMM_X_SEPARATED && gmmAttrs.splitItem != GMM_NO_SEPARATED,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "splitItem", std::to_string(gmmAttrs.splitItem), "Invalid splitItem, which can only be one of 2/3."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gmmAttrs.groupType != GMM_SPLIT_M && gmmAttrs.groupType != GMM_SPLIT_K,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "groupType", std::to_string(gmmAttrs.groupType), "Invalid groupType, which can only be one of 0/2."),
                return ge::GRAPH_FAILED);
    // non split axis cannot be empty tensor
    OP_CHECK_IF(CheckNotZeroValueForNoneSplitAxis(context, gmmAttrs) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", Ops::Transformer::Gmm::FormatString("CheckNotZeroValueForNoneSplitAxis Failed.").c_str()), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckFormatValid(const gert::InferShapeContext *context) const
{
    const auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const auto xFormat = xDesc->GetOriginFormat();
    OP_CHECK_IF(xFormat != ge::FORMAT_ND && xFormat != ge::FORMAT_NCL && xFormat != ge::FORMAT_NCHW,
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                    context->GetNodeName(), "x", ge::TypeUtils::FormatToAscendString(xFormat).GetString(),
                    "Format of x should be ND, NCL or NCHW."),
                return ge::GRAPH_FAILED);
    const auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    const auto weightFormat = weightDesc->GetOriginFormat();
    OP_CHECK_IF(weightFormat != ge::FORMAT_ND && weightFormat != ge::FORMAT_NCL &&
                    weightFormat != ge::FORMAT_NCHW && weightFormat != ge::FORMAT_FRACTAL_NZ,
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                    context->GetNodeName(), "weight", ge::TypeUtils::FormatToAscendString(weightFormat).GetString(),
                    "Format of weight should be ND, NCL, FORMAT_FRACTAL_NZ or NCHW."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShapeForBias(const gert::InferShapeContext *context) const
{
    auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, 0);
    if (IsNonEmpty(biasShape)) {
        size_t biasDimNum = biasShape->GetDimNum();
        OP_CHECK_IF(biasDimNum != 2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "bias", std::to_string(biasDimNum),
                                                         "When bias is not null, its dim should be 2."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            biasShape->GetDim(0) != groupNum_ || biasShape->GetDim(1) != static_cast<int64_t>(weightNDim_),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), "bias",
                Ops::Transformer::Gmm::FormatString("[%ld, %ld]", biasShape->GetDim(0), biasShape->GetDim(1)).c_str(),
                Ops::Transformer::Gmm::FormatString("The shape of bias should be [%ld, %ld].", groupNum_, weightNDim_).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool GroupedMatmulQuantChecker::IsDoubleScaleScenario(const gert::Shape *scaleShape,
                                                      const gert::Shape *perTokenScaleShape) const
{ // pertoken dim num support 2 or 1 for now
    if (perTokenScaleShape->GetDimNum() != 2 && perTokenScaleShape->GetDimNum() != 1) {
        return false;
    }
    bool IsScaleMatchedDoubleScale = false;
    bool IsPertokenScaleMatchedDoubleScale = false;
    if (scaleShape->GetDimNum() == 2) { // dim num 2 with shape (e, 1)
        IsScaleMatchedDoubleScale = (scaleShape->GetDim(0) == groupNum_) && (scaleShape->GetDim(1) == 1);
    } else {
        IsScaleMatchedDoubleScale = scaleShape->GetDim(0) == groupNum_;
    }

    if (perTokenScaleShape->GetDimNum() == 2) { // dim num 2 with shape (e, 1)
        IsPertokenScaleMatchedDoubleScale =
            (perTokenScaleShape->GetDim(0) == groupNum_) && (perTokenScaleShape->GetDim(1) == 1);
    } else {
        IsPertokenScaleMatchedDoubleScale = perTokenScaleShape->GetDim(0) == groupNum_;
    }
    return IsScaleMatchedDoubleScale && IsPertokenScaleMatchedDoubleScale;
}

bool GroupedMatmulQuantChecker::IsPerTileQuantMode(const gert::InferShapeContext *context,
                                                   const GMMAttrs &gmmAttrs) const
{
    auto perTokenScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    bool isPerTileQuantMode = false;
    if (!IsNonEmpty(perTokenScaleShape)) {
        return isPerTileQuantMode;
    }
    auto perTokenScaleDimNum = perTokenScaleShape->GetDimNum();
    // 2 is the minimum dimension for perTokenScale in perTile case
    if (perTokenScaleDimNum < 2 || perTokenScaleDimNum != xdimNum_) {
        return false;
    }
    auto perTokenMDim = gmmAttrs.transposeX ? perTokenScaleShape->GetDim(xdimNum_ - 1) :
                                              perTokenScaleShape->GetDim(xdimNum_ - PENULTIMATE_DIM);
    if (perTokenMDim != xMDim_) {
        return false;
    }
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    auto scaleDimNum = scaleShape->GetDimNum();
    // 2 is the minimum dimension for scale in perTile case
    if (scaleDimNum < 2 || scaleDimNum != weightdimNum_) {
        return false;
    }
    auto scaleKDim = gmmAttrs.transposeWeight ? scaleShape->GetDim(scaleDimNum - 1) :
                                                scaleShape->GetDim(scaleDimNum - PENULTIMATE_DIM);
    auto scaleNDim = gmmAttrs.transposeWeight ? scaleShape->GetDim(scaleDimNum - PENULTIMATE_DIM) :
                                                scaleShape->GetDim(scaleDimNum - 1);
    bool isKdimValid =
        ((gmmAttrs.groupType == GMM_SPLIT_K && scaleKDim == (weightKDim_ / PERTILE_GROUP_SIZE) + groupNum_) ||
         (gmmAttrs.groupType == GMM_SPLIT_M &&
          scaleKDim == (weightKDim_ + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE));
    if (scaleNDim == (weightNDim_ + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE && isKdimValid) {
        isPerTileQuantMode = true;
    }
    return isPerTileQuantMode;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckGroupedMatmulPerGroupDim(const gert::InferShapeContext *context,
                                                                         const GMMAttrs &gmmAttrs) const
{
    auto scaleDimNum = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0)->GetDimNum();
    auto perTokenDimNum = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0)->GetDimNum();
    OP_CHECK_IF(xdimNum_ != GMM_MIN_FM_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(xdimNum_),
                                                         "The dim num of x should be 2."),
                return ge::GRAPH_FAILED);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        OP_CHECK_IF(weightdimNum_ != GMM_SPLIT_M_SINGLE_WEIGHT_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "weight", std::to_string(weightdimNum_),
                                                             "The dim num of weight should be 3 when groupType is 0 (split M)."),
                    return ge::GRAPH_FAILED);
    } else if (gmmAttrs.groupType == GMM_SPLIT_K) {
        OP_CHECK_IF(weightdimNum_ != GMM_SPLIT_K_SINGLE_WEIGHT_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "weight", std::to_string(weightdimNum_),
                                                             "The dim num of weight should be 2 when groupType is 2 (split K)."),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(scaleDimNum != weightdimNum_,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context->GetNodeName(), "scale, weight",
                Ops::Transformer::Gmm::FormatString("%zu, %zu", scaleDimNum, weightdimNum_).c_str(),
                Ops::Transformer::Gmm::FormatString("The dim num of scale should equal that of weight when groupType is %ld.",
                                                    gmmAttrs.groupType).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(perTokenDimNum != xdimNum_,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context->GetNodeName(), "perTokenScale, x",
                Ops::Transformer::Gmm::FormatString("%zu, %zu", perTokenDimNum, xdimNum_).c_str(),
                "The dim num of perTokenScale should equal that of x."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckGroupedMatmulPerTileShape(const gert::InferShapeContext *context,
                                                                          const GMMAttrs &gmmAttrs) const
{
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    auto perTokenShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    auto perTokenMDim =
        gmmAttrs.transposeX ? perTokenShape->GetDim(xdimNum_ - 1) : perTokenShape->GetDim(xdimNum_ - PENULTIMATE_DIM);
    auto perTokenKDim =
        gmmAttrs.transposeX ? perTokenShape->GetDim(xdimNum_ - PENULTIMATE_DIM) : perTokenShape->GetDim(xdimNum_ - 1);
    auto scaleKDim = gmmAttrs.transposeWeight ? scaleShape->GetDim(weightdimNum_ - 1) :
                                                scaleShape->GetDim(weightdimNum_ - PENULTIMATE_DIM);
    auto scaleNDim = gmmAttrs.transposeWeight ? scaleShape->GetDim(weightdimNum_ - PENULTIMATE_DIM) :
                                                scaleShape->GetDim(weightdimNum_ - 1);
    OP_CHECK_IF(perTokenMDim != xMDim_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "perTokenScale M", std::to_string(perTokenMDim), Ops::Transformer::Gmm::FormatString("perTokenScale M should equal x M [%ld].", xMDim_).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleNDim != (weightNDim_ + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale N", std::to_string(scaleNDim), Ops::Transformer::Gmm::FormatString("scale N should equal ceil(weight N [%ld] / 128).", weightNDim_).c_str()),
                return ge::GRAPH_FAILED);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        int64_t expectScaleKValue = (weightKDim_ + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE;
        OP_CHECK_IF(perTokenKDim != scaleKDim || scaleKDim != expectScaleKValue,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "perTokenScale K, scale K, weight K", Ops::Transformer::Gmm::FormatString("%ld, %ld, %ld", perTokenKDim, scaleKDim, weightKDim_).c_str(), Ops::Transformer::Gmm::FormatString("When quantification mode is G-B quantification and groupType is 0 (split M), perTokenScale K and scale K must equal ceil(weight K / 128).").c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        int64_t expectScaleKValue = (weightKDim_ / PERTILE_GROUP_SIZE) + groupNum_;
        OP_CHECK_IF(perTokenKDim != scaleKDim || scaleKDim != expectScaleKValue,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "perTokenScale K, scale K, weight K", Ops::Transformer::Gmm::FormatString("%ld, %ld, %ld", perTokenKDim, scaleKDim, weightKDim_).c_str(), Ops::Transformer::Gmm::FormatString("When quantification mode is G-B quantification and groupType is 2 (split K), perTokenScale K and scale K must equal weight K / 128 plus group count %ld.", groupNum_).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShapeForPerGroupQuantParam(const gert::InferShapeContext *context,
                                                                           const GMMAttrs &gmmAttrs) const
{
    if (IsPerTileQuantMode(context, gmmAttrs)) {
        OP_CHECK_IF(CheckGroupedMatmulPerGroupDim(context, gmmAttrs) != ge::GRAPH_SUCCESS ||
                        CheckGroupedMatmulPerTileShape(context, gmmAttrs) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", Ops::Transformer::Gmm::FormatString("CheckShapeForPerGroupQuantParam failed").c_str()), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckPertokenShapeInNormalQuantMode(
    const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs, const gert::Shape *perTokenScaleShape) const
{
    OP_CHECK_IF(perTokenScaleShape->GetDimNum() != 1 && perTokenScaleShape->GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "perTokenScale", std::to_string(perTokenScaleShape->GetDimNum()), "In T-C or K-C quant mode, perTokenScale rank should be 1 or 2."),
                return ge::GRAPH_FAILED);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        if (perTokenScaleShape->GetDimNum() == 1) {
            OP_CHECK_IF(perTokenScaleShape->GetDim(0) != xMDim_ && perTokenScaleShape->GetDim(0) != groupNum_,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "perTokenScale", Ops::Transformer::Gmm::FormatString("(%ld)", perTokenScaleShape->GetDim(0)).c_str(), Ops::Transformer::Gmm::FormatString("perTokenScale shape should be (%ld,) or (%ld,).", xMDim_, groupNum_).c_str()),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(perTokenScaleShape->GetDim(0) != groupNum_ || perTokenScaleShape->GetDim(1) != 1,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "perTokenScale", Ops::Transformer::Gmm::FormatString("(%ld, %ld)", perTokenScaleShape->GetDim(0), perTokenScaleShape->GetDim(1)).c_str(), Ops::Transformer::Gmm::FormatString("perTokenScale shape should be (%ld, 1).", groupNum_).c_str()),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // split k
        if (perTokenScaleShape->GetDimNum() == 1) {
            OP_CHECK_IF(perTokenScaleShape->GetDim(0) != groupNum_,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "perTokenScale", Ops::Transformer::Gmm::FormatString("(%ld)", perTokenScaleShape->GetDim(0)).c_str(), Ops::Transformer::Gmm::FormatString("perTokenScale shape should be (%ld,).", groupNum_).c_str()),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                perTokenScaleShape->GetDim(0) != groupNum_ ||
                    (perTokenScaleShape->GetDim(1) != xMDim_ && perTokenScaleShape->GetDim(1) != 1),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "perTokenScale", Ops::Transformer::Gmm::FormatString("(%ld, %ld)", perTokenScaleShape->GetDim(0), perTokenScaleShape->GetDim(1)).c_str(), Ops::Transformer::Gmm::FormatString("perTokenScale shape should be (%ld,%ld) or (%ld,1).", groupNum_, xMDim_, groupNum_).c_str()),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShapeForQuantParam(const gert::InferShapeContext *context,
                                                                   const GMMAttrs &gmmAttrs) const
{
    if (gmmAttrs.outputDtype == GMM_OUT_DTYPE_INT32) {
        return ge::GRAPH_SUCCESS;
    }
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    auto perTokenScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    if (IsPerTileQuantMode(context, gmmAttrs)) {
        return CheckShapeForPerGroupQuantParam(context, gmmAttrs);
    }
    // mx在tiling中校验异常场景
    if (scaleShape->GetDimNum() == MXFP_TYPEM_SCALE_DIM_NUM || scaleShape->GetDimNum() == MXFP_TYPEK_SCALE_DIM_NUM) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(scaleShape->GetDimNum() != 2 && scaleShape->GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "scale",
                                                         std::to_string(scaleShape->GetDimNum()),
                                                         "The dim num of scale should be 1 or 2."),
                return ge::GRAPH_FAILED);
    if (scaleShape->GetDimNum() == 1) {
        OP_CHECK_IF(scaleShape->GetDim(0) != groupNum_,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale", std::to_string(scaleShape->GetDim(0)),
                                                       Ops::Transformer::Gmm::FormatString("The 1st dim value of scale should be g[%ld].", groupNum_).c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            scaleShape->GetDim(0) != groupNum_ || (scaleShape->GetDim(1) != 1 && scaleShape->GetDim(1) != weightNDim_),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "scale",
            Ops::Transformer::Gmm::FormatString("[%ld, %ld]", scaleShape->GetDim(0), scaleShape->GetDim(1)).c_str(),
            Ops::Transformer::Gmm::FormatString("The shape of scale should be [%ld, 1] or [%ld, %ld].", groupNum_, groupNum_, weightNDim_).c_str()),
            return ge::GRAPH_FAILED);
    }
    if (IsNonEmpty(perTokenScaleShape)) {
        if (IsDoubleScaleScenario(scaleShape, perTokenScaleShape)) {
            return ge::GRAPH_SUCCESS;
        }
        if (CheckPertokenShapeInNormalQuantMode(context, gmmAttrs, perTokenScaleShape) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::GetGroupNumValue(const gert::InferShapeContext *context)
{
    auto groupListShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListShape);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_GROUP_LIST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListTypePtr);
    if (*groupListTypePtr == GROUP_LIST_TYPE_SPARSE_M) {
        OP_CHECK_IF(groupListShape->GetDimNum() != GROUP_LIST_SPARSE_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "groupList", std::to_string(groupListShape->GetDimNum()),
                        Ops::Transformer::Gmm::FormatString("groupList rank should be %zu.", GROUP_LIST_SPARSE_DIM_NUM).c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(groupListShape->GetDimNum() != GROUP_LIST_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "groupList", std::to_string(groupListShape->GetDimNum()),
                        Ops::Transformer::Gmm::FormatString("groupList rank should be %zu.", GROUP_LIST_DIM_NUM).c_str()),
                    return ge::GRAPH_FAILED);
    }
    groupNum_ = groupListShape->GetDim(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShapeForGrouplist(const gert::InferShapeContext *context) const
{
    OP_CHECK_IF(groupNum_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "groupList", std::to_string(groupNum_),
                    "The groupList first dimension value should be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupNum_ > 1024,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "groupList", std::to_string(groupNum_),
                    "The groupList number must not exceed 1024."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShapeValid(const gert::InferShapeContext *context,
                                                           const GMMAttrs &gmmAttrs) const
{
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_IF(
        xKDim_ != weightKDim_,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "x K, weight K",
            Ops::Transformer::Gmm::FormatString("%ld, %ld", xKDim_, weightKDim_).c_str(),
            "The K dimensions of x and weight must be equal."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeForGrouplist(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "groupList", Ops::Transformer::Gmm::FormatString("CheckShapeForGrouplist failed.").c_str()), return ge::GRAPH_FAILED);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        OP_CHECK_IF(xdimNum_ != 2 && weightdimNum_ != 3,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context->GetNodeName(), "x, weight",
                        Ops::Transformer::Gmm::FormatString("%zu, %zu", xdimNum_, weightdimNum_).c_str(),
                        "When split m, x rank should be 2 and weight rank should be 3."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            weightShape->GetDim(0) != groupNum_,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "weight", std::to_string(weightShape->GetDim(0)),
                Ops::Transformer::Gmm::FormatString("Weight first dimension should equal group count %ld.", groupNum_).c_str()),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(xdimNum_ != 2 && weightdimNum_ != 2,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context->GetNodeName(), "x, weight",
                        Ops::Transformer::Gmm::FormatString("%zu, %zu", xdimNum_, weightdimNum_).c_str(),
                        "When split k, x rank should be 2 and weight rank should be 2."),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(CheckShapeForBias(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", Ops::Transformer::Gmm::FormatString("CheckShapeForBias failed.").c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        gmmAttrs.outputDtype != GMM_OUT_DTYPE_INT32 &&
            !IsNonEmpty(context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0)),
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "scale"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeForQuantParam(context, gmmAttrs) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", Ops::Transformer::Gmm::FormatString("CheckShapeForQuantParam failed.").c_str()), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool GroupedMatmulQuantChecker::CheckDataContainsNegativeValue() const
{
    return xMDim_ < 0L || xKDim_ < 0L || weightNDim_ < 0L || weightKDim_ < 0L;
}

bool GroupedMatmulQuantChecker::CheckShapeContainsNegativeValue(const gert::Shape *shape) const
{
    if (shape == nullptr) {
        return false;
    }
    size_t dimNum = shape->GetDimNum();
    for (size_t i = 0; i < dimNum; i++) {
        if (shape->GetDim(i) < 0L) {
            return true;
        }
    }
    return false;
}

bool GroupedMatmulQuantChecker::CheckNonDataInputContainsNegativeValue(const gert::InferShapeContext *context) const
{
    if (groupNum_ < 0L) {
        return true;
    }
    auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, 0);
    if (IsNonEmpty(biasShape) && CheckShapeContainsNegativeValue(biasShape)) {
        return true;
    }
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    if (IsNonEmpty(scaleShape) && CheckShapeContainsNegativeValue(scaleShape)) {
        return true;
    }
    auto pertokenScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    if (IsNonEmpty(pertokenScaleShape) && CheckShapeContainsNegativeValue(pertokenScaleShape)) {
        return true;
    }
    return false;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShape(const gert::InferShapeContext *context,
                                                      const GroupedMatmulCommonUtil &commonUtil) const
{
    if (CheckDataContainsNegativeValue() || CheckNonDataInputContainsNegativeValue(context)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(CheckScenarioValidForShape(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", Ops::Transformer::Gmm::FormatString("CheckScenarioValidForShape failed.").c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeValid(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", Ops::Transformer::Gmm::FormatString("CheckShapeValid failed.").c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckFormatValid(context) != ge::GRAPH_SUCCESS, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", Ops::Transformer::Gmm::FormatString("CheckFormatValid failed.").c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::UpdateShapeY(gert::InferShapeContext *context, size_t idxY,
                                                        std::vector<int64_t> &yDims)
{
    gert::Shape *yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(yDims.size());
    for (size_t dim = 0; dim < yDims.size(); ++dim) {
        yShape->SetDim(dim, yDims[dim]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::InferOutShape(gert::InferShapeContext *context, const GMMAttrs &gmmAttrs)
{
    std::vector<int64_t> yDims = {xMDim_, weightNDim_};
    if (gmmAttrs.groupType == GMM_SPLIT_K) {
        yDims.insert(yDims.begin(), groupNum_);
    }
    OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y, yDims) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "y", Ops::Transformer::Gmm::FormatString("Failed to update y shape.").c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckScenarioValidForDtype(const gert::InferDataTypeContext *context,
                                                                      const GMMAttrs &gmmAttrs) const
{
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto scaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_SCALE, 0);
    auto perTokenScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_PERTOKEN_SCALE, 0);

    if (gmmAttrs.groupType == GMM_SPLIT_K) {
        OP_CHECK_IF(xDtype == ge::DT_INT8 || scaleDtype == ge::DT_INT64 || scaleDtype == ge::DT_UINT64,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context->GetNodeName(), "x, scale",
                        Ops::Transformer::Gmm::FormatString("%s, %s",
                            ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                            ge::TypeUtils::DataTypeToAscendString(scaleDtype).GetString()).c_str(),
                        "Split k does not support int8 x or int64/uint64 scale."),
                    return ge::GRAPH_FAILED);
    }
    if ((xDtype == ge::DT_HIFLOAT8 || xDtype == ge::DT_FLOAT8_E5M2 ||
         xDtype == ge::DT_FLOAT8_E4M3FN) && perTokenScaleDtype != ge::DT_UNDEFINED) {
        OP_CHECK_IF(
            scaleDtype != perTokenScaleDtype || (scaleDtype != ge::DataType::DT_FLOAT && scaleDtype != ge::DataType::DT_FLOAT8_E8M0),
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context->GetNodeName(), "scale, perTokenScale",
                Ops::Transformer::Gmm::FormatString("%s, %s",
                    ge::TypeUtils::DataTypeToAscendString(scaleDtype).GetString(),
                    ge::TypeUtils::DataTypeToAscendString(perTokenScaleDtype).GetString()).c_str(),
                "Scale and perTokenScale dtypes should match and be FLOAT or FLOAT8_E8M0."), return ge::GRAPH_FAILED);
    } else if (xDtype == ge::DT_FLOAT4_E2M1 || xDtype == ge::DT_FLOAT4_E1M2) {
        OP_CHECK_IF(
            scaleDtype != ge::DataType::DT_FLOAT8_E8M0,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "scale", ge::TypeUtils::DataTypeToAscendString(scaleDtype).GetString(),
                "When x dtype is FLOAT4, scale dtype should be FLOAT8_E8M0."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckDtype(const gert::InferDataTypeContext *context,
                                                      const GroupedMatmulCommonUtil &commonUtil) const
{
    OP_CHECK_IF(CheckDtypeValid(context) != ge::GRAPH_SUCCESS, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", Ops::Transformer::Gmm::FormatString("CheckDtypeValid failed.").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckScenarioValidForDtype(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", Ops::Transformer::Gmm::FormatString("CheckScenarioValidForDtype failed.").c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::InferOutDtype(gert::InferDataTypeContext *context)
{
    auto attrs = context->GetAttrs();
    const int64_t *outputDtype = attrs->GetInt(GMM_INDEX_ATTR_OUTPUT_DTYPE);
    ge::DataType yDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto it = GMM_OUTPUT_DTYPE_MAP.find(*outputDtype);
    OP_CHECK_IF(it == GMM_OUTPUT_DTYPE_MAP.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "outputDtype", std::to_string(*outputDtype),
                    "The output dtype should be INT8, BF16, INT32, FLOAT16 or FLOAT32."),
                return ge::GRAPH_FAILED);
    yDtype = it->second;
    context->SetOutputDataType(GMM_INDEX_OUT_Y, yDtype);
    return ge::GRAPH_SUCCESS;
}
} // namespace ops