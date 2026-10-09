/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_grouped_matmul_activation_weight_nz.h"

#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"

using namespace op;

namespace {
constexpr char API_NAME[] = "aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize";
constexpr int64_t MX_DEQUANT_MODE = 2L;
constexpr int64_t FLOAT_DEQUANT_DTYPE = 0L;
constexpr int64_t SWIGLU_ACT_TYPE = 0L;
constexpr int64_t MX_QUANT_MODE = 2L;
constexpr int64_t MX_SCALE_ALG_OCP = 0L;
constexpr double MX_DST_TYPE_MAX = 0.0;
constexpr char MX_ROUND_MODE[] = "rint";
constexpr size_t SINGLE_TENSOR_SIZE = 1UL;
constexpr size_t FIRST_TENSOR_INDEX = 0UL;
constexpr size_t MATRIX_DIM_NUM = 2UL;
constexpr size_t BATCHED_MATRIX_DIM_NUM = 3UL;
constexpr size_t MATRIX_K_DIM = 1UL;
constexpr size_t BATCHED_WEIGHT_K_DIM = 1UL;
constexpr size_t TRANSPOSED_BATCHED_WEIGHT_K_DIM = 2UL;

#define GMMAWN_CHECK(cond, code, logExpr) \
    do { \
        if (!(cond)) { \
            logExpr; \
            return code; \
        } \
    } while (0)

bool IsA4W4(const aclTensor* x, const aclTensorList* weight)
{
    return x != nullptr && weight != nullptr && weight->Size() == SINGLE_TENSOR_SIZE &&
           (*weight)[FIRST_TENSOR_INDEX] != nullptr && x->GetDataType() == DataType::DT_FLOAT4_E2M1 &&
           (*weight)[FIRST_TENSOR_INDEX]->GetDataType() == DataType::DT_FLOAT4_E2M1;
}

bool IsNzTensor(const aclTensor* tensor)
{
    return tensor != nullptr && ge::GetPrimaryFormat(tensor->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ;
}

aclnnStatus CheckParams(const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList,
                        const aclTensorList* weight, const aclTensorList* weightScale, const aclTensor* bias,
                        int64_t dequantMode, int64_t dequantDtype, int64_t actType, bool transposeWeight,
                        const aclTensor* y)
{
    GMMAWN_CHECK(x != nullptr && xScale != nullptr && groupList != nullptr && weight != nullptr &&
                     weightScale != nullptr && y != nullptr,
                 ACLNN_ERR_PARAM_NULLPTR,
                 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "required input", "does not support nullptr"));
    GMMAWN_CHECK(IsA4W4(x, weight), ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "x, weight", "unsupported",
                                                       "only FLOAT4_E2M1/FLOAT4_E2M1 is supported"));
    GMMAWN_CHECK(weightScale->Size() == SINGLE_TENSOR_SIZE && (*weightScale)[FIRST_TENSOR_INDEX] != nullptr,
                 ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_TENSORNUM(API_NAME, "weightScale", weightScale->Size(), "1"));
    GMMAWN_CHECK(xScale->GetDataType() == DataType::DT_FLOAT8_E8M0 &&
                     (*weightScale)[FIRST_TENSOR_INDEX]->GetDataType() == DataType::DT_FLOAT8_E8M0,
                 ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "xScale, weightScale", "unsupported",
                                                       "both must be FLOAT8_E8M0"));
    GMMAWN_CHECK(groupList->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "groupList", "unsupported", "must be INT64"));
    GMMAWN_CHECK(IsNzTensor((*weight)[FIRST_TENSOR_INDEX]), ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(API_NAME, "weight", "unsupported", "must be FRACTAL_NZ"));
    const auto& xShape = x->GetViewShape();
    const auto& weightShape = (*weight)[FIRST_TENSOR_INDEX]->GetViewShape();
    GMMAWN_CHECK(xShape.GetDimNum() == MATRIX_DIM_NUM && weightShape.GetDimNum() == BATCHED_MATRIX_DIM_NUM,
                 ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "x, weight", "unsupported",
                                                          "x must be 2D and weight must be 3D"));
    const int64_t weightK = transposeWeight ? weightShape.GetDim(TRANSPOSED_BATCHED_WEIGHT_K_DIM) :
                                              weightShape.GetDim(BATCHED_WEIGHT_K_DIM);
    GMMAWN_CHECK(xShape.GetDim(MATRIX_K_DIM) == weightK, ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(API_NAME, "weight", "inconsistent K dimension",
                                                       "transposeWeight must match the weight logical shape"));
    GMMAWN_CHECK(y->GetDataType() == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "y", "unsupported", "only BF16 is supported"));
    GMMAWN_CHECK(bias == nullptr, ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "bias", "current A4W4 path does not support bias"));
    GMMAWN_CHECK(
        dequantMode == MX_DEQUANT_MODE && dequantDtype == FLOAT_DEQUANT_DTYPE, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "dequantMode/dequantDtype", "unsupported",
                                              "A4W4 only supports MX dequant mode 2 and float32 accumulation"));
    GMMAWN_CHECK(actType == SWIGLU_ACT_TYPE, ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "actType", "unsupported",
                                                       "only SwiGLU actType 0 is supported"));
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
    const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList, const aclTensorList* weight,
    const aclTensorList* weightScale, const aclTensor* bias, int64_t dequantMode, int64_t dequantDtype, int64_t actType,
    bool transposeWeight, int64_t groupListType, const aclIntArray* tuningConfig, int64_t swigluMode, double clampLimit,
    double gluAlpha, double gluBias, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    GMMAWN_CHECK(
        workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "workspaceSize/executor", "does not support nullptr"));
    aclnnStatus status = CheckParams(x, xScale, groupList, weight, weightScale, bias, dequantMode, dequantDtype,
                                     actType, transposeWeight, y);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    // The new public IR has no worker-only auxiliary inputs or yScale output.
    // A4W4 direct BF16 does not read or write them.
    return aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
        x, weight, weightScale, nullptr, bias, xScale, nullptr, groupList, dequantMode, dequantDtype, MX_QUANT_MODE,
        groupListType, tuningConfig, swigluMode, clampLimit, gluAlpha, gluBias, MX_ROUND_MODE, MX_SCALE_ALG_OCP,
        MX_DST_TYPE_MAX, y, nullptr, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulActivationWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream)
{
    return aclnnGroupedMatmulSwigluQuantWeightNzV3(workspace, workspaceSize, executor, stream);
}
} // extern "C"
