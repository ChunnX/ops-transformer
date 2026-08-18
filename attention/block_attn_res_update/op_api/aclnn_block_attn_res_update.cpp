/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <limits>
#include <string>

#include "aclnn_kernels/contiguous.h"
#include "aclnn_block_attn_res_update.h"
#include "block_attn_res_update.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "log/log.h"

using namespace op;

namespace {
constexpr size_t MATRIX_RANK = 2UL;
constexpr size_t VECTOR_RANK = 1UL;
constexpr int64_t MAX_D_SIZE = 8192L;
constexpr const char *ACLNN_API_NAME = "aclnnBlockAttnResUpdateGetWorkspaceSize";

struct BlockAttnResUpdateParams {
    aclTensor *partialBlock{nullptr};
    const aclTensor *delta{nullptr};
    const aclTensor *pseudoQuery{nullptr};
    const aclTensor *numerator{nullptr};
    const aclTensor *logitMax{nullptr};
    const aclTensor *expSum{nullptr};
    float eps{0.0F};
    aclTensor *h{nullptr};
};

static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BF16_DTYPE_SUPPORT_LIST = {DataType::DT_BF16};

static bool CheckNotNull(const BlockAttnResUpdateParams &params)
{
    OP_CHECK(params.partialBlock != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "partialBlock", "tensor can not be null"),
             return false);
    OP_CHECK(params.delta != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "delta", "tensor can not be null"),
             return false);
    OP_CHECK(params.pseudoQuery != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "pseudoQuery", "tensor can not be null"),
             return false);
    OP_CHECK(params.numerator != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "numerator", "tensor can not be null"),
             return false);
    OP_CHECK(params.logitMax != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "logitMax", "tensor can not be null"),
             return false);
    OP_CHECK(params.expSum != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "expSum", "tensor can not be null"),
             return false);
    OP_CHECK(params.h != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "h", "tensor can not be null"), return false);
    return true;
}

static bool CheckDtypeValid(const BlockAttnResUpdateParams &params)
{
    OP_CHECK(CheckType(params.partialBlock->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, "partialBlock",
                                       op::ToString(params.partialBlock->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()), return false);
    OP_CHECK(CheckType(params.pseudoQuery->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, "pseudoQuery",
                                       op::ToString(params.pseudoQuery->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()), return false);
    OP_CHECK(CheckType(params.numerator->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, "numerator",
                                       op::ToString(params.numerator->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()), return false);
    OP_CHECK(CheckType(params.logitMax->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, "logitMax",
                                       op::ToString(params.logitMax->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()), return false);
    OP_CHECK(CheckType(params.expSum->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, "expSum",
                                       op::ToString(params.expSum->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()), return false);
    OP_CHECK(CheckType(params.delta->GetDataType(), BF16_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, "delta",
                                       op::ToString(params.delta->GetDataType()).GetString(),
                                       op::ToString(BF16_DTYPE_SUPPORT_LIST).GetString()), return false);
    OP_CHECK(CheckType(params.h->GetDataType(), BF16_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, "h",
                                       op::ToString(params.h->GetDataType()).GetString(),
                                       op::ToString(BF16_DTYPE_SUPPORT_LIST).GetString()), return false);
    return true;
}

static bool CheckFormat(const BlockAttnResUpdateParams &params)
{
    const std::string formats = std::string(op::ToString(params.partialBlock->GetStorageFormat()).GetString()) +
                                ", " + op::ToString(params.delta->GetStorageFormat()).GetString() + ", " +
                                op::ToString(params.pseudoQuery->GetStorageFormat()).GetString() + ", " +
                                op::ToString(params.numerator->GetStorageFormat()).GetString() + ", " +
                                op::ToString(params.logitMax->GetStorageFormat()).GetString() + ", " +
                                op::ToString(params.expSum->GetStorageFormat()).GetString() + ", " +
                                op::ToString(params.h->GetStorageFormat()).GetString();
    OP_CHECK(params.partialBlock->GetStorageFormat() == Format::FORMAT_ND &&
                 params.delta->GetStorageFormat() == Format::FORMAT_ND &&
                 params.pseudoQuery->GetStorageFormat() == Format::FORMAT_ND &&
                 params.numerator->GetStorageFormat() == Format::FORMAT_ND &&
                 params.logitMax->GetStorageFormat() == Format::FORMAT_ND &&
                 params.expSum->GetStorageFormat() == Format::FORMAT_ND &&
                 params.h->GetStorageFormat() == Format::FORMAT_ND,
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                 ACLNN_API_NAME, "partialBlock, delta, pseudoQuery, numerator, logitMax, expSum, h", formats.c_str(),
                 "all tensor formats must be ND"), return false);
    return true;
}

static bool CheckShape(const BlockAttnResUpdateParams &params)
{
    const auto &partialBlockShape = params.partialBlock->GetViewShape();
    const auto &deltaShape = params.delta->GetViewShape();
    const auto &pseudoQueryShape = params.pseudoQuery->GetViewShape();
    const auto &numeratorShape = params.numerator->GetViewShape();
    const auto &logitMaxShape = params.logitMax->GetViewShape();
    const auto &expSumShape = params.expSum->GetViewShape();
    const auto &hShape = params.h->GetViewShape();

    const std::string matrixDims = std::to_string(partialBlockShape.GetDimNum()) + "D, " +
                                   std::to_string(deltaShape.GetDimNum()) + "D, " +
                                   std::to_string(numeratorShape.GetDimNum()) + "D, " +
                                   std::to_string(hShape.GetDimNum()) + "D";
    OP_CHECK(partialBlockShape.GetDimNum() == MATRIX_RANK && deltaShape.GetDimNum() == MATRIX_RANK &&
                 numeratorShape.GetDimNum() == MATRIX_RANK && hShape.GetDimNum() == MATRIX_RANK,
             OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(ACLNN_API_NAME, "partialBlock, delta, numerator, h",
                                                       matrixDims.c_str(), "all listed tensors must be 2D"),
             return false);
    const std::string vectorDims = std::to_string(pseudoQueryShape.GetDimNum()) + "D, " +
                                   std::to_string(logitMaxShape.GetDimNum()) + "D, " +
                                   std::to_string(expSumShape.GetDimNum()) + "D";
    OP_CHECK(pseudoQueryShape.GetDimNum() == VECTOR_RANK && logitMaxShape.GetDimNum() == VECTOR_RANK &&
                 expSumShape.GetDimNum() == VECTOR_RANK,
             OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(ACLNN_API_NAME, "pseudoQuery, logitMax, expSum",
                                                       vectorDims.c_str(), "all listed tensors must be 1D"),
             return false);

    const std::string matrixShapes = std::string(op::ToString(partialBlockShape).GetString()) + ", " +
                                     op::ToString(deltaShape).GetString() + ", " +
                                     op::ToString(numeratorShape).GetString() + ", " +
                                     op::ToString(hShape).GetString();
    OP_CHECK(deltaShape == partialBlockShape && numeratorShape == partialBlockShape && hShape == partialBlockShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_API_NAME, "partialBlock, delta, numerator, h",
                                                    matrixShapes.c_str(),
                                                    "all listed tensors must have the same [T, D] shape"),
             return false);

    const int64_t tSize = partialBlockShape.GetDim(0);
    const int64_t dSize = partialBlockShape.GetDim(1);
    const std::string statsShapes = std::string(op::ToString(logitMaxShape).GetString()) + ", " +
                                    op::ToString(expSumShape).GetString() + ", " +
                                    op::ToString(partialBlockShape).GetString();
    OP_CHECK(logitMaxShape.GetDim(0) == tSize && expSumShape.GetDim(0) == tSize,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 ACLNN_API_NAME, "logitMax, expSum, partialBlock", statsShapes.c_str(),
                 "logitMax and expSum must have shape [T], where T is the first dimension of partialBlock"),
             return false);
    const std::string queryShapes = std::string(op::ToString(pseudoQueryShape).GetString()) + ", " +
                                    op::ToString(partialBlockShape).GetString();
    OP_CHECK(pseudoQueryShape.GetDim(0) == dSize,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 ACLNN_API_NAME, "pseudoQuery, partialBlock", queryShapes.c_str(),
                 "pseudoQuery must have shape [D], where D is the second dimension of partialBlock"), return false);

    OP_CHECK(tSize >= 0,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_API_NAME, "T", std::to_string(tSize),
                                                   "the T dimension must be greater than or equal to 0"), return false);
    OP_CHECK(dSize >= 0 && dSize <= MAX_D_SIZE,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_API_NAME, "D", std::to_string(dSize),
                                                   "the D dimension must be in range [0, 8192]"), return false);

    // An empty [T, D] workload has no elements to update. All rank and shape relations above still apply;
    // only InferShape, tiling and kernel launch are skipped by the L0 operator.
    if (tSize == 0 || dSize == 0) {
        return true;
    }

    const std::string tdValues = std::to_string(tSize) + ", " + std::to_string(dSize);
    OP_CHECK(tSize <= std::numeric_limits<int64_t>::max() / dSize,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(ACLNN_API_NAME, "T, D", tdValues.c_str(),
                                                    "the product T * D must be less than or equal to INT64_MAX"),
             return false);
    return true;
}

static bool CheckContiguous(const BlockAttnResUpdateParams &params)
{
    OP_CHECK(IsContiguous(params.partialBlock),
             OP_LOGE_FOR_INVALID_STRIDE(ACLNN_API_NAME, "partialBlock",
                                        op::ToString(params.partialBlock->GetViewStrides()).GetString(),
                                        "contiguous strides"), return false);
    OP_CHECK(IsContiguous(params.delta),
             OP_LOGE_FOR_INVALID_STRIDE(ACLNN_API_NAME, "delta",
                                        op::ToString(params.delta->GetViewStrides()).GetString(),
                                        "contiguous strides"), return false);
    OP_CHECK(IsContiguous(params.pseudoQuery),
             OP_LOGE_FOR_INVALID_STRIDE(ACLNN_API_NAME, "pseudoQuery",
                                        op::ToString(params.pseudoQuery->GetViewStrides()).GetString(),
                                        "contiguous strides"), return false);
    OP_CHECK(IsContiguous(params.numerator),
             OP_LOGE_FOR_INVALID_STRIDE(ACLNN_API_NAME, "numerator",
                                        op::ToString(params.numerator->GetViewStrides()).GetString(),
                                        "contiguous strides"), return false);
    OP_CHECK(IsContiguous(params.logitMax),
             OP_LOGE_FOR_INVALID_STRIDE(ACLNN_API_NAME, "logitMax",
                                        op::ToString(params.logitMax->GetViewStrides()).GetString(),
                                        "contiguous strides"), return false);
    OP_CHECK(IsContiguous(params.expSum),
             OP_LOGE_FOR_INVALID_STRIDE(ACLNN_API_NAME, "expSum",
                                        op::ToString(params.expSum->GetViewStrides()).GetString(),
                                        "contiguous strides"), return false);
    OP_CHECK(IsContiguous(params.h),
             OP_LOGE_FOR_INVALID_STRIDE(ACLNN_API_NAME, "h",
                                        op::ToString(params.h->GetViewStrides()).GetString(),
                                        "contiguous strides"), return false);
    return true;
}

static bool CheckEps(float eps)
{
    OP_CHECK(std::isfinite(eps) && eps > 0.0F,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_API_NAME, "eps", std::to_string(eps),
                                                   "eps must be finite and greater than 0"), return false);
    return true;
}

static aclnnStatus CheckParams(const BlockAttnResUpdateParams &params)
{
    CHECK_COND(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed.");
    CHECK_COND(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed.");
    CHECK_COND(CheckFormat(params), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed.");
    CHECK_COND(CheckShape(params), ACLNN_ERR_PARAM_INVALID, "CheckShape failed.");
    CHECK_COND(CheckContiguous(params), ACLNN_ERR_PARAM_INVALID, "CheckContiguous failed.");
    CHECK_COND(CheckEps(params.eps), ACLNN_ERR_PARAM_INVALID, "CheckEps failed.");
    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(BlockAttnResUpdateParams &params)
{
    // Torch tensors may use a flattened storage shape. Tiling reads the original shape,
    // so synchronize it with the logical view shape before invoking the L0 operator.
    params.partialBlock->SetOriginalShape(params.partialBlock->GetViewShape());
    params.delta->SetOriginalShape(params.delta->GetViewShape());
    params.pseudoQuery->SetOriginalShape(params.pseudoQuery->GetViewShape());
    params.numerator->SetOriginalShape(params.numerator->GetViewShape());
    params.logitMax->SetOriginalShape(params.logitMax->GetViewShape());
    params.expSum->SetOriginalShape(params.expSum->GetViewShape());
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(
    aclTensor *partialBlock, const aclTensor *delta, const aclTensor *pseudoQuery, const aclTensor *numerator,
    const aclTensor *logitMax, const aclTensor *expSum, float eps, aclTensor *h, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK(workspaceSize != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "workspaceSize", "pointer can not be null"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "executor", "pointer can not be null"),
             return ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnBlockAttnResUpdate,
                   DFX_IN(partialBlock, delta, pseudoQuery, numerator, logitMax, expSum, eps),
                   DFX_OUT(partialBlock, h));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    BlockAttnResUpdateParams params{partialBlock, delta, pseudoQuery, numerator, logitMax, expSum, eps, h};
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto opResult = l0op::BlockAttnResUpdate(
        partialBlock, delta, pseudoQuery, numerator, logitMax, expSum, eps, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyH = l0op::ViewCopy(opResult, h, uniqueExecutor.get());
    CHECK_RET(viewCopyH != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnBlockAttnResUpdate(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockAttnResUpdate);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
