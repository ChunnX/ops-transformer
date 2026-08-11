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

using namespace op;

namespace {
constexpr size_t MATRIX_RANK = 2UL;
constexpr size_t VECTOR_RANK = 1UL;
constexpr int64_t MAX_D_SIZE = 8192L;

struct BlockAttnResUpdateParams {
    aclTensor *partialBlock{nullptr};
    const aclTensor *delta{nullptr};
    const aclTensor *interNumerator{nullptr};
    const aclTensor *interMax{nullptr};
    const aclTensor *interExpSum{nullptr};
    const aclTensor *effectiveQuery{nullptr};
    float eps{0.0F};
    aclTensor *h{nullptr};
};

static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BF16_DTYPE_SUPPORT_LIST = {DataType::DT_BF16};

static bool CheckNotNull(const BlockAttnResUpdateParams &params)
{
    OP_CHECK_NULL(params.partialBlock, return false);
    OP_CHECK_NULL(params.delta, return false);
    OP_CHECK_NULL(params.interNumerator, return false);
    OP_CHECK_NULL(params.interMax, return false);
    OP_CHECK_NULL(params.interExpSum, return false);
    OP_CHECK_NULL(params.effectiveQuery, return false);
    OP_CHECK_NULL(params.h, return false);
    return true;
}

static bool CheckDtypeValid(const BlockAttnResUpdateParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.partialBlock, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.interNumerator, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.interMax, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.interExpSum, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.effectiveQuery, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.delta, BF16_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.h, BF16_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckFormat(const BlockAttnResUpdateParams &params)
{
    OP_CHECK(params.partialBlock->GetStorageFormat() == Format::FORMAT_ND &&
                 params.delta->GetStorageFormat() == Format::FORMAT_ND &&
                 params.interNumerator->GetStorageFormat() == Format::FORMAT_ND &&
                 params.interMax->GetStorageFormat() == Format::FORMAT_ND &&
                 params.interExpSum->GetStorageFormat() == Format::FORMAT_ND &&
                 params.effectiveQuery->GetStorageFormat() == Format::FORMAT_ND &&
                 params.h->GetStorageFormat() == Format::FORMAT_ND,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResUpdate only supports ND format."), return false);
    return true;
}

static bool CheckShape(const BlockAttnResUpdateParams &params)
{
    const auto &partialBlockShape = params.partialBlock->GetViewShape();
    const auto &deltaShape = params.delta->GetViewShape();
    const auto &interNumeratorShape = params.interNumerator->GetViewShape();
    const auto &interMaxShape = params.interMax->GetViewShape();
    const auto &interExpSumShape = params.interExpSum->GetViewShape();
    const auto &effectiveQueryShape = params.effectiveQuery->GetViewShape();
    const auto &hShape = params.h->GetViewShape();

    OP_CHECK(partialBlockShape.GetDimNum() == MATRIX_RANK && deltaShape.GetDimNum() == MATRIX_RANK &&
                 interNumeratorShape.GetDimNum() == MATRIX_RANK && hShape.GetDimNum() == MATRIX_RANK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "partialBlock, delta, interNumerator and h must all be rank-2 tensors."),
             return false);
    OP_CHECK(interMaxShape.GetDimNum() == VECTOR_RANK && interExpSumShape.GetDimNum() == VECTOR_RANK &&
                 effectiveQueryShape.GetDimNum() == VECTOR_RANK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "interMax, interExpSum and effectiveQuery must all be rank-1 tensors."),
             return false);

    OP_CHECK(deltaShape == partialBlockShape && interNumeratorShape == partialBlockShape && hShape == partialBlockShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "partialBlock, delta, interNumerator and h must have the same [T, D] shape."),
             return false);

    const int64_t tSize = partialBlockShape.GetDim(0);
    const int64_t dSize = partialBlockShape.GetDim(1);
    OP_CHECK(tSize > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "T must be greater than 0, but got %ld.", tSize),
             return false);
    OP_CHECK(dSize >= 1 && dSize <= MAX_D_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "D must be in [1, 8192], but got %ld.", dSize), return false);
    OP_CHECK(tSize <= std::numeric_limits<int64_t>::max() / dSize,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "T * D exceeds the signed 64-bit tensor range."), return false);
    OP_CHECK(interMaxShape.GetDim(0) == tSize && interExpSumShape.GetDim(0) == tSize,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "interMax and interExpSum must have shape [T]."), return false);
    OP_CHECK(effectiveQueryShape.GetDim(0) == dSize,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "effectiveQuery must have shape [D]."), return false);
    return true;
}

static bool CheckContiguous(const BlockAttnResUpdateParams &params)
{
    OP_CHECK(IsContiguous(params.partialBlock),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "partialBlock must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.delta), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "delta must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.interNumerator),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "interNumerator must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.interMax),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "interMax must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.interExpSum),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "interExpSum must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.effectiveQuery),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "effectiveQuery must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.h), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h must be contiguous."), return false);
    return true;
}

static bool CheckEps(float eps)
{
    OP_CHECK(std::isfinite(eps) && eps > 0.0F,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "eps must be finite and greater than 0, but got %f.", eps),
             return false);
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
} // namespace

extern "C" aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(
    aclTensor *partialBlock, const aclTensor *delta, const aclTensor *interNumerator, const aclTensor *interMax,
    const aclTensor *interExpSum, const aclTensor *effectiveQuery, float eps, aclTensor *h, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize is nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr.");
    L2_DFX_PHASE_1(aclnnBlockAttnResUpdate,
                   DFX_IN(partialBlock, delta, interNumerator, interMax, interExpSum, effectiveQuery, eps),
                   DFX_OUT(partialBlock, h));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    BlockAttnResUpdateParams params{
        partialBlock, delta, interNumerator, interMax, interExpSum, effectiveQuery, eps, h};
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto opResult = l0op::BlockAttnResUpdate(
        partialBlock, delta, interNumerator, interMax, interExpSum, effectiveQuery, eps, uniqueExecutor.get());
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
