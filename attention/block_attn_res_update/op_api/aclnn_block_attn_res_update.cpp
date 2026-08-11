/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_block_attn_res_update.h"
#include "block_attn_res_update.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
constexpr size_t ACLNN_MAX_SHAPE_RANK = 8UL;

struct BlockAttnResUpdateParams {
    const aclTensor *partialBlock{nullptr};
    const aclTensor *partialDelta{nullptr};
    const aclTensor *effectiveQuery{nullptr};
    const aclTensor *interMax{nullptr};
    const aclTensor *interExpSum{nullptr};
    const aclTensor *interNumerator{nullptr};
    float epsilon{0.0F};
    const aclTensor *h{nullptr};
    const aclTensor *partialBlocks{nullptr};
};

static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> LOW_PRECISION_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16};

static bool CheckNotNull(const BlockAttnResUpdateParams &params)
{
    OP_CHECK_NULL(params.partialBlock, return false);
    OP_CHECK_NULL(params.partialDelta, return false);
    OP_CHECK_NULL(params.effectiveQuery, return false);
    OP_CHECK_NULL(params.interMax, return false);
    OP_CHECK_NULL(params.interExpSum, return false);
    OP_CHECK_NULL(params.interNumerator, return false);
    OP_CHECK_NULL(params.h, return false);
    OP_CHECK_NULL(params.partialBlocks, return false);
    return true;
}

static bool CheckDtypeValid(const BlockAttnResUpdateParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.partialBlock, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.effectiveQuery, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.interMax, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.interExpSum, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.interNumerator, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.partialBlocks, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.partialDelta, LOW_PRECISION_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.h, LOW_PRECISION_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(params.h, params.partialDelta->GetDataType(), return false);
    return true;
}

static bool IsFormatValid(const aclTensor *tensor)
{
    return !IsPrivateFormat(tensor->GetStorageFormat());
}

static bool CheckFormat(const BlockAttnResUpdateParams &params)
{
    OP_CHECK(IsFormatValid(params.partialBlock) && IsFormatValid(params.partialDelta) &&
                 IsFormatValid(params.effectiveQuery) && IsFormatValid(params.interMax) &&
                 IsFormatValid(params.interExpSum) && IsFormatValid(params.interNumerator) && IsFormatValid(params.h) &&
                 IsFormatValid(params.partialBlocks),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResUpdate does not support private formats."), return false);
    return true;
}

static bool CheckShape(const BlockAttnResUpdateParams &params)
{
    OP_CHECK_MAX_DIM(params.partialBlock, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(params.partialDelta, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(params.effectiveQuery, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(params.interMax, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(params.interExpSum, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(params.interNumerator, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(params.h, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(params.partialBlocks, ACLNN_MAX_SHAPE_RANK, return false);

    OP_CHECK(params.h->GetViewShape() == params.partialDelta->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of h must be the same as partialDelta."), return false);
    OP_CHECK(params.partialBlocks->GetViewShape() == params.partialBlock->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of partialBlocks must be the same as partialBlock."),
             return false);
    return true;
}

static bool CheckContiguous(const BlockAttnResUpdateParams &params)
{
    OP_CHECK(IsContiguous(params.partialBlock),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "partialBlock must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.partialDelta),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "partialDelta must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.effectiveQuery),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "effectiveQuery must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.interMax),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "interMax must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.interExpSum),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "interExpSum must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.interNumerator),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "interNumerator must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.h),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h must be contiguous."), return false);
    OP_CHECK(IsContiguous(params.partialBlocks),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "partialBlocks must be contiguous."), return false);
    return true;
}

static aclnnStatus CheckParams(const BlockAttnResUpdateParams &params)
{
    CHECK_COND(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed.");
    CHECK_COND(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed.");
    CHECK_COND(CheckFormat(params), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed.");
    CHECK_COND(CheckShape(params), ACLNN_ERR_PARAM_INVALID, "CheckShape failed.");
    CHECK_COND(CheckContiguous(params), ACLNN_ERR_PARAM_INVALID, "CheckContiguous failed.");
    return ACLNN_SUCCESS;
}

static bool HasEmptyOutput(const BlockAttnResUpdateParams &params)
{
    return params.h->IsEmpty() && params.partialBlocks->IsEmpty();
}
} // namespace

extern "C" aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(
    const aclTensor *partialBlock, const aclTensor *partialDelta, const aclTensor *effectiveQuery,
    const aclTensor *interMax, const aclTensor *interExpSum, const aclTensor *interNumerator, float epsilon,
    const aclTensor *h, const aclTensor *partialBlocks, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize is nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr.");
    L2_DFX_PHASE_1(aclnnBlockAttnResUpdate,
                   DFX_IN(partialBlock, partialDelta, effectiveQuery, interMax, interExpSum, interNumerator, epsilon),
                   DFX_OUT(h, partialBlocks));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    BlockAttnResUpdateParams params{partialBlock, partialDelta, effectiveQuery, interMax, interExpSum,
                                    interNumerator, epsilon, h, partialBlocks};
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyOutput(params)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto opResult = l0op::BlockAttnResUpdate(
        partialBlock, partialDelta, effectiveQuery, interMax, interExpSum, interNumerator, epsilon,
        uniqueExecutor.get());
    CHECK_RET(opResult[0] != nullptr && opResult[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyH = l0op::ViewCopy(opResult[0], h, uniqueExecutor.get());
    CHECK_RET(viewCopyH != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyPartialBlocks = l0op::ViewCopy(opResult[1], partialBlocks, uniqueExecutor.get());
    CHECK_RET(viewCopyPartialBlocks != nullptr, ACLNN_ERR_INNER_NULLPTR);

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
