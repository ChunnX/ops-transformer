/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_token_permute.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "moe_token_permute.h"

using namespace op;

namespace MoeTokenPermuteCheck {

static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_X = {DataType::DT_FLOAT16, DataType::DT_BF16,
                                                                             DataType::DT_FLOAT, DataType::DT_INT8};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE = {DataType::DT_INT32,
                                                                                              DataType::DT_INT64};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor *tokens, const aclTensor *indices, const aclTensor *permuteTokensOut,
                                const aclTensor *sortedIndicesOut)
{
    OP_CHECK_NULL(tokens, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(permuteTokensOut, return false);
    OP_CHECK_NULL(sortedIndicesOut, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *tokens, const aclTensor *indices, const aclTensor *permuteTokensOut,
                                   const aclTensor *sortedIndicesOut)
{
    if (tokens != nullptr && tokens->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(tokens, MOE_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (indices != nullptr && indices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(indices, MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE, return false);
    }
    if (permuteTokensOut != nullptr && permuteTokensOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokensOut, MOE_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(tokens, permuteTokensOut, return false);
    }
    if (sortedIndicesOut != nullptr && sortedIndicesOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndicesOut, MOE_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *tokens, const aclTensor *indices, const aclTensor *permuteTokensOut,
                               const aclTensor *sortedIndicesOut)
{
    CHECK_RET(CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(tokens, indices, permuteTokensOut, sortedIndicesOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeTokenPermuteCheck

static aclnnStatus BuildMoeTokenPermuteExecutor(const aclTensor *tokens, const aclTensor *indices, int64_t numOutTokens,
                                                bool paddedMode, const aclTensor *permuteTokensOut,
                                                const aclTensor *sortedIndicesOut, uint64_t *workspaceSize,
                                                aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto tokensContiguous = l0op::Contiguous(tokens, uniqueExecutor.get());
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::MoeTokenPermute(tokensContiguous, indicesContiguous, numOutTokens, paddedMode, permuteTokensOut,
                                        sortedIndicesOut, uniqueExecutor.get());
    auto permuteTokens = std::get<0>(result);
    auto sortedIndices = std::get<1>(result);
    CHECK_RET(permuteTokens != nullptr && sortedIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(permuteTokens, permuteTokensOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(sortedIndices, sortedIndicesOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(const aclTensor *tokens, const aclTensor *indices,
                                                 int64_t numOutTokens, bool paddedMode,
                                                 const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenPermute, DFX_IN(tokens, indices, numOutTokens, paddedMode),
                   DFX_OUT(permuteTokensOut, sortedIndicesOut));

    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);

    aclnnStatus ret = MoeTokenPermuteCheck::CheckParams(tokens, indices, permuteTokensOut, sortedIndicesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return BuildMoeTokenPermuteExecutor(tokens, indices, numOutTokens, paddedMode, permuteTokensOut, sortedIndicesOut,
                                        workspaceSize, executor);
}

aclnnStatus aclnnMoeTokenPermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeTokenPermute);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
