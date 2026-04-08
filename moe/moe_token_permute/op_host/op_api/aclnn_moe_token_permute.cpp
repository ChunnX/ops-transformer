/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe/moe_init_routing_v2/op_host/op_api/moe_init_routing_v2.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMoeTokenPermuteGetWorkspaceSize(
    const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens, bool paddedMode,
    const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeTokenPermute(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(
    const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens, bool paddedMode,
    const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenPermute,
        DFX_IN(tokens, indices, numOutTokens, paddedMode),
        DFX_OUT(permuteTokensOut, sortedIndicesOut));
                
    static bool useMoeInitRoutingV2 = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRoutingV2) {
        return aclnnInnerMoeTokenPermuteGetWorkspaceSize(
            tokens, indices, numOutTokens, paddedMode, permuteTokensOut, sortedIndicesOut, workspaceSize, executor);
    }
    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);

    // 参数检查
    OP_CHECK_NULL(tokens, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(indices, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(permuteTokensOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(sortedIndicesOut, return ACLNN_ERR_PARAM_NULLPTR);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入转换成连续的tensor
    auto tokensContiguous = l0op::Contiguous(tokens, uniqueExecutor.get());
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0接口进行计算
    auto result = l0op::MoeInitRoutingV2(tokensContiguous, indicesContiguous,
        numOutTokens, 0, 0, 0, 0, false,
        permuteTokensOut, sortedIndicesOut,
        nullptr, nullptr, uniqueExecutor.get());

    auto expandedXOut_ = std::get<0>(result);
    auto expandedRowIdxOut_ = std::get<1>(result);
    bool hasNullptr = (expandedXOut_ == nullptr) || (expandedRowIdxOut_ == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);

    // copyout结果，如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyExpandedXOutResult = l0op::ViewCopy(expandedXOut_, permuteTokensOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedXOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyExpandedRowIdxOutResult = l0op::ViewCopy(expandedRowIdxOut_, sortedIndicesOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedRowIdxOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenPermute(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static bool useMoeInitRoutingV2 = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRoutingV2) {
        return aclnnInnerMoeTokenPermute(workspace, workspaceSize, executor, stream);
    }
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif