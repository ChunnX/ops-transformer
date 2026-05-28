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
 * \file aclnn_moe_token_unpermute_grad.cpp
 * \brief
 */
#include "aclnn_moe_token_unpermute_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_moe_token_unpermute_grad.h"  // 该文件为自动生成，在build/autogen/inner路径下
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe/moe_finalize_routing_v2_grad/op_host/op_api/moe_finalize_routing_v2_grad.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(
    const aclTensor* permuteTokens, const aclTensor* unpermutedTokensGrad, const aclTensor* sortedIndices,
    const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional,
    aclTensor* permutedTokensGradOut, aclTensor* probsGradOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenUnpermuteGrad,
        DFX_IN(permuteTokens, unpermutedTokensGrad, sortedIndices,
            probsOptional, paddedMode, restoreShapeOptional),
        DFX_OUT(permutedTokensGradOut, probsGradOut));

    static bool useMoeFinalizeRoutingV2Grad = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeFinalizeRoutingV2Grad) {
        return aclnnInnerMoeTokenUnpermuteGradGetWorkspaceSize(
            permuteTokens, unpermutedTokensGrad, sortedIndices, probsOptional, paddedMode, restoreShapeOptional,
            permutedTokensGradOut, probsGradOut, workspaceSize, executor);
    }
    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);

    // 参数检查
    OP_CHECK_NULL(permuteTokens, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(unpermutedTokensGrad, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(sortedIndices, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(permutedTokensGradOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(probsGradOut, return ACLNN_ERR_PARAM_NULLPTR);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入转换成连续的tensor
    auto unpermutedTokensGradContiguous = l0op::Contiguous(unpermutedTokensGrad, uniqueExecutor.get());
    CHECK_RET(unpermutedTokensGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sortedIndicesContiguous = l0op::Contiguous(sortedIndices, uniqueExecutor.get());
    CHECK_RET(sortedIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* permuteTokensContiguous = l0op::Contiguous(permuteTokens, uniqueExecutor.get());
    CHECK_RET(permuteTokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    const aclTensor* probsContiguous = nullptr;
    if (probsOptional != nullptr) {
        probsContiguous = l0op::Contiguous(probsOptional, uniqueExecutor.get());
        CHECK_RET(probsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 获取activeNum
    auto permuteTokensShape = permuteTokens->GetViewShape();
    CHECK_RET(permuteTokensShape.GetDimNum() > 0, ACLNN_ERR_PARAM_INVALID);
    int64_t activeNum = permuteTokensShape.GetDim(0);

    // 调用l0接口进行计算
    auto result = l0op::MoeFinalizeRoutingV2Grad(unpermutedTokensGradContiguous, sortedIndicesContiguous,
        permuteTokensContiguous, probsContiguous, nullptr, nullptr,
        0, activeNum, 0, 0, permutedTokensGradOut,
        probsGradOut, uniqueExecutor.get());
    auto [gradExpandedXOut_, gradScalesOut_] = result;
    bool hasNullptr = (gradExpandedXOut_ == nullptr) || (gradScalesOut_ == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);

    // copyout结果，如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyGradExpandedXOutResult = l0op::ViewCopy(gradExpandedXOut_,
        permutedTokensGradOut, uniqueExecutor.get());
    CHECK_RET(viewCopyGradExpandedXOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyGradScalesOutResult = l0op::ViewCopy(gradScalesOut_, probsGradOut, uniqueExecutor.get());
    CHECK_RET(viewCopyGradScalesOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenUnpermuteGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static bool useMoeFinalizeRoutingV2Grad = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeFinalizeRoutingV2Grad) {
        return aclnnInnerMoeTokenUnpermuteGrad(workspace, workspaceSize, executor, stream);
    }

    L2_DFX_PHASE_2(aclnnMoeTokenUnpermuteGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif