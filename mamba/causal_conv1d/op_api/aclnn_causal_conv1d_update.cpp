/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <tuple>
#include <cstddef>
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "causal_conv1d.h"
#include "aclnn_causal_conv1d_update.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

aclnnStatus CausalConv1dUpdateCommonProcess(const aclTensor *x, aclTensor *convStates, const aclTensor *weight,
                                            const aclTensor *bias, int64_t activationMode,
                                            const aclTensor *cacheIndices, const aclTensor *numAcceptedTokens,
                                            const aclTensor *queryStartLoc, int64_t maxQueryLen, int64_t nullBlockId,
                                            const aclTensor *blockIdxLastScheduledToken,
                                            const aclTensor *initialStateIdx, aclTensor *y, uint64_t *workspaceSize,
                                            aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();

    const aclTensor *xFinal = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_COND(xFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous x failed.");

    aclTensor *convStatesFinal = const_cast<aclTensor *>(l0op::Contiguous(convStates, uniqueExecutor.get()));
    CHECK_COND(convStatesFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous convStates failed.");

    weight = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_COND(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous weight failed.");

    if (bias != nullptr) {
        bias = l0op::Contiguous(bias, uniqueExecutor.get());
        CHECK_COND(bias != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous bias failed.");
    }
    if (cacheIndices != nullptr) {
        cacheIndices = l0op::Contiguous(cacheIndices, uniqueExecutor.get());
        CHECK_COND(cacheIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous cacheIndices failed.");
    }
    if (queryStartLoc != nullptr) {
        queryStartLoc = l0op::Contiguous(queryStartLoc, uniqueExecutor.get());
        CHECK_COND(queryStartLoc != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous queryStartLoc failed.");
    }
    if (numAcceptedTokens != nullptr) {
        numAcceptedTokens = l0op::Contiguous(numAcceptedTokens, uniqueExecutor.get());
        CHECK_COND(numAcceptedTokens != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous numAcceptedTokens failed.");
    }

    bool ok = l0op::CausalConv1d(xFinal, weight, convStatesFinal, bias, queryStartLoc, cacheIndices, nullptr,
                                 numAcceptedTokens, activationMode, nullBlockId, y, uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_TILING_ERROR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

ACLNN_API aclnnStatus aclnnCausalConv1dUpdateGetWorkspaceSize(
    const aclTensor *x, aclTensor *convStates, const aclTensor *weight, const aclTensor *bias, int64_t activationMode,
    const aclTensor *cacheIndices, const aclTensor *numAcceptedTokens, const aclTensor *queryStartLoc,
    int64_t maxQueryLen, int64_t nullBlockId, const aclTensor *blockIdxLastScheduledToken,
    const aclTensor *initialStateIdx, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnCausalConv1dUpdate,
                   DFX_IN(x, convStates, weight, bias, cacheIndices, numAcceptedTokens, queryStartLoc),
                   DFX_OUT(convStates, y));
    return CausalConv1dUpdateCommonProcess(x, convStates, weight, bias, activationMode, cacheIndices, numAcceptedTokens,
                                           queryStartLoc, maxQueryLen, nullBlockId, blockIdxLastScheduledToken,
                                           initialStateIdx, y, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnCausalConv1dUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCausalConv1dUpdate);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
