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
#include "aclnn_causal_conv1d.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

aclnnStatus CausalConv1dCommonProcess(const aclTensor *x, const aclTensor *weight, aclTensor *convStates,
                                      const aclTensor *bias, const aclTensor *queryStartLoc,
                                      const aclTensor *cacheIndices, const aclTensor *initialStateMode,
                                      const aclTensor *numAcceptedTokens, int64_t activationMode, int64_t nullBlockId,
                                      aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
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
    if (initialStateMode != nullptr) {
        initialStateMode = l0op::Contiguous(initialStateMode, uniqueExecutor.get());
        CHECK_COND(initialStateMode != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous initialStateMode failed.");
    }
    if (numAcceptedTokens != nullptr) {
        numAcceptedTokens = l0op::Contiguous(numAcceptedTokens, uniqueExecutor.get());
        CHECK_COND(numAcceptedTokens != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous numAcceptedTokens failed.");
    }

    bool ok = l0op::CausalConv1d(xFinal, weight, convStatesFinal, bias, queryStartLoc, cacheIndices, initialStateMode,
                                 numAcceptedTokens, activationMode, nullBlockId, y, uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_TILING_ERROR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

ACLNN_API aclnnStatus aclnnCausalConv1dGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
                                                        aclTensor *convStates, const aclTensor *bias,
                                                        const aclTensor *queryStartLoc, const aclTensor *cacheIndices,
                                                        const aclTensor *initialStateMode,
                                                        const aclTensor *numAcceptedTokens, int64_t activationMode,
                                                        int64_t nullBlockId, aclTensor *y, uint64_t *workspaceSize,
                                                        aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnCausalConv1d,
        DFX_IN(x, weight, convStates, bias, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens),
        DFX_OUT(convStates, y));
    return CausalConv1dCommonProcess(x, weight, convStates, bias, queryStartLoc, cacheIndices, initialStateMode,
                                     numAcceptedTokens, activationMode, nullBlockId, y, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnCausalConv1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCausalConv1d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
