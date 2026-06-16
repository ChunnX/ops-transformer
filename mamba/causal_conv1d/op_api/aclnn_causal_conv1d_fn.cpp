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
#include "aclnn_causal_conv1d_fn.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

aclnnStatus CausalConv1dFnCommonProcess(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
                                        aclTensor *convStates, const aclTensor *queryStartLoc,
                                        const aclTensor *cacheIndices, const aclTensor *initialStateMode,
                                        int64_t activationMode, int64_t padSlotId, int64_t nullBlockId,
                                        const aclTensor *blockIdxFirstScheduledToken,
                                        const aclTensor *blockIdxLastScheduledToken, const aclTensor *initialStateIdx,
                                        const aclTensor *numComputedTokens, int64_t blockSizeToAlign, aclTensor *y,
                                        uint64_t *workspaceSize, aclOpExecutor **executor)
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

    bool ok = l0op::CausalConv1d(xFinal, weight, convStatesFinal, bias, queryStartLoc, cacheIndices, initialStateMode,
                                 nullptr, activationMode, nullBlockId, y, uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_TILING_ERROR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

ACLNN_API aclnnStatus aclnnCausalConv1dFnGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *bias, aclTensor *convStates,
    const aclTensor *queryStartLoc, const aclTensor *cacheIndices, const aclTensor *initialStateMode,
    int64_t activationMode, int64_t padSlotId, int64_t nullBlockId, const aclTensor *blockIdxFirstScheduledToken,
    const aclTensor *blockIdxLastScheduledToken, const aclTensor *initialStateIdx, const aclTensor *numComputedTokens,
    int64_t blockSizeToAlign, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnCausalConv1dFn,
                   DFX_IN(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode),
                   DFX_OUT(convStates, y));
    return CausalConv1dFnCommonProcess(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode,
                                       activationMode, padSlotId, nullBlockId, blockIdxFirstScheduledToken,
                                       blockIdxLastScheduledToken, initialStateIdx, numComputedTokens, blockSizeToAlign,
                                       y, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnCausalConv1dFn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCausalConv1dFn);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
