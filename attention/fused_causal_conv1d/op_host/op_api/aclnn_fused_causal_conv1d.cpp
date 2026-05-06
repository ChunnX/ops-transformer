/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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
#include "fused_causal_conv1d.h"
#include "aclnn_fused_causal_conv1d.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

aclnnStatus FusedCausalConv1dCommonProcess(const aclTensor *x, const aclTensor *weight, aclTensor *convStates,
                                      const aclTensor *queryStartLoc, const aclTensor *cacheIndices,
                                      const aclTensor *initialStateMode, const aclTensor *bias,
                                      const aclTensor *numAcceptedTokens, int64_t activationMode, int64_t padSlotId,
                                      int64_t runMode, int64_t residualConnection, aclTensor *y,
                                      uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();

    // Handle non-contiguous x input via CreateView (dual shape descriptor, zero-copy).
    // For contiguous tensors, view shape == storage shape, so this is a no-op.
    const aclTensor *xFinal =
        uniqueExecutor->CreateView(x, x->GetViewShape(), x->GetStorageShape(), x->GetViewStrides(), x->GetViewOffset());
    CHECK_COND(xFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "CreateView for x failed.");

    aclTensor *convStatesFinal =
        uniqueExecutor->CreateView(convStates, convStates->GetViewShape(), convStates->GetStorageShape(),
                                   convStates->GetViewStrides(), convStates->GetViewOffset());
    CHECK_COND(convStatesFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "CreateView for convStatesFinal failed.");

    weight = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_COND(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous weight failed.");

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
    if (bias != nullptr) {
        bias = l0op::Contiguous(bias, uniqueExecutor.get());
        CHECK_COND(bias != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous bias failed.");
    }
    if (numAcceptedTokens != nullptr) {
        numAcceptedTokens = l0op::Contiguous(numAcceptedTokens, uniqueExecutor.get());
        CHECK_COND(numAcceptedTokens != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous numAcceptedTokens failed.");
    }

    // convStates is an in-place update: the same tensor serves as both input and
    // output. y is always contiguous. Both are passed directly to l0op.
    bool ok = l0op::FusedCausalConv1d(xFinal, weight, convStatesFinal, queryStartLoc, cacheIndices, initialStateMode, bias,
                                 numAcceptedTokens, activationMode, padSlotId, runMode, residualConnection, y,
                                 uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_TILING_ERROR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

ACLNN_API aclnnStatus aclnnFusedCausalConv1dGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, aclTensor *convStates, const aclTensor *queryStartLoc,
    const aclTensor *cacheIndices, const aclTensor *initialStateMode, const aclTensor *bias,
    const aclTensor *numAcceptedTokens, int64_t activationMode, int64_t padSlotId, int64_t runMode,
    int64_t residualConnection, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnFusedCausalConv1d,
        DFX_IN(x, weight, convStates, queryStartLoc, cacheIndices, initialStateMode, bias, numAcceptedTokens),
        DFX_OUT(y, convStates));
    return FusedCausalConv1dCommonProcess(x, weight, convStates, queryStartLoc, cacheIndices, initialStateMode, bias,
                                     numAcceptedTokens, activationMode, padSlotId, runMode, residualConnection, y,
                                     workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnFusedCausalConv1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedCausalConv1d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
