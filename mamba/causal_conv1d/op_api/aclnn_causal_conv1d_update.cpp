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

aclnnStatus CausalConv1dUpdateCommonProcess(const aclTensor *x, const aclTensor *weight, aclTensor *convStatesRef,
                                            const aclTensor *biasOptional,
                                            const aclTensor *queryStartLocOptional,
                                            const aclTensor *cacheIndicesOptional,
                                            const aclTensor *numAcceptedTokensOptional,
                                            const aclTensor *blockIdxLastScheduledTokenOptional,
                                            const aclTensor *initialStateIdxOptional,
                                            int64_t activationMode, int64_t nullBlockId, int64_t maxQueryLen,
                                            aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();

    const aclTensor *xFinal = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_COND(xFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous x failed.");

    aclTensor *convStatesFinal = const_cast<aclTensor *>(l0op::Contiguous(convStatesRef, uniqueExecutor.get()));
    CHECK_COND(convStatesFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous convStatesRef failed.");

    weight = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_COND(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous weight failed.");

    if (biasOptional != nullptr) {
        biasOptional = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_COND(biasOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous biasOptional failed.");
    }
    if (cacheIndicesOptional != nullptr) {
        cacheIndicesOptional = l0op::Contiguous(cacheIndicesOptional, uniqueExecutor.get());
        CHECK_COND(cacheIndicesOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous cacheIndicesOptional failed.");
    }
    if (queryStartLocOptional != nullptr) {
        queryStartLocOptional = l0op::Contiguous(queryStartLocOptional, uniqueExecutor.get());
        CHECK_COND(queryStartLocOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous queryStartLocOptional failed.");
    }
    if (numAcceptedTokensOptional != nullptr) {
        numAcceptedTokensOptional = l0op::Contiguous(numAcceptedTokensOptional, uniqueExecutor.get());
        CHECK_COND(numAcceptedTokensOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "Contiguous numAcceptedTokensOptional failed.");
    }

    bool ok = l0op::CausalConv1d(xFinal, weight, convStatesFinal, biasOptional, queryStartLocOptional,
                                 cacheIndicesOptional, nullptr, numAcceptedTokensOptional, activationMode, nullBlockId,
                                 y, uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_TILING_ERROR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

ACLNN_API aclnnStatus aclnnCausalConv1dUpdateGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, aclTensor *convStatesRef, const aclTensor *biasOptional,
    const aclTensor *queryStartLocOptional, const aclTensor *cacheIndicesOptional,
    const aclTensor *numAcceptedTokensOptional,
    const aclTensor *blockIdxLastScheduledTokenOptional,
    const aclTensor *initialStateIdxOptional,
    int64_t activationMode, int64_t nullBlockId, int64_t maxQueryLen,
    aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnCausalConv1dUpdate,
                   DFX_IN(x, weight, convStatesRef, biasOptional, queryStartLocOptional, cacheIndicesOptional,
                          numAcceptedTokensOptional),
                   DFX_OUT(convStatesRef, y));
    return CausalConv1dUpdateCommonProcess(x, weight, convStatesRef, biasOptional,
                                           queryStartLocOptional, cacheIndicesOptional,
                                           numAcceptedTokensOptional,
                                           blockIdxLastScheduledTokenOptional,
                                           initialStateIdxOptional,
                                           activationMode, nullBlockId, maxQueryLen, y, workspaceSize, executor);
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
