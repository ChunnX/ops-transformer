/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_block_attn_res_prepare.cpp
 * \brief ACLNN implementation for BlockAttnResPrepare.
 */

#include "aclnn_block_attn_res_prepare.h"
#include "block_attn_res_prepare.h"

#include <cmath>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

constexpr int64_t BLOCK_RESIDUAL_DIM_NUM = 3;
constexpr int64_t EFFECTIVE_QUERY_DIM_NUM = 2;
constexpr int64_t VALID_BLOCKS_DIM_NUM = 1;
constexpr int64_t STATS_DIM_NUM = 2;
constexpr int64_t WEIGHTED_OUTPUT_DIM_NUM = 3;
constexpr int64_t MAX_SOFTMAX_N = 64;

bool HasPositiveDims(const aclTensor *tensor)
{
    const auto &shape = tensor->GetViewShape();
    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
        if (shape.GetDim(index) <= 0) {
            return false;
        }
    }
    return true;
}

aclnnStatus CheckNotNull(const aclTensor *blockResidual, const aclTensor *effectiveQuery, const aclTensor *validBlocks,
                         const aclTensor *softmaxMax, const aclTensor *weightedOutput, const aclTensor *softmaxSum,
                         uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(blockResidual != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(effectiveQuery != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(validBlocks != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxMax != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(weightedOutput != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxSum != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const aclTensor *blockResidual, const aclTensor *effectiveQuery, const aclTensor *validBlocks,
                       const aclTensor *softmaxMax, const aclTensor *weightedOutput, const aclTensor *softmaxSum)
{
    CHECK_RET(blockResidual->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(effectiveQuery->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(validBlocks->GetDataType() == DataType::DT_INT32, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(softmaxMax->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(weightedOutput->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(softmaxSum->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckShape(const aclTensor *blockResidual, const aclTensor *effectiveQuery, const aclTensor *validBlocks,
                       const aclTensor *softmaxMax, const aclTensor *weightedOutput, const aclTensor *softmaxSum)
{
    const auto &vShape = blockResidual->GetViewShape();
    const auto &qShape = effectiveQuery->GetViewShape();
    const auto &validShape = validBlocks->GetViewShape();
    const auto &maxShape = softmaxMax->GetViewShape();
    const auto &outputShape = weightedOutput->GetViewShape();
    const auto &sumShape = softmaxSum->GetViewShape();

    if (vShape.GetDimNum() != BLOCK_RESIDUAL_DIM_NUM || qShape.GetDimNum() != EFFECTIVE_QUERY_DIM_NUM ||
        validShape.GetDimNum() != VALID_BLOCKS_DIM_NUM || maxShape.GetDimNum() != STATS_DIM_NUM ||
        outputShape.GetDimNum() != WEIGHTED_OUTPUT_DIM_NUM || sumShape.GetDimNum() != STATS_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResPrepare input or output rank is invalid.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!HasPositiveDims(blockResidual) || !HasPositiveDims(effectiveQuery) || vShape.GetDim(1) > MAX_SOFTMAX_N ||
        validShape.GetDim(0) != 1 || qShape.GetDim(1) != vShape.GetDim(2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResPrepare input shape is invalid.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t totalT = vShape.GetDim(0);
    const int64_t totalS = qShape.GetDim(0);
    const int64_t totalD = vShape.GetDim(2);
    if (maxShape.GetDim(0) != totalS || maxShape.GetDim(1) != totalT || sumShape.GetDim(0) != totalS ||
        sumShape.GetDim(1) != totalT || outputShape.GetDim(0) != totalS || outputShape.GetDim(1) != totalT ||
        outputShape.GetDim(2) != totalD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResPrepare output shape does not match the input shape.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *blockResidual, const aclTensor *effectiveQuery, const aclTensor *validBlocks,
                        float epsilon, const aclTensor *softmaxMax, const aclTensor *weightedOutput,
                        const aclTensor *softmaxSum, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckNotNull(blockResidual, effectiveQuery, validBlocks, softmaxMax, weightedOutput, softmaxSum,
                           workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "epsilon must be finite and greater than zero.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckDtype(blockResidual, effectiveQuery, validBlocks, softmaxMax, weightedOutput, softmaxSum) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    return CheckShape(blockResidual, effectiveQuery, validBlocks, softmaxMax, weightedOutput, softmaxSum);
}

} // namespace

extern "C" {

aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(
    const aclTensor *blockResidual, const aclTensor *effectiveQuery, const aclTensor *validBlocks, float epsilon,
    aclTensor *softmaxMax, aclTensor *weightedOutput, aclTensor *softmaxSum, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnBlockAttnResPrepare, DFX_IN(blockResidual, effectiveQuery, validBlocks, epsilon),
                   DFX_OUT(softmaxMax, weightedOutput, softmaxSum));

    CHECK_RET(CheckNotNull(blockResidual, effectiveQuery, validBlocks, softmaxMax, weightedOutput, softmaxSum,
                           workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "BlockAttnResPrepare only supports Ascend 950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorImpl = uniqueExecutor.get();

    const aclnnStatus checkRet = CheckParams(blockResidual, effectiveQuery, validBlocks, epsilon, softmaxMax,
                                             weightedOutput, softmaxSum, workspaceSize, executor);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    blockResidual = l0op::Contiguous(blockResidual, executorImpl);
    effectiveQuery = l0op::Contiguous(effectiveQuery, executorImpl);
    validBlocks = l0op::Contiguous(validBlocks, executorImpl);
    CHECK_RET(blockResidual != nullptr && effectiveQuery != nullptr && validBlocks != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const auto outputs =
        l0op::BlockAttnResPrepare(blockResidual, effectiveQuery, validBlocks, epsilon, executorImpl);
    CHECK_RET(outputs[0] != nullptr && outputs[1] != nullptr && outputs[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[0], softmaxMax, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[1], weightedOutput, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[2], softmaxSum, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBlockAttnResPrepare(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockAttnResPrepare);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // extern "C"
