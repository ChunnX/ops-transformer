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

constexpr int64_t BLOCK_RES_DIM_NUM = 3;
constexpr int64_t VALID_BLOCKS_DIM_NUM = 1;
constexpr int64_t PSEUDO_QUERY_DIM_NUM = 2;
constexpr int64_t STATS_DIM_NUM = 2;
constexpr int64_t NUMERATOR_DIM_NUM = 3;
constexpr int64_t MIN_HEAD_DIM = 1;
constexpr int64_t MAX_HEAD_DIM = 8192;
constexpr int64_t MIN_BLOCK_NUM = 1;
constexpr int64_t MAX_BLOCK_NUM = 64;

aclnnStatus CheckNotNull(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery,
                         const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum,
                         uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(blockRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(validBlocks != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(pseudoQuery != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(numerator != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(logitMax != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(expSum != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery,
                       const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum)
{
    CHECK_RET(blockRes->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(validBlocks->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(pseudoQuery->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numerator->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(logitMax->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(expSum->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckShape(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery,
                       const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum)
{
    const auto &blockResShape = blockRes->GetViewShape();
    const auto &validShape = validBlocks->GetViewShape();
    const auto &pseudoQueryShape = pseudoQuery->GetViewShape();
    const auto &numeratorShape = numerator->GetViewShape();
    const auto &logitMaxShape = logitMax->GetViewShape();
    const auto &expSumShape = expSum->GetViewShape();

    if (blockResShape.GetDimNum() != BLOCK_RES_DIM_NUM || validShape.GetDimNum() != VALID_BLOCKS_DIM_NUM ||
        pseudoQueryShape.GetDimNum() != PSEUDO_QUERY_DIM_NUM || numeratorShape.GetDimNum() != NUMERATOR_DIM_NUM ||
        logitMaxShape.GetDimNum() != STATS_DIM_NUM || expSumShape.GetDimNum() != STATS_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResPrepare input or output rank is invalid.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t totalT = blockResShape.GetDim(0);
    const int64_t totalN = blockResShape.GetDim(1);
    const int64_t totalD = blockResShape.GetDim(2);
    const int64_t totalS = pseudoQueryShape.GetDim(0);
    if (totalT < 0 || totalS < 0 || totalN < MIN_BLOCK_NUM || totalN > MAX_BLOCK_NUM ||
        totalD < MIN_HEAD_DIM || totalD > MAX_HEAD_DIM || validShape.GetDim(0) != 1 ||
        pseudoQueryShape.GetDim(1) != totalD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResPrepare input shape is invalid.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (numeratorShape.GetDim(0) != totalS || numeratorShape.GetDim(1) != totalT ||
        numeratorShape.GetDim(2) != totalD || logitMaxShape.GetDim(0) != totalS ||
        logitMaxShape.GetDim(1) != totalT || expSumShape.GetDim(0) != totalS || expSumShape.GetDim(1) != totalT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResPrepare output shape does not match the input shape.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery,
                        const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum, float eps,
                        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckNotNull(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum,
                           workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    if (!std::isfinite(eps) || eps <= 0.0F) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "eps must be finite and greater than zero.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckDtype(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    return CheckShape(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum);
}

} // namespace

extern "C" {

aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(
    const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery, aclTensor *numerator,
    aclTensor *logitMax, aclTensor *expSum, float eps, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnBlockAttnResPrepare, DFX_IN(blockRes, validBlocks, pseudoQuery, eps),
                   DFX_OUT(numerator, logitMax, expSum));

    CHECK_RET(CheckNotNull(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum,
                           workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "BlockAttnResPrepare only supports Ascend 950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorImpl = uniqueExecutor.get();

    const aclnnStatus checkRet = CheckParams(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum, eps,
                                             workspaceSize, executor);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    blockRes = l0op::Contiguous(blockRes, executorImpl);
    validBlocks = l0op::Contiguous(validBlocks, executorImpl);
    pseudoQuery = l0op::Contiguous(pseudoQuery, executorImpl);
    CHECK_RET(blockRes != nullptr && validBlocks != nullptr && pseudoQuery != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const auto outputs =
        l0op::BlockAttnResPrepare(blockRes, validBlocks, pseudoQuery, eps, executorImpl);
    CHECK_RET(outputs[0] != nullptr && outputs[1] != nullptr && outputs[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[0], numerator, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[1], logitMax, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[2], expSum, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);

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
