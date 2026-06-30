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
 * \file aclnn_lightning_indexer_kl_loss.cpp
 * \brief
 */

#include "aclnn_lightning_indexer_kl_loss.h"
#include "lightning_indexer_kl_loss.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct LightningIndexerKLLossParams {
    const aclTensor *target_score = nullptr;
    const aclTensor *index_probs = nullptr;
    double eps;
    const aclTensor *loss = nullptr;
};

static aclnnStatus CheckParams(const LightningIndexerKLLossParams &params)
{
    CHECK_COND(params.target_score != nullptr, ACLNN_ERR_PARAM_NULLPTR, "target_score must not be nullptr.");
    CHECK_COND(params.index_probs != nullptr, ACLNN_ERR_PARAM_NULLPTR, "index_probs must not be nullptr.");
    CHECK_COND(params.loss != nullptr, ACLNN_ERR_PARAM_NULLPTR, "loss must not be nullptr.");
    return ACLNN_SUCCESS;
}

static bool CheckDataType(const aclTensor *target_score, const aclTensor *index_probs)
{
    const DataType scoreDtype = target_score->GetDataType();
    const DataType probDtype = index_probs->GetDataType();

    if (scoreDtype != probDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "target_score and index_probs must have the same dtype.");
        return false;
    }

    static const std::unordered_map<DataType, std::vector<DataType>> validDtype = {
        {DataType::DT_FLOAT, {DataType::DT_FLOAT}},
        {DataType::DT_FLOAT16, {DataType::DT_FLOAT16}},
        {DataType::DT_BF16, {DataType::DT_BF16}},
    };

    auto iter = validDtype.find(scoreDtype);
    if (iter == validDtype.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported target_score datatype %d.", static_cast<int>(scoreDtype));
        return false;
    }

    return true;
}

static aclnnStatus ValidateParams(const aclTensor *target_score, const aclTensor *index_probs, double eps)
{
    CHECK_RET(target_score != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(index_probs != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_COND(eps > 0, ACLNN_ERR_PARAM_INVALID, "eps should be greater than 0.");

    if (!CheckDataType(target_score, index_probs)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLightningIndexerKLLossGetWorkspaceSize(
    const aclTensor *target_score, const aclTensor *index_probs,
    double eps, const aclTensor *loss,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus validateRet = ValidateParams(target_score, index_probs, eps);
    if (validateRet != ACLNN_SUCCESS) {
        return validateRet;
    }

    L2_DFX_PHASE_1(aclnnLightningIndexerKLLoss,
                   DFX_IN(target_score, index_probs, eps),
                   DFX_OUT(loss));
    LightningIndexerKLLossParams params{target_score, index_probs, eps, loss};

    aclnnStatus ret = CheckParams(params);
    CHECK_COND(ret == ACLNN_SUCCESS, ret, "aclnnLightningIndexerKLLossGetWorkspaceSize checkParams failed.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (target_score->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Make contiguous
    auto targetScoreContiguous = l0op::Contiguous(params.target_score, uniqueExecutor.get());
    CHECK_RET(targetScoreContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indexProbsContiguous = l0op::Contiguous(params.index_probs, uniqueExecutor.get());
    CHECK_RET(indexProbsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // call l0 interface
    auto result = l0op::LightningIndexerKLLoss(
        targetScoreContiguous, indexProbsContiguous, params.eps, uniqueExecutor.get());

    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(result[0], loss, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLightningIndexerKLLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLightningIndexerKLLoss);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
