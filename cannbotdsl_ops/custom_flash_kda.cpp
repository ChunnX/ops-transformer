/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#include "aclnn_custom_flash_kda.h"

#include <cmath>
#include <limits>

#include "flash_kda_l0.h"
#include "opdev/framework_op.h"
#include "opdev/make_op_executor.h"

namespace {

bool CheckScalarContract(double scale, double lowerBound)
{
    return std::isfinite(scale) && std::isfinite(lowerBound) &&
           lowerBound >= -5.0 && lowerBound <= 0.0 &&
           scale <= std::numeric_limits<float>::max() &&
           scale >= -std::numeric_limits<float>::max() &&
           lowerBound <= std::numeric_limits<float>::max() &&
           lowerBound >= -std::numeric_limits<float>::max();
}

}  // namespace

extern "C" aclnnStatus aclnnCustomFlashKdaGetWorkspaceSize(
    const aclTensor* k, const aclTensor* v, const aclTensor* q,
    const aclTensor* beta, const aclTensor* g, const aclTensor* aLog,
    const aclTensor* dtBias, const aclTensor* initialState, double scale,
    double lowerBound, const aclTensor* out, const aclTensor* finalState,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (workspaceSize != nullptr) {
        *workspaceSize = 0U;
    }
    if (executor != nullptr) {
        *executor = nullptr;
    }
    if (workspaceSize == nullptr || executor == nullptr ||
        !CheckScalarContract(scale, lowerBound)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }

    // This is the L2 assembly boundary. Only the public L0 contract is visible
    // here; Provider selection, hidden resources and physical binding remain
    // encapsulated by l0op::FlashKda.
    const aclnnStatus status = l0op::FlashKda(
        k, v, q, beta, g, aLog, dtBias, initialState,
        static_cast<float>(scale), static_cast<float>(lowerBound), out,
        finalState, uniqueExecutor.get());
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnCustomFlashKda(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
