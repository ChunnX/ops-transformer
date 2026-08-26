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
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_moe_token_unpermute_grad.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/cast.h"
#include "level0/gather_v2.h"
#include "level0/mul.h"
#include "level0/reduce_sum_op.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"

using namespace op;

namespace MoeTokenUnpermuteGradCheck {

static const std::initializer_list<op::DataType> MOE_GRAD_DTYPE_SUPPORT_LIST_X = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices,
                                const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut,
                                const aclTensor *permuteTokens)
{
    OP_CHECK_NULL(permuteTokens, return false);
    OP_CHECK_NULL(unpermutedTokensGrad, return false);
    OP_CHECK_NULL(sortedIndices, return false);
    OP_CHECK_NULL(permutedTokensGradOut, return false);
    OP_CHECK_NULL(probsGradOut, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices,
                                   const aclTensor *permuteTokens, const aclTensor *probsOptional,
                                   const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut)
{
    if (unpermutedTokensGrad != nullptr && unpermutedTokensGrad->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(unpermutedTokensGrad, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (sortedIndices != nullptr && sortedIndices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndices, MOE_GRAD_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (permuteTokens != nullptr && permuteTokens->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokens, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(unpermutedTokensGrad, permuteTokens, return false);
    }
    if (probsOptional != nullptr && probsOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(probsOptional, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (permutedTokensGradOut != nullptr && permutedTokensGradOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permutedTokensGradOut, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(unpermutedTokensGrad, permutedTokensGradOut, return false);
    }
    if (probsGradOut != nullptr && probsGradOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(probsGradOut, MOE_GRAD_DTYPE_SUPPORT_LIST_X, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *unpermutedTokensGrad, const aclTensor *sortedIndices,
                               const aclTensor *permuteTokens, const aclTensor *probsOptional,
                               const aclTensor *permutedTokensGradOut, const aclTensor *probsGradOut)
{
    CHECK_RET(CheckNotNull(unpermutedTokensGrad, sortedIndices, permutedTokensGradOut, probsGradOut, permuteTokens),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(unpermutedTokensGrad, sortedIndices, permuteTokens, probsOptional, permutedTokensGradOut,
                              probsGradOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeTokenUnpermuteGradCheck

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(
    const aclTensor* permuteTokensOptional, const aclTensor* unpermutedTokensGrad, const aclTensor* sortedIndices,
    const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional,
    aclTensor* permutedTokensGradOut, aclTensor* probsGradOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenUnpermuteGrad,
        DFX_IN(permuteTokensOptional, unpermutedTokensGrad, sortedIndices,
            probsOptional, paddedMode, restoreShapeOptional),
        DFX_OUT(permutedTokensGradOut, probsGradOut));

    // ascend950: use l0op basic ops (GatherV2 + Mul + ReduceSum)
    // ascend910b/910_93: use native kernel via aclnnInner
    bool useL0op = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useL0op) {
        return aclnnInnerMoeTokenUnpermuteGradGetWorkspaceSize(
            permuteTokensOptional, unpermutedTokensGrad, sortedIndices, probsOptional, paddedMode, restoreShapeOptional,
            permutedTokensGradOut, probsGradOut, workspaceSize, executor);
    }
    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);

    aclnnStatus ret = MoeTokenUnpermuteGradCheck::CheckParams(unpermutedTokensGrad, sortedIndices, permuteTokensOptional,
                                                              probsOptional, permutedTokensGradOut, probsGradOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto* exec = uniqueExecutor.get();

    auto gradYContig = l0op::Contiguous(unpermutedTokensGrad, exec);
    CHECK_RET(gradYContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sortedIdxContig = l0op::Contiguous(sortedIndices, exec);
    CHECK_RET(sortedIdxContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto permTokensContig = l0op::Contiguous(permuteTokensOptional, exec);
    CHECK_RET(permTokensContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* probsContig = nullptr;
    if (probsOptional != nullptr) {
        probsContig = l0op::Contiguous(probsOptional, exec);
        CHECK_RET(probsContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Cast to FP32 for precision (all before computation)
    auto origDtype = unpermutedTokensGrad->GetDataType();
    auto gradYFp32 = l0op::Cast(gradYContig, ge::DT_FLOAT, exec);
    CHECK_RET(gradYFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto permTokensFp32 = l0op::Cast(permTokensContig, ge::DT_FLOAT, exec);
    CHECK_RET(permTokensFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* probsFp32 = nullptr;
    if (probsContig != nullptr) {
        probsFp32 = l0op::Cast(probsContig, ge::DT_FLOAT, exec);
        CHECK_RET(probsFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Gather: gathered_grad_y = grad_y[sorted_indices]
    auto gatheredGradY = l0op::GatherV2(gradYFp32, 0, sortedIdxContig, exec);
    CHECK_RET(gatheredGradY != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* gradXResult = nullptr;
    const aclTensor* gradProbsResult = nullptr;

    if (probsFp32 != nullptr) {
        int64_t numExpertTokens = static_cast<int64_t>(sortedIdxContig->GetViewShape().GetDim(0));
        auto probsFlat = l0op::Reshape(probsFp32, exec->AllocIntArray(&numExpertTokens, 1), exec);
        CHECK_RET(probsFlat != nullptr, ACLNN_ERR_INNER_NULLPTR);
        int64_t bcShape[2] = {numExpertTokens, 1};
        auto probsBc = l0op::Reshape(probsFlat, exec->AllocIntArray(bcShape, 2), exec);
        CHECK_RET(probsBc != nullptr, ACLNN_ERR_INNER_NULLPTR);

        gradXResult = l0op::Mul(gatheredGradY, probsBc, exec);
        CHECK_RET(gradXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto prod = l0op::Mul(gatheredGradY, permTokensFp32, exec);
        CHECK_RET(prod != nullptr, ACLNN_ERR_INNER_NULLPTR);
        int64_t reduceAxis = -1;
        gradProbsResult = l0op::ReduceSumOp(prod, exec->AllocIntArray(&reduceAxis, 1), false, exec);
        CHECK_RET(gradProbsResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        int64_t probsShapeArr[2] = {static_cast<int64_t>(probsContig->GetViewShape().GetDim(0)),
                                     static_cast<int64_t>(probsContig->GetViewShape().GetDim(1))};
        gradProbsResult = l0op::Reshape(gradProbsResult, exec->AllocIntArray(probsShapeArr, 2), exec);
        CHECK_RET(gradProbsResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        gradXResult = gatheredGradY;
    }

    // Cast back to original dtype
    gradXResult = l0op::Cast(gradXResult, origDtype, exec);
    CHECK_RET(gradXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (gradProbsResult != nullptr) {
        auto probsDtype = probsOptional->GetDataType();
        gradProbsResult = l0op::Cast(gradProbsResult, probsDtype, exec);
        CHECK_RET(gradProbsResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // ViewCopy to output
    auto viewCopyGradX = l0op::ViewCopy(gradXResult, permutedTokensGradOut, exec);
    CHECK_RET(viewCopyGradX != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (gradProbsResult != nullptr) {
        auto viewCopyGradProbs = l0op::ViewCopy(gradProbsResult, probsGradOut, exec);
        CHECK_RET(viewCopyGradProbs != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenUnpermuteGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static bool useL0op = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useL0op) {
        return aclnnInnerMoeTokenUnpermuteGrad(workspace, workspaceSize, executor, stream);
    }

    L2_DFX_PHASE_2(aclnnMoeTokenUnpermuteGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif