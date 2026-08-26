/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_token_unpermute_with_ep.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "moe_token_unpermute_with_ep.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace {
constexpr uint64_t RANK_ONE = 1;
constexpr uint64_t RANK_TWO = 2;
constexpr size_t RANGE_SIZE = 2;
constexpr int64_t MAX_TOP_K = 512;

const std::initializer_list<DataType> kFloatTypes = {DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};

bool IsNd(const aclTensor *tensor)
{
    return tensor != nullptr && tensor->GetStorageFormat() == Format::FORMAT_ND &&
           tensor->GetViewFormat() == Format::FORMAT_ND;
}

bool CheckTensorContract(const aclTensor *permutedTokens, const aclTensor *sortedIndices,
                         const aclTensor *probsOptional, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(permutedTokens, kFloatTypes, return false);
    OP_CHECK_DTYPE_NOT_MATCH(sortedIndices, DataType::DT_INT32, return false);
    if (probsOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(probsOptional, kFloatTypes, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, kFloatTypes, return false);
    OP_CHECK_DTYPE_NOT_SAME(permutedTokens, out, return false);

    OP_CHECK(IsNd(permutedTokens), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "permutedTokens only supports ND."), return false);
    OP_CHECK(IsNd(sortedIndices), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sortedIndices only supports ND."), return false);
    OP_CHECK(probsOptional == nullptr || IsNd(probsOptional),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "probsOptional only supports ND."), return false);
    OP_CHECK(IsNd(out), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out only supports ND."), return false);

    const Shape &tokensShape = permutedTokens->GetViewShape();
    const Shape &indicesShape = sortedIndices->GetViewShape();
    const Shape &outShape = out->GetViewShape();
    OP_CHECK(tokensShape.GetDimNum() == RANK_TWO, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "permutedTokens must be rank 2."),
             return false);
    OP_CHECK(indicesShape.GetDimNum() == RANK_ONE, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sortedIndices must be rank 1."),
             return false);
    OP_CHECK(outShape.GetDimNum() == RANK_TWO, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out must be rank 2."), return false);
    OP_CHECK(tokensShape.GetShapeSize() > 0 && indicesShape.GetShapeSize() > 0 && outShape.GetShapeSize() > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "empty tensors are not supported."), return false);
    if (probsOptional != nullptr) {
        const Shape &probsShape = probsOptional->GetViewShape();
        OP_CHECK(probsShape.GetDimNum() == RANK_TWO, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "probsOptional must be rank 2."),
                 return false);
        OP_CHECK(probsShape.GetShapeSize() > 0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "empty probsOptional is not supported."), return false);
    }
    return true;
}

bool CheckShapeAndAttrs(const aclTensor *permutedTokens, const aclTensor *sortedIndices, const aclTensor *probsOptional,
                        int64_t numTopk, const aclIntArray *rangeOptional, bool paddedMode,
                        const aclIntArray *restoreShapeOptional, const aclTensor *out)
{
    OP_CHECK(!paddedMode, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "paddedMode currently only supports false."), return false);
    OP_CHECK(restoreShapeOptional == nullptr || restoreShapeOptional->Size() == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "restoreShapeOptional currently only supports nullptr or empty."),
             return false);

    bool fallback = rangeOptional == nullptr || rangeOptional->Size() == 0;
    int64_t rangeStart = 0;
    int64_t rangeEnd = 0;
    if (!fallback) {
        OP_CHECK(rangeOptional->Size() == RANGE_SIZE,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rangeOptional must contain exactly two elements."), return false);
        rangeStart = (*rangeOptional)[0];
        rangeEnd = (*rangeOptional)[1];
        fallback = rangeStart == 0 && rangeEnd == 0;
    }
    if (!fallback) {
        OP_CHECK(rangeStart >= 0 && rangeEnd > rangeStart,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rangeOptional must satisfy 0 <= start < end."), return false);
        OP_CHECK(numTopk >= 1 && numTopk <= MAX_TOP_K, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numTopk must be in [1, 512]."),
                 return false);
        OP_CHECK(permutedTokens->GetViewShape().GetDim(0) == rangeEnd - rangeStart,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "permutedTokens dim 0 must equal rangeOptional end minus start."),
                 return false);
    }

    const int64_t indicesLength = sortedIndices->GetViewShape().GetDim(0);
    if (fallback) {
        OP_CHECK(
            permutedTokens->GetViewShape().GetDim(0) == indicesLength,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "permutedTokens dim 0 must equal sortedIndices length in fallback mode."),
            return false);
    }
    int64_t tokensNum = 0;
    if (probsOptional != nullptr) {
        const Shape &probsShape = probsOptional->GetViewShape();
        const int64_t probsTopK = probsShape.GetDim(1);
        tokensNum = probsShape.GetDim(0);
        OP_CHECK(probsTopK >= 1 && probsTopK <= MAX_TOP_K,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "probsOptional dim 1 must be in [1, 512]."), return false);
        OP_CHECK(fallback || probsTopK == numTopk,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numTopk must equal probsOptional dim 1."), return false);
        OP_CHECK(indicesLength % probsTopK == 0 && indicesLength / probsTopK == tokensNum,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sortedIndices length must equal probsOptional dim 0 times dim 1."),
                 return false);
    } else {
        const int64_t effectiveTopK = fallback ? 1 : numTopk;
        OP_CHECK(indicesLength % effectiveTopK == 0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sortedIndices length must be divisible by effective TopK."),
                 return false);
        tokensNum = indicesLength / effectiveTopK;
    }

    const Shape &outShape = out->GetViewShape();
    OP_CHECK(outShape.GetDim(0) == tokensNum,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out dim 0 does not match inferred token count."), return false);
    OP_CHECK(outShape.GetDim(1) == permutedTokens->GetViewShape().GetDim(1),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out hidden size must equal permutedTokens hidden size."), return false);
    return true;
}

aclnnStatus CheckParams(const aclTensor *permutedTokens, const aclTensor *sortedIndices, const aclTensor *probsOptional,
                        int64_t numTopk, const aclIntArray *rangeOptional, bool paddedMode,
                        const aclIntArray *restoreShapeOptional, const aclTensor *out, const uint64_t *workspaceSize,
                        aclOpExecutor *const *executor)
{
    CHECK_RET(permutedTokens != nullptr && sortedIndices != nullptr && out != nullptr && workspaceSize != nullptr &&
                  executor != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorContract(permutedTokens, sortedIndices, probsOptional, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeAndAttrs(permutedTokens, sortedIndices, probsOptional, numTopk, rangeOptional, paddedMode,
                                 restoreShapeOptional, out),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize(const aclTensor *permutedTokens,
                                                         const aclTensor *sortedIndices, const aclTensor *probsOptional,
                                                         int64_t numTopk, const aclIntArray *rangeOptional,
                                                         bool paddedMode, const aclIntArray *restoreShapeOptional,
                                                         const aclTensor *out, uint64_t *workspaceSize,
                                                         aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnMoeTokenUnpermuteWithEp,
        DFX_IN(permutedTokens, sortedIndices, probsOptional, numTopk, rangeOptional, paddedMode, restoreShapeOptional),
        DFX_OUT(out));
    const aclnnStatus checkRet = CheckParams(permutedTokens, sortedIndices, probsOptional, numTopk, rangeOptional,
                                             paddedMode, restoreShapeOptional, out, workspaceSize, executor);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *l0Executor = uniqueExecutor.get();

    const aclTensor *tokensContiguous = l0op::Contiguous(permutedTokens, l0Executor);
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *indicesContiguous = l0op::Contiguous(sortedIndices, l0Executor);
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *probsContiguous = nullptr;
    if (probsOptional != nullptr) {
        probsContiguous = l0op::Contiguous(probsOptional, l0Executor);
        CHECK_RET(probsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *contiguousOut =
        l0Executor->AllocTensor(out->GetViewShape(), out->GetDataType(), Format::FORMAT_ND);
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *result =
        l0op::MoeTokenUnpermuteWithEp(tokensContiguous, indicesContiguous, probsContiguous, numTopk, rangeOptional,
                                      paddedMode, restoreShapeOptional, contiguousOut, l0Executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *viewCopyResult = l0op::ViewCopy(result, out, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenUnpermuteWithEp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeTokenUnpermuteWithEp);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
