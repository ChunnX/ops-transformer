/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_gating_top_k.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe_gating_top_k.h"

using namespace op;

namespace MoeGatingTopKCheck {

static const std::initializer_list<op::DataType> MOE_GATING_TOP_K_DTYPE_SUPPORT_LIST_X = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_GATING_TOP_K_DTYPE_SUPPORT_LIST_INT = {
    DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor* x, const aclTensor* yOut,
                                const aclTensor* expertIdxOut, const aclTensor* outOut)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(yOut, return false);
    OP_CHECK_NULL(expertIdxOut, return false);
    OP_CHECK_NULL(outOut, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* x, const aclTensor* biasOptional,
                                   const aclTensor* yOut, const aclTensor* expertIdxOut,
                                   const aclTensor* outOut)
{
    if (x != nullptr && x->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x, MOE_GATING_TOP_K_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (biasOptional != nullptr && biasOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, MOE_GATING_TOP_K_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(x, biasOptional, return false);
    }
    if (yOut != nullptr && yOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(yOut, MOE_GATING_TOP_K_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(x, yOut, return false);
    }
    if (expertIdxOut != nullptr && expertIdxOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdxOut, MOE_GATING_TOP_K_DTYPE_SUPPORT_LIST_INT, return false);
    }
    if (outOut != nullptr && outOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(outOut, MOE_GATING_TOP_K_DTYPE_SUPPORT_LIST_X, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* biasOptional,
                               const aclTensor* yOut, const aclTensor* expertIdxOut,
                               const aclTensor* outOut)
{
    CHECK_RET(CheckNotNull(x, yOut, expertIdxOut, outOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, biasOptional, yOut, expertIdxOut, outOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeGatingTopKCheck

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMoeGatingTopKGetWorkspaceSize(
    const aclTensor* x, const aclTensor* biasOptional,
    int64_t k, int64_t kGroup, int64_t groupCount, int64_t groupSelectMode,
    int64_t renorm, int64_t normType, bool outFlag,
    double routedScalingFactor, double eps,
    const aclTensor* yOut, const aclTensor* expertIdxOut, const aclTensor* outOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMoeGatingTopK,
        DFX_IN(x, biasOptional, k, kGroup, groupCount, groupSelectMode, renorm,
              normType, outFlag, routedScalingFactor, eps),
        DFX_OUT(yOut, expertIdxOut, outOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclnnStatus ret = MoeGatingTopKCheck::CheckParams(x, biasOptional, yOut, expertIdxOut, outOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    OP_LOGI("MoeGatingTopK", "[Before Kernel] k=%ld, kGroup=%ld, groupCount=%ld, groupSelectMode=%ld, "
            "renorm=%ld, normType=%ld, outFlag=%d, routedScalingFactor=%f, eps=%f",
            k, kGroup, groupCount, groupSelectMode, renorm, normType, outFlag, routedScalingFactor, eps);

    OP_LOGI("MoeGatingTopK", "[Before Kernel] x dtype=%d, shapeSize=%ld, stride0=%ld, stride1=%ld, addr=%p",
            (int)x->GetDataType(), x->GetViewShape().GetShapeSize(),
            x->GetViewShape().GetDim(0), x->GetViewShape().GetDimNum() > 1 ? x->GetViewShape().GetDim(1) : -1,
            x->GetTensorAddr());
    OP_LOGI("MoeGatingTopK", "[Before Kernel] yOut dtype=%d, shapeSize=%ld, dim0=%ld, dim1=%ld, addr=%p",
            (int)yOut->GetDataType(), yOut->GetViewShape().GetShapeSize(),
            yOut->GetViewShape().GetDim(0), yOut->GetViewShape().GetDimNum() > 1 ? yOut->GetViewShape().GetDim(1) : -1,
            yOut->GetTensorAddr());
    OP_LOGI("MoeGatingTopK", "[Before Kernel] expertIdxOut dtype=%d, shapeSize=%ld, dim0=%ld, dim1=%ld, addr=%p",
            (int)expertIdxOut->GetDataType(), expertIdxOut->GetViewShape().GetShapeSize(),
            expertIdxOut->GetViewShape().GetDim(0), expertIdxOut->GetViewShape().GetDimNum() > 1 ? expertIdxOut->GetViewShape().GetDim(1) : -1,
            expertIdxOut->GetTensorAddr());
    OP_LOGI("MoeGatingTopK", "[Before Kernel] outOut dtype=%d, shapeSize=%ld, dim0=%ld, dim1=%ld, addr=%p",
            (int)outOut->GetDataType(), outOut->GetViewShape().GetShapeSize(),
            outOut->GetViewShape().GetDim(0), outOut->GetViewShape().GetDimNum() > 1 ? outOut->GetViewShape().GetDim(1) : -1,
            outOut->GetTensorAddr());
    if (biasOptional != nullptr) {
        OP_LOGI("MoeGatingTopK", "[Before Kernel] bias dtype=%d, shapeSize=%ld, addr=%p",
                (int)biasOptional->GetDataType(), biasOptional->GetViewShape().GetShapeSize(),
                biasOptional->GetTensorAddr());
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* biasContiguous = nullptr;
    if (biasOptional != nullptr) {
        biasContiguous = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    OP_LOGI("MoeGatingTopK", "[Before Kernel] xContiguous dtype=%d, shapeSize=%ld, dim0=%ld, dim1=%ld, addr=%p",
            (int)xContiguous->GetDataType(), xContiguous->GetViewShape().GetShapeSize(),
            xContiguous->GetViewShape().GetDim(0), xContiguous->GetViewShape().GetDimNum() > 1 ? xContiguous->GetViewShape().GetDim(1) : -1,
            xContiguous->GetTensorAddr());

    auto result = l0op::MoeGatingTopK(xContiguous, biasContiguous, nullptr, nullptr,
                                      k, kGroup, groupCount, groupSelectMode,
                                      renorm, normType, outFlag, routedScalingFactor, eps,
                                      yOut, expertIdxOut, outOut,
                                      uniqueExecutor.get());
    CHECK_RET(result.y != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(result.expertIdx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(result.outOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    OP_LOGI("MoeGatingTopK", "[After Kernel] result.y dtype=%d, shapeSize=%ld, dim0=%ld, dim1=%ld, addr=%p",
            (int)result.y->GetDataType(), result.y->GetViewShape().GetShapeSize(),
            result.y->GetViewShape().GetDim(0), result.y->GetViewShape().GetDimNum() > 1 ? result.y->GetViewShape().GetDim(1) : -1,
            result.y->GetTensorAddr());
    OP_LOGI("MoeGatingTopK", "[After Kernel] result.expertIdx dtype=%d, shapeSize=%ld, dim0=%ld, dim1=%ld, addr=%p",
            (int)result.expertIdx->GetDataType(), result.expertIdx->GetViewShape().GetShapeSize(),
            result.expertIdx->GetViewShape().GetDim(0), result.expertIdx->GetViewShape().GetDimNum() > 1 ? result.expertIdx->GetViewShape().GetDim(1) : -1,
            result.expertIdx->GetTensorAddr());
    OP_LOGI("MoeGatingTopK", "[After Kernel] result.outOut dtype=%d, shapeSize=%ld, dim0=%ld, dim1=%ld, addr=%p",
            (int)result.outOut->GetDataType(), result.outOut->GetViewShape().GetShapeSize(),
            result.outOut->GetViewShape().GetDim(0), result.outOut->GetViewShape().GetDimNum() > 1 ? result.outOut->GetViewShape().GetDim(1) : -1,
            result.outOut->GetTensorAddr());

    OP_LOGI("MoeGatingTopK", "[ViewCopy] result.y -> yOut: src dtype=%d dst dtype=%d, src addr=%p dst addr=%p",
            (int)result.y->GetDataType(), (int)yOut->GetDataType(),
            result.y->GetTensorAddr(), yOut->GetTensorAddr());
    auto viewCopyYOutResult = l0op::ViewCopy(result.y, yOut, uniqueExecutor.get());
    CHECK_RET(viewCopyYOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    OP_LOGI("MoeGatingTopK", "[ViewCopy] result.expertIdx -> expertIdxOut: src dtype=%d dst dtype=%d, src addr=%p dst addr=%p",
            (int)result.expertIdx->GetDataType(), (int)expertIdxOut->GetDataType(),
            result.expertIdx->GetTensorAddr(), expertIdxOut->GetTensorAddr());
    auto viewCopyExpertIdxResult = l0op::ViewCopy(result.expertIdx, expertIdxOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpertIdxResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    OP_LOGI("MoeGatingTopK", "[ViewCopy] result.outOut -> outOut: src dtype=%d dst dtype=%d, src addr=%p dst addr=%p",
            (int)result.outOut->GetDataType(), (int)outOut->GetDataType(),
            result.outOut->GetTensorAddr(), outOut->GetTensorAddr());
    auto viewCopyOutOutResult = l0op::ViewCopy(result.outOut, outOut, uniqueExecutor.get());
    CHECK_RET(viewCopyOutOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnMoeGatingTopK(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeGatingTopK);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif