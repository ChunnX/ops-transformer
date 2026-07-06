/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_sparse_flash_mla_softmax_l1_norm.cpp
 * \brief
 */
#include "aclnn_sparse_flash_mla_softmax_l1_norm.h"
#include "l0_sparse_flash_mla_softmax_l1_norm.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_errno.h"
#include "opdev/platform.h"
#include <string>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static bool CheckInputDataType(const aclTensor *query, const aclTensor *key)
{
    const DataType qDtype = query->GetDataType();
    const DataType kDtype = key->GetDataType();
    if (qDtype != kDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query dtype %d and key dtype %d must be same.",
                static_cast<int>(qDtype), static_cast<int>(kDtype));
        return false;
    }
    if (qDtype != DataType::DT_FLOAT16 && qDtype != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query/key dtype only support FLOAT16 or BF16, got %d.",
                static_cast<int>(qDtype));
        return false;
    }
    return true;
}

static aclnnStatus ValidateParams(const aclTensor *query,
                                  const aclTensor *key,
                                  const aclTensor *lse,
                                  int64_t cmpRatio,
                                  int64_t maskMode,
                                  char *layoutQOptional,
                                  char *layoutKOptional)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(lse != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    if (!CheckInputDataType(query, key)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (cmpRatio < 1 || cmpRatio > 128) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmpRatio must be in [1, 128], got %ld.", cmpRatio);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (maskMode != 0 && maskMode != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskMode only supports 0 or 3, got %ld.", maskMode);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (layoutQOptional != nullptr) {
        std::string layoutQ(layoutQOptional);
        if (layoutQ != "BSND" && layoutQ != "TND") {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutQ only supports BSND or TND, got %s.", layoutQ.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (layoutKOptional != nullptr) {
        std::string layoutK(layoutKOptional);
        if (layoutK != "BSND" && layoutK != "TND") {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutK only supports BSND or TND, got %s.", layoutK.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static const aclTensor *ContiguousTensorWithCheck(const aclTensor *tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    const aclTensor *contiguousTensor = l0op::Contiguous(tensor, executor);
    return contiguousTensor;
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *lse,
    const aclTensor *sparseIndicesOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *topkLengthOptional,
    const aclTensor *metadataOptional,
    double scaleValue,
    int64_t cmpRatio,
    int64_t maskMode,
    char *layoutQOptional,
    char *layoutKOptional,
    const aclTensor *softmaxL1NormOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    aclnnStatus ret = ValidateParams(query, key, lse, cmpRatio, maskMode, layoutQOptional, layoutKOptional);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    L2_DFX_PHASE_1(aclnnSparseFlashMlaSoftmaxL1Norm,
                   DFX_IN(query, key, lse, sparseIndicesOptional, cuSeqlensQOptional, cuSeqlensKOptional,
                          sequsedQOptional, sequsedKOptional, cmpResidualKOptional, topkLengthOptional,
                          metadataOptional, scaleValue, cmpRatio, maskMode, layoutQOptional, layoutKOptional),
                   DFX_OUT(softmaxL1NormOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorImpl = uniqueExecutor.get();

    query = l0op::Contiguous(query, executorImpl);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    key = l0op::Contiguous(key, executorImpl);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    lse = l0op::Contiguous(lse, executorImpl);
    CHECK_RET(lse != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    const aclTensor *sparseIndicesContiguous = ContiguousTensorWithCheck(sparseIndicesOptional, executorImpl);
    const aclTensor *cuSeqQContiguous = ContiguousTensorWithCheck(cuSeqlensQOptional, executorImpl);
    const aclTensor *cuSeqKContiguous = ContiguousTensorWithCheck(cuSeqlensKOptional, executorImpl);
    const aclTensor *seqUsedQContiguous = ContiguousTensorWithCheck(sequsedQOptional, executorImpl);
    const aclTensor *seqUsedKContiguous = ContiguousTensorWithCheck(sequsedKOptional, executorImpl);
    const aclTensor *cmpResidualKContiguous = ContiguousTensorWithCheck(cmpResidualKOptional, executorImpl);
    const aclTensor *topkLengthContiguous = ContiguousTensorWithCheck(topkLengthOptional, executorImpl);
    const aclTensor *metadataContiguous = ContiguousTensorWithCheck(metadataOptional, executorImpl);

    auto output = l0op::SparseFlashMlaSoftmaxL1Norm(
        query, key, lse, sparseIndicesContiguous, cuSeqQContiguous, cuSeqKContiguous, seqUsedQContiguous,
        seqUsedKContiguous, cmpResidualKContiguous, topkLengthContiguous, metadataContiguous,
        layoutQOptional, layoutKOptional, scaleValue, cmpRatio, maskMode,
        softmaxL1NormOptional, executorImpl);
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaSoftmaxL1Norm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseFlashMlaSoftmaxL1Norm);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
