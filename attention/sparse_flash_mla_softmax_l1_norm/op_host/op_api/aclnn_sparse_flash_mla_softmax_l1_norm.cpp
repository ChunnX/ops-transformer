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
#include "opdev/op_log.h"

#include <string>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static inline aclnnStatus ContiguousTensorWithCheck(const aclTensor *inputTensor, const aclTensor **outTensor,
                                                    aclOpExecutor *executor)
{
    if (inputTensor != nullptr && inputTensor->GetViewShape().GetDimNum() != 0) {
        *outTensor = l0op::Contiguous(inputTensor, executor);
        CHECK_RET(*outTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize(
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
    const aclTensor *softmaxL1Norm,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSparseFlashMlaSoftmaxL1Norm,
                   DFX_IN(query, key, lse, sparseIndicesOptional, cuSeqlensQOptional, cuSeqlensKOptional,
                          sequsedQOptional, sequsedKOptional, cmpResidualKOptional, topkLengthOptional,
                          metadataOptional, scaleValue, cmpRatio, maskMode, layoutQOptional, layoutKOptional),
                   DFX_OUT(softmaxL1Norm));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(query, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(key, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(lse, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(softmaxL1Norm, return ACLNN_ERR_PARAM_NULLPTR);

    OP_CHECK(query->GetDataType() == key->GetDataType(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query and key dtype must be the same."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(query->GetDataType() == DataType::DT_FLOAT16 || query->GetDataType() == DataType::DT_BF16,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query/key dtype only support FLOAT16 or BF16."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(cmpRatio >= 1 && cmpRatio <= 128,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmpRatio must be in [1, 128], got %ld.", cmpRatio),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(maskMode == 0 || maskMode == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskMode only supports 0 or 3, got %ld.", maskMode),
             return ACLNN_ERR_PARAM_INVALID);

    const aclTensor *queryContiguous = nullptr;
    const aclTensor *keyContiguous = nullptr;
    const aclTensor *lseContiguous = nullptr;
    const aclTensor *sparseIndicesContiguous = nullptr;
    const aclTensor *cuSeqQContiguous = nullptr;
    const aclTensor *cuSeqKContiguous = nullptr;
    const aclTensor *seqUsedQContiguous = nullptr;
    const aclTensor *seqUsedKContiguous = nullptr;
    const aclTensor *cmpResidualKContiguous = nullptr;
    const aclTensor *topkLengthContiguous = nullptr;
    const aclTensor *metadataContiguous = nullptr;

    auto ret = ContiguousTensorWithCheck(query, &queryContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(key, &keyContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(lse, &lseContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(sparseIndicesOptional, &sparseIndicesContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(cuSeqlensQOptional, &cuSeqQContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(cuSeqlensKOptional, &cuSeqKContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(sequsedQOptional, &seqUsedQContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(sequsedKOptional, &seqUsedKContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(cmpResidualKOptional, &cmpResidualKContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(topkLengthOptional, &topkLengthContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ContiguousTensorWithCheck(metadataOptional, &metadataContiguous, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    l0op::SparseFlashMlaSoftmaxL1Norm(
        queryContiguous, keyContiguous, lseContiguous, sparseIndicesContiguous,
        cuSeqQContiguous, cuSeqKContiguous, seqUsedQContiguous, seqUsedKContiguous,
        cmpResidualKContiguous, topkLengthContiguous, metadataContiguous,
        scaleValue, cmpRatio, maskMode, layoutQOptional, layoutKOptional,
        softmaxL1Norm, uniqueExecutor.get());

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
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
