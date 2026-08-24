/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "aclnn_minimax_sparse_attention_split_kv.h"

#include "minimax_sparse_attention_split_kv.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include <acl/acl.h>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static bool IsSupportedQkvDtype(DataType dtype)
{
    return dtype == DataType::DT_BF16 || dtype == DataType::DT_FLOAT8_E4M3FN;
}

static aclnnStatus ValidateParams(const aclTensor *query,
                                  const aclTensor *key,
                                  const aclTensor *value,
                                  const aclTensor *blockTable,
                                  const aclTensor *k2qRowPtr,
                                  const aclTensor *k2qQIndices,
                                  const aclTensor *k2qSlotIndices,
                                  const aclTensor *actualSeqLengths,
                                  const aclTensor *actualSeqLengthsKv,
                                  const aclTensor *attentionOut,
                                  int64_t innerPrecise)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qRowPtr != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qQIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qSlotIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengths != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengthsKv != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    DataType qDtype = query->GetDataType();
    DataType kDtype = key->GetDataType();
    DataType vDtype = value->GetDataType();
    if (!IsSupportedQkvDtype(qDtype) || !IsSupportedQkvDtype(kDtype) ||
        !IsSupportedQkvDtype(vDtype)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "MinimaxSparseAttentionSplitKv Q/K/V only support BF16 or FLOAT8_E4M3FN.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qDtype != kDtype || qDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "MinimaxSparseAttentionSplitKv Q/K/V dtype must be consistent.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (blockTable->GetDataType() != DataType::DT_INT32 ||
        k2qRowPtr->GetDataType() != DataType::DT_INT32 ||
        k2qQIndices->GetDataType() != DataType::DT_INT32 ||
        k2qSlotIndices->GetDataType() != DataType::DT_INT32 ||
        actualSeqLengths->GetDataType() != DataType::DT_INT32 ||
        actualSeqLengthsKv->GetDataType() != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "MinimaxSparseAttentionSplitKv index/seqlen inputs must be INT32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (attentionOut->GetDataType() != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "MinimaxSparseAttentionSplitKv attentionOut must be BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    constexpr int64_t kFp8InnerPrecise = 4;
    if (qDtype == DataType::DT_FLOAT8_E4M3FN && innerPrecise != kFp8InnerPrecise) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "MinimaxSparseAttentionSplitKv fp8 path uses FP32 O_partial only "
                "(innerPrecise=%ld); got %ld. innerPrecise=1 BF16 O_partial is not implemented.",
                kFp8InnerPrecise, innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus MakeContiguous(const aclTensor *&query,
                                  const aclTensor *&key,
                                  const aclTensor *&value,
                                  const aclTensor *&blockTable,
                                  const aclTensor *&k2qRowPtr,
                                  const aclTensor *&k2qQIndices,
                                  const aclTensor *&k2qSlotIndices,
                                  const aclTensor *&actualSeqLengths,
                                  const aclTensor *&actualSeqLengthsKv,
                                  aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    blockTable = l0op::Contiguous(blockTable, executor);
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    k2qRowPtr = l0op::Contiguous(k2qRowPtr, executor);
    CHECK_RET(k2qRowPtr != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    k2qQIndices = l0op::Contiguous(k2qQIndices, executor);
    CHECK_RET(k2qQIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    k2qSlotIndices = l0op::Contiguous(k2qSlotIndices, executor);
    CHECK_RET(k2qSlotIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    actualSeqLengths = l0op::Contiguous(actualSeqLengths, executor);
    CHECK_RET(actualSeqLengths != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    actualSeqLengthsKv = l0op::Contiguous(actualSeqLengthsKv, executor);
    CHECK_RET(actualSeqLengthsKv != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnMinimaxSparseAttentionSplitKvGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockTable,
    const aclTensor *k2qRowPtr,
    const aclTensor *k2qQIndices,
    const aclTensor *k2qSlotIndices,
    const aclTensor *actualSeqLengths,
    const aclTensor *actualSeqLengthsKv,
    int64_t numKeyValueHeads,
    double scaleValue,
    int64_t blockSize,
    int64_t topK,
    int64_t innerPrecise,
    aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qRowPtr != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qQIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qSlotIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengths != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengthsKv != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    aclnnStatus validateRet = ValidateParams(query, key, value, blockTable,
                                             k2qRowPtr, k2qQIndices, k2qSlotIndices,
                                             actualSeqLengths, actualSeqLengthsKv,
                                             attentionOut, innerPrecise);
    if (validateRet != ACLNN_SUCCESS) {
        return validateRet;
    }

    L2_DFX_PHASE_1(aclnnMinimaxSparseAttentionSplitKv,
                   DFX_IN(query, key, value, blockTable,
                          k2qRowPtr, k2qQIndices, k2qSlotIndices,
                          actualSeqLengths, actualSeqLengthsKv,
                          numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise),
                   DFX_OUT(attentionOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    aclnnStatus ret = MakeContiguous(query, key, value, blockTable,
                                     k2qRowPtr, k2qQIndices, k2qSlotIndices,
                                     actualSeqLengths, actualSeqLengthsKv,
                                     executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto output = l0op::MinimaxSparseAttentionSplitKv(
        query, key, value, blockTable,
        k2qRowPtr, k2qQIndices, k2qSlotIndices,
        actualSeqLengths, actualSeqLengthsKv,
        numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise,
        executorImpl);
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "MinimaxSparseAttentionSplitKv returned nullptr output.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(output, attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnMinimaxSparseAttentionSplitKv(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMinimaxSparseAttentionSplitKv);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
