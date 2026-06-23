/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "aclnn_sparse_attention_score_prefill.h"

#include "sparse_attention_score_prefill.h"
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

static aclnnStatus MakeContiguous(const aclTensor *&query,
                                  const aclTensor *&key,
                                  const aclTensor *&value,
                                  const aclTensor *&selectIdx,
                                  const aclTensor *&blockTable,
                                  const aclTensor *&kvToQSelectIdx,
                                  const aclTensor *&kvToQSelectNum,
                                  const aclTensor *&kvToQSlotIdx,
                                  const aclTensor *&kvBlockValidSize,
                                  aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    selectIdx = l0op::Contiguous(selectIdx, executor);
    CHECK_RET(selectIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    blockTable = l0op::Contiguous(blockTable, executor);
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    kvToQSelectIdx = l0op::Contiguous(kvToQSelectIdx, executor);
    CHECK_RET(kvToQSelectIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    kvToQSelectNum = l0op::Contiguous(kvToQSelectNum, executor);
    CHECK_RET(kvToQSelectNum != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    kvToQSlotIdx = l0op::Contiguous(kvToQSlotIdx, executor);
    CHECK_RET(kvToQSlotIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    kvBlockValidSize = l0op::Contiguous(kvBlockValidSize, executor);
    CHECK_RET(kvBlockValidSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnSparseAttentionScorePrefillGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *selectIdx,
    const aclTensor *blockTable,
    const aclTensor *kvToQSelectIdx,
    const aclTensor *kvToQSelectNum,
    const aclTensor *kvToQSlotIdx,
    const aclTensor *kvBlockValidSize,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
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
    CHECK_RET(selectIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(kvToQSelectIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(kvToQSelectNum != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(kvToQSlotIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(kvBlockValidSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    L2_DFX_PHASE_1(aclnnSparseAttentionScorePrefill,
                   DFX_IN(query, key, value, selectIdx, blockTable,
                          kvToQSelectIdx, kvToQSelectNum, kvToQSlotIdx,
                          kvBlockValidSize,
                          actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                          numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise),
                   DFX_OUT(attentionOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    aclnnStatus ret = MakeContiguous(query, key, value, selectIdx, blockTable,
                                     kvToQSelectIdx, kvToQSelectNum,
                                     kvToQSlotIdx, kvBlockValidSize,
                                     executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto output = l0op::SparseAttentionScorePrefill(
        query, key, value, selectIdx, blockTable,
        kvToQSelectIdx, kvToQSelectNum, kvToQSlotIdx, kvBlockValidSize,
        actualSeqLengthsOptional, actualSeqLengthsKvOptional,
        numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise,
        executorImpl);

    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "SparseAttentionScorePrefill returned nullptr output.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(output, attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnSparseAttentionScorePrefill(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseAttentionScorePrefill);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
