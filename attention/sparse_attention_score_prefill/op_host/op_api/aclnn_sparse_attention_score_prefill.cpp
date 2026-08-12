/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

/*
 * 本文件实现 SparseAttentionScorePrefill 算子的两阶段 aclnn API：
 *   1. aclnnSparseAttentionScorePrefillGetWorkspaceSize（规划阶段）：
 *      校验参数、创建 executor、计算所需 workspace 大小。
 *   2. aclnnSparseAttentionScorePrefill（执行阶段）：
 *      在指定 stream 上启动算子执行。
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

// MakeContiguous: 确保所有输入 tensor 在内存中连续（contiguous）。
// 对于非连续的 tensor，会通过 l0op::Contiguous 创建连续副本；
// 任一输入转换失败（返回 nullptr）则整体返回错误码。
static aclnnStatus MakeContiguous(const aclTensor *&query,
                                  const aclTensor *&key,
                                  const aclTensor *&value,
                                  const aclTensor *&blockTable,
                                  const aclTensor *&k2qRowPtr,
                                  const aclTensor *&k2qQIndices,
                                  const aclTensor *&k2qSlotIndices,
                                  const aclTensor *&actualSeqLengthsOptional,
                                  const aclTensor *&actualSeqLengthsKvOptional,
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

    actualSeqLengthsOptional = l0op::Contiguous(actualSeqLengthsOptional, executor);
    CHECK_RET(actualSeqLengthsOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    actualSeqLengthsKvOptional = l0op::Contiguous(actualSeqLengthsKvOptional, executor);
    CHECK_RET(actualSeqLengthsKvOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

} // namespace

// aclnnSparseAttentionScorePrefillGetWorkspaceSize: aclnn 两阶段 API 的第一阶段（规划阶段）。
// 功能：
//   1. 校验所有输入/输出指针非空；
//   2. 创建 executor；
//   3. 调用 MakeContiguous 保证输入连续；
//   4. 调用 L0 层 SparseAttentionScorePrefill 构建算子计算图；
//   5. 通过 ViewCopy 将内部输出拷贝到用户提供的 attentionOut；
//   6. 计算并返回 workspace 大小，同时输出 executor 供第二阶段使用。
__attribute__((visibility("default"))) aclnnStatus aclnnSparseAttentionScorePrefillGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockTable,
    const aclTensor *k2qRowPtr,
    const aclTensor *k2qQIndices,
    const aclTensor *k2qSlotIndices,
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
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qRowPtr != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qQIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qSlotIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengthsOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengthsKvOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    L2_DFX_PHASE_1(aclnnSparseAttentionScorePrefill,
                   DFX_IN(query, key, value, blockTable,
                          k2qRowPtr, k2qQIndices, k2qSlotIndices,
                          actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                          numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise),
                   DFX_OUT(attentionOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    aclnnStatus ret = MakeContiguous(query, key, value, blockTable,
                                     k2qRowPtr, k2qQIndices, k2qSlotIndices,
                                     actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                                     executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto output = l0op::SparseAttentionScorePrefill(
        query, key, value, blockTable,
        k2qRowPtr, k2qQIndices, k2qSlotIndices,
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

// aclnnSparseAttentionScorePrefill: aclnn 两阶段 API 的第二阶段（执行阶段）。
// 功能：在指定的 aclrtStream 上启动算子执行，复用第一阶段生成的 executor 与 workspace。
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
