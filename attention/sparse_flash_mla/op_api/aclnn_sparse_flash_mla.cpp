/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sparse_flash_mla.h"
#include "sparse_flash_mla.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include <vector>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr const char *DEFAULT_LAYOUT = "BSND";

static aclnnStatus CheckMandatory(
    const aclTensor *q,
    const aclTensor *attnOut,
    const aclTensor *softmaxLse,
    const uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(q != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attnOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus MakeContiguousOptional(
    const aclTensor *input,
    const aclTensor *&output,
    aclOpExecutor *executor,
    const char *name)
{
    if (input == nullptr) {
        output = nullptr;
        return ACLNN_SUCCESS;
    }
    output = l0op::Contiguous(input, executor);
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "%s contiguous failed.", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static int64_t GetFirstStride(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return 0;
    }
    auto viewShape = tensor->GetViewShape();
    if (viewShape.GetDimNum() == 0) {
        return 0;
    }
    auto viewStrides = tensor->GetViewStrides();
    return viewStrides[0];
}

static aclnnStatus ReshapeOptionalToViewShape(
    const aclTensor *input,
    const aclTensor *&output,
    aclOpExecutor *executor,
    const char *name)
{
    if (input == nullptr) {
        output = nullptr;
        return ACLNN_SUCCESS;
    }

    auto viewShape = input->GetViewShape();
    const size_t dimNum = viewShape.GetDimNum();
    if (dimNum == 0) {
        output = input;
        return ACLNN_SUCCESS;
    }

    std::vector<int64_t> shape(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        shape[i] = viewShape.GetDim(i);
    }

    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());
    if (shapeArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "%s alloc view shape failed.", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    output = l0op::Reshape(input, shapeArray, executor);
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "%s reshape to view shape failed.", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

}  // namespace

aclnnStatus aclnnSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *oriKvOptional,
    const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional,
    const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional,
    const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    double softmaxScale,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    const char *layoutQ,
    const char *layoutKv,
    int64_t topkValueMode,
    bool returnSoftmaxLse,
    const aclTensor *attnOut,
    const aclTensor *softmaxLse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSparseFlashMla,
                   DFX_IN(q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional,
                          oriBlockTableOptional, cmpBlockTableOptional, cuSeqlensQOptional, cuSeqlensOriKvOptional,
                          cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional,
                          cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional, sinksOptional,
                          metadataOptional, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft,
                          oriWinRight, layoutQ, layoutKv, topkValueMode, returnSoftmaxLse),
                   DFX_OUT(attnOut, softmaxLse));

    CHECK_RET(CheckMandatory(q, attnOut, softmaxLse, workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto qContiguous = l0op::Contiguous(q, uniqueExecutor.get());
    CHECK_RET(qContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *oriSparseIndices = nullptr;
    const aclTensor *cmpSparseIndices = nullptr;
    const aclTensor *oriBlockTable = nullptr;
    const aclTensor *cmpBlockTable = nullptr;
    const aclTensor *cuSeqlensQ = nullptr;
    const aclTensor *cuSeqlensOriKv = nullptr;
    const aclTensor *cuSeqlensCmpKv = nullptr;
    const aclTensor *sequsedQ = nullptr;
    const aclTensor *sequsedOriKv = nullptr;
    const aclTensor *sequsedCmpKv = nullptr;
    const aclTensor *cmpResidualKv = nullptr;
    const aclTensor *oriTopkLength = nullptr;
    const aclTensor *cmpTopkLength = nullptr;
    const aclTensor *sinks = nullptr;
    const aclTensor *metadata = nullptr;

    CHECK_RET(MakeContiguousOptional(oriSparseIndicesOptional, oriSparseIndices, uniqueExecutor.get(),
                                     "ori_sparse_indices") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(cmpSparseIndicesOptional, cmpSparseIndices, uniqueExecutor.get(),
                                     "cmp_sparse_indices") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(oriBlockTableOptional, oriBlockTable, uniqueExecutor.get(),
                                     "ori_block_table") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(cmpBlockTableOptional, cmpBlockTable, uniqueExecutor.get(),
                                     "cmp_block_table") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(cuSeqlensQOptional, cuSeqlensQ, uniqueExecutor.get(),
                                     "cu_seqlens_q") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(cuSeqlensOriKvOptional, cuSeqlensOriKv, uniqueExecutor.get(),
                                     "cu_seqlens_ori_kv") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(cuSeqlensCmpKvOptional, cuSeqlensCmpKv, uniqueExecutor.get(),
                                     "cu_seqlens_cmp_kv") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(sequsedQOptional, sequsedQ, uniqueExecutor.get(),
                                     "seqused_q") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(sequsedOriKvOptional, sequsedOriKv, uniqueExecutor.get(),
                                     "seqused_ori_kv") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(sequsedCmpKvOptional, sequsedCmpKv, uniqueExecutor.get(),
                                     "seqused_cmp_kv") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(cmpResidualKvOptional, cmpResidualKv, uniqueExecutor.get(),
                                     "cmp_residual_kv") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(oriTopkLengthOptional, oriTopkLength, uniqueExecutor.get(),
                                     "ori_topk_length") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(cmpTopkLengthOptional, cmpTopkLength, uniqueExecutor.get(),
                                     "cmp_topk_length") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(sinksOptional, sinks, uniqueExecutor.get(),
                                     "sinks") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(metadataOptional, metadata, uniqueExecutor.get(),
                                     "metadata") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    const char *layoutQValue = (layoutQ == nullptr) ? DEFAULT_LAYOUT : layoutQ;
    const char *layoutKvValue = (layoutKv == nullptr) ? DEFAULT_LAYOUT : layoutKv;
    int64_t oriKvStride = GetFirstStride(oriKvOptional);
    int64_t cmpKvStride = GetFirstStride(cmpKvOptional);
    const aclTensor *oriKv = nullptr;
    const aclTensor *cmpKv = nullptr;
    CHECK_RET(ReshapeOptionalToViewShape(oriKvOptional, oriKv, uniqueExecutor.get(), "ori_kv") == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ReshapeOptionalToViewShape(cmpKvOptional, cmpKv, uniqueExecutor.get(), "cmp_kv") == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    auto outputs = l0op::SparseFlashMla(
        qContiguous, oriKv, cmpKv, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
        cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv, sequsedCmpKv, cmpResidualKv,
        oriTopkLength, cmpTopkLength, sinks, metadata, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode,
        oriWinLeft, oriWinRight, layoutQValue, layoutKvValue, topkValueMode, oriKvStride, cmpKvStride,
        returnSoftmaxLse, uniqueExecutor.get());
    CHECK_RET(outputs[0] != nullptr && outputs[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyOut = l0op::ViewCopy(outputs[0], attnOut, uniqueExecutor.get());
    CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyLse = l0op::ViewCopy(outputs[1], softmaxLse, uniqueExecutor.get());
    CHECK_RET(viewCopyLse != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSparseFlashMla(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseFlashMla);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
