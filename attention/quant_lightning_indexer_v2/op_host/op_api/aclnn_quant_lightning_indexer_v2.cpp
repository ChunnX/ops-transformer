/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_lightning_indexer_v2.h"

#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerQuantLightningIndexerV2GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *w, const aclTensor *qDescale,
    const aclTensor *kDescale, const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
    const aclTensor *blockTableOptional, const aclTensor *outputIdxOffsetOptional, const aclTensor *metadataOptional,
    int64_t topk, int64_t quantMode, int64_t maxSeqlenQ, char *layoutQOptional, char *layoutKOptional,
    int64_t maskMode, int64_t cmpRatio, int64_t returnValue, int64_t keyStride0,
    int64_t keyDequantScaleStride0, const aclTensor *sparseIndicesOut, const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerQuantLightningIndexerV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnQuantLightningIndexerV2GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *w, const aclTensor *qDescale,
    const aclTensor *kDescale, const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
    const aclTensor *blockTableOptional, const aclTensor *outputIdxOffsetOptional, const aclTensor *metadataOptional,
    int64_t topk, int64_t quantMode, int64_t maxSeqlenQ, char *layoutQOptional, char *layoutKOptional,
    int64_t maskMode, int64_t cmpRatio, int64_t returnValue, const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (k == nullptr || kDescale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "k and k_descale must not be null when calculating input strides.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    int64_t keyStride0 = -1;
    int64_t keyDequantScaleStride0 = -1;
    if (!IsContiguous(k)) {
        const auto keyStrides = k->GetViewStrides();
        keyStride0 = keyStrides[0];
    }
    if (!IsContiguous(kDescale)) {
        const auto keyScaleStrides = kDescale->GetViewStrides();
        keyDequantScaleStride0 = keyScaleStrides[0];
    }

    return aclnnInnerQuantLightningIndexerV2GetWorkspaceSize(
        q, k, w, qDescale, kDescale, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
        cmpResidualKOptional, blockTableOptional, outputIdxOffsetOptional, metadataOptional, topk, quantMode,
        maxSeqlenQ, layoutQOptional, layoutKOptional, maskMode, cmpRatio, returnValue, keyStride0,
        keyDequantScaleStride0, sparseIndicesOut, sparseValuesOut, workspaceSize, executor);
}

aclnnStatus aclnnQuantLightningIndexerV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerQuantLightningIndexerV2(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
