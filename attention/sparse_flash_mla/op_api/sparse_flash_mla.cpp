/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_flash_mla.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SparseFlashMla);

const std::array<const aclTensor *, 2> SparseFlashMla(
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
    int64_t oriKvStride,
    int64_t cmpKvStride,
    bool returnSoftmaxLse,
    aclOpExecutor *executor)
{
    L0_DFX(SparseFlashMla, q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional,
           oriBlockTableOptional, cmpBlockTableOptional, cuSeqlensQOptional, cuSeqlensOriKvOptional,
           cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional,
           cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional, sinksOptional, metadataOptional,
           softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQ, layoutKv,
           topkValueMode, oriKvStride, cmpKvStride, returnSoftmaxLse);

    auto attnOut = executor->AllocTensor(q->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLse = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    if (attnOut == nullptr || softmaxLse == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SparseFlashMla AllocTensor failed.");
        return {nullptr, nullptr};
    }

    float softmaxScaleAttr = static_cast<float>(softmaxScale);
    auto ret = INFER_SHAPE(
        SparseFlashMla,
        OP_INPUT(q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional,
                 oriBlockTableOptional, cmpBlockTableOptional, cuSeqlensQOptional, cuSeqlensOriKvOptional,
                 cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional,
                 cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional, sinksOptional, metadataOptional),
        OP_OUTPUT(attnOut, softmaxLse),
        OP_ATTR(softmaxScaleAttr, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQ, layoutKv,
                topkValueMode, oriKvStride, cmpKvStride, returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SparseFlashMla InferShape failed.");
        return {nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        SparseFlashMla,
        OP_INPUT(q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional,
                 oriBlockTableOptional, cmpBlockTableOptional, cuSeqlensQOptional, cuSeqlensOriKvOptional,
                 cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional,
                 cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional, sinksOptional, metadataOptional),
        OP_OUTPUT(attnOut, softmaxLse),
        OP_ATTR(softmaxScaleAttr, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQ, layoutKv,
                topkValueMode, oriKvStride, cmpKvStride, returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SparseFlashMla ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }

    return {attnOut, softmaxLse};
}

}  // namespace l0op
