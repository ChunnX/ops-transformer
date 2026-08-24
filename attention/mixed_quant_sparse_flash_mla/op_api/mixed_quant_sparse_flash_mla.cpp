/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mixed_quant_sparse_flash_mla.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MixedQuantSparseFlashMla);

const std::array<const aclTensor *, 2> MixedQuantSparseFlashMla(
    const aclTensor *q, const aclTensor *oriKvOptional, const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional, const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional, const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional, const aclTensor *sinksOptional, const aclTensor *metadataOptional,
    int64_t quantMode, int64_t ropeHeadDim, double softmaxScale, int64_t cmpRatio,
    int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
    const char *layoutQOptional, const char *layoutKvOptional, int64_t topkValueMode, bool returnSoftmaxLse,
    aclOpExecutor *executor)
{
    L0_DFX(MixedQuantSparseFlashMla, q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional,
           cmpSparseIndicesOptional, oriBlockTableOptional, cmpBlockTableOptional, cuSeqlensQOptional,
           cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional,
           sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional,
           sinksOptional, metadataOptional, quantMode, ropeHeadDim, softmaxScale, cmpRatio, oriMaskMode,
           cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, topkValueMode, returnSoftmaxLse);

    if (oriKvOptional == nullptr) {
        oriKvOptional = executor->AllocTensor(DataType::DT_FLOAT8_E4M3FN, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cmpKvOptional == nullptr) {
        cmpKvOptional = executor->AllocTensor(DataType::DT_FLOAT8_E4M3FN, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (oriSparseIndicesOptional == nullptr) {
        oriSparseIndicesOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cmpSparseIndicesOptional == nullptr) {
        cmpSparseIndicesOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (oriBlockTableOptional == nullptr) {
        oriBlockTableOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cmpBlockTableOptional == nullptr) {
        cmpBlockTableOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cuSeqlensQOptional == nullptr) {
        cuSeqlensQOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cuSeqlensOriKvOptional == nullptr) {
        cuSeqlensOriKvOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cuSeqlensCmpKvOptional == nullptr) {
        cuSeqlensCmpKvOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (sequsedQOptional == nullptr) {
        sequsedQOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (sequsedOriKvOptional == nullptr) {
        sequsedOriKvOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (sequsedCmpKvOptional == nullptr) {
        sequsedCmpKvOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cmpResidualKvOptional == nullptr) {
        cmpResidualKvOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (oriTopkLengthOptional == nullptr) {
        oriTopkLengthOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cmpTopkLengthOptional == nullptr) {
        cmpTopkLengthOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (sinksOptional == nullptr) {
        sinksOptional = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (metadataOptional == nullptr) {
        metadataOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto attentionOutAlloc = executor->AllocTensor(DataType::DT_BF16, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseAlloc = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(MixedQuantSparseFlashMla,
                           OP_INPUT(q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional,
                                    cmpSparseIndicesOptional, oriBlockTableOptional, cmpBlockTableOptional,
                                    cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional,
                                    sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional,
                                    cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional,
                                    sinksOptional, metadataOptional),
                           OP_OUTPUT(attentionOutAlloc, softmaxLseAlloc),
                           OP_ATTR(quantMode, ropeHeadDim, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode,
                                   oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, topkValueMode,
                                   returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MixedQuantSparseFlashMla InferShape failed.");
        return {nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MixedQuantSparseFlashMla,
        OP_INPUT(q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional,
                 oriBlockTableOptional, cmpBlockTableOptional, cuSeqlensQOptional, cuSeqlensOriKvOptional,
                 cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional,
                 cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional, sinksOptional, metadataOptional),
        OP_OUTPUT(attentionOutAlloc, softmaxLseAlloc),
        OP_ATTR(quantMode, ropeHeadDim, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                layoutQOptional, layoutKvOptional, topkValueMode, returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MixedQuantSparseFlashMla launch kernel failed.");
        return {nullptr, nullptr};
    }

    return {attentionOutAlloc, softmaxLseAlloc};
}

} // namespace l0op
