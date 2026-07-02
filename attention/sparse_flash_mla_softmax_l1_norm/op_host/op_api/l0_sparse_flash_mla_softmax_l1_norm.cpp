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
 * \file l0_sparse_flash_mla_softmax_l1_norm.cpp
 * \brief
 */
#include "l0_sparse_flash_mla_softmax_l1_norm.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SparseFlashMlaSoftmaxL1Norm);

const aclTensor* SparseFlashMlaSoftmaxL1Norm(
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
    float scaleValue,
    int32_t cmpRatio,
    int32_t maskMode,
    char *layoutQOptional,
    char *layoutKOptional,
    const aclTensor *softmaxL1NormOptional,
    aclOpExecutor *executor)
{
    L0_DFX(SparseFlashMlaSoftmaxL1Norm, query, key, lse, sparseIndicesOptional, cuSeqlensQOptional,
           cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional, topkLengthOptional,
           metadataOptional, scaleValue, cmpRatio, maskMode, layoutQOptional, layoutKOptional,
           softmaxL1NormOptional);

    const char *safeLayoutQ = (layoutQOptional != nullptr) ? layoutQOptional : "BSND";
    const char *safeLayoutK = (layoutKOptional != nullptr) ? layoutKOptional : "BSND";

    const aclTensor *metadataTensor = (metadataOptional != nullptr) ? metadataOptional :
        executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);

    const aclTensor *outputTensor = (softmaxL1NormOptional != nullptr) ? softmaxL1NormOptional :
        executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(SparseFlashMlaSoftmaxL1Norm,
                           OP_INPUT(query, key, lse, sparseIndicesOptional, cuSeqlensQOptional,
                                    cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional,
                                    topkLengthOptional, metadataTensor),
                           OP_OUTPUT(outputTensor),
                           OP_ATTR(scaleValue, cmpRatio, maskMode, safeLayoutQ, safeLayoutK));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "SparseFlashMlaSoftmaxL1Norm infer shape failed, scaleValue: %f.", scaleValue);
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(SparseFlashMlaSoftmaxL1Norm,
                                OP_INPUT(query, key, lse, sparseIndicesOptional, cuSeqlensQOptional,
                                         cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                                         cmpResidualKOptional, topkLengthOptional, metadataTensor),
                                OP_OUTPUT(outputTensor),
                                OP_ATTR(scaleValue, cmpRatio, maskMode, safeLayoutQ, safeLayoutK));

    return outputTensor;
}
} // namespace l0op
