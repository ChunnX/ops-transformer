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
#include "opdev/op_log.h"

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
    const char *layoutQ,
    const char *layoutK,
    double scaleValue,
    int64_t cmpRatio,
    int64_t maskMode,
    const aclTensor *softmaxL1NormOptional,
    aclOpExecutor *executor)
{
    L0_DFX(SparseFlashMlaSoftmaxL1Norm, query, key, lse, sparseIndicesOptional, cuSeqlensQOptional,
           cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional, topkLengthOptional,
           metadataOptional, layoutQ, layoutK, scaleValue, cmpRatio, maskMode, softmaxL1NormOptional);

    ADD_TO_LAUNCHER_LIST_AICORE(SparseFlashMlaSoftmaxL1Norm,
                                OP_INPUT(query, key, lse, sparseIndicesOptional, cuSeqlensQOptional,
                                         cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                                         cmpResidualKOptional, topkLengthOptional, metadataOptional),
                                OP_OUTPUT(softmaxL1NormOptional),
                                OP_ATTR(layoutQ, layoutK, scaleValue, cmpRatio, maskMode));

    OP_LOGI("SparseFlashMlaSoftmaxL1Norm",
            "l0 registered to launcher list, layoutQ=%s, maskMode=%ld, cmpRatio=%ld.",
            layoutQ != nullptr ? layoutQ : "null", maskMode, cmpRatio);

    return softmaxL1NormOptional;
}

} // namespace l0op
