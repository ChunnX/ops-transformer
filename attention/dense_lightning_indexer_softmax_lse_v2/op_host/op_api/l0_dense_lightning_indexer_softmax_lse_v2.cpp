/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file l0_dense_lightning_indexer_softmax_lse_v2.cpp
 * \brief L0 operator implementation for DenseLightningIndexerSoftmaxLseV2
 */

#include "l0_dense_lightning_indexer_softmax_lse_v2.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(DenseLightningIndexerSoftmaxLseV2);

const aclTensor *DenseLightningIndexerSoftmaxLseV2(
    const aclTensor *queryIndex,
    const aclTensor *keyIndex,
    const aclTensor *weight,
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *metadataOptional,
    const char *layoutQ,
    const char *layoutK,
    int64_t maskMode,
    int64_t cmpRatio,
    const aclTensor *softmaxLseOut,
    aclOpExecutor *executor)
{
    L0_DFX(DenseLightningIndexerSoftmaxLseV2,
           queryIndex, keyIndex, weight,
           cuSeqLensQOptional, cuSeqLensKOptional,
           seqUsedQOptional, seqUsedKOptional,
           cmpResidualKOptional, metadataOptional,
           layoutQ, layoutK, maskMode, cmpRatio,
           softmaxLseOut);

    ADD_TO_LAUNCHER_LIST_AICORE(
        DenseLightningIndexerSoftmaxLseV2,
        OP_INPUT(queryIndex, keyIndex, weight,
                 cuSeqLensQOptional, cuSeqLensKOptional,
                 seqUsedQOptional, seqUsedKOptional,
                 cmpResidualKOptional, metadataOptional),
        OP_OUTPUT(softmaxLseOut),
        OP_ATTR(layoutQ, layoutK, maskMode, cmpRatio));

    return softmaxLseOut;
}

}  // namespace l0op
