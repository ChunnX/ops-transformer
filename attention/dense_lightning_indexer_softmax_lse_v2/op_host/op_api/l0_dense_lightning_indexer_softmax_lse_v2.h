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
 * \file l0_dense_lightning_indexer_softmax_lse_v2.h
 * \brief L0 operator declaration for DenseLightningIndexerSoftmaxLseV2
 */

#ifndef OP_API_INC_LEVEL0_L0_DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_H_
#define OP_API_INC_LEVEL0_L0_DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_H_

#include "opdev/op_executor.h"

namespace l0op {

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
    aclOpExecutor *executor);

}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_L0_DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_H_
