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
 * \file aclnn_dense_lightning_indexer_softmax_lse_v2.h
 * \brief
 */

#ifndef ACLNN_DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_H
#define ACLNN_DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize(
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
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnDenseLightningIndexerSoftmaxLseV2(void *workspace,
                                        uint64_t workspaceSize,
                                        aclOpExecutor *executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_H
