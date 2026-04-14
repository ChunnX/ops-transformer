/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_QUANT_LIGHTNING_INDEXER_METADATA_AICPU_H
#define ACLNN_QUANT_LIGHTNING_INDEXER_METADATA_AICPU_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnQuantLightningIndexerMetadataGetWorkspaceSize(
    const aclTensor* actualSeqLengthsQueryOptional, const aclTensor* actualSeqLengthsKeyOptional, int64_t numHeadsQ,
    int64_t numHeadsK, int64_t headDim, int64_t queryQuantMode, int64_t keyQuantMode, int64_t batchSizeOptional,
    int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, char* layoutQueryOptional, char* layoutKeyOptional,
    int64_t sparseCountOptional, int64_t sparseModeOptional, int64_t preTokensOptional, int64_t nextTokensOptional,
    const aclTensor* metaData, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) 
aclnnStatus CheckSingleParamQli(int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, 
                             int64_t numHeadsQ, int64_t numHeadsK, char* layoutQueryOptional, char* layoutKeyOptional, 
                             int64_t sparseModeOptional, int64_t preTokensOptional, int64_t nextTokensOptional);

__attribute__((visibility("default")))
aclnnStatus CheckExistenceQli(char* layoutQueryOptional, char* layoutKeyOptional, 
                           const aclTensor* actualSeqLengthsQueryOptional, const aclTensor* actualSeqLengthsKeyOptional);

__attribute__((visibility("default")))
aclnnStatus CheckConsistencyQli(char* layoutQueryOptional, char* layoutKeyOptional,
                             const aclTensor* actualSeqLengthsQueryOptional, 
                             const aclTensor* actualSeqLengthsKeyOptional, int64_t batchSizeOptional);

__attribute__((visibility("default")))
aclnnStatus CheckFeatureQli();

__attribute__((visibility("default"))) aclnnStatus
aclnnQuantLightningIndexerMetadata(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_QUANT_LIGHTNING_INDEXER_METADATA_AICPU_H