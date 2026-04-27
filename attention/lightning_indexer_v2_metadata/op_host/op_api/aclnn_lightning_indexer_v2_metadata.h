/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_LIGHTNING_INDEXER_V2_METADATA_AICPU_H
#define ACLNN_LIGHTNING_INDEXER_V2_METADATA_AICPU_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnLightningIndexerV2MetadataGetWorkspaceSize(
    const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensKOptional, const aclTensor* sequsedQOptional, 
    const aclTensor* sequsedKOptional, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topkOptional, 
    int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, char* layoutQOptional, 
    char* layoutKOptional, int64_t maskModeOptional, const aclTensor* metaData, uint64_t* workspaceSize, 
    aclOpExecutor** executor);

__attribute__((visibility("default"))) 
aclnnStatus CheckSingleParamLiV2(int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, 
                                 int64_t numHeadsQ, int64_t numHeadsK, char* layoutQOptional, char* layoutKOptional, 
                                 int64_t maskModeOptional);

__attribute__((visibility("default")))
aclnnStatus CheckExistenceLiV2(char* layoutQOptional, char* layoutKOptional, const aclTensor* cuSeqLensQOptional, 
                               const aclTensor* cuSeqLensKOptional, int64_t maskModeOptional);

__attribute__((visibility("default")))
aclnnStatus CheckConsistencyLiV2(char* layoutQOptional, char* layoutKOptional, const aclTensor* cuSeqLensQOptional, 
                                 const aclTensor* cuSeqLensKOptional, const aclTensor* sequsedQOptional, 
                                 const aclTensor* sequsedKOptional, int64_t batchSizeOptional);

__attribute__((visibility("default")))
aclnnStatus CheckFeatureLiV2();

__attribute__((visibility("default"))) aclnnStatus
aclnnLightningIndexerV2Metadata(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_LIGHTNING_INDEXER_V2_METADATA_AICPU_H