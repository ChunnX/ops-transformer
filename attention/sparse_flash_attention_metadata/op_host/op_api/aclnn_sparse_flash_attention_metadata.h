/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SPARSE_FLASH_ATTENTION_METADATA_AICPU_H
#define ACLNN_SPARSE_FLASH_ATTENTION_METADATA_AICPU_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) 
aclnnStatus aclnnSparseFlashAttentionMetadataGetWorkspaceSize(
    const aclTensor* actualSeqLengthsQueryOptional, const aclTensor* actualSeqLengthsKvOptional, int64_t numHeadsQ, 
    int64_t numHeadsKv, int64_t headDim, int64_t batchSizeOptional, int64_t maxSeqlenQOptional, 
    int64_t maxSeqlenKvOptional, int64_t ropeHeadDimOptional, int64_t topKOptional, int64_t sparseBlockSizeOptional, 
    char *layoutQueryOptional, char *layoutKvOptional, int64_t sparseModeOptional, int64_t preTokensOptional, 
    int64_t  nextTokensOptional, int64_t  attentionModeOptional, const aclTensor* metaData, uint64_t* workspaceSize, 
    aclOpExecutor** executor);

__attribute__((visibility("default"))) 
aclnnStatus CheckSingleParamSfa(int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKvOptional, 
                                 int64_t numHeadsQ, int64_t numHeadsKv, int64_t topKOptional, int64_t sparseModeOptional, 
                                 int64_t preTokensOptional, int64_t  nextTokensOptional, char* layoutQueryOptional, 
                                 char* layoutKvOptional);

__attribute__((visibility("default"))) 
aclnnStatus CheckExistenceSfa(const aclTensor* actualSeqLengthsQueryOptional, 
                               const aclTensor* actualSeqLengthsKvOptional, 
                               char* layoutQueryOptional, char* layoutKvOptional);

__attribute__((visibility("default"))) 
aclnnStatus CheckConsistencySfa(const aclTensor* actualSeqLengthsQueryOptional, 
                                 const aclTensor* actualSeqLengthsKvOptional, int64_t batchSizeOptional);

__attribute__((visibility("default"))) 
aclnnStatus CheckFeatureSfa();

__attribute__((visibility("default"))) 
aclnnStatus aclnnSparseFlashAttentionMetadata(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, 
                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_AICPU_H
