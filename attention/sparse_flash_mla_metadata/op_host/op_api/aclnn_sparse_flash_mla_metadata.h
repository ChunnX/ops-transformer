/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SPARSE_FLASH_MLA_METADATA_AICPU_H
#define ACLNN_SPARSE_FLASH_MLA_METADATA_AICPU_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnSparseFlashMlaMetadataGetWorkspaceSize的第一段接口，计算workspace大小。
 * 功能描述：该算子实现aclnnSparseFlashMla的tiling metadata数据计算。
 * @domain aclnn_ops_train
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional,
    const aclTensor* cuSeqLensCmpKvOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedOriKvOptional,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    int64_t oriTopK, int64_t cmpTopK, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft,
    int64_t oriWinRight, char *layoutQOptional, char *layoutKvOptional, bool hasOriKv, bool hasCmpKv,
    const aclTensor* metadata, uint64_t* workspaceSize, aclOpExecutor** executor);

    /**
 * @brief aclnnSparseFlashMlaMetadata的第二段接口，用于执行计算。
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnSparseFlashMlaMetadata(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPARSE_FLASH_MLA_METADATA_AICPU_H
