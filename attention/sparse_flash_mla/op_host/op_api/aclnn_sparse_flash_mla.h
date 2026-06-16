/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SPARSE_FLASH_MLA_H
#define ACLNN_SPARSE_FLASH_MLA_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *oriKv,
    const aclTensor *cmpKv,
    const aclTensor *oriSparseIndices,
    const aclTensor *cmpSparseIndices,
    const aclTensor *oriBlockTable,
    const aclTensor *cmpBlockTable,
    const aclTensor *cuSeqLensQ,
    const aclTensor *cuSeqLensOriKv,
    const aclTensor *cuSeqLensCmpKv,
    const aclTensor *seqUsedQ,
    const aclTensor *seqUsedOriKv,
    const aclTensor *seqUsedCmpKv,
    const aclTensor *cmpResidualKv,
    const aclTensor *oriTopkLength,
    const aclTensor *cmpTopkLength,
    const aclTensor *sinks,
    const aclTensor *metadata,
    double           softmaxScale,
    int64_t          cmpRatio,
    int64_t          oriMaskMode,
    int64_t          cmpMaskMode,
    int64_t          oriWinLeft,
    int64_t          oriWinRight,
    char            *layoutQ,
    char            *layoutKv,
    int64_t          topkValueMode,
    int64_t          oriKvStride,
    int64_t          cmpKvStride,
    bool             returnSoftmaxLse,
    const aclTensor *attnOut,
    const aclTensor *softmaxLse,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMla(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
