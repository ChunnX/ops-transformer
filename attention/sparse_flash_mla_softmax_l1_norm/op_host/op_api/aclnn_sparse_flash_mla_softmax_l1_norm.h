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
 * \file aclnn_sparse_flash_mla_softmax_l1_norm.h
 * \brief
 */
#ifndef ACLNN_SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_H
#define ACLNN_SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_H

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize(
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
    double scaleValue,
    int64_t cmpRatio,
    int64_t maskMode,
    char *layoutQOptional,
    char *layoutKOptional,
    const aclTensor *softmaxL1NormOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaSoftmaxL1Norm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_H
