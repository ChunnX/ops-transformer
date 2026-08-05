/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SPARSE_FLASH_ATTENTION_GRAD_V2_H
#define ACLNN_SPARSE_FLASH_ATTENTION_GRAD_V2_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnSparseFlashAttentionGradV2
 *        calculates the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
 * @note  V2 相比 V1（aclnnSparseFlashAttentionGrad，商发兼容、5 输出）：
 *        1) 新增 oss-sink 输入 sinks 与 sink 梯度输出 dSinks（6 输出）；
 *        2) layout 拆分为 layoutQuery/layoutKv（约定 q==kv 恒等，tiling 校验相等）。
 *        V1/V2 复用同一 OpDef（SparseFlashAttentionGrad）与同一 inner aclnn。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashAttentionGradV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *dOut,
    const aclTensor *out,
    const aclTensor *softmaxMax,
    const aclTensor *softmaxSum,
    const aclTensor *sinks,                    // oss-sink 输入 [N1] FP32，OPTIONAL；与 dSinks 成对出现
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    double scaleValue,
    int64_t sparseBlockSize,
    char *layoutQueryOptional,
    char *layoutKvOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    bool deterministic,
    const aclTensor *dQueryOut,
    const aclTensor *dKeyOut,
    const aclTensor *dValueOut,
    const aclTensor *dQueryRopeOutOptional,
    const aclTensor *dKeyRopeOutOptional,
    const aclTensor *dSinksOptional,           // sink 梯度 [N1] FP32，OPTIONAL；与 sinks 成对出现
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of ACLNN_SPARSE_FLASH_ATTENTION_GRAD_V2_H is used to perform calculations.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashAttentionGradV2(void *workspace,
                                                                                 uint64_t workspaceSize,
                                                                                 aclOpExecutor *executor,
                                                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPARSE_FLASH_ATTENTION_GRAD_V2_H
