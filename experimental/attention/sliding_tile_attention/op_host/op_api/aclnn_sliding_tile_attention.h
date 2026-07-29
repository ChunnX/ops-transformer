/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_sliding_tile_attention.h
 * @brief ACLNN interface declarations for the runtime-metadata SlidingTileAttention implementation.
 */

#ifndef ACLNN_SLIDING_TILE_ATTENTION_H_
#define ACLNN_SLIDING_TILE_ATTENTION_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates the SlidingTileAttention executor and returns its workspace size.
 * @param q Required rank-4 BNSD fp16/bf16 query tensor.
 * @param k Required tensor with the same shape and dtype as q.
 * @param v Required tensor with the same shape and dtype as q.
 * @param windowSize One positive-odd [t,h,w] triplet or 3*N positive-odd per-head values.
 * @param textLength Text length required by the selected sequence family.
 * @param hasText Whether the selected sequence family contains text tokens.
 * @param seqShape Canvas enum: 1=30x48x80, 2=36x48x48, 3=18x48x80.
 * @param output Required output tensor with the same shape and dtype as q.
 * @param workspaceSize Output workspace size in bytes.
 * @param executor Output ACLNN executor.
 * @return ACLNN_SUCCESS on success; ACLNN_ERR_PARAM_NULLPTR or ACLNN_ERR_PARAM_INVALID for invalid inputs.
 * @note Non-empty tensors require 1<=N<=128 and a positive D divisible by 8. Sequence, text, canvas, and tile
 *       families remain restricted to the task contract. Unsupported metadata fails explicitly.
 */
ACLNN_API aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *k,
                                                                const aclTensor *v, const aclIntArray *windowSize,
                                                                int64_t textLength, bool hasText, int64_t seqShape,
                                                                aclTensor *output, uint64_t *workspaceSize,
                                                                aclOpExecutor **executor);

/**
 * @brief Executes the previously created SlidingTileAttention executor.
 * @param workspace Device workspace with at least workspaceSize bytes; may be null when workspaceSize is zero.
 * @param workspaceSize Workspace size returned by aclnnSlidingTileAttentionGetWorkspaceSize.
 * @param executor Executor returned by aclnnSlidingTileAttentionGetWorkspaceSize.
 * @param stream ACL runtime stream used for the single fused kernel launch.
 * @return ACLNN_SUCCESS on success, otherwise an ACLNN runtime error code.
 */
ACLNN_API aclnnStatus aclnnSlidingTileAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
