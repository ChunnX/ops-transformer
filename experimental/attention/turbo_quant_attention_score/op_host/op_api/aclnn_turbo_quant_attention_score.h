/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_turbo_quant_attention_score.h
 * \brief ACLNN C-API for TurboQuantAttentionScore.
 */

#ifndef OP_API_ACLNN_TURBO_QUANT_ATTENTION_SCORE_H
#define OP_API_ACLNN_TURBO_QUANT_ATTENTION_SCORE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief TurboQuantAttentionScore phase-1: compute required workspace size.
 * @param [in]  queryRotated  : [T, H, D], BF16.
 * @param [in]  queryQjl      : [T, H, D], BF16.
 * @param [in]  keyCacheIdx   : [num_blocks, 128, KvH, idxBytes], UINT8.
 * @param [in]  keyCacheQjl   : [num_blocks, 128, KvH, qjlBytes], UINT8.
 * @param [in]  keyCacheNorm  : [num_blocks, 128, KvH], BF16 / FLOAT16.
 * @param [in]  keyCacheGamma : [num_blocks, 128, KvH], BF16 / FLOAT16.
 * @param [in]  blockTable    : [B, maxBlocks], INT32.
 * @param [in]  seqLens       : [B], INT32.
 * @param [in]  mseBits       : {2, 3, 4}, default 3.
 * @param [out] out           : [T, H, maxKvLen], FLOAT32.
 * @param [out] workspaceSize : required workspace bytes on device.
 * @param [out] executor      : op executor handle.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnTurboQuantAttentionScoreGetWorkspaceSize(
    const aclTensor *queryRotated, const aclTensor *queryQjl, const aclTensor *keyCacheIdx,
    const aclTensor *keyCacheQjl, const aclTensor *keyCacheNorm, const aclTensor *keyCacheGamma,
    const aclTensor *blockTable, const aclTensor *seqLens, int64_t mseBits, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief TurboQuantAttentionScore phase-2: launch the kernel.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnTurboQuantAttentionScore(void *workspace,
                                                                                 uint64_t workspaceSize,
                                                                                 aclOpExecutor *executor,
                                                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_TURBO_QUANT_ATTENTION_SCORE_H
