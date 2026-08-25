/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_GATHER_PA_KV_CACHE_WITH_DEQUANT_H_
#define ACLNN_GATHER_PA_KV_CACHE_WITH_DEQUANT_H_
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fused GatherPaKvCache + Dequantization operator
 * 
 * This operator fuses the GatherPaKvCache operation with dequantization:
 * - Key: INT8 cache -> BF16 output (with per-token-per-head scale)
 * - Value: FP8 E4M3FN cache -> BF16 output (with per-channel scale)
 * 
 * @param keyCache INT8 key cache tensor [num_blocks, block_size, num_heads, head_dim_k]
 * @param keyScaleCache FLOAT key scale tensor [num_blocks, block_size, num_heads]
 * @param valueCache FP8_E4M3FN value cache tensor [num_blocks, block_size, num_heads, head_dim_v]
 * @param valueScale FLOAT value scale tensor [num_heads, head_dim_v]
 * @param blockTables INT32/INT64 block tables tensor [batch, block_indices]
 * @param seqLens INT32/INT64 sequence lengths tensor [batch] or [batch+1]
 * @param keyRef BF16 key reference tensor [num_tokens, num_heads, head_dim_k]
 * @param valueRef BF16 value reference tensor [num_tokens, num_heads, head_dim_v]
 * @param seqOffsetOptional Optional INT32/INT64 sequence offset tensor [batch]
 * @param cacheMode Cache mode: "Norm" (ND format)
 * @param isSeqLensCumsum Whether seq_lens is cumulative sum
 * @param workspaceSize Output workspace size
 * @param executor Output executor
 * @return aclnnStatus
 */
aclnnStatus aclnnGatherPaKvCacheWithDequantGetWorkspaceSize(
    const aclTensor *keyCache,
    const aclTensor *keyScaleCache,
    const aclTensor *valueCache,
    const aclTensor *valueScale,
    const aclTensor *blockTables,
    const aclTensor *seqLens,
    aclTensor *keyRef,
    aclTensor *valueRef,
    const aclTensor *seqOffsetOptional,
    char *cacheMode,
    bool isSeqLensCumsum,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute the fused GatherPaKvCache + Dequantization operator
 * 
 * @param workspace Device workspace memory
 * @param workspaceSize Workspace size
 * @param executor Op executor from GetWorkspaceSize
 * @param stream Execution stream
 * @return aclnnStatus
 */
aclnnStatus aclnnGatherPaKvCacheWithDequant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_GATHER_PA_KV_CACHE_WITH_DEQUANT_H_
