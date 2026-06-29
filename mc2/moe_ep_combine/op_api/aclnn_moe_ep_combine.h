/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_EP_COMBINE_
#define OP_API_INC_MOE_EP_COMBINE_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

extern "C" {
/**
 * @brief MoE Expert-Parallel Combine: reverse routing + sum (corresponding to DeepEP V2 launch_combine).
 *
 * @param [in] context             communication context int32 tensor
 * @param [in] x                     expert-side tokens [A, hidden] bf16
 * @param [in] topkIdx      [num_tokens, top_k] int32 handle field
 * @param [in] recvSrcMetadata       [A, 4] int32 handle field
 * @param [in] numRecvTokensPerExpert  [num_local_experts] int64 handle field
 * @param [in] topkWeightsOptional   [A] float; nullptr for pure accumulate combine
 * @param [in] biasOptional0         nullptr (reserved, first release must be null)
 * @param [in] biasOptional1         nullptr (reserved, first release must be null)
 * @param [in] epWorldSize           EP world size
 * @param [in] epRankId              EP rank id
 * @param [in] numExperts            total expert count
 * @param [in] numMaxTokensPerRank   max tokens per rank
 * @param [in] cclBufferSize         HCCL buffer size
 * @param [out] combinedX            [num_tokens, hidden] bf16
 * @param [out] combinedTopkWeightsOptional  [num_tokens, top_k] float; only when topkWeightsOptional != nullptr
 * @param [out] workspaceSize
 * @param [out] executor
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnMoeEpCombineGetWorkspaceSize(
    const aclTensor* context,
    const aclTensor* x,
    const aclTensor* topkIdx,
    const aclTensor* recvSrcMetadata,
    const aclTensor* numRecvTokensPerExpert,
    const aclTensor* topkWeightsOptional,
    const aclTensor* biasOptional0,
    const aclTensor* biasOptional1,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t numExperts,
    int64_t numMaxTokensPerRank,
    int64_t cclBufferSize,
    aclTensor* combinedX,
    aclTensor* combinedTopkWeightsOptional,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief execute moe_ep_combine operator
 */
ACLNN_API aclnnStatus aclnnMoeEpCombine(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

}  // extern "C"

#endif  // OP_API_INC_MOE_EP_COMBINE_