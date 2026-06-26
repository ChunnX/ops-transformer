/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MOE_EP_DISPATCH_EPILOGUE_
#define OP_API_INC_MOE_EP_DISPATCH_EPILOGUE_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

extern "C" {

/**
 * @brief MoE EP Dispatch Epilogue: compact copy from HCCL Window to caller-preallocated
 *        recv_x (corresponding to DeepEP V2 launch_dispatch_copy_epilogue).
 *
 * @param [in] context             communication context int32 tensor
 * @param [in] dstBufferSlotIdx           [num_tokens, top_k] int32 from Dispatch
 * @param [in] numRecvTokensPerRank       [ep_world_size] int32 from Dispatch
 * @param [in] numRecvTokensPerExpert      [num_local_experts] int64 from Dispatch
 * @param [in] cachedRecvSrcMetadata       OPTIONAL cached metadata tensor
 *                                        (cached path: [A_actual, 3] int32; nullptr in non-cached path)
 * @param [in] epWorldSize                EP world size
 * @param [in] epRankId                   EP rank id
 * @param [in] numExperts                 total expert count
 * @param [in] numMaxTokensPerRank        max tokens per rank
 * @param [in] cclBufferSize              HCCL buffer size
 * @param [in] expertAlignment            alignment for expert token count
 * @param [out] recvX                     [A_alloc, hidden] bf16/fp16/fp8_e5m2/fp8_e4m3fn caller preallocated
 * @param [out] recvSrcMetadata           [A_alloc, 3] int32 handle field: [src_token_idx, src_rank_id, src_topk_idx]
 * @param [out] recvTopkWeights           OPTIONAL [A_alloc] float; nullptr if not needed
 * @param [out] recvScales               OPTIONAL [A_alloc, num_topk] float/fp8_e8m0; nullptr if not quantized
 * @param [out] workspaceSize
 * @param [out] executor
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnMoeEpDispatchEpilogueGetWorkspaceSize(
    const aclTensor* context,
    const aclTensor* dstBufferSlotIdx,
    const aclTensor* numRecvTokensPerRank,
    const aclTensor* numRecvTokensPerExpert,
    const aclTensor* cachedRecvSrcMetadata,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t numExperts,
    int64_t numMaxTokensPerRank,
    int64_t cclBufferSize,
    int64_t expertAlignment,
    aclTensor* recvX,
    aclTensor* recvSrcMetadata,
    aclTensor* recvTopkWeights,
    aclTensor* recvScales,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief execute moe_ep_dispatch_epilogue operator
 */
ACLNN_API aclnnStatus aclnnMoeEpDispatchEpilogue(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

}  // extern "C"

#endif  // OP_API_INC_MOE_EP_DISPATCH_EPILOGUE_
