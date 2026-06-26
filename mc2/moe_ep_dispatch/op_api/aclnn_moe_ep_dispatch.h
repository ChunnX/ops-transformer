/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MOE_EP_DISPATCH_
#define OP_API_INC_MOE_EP_DISPATCH_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

extern "C" {
/**
 * @brief MoE Expert-Parallel Dispatch: SendPhase + RecvPhase, write HCCL Window
 *        + host-pinned counter (corresponding to DeepEP V2 launch_dispatch).
 *
 * @param [in] context             communication context int32 tensor
 * @param [in] x                   token data [num_tokens, hidden] bf16/fp16/fp8_e5m2/fp8_e4m3fn
 * @param [in] topkIdx             top-k expert indices [num_tokens, top_k] int32; nullptr in cached mode
 * @param [in] topkWeights         top-k weights [num_tokens, top_k] float; nullptr in cached mode
 * @param [in] scales              dynamic quantization scales (float or fp8_e8m0); nullptr if not quantized
 * @param [in] cachedHandleDstBufferSlotIdx  [num_tokens, top_k] int32; nullptr in non-cached mode
 * @param [in] epWorldSize         EP world size
 * @param [in] epRankId            EP rank id
 * @param [in] numExperts          total expert count
 * @param [in] numMaxTokensPerRank max tokens per rank upper bound
 * @param [in] cclBufferSize       HCCL buffer size
 * @param [in] expertAlignment     alignment for expert token count
 * @param [in] doCpuSync           true: write host-pinned counter for precise alloc
 * @param [in] hostPinnedCounterAddr device address of host-pinned counter (0 if doCpuSync=false)
 * @param [out] numRecvTokensPerRank   [ep_world_size] int32 device
 * @param [out] numRecvTokensPerExpert [num_local_experts] int64 device
 * @param [out] dstBufferSlotIdx       [num_tokens, top_k] int32 device
 * @param [out] workspaceSize      workspace size output
 * @param [out] executor           executor output
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnMoeEpDispatchGetWorkspaceSize(
    const aclTensor* context,
    const aclTensor* x,
    const aclTensor* topkIdx,
    const aclTensor* topkWeights,
    const aclTensor* scales,
    const aclTensor* cachedHandleDstBufferSlotIdx,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t numExperts,
    int64_t numMaxTokensPerRank,
    int64_t cclBufferSize,
    int64_t expertAlignment,
    bool doCpuSync,
    int64_t hostPinnedCounterAddr,
    aclTensor* numRecvTokensPerRank,
    aclTensor* numRecvTokensPerExpert,
    aclTensor* dstBufferSlotIdx,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief execute moe_ep_dispatch operator
 */
ACLNN_API aclnnStatus aclnnMoeEpDispatch(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

}  // extern "C"

#endif  // OP_API_INC_MOE_EP_DISPATCH_