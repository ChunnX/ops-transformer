/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_DISPATCH_LAYOUT_H_
#define OP_API_INC_DISPATCH_LAYOUT_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute dispatch layout information for MoE distributed dispatch.
 * First segment API: calculates workspace size.
 * @domain aclnn_ops_infer
 * @param [in] topkIdx: Top-k expert indices tensor [num_tokens, num_topk], dtype: int64
 * @param [in] num_tokens: Number of tokens
 * @param [in] num_ranks: Number of ranks in distributed group
 * @param [in] num_experts: Total number of experts
 * @param [in] num_topk: Number of top-k experts selected per token
 * @param [in] local_ranksize: Local rank size for multi-machine scenario
 * @param [out] numTokensPerRank: Token count per rank [num_ranks], dtype: int32
 * @param [out] numTokensPerExpert: Token count per expert [num_experts], dtype: int32
 * @param [out] isTokenInRank: Token-to-rank mapping [num_tokens, num_ranks], dtype: int32
 * @param [out] notifySendData: Temporary buffer [num_tokens, num_experts], dtype: int32
 * @param [out] sendTokenIdxSmall: Token indices for sending [num_tokens, num_topk], dtype: int32
 * @param [out] workspaceSize: Returns required workspace size on NPU device side
 * @param [out] executor: Returns op executor containing computation process
 * @return aclnnStatus: Returns status code
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchLayoutGetWorkspaceSize(
    const aclTensor* topkIdx,
    int64_t num_tokens,
    int64_t num_ranks,
    int64_t num_experts,
    int64_t num_topk,
    int64_t local_ranksize,
    aclTensor* numTokensPerRank,
    aclTensor* numTokensPerExpert,
    aclTensor* isTokenInRank,
    aclTensor* notifySendData,
    aclTensor* sendTokenIdxSmall,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief Second segment API: executes computation.
 * @param [in] workspace: Workspace memory address on NPU device side
 * @param [in] workspaceSize: Workspace size on NPU device side, obtained from first segment API
 * @param [in] executor: Op executor containing computation process
 * @param [in] stream: ACL stream
 * @return aclnnStatus: Returns status code
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchLayout(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_DISPATCH_LAYOUT_H_