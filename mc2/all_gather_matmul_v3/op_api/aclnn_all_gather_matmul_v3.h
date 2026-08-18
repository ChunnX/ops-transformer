/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_all_gather_matmul_v3.h
 * \brief aclnn API for AllGatherMatmulV3 (MX-quant FP8/FP4, apace UDMA path)
 */

#ifndef OP_API_INC_ALL_GATHER_MATMUL_APACE_H_
#define OP_API_INC_ALL_GATHER_MATMUL_APACE_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Phase 1: compute workspace size for AllGatherQuantMatmulV3.
 * @domain aclnn_ops_infer
 * @param [in] context: apace communication context tensor. Type: int32/uint8.
 * @param [in] x1: Left matrix (per-rank A). Type: float8_e4m3fn / float8_e5m2 / float4_e2m1.
 *                 Shape: [M_per_rank, K].
 * @param [in] x2: Right matrix (transposed B). Type: same family as x1. Shape: [N, K].
 * @param [in] biasOptional: Optional bias (not supported yet, must be nullptr).
 * @param [in] x1ScaleOptional: MX scale for x1. Type: float8_e8m0. Shape: [M_per_rank, Ceil(K/64), 2].
 * @param [in] x2ScaleOptional: MX scale for x2. Type: float8_e8m0. Shape: [N, Ceil(K/64), 2].
 * @param [in] group: Communication group identifier.
 * @param [in] rankSize: Number of ranks.
 * @param [in] hcclBufferSize: HCCL communication buffer size.
 * @param [in] groupSize: Packed quant group sizes (MX: [1,1,32]).
 * @param [in] commMode: Communication mode. Only "urma" is supported.
 * @param [out] output: Output tensor. Type: bfloat16 / float16. Shape: [worldSize * M_per_rank, N].
 * @param [out] workspaceSize: Computed workspace size.
 * @param [out] executor: Executor handle.
 * @return aclnnStatus: 0 on success.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAllGatherQuantMatmulV3GetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional, const aclTensor *x2ScaleOptional, const char *group, int64_t rankSize,
    int64_t hcclBufferSize, int64_t groupSize, const char *commMode, const aclTensor *output,
    const aclTensor *gatherOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Phase 2: execute AllGatherQuantMatmulV3 on the given stream.
 *
 * @param [in] workspace: Device workspace memory.
 * @param [in] workspaceSize: Workspace size (from GetWorkspaceSize).
 * @param [in] executor: Executor handle (from GetWorkspaceSize).
 * @param [in] stream: aclrtStream for kernel launch.
 * @return aclnnStatus: 0 on success.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAllGatherQuantMatmulV3(void *workspace,
                                                                               uint64_t workspaceSize,
                                                                               aclOpExecutor *executor,
                                                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ALL_GATHER_MATMUL_APACE_H_
