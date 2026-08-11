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
 * \file aclnn_allto_all_matmul_v2.h
 * \brief aclnn API for AlltoAllMatmulV2 (MX-quant FP8, apace UDMA path)
 */

#ifndef OP_API_INC_ALL_TO_ALL_MATMUL_APACE_H_
#define OP_API_INC_ALL_TO_ALL_MATMUL_APACE_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Phase 1: compute workspace size for AlltoAllQuantMatmulV3.
 * @domain aclnn_ops_infer
 * @param [in] context: apace communication context tensor. Type: int32/uint8.
 * @param [in] x1: Left matrix. Type: float8_e4m3fn / float8_e5m2. Shape: [M, K].
 * @param [in] x2: Right matrix. Type: float8_e4m3fn / float8_e5m2. Shape: [N, K].
 * @param [in] biasOptional: Optional bias (unused, must be nullptr). Type: float.
 * @param [in] x1ScaleOptional: MX scale for x1. Type: float8_e8m0.
 * @param [in] x2ScaleOptional: MX scale for x2. Type: float8_e8m0.
 * @param [in] group: Communication group identifier.
 * @param [in] worldSize: Number of ranks.
 * @param [in] hcclBufferSize: HCCL communication buffer size.
 * @param [in] x1QuantMode: x1 quantization mode. Default 6 (MX quant).
 * @param [in] x2QuantMode: x2 quantization mode. Default 6 (MX quant).
 * @param [in] groupSize: Group size for quantization. Default 0.
 * @param [in] commMode: Communication mode. Default "urma".
 * @param [in] precisionMode: Precision mode. Default 0.
 * @param [out] output: Output tensor. Type: bfloat16 / float16.
 * @param [out] alltoAllOutOptional: Optional all2all intermediate output.
 * @param [out] workspaceSize: Computed workspace size.
 * @param [out] executor: Executor handle.
 * @return aclnnStatus: 0 on success.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllQuantMatmulV3GetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional, const aclTensor *x2ScaleOptional, const char *group, int64_t worldSize,
    int64_t hcclBufferSize, int64_t x1QuantMode, int64_t x2QuantMode,
    int64_t groupSize, const char *commMode, int64_t precisionMode, const aclTensor *output,
    const aclTensor *alltoAllOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Phase 2: execute AlltoAllQuantMatmulV3 on the given stream.
 *
 * @param [in] workspace: Device workspace memory.
 * @param [in] workspaceSize: Workspace size (from GetWorkspaceSize).
 * @param [in] executor: Executor handle (from GetWorkspaceSize).
 * @param [in] stream: aclrtStream for kernel launch.
 * @return aclnnStatus: 0 on success.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllQuantMatmulV3(void *workspace, uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ALL_TO_ALL_MATMUL_APACE_H_