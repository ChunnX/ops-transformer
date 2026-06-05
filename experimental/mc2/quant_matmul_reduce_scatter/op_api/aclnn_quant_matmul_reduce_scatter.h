/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_QUANT_MATMUL_REDUCE_SCATTER_
#define OP_API_INC_QUANT_MATMUL_REDUCE_SCATTER_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief QBMM + ReduceScatter 融合算子
 * @param [in] x1: fp8 A 矩阵 [M, K]，数据类型 FLOAT8_E4M3FN，格式 ND
 * @param [in] x2: fp8 B/权重矩阵，数据类型 FLOAT8_E4M3FN，格式 ND 或 FRACTAL_NZ
 * @param [in] x1Scale: A 矩阵 e8m0 scale，数据类型 FLOAT8_E8M0，格式 ND
 * @param [in] x2Scale: B 矩阵 e8m0 scale，数据类型 FLOAT8_E8M0，格式 ND
 * @param [in] group: 通信域标识，字符串
 * @param [in] rankSize: 通信域大小，int64
 * @param [in] transposeX2: x2 是否转置，bool
 * @param [out] y: ReduceScatter 输出 bf16 [M/rs, N]，数据类型 BFLOAT16，格式 ND
 * @param [out] workspaceSize: Device 侧 workspace 大小
 * @param [out] executor: op 执行器
 * @return aclnnStatus: 状态码
 */
ACLNN_API aclnnStatus aclnnQuantMatmulReduceScatterGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale,
    const char* group, int64_t rankSize, bool transposeX2,
    aclTensor* x, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnQuantMatmulReduceScatter(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_REDUCE_SCATTER_
