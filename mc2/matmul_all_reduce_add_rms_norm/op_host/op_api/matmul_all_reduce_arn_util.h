/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_ARN_UTIL_H_
#define OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_ARN_UTIL_H_

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "acl/acl.h"
#include "matmul_all_reduce/op_host/op_api/matmul_all_reduce_util.h"

#ifdef __cplusplus
extern "C" {
#endif

// MatmulAllReduceAddRmsNorm
bool ArnCheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* residual, const aclTensor* gamma);
bool ArnCheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* residual);

aclnnStatus InnerMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffset, const aclTensor* dequant, const aclTensor* residual, const aclTensor* gamma,
    double epsilon, const char* group, const char* reduceOp, int64_t commTurn, int64_t antiquantGroupSize,
    const aclTensor* y, const aclTensor* normOut, uint64_t* workspaceSize, aclOpExecutor** executor);

#ifdef __cplusplus
}
#endif

#endif // OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_ARN_UTIL_H_