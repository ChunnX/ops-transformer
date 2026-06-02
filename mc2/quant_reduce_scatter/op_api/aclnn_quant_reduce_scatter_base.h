/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
**/

/*!
 * \file aclnn_quant_reduce_scatter_base.h
 * \brief
 */
#ifndef OP_API_INC_QUANT_REDUCE_SCATTER_BASE_
#define OP_API_INC_QUANT_REDUCE_SCATTER_BASE_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnQuantReduceScatterBaseGetWorkspaceSize(const aclTensor* x, const aclTensor* scales,
                                                                  const char* group, const char* reduceOp,
                                                                  aclTensor* output, uint64_t* workspaceSize,
                                                                  aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnQuantReduceScatterBase(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                  const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_REDUCE_SCATTER_BASE_