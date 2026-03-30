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
 * \file aclnn_mmhc_pre_sinkhorn_sinkhorn.h
 * \brief aclnn_mmhc_pre_sinkhorn_sinkhorn
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_MMHC_PRE_SINKHORN_SINKHORN_H_
#define OP_API_INC_LEVEL2_ACLNN_MMHC_PRE_SINKHORN_SINKHORN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPreSinkhorn的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnMhcPreSinkhornGetWorkspaceSize(const aclTensor *x, float eps, int64_t numIters,
                                                          int64_t outFlag, aclTensor *output, aclTensor *normOut,
                                                          aclTensor *sumOut, uint64_t *workspaceSize,
                                                          aclOpExecutor **executor);

/**
 * @brief aclnnMhcPreSinkhorn的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnMhcPreSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_MMHC_PRE_SINKHORN_SINKHORN_H_
