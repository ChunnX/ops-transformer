/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef OP_API_INC_RMS_NORM_QUANT_H_
#define OP_API_INC_RMS_NORM_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnRmsNormQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* scale, const aclTensor* offset,
    double epsilon, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnRmsNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_RMS_NORM_QUANT_H_
