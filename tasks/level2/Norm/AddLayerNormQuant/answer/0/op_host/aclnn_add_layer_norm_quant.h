/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef OP_API_INC_LEVEL2_ADD_LAYER_NORM_QUANT_H_
#define OP_API_INC_LEVEL2_ADD_LAYER_NORM_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta,
    const aclTensor* biasOptional, const aclTensor* scales1Optional, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const char* quantMode, double epsilon,
    bool additionalOutput, bool divMode, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* outScales1Out,
    aclTensor* outScales2Out, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnAddLayerNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ADD_LAYER_NORM_QUANT_H_
