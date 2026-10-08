/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_GROUPED_MATMUL_ACTIVATION_WEIGHT_NZ_H
#define OP_API_ACLNN_GROUPED_MATMUL_ACTIVATION_WEIGHT_NZ_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief GroupedMatmulActivationWeightNZ 的第一段接口。
 *
 * 对外参数仅包含 x、xScale、groupList、weight、weightScale、bias 和 y，属性为
 * dequantMode、dequantDtype、actType、transposeWeight、groupListType、tuningConfig、
 * swigluMode、clampLimit、gluAlpha、gluBias。
 *
 * 当前仅支持 Ascend 950 上的 MXFP4 A4W4 WeightNZ SwiGLU：x/weight 为
 * FLOAT4_E2M1，xScale/weightScale 为 FLOAT8_E8M0，y 为 BF16。实现复用既有
 * concat-N TensorAPI worker；yScale、辅助矩阵、smooth、round 和 scale 算法均为内部固定参数。
 * swigluMode=0 执行不带 clamp 的传统 SwiGLU；swigluMode=2 执行现有带 clamp 的前后半分支 SwiGLU。
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
    const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList, const aclTensorList* weight,
    const aclTensorList* weightScale, const aclTensor* bias, int64_t dequantMode, int64_t dequantDtype, int64_t actType,
    bool transposeWeight, int64_t groupListType, const aclIntArray* tuningConfig, int64_t swigluMode, double clampLimit,
    double gluAlpha, double gluBias, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnGroupedMatmulActivationWeightNz(void* workspace, uint64_t workspaceSize,
                                                           aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_GROUPED_MATMUL_ACTIVATION_WEIGHT_NZ_H
