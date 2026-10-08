/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_causal_conv1d_grad.h
 * @brief ACLNN L2 API 接口声明 - CausalConv1dGrad
 *
 * 两段式设计：
 * - GetWorkspaceSize: 参数检查、workspace 计算、创建执行器
 * - aclnnCausalConv1dGrad: 执行计算
 */

#ifndef ACLNN_CAUSAL_CONV1D_GRAD_H_
#define ACLNN_CAUSAL_CONV1D_GRAD_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnCausalConv1dGrad 所需的 workspace 大小
 * @param gradY   [in]  上游梯度 [B, T, D]
 * @param x       [in]  前向输入 [B, T, D]
 * @param weight  [in]  卷积权重 [kW, D]
 * @param bias    [in]  偏置 [D]
 * @param dx      [out] 输入梯度 [B, T, D]
 * @param dweight [out] 权重梯度 [kW, D]
 * @param dbias   [out] 偏置梯度 [D]
 * @param workspaceSize [out] 返回所需 workspace 大小
 * @param executor      [out] 返回执行器
 */
ACLNN_API aclnnStatus aclnnCausalConv1dGradGetWorkspaceSize(
    const aclTensor* gradY,
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* bias,
    const aclTensor* dx,
    const aclTensor* dweight,
    const aclTensor* dbias,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief 执行 CausalConv1dGrad 算子计算
 */
ACLNN_API aclnnStatus aclnnCausalConv1dGrad(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_CAUSAL_CONV1D_GRAD_H_
