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
 * \file aclnn_causal_conv1d_backward.h
 * \brief ACLNN L2 API 接口声明 - CausalConv1dBackward
 *
 * 两段式设计：
 * - GetWorkspaceSize: 参数检查、workspace 计算、创建执行器
 * - aclnnCausalConv1dBackward: 执行计算
 */

#ifndef ACLNN_CAUSAL_CONV1D_BACKWARD_H_
#define ACLNN_CAUSAL_CONV1D_BACKWARD_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnCausalConv1dBackward 所需的 workspace 大小
 * @param gradY   [in]  上游梯度 [B, L, D] 或 [cu_seq_len, D]
 * @param x       [in]  前向输入，与 gradY 同形
 * @param weight  [in]  卷积权重 [W, D]，W∈{2,3,4}
 * @param queryStartLocOptional [in] 序列起始位置 [num_seq+1] int32，2D 输入必须提供
 * @param activation [in] 正向激活类型 "silu" 或 "none"
 * @param gradXOut      [out] 输入梯度，与 x 同形同 dtype
 * @param gradWeightOut [out] 权重梯度，与 weight 同形同 dtype
 * @param workspaceSize [out] 返回所需 workspace 大小
 * @param executor      [out] 返回执行器
 */
ACLNN_API aclnnStatus aclnnCausalConv1dBackwardGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *queryStartLocOptional,
    const char *activation,
    aclTensor *gradXOut,
    aclTensor *gradWeightOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief 执行 CausalConv1dBackward 算子计算
 */
ACLNN_API aclnnStatus aclnnCausalConv1dBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_CAUSAL_CONV1D_BACKWARD_H_
