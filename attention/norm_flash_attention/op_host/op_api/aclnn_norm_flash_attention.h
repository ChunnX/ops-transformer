/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_NORM_FLASH_ATTENTION_H
#define OP_API_INC_NORM_FLASH_ATTENTION_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnNormFlashAttentionGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：NormFlashAttention正向算子
 * @param [in] query: 输入tensor，shape为[B, S, H, D]，数据类型支持：DT_FLOAT16, DT_FLOAT, DT_BF16。
 * @param [in] key: 输入tensor，shape为[B, S, H, D]，数据类型支持：DT_FLOAT16, DT_FLOAT, DT_BF16。
 * @param [in] value: 输入tensor，shape为[B, S, H, D]，数据类型支持：DT_FLOAT16, DT_FLOAT, DT_BF16。
 * @param [out] attentionOut: 输出tensor，shape为[B, S, H, D]，数据类型：DT_FLOAT16, DT_FLOAT, DT_BF16。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnNormFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnNormFlashAttention 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 aclnnNormFlashAttentionGetWorkspaceSize 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnNormFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_NORM_FLASH_ATTENTION_H
