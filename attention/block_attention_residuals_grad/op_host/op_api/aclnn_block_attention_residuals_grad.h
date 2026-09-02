/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_BLOCK_ATTENTION_RESIDUALS_GRAD_H
#define OP_API_INC_ACLNN_BLOCK_ATTENTION_RESIDUALS_GRAD_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBlockAttentionResidualsGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] partial_block: 表示公式中的partial_block，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [in] block_res: 表示公式中的block_res，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [in] proj_weight: 表示公式中的proj_weight，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [in] norm_weight: 表示公式中的norm_weight，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [in] grad_hidden_states:
 * 表示上游梯度grad_hidden_states，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [in] inv_norm: 表示逆RMS归一化系数inv_norm，数据类型支持FLOAT32，数据格式支持ND。
 * @param [in] probs: 表示softmax概率probs，数据类型支持FLOAT32，数据格式支持ND。
 * @param [in] validBlockNums: 表示预留属性，当前内核未使用，默认值为0。
 * @param [out] grad_partial_block: 表示partial_block的梯度，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [out] grad_block_res: 表示block_res的梯度，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [out] grad_proj_weight: 表示proj_weight的梯度，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [out] grad_norm_weight: 表示norm_weight的梯度，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBlockAttentionResidualsGradGetWorkspaceSize(
    const aclTensor *partial_block, const aclTensor *block_res, const aclTensor *proj_weight,
    const aclTensor *norm_weight, const aclTensor *grad_hidden_states, const aclTensor *inv_norm,
    const aclTensor *probs, int64_t validBlockNums, const aclTensor *grad_partial_block,
    const aclTensor *grad_block_res, const aclTensor *grad_proj_weight, const aclTensor *grad_norm_weight,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnBlockAttentionResidualsGrad的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnBlockAttentionResidualsGradGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBlockAttentionResidualsGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_BLOCK_ATTENTION_RESIDUALS_GRAD_H
