/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_ROTATE_DYNAMIC_QUANT_H
#define OP_API_INC_ACLNN_ROTATE_DYNAMIC_QUANT_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRotateDynamicQuantGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：融合旋转位置编码和动态量化的算子
 * @param [in] x: 输入tensor，shape为[M, N]，数据类型支持：DT_FLOAT16, DT_BFLOAT16。
 * @param [in] rot: 旋转矩阵tensor，shape为[N, N]，数据类型与x保持一致。
 * @param [in] dstType: 输出yOut的数据类型属性，0表示DT_INT8，1表示DT_INT32。
 * @param [out] yOut: 输出tensor，shape为[M, N]，数据类型由dstType决定。
 * @param [out] scaleOut: 输出scale tensor，shape为[M]，数据类型为DT_FLOAT。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnRotateDynamicQuantGetWorkspaceSize(
    const aclTensor *x, const aclTensor *rot,
    int64_t dstType,
    aclTensor *yOut, aclTensor *scaleOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnRotateDynamicQuant 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnRotateDynamicQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ACLNN_ROTATE_DYNAMIC_QUANT_H
