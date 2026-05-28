/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNNINNER_ALLTO_ALLV_GROUPED_MAT_MUL_V2_H_
#define ACLNNINNER_ALLTO_ALLV_GROUPED_MAT_MUL_V2_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInnerAlltoAllvGroupedMatMulV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @param [in] gmmX: 计算输入，Tensor，数据类型支持float16，bfloat16
 * @param [in] gmmWeight: 计算输入，Tensor，数据类型支持float16, bfloat16
 * @param [in] sendCountsTensorOptional: 可选入参，计算输入，Tensor
 * @param [in] recvCountsTensorOptional: 可选入参，计算输入，Tensor
 * @param [in] mmXOptional: 可选入参，计算输入，Tensor
 * @param [in] mmWeightOptional: 可选入参，计算输入，Tensor
 * @param [in] group: 计算输入，str。ep通信域名称
 * @param [in] epWorldSize: 计算输入，int。ep通信域size
 * @param [in] sendCounts: 计算输入，list int。通信发送的数据量
 * @param [in] recvCounts: 计算输入，list int。通信接受的数据量
 * @param [in] transGmmWeight: 可选入参，计算输入。表明gmm的右矩阵是否需要转置
 * @param [in] transMmWeight: 可选入参，计算输入。表明共享专家mm的右矩阵是否需要转置
 * @param [in] permuteOutFlag: 可选入参，计算输入。表明是否需要输出permuteOut
 * @param [in] commMode: 计算输入，str。通信模式，支持 '', 'ccu', 'ai_cpu'
 * @param [out] gmmY: 计算输出，Tensor，数据类型支持float16, bfloat16
 * @param [out] mmYOptional: 可选输出，计算输出，Tensor
 * @param [out] permuteOutOptional: 可选输出，计算输出，Tensor
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程
 * @return aclnnStatus: 返回值，返回状态码
 */
aclnnStatus aclnnInnerAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional,
    const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
    char* group, int64_t epWorldSize, const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, char* commMode,
    aclTensor* gmmY, aclTensor* mmYOptional, aclTensor* permuteOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInnerAlltoAllvGroupedMatMulV2的第二段接口，用于执行计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小
 * @param [in] executor: op执行器，包含了算子计算流程
 * @param [in] stream: acl stream流
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnInnerAlltoAllvGroupedMatMulV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNNINNER_ALLTO_ALLV_GROUPED_MAT_MUL_V2_H_