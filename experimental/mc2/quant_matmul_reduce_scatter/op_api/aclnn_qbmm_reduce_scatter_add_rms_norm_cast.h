/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_
#define OP_API_INC_QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现xxx。
 * @brief aclnnQbmmReduceScatterAddRmsNormCast的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x1: 计算输入，Tensor，数据类型int8，数据格式支持ND。xx解释。
 * @param [in] x2: 计算输入，Tensor，数据类型int32，数据格式支持ND/NZ。xx解释。
 * @param [in] y: 计算输入，Tensor，数据类型bfloat16，数据格式支持ND。
 * @param [in] gamma: 计算输入，Tensor，数据类型float32，数据格式支持ND。
 * @param [in] scale: 计算输入，Tensor，数据类型bfloat16，数据格式支持ND。
 * @param [in] bias: 计算可选输入，Tensor，数据类型bfloat16, int32, float16, float32，数据格式支持ND。
 * @param [in] pertoken_scale: 计算可选输入，Tensor，数据类型bfloat16，数据格式支持ND。
 * @param [out] y1: 计算输出，Tensor，必选输出，数据类型支持float32，数据格式支持ND。
 * @param [out] y2: 计算输出，Tensor，必选输出，数据类型float16, bfloat16，数据格式支持ND。
 * @param [out] x: 计算输出，Tensor，必选输出，数据类型float16, bfloat16，数据格式支持ND。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 */
ACLNN_API aclnnStatus aclnnQbmmReduceScatterAddRmsNormCastGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* y,
                                                                           const aclTensor* gamma, const aclTensor* scale, const aclTensor* bias, 
                                                                           const aclTensor* pertokenScale, 
                                                                           const char* group, int64_t rankSize, bool transposeX2, int64_t dtype, float epsilon,
                                                                           aclTensor* y1, aclTensor* y2, aclTensor* x, 
                                                                           uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnQbmmReduceScatterAddRmsNormCast的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnQbmmReduceScatterAddRmsNormCastGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQbmmReduceScatterAddRmsNormCast(void* workspace, uint64_t workspaceSize,
                                                           aclOpExecutor* executor, const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_