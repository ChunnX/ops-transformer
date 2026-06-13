/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H_
#define OP_API_INC_INPLACE_PARTIAL_ROTARY_MUL_GRAD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplacePartialRotaryMulGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] dyRef: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,数据格式支持ND。
 * Inplace模式，dyRef同时作为输出写入结果。
 * @param [in] cos: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,数据类型与sin一致,
 * 数据格式支持ND。
 * @param [in] sin: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,数据类型与cos一致,shape与cos一致,
 * 数据格式支持ND。
 * @param [in] xOptional: npu
 * device侧的aclTensor, 可选输入, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,数据类型与dyRef一致。
 * 当前版本暂不支持，请传入空指针。
 * @param [in] rotaryMode: 支持int64_t类型, 旋转模式，0=half，1=interleave，2=quarter，3=interleave-half。
 * @param [in] partialSlice: 部分旋转的切片范围 [start, end)，作用于最后一维。
 * @param [in] dcosOut: npu
 * device侧的aclTensor, 正向计算输入cos的导数。当前版本暂不支持，输出无意义。
 * @param [in] dsinOut: npu
 * device侧的aclTensor, 正向计算输入sin的导数。当前版本暂不支持，输出无意义。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplacePartialRotaryMulGradGetWorkspaceSize(
    const aclTensor* dyRef, const aclTensor* cos, const aclTensor* sin, const aclTensor* xOptional,
    int64_t rotaryMode, const aclIntArray* partialSlice,
    aclTensor* dcosOut, aclTensor* dsinOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/* @brief aclnnInplacePartialRotaryMulGrad的第二段接口，用于执行计算.
 * @param [in] workspace: 在npu device侧申请的workspace内存起址
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplacePartialRotaryMulGrad(void* workspace, uint64_t workspaceSize,
                                                       aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
