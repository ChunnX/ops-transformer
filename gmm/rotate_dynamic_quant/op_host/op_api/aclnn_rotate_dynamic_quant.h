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
 *
 * 算子功能：RotateDynamicQuant正向算子，融合旋转和动态量化
 *
 * @param [in] x: 输入tensor，shape为[M, N]，数据类型支持：DT_FLOAT16, DT_BFLOAT16。进行int4量化时，要求形状最后一维是8的整数倍，支持非连续tensor。
 * @param [in] rot: 旋转矩阵tensor，shape为[N, N]，数据类型支持：DT_FLOAT16, DT_BFLOAT16，与x保持一致，支持非连续tensor。
 * @param [in] dst_type: 属性，指定输出y的数据类型，0=DT_INT8, 1=DT_INT32，默认为0。
 * @param [out] yOut: 输出tensor，shape为[M, N]，数据类型：DT_INT8或DT_INT32，shape与输入x一致。
 * @param [out] scaleOut: 输出tensor，shape为[M]，数据类型：DT_FLOAT，表示动态量化过程中计算出的缩放系数。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnRotateDynamicQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *rot,
    int32_t dst_type,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnRotateDynamicQuant 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 aclnnRotateDynamicQuantGetWorkspaceSize 获取。
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
