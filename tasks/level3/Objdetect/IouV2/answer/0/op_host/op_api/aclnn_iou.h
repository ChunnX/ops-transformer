/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_IOU_H_
#define OP_API_INC_LEVEL2_ACLNN_IOU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief aclnnIou的第一段接口，根据具体的计算流程，计算workspace大小。
     * @domain aclnn_ops_train
     *
     * 算子功能：对两个输入矩形框集合，计算交并比(IOU)或前景交叉比(IOF)，用于评价预测框(bBox)和真值框(gtBox)的重叠度。
     *
     * @param [in] bBoxes: npu device侧的aclTensor，预测矩形框，数据类型支持FLOAT32，FLOAT16，BFLOAT16，
     * 支持非连续的Tensor，数据格式支持ND。
     * @param [in] gtBoxes: npu device侧的aclTensor，真值矩形框，数据类型支持FLOAT32，FLOAT16，BFLOAT16，
     * 支持非连续的Tensor，数据格式支持ND。
     * @param [in] mode: host侧的char*类型，用于选择计算方式"iou"或"iof"。
     * @param [in] eps: host侧的float类型，防止除零，计算面积时长和宽都会加上eps。
     * @param [in] aligned: host侧的Bool类型，用于标识两个输入的shape是否相同。
     * @param [out] overlap: npu device侧的aclTensor，数据类型支持FLOAT32，FLOAT16，BFLOAT16，
     * 数据类型、数据格式、tensor shape需要与bBoxes保持一致。
     * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
     * @param [out] executor: 返回op执行器，包含算子计算流程。
     * @return aclnnStatus: 返回状态码。
     */
    ACLNN_API aclnnStatus aclnnIouGetWorkspaceSize(const aclTensor *bBoxes, const aclTensor *gtBoxes,
                                                   const char *mode, float eps, bool aligned,
                                                   aclTensor *overlap, uint64_t *workspaceSize, aclOpExecutor **executor);

    /**
     * @brief aclnnIou的第二段接口，用于执行计算。
     * @param [in] workspace: 在npu device侧申请的workspace内存起址。
     * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnIouGetWorkspaceSize获取。
     * @param [in] exector: op执行器，包含了算子计算流程。
     * @param [in] stream: acl stream流。
     * @return aclnnStatus: 返回状态码
     */
    ACLNN_API aclnnStatus aclnnIou(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_IOU_H_
