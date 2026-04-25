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
 * \file aclnn_mhc_pre_sinkhorn.h
 * \brief aclnn_mhc_pre_sinkhorn
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_MHC_PRE_SINKHORN_H_
#define OP_API_INC_LEVEL2_ACLNN_MHC_PRE_SINKHORN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPreSinkhorn的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 
 * @param x 输入张量，BF16类型
 * @param phi 权重矩阵，FP32类型
 * @param alpha 缩放系数，FP32类型
 * @param bias 偏置向量，FP32类型
 * @param hcMult 多头数量
 * @param numIters Sinkhorn迭代次数
 * @param hcEps hc epsilon值
 * @param normEps 归一化epsilon值
 * @param needBackward 是否需要反向传播（控制可选输出）
 * 
 * 必选输出：
 * @param hin 输出张量，BF16类型，Attention/MLP层输入
 * @param hPost 后处理变换矩阵，FP32类型
 * @param hRes 未做Sinkhorn的组合矩阵，FP32类型
 * 
 * 可选输出（needBackward=true时有效）：
 * @param hPre sigmoid后的H_pre结果，FP32类型
 * @param hcBeforeNorm x @ phi矩阵乘法结果，FP32类型
 * @param invRms RmsNorm的1/r结果，FP32类型
 * @param sumOut Sinkhorn迭代中间sum结果，FP32类型
 * @param normOut Sinkhorn迭代中间norm结果，FP32类型
 * 
 * @param workspaceSize workspace大小
 * @param executor 执行器
 */
ACLNN_API aclnnStatus aclnnMhcPreSinkhornGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
    int64_t hcMult, int64_t numIters, float hcEps, float normEps, bool needBackward,
    aclTensor *hin, aclTensor *hPost, aclTensor *hRes,
    aclTensor *hPre, aclTensor *hcBeforeNorm, aclTensor *invRms,
    aclTensor *sumOut, aclTensor *normOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMhcPreSinkhorn的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnMhcPreSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_MHC_PRE_SINKHORN_H_
