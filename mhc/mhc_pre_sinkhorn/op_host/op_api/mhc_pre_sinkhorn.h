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
 * \file mhc_pre_sinkhorn.h
 * \brief mhc_pre_sinkhorn
 */

#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_MHC_PRE_SINKHORN_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_MHC_PRE_SINKHORN_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief MHC Pre Sinkhorn 算子 Level 0 接口
 * 
 * @param x 输入张量，BF16类型
 * @param phi 权重矩阵，FP32类型
 * @param alpha 缩放系数，FP32类型
 * @param bias 偏置向量，FP32类型
 * @param hcMult 多头数量
 * @param numIters Sinkhorn迭代次数
 * @param hcEps hc epsilon值
 * @param normEps 归一化epsilon值
 * @param needBackward 是否需要反向传播
 * 
 * 必选输出：
 * @param hin 输出张量，BF16类型
 * @param hPost 后处理变换矩阵，FP32类型
 * @param hRes 未做Sinkhorn的组合矩阵，FP32类型
 * 
 * 可选输出：
 * @param hPre sigmoid后的H_pre结果，FP32类型
 * @param hcBeforeNorm x @ phi矩阵乘法结果，FP32类型
 * @param invRms RmsNorm的1/r结果，FP32类型
 * @param sumOut Sinkhorn迭代中间sum结果，FP32类型
 * @param normOut Sinkhorn迭代中间norm结果，FP32类型
 * 
 * @param executor 执行器
 * @return const aclTensor* 返回hin输出张量
 */
const aclTensor *MhcPreSinkhorn(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                const aclTensor *bias, int64_t hcMult, int64_t numIters, float hcEps,
                                float normEps, bool needBackward, 
                                aclTensor *hin, aclTensor *hPost, aclTensor *hRes,
                                aclTensor *hPre, aclTensor *hcBeforeNorm, aclTensor *invRms,
                                aclTensor *sumOut, aclTensor *normOut,
                                aclOpExecutor *executor);
}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_MHC_PRE_SINKHORN_OP_H_
