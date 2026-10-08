/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OP_API_INC_ACLNN_MHC_SANDWICH_NORM_POST_PREONLY_H_
#define OP_API_INC_ACLNN_MHC_SANDWICH_NORM_POST_PREONLY_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize 的第一段接口，根据计算流程计算 workspace 大小。
 * @domain aclnn_ops_infer
 * 算子功能：融合 RMSNorm_0、MHC_Post、可选 RMSNorm_mid、MHC_Pre（pre 分支）和 RMSNorm_1。
 * @param [in] hOut: post 输入，shape为[T,D]或[B,S,D]，数据类型支持：FLOAT16、BFLOAT16。
 * @param [in] residual: residual输入，shape为[T,N,D]或[B,S,N,D]，数据类型与hOut一致。
 * @param [in] hPost: post权重，shape为[T,N]或[B,S,N]，数据类型：FLOAT。
 * @param [in] hResOptional: 可选的mHC hRes变换矩阵，是做完sinkhorn变换后的双随机矩阵；
 * 该参数为接口预留，暂不支持传nullptr，shape为[T,N,N]或[B,S,N,N]，数据类型：FLOAT。
 * @param [in] phi: pre投影参数，shape为[N,N*D]，数据类型：FLOAT。
 * @param [in] alpha0: pre缩放系数，shape为[1]，数据类型：FLOAT。
 * @param [in] bias: pre偏置，shape为[N]，数据类型：FLOAT。
 * @param [in] gamma1: 第二次RMSNorm权重，shape为[D]，数据类型：FLOAT。
 * @param [in] gamma0Optional: 可选的第一次RMSNorm权重，shape为[D]，数据类型：FLOAT。
 * @param [in] gamma2Optional: 可选的中间RMSNorm权重，传nullptr时跳过，数据类型：FLOAT。
 * @param [in] normEps: RMSNorm数值稳定项，默认值为1e-6。
 * @param [in] mhcPreNormEps: MHC_Pre内部RMS数值稳定项，默认值为1e-6。
 * @param [in] hcEps: Sigmoid门控结果的加性偏移，默认值为1e-6。
 * @param [in] returnHInF32: 是否返回FLOAT类型中间结果hInF32OutOptional。
 * @param [out] hInPrimeOut: 输出tensor，shape为[T,D]或[B,S,D]，数据类型与hOut一致。
 * @param [out] x2Out: 输出tensor，shape为[T,N,D]或[B,S,N,D]，数据类型与hOut一致。
 * @param [out] hInF32OutOptional: 可选输出tensor，returnHInF32为true时shape为[T,D]或[B,S,D]，
 * 数据类型为FLOAT；否则传nullptr。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
aclnnStatus aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize(
    const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost, const aclTensor *hResOptional,
    const aclTensor *phi, const aclTensor *alpha0, const aclTensor *bias, const aclTensor *gamma1,
    const aclTensor *gamma0Optional, const aclTensor *gamma2Optional, float normEps, float mhcPreNormEps, float hcEps,
    bool returnHInF32, const aclTensor *hInPrimeOut, const aclTensor *x2Out, const aclTensor *hInF32OutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMhcSandwichNormPostPreonly 的第二段接口。
 * @param [in] workspace: npu device侧workspace内存起址，workspaceSize为0时可传nullptr。
 * @param [in] workspaceSize: workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，由第一段接口获取。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
aclnnStatus aclnnMhcSandwichNormPostPreonly(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_MHC_SANDWICH_NORM_POST_PREONLY_H_
