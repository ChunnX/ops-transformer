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
 * \file aclnn_mhc_pre_sinkhorn_backward.h
 * \brief MhcPreSinkhornBackward ACLNN API
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_MHC_PRE_SINKHORN_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_MHC_PRE_SINKHORN_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPreSinkhornBackwardGetWorkspaceSize的第一段接口，计算workspace大小。
 * 功能描述：该算子实现Manifold-Constraint Hyper-Connection的Pre-Sinkhorn反向传播，
 *          计算输入梯度和参数梯度。
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnMhcPreSinkhornBackwardGetWorkspaceSize(
    const aclTensor *grad_hin, const aclTensor *grad_h_post, const aclTensor *grad_h_res,
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
    const aclTensor *h_pre, const aclTensor *hc_before_norm, const aclTensor *inv_rms,
    const aclTensor *sum_out, const aclTensor *norm_out,
    double hc_eps,
    aclTensor *grad_x, aclTensor *grad_phi, aclTensor *grad_alpha, aclTensor *grad_bias,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMhcPreSinkhornBackward的第二段接口，用于执行计算。
 */
aclnnStatus aclnnMhcPreSinkhornBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_MHC_PRE_SINKHORN_BACKWARD_H_
