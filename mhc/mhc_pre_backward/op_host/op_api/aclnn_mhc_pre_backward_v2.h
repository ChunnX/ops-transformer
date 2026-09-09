/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MHC_PRE_BACKWARD_V2_H
#define OP_API_INC_MHC_PRE_BACKWARD_V2_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first-stage interface of aclnnMhcPreBackwardV2. Calculates the required workspace size.
 * @domain aclnn_ops_infer
 * @param [in] x: Input tensor.
 * @param [in] phi: Input projection tensor.
 * @param [in] alpha: Input scaling tensor.
 * @param [in] gradHIn: Input gradient of hIn.
 * @param [in] gradHPost: Input gradient of hPost.
 * @param [in] gradHRes: Input gradient of hRes.
 * @param [in] invRms: Forward intermediate tensor.
 * @param [in] hMix: Forward intermediate tensor.
 * @param [in] hPre: Forward intermediate tensor.
 * @param [in] hPost: Forward output tensor.
 * @param [in] gamma: Optional RMSNorm scale tensor.
 * @param [in] gradXPostOptional: Optional input gradient.
 * @param [in] hcEps: Hyper-connection epsilon.
 * @param [in] opImplMode: Cube precision mode. 0 uses FP32 and 1 uses HF32.
 * @param [out] gradX: Output gradient of x.
 * @param [out] gradPhi: Output gradient of phi.
 * @param [out] gradAlpha: Output gradient of alpha.
 * @param [out] gradBias: Output gradient of bias.
 * @param [out] gradGamma: Optional output gradient of gamma.
 * @param [out] workspaceSize: Required device workspace size.
 * @param [out] executor: Operator executor.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnMhcPreBackwardV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *gradHIn,
    const aclTensor *gradHPost, const aclTensor *gradHRes, const aclTensor *invRms, const aclTensor *hMix,
    const aclTensor *hPre, const aclTensor *hPost, const aclTensor *gamma, const aclTensor *gradXPostOptional,
    float hcEps, int64_t opImplMode, const aclTensor *gradX, const aclTensor *gradPhi, const aclTensor *gradAlpha,
    const aclTensor *gradBias, const aclTensor *gradGamma, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second-stage interface of aclnnMhcPreBackwardV2. Executes the operator.
 * @domain aclnn_ops_infer
 * @param [in] workspace: Device workspace address.
 * @param [in] workspaceSize: Workspace size returned by aclnnMhcPreBackwardV2GetWorkspaceSize.
 * @param [in] executor: Operator executor returned by aclnnMhcPreBackwardV2GetWorkspaceSize.
 * @param [in] stream: ACL stream used to execute the operator.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnMhcPreBackwardV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MHC_PRE_BACKWARD_V2_H
