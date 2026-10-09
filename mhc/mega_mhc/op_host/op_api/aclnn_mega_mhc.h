/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_mega_mhc.h
 * \brief Declares the public MegaMhc aclnn workspace and execution interfaces.
 */

#ifndef OP_API_INC_ACLNN_MEGA_MHC_H_
#define OP_API_INC_ACLNN_MEGA_MHC_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMegaMhcGetWorkspaceSize(const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost,
                                         const aclTensor *hRes, const aclTensor *phi, const aclTensor *alpha,
                                         const aclTensor *bias, const aclTensor *gamma1,
                                         const aclTensor *preMixOptional, const aclTensor *gamma0Optional,
                                         const aclTensor *gamma2Optional, double normEps, double mhcPreNormEps,
                                         double hcEps, int64_t computeMode, const aclTensor *hInPrimeOut,
                                         const aclTensor *hPostNewOut, const aclTensor *hResNewOut,
                                         const aclTensor *x2Out, const aclTensor *hPreOut, uint64_t *workspaceSize,
                                         aclOpExecutor **executor);

aclnnStatus aclnnMegaMhc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_MEGA_MHC_H_
