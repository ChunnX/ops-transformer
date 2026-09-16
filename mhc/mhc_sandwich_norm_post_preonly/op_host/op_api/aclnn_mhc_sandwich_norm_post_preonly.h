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

aclnnStatus aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize(
    const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost, const aclTensor *hRes,
    const aclTensor *phi, const aclTensor *alpha0, const aclTensor *bias, const aclTensor *gamma0,
    const aclTensor *gamma1, const aclTensor *gamma2Optional, double normEps, double mhcPreNormEps,
    double hcEps, bool returnHInF32, const aclTensor *hInPrime, const aclTensor *x2Out,
    const aclTensor *hInF32, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnMhcSandwichNormPostPreonly(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_MHC_SANDWICH_NORM_POST_PREONLY_H_
