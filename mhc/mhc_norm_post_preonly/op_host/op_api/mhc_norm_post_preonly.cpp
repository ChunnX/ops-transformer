/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "mhc_norm_post_preonly.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace MhcNormPostPreonlyl0op {

OP_TYPE_REGISTER(MhcNormPostPreonly);

bool MhcNormPostPreonly(const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost,
                                const aclTensor *hResOptional, const aclTensor *phi, const aclTensor *alpha0,
                                const aclTensor *bias, const aclTensor *gamma0Optional, const aclTensor *gamma1,
                                const aclTensor *gamma2Optional, float normEps, float mhcPreNormEps, float hcEps,
                                bool returnHInF32, const aclTensor *hInPrimeOut, const aclTensor *x2Out,
                                const aclTensor *hInF32OutOptional, aclOpExecutor *executor)
{
    const aclTensor *hInF32ForKernel = hInF32OutOptional;
    if (hInF32ForKernel == nullptr) {
        hInF32ForKernel = executor->AllocTensor(hPost->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
        if (hInF32ForKernel == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "hInF32ForKernel AllocTensor failed.");
            return false;
        }
    }

    L0_DFX(MhcNormPostPreonly, hOut, residual, hPost, hResOptional, phi, alpha0, bias, gamma0Optional, gamma1,
           gamma2Optional, hInPrimeOut, x2Out, hInF32OutOptional);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MhcNormPostPreonly,
        OP_INPUT(hOut, residual, hPost, hResOptional, phi, alpha0, bias, gamma0Optional, gamma1, gamma2Optional),
        OP_OUTPUT(hInPrimeOut, x2Out, hInF32ForKernel),
        OP_ATTR(static_cast<float>(normEps), static_cast<float>(mhcPreNormEps), static_cast<float>(hcEps),
                returnHInF32));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return false,
                                         "MhcNormPostPreonly AddToLauncherList failed.");

    return true;
}

} // namespace MhcNormPostPreonlyl0op
