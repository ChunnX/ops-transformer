/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "mhc_sandwich_norm_post_preonly.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace MhcSandwichNormPostPreonlyl0op {

OP_TYPE_REGISTER(MhcSandwichNormPostPreonly);

bool MhcSandwichNormPostPreonly(const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost,
                                       const aclTensor *hRes, const aclTensor *phi, const aclTensor *alpha0,
                                       const aclTensor *bias, const aclTensor *gamma0, const aclTensor *gamma1,
                                       const aclTensor *gamma2Optional, double normEps, double mhcPreNormEps,
                                       double hcEps, bool returnHInF32,
                                       const aclTensor *hInPrime, const aclTensor *x2Out, const aclTensor *hInF32,
                                       aclOpExecutor *executor)
{
    if (returnHInF32 == false) {
        hInF32 = executor->AllocTensor(hPost->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
        if (hInF32 == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "hInF32 AllocTensor failed.");
            return false;
        }
    }

    L0_DFX(MhcSandwichNormPostPreonly, hOut, residual, hPost, hRes, phi, alpha0, bias, gamma0, gamma1,
           gamma2Optional, hInPrime, x2Out, hInF32);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MhcSandwichNormPostPreonly,
        OP_INPUT(hOut, residual, hPost, hRes, phi, alpha0, bias, gamma0, gamma1, gamma2Optional),
        OP_OUTPUT(hInPrime, x2Out, hInF32),
        OP_ATTR(static_cast<float>(normEps), static_cast<float>(mhcPreNormEps), static_cast<float>(hcEps),
                returnHInF32));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return false,
                                         "MhcSandwichNormPostPreonly AddToLauncherList failed.");

    return true;
}

} // namespace l0op
