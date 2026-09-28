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
 * \file mega_mhc.cpp
 * \brief Registers the MegaMhc operator with the L0 execution framework.
 */

#include "mega_mhc.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace MegaMhcl0op {

OP_TYPE_REGISTER(MegaMhc);

/** Launch RMS0 -> Post -> optional Mid -> Pre -> RMS1 with supplied contiguous output views. */
bool MegaMhc(const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost, const aclTensor *hRes,
             const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *gamma1,
             const aclTensor *preMixOptional, const aclTensor *gamma0Optional, const aclTensor *gamma2Optional,
             double normEps, double mhcPreNormEps, double hcEps, int64_t computeMode, const aclTensor *hInPrimeOut,
             const aclTensor *hPostNewOut, const aclTensor *hResNewOut, const aclTensor *x2Out,
             const aclTensor *hPreOut, aclOpExecutor *executor)
{
    L0_DFX(MegaMhc, hOut, residual, hPost, hRes, phi, alpha, bias, gamma1, preMixOptional, gamma0Optional,
           gamma2Optional, normEps, mhcPreNormEps, hcEps, computeMode, hInPrimeOut, hPostNewOut, hResNewOut, x2Out,
           hPreOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MegaMhc,
        OP_INPUT(hOut, residual, hPost, hRes, phi, alpha, bias, gamma1, preMixOptional, gamma0Optional, gamma2Optional),
        OP_OUTPUT(hInPrimeOut, hPostNewOut, hResNewOut, x2Out, hPreOut),
        OP_ATTR(static_cast<float>(normEps), static_cast<float>(mhcPreNormEps), static_cast<float>(hcEps),
                computeMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(static_cast<int32_t>(ret), "AddToLauncherList result invalid: actual=%d, expected=%d(ACLNN_SUCCESS)",
                static_cast<int32_t>(ret), static_cast<int32_t>(ACLNN_SUCCESS));
        return false;
    }
    return true;
}

} // namespace MegaMhcl0op
