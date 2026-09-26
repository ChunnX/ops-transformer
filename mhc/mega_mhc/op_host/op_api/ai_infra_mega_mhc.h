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
 * \file ai_infra_mega_mhc.h
 * \brief Declares the AiInfraMegaMhc L0 operator interface.
 */

#ifndef AI_INFRA_MEGA_MHC_L0OP_H_
#define AI_INFRA_MEGA_MHC_L0OP_H_

#include <cstdint>

#include "opdev/op_executor.h"

namespace AiInfraMegaMhcl0op {

/** Launch RMS0 -> Post -> optional Mid -> Pre -> RMS1 with supplied contiguous output views. */
bool AiInfraMegaMhc(const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost, const aclTensor *hRes,
    const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *gamma1,
    const aclTensor *gamma0Optional, const aclTensor *gamma2Optional, double normEps, double hcEps, int64_t computeMode,
    const aclTensor *hInPrimeOut, const aclTensor *hPostNewOut, const aclTensor *hResNewOut, const aclTensor *x2Out,
    aclOpExecutor *executor);

}  // namespace AiInfraMegaMhcl0op

#endif  // AI_INFRA_MEGA_MHC_L0OP_H_
