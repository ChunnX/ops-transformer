/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_comm_utils.h"
#include <cstdlib>
#include "mc2_log.h"

namespace mc2 {

uint8_t GetCommModeFromEnv()
{
    auto commModeEnv = getenv(COMM_MODE_ENV);
    uint8_t commMode = COMM_MODE_CCU;
    if (commModeEnv != nullptr) {
        commMode = COMM_MODE_AICPU;
    }
    OPS_LOG_D("", "Get ENV_MC2_COMM_MODE_AICPU is %u (0=CCU, 1=AICPU)", commMode);
    return commMode;
}

}