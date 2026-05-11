/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_COMM_UTILS_H
#define MC2_COMM_UTILS_H

#include <cstdint>

namespace mc2 {

constexpr char COMM_MODE_ENV[] = "ENV_MC2_COMM_MODE_AICPU";

constexpr uint8_t COMM_MODE_CCU = 0;
constexpr uint8_t COMM_MODE_AICPU = 1;

uint8_t GetCommModeFromEnv();

}

#endif