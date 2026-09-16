/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#ifndef MEGA_MOE_A8W4_L0_REUSE
#define MEGA_MOE_A8W4_L0_REUSE 1
#endif
#ifndef MEGA_MOE_A8W4_EARLY_FIX
#define MEGA_MOE_A8W4_EARLY_FIX 1
#endif
#ifndef MEGA_MOE_PROLOGUE_PREFETCH
#define MEGA_MOE_PROLOGUE_PREFETCH 1
#endif
namespace Blaze::Gemm {
constexpr uint64_t MX_FP8FP4_M_STRIP = 128;
}
