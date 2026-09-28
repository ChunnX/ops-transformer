/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_INPUT_TRACE_CONFIG_H
#define MEGA_MOE_INPUT_TRACE_CONFIG_H

#include <cstdint>

// Host、torch extension 和 kernel 使用相同配置并重新构建。
// 0：关闭；1：记录下发及已有完成边界；2：额外排空共享阶段流水测完成上界。
#ifndef MEGA_MOE_INPUT_TRACE
#define MEGA_MOE_INPUT_TRACE 0
#endif
#if MEGA_MOE_INPUT_TRACE < 0 || MEGA_MOE_INPUT_TRACE > 2
#error "MEGA_MOE_INPUT_TRACE must be 0, 1 or 2"
#endif

namespace MegaMoeImpl {
constexpr uint32_t INPUT_TRACE_MAGIC = 0x4D495452U; // MITR
constexpr uint32_t INPUT_TRACE_VERSION = 1U;
constexpr uint32_t INPUT_TRACE_HEADER_WORDS = 16U;
constexpr uint32_t INPUT_TRACE_RECORD_HEADER_WORDS = 8U;
constexpr uint32_t INPUT_TRACE_EVENT_WORDS = 9U;
constexpr uint32_t INPUT_TRACE_EVENT_CAPACITY = 32U;
constexpr uint32_t INPUT_TRACE_ROLE_COUNT = 3U;
constexpr uint32_t INPUT_TRACE_RECORD_WORDS =
    INPUT_TRACE_RECORD_HEADER_WORDS + INPUT_TRACE_EVENT_CAPACITY * INPUT_TRACE_EVENT_WORDS;
} // namespace MegaMoeImpl
#endif
