/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_EP_DISPATCH_SCHEDULE_H
#define MOE_EP_DISPATCH_SCHEDULE_H

#include <cstdint>

namespace MoeEpDispatchSchedule {
struct Range {
    uint32_t begin;
    uint32_t end;
};

// Callers validate count > 0 and index < count before launching the kernel.
__aicore__ inline Range Partition(uint32_t total, uint32_t count, uint32_t index)
{
    uint32_t base = total / count;
    uint32_t remainder = total % count;
    uint32_t begin = base * index + (index < remainder ? index : remainder);
    return {begin, begin + base + (index < remainder ? 1U : 0U)};
}

// token must be strictly below total; empty partitions cannot own a token.
__aicore__ inline uint32_t Owner(uint32_t total, uint32_t count, uint32_t token)
{
    uint32_t base = total / count;
    uint32_t remainder = total % count;
    uint32_t largerTokens = remainder * (base + 1U);
    if (token < largerTokens) {
        return token / (base + 1U);
    }
    return remainder + (token - largerTokens) / base;
}

__aicore__ inline Range Overlap(uint32_t total, uint32_t count, uint32_t begin, uint32_t end)
{
    if (begin == end) {
        return {0U, 0U};
    }
    return {Owner(total, count, begin), Owner(total, count, end - 1U) + 1U};
}
} // namespace MoeEpDispatchSchedule

#endif
