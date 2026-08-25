/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_trace_utils.h
 * \brief MegaMoe 诊断打点的通用事件写入与区间记录接口。
 */

#ifndef MEGA_MOE_TRACE_UTILS_H
#define MEGA_MOE_TRACE_UTILS_H

#include <cstdint>

#include "kernel_operator.h"
#include "mega_moe_trace_config.h"

namespace MegaMoeImpl {

#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0

struct CriticalPathTraceEventWriter {
    __gm__ uint32_t *record = nullptr;
    uint32_t *eventCount = nullptr;
    uint32_t *truncated = nullptr;
    uint32_t eventCapacity = 0U;
    uint32_t recordHeaderWords = 0U;
    uint32_t eventWords = 0U;

    __aicore__ inline bool IsEnabled() const
    {
        return record != nullptr && eventCount != nullptr && truncated != nullptr;
    }

    __aicore__ inline void Record(uint32_t eventType, uint32_t iterationIndex, uint32_t logicalUnitIndex,
                                  uint32_t unitStart, uint32_t unitCount, uint32_t workCount, uint32_t parentEventType,
                                  uint32_t beginCycle, uint32_t endCycle)
    {
        if (!IsEnabled()) {
            return;
        }
        if (*eventCount >= eventCapacity) {
            *truncated = 1U;
            return;
        }
        __gm__ uint32_t *dst = record + recordHeaderWords + static_cast<uint64_t>(*eventCount) * eventWords;
        AscendC::WriteGmByPassDCache(dst + 0U, eventType);
        AscendC::WriteGmByPassDCache(dst + 1U, iterationIndex);
        AscendC::WriteGmByPassDCache(dst + 2U, logicalUnitIndex);
        AscendC::WriteGmByPassDCache(dst + 3U, unitStart);
        AscendC::WriteGmByPassDCache(dst + 4U, unitCount);
        AscendC::WriteGmByPassDCache(dst + 5U, workCount);
        AscendC::WriteGmByPassDCache(dst + 6U, parentEventType);
        AscendC::WriteGmByPassDCache(dst + 7U, beginCycle);
        AscendC::WriteGmByPassDCache(dst + 8U, endCycle);
        ++(*eventCount);
    }
};

// 将一次调用中的真实 tile/chunk 坐标映射回 wave 内专家行坐标。
struct CriticalPathTileTraceContext {
    CriticalPathTraceEventWriter writer{};
    uint32_t stageEventType = 0U;
    uint32_t waitEventType = 0U;
    uint32_t iterationIndex = 0U;
    uint32_t logicalUnitIndex = 0U;
    uint32_t unitBase = 0U;
};

__aicore__ inline uint32_t BeginTraceInterval(CriticalPathTileTraceContext *context)
{
    return context == nullptr || !context->writer.IsEnabled() ? 0U : static_cast<uint32_t>(AscendC::GetSystemCycle());
}

__aicore__ inline void EndTraceInterval(CriticalPathTileTraceContext *context, bool isWait, uint32_t localUnitStart,
                                        uint32_t unitCount, uint32_t workCount, uint32_t beginCycle)
{
    if (context == nullptr || !context->writer.IsEnabled()) {
        return;
    }
    uint32_t eventType = isWait ? context->waitEventType : context->stageEventType;
    if (eventType == 0U) {
        return;
    }
    context->writer.Record(
        eventType, context->iterationIndex, context->logicalUnitIndex, context->unitBase + localUnitStart, unitCount,
        workCount, isWait ? context->stageEventType : 0U, beginCycle, static_cast<uint32_t>(AscendC::GetSystemCycle()));
}

#else

// 保留统一的内联接口，但正常构建中所有 trace 逻辑都在编译期化为空操作。
struct CriticalPathTraceEventWriter {
    __aicore__ inline bool IsEnabled() const
    {
        return false;
    }
};

struct CriticalPathTileTraceContext {
    CriticalPathTraceEventWriter writer{};
};

__aicore__ inline uint32_t BeginTraceInterval(CriticalPathTileTraceContext *)
{
    return 0U;
}

__aicore__ inline void EndTraceInterval(CriticalPathTileTraceContext *, bool, uint32_t, uint32_t, uint32_t, uint32_t) {}

#endif

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TRACE_UTILS_H
