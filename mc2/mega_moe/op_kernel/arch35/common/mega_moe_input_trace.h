/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_INPUT_TRACE_H
#define MEGA_MOE_INPUT_TRACE_H

#include "kernel_operator.h"
#include "mega_moe_input_trace_config.h"

namespace MegaMoeImpl {

enum class InputTraceEvent : uint32_t {
    INPUT_PREPARE = 1,
    BUFFER_INIT_SUBMIT = 2,
    RESET_SUBMIT = 3,
    TOKEN_DATA_SUBMIT = 4,
    SEND_INDEX_SUBMIT = 5,
    SEND_COUNT_SUBMIT = 6,
    AIC_INPUT_SYNC_SUBMIT = 7,
    SHARED_STAGE_SUBMIT = 8,
    SHARED_COMPLETION_FENCE = 9,
    MOE_ENTRY_SYNC_SUBMIT = 10,
    MOE_ENTRY = 11,
    INPUT_RESET_SCALAR_WAIT = 12,
    COUNT_TABLE_WAIT = 13,
    COUNT_TABLE_PREPARE = 14,
    MOE_COUNT_READY_WAIT = 15,
};

// 每角色独占 GM record；完整算子输出完成后才复制到 y。关闭打点时内联消除。
class InputTrace {
public:
    static __aicore__ inline uint32_t Now()
    {
#if MEGA_MOE_INPUT_TRACE
        return static_cast<uint32_t>(AscendC::GetSystemCycle());
#else
        return 0U;
#endif
    }

#if MEGA_MOE_INPUT_TRACE
    __aicore__ inline void Init(GM_ADDR scratch, GM_ADDR output, uint64_t outputBytes, uint32_t cores, uint32_t core,
                                uint32_t role, uint32_t bs, uint32_t h, uint32_t worldSize, uint32_t sharedMode,
                                uint32_t moeMode)
    {
        cores_ = cores;
        core_ = core;
        role_ = role;
        bs_ = bs;
        h_ = h;
        worldSize_ = worldSize;
        sharedMode_ = sharedMode;
        moeMode_ = moeMode;
        const uint64_t fixedWords = INPUT_TRACE_HEADER_WORDS + static_cast<uint64_t>(cores) * INPUT_TRACE_ROLE_COUNT *
                                                                   INPUT_TRACE_RECORD_HEADER_WORDS;
        const uint64_t words = outputBytes / sizeof(uint32_t);
        if (cores == 0U || words <= fixedWords) {
            return;
        }
        const uint64_t capacity =
            (words - fixedWords) / (static_cast<uint64_t>(cores) * INPUT_TRACE_ROLE_COUNT * INPUT_TRACE_EVENT_WORDS);
        capacity_ =
            static_cast<uint32_t>(capacity < INPUT_TRACE_EVENT_CAPACITY ? capacity : INPUT_TRACE_EVENT_CAPACITY);
        if (capacity_ == 0U) {
            return;
        }
        const uint32_t index = role * cores + core;
        scratch_ =
            reinterpret_cast<__gm__ uint32_t *>(scratch) + static_cast<uint64_t>(index) * INPUT_TRACE_RECORD_WORDS;
        output_ = reinterpret_cast<__gm__ uint32_t *>(output);
    }
#endif

    __aicore__ inline void Begin()
    {
#if MEGA_MOE_INPUT_TRACE
        begin_ = Now();
#endif
    }

    __aicore__ inline void Record(InputTraceEvent event, uint32_t begin, uint32_t end, uint32_t rangeBegin = 0U,
                                  uint32_t rangeCount = 0U, uint32_t parent = 0U)
    {
#if MEGA_MOE_INPUT_TRACE
        if (scratch_ == nullptr) {
            return;
        }
        if (count_ == capacity_) {
            truncated_ = 1U;
            return;
        }
        __gm__ uint32_t *dst = scratch_ + INPUT_TRACE_RECORD_HEADER_WORDS + count_ * INPUT_TRACE_EVENT_WORDS;
        AscendC::WriteGmBypassDCache(dst + 0U, static_cast<uint32_t>(event));
        AscendC::WriteGmBypassDCache(dst + 1U, 0U);
        AscendC::WriteGmBypassDCache(dst + 2U, 0U);
        AscendC::WriteGmBypassDCache(dst + 3U, rangeBegin);
        AscendC::WriteGmBypassDCache(dst + 4U, rangeCount);
        AscendC::WriteGmBypassDCache(dst + 5U, rangeCount);
        AscendC::WriteGmBypassDCache(dst + 6U, parent);
        AscendC::WriteGmBypassDCache(dst + 7U, begin);
        AscendC::WriteGmBypassDCache(dst + 8U, end);
        ++count_;
#endif
    }

    __aicore__ inline void Finish()
    {
#if MEGA_MOE_INPUT_TRACE
        if (scratch_ == nullptr) {
            return;
        }
        // 完整算子本核流水完成后记录结束；发布屏障和复制时间不计入 Process。
        AscendC::PipeBarrier<PIPE_ALL>();
        const uint32_t end = Now();
        AscendC::SyncAll<false>();
        const uint32_t recordWords = INPUT_TRACE_RECORD_HEADER_WORDS + capacity_ * INPUT_TRACE_EVENT_WORDS;
        __gm__ uint32_t *dst =
            output_ + INPUT_TRACE_HEADER_WORDS + static_cast<uint64_t>(role_ * cores_ + core_) * recordWords;
        AscendC::WriteGmBypassDCache(dst + 0U, role_);
        AscendC::WriteGmBypassDCache(dst + 1U, core_);
        AscendC::WriteGmBypassDCache(dst + 2U, count_);
        AscendC::WriteGmBypassDCache(dst + 3U, truncated_);
        AscendC::WriteGmBypassDCache(dst + 4U, begin_);
        AscendC::WriteGmBypassDCache(dst + 5U, end);
        AscendC::WriteGmBypassDCache(dst + 6U, 0U);
        AscendC::WriteGmBypassDCache(dst + 7U, 0U);
        for (uint32_t word = 0U; word < count_ * INPUT_TRACE_EVENT_WORDS; ++word) {
            AscendC::WriteGmBypassDCache(
                dst + INPUT_TRACE_RECORD_HEADER_WORDS + word,
                AscendC::ReadGmBypassDCache(scratch_ + INPUT_TRACE_RECORD_HEADER_WORDS + word));
        }
        AscendC::SyncAll<false>();
        if (role_ == 0U && core_ == 0U) {
            AscendC::WriteGmBypassDCache(output_ + 1U, INPUT_TRACE_VERSION);
            AscendC::WriteGmBypassDCache(output_ + 2U, INPUT_TRACE_HEADER_WORDS);
            AscendC::WriteGmBypassDCache(output_ + 3U, recordWords);
            AscendC::WriteGmBypassDCache(output_ + 4U, INPUT_TRACE_EVENT_WORDS);
            AscendC::WriteGmBypassDCache(output_ + 5U, cores_ * INPUT_TRACE_ROLE_COUNT);
            AscendC::WriteGmBypassDCache(output_ + 6U, cores_);
            AscendC::WriteGmBypassDCache(output_ + 7U, capacity_);
            AscendC::WriteGmBypassDCache(output_ + 8U, bs_);
            AscendC::WriteGmBypassDCache(output_ + 9U, h_);
            AscendC::WriteGmBypassDCache(output_ + 10U, worldSize_);
            AscendC::WriteGmBypassDCache(output_ + 11U, static_cast<uint32_t>(MEGA_MOE_INPUT_TRACE));
            AscendC::WriteGmBypassDCache(output_ + 12U, sharedMode_);
            AscendC::WriteGmBypassDCache(output_ + 13U, moeMode_);
            AscendC::WriteGmBypassDCache(output_ + 14U, 0U);
            AscendC::WriteGmBypassDCache(output_ + 15U, 0U);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::WriteGmBypassDCache(output_, INPUT_TRACE_MAGIC);
        }
#endif
    }

private:
#if MEGA_MOE_INPUT_TRACE
    __gm__ uint32_t *scratch_ = nullptr;
    __gm__ uint32_t *output_ = nullptr;
    uint32_t cores_ = 0U, core_ = 0U, role_ = 0U, capacity_ = 0U;
    uint32_t count_ = 0U, truncated_ = 0U, begin_ = 0U;
    uint32_t bs_ = 0U, h_ = 0U, worldSize_ = 0U, sharedMode_ = 0U, moeMode_ = 0U;
#endif
};
} // namespace MegaMoeImpl
#endif
