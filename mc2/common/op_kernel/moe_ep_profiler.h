/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_PROFILER_H
#define MOE_EP_PROFILER_H

#include <cstdint>

#ifdef ENABLE_MOE_EP_PROFILING

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "mc2_moe_context.h"

namespace Mc2Profile {

constexpr uint32_t MOE_EP_PROFILE_MAGIC = 0x4D455046U;
constexpr uint32_t MOE_EP_PROFILE_VERSION = 0x00010000U;
constexpr uint32_t MOE_EP_PROFILE_HEADER_BYTES = 64U;
constexpr uint32_t MOE_EP_PROFILE_CORE_CONTROL_BYTES = 64U;
constexpr uint32_t MOE_EP_PROFILE_RECORD_BYTES = 16U;
constexpr uint32_t MOE_EP_PROFILE_LOCAL_RECORDS = 32U;
constexpr uint32_t MOE_EP_PROFILE_AIV_ONLY = 1U;

enum class MoeEpProfileOp : uint32_t {
    DISPATCH = 1U,
    DISPATCH_EPILOGUE = 2U,
    COMBINE = 3U,
    COMBINE_EPILOGUE = 4U,
};

enum class MoeEpProfileEngine : uint32_t {
    SCALAR = 1U,
    MTE2 = 2U,
    VECTOR = 3U,
    MTE3 = 4U,
    URMA = 5U,
    WAIT = 6U,
};

enum class MoeEpProfileBoundary : uint32_t {
    BEGIN = 0U,
    END = 1U,
};

// [31:28] operator, [27:24] physical role (1=AIV), [23:20]
// logical engine, [19:1] stage, [0] boundary (0=begin, 1=end).
__aicore__ inline constexpr uint32_t MakeMoeEpProfileEvent(MoeEpProfileOp op, MoeEpProfileEngine engine, uint32_t stage,
                                                           MoeEpProfileBoundary boundary)
{
    return ((static_cast<uint32_t>(op) & 0xFU) << 28U) | (1U << 24U) | ((static_cast<uint32_t>(engine) & 0xFU) << 20U) |
           ((stage & 0x7FFFFU) << 1U) | (static_cast<uint32_t>(boundary) & 1U);
}

struct MoeEpProfileRecord {
    uint32_t eventId;
    uint32_t payload;
    uint64_t cycle;
};

static_assert(sizeof(MoeEpProfileRecord) == MOE_EP_PROFILE_RECORD_BYTES, "Unexpected profiler record layout");

class MoeEpProfiler {
public:
    __aicore__ inline void Init(__gm__ Mc2Aclnn::MoeCommContext *context, MoeEpProfileOp op, uint32_t actualAivCores)
    {
        profBuf_ = nullptr;
        maxCores_ = 0U;
        slotsPerCore_ = 0U;
        op_ = static_cast<uint32_t>(op);
        actualAivCores_ = actualAivCores;
        count_ = 0U;
        localDropped_ = 0U;
        if (context == nullptr) {
            return;
        }
        maxCores_ = context->moeEpProfilerMaxCores;
        slotsPerCore_ = context->moeEpProfilerSlotsPerCore;
        uint32_t opIndex = op_ - 1U;
        if (opIndex >= 4U || maxCores_ == 0U || slotsPerCore_ == 0U) {
            return;
        }
        profBuf_ = reinterpret_cast<__gm__ uint8_t *>(context->moeEpProfilerBuffers[opIndex]);
    }

    // Mark is deliberately local-only: one cycle read plus local record stores.
    // It performs no GM access, DCCI, or synchronization.
    __aicore__ inline void Mark(uint32_t eventId, uint32_t payload = 0U)
    {
        if (profBuf_ == nullptr) {
            return;
        }
        if (count_ >= MOE_EP_PROFILE_LOCAL_RECORDS) {
            ++localDropped_;
            return;
        }
        uint64_t cycle = AscendC::GetSystemCycle();
        localRecords_[count_].eventId = eventId;
        localRecords_[count_].payload = payload;
        localRecords_[count_].cycle = cycle;
        ++count_;
    }

    // Flush must be called only after recording KERNEL_END. All profiling GM
    // traffic and DCCI operations are intentionally deferred to this method.
    // No barrier is introduced here.
    __aicore__ inline void Flush()
    {
        if (profBuf_ == nullptr || count_ == 0U) {
            return;
        }
        uint32_t coreId = AscendC::GetBlockIdx();
        if (coreId >= maxCores_) {
            return;
        }

        if (coreId == 0U) {
            WriteGlobalHeader();
        }

        uint64_t coreStride = static_cast<uint64_t>(MOE_EP_PROFILE_CORE_CONTROL_BYTES) +
                              static_cast<uint64_t>(slotsPerCore_) * MOE_EP_PROFILE_RECORD_BYTES;
        __gm__ uint8_t *coreBase = profBuf_ + MOE_EP_PROFILE_HEADER_BYTES + coreId * coreStride;
        __gm__ uint32_t *control = reinterpret_cast<__gm__ uint32_t *>(coreBase);
        FlushCacheLine(control, MOE_EP_PROFILE_CORE_CONTROL_BYTES / sizeof(uint32_t));

        uint64_t writeCount = static_cast<uint64_t>(control[0]) | (static_cast<uint64_t>(control[1]) << 32U);
        uint64_t droppedCount = static_cast<uint64_t>(control[4]) | (static_cast<uint64_t>(control[5]) << 32U);
        uint64_t flushCount = static_cast<uint64_t>(control[6]) | (static_cast<uint64_t>(control[7]) << 32U);

        __gm__ uint8_t *recordsBase = coreBase + MOE_EP_PROFILE_CORE_CONTROL_BYTES;
        for (uint32_t i = 0U; i < count_; ++i) {
            uint32_t slot = static_cast<uint32_t>((writeCount + i) % slotsPerCore_);
            __gm__ uint32_t *dst = reinterpret_cast<__gm__ uint32_t *>(recordsBase + static_cast<uint64_t>(slot) *
                                                                                         MOE_EP_PROFILE_RECORD_BYTES);
            dst[0] = localRecords_[i].eventId;
            dst[1] = localRecords_[i].payload;
            dst[2] = static_cast<uint32_t>(localRecords_[i].cycle);
            dst[3] = static_cast<uint32_t>(localRecords_[i].cycle >> 32U);
            FlushCacheLine(dst, MOE_EP_PROFILE_RECORD_BYTES / sizeof(uint32_t));
        }

        uint64_t nextWriteCount = writeCount + count_;
        uint64_t nextDroppedCount = droppedCount + localDropped_;
        uint32_t flags = control[3];
        if (nextWriteCount > slotsPerCore_) {
            flags |= 1U;
        }
        if (localDropped_ != 0U) {
            flags |= 2U;
        }
        StoreU64(control, 0U, nextWriteCount);
        control[2] = static_cast<uint32_t>(nextWriteCount % slotsPerCore_);
        control[3] = flags;
        StoreU64(control, 4U, nextDroppedCount);
        StoreU64(control, 6U, flushCount + 1U);
        control[8] = count_;
        FlushCacheLine(control, MOE_EP_PROFILE_CORE_CONTROL_BYTES / sizeof(uint32_t));
    }

private:
    __aicore__ static inline void StoreU64(__gm__ uint32_t *dst, uint32_t wordOffset, uint64_t value)
    {
        dst[wordOffset] = static_cast<uint32_t>(value);
        dst[wordOffset + 1U] = static_cast<uint32_t>(value >> 32U);
    }

    __aicore__ static inline void FlushCacheLine(__gm__ uint32_t *address, uint32_t elements)
    {
        AscendC::GlobalTensor<uint32_t> tensor;
        tensor.SetGlobalBuffer(address, elements);
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(tensor);
    }

    __aicore__ inline void WriteGlobalHeader()
    {
        __gm__ uint32_t *header = reinterpret_cast<__gm__ uint32_t *>(profBuf_);
        uint32_t coreStride = MOE_EP_PROFILE_CORE_CONTROL_BYTES + slotsPerCore_ * MOE_EP_PROFILE_RECORD_BYTES;
        header[0] = MOE_EP_PROFILE_MAGIC;
        header[1] = MOE_EP_PROFILE_VERSION;
        header[2] = MOE_EP_PROFILE_HEADER_BYTES;
        header[3] = op_;
        header[4] = actualAivCores_;
        header[5] = maxCores_;
        header[6] = slotsPerCore_;
        header[7] = MOE_EP_PROFILE_RECORD_BYTES;
        header[8] = MOE_EP_PROFILE_CORE_CONTROL_BYTES;
        header[9] = coreStride;
        header[10] = MOE_EP_PROFILE_AIV_ONLY;
        header[11] = MOE_EP_PROFILE_LOCAL_RECORDS;
        FlushCacheLine(header, MOE_EP_PROFILE_HEADER_BYTES / sizeof(uint32_t));
    }

    MoeEpProfileRecord localRecords_[MOE_EP_PROFILE_LOCAL_RECORDS];
    __gm__ uint8_t *profBuf_{nullptr};
    uint32_t count_{0U};
    uint32_t localDropped_{0U};
    uint32_t op_{0U};
    uint32_t actualAivCores_{0U};
    uint32_t maxCores_{0U};
    uint32_t slotsPerCore_{0U};
};

} // namespace Mc2Profile

using Mc2Profile::MoeEpProfileBoundary;
using Mc2Profile::MoeEpProfileEngine;
using Mc2Profile::MoeEpProfileOp;

#define MOE_EP_PROFILE_EVENT(op, engine, stage, boundary) \
    Mc2Profile::MakeMoeEpProfileEvent((op), (engine), static_cast<uint32_t>(stage), (boundary))
#define MOE_EP_PROFILE_DECLARE() Mc2Profile::MoeEpProfiler moeEpProfiler_
#define MOE_EP_PROFILE_INIT(context, op, actualAivCores) moeEpProfiler_.Init((context), (op), (actualAivCores))
#define MOE_EP_PROFILE_MARK(eventId) moeEpProfiler_.Mark((eventId))
#define MOE_EP_PROFILE_MARK_PAYLOAD(eventId, payload) moeEpProfiler_.Mark((eventId), (payload))
#define MOE_EP_PROFILE_FLUSH() moeEpProfiler_.Flush()

#else

#define MOE_EP_PROFILE_EVENT(op, engine, stage, boundary) (0U)
#define MOE_EP_PROFILE_DECLARE()
#define MOE_EP_PROFILE_INIT(context, op, actualAivCores) ((void)0)
#define MOE_EP_PROFILE_MARK(eventId) ((void)0)
#define MOE_EP_PROFILE_MARK_PAYLOAD(eventId, payload) ((void)0)
#define MOE_EP_PROFILE_FLUSH() ((void)0)

#endif // ENABLE_MOE_EP_PROFILING

#endif // MOE_EP_PROFILER_H
