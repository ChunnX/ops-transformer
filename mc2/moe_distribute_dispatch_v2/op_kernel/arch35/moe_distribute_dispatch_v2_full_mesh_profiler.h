/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_full_mesh_profiler.h
 * \brief Device-side cycle profiler for the A5 FullMeshV2 path.
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILER_H
#define MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILER_H

#include <cstdint>

// Debug switch shared by host tiling and device code. Set to 0 before performance acceptance.
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILING
#define MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILING 1
#endif

namespace MoeDistributeDispatchV2FullMeshProfiler {
constexpr uint32_t PROFILER_MAX_CORE_NUM = 72U;
constexpr uint32_t PROFILER_MAX_EVENT_NUM = 32U;
constexpr uint32_t PROFILER_RECORD_SIZE = 16U;
constexpr uint32_t PROFILER_HEADER_SIZE = 64U;
constexpr uint32_t PROFILER_CORE_COUNT_ALIGN = 64U;
constexpr uint32_t PROFILER_CORE_COUNT_OFFSET = PROFILER_HEADER_SIZE;
constexpr uint32_t PROFILER_DATA_OFFSET =
    PROFILER_CORE_COUNT_OFFSET + PROFILER_MAX_CORE_NUM * PROFILER_CORE_COUNT_ALIGN;
constexpr uint64_t PROFILER_BUFFER_SIZE =
    PROFILER_DATA_OFFSET + PROFILER_MAX_CORE_NUM * PROFILER_MAX_EVENT_NUM * PROFILER_RECORD_SIZE;
constexpr uint32_t PROFILER_MAGIC = 0x50524F46U;
constexpr uint32_t PROFILER_FORMAT_VERSION = 4U;
constexpr uint32_t PROFILER_LAYOUT_AIV_ONLY = 1U;
constexpr uint32_t PROFILER_OVERFLOW_EVENT = 0xFFFFU;

enum FullMeshV2ProfilerEvent : uint32_t {
    EVT_KERNEL_BEGIN = 0x0001U,
    EVT_SEND_TOKENS_BEGIN = 0x0010U,
    EVT_SEND_TOKENS_END = 0x0011U,
    EVT_SEND_COUNTS_BEGIN = 0x0020U,
    EVT_SEND_COUNTS_END = 0x0021U,
    EVT_WAIT_RECV_COUNTS_BEGIN = 0x0030U,
    EVT_WAIT_RECV_COUNTS_END = 0x0031U,
    EVT_LOCAL_WINDOW_COPY_BEGIN = 0x0040U,
    EVT_LOCAL_WINDOW_COPY_END = 0x0041U,
    EVT_WAIT_CUMSUM_BEGIN = 0x0050U,
    EVT_WAIT_CUMSUM_END = 0x0051U,
    EVT_FORMAT_OUTPUT_BEGIN = 0x0060U,
    EVT_FORMAT_OUTPUT_END = 0x0061U,
    EVT_PREPARE_TOKEN_INPUT_BEGIN = 0x0070U,
    EVT_PREPARE_TOKEN_INPUT_END = 0x0071U,
    EVT_TRANSFER_TOKENS_BEGIN = 0x0080U,
    EVT_TRANSFER_TOKENS_END = 0x0081U,
    EVT_PREPARE_COUNT_INPUT_BEGIN = 0x0090U,
    EVT_PREPARE_COUNT_INPUT_END = 0x0091U,
    EVT_CLEAR_RECV_STATUS_BEGIN = 0x00A0U,
    EVT_CLEAR_RECV_STATUS_END = 0x00A1U,
    EVT_GATHER_RECV_COUNTS_BEGIN = 0x00B0U,
    EVT_GATHER_RECV_COUNTS_END = 0x00B1U,
    EVT_GET_CUMSUM_BEGIN = 0x00C0U,
    EVT_GET_CUMSUM_END = 0x00C1U,
    EVT_BUILD_RECV_CUMSUM_BEGIN = 0x00D0U,
    EVT_BUILD_RECV_CUMSUM_END = 0x00D1U,
    EVT_PUBLISH_RECV_COUNTS_BEGIN = 0x00E0U,
    EVT_PUBLISH_RECV_COUNTS_END = 0x00E1U,
    EVT_RESET_CUMSUM_FLAG_BEGIN = 0x0100U,
    EVT_RESET_CUMSUM_FLAG_END = 0x0101U,
    EVT_BUILD_EXPERT_TASKS_BEGIN = 0x0110U,
    EVT_BUILD_EXPERT_TASKS_END = 0x0111U,
    EVT_CLEAR_LOCAL_FLAGS_BEGIN = 0x0120U,
    EVT_CLEAR_LOCAL_FLAGS_END = 0x0121U,
    EVT_DRAIN_PIPELINES_BEGIN = 0x0130U,
    EVT_DRAIN_PIPELINES_END = 0x0131U,
    EVT_NO_ACTIVE_TOKENS_BEGIN = 0x0140U,
    EVT_NO_ACTIVE_TOKENS_END = 0x0141U,
    EVT_NO_ASSIGNED_STATUS_BEGIN = 0x0150U,
    EVT_NO_ASSIGNED_STATUS_END = 0x0151U,
    EVT_NO_VALID_EXPERT_BEGIN = 0x0160U,
    EVT_NO_VALID_EXPERT_END = 0x0161U,
    EVT_SEND_TASK_COUNT = 0x0171U,
    EVT_SEND_BUFFER_NUM = 0x0172U,
    EVT_SEND_POSITION_CYCLES = 0x0173U,
    EVT_SEND_INPUT_CYCLES = 0x0174U,
    EVT_SEND_COMPUTE_CYCLES = 0x0175U,
    EVT_SEND_PACK_CYCLES = 0x0176U,
    EVT_SEND_BUFFER_WAIT_CYCLES = 0x0177U,
    EVT_SERIAL_BUFFER_INIT_CYCLES = 0x0178U,
    EVT_READ_RECV_COUNTS_BEGIN = 0x0180U,
    EVT_READ_RECV_COUNTS_END = 0x0181U,
    EVT_BUILD_LOCAL_PREFIX_BEGIN = 0x0190U,
    EVT_BUILD_LOCAL_PREFIX_END = 0x0191U,
    EVT_PUBLISH_LOCAL_COUNTS_BEGIN = 0x01A0U,
    EVT_PUBLISH_LOCAL_COUNTS_END = 0x01A1U,
    EVT_KERNEL_END = 0x00FFU,
};

#if MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILING && defined(__NPU_ARCH__)
class ProfilerState {
public:
    __aicore__ inline void Init(GM_ADDR buffer, uint32_t coreNum)
    {
        buffer_ = reinterpret_cast<__gm__ uint8_t *>(buffer);
        coreNum_ = coreNum < PROFILER_MAX_CORE_NUM ? coreNum : PROFILER_MAX_CORE_NUM;
        eventCount_ = 0U;
        overflow_ = false;
    }

    __aicore__ inline void Mark(uint32_t eventId, uint32_t payload = 0U)
    {
        if (buffer_ == nullptr) {
            return;
        }
        if (eventCount_ >= PROFILER_MAX_EVENT_NUM) {
            overflow_ = true;
            return;
        }
        eventIds_[eventCount_] = eventId;
        payloads_[eventCount_] = payload;
        cycles_[eventCount_] = AscendC::GetSystemCycle();
        ++eventCount_;
    }

    __aicore__ inline void Flush(uint32_t coreId)
    {
        if ((buffer_ == nullptr) || (coreId >= PROFILER_MAX_CORE_NUM)) {
            return;
        }
        uint32_t recordCount = eventCount_;
        if (overflow_) {
            recordCount = PROFILER_MAX_EVENT_NUM;
            eventIds_[PROFILER_MAX_EVENT_NUM - 1U] = PROFILER_OVERFLOW_EVENT;
            payloads_[PROFILER_MAX_EVENT_NUM - 1U] = 0U;
            cycles_[PROFILER_MAX_EVENT_NUM - 1U] = AscendC::GetSystemCycle();
        }
        __gm__ uint32_t *recordBase = reinterpret_cast<__gm__ uint32_t *>(
            buffer_ + PROFILER_DATA_OFFSET + coreId * PROFILER_MAX_EVENT_NUM * PROFILER_RECORD_SIZE);
        for (uint32_t i = 0U; i < recordCount; ++i) {
            __gm__ uint32_t *record = recordBase + i * (PROFILER_RECORD_SIZE / sizeof(uint32_t));
            record[0] = eventIds_[i];
            record[1] = payloads_[i];
            record[2] = static_cast<uint32_t>(cycles_[i]);
            record[3] = static_cast<uint32_t>(cycles_[i] >> 32U);
            FlushCacheLine(record, PROFILER_RECORD_SIZE / sizeof(uint32_t));
        }
        __gm__ uint32_t *count = reinterpret_cast<__gm__ uint32_t *>(
            buffer_ + PROFILER_CORE_COUNT_OFFSET + coreId * PROFILER_CORE_COUNT_ALIGN);
        *count = recordCount;
        FlushCacheLine(count, 1U);
        if (coreId == 0U) {
            __gm__ uint32_t *header = reinterpret_cast<__gm__ uint32_t *>(buffer_);
            header[0] = PROFILER_MAGIC;
            header[1] = (PROFILER_LAYOUT_AIV_ONLY << 16U) | PROFILER_FORMAT_VERSION;
            header[2] = coreNum_;
            header[3] = PROFILER_RECORD_SIZE;
            header[4] = PROFILER_MAX_EVENT_NUM;
            header[5] = PROFILER_CORE_COUNT_ALIGN;
            header[6] = PROFILER_DATA_OFFSET;
            header[7] = 0U;
            FlushCacheLine(header, PROFILER_HEADER_SIZE / sizeof(uint32_t));
        }
    }

private:
    __aicore__ inline void FlushCacheLine(__gm__ uint32_t *address, uint32_t size)
    {
        AscendC::GlobalTensor<uint32_t> tensor;
        tensor.SetGlobalBuffer(address, size);
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(tensor);
    }

    __gm__ uint8_t *buffer_{nullptr};
    uint64_t cycles_[PROFILER_MAX_EVENT_NUM];
    uint32_t eventIds_[PROFILER_MAX_EVENT_NUM];
    uint32_t payloads_[PROFILER_MAX_EVENT_NUM];
    uint32_t coreNum_{0U};
    uint32_t eventCount_{0U};
    bool overflow_{false};
};
#else
class ProfilerState {
public:
#if defined(__NPU_ARCH__)
    template <typename BufferType>
    __aicore__ inline void Init(BufferType, uint32_t) {}
    __aicore__ inline void Mark(uint32_t, uint32_t = 0U) {}
    __aicore__ inline void Flush(uint32_t) {}
#else
    template <typename BufferType>
    inline void Init(BufferType, uint32_t) {}
    inline void Init(uint8_t *, uint32_t) {}
    inline void Mark(uint32_t, uint32_t = 0U) {}
    inline void Flush(uint32_t) {}
#endif
};
#endif
} // namespace MoeDistributeDispatchV2FullMeshProfiler

#endif // MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILER_H
