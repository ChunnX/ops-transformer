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
constexpr uint32_t PROFILER_MAX_EVENT_NUM = 96U;
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
    EVT_SEND_PROFILE_MODE = 0x0170U,
    EVT_SEND_TASK_COUNT = 0x0171U,
    EVT_SEND_POSITION_CYCLES = 0x0172U,
    EVT_SEND_INPUT_CYCLES = 0x0173U,
    EVT_SEND_QUANT_CYCLES = 0x0174U,
    EVT_SEND_BUFFER_WAIT_CYCLES = 0x0175U,
    EVT_CAL_CUMSUM_BEGIN = 0x0200U,
    EVT_CAL_CUMSUM_END = 0x0201U,
    EVT_INIT_COUNT_STATUS_BEGIN = 0x0210U,
    EVT_INIT_COUNT_STATUS_END = 0x0211U,
    EVT_COMPUTE_EXPERT_COUNTS_BEGIN = 0x0220U,
    EVT_COMPUTE_EXPERT_COUNTS_END = 0x0221U,
    EVT_TRANSFER_COUNTS_BEGIN = 0x0230U,
    EVT_TRANSFER_COUNTS_END = 0x0231U,
    EVT_POLL_RECV_COUNTS_BEGIN = 0x0240U,
    EVT_POLL_RECV_COUNTS_END = 0x0241U,
    EVT_REDUCE_LOCAL_COUNTS_BEGIN = 0x0250U,
    EVT_REDUCE_LOCAL_COUNTS_END = 0x0251U,
    EVT_PUBLISH_LOCAL_SUM_BEGIN = 0x0260U,
    EVT_PUBLISH_LOCAL_SUM_END = 0x0261U,
    EVT_POLL_CORE_SUMS_BEGIN = 0x0270U,
    EVT_POLL_CORE_SUMS_END = 0x0271U,
    EVT_REDUCE_CORE_PREFIX_BEGIN = 0x0280U,
    EVT_REDUCE_CORE_PREFIX_END = 0x0281U,
    EVT_CLEAR_CORE_SUMS_BEGIN = 0x0290U,
    EVT_CLEAR_CORE_SUMS_END = 0x0291U,
    EVT_WRITE_COUNTS_OUTPUT_BEGIN = 0x02A0U,
    EVT_WRITE_COUNTS_OUTPUT_END = 0x02A1U,
    EVT_BROADCAST_PREFIX_BEGIN = 0x02B0U,
    EVT_BROADCAST_PREFIX_END = 0x02B1U,
    EVT_PUBLISH_PREFIX_FLAGS_BEGIN = 0x02C0U,
    EVT_PUBLISH_PREFIX_FLAGS_END = 0x02C1U,
    EVT_READ_PREFIX_BEGIN = 0x02D0U,
    EVT_READ_PREFIX_END = 0x02D1U,
    EVT_DRAIN_SEND_BUFFERS_BEGIN = 0x02E0U,
    EVT_DRAIN_SEND_BUFFERS_END = 0x02E1U,
    EVT_INIT_COUNT_BUFFERS_BEGIN = 0x02F0U,
    EVT_INIT_COUNT_BUFFERS_END = 0x02F1U,
    EVT_SEND_METADATA_CYCLES = 0x0300U,
    EVT_SEND_PACK_CYCLES = 0x0301U,
    EVT_SEND_MTE3_ISSUE_CYCLES = 0x0302U,
    EVT_COUNT_COMPUTE_CYCLES = 0x0303U,
    EVT_COUNT_ISSUE_CYCLES = 0x0304U,
    EVT_SEND_INPUT_COUNT = 0x0305U,
    EVT_SEND_QUANT_COUNT = 0x0306U,
    EVT_KERNEL_END = 0x00FFU,
};

enum ProfileMetric : uint32_t {
    METRIC_SEND_POSITION,
    METRIC_SEND_INPUT,
    METRIC_SEND_QUANT,
    METRIC_SEND_BUFFER_WAIT,
    METRIC_SEND_METADATA,
    METRIC_SEND_PACK,
    METRIC_SEND_MTE3_ISSUE,
    METRIC_COUNT_COMPUTE,
    METRIC_COUNT_ISSUE,
    METRIC_COUNT
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
        for (uint32_t i = 0; i < METRIC_COUNT; ++i) {
            metricCycles_[i] = 0;
            metricCalls_[i] = 0;
        }
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

    __aicore__ inline void Accumulate(uint32_t metric, uint64_t begin)
    {
        metricCycles_[metric] += AscendC::GetSystemCycle() - begin;
        ++metricCalls_[metric];
    }

    __aicore__ inline void EmitMetrics()
    {
        if (metricCalls_[METRIC_SEND_POSITION] != 0U) {
            Mark(EVT_SEND_POSITION_CYCLES, Narrow(metricCycles_[METRIC_SEND_POSITION]));
        }
        if (metricCalls_[METRIC_SEND_INPUT] != 0U) {
            Mark(EVT_SEND_INPUT_CYCLES, Narrow(metricCycles_[METRIC_SEND_INPUT]));
        }
        if (metricCalls_[METRIC_SEND_QUANT] != 0U) {
            Mark(EVT_SEND_QUANT_CYCLES, Narrow(metricCycles_[METRIC_SEND_QUANT]));
        }
        if (metricCalls_[METRIC_SEND_BUFFER_WAIT] != 0U) {
            Mark(EVT_SEND_BUFFER_WAIT_CYCLES, Narrow(metricCycles_[METRIC_SEND_BUFFER_WAIT]));
        }
        if (metricCalls_[METRIC_SEND_METADATA] != 0U) {
            Mark(EVT_SEND_METADATA_CYCLES, Narrow(metricCycles_[METRIC_SEND_METADATA]));
        }
        if (metricCalls_[METRIC_SEND_PACK] != 0U) {
            Mark(EVT_SEND_PACK_CYCLES, Narrow(metricCycles_[METRIC_SEND_PACK]));
        }
        if (metricCalls_[METRIC_SEND_MTE3_ISSUE] != 0U) {
            Mark(EVT_SEND_MTE3_ISSUE_CYCLES, Narrow(metricCycles_[METRIC_SEND_MTE3_ISSUE]));
        }
        if (metricCalls_[METRIC_COUNT_COMPUTE] != 0U) {
            Mark(EVT_COUNT_COMPUTE_CYCLES, Narrow(metricCycles_[METRIC_COUNT_COMPUTE]));
        }
        if (metricCalls_[METRIC_COUNT_ISSUE] != 0U) {
            Mark(EVT_COUNT_ISSUE_CYCLES, Narrow(metricCycles_[METRIC_COUNT_ISSUE]));
        }
        if (metricCalls_[METRIC_SEND_INPUT] != 0U) {
            Mark(EVT_SEND_INPUT_COUNT, metricCalls_[METRIC_SEND_INPUT]);
        }
        if (metricCalls_[METRIC_SEND_QUANT] != 0U) {
            Mark(EVT_SEND_QUANT_COUNT, metricCalls_[METRIC_SEND_QUANT]);
        }
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
        __gm__ uint32_t *count = reinterpret_cast<__gm__ uint32_t *>(buffer_ + PROFILER_CORE_COUNT_OFFSET +
                                                                     coreId * PROFILER_CORE_COUNT_ALIGN);
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
    // Saturation is explicit in the parser; accumulated metrics never silently wrap.
    __aicore__ inline uint32_t Narrow(uint64_t value)
    {
        return value > 0xFFFFFFFFULL ? 0xFFFFFFFFU : static_cast<uint32_t>(value);
    }
    uint64_t metricCycles_[METRIC_COUNT]{};
    uint32_t metricCalls_[METRIC_COUNT]{};
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
    __aicore__ inline void Init(BufferType, uint32_t)
    {}
    __aicore__ inline void Mark(uint32_t, uint32_t = 0U) {}
    __aicore__ inline void Flush(uint32_t) {}
    __aicore__ inline void EmitMetrics() {}
#else
    template <typename BufferType>
    inline void Init(BufferType, uint32_t)
    {}
    inline void Init(uint8_t *, uint32_t) {}
    inline void Mark(uint32_t, uint32_t = 0U) {}
    inline void Flush(uint32_t) {}
    inline void EmitMetrics() {}
#endif
};
#endif
} // namespace MoeDistributeDispatchV2FullMeshProfiler

// No extra pipeline barrier is inserted inside measured regions. These are issue-side
// intervals including existing waits, not isolated hardware-engine execution times.
#if MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILING && defined(__NPU_ARCH__)
#define FULLMESH_PROFILE_BEGIN(name) const uint64_t name = AscendC::GetSystemCycle()
#define FULLMESH_PROFILE_END(metric, name) profiler_.Accumulate(metric, name)
#else
#define FULLMESH_PROFILE_BEGIN(name) ((void)0)
#define FULLMESH_PROFILE_END(metric, name) ((void)0)
#endif

#endif // MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILER_H
