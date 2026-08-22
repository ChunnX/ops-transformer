/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_RUNTIME_INFO_H
#define MC2_RUNTIME_INFO_H

#include "mc2_tiling_struct.h"

#if defined(__CCE_AICORE__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Mc2Kernel {

struct DefaultRtPolicy {
    using StageEnumT = Mc2Tiling::RuntimePhase;
};

template <typename Policy = DefaultRtPolicy>
class Mc2RuntimeInfo {
public:
    __aicore__ inline Mc2RuntimeInfo() = default;

    __aicore__ inline void Init(GM_ADDR workspaceGM,
                                Mc2Tiling::WorkspaceLayoutInfo *wsLayout,
                                uint32_t aicCoreNum)
    {
        runtimeInfoBase_ = nullptr;
        slotIdx_ = 0;

        if (wsLayout == nullptr) {
            return;
        }

        for (uint32_t i = 0; i < wsLayout->segCount && i < Mc2Tiling::MAX_WORKSPACE_SEGMENTS; i++) {
            if (wsLayout->segments[i].type == Mc2Tiling::WS_SEG_RUNTIME_INFO) {
                runtimeInfoBase_ = reinterpret_cast<__gm__ Mc2Tiling::RuntimeInfoPerCore *>(
                    workspaceGM + wsLayout->segments[i].offset);
                break;
            }
        }

        if (runtimeInfoBase_ == nullptr) {
            return;
        }

        if ASCEND_IS_AIC {
            slotIdx_ = AscendC::GetBlockIdx();
        } else {
            slotIdx_ = aicCoreNum + AscendC::GetBlockIdx();
        }

        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->magicNum = 0x5A5A5A5AU;
        mySlot->coreId = static_cast<uint16_t>(slotIdx_);
        mySlot->execTurn = 0;
        mySlot->execPosition = 0;
        mySlot->commPhase = static_cast<uint8_t>(Mc2Tiling::RT_PHASE_INIT);
        mySlot->reserved1 = 0;
        mySlot->traceHead = 0;
        mySlot->traceCount = 0;
        mySlot->reserved2[0] = 0;
        mySlot->reserved2[1] = 0;

        FlushCacheLine(reinterpret_cast<GM_ADDR>(mySlot), Mc2Tiling::RUNTIME_INFO_PER_CORE_SIZE);
    }

    __aicore__ inline void Record(uint8_t turn, uint8_t position, uint8_t phase)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->execTurn = turn;
        mySlot->execPosition = position;
        mySlot->commPhase = phase;
        FlushCacheLine(reinterpret_cast<GM_ADDR>(mySlot), Mc2Tiling::RUNTIME_INFO_PER_CORE_SIZE);
        AppendTrace(turn, position, phase);
    }

    __aicore__ inline void RecordPhase(uint8_t phase)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->commPhase = phase;
        FlushCacheLine(reinterpret_cast<GM_ADDR>(mySlot), Mc2Tiling::RUNTIME_INFO_PER_CORE_SIZE);
        AppendTrace(mySlot->execTurn, mySlot->execPosition, phase);
    }

    __aicore__ inline void RecordPosition(uint8_t turn, uint8_t position)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->execTurn = turn;
        mySlot->execPosition = position;
        FlushCacheLine(reinterpret_cast<GM_ADDR>(mySlot), Mc2Tiling::RUNTIME_INFO_PER_CORE_SIZE);
        AppendTrace(turn, position, mySlot->commPhase);
    }

    __aicore__ inline void WriteGmByPassDCache(uint8_t turn, uint8_t position, uint8_t phase)
    {
        Record(turn, position, phase);
    }

    __aicore__ inline void UpdatePhase(uint8_t phase)
    {
        RecordPhase(phase);
    }

    __aicore__ inline void UpdatePhase(uint8_t turn, uint8_t position, uint8_t phase)
    {
        Record(turn, position, phase);
    }

private:
    __aicore__ inline void AppendTrace(uint8_t turn, uint8_t position, uint8_t phase)
    {
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        uint16_t idx = mySlot->traceHead % Mc2Tiling::RT_TRACE_CAPACITY;
        __gm__ Mc2Tiling::RtTraceEntry *entry = &mySlot->trace[idx];
        entry->turn = turn;
        entry->pos = position;
        entry->phase = phase;
        entry->reserved = 0;
        entry->cycle = static_cast<uint32_t>(AscendC::GetSystemCycle());
        mySlot->traceHead++;
        if (mySlot->traceCount < Mc2Tiling::RT_TRACE_CAPACITY) {
            mySlot->traceCount++;
        }
        FlushCacheLine(reinterpret_cast<GM_ADDR>(entry), sizeof(Mc2Tiling::RtTraceEntry));
    }

    __aicore__ inline void FlushCacheLine(GM_ADDR addr, size_t size)
    {
        AscendC::GlobalTensor<uint8_t> cacheLine;
        cacheLine.SetGlobalBuffer(addr, size);
        DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                 AscendC::DcciDst::CACHELINE_OUT>(cacheLine);
    }

    __gm__ Mc2Tiling::RuntimeInfoPerCore *runtimeInfoBase_ = nullptr;
    uint32_t slotIdx_ = 0;
};

} // namespace Mc2Kernel
#endif // __CCE_AICORE__
#endif // MC2_RUNTIME_INFO_H
