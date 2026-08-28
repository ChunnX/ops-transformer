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

class Mc2RuntimeInfo {
public:
    __aicore__ inline Mc2RuntimeInfo() = default;

    __aicore__ inline void Init(GM_ADDR workspaceGM,
                                Mc2Tiling::WorkspaceLayoutInfo *wsLayout,
                                uint32_t aicCoreNum,
                                uint8_t initialPhase = Mc2Tiling::RT_PHASE_COMM_INIT)
    {
        runtimeInfoBase_ = nullptr;
        slotIdx_ = 0;

        if (workspaceGM == nullptr || wsLayout == nullptr ||
            wsLayout->segCount > Mc2Tiling::MAX_WORKSPACE_SEGMENTS) {
            return;
        }

        // 遍历 workspace 段表，定位 RUNTIME_INFO 段。
        // 注意: kernel 收到的 workspaceGM 已跳过 LIB_API 段(由框架管理)，
        // 但段表中的 offset 是从 0 累加的(含 LIB_API)，因此需减去 libApiSize
        // 得到用户段实际偏移，否则基地址越界导致 GM 写入崩溃。
        uint64_t libApiSize = 0;
        for (uint32_t i = 0; i < wsLayout->segCount && i < Mc2Tiling::MAX_WORKSPACE_SEGMENTS; i++) {
            if (wsLayout->segments[i].type == Mc2Tiling::WS_SEG_LIB_API) {
                if (wsLayout->segments[i].offset != 0 || wsLayout->segments[i].size > wsLayout->totalSize) {
                    return;
                }
                libApiSize = wsLayout->segments[i].size;
            }
        }
        for (uint32_t i = 0; i < wsLayout->segCount; i++) {
            if (wsLayout->segments[i].type == Mc2Tiling::WS_SEG_RUNTIME_INFO) {
                const auto &seg = wsLayout->segments[i];
                if (seg.offset < libApiSize || seg.offset > wsLayout->totalSize ||
                    seg.size > wsLayout->totalSize - seg.offset || seg.size < Mc2Tiling::RUNTIME_INFO_TOTAL_SIZE) {
                    return;
                }
                uint64_t userOffset = wsLayout->segments[i].offset - libApiSize;
                runtimeInfoBase_ = reinterpret_cast<__gm__ Mc2Tiling::RuntimeInfoPerCore *>(
                    workspaceGM + userOffset);
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
        if (slotIdx_ >= Mc2Tiling::RUNTIME_INFO_CORE_MAX) {
            runtimeInfoBase_ = nullptr;
            return;
        }

        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->magicNum = 0x5A5A5A5AU;
        mySlot->coreId = static_cast<uint16_t>(slotIdx_);
        mySlot->execTurn = 0;
        mySlot->execPosition = 0;
        mySlot->commPhase = initialPhase;
        mySlot->commCommitCount = 0;
        mySlot->commWaitCount = 0;

        FlushCache();
    }

    // 组合写入: 阶段入口用，三字段一次写入+单次FlushCache，
    // 避免分两次写(WritePosTag+WritePhase)崩溃导致 posTag/phase 状态不一致
    __aicore__ inline void WriteGmByPassDCache(uint8_t turn, uint8_t position, uint8_t phase)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->execTurn = turn;
        mySlot->execPosition = position;
        mySlot->commPhase = phase;
        FlushCache();
    }

    // Change the current tile/operation without overwriting the last communication state.
    __aicore__ inline void WritePositionByPassDCache(uint32_t turn, uint8_t position)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->execTurn = static_cast<uint8_t>(turn);
        mySlot->execPosition = position;
        FlushCache();
    }

    // 仅更新 commPhase: finalize 或仅切换通信阶段时用
    __aicore__ inline void WritePhaseByPassDCache(uint8_t phase)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->commPhase = phase;
        FlushCache();
    }

    // 仅更新 execTurn: tile 进度打点用，崩溃时 turn 值即当前卡死的 tile 序号
    __aicore__ inline void WriteTurnByPassDCache(uint8_t turn)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->execTurn = turn;
        FlushCache();
    }

    // 递增 HCCL Commit 次数: 在 hccl_.Commit() 返回后调用
    __aicore__ inline void IncCommitCount(uint8_t phase = Mc2Tiling::RT_PHASE_RESERVED)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->commCommitCount++;
        if (phase != Mc2Tiling::RT_PHASE_RESERVED) {
            mySlot->commPhase = phase;
        }
        FlushCache();
    }

    // 递增 HCCL Wait 次数: 在 hccl_.Wait() 返回后调用
    // commit > wait means outstanding communication, not necessarily a blocked Wait.
    __aicore__ inline void IncWaitCount(uint8_t phase = Mc2Tiling::RT_PHASE_RESERVED)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->commWaitCount++;
        if (phase != Mc2Tiling::RT_PHASE_RESERVED) {
            mySlot->commPhase = phase;
        }
        FlushCache();
    }

private:
    __aicore__ inline void FlushCache()
    {
        AscendC::GlobalTensor<uint8_t> cacheLine;
        cacheLine.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(&runtimeInfoBase_[slotIdx_]),
                                  Mc2Tiling::RUNTIME_INFO_PER_CORE_SIZE);
        DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(cacheLine);
    }

    __gm__ Mc2Tiling::RuntimeInfoPerCore *runtimeInfoBase_ = nullptr;
    uint32_t slotIdx_ = 0;
};

} // namespace Mc2Kernel
#endif // __CCE_AICORE__
#endif // MC2_RUNTIME_INFO_H
