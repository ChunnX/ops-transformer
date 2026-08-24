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
        mySlot->commPhase = static_cast<uint8_t>(Mc2Tiling::RT_PHASE_COMM_INIT);

        WriteGMByPassDCache();
    }

    __aicore__ inline void WriteGmByPassDCache(uint8_t turn, uint8_t position, uint8_t phase)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->execTurn = turn;
        mySlot->execPosition = position;
        mySlot->commPhase = phase;
        WriteGMByPassDCache();
    }

    __aicore__ inline void WritePhaseByPassDCache(uint8_t phase)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->commPhase = phase;
        WriteGMByPassDCache();
    }

    __aicore__ inline void WriteTurnByPassDCache(uint8_t tileIdx)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->execTurn = tileIdx;
        WriteGMByPassDCache();
    }

    __aicore__ inline void WritePosTagByPassDCache(uint8_t tag)
    {
        if (runtimeInfoBase_ == nullptr) {
            return;
        }
        __gm__ Mc2Tiling::RuntimeInfoPerCore *mySlot = &runtimeInfoBase_[slotIdx_];
        mySlot->execPosition = tag;
        WriteGMByPassDCache();
    }

private:
    __aicore__ inline void WriteGMByPassDCache()
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
