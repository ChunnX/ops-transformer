/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "lib/hccl/hccl.h"
#include "basic_api/kernel_operator_block_sync_intf.h"

namespace Apace {
static constexpr AscendC::HcclServerConfig HCOMM_CONFIG = {AscendC::CoreType::ON_AIC, 0};
// Separate from SyncAll's reserved flags 11-14 and the matmul API's flags 0-7.
static constexpr uint16_t HCOMM_DATA_READY_FLAG = 9;
static constexpr uint16_t HCOMM_WINDOW_ACK_FLAG = 10;
static constexpr uint32_t HCOMM_NOTIFY_WINDOW = 8;

template <AscendC::HcclServerType ServerType>
struct HcommCommState {
    AscendC::Hccl<ServerType, HCOMM_CONFIG> hccl_;
    AscendC::HcclHandle scaleHandle_{0};
    AscendC::HcclHandle dataHeadHandle_{0};
    AscendC::HcclHandle dataTailHandle_{0};
    uint32_t headTileCnt_{0};
    uint32_t totalTileCnt_{0};
    uint32_t nextNotifyTile_{0};
    uint32_t acknowledgedTiles_{0};
};

// Both paired AIVs acknowledge a window; mode 2 combines them into one AIC event.
// The scalar wait is intentional: a pipe-only wait must not allow the sender loop to run ahead.
struct HcommOutputTileSync {
    __aicore__ static inline void NotifyReady()
    {
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(HCOMM_DATA_READY_FLAG);
    }

    __aicore__ static inline void WaitWindowAck()
    {
        AscendC::CrossCoreWaitFlag(HCOMM_WINDOW_ACK_FLAG);
    }

    __aicore__ static inline void WaitReady()
    {
        AscendC::CrossCoreWaitFlag(HCOMM_DATA_READY_FLAG);
        AscendC::SyncAll<true>();
    }

    __aicore__ static inline void CompleteTile(uint32_t tileIdx, uint32_t totalTiles)
    {
        if ((tileIdx + 1) % HCOMM_NOTIFY_WINDOW == 0 || tileIdx + 1 == totalTiles) {
            // Step4 must finish/enqueue this tile's GM writes before reaching this point.
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(HCOMM_WINDOW_ACK_FLAG);
        }
    }
};

template <AscendC::HcclServerType ServerType, bool EnableHcommAllToAllOut>
struct HcommCommWaitPolicy {
    HcommCommState<ServerType> *state_{nullptr};
    uint32_t slotNum{0};

    __aicore__ inline void Release(uint32_t) {}

    __aicore__ inline void Acquire(uint32_t tileIdx)
    {
        if constexpr (!EnableHcommAllToAllOut) {
            // Keep legacy Wait behavior, including its original error handling.
            if (tileIdx == 0) {
                state_->hccl_.Wait(state_->scaleHandle_);
            }
            if (tileIdx < state_->headTileCnt_) {
                state_->hccl_.Wait(state_->dataHeadHandle_);
            } else {
                state_->hccl_.Wait(state_->dataTailHandle_);
            }
        } else {
            static_assert(ServerType == AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU,
                          "Hcomm output notifications require CCU");
            ASSERT_MSG(tileIdx < state_->totalTileCnt_, "hcomm tile index out of range");
            while (state_->nextNotifyTile_ <= tileIdx) {
                if (state_->nextNotifyTile_ - state_->acknowledgedTiles_ == HCOMM_NOTIFY_WINDOW) {
                    HcommOutputTileSync::WaitWindowAck();
                    state_->acknowledgedTiles_ = state_->nextNotifyTile_;
                }
                const uint32_t nextTile = state_->nextNotifyTile_;
                if (nextTile == 0) {
                    const int32_t ret = state_->hccl_.Wait(state_->scaleHandle_);
                    ASSERT_MSG(ret == 0, "hcomm scale Wait failed: %d", ret);
                }
                const auto handle = nextTile < state_->headTileCnt_ ? state_->dataHeadHandle_ : state_->dataTailHandle_;
                const int32_t ret = state_->hccl_.Wait(handle);
                // Fail before announcing unreadable data. This is a fatal diagnostic, not collective recovery.
                ASSERT_MSG(ret == 0, "hcomm data Wait failed at tile %u: %d", nextTile, ret);
                HcommOutputTileSync::NotifyReady();
                ++state_->nextNotifyTile_;
            }
        }
    }

    __aicore__ inline void FinishNotifications()
    {
        if constexpr (EnableHcommAllToAllOut) {
            // Covers idle cores and an early return from the matmul scheduler. Never re-wait a completed tile.
            if (state_->nextNotifyTile_ < state_->totalTileCnt_) {
                Acquire(state_->totalTileCnt_ - 1);
            }
            if (state_->acknowledgedTiles_ < state_->nextNotifyTile_) {
                HcommOutputTileSync::WaitWindowAck();
                state_->acknowledgedTiles_ = state_->nextNotifyTile_;
            }
        }
    }
};
} // namespace Apace
