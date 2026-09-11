/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEGA_MOE_GMM_EPILOGUE_SYNC_H
#define MEGA_MOE_GMM_EPILOGUE_SYNC_H

#include "kernel_operator.h"

namespace MegaMoeImpl {

// 每条同步链路各允许 AIC 领先 15 个逻辑 tile，计数跨 expert/wave 保留。
// 每个 tile 的所有 GM 输出完成后通知消费者，消费者处理完所有子 tile 后通知 AIC。
constexpr uint32_t GMM_MAX_PENDING_TILES = 15;

// Mode 4 flag IDs: each pair holds a producer-ready and consumer-ACK event.
enum class GmmEventPair : uint16_t {
    MOE_GMM1 = 8,
    MOE_GMM2 = 10,
    SHARED_GMM1 = 12,
};
constexpr uint16_t GMM_AIV_FLAG_OFFSET = 16U;

// GMM1 -> Activation：A8W8/A4W4 使用 AIV0，A8W4 使用 AIV1。
class Gmm1ActivationSync {
public:
    __aicore__ explicit inline Gmm1ActivationSync(uint16_t activationSubBlockIdx,
                                                  GmmEventPair eventPair = GmmEventPair::MOE_GMM1)
        : aivFlagOffset_(activationSubBlockIdx * GMM_AIV_FLAG_OFFSET),
          gmm1FinishedFlag_(static_cast<uint16_t>(eventPair)),
          activationFinishedFlag_(gmm1FinishedFlag_ + 1U)
    {}

    // AIC：提交前预留额度。累计第 16 个 tile 起，每次消费一个 ACK；已到达的 ACK 可直接通过。
    // PIPE_S gates submission itself. PIPE_MTE2 would only stall loads and cannot
    // enforce this bound for work that can advance without another MTE2 load.
    __aicore__ inline void WaitForActivation()
    {
        if (pendingTiles_ == GMM_MAX_PENDING_TILES) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(activationFinishedFlag_ + aivFlagOffset_);
            --pendingTiles_;
        }
        ++pendingTiles_;
    }

    // AIC：整个交织 tile 的 Fixpipe 已提交，由 FIX 通知 Activation 读取 GM。
    __aicore__ inline void NotifyActivation()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(gmm1FinishedFlag_ + aivFlagOffset_);
    }

    // AIV：在 MTE2 读取 GMM1 结果前等待。
    __aicore__ inline void WaitForGmm1()
    {
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(gmm1FinishedFlag_);
    }

    // AIV：整个逻辑 tile 的 Activation/量化输出写回后，归还一个额度。
    __aicore__ inline void NotifyGmm1()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(activationFinishedFlag_);
    }

    // 完整 MoE 流（包括滞后的 GMM2）结束后排空；共享 GMM1 的 ACK 也延后至此回收。
    __aicore__ inline void EndSync()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            while (pendingTiles_ != 0U) {
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(activationFinishedFlag_ + aivFlagOffset_);
                --pendingTiles_;
            }
        }
    }

private:
    // Mode 4：AIC 的 flagId 加 16 选择 AIV1；AIV 使用本地 flagId。
    const uint16_t aivFlagOffset_;
    const uint16_t gmm1FinishedFlag_;
    const uint16_t activationFinishedFlag_;
    uint32_t pendingTiles_ = 0; // 已提交的 tile 数减去 AIC 已消费的 Activation 完成通知数。
};

// GMM2 -> 非量化 Combine：固定使用 AIV1，事件和计数独立于 GMM1/Activation。
class Gmm2CombineSync {
public:
    // AIC：提交前预留额度。累计第 16 个 tile 起，每次消费一个 ACK；已到达的 ACK 可直接通过。
    // PIPE_S gates submission itself. PIPE_MTE2 would only stall loads and cannot
    // enforce this bound for work that can advance without another MTE2 load.
    __aicore__ inline void WaitForCombine()
    {
        if (pendingTiles_ == GMM_MAX_PENDING_TILES) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(COMBINE_FINISHED_FLAG + GMM_AIV_FLAG_OFFSET);
            --pendingTiles_;
        }
        ++pendingTiles_;
    }

    __aicore__ inline void NotifyCombine()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(GMM2_FINISHED_FLAG + GMM_AIV_FLAG_OFFSET);
    }

    __aicore__ inline void WaitForGmm2()
    {
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(GMM2_FINISHED_FLAG);
    }

    // AIV1：Combine 输出写回后，归还一个额度。
    __aicore__ inline void NotifyGmm2()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(COMBINE_FINISHED_FLAG);
    }

    __aicore__ inline void EndSync()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            while (pendingTiles_ != 0U) {
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(COMBINE_FINISHED_FLAG + GMM_AIV_FLAG_OFFSET);
                --pendingTiles_;
            }
        }
    }

private:
    static constexpr uint16_t GMM2_FINISHED_FLAG = static_cast<uint16_t>(GmmEventPair::MOE_GMM2);
    static constexpr uint16_t COMBINE_FINISHED_FLAG = GMM2_FINISHED_FLAG + 1U;
    uint32_t pendingTiles_ = 0;
};

} // namespace MegaMoeImpl
#endif
