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

// 两条同步链路各允许 AIC 领先 15 个逻辑 tile，计数跨 expert/wave 保留。
// 每个 tile 的所有 GM 输出完成后通知消费者，消费者处理完所有子 tile 后通知 AIC。
constexpr uint32_t GMM_MAX_PENDING_TILES = 15;

// GMM1 -> SwiGLU：A8W8/A4W4 使用 AIV0，A8W4 使用 AIV1。
class Gmm1SwigluSync {
public:
    __aicore__ explicit inline Gmm1SwigluSync(uint16_t swigluSubBlockIdx)
        : aivFlagOffset_(swigluSubBlockIdx * 16U)
    {}

    // AIC：提交前预留额度。累计第 16 个 tile 起，每次消费一个 ACK；已到达的 ACK 可直接通过。
    __aicore__ inline void WaitForSwiglu()
    {
        if (pendingTiles_ == GMM_MAX_PENDING_TILES) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(SWIGLU_FINISHED_FLAG + aivFlagOffset_);
            --pendingTiles_;
        }
        ++pendingTiles_;
    }

    // AIC：整个交织 tile 的 Fixpipe 已提交，由 FIX 通知 SwiGLU 读取 GM。
    __aicore__ inline void NotifySwiglu()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(GMM1_FINISHED_FLAG + aivFlagOffset_);
    }

    // AIV：在 MTE2 读取 GMM1 结果前等待。
    __aicore__ inline void WaitForGmm1()
    {
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(GMM1_FINISHED_FLAG);
    }

    // AIV：整个逻辑 tile 的 SwiGLU/量化输出写回后，归还一个额度。
    __aicore__ inline void NotifyGmm1()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(SWIGLU_FINISHED_FLAG);
    }

    // 仅在整个专家流（包括滞后的 GMM2）结束时收回剩余额度。
    __aicore__ inline void EndSync()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            while (pendingTiles_ != 0U) {
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(SWIGLU_FINISHED_FLAG + aivFlagOffset_);
                --pendingTiles_;
            }
        }
    }

private:
    // Mode 4：AIC 的 flagId 加 16 选择 AIV1；AIV 使用本地 flagId。
    static constexpr uint16_t GMM1_FINISHED_FLAG = 8;
    static constexpr uint16_t SWIGLU_FINISHED_FLAG = 9;
    const uint16_t aivFlagOffset_;
    uint32_t pendingTiles_ = 0; // 已提交的 tile 数减去 AIC 已消费的 SwiGLU 完成通知数。
};

// GMM2 -> 非量化 Combine：固定使用 AIV1，事件和计数独立于 GMM1/SwiGLU。
class Gmm2CombineSync {
public:
    // AIC：提交前预留额度。累计第 16 个 tile 起，每次消费一个 ACK；已到达的 ACK 可直接通过。
    __aicore__ inline void WaitForCombine()
    {
        if (pendingTiles_ == GMM_MAX_PENDING_TILES) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(COMBINE_FINISHED_FLAG + AIV1_FLAG_OFFSET);
            --pendingTiles_;
        }
        ++pendingTiles_;
    }

    __aicore__ inline void NotifyCombine()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(GMM2_FINISHED_FLAG + AIV1_FLAG_OFFSET);
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
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(COMBINE_FINISHED_FLAG + AIV1_FLAG_OFFSET);
                --pendingTiles_;
            }
        }
    }

private:
    static constexpr uint16_t GMM2_FINISHED_FLAG = 10;
    static constexpr uint16_t COMBINE_FINISHED_FLAG = 11;
    static constexpr uint16_t AIV1_FLAG_OFFSET = 16;
    uint32_t pendingTiles_ = 0;
};

} // namespace MegaMoeImpl
#endif
