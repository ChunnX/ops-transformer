/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEGA_MOE_GMM_TILE_SYNC_H
#define MEGA_MOE_GMM_TILE_SYNC_H

#include "kernel_operator.h"

#ifndef MEGA_MOE_GMM_HARD_TILE_SYNC
#define MEGA_MOE_GMM_HARD_TILE_SYNC 1
#endif
#ifndef MEGA_MOE_GMM_MAX_INFLIGHT_TILES
#define MEGA_MOE_GMM_MAX_INFLIGHT_TILES 15
#endif

namespace MegaMoeImpl {

// Separate GM-output GMM1/GMM2 FIFOs, retained across experts and waves.
// A logical GMM1 tile includes BOTH gate/up results; a GMM2 tile has one result.
// Output tensors retain their existing GM storage; this is a notification window,
// not an L1/L0C buffer count. Blaze's buffer and UnitFlag synchronization stays intact.
class GmmTileSync {
public:
    enum class Stage : uint16_t {
        GMM1 = 8,
        GMM2 = 10
    };

    enum class Consumer : uint16_t {
        AIV0 = 0,
        AIV1 = 16
    };

    static constexpr uint32_t MAX_INFLIGHT = MEGA_MOE_GMM_MAX_INFLIGHT_TILES;
    static_assert(MAX_INFLIGHT > 0U && MAX_INFLIGHT <= 15U, "Cross-core counters must stay in [0, 15]");

    __aicore__ explicit inline GmmTileSync(Stage stage, Consumer consumer)
        : aivReadyId_(static_cast<uint16_t>(stage)),
          aicReadyId_(static_cast<uint16_t>(stage) + static_cast<uint16_t>(consumer))
    {}

    __aicore__ inline void Acquire()
    {
        if (outstanding_ == MAX_INFLIGHT) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(aicReadyId_ + 1);
            --outstanding_;
        }
        ++outstanding_;
    }

    __aicore__ inline void PublishReady()
    {
        // Issued after both GMM1 Fixpipes (or the one GMM2 Fixpipe).
        // FIX, rather than S, publishes completion without draining Scalar.
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(aicReadyId_);
    }

    __aicore__ inline void WaitReady()
    {
        // Results are in GM: fence the MTE2 reader, not just Vector arithmetic.
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(aivReadyId_);
    }

    __aicore__ inline void Release()
    {
        // The consumer has consumed the tile; order acknowledgement after its output stores.
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(aivReadyId_ + 1);
    }

    __aicore__ inline void Drain()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            while (outstanding_ != 0U) {
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(aicReadyId_ + 1);
                --outstanding_;
            }
        }
    }

private:
    // Mode 4: AIC ids 0..15 address AIV0; ids 16..31 address AIV1.
    // GMM1 uses AIV ready/ack ids 8/9; GMM2 uses 10/11. AIC ids add the peer offset.
    // These do not overlap W4 prologue ids 0/1 or generic UB epilogue ids 4..7.
    const uint16_t aivReadyId_;
    const uint16_t aicReadyId_;
    // Count acquired tiles minus acknowledgements consumed by AIC, including
    // issued-but-not-yet-completed notifications. Both hardware counters <= this.
    uint32_t outstanding_ = 0;
};

} // namespace MegaMoeImpl
#endif
