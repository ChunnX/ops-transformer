/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_BLOCK_SYNC_H_
#define FLASH_ATTN_BLOCK_SYNC_H_

#include "../../../common/op_kernel/attn_buffer.h"

namespace FlashAttnKernel {

// Shared event IDs keep the Cube side of the DN/ND protocol consistent.
struct FlashAttnCubeSync {
    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CROSSCORE_MM_0 = 0U;
    static constexpr uint32_t CROSSCORE_MM_1 = 1U;
    static constexpr uint32_t CROSSCORE_MM_2 = 2U;
    static constexpr uint32_t CROSSCORE_MM_3 = 3U;
    static constexpr uint32_t CROSSCORE_L1P_0 = 5U;
    static constexpr uint32_t CROSSCORE_L1P_1 = 6U;
    static constexpr uint32_t CROSSCORE_L1P_2 = 7U;

    // 核内同步ID
    static constexpr uint32_t Q_L1_BUFFER_ID0 = 0U;
    static constexpr uint32_t Q_L1_BUFFER_ID1 = 1U;
    static constexpr uint32_t KV_L1_BUFFER_ID0 = 2U;
    static constexpr uint32_t KV_L1_BUFFER_ID1 = 3U;
    static constexpr uint32_t KV_L1_BUFFER_ID2 = 4U;
    static constexpr uint32_t KV_L1_BUFFER_ID3 = 5U;
    static constexpr uint32_t L0A_BUFFER_ID0 = 6U;
    static constexpr uint32_t L0A_BUFFER_ID1 = 7U;
    static constexpr uint32_t L0B_BUFFER_ID0 = 8U;
    static constexpr uint32_t L0B_BUFFER_ID1 = 9U;
    static constexpr uint32_t L0C_BUFFER_ID0 = 10U;
    static constexpr uint32_t L0C_BUFFER_ID1 = 11U;
    static constexpr uint32_t L0C_BUFFER_ID2 = 12U;
    static constexpr uint32_t L0C_BUFFER_ID3 = 13U;

    template <uint32_t D_BASE_SIZE>
    __aicore__ inline static void WaitMmResults()
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_1 + AIV0_AIV1_OFFSET);
        if constexpr (D_BASE_SIZE <= 128) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_2);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_2 + AIV0_AIV1_OFFSET);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_3);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_3 + AIV0_AIV1_OFFSET);
        }
    }
};

// Shared event IDs keep the Vec side of the DN/ND protocol consistent.
struct FlashAttnVecSync {
    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint32_t CC_MM_0 = 0U;
    static constexpr uint32_t CC_MM_1 = 1U;
    static constexpr uint32_t CC_MM_2 = 2U;
    static constexpr uint32_t CC_MM_3 = 3U;
    static constexpr uint32_t CC_L1P_0 = 5U;
    static constexpr uint32_t CC_L1P_1 = 6U;
    static constexpr uint32_t CC_L1P_2 = 7U;

    // 核内同步ID
    // MTE3<->V, 输出buffer
    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT1 = 3;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT1 = 5;

    template <uint32_t D_BASE_SIZE>
    __aicore__ inline static void InitMmResults()
    {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_1);
        if constexpr (D_BASE_SIZE <= 128) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_2);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_3);
        }
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}
};

} // namespace FlashAttnKernel

#endif // FLASH_ATTN_BLOCK_SYNC_H_
