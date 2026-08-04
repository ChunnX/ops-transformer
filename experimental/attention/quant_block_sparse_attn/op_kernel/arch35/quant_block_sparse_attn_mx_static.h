/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_H_

#include "kernel_operator.h"

namespace BaseApi {
namespace MxStatic {

constexpr uint32_t UB_CAPACITY = 256U * 1024U;
constexpr uint32_t L1_CAPACITY = 512U * 1024U;
constexpr uint32_t L0A_CAPACITY = 64U * 1024U;
constexpr uint32_t L0B_CAPACITY = 64U * 1024U;
constexpr uint32_t L0C_CAPACITY = 256U * 1024U;

constexpr uint32_t BMM1_UB_SLOT_BYTES = 64U * 512U * sizeof(float) / 2U;
constexpr uint32_t BMM2_UB_SLOT_BYTES = 64U * 128U * sizeof(float);
constexpr uint32_t BMM1_UB_ADDR0 = 0U;
constexpr uint32_t BMM1_UB_ADDR1 = BMM1_UB_ADDR0 + BMM1_UB_SLOT_BYTES;
constexpr uint32_t BMM2_UB_ADDR = BMM1_UB_ADDR1 + BMM1_UB_SLOT_BYTES;
constexpr uint32_t VEC_PRIVATE_UB_BASE = BMM2_UB_ADDR + BMM2_UB_SLOT_BYTES;

constexpr uint32_t L1_P_SLOT_BYTES = 128U * 512U + 128U * 512U / 32U;
constexpr uint32_t L1_P_ADDR0 = 0U;
constexpr uint32_t L1_P_ADDR1 = L1_P_ADDR0 + L1_P_SLOT_BYTES;
constexpr uint32_t L1_P_ADDR2 = L1_P_ADDR1 + L1_P_SLOT_BYTES;
constexpr uint32_t CUBE_PRIVATE_L1_BASE = L1_P_ADDR2 + L1_P_SLOT_BYTES;

constexpr uint32_t L0_SLOT_BYTES = 32U * 1024U;
constexpr uint32_t L0C_SLOT_BYTES = 128U * 1024U;

static_assert(VEC_PRIVATE_UB_BASE == 163840U, "Unexpected MX shared UB layout");
static_assert(CUBE_PRIVATE_L1_BASE == 202752U, "Unexpected MX shared L1 layout");
static_assert(VEC_PRIVATE_UB_BASE <= UB_CAPACITY, "MX shared UB exceeds capacity");
static_assert(CUBE_PRIVATE_L1_BASE <= L1_CAPACITY, "MX shared L1 exceeds capacity");
static_assert(2U * L0_SLOT_BYTES <= L0A_CAPACITY, "MX L0A exceeds capacity");
static_assert(2U * L0_SLOT_BYTES <= L0B_CAPACITY, "MX L0B exceeds capacity");
static_assert(2U * L0C_SLOT_BYTES <= L0C_CAPACITY, "MX L0C exceeds capacity");

constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
constexpr uint32_t MX_AIV0_AIV1_OFFSET = 16U;
// CROSS_CORE_SYNC_BOTH requires independent producer->consumer (ready) and
// consumer->producer (free) counters. Sharing one counter lets a ready token
// satisfy the producer's free wait, so the AIC can overwrite UB before both
// AIVs have consumed it. Event IDs 6 and 7 are reserved by static Tensor.
constexpr uint32_t CC_BMM1_READY_EVENT_ID0 = 0U;
constexpr uint32_t CC_BMM1_FREE_EVENT_ID0 = 1U;
constexpr uint32_t CC_BMM1_READY_EVENT_ID1 = 2U;
constexpr uint32_t CC_BMM1_FREE_EVENT_ID1 = 3U;
constexpr uint32_t CC_BMM2_READY_EVENT_ID = 4U;
constexpr uint32_t CC_BMM2_FREE_EVENT_ID = 5U;
constexpr uint32_t CC_L1P_READY_EVENT_ID0 = 8U;
constexpr uint32_t CC_L1P_READY_EVENT_ID1 = 9U;
constexpr uint32_t CC_L1P_READY_EVENT_ID2 = 10U;

constexpr AscendC::TEventID L1_Q_EVENT_ID0 = 0U;
constexpr AscendC::TEventID L1_Q_EVENT_ID1 = 1U;
constexpr AscendC::TEventID L1_K_EVENT_ID0 = 2U;
constexpr AscendC::TEventID L1_K_EVENT_ID1 = 3U;
constexpr AscendC::TEventID L1_K_EVENT_ID2 = 4U;
constexpr AscendC::TEventID L1_K_EVENT_ID3 = 5U;
constexpr AscendC::TEventID L1_V_EVENT_ID0 = 8U;
constexpr AscendC::TEventID L1_V_EVENT_ID1 = 9U;

constexpr AscendC::TEventID L0A_EVENT_ID0 = 0U;
constexpr AscendC::TEventID L0A_EVENT_ID1 = 1U;
constexpr AscendC::TEventID L0B_EVENT_ID0 = 2U;
constexpr AscendC::TEventID L0B_EVENT_ID1 = 3U;
constexpr AscendC::TEventID L0C_EVENT_ID0 = 0U;
constexpr AscendC::TEventID L0C_EVENT_ID1 = 1U;

constexpr AscendC::TEventID STAGE1_EVENT_ID0 = 0U;
constexpr AscendC::TEventID STAGE1_EVENT_ID1 = 1U;
constexpr AscendC::TEventID PSCALE_EVENT_ID = 2U;
constexpr AscendC::TEventID STAGE2_EVENT_ID = 3U;
constexpr AscendC::TEventID LSE_EVENT_ID = 4U;
constexpr AscendC::TEventID INIT_OUTPUT_FREE_EVENT_ID = 5U;
constexpr AscendC::TEventID INIT_OUTPUT_READY_EVENT_ID = 10U;
constexpr AscendC::TEventID MASK_EVENT_ID = 0U;

__aicore__ inline uint32_t Bmm1ReadyEventId(uint32_t slot)
{
    return slot == 0U ? CC_BMM1_READY_EVENT_ID0 : CC_BMM1_READY_EVENT_ID1;
}

__aicore__ inline uint32_t Bmm1FreeEventId(uint32_t slot)
{
    return slot == 0U ? CC_BMM1_FREE_EVENT_ID0 : CC_BMM1_FREE_EVENT_ID1;
}

__aicore__ inline AscendC::TEventID Stage1EventId(uint32_t slot)
{
    return slot == 0U ? STAGE1_EVENT_ID0 : STAGE1_EVENT_ID1;
}

__aicore__ inline uint32_t L1PEventId(uint32_t slot)
{
    if (slot == 0U) {
        return CC_L1P_READY_EVENT_ID0;
    }
    return slot == 1U ? CC_L1P_READY_EVENT_ID1 : CC_L1P_READY_EVENT_ID2;
}

__aicore__ inline void AicWaitUbFree(uint32_t eventId)
{
    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(eventId);
    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(eventId + MX_AIV0_AIV1_OFFSET);
}

__aicore__ inline void AicSetUbReady(uint32_t eventId)
{
    AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(eventId);
    AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(eventId + MX_AIV0_AIV1_OFFSET);
}

__aicore__ inline void AivWaitUbReady(uint32_t eventId)
{
    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(eventId);
}

__aicore__ inline void AivSetUbFree(uint32_t eventId)
{
    // Mode 4 maps AIV1 to the AIC eventId+16 counter in hardware. Do not add 16 here.
    AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(eventId);
}

__aicore__ inline void AivSetL1PReady(uint32_t eventId)
{
    AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(eventId);
}

__aicore__ inline void AicWaitL1PReady(uint32_t eventId)
{
    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(eventId);
    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(eventId + MX_AIV0_AIV1_OFFSET);
}

} // namespace MxStatic
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_H_
