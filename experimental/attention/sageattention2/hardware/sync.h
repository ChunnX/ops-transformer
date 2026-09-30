/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "../common/constants.h"
#include "../common/types.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#endif

namespace sage::hardware {

// ============================================================================
// Asynchronous Pipeline Token Signals
// ============================================================================
enum class TokenSignal : uint32_t {
    QK_READY = 0,
    P_READY = 1,
    PV_READY = 2,
    PV_RELEASE = 3,
    Q_READY = 4,
    AIV_FLAG_OFFSET = 16,
};

// ============================================================================
// Strongly-Typed Hardware Event Slots and Cross-Core Handshake State Machine
// ============================================================================
struct EventHandshake {
    // MTE1 slot events arbitrating the shared QK / PV L1 tile regions on the AIC
    static constexpr uint32_t QK_EVENT = 2;
    static constexpr uint32_t PV_EVENT = 3;
    // AIC MTE2 parity slots for the wide-pipeline GM->L1 prefetch engines (K then V)
    static constexpr uint32_t K_PREFETCH_EVENT_BASE = 4;
    static constexpr uint32_t V_PREFETCH_EVENT_BASE = 6;
    // AIV MTE2 slots double-buffering the per-group K scale tiles (AIV-local domain)
    static constexpr uint32_t SCALE_EVENT_BASE = 4;
    // FIX slot event pairing the second QK sub-block fixpipe with its MMA
    static constexpr uint32_t QK_SECOND_FIX_EVENT = 1;

#if defined(__CCE__)
    // Explicit cross-core synchronization primitives
    template <auto PIPE>
    __aicore__ static inline void NotifyAiv(TokenSignal sig, uint32_t aivIdx = 0) {
        AscendC::CrossCoreSetFlag<4, PIPE>(static_cast<uint32_t>(sig) + aivIdx * static_cast<uint32_t>(TokenSignal::AIV_FLAG_OFFSET));
    }

    template <auto PIPE>
    __aicore__ static inline void NotifyBothAiv(TokenSignal sig) {
        AscendC::CrossCoreSetFlag<4, PIPE>(static_cast<uint32_t>(sig));
        AscendC::CrossCoreSetFlag<4, PIPE>(static_cast<uint32_t>(sig) + static_cast<uint32_t>(TokenSignal::AIV_FLAG_OFFSET));
    }

    template <auto PIPE>
    __aicore__ static inline void WaitFromCube(TokenSignal sig) {
        uint32_t subBlock = AscendC::GetSubBlockIdx();
        AscendC::CrossCoreWaitFlag<4, PIPE>(static_cast<uint32_t>(sig) + subBlock * static_cast<uint32_t>(TokenSignal::AIV_FLAG_OFFSET));
    }

    template <auto PIPE>
    __aicore__ static inline void NotifyCube(TokenSignal sig) {
        uint32_t subBlock = AscendC::GetSubBlockIdx();
        AscendC::CrossCoreSetFlag<4, PIPE>(static_cast<uint32_t>(sig) + subBlock * static_cast<uint32_t>(TokenSignal::AIV_FLAG_OFFSET));
    }

    template <auto PIPE>
    __aicore__ static inline void WaitFromBothAiv(TokenSignal sig) {
        AscendC::CrossCoreWaitFlag<4, PIPE>(static_cast<uint32_t>(sig));
        AscendC::CrossCoreWaitFlag<4, PIPE>(static_cast<uint32_t>(sig) + static_cast<uint32_t>(TokenSignal::AIV_FLAG_OFFSET));
    }
#endif
};

#if defined(__CCE__)
template <AscendC::HardEvent E>
__aicore__ inline void Sync() {
    AscendC::SetFlag<E>(0);
    AscendC::WaitFlag<E>(0);
}
#endif

} // namespace sage::hardware

namespace sage {
using hardware::TokenSignal;
using hardware::EventHandshake;
#if defined(__CCE__)
using hardware::Sync;
#endif
} // namespace sage
