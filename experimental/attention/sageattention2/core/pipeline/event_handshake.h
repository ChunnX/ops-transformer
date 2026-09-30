/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#if defined(__CCE__)
#include "kernel_operator.h"
#endif

namespace sage::pipeline {

// Cross-core / inter-engine token signaling states
enum class TokenSignal : uint32_t {
    QK_READY = 0,
    P_READY = 1,
    PV_READY = 2,
    PV_RELEASE = 3,
    Q_READY = 4,
    AIV_FLAG_OFFSET = 16,
};

// Strongly-typed hardware event slots and synchronization primitives for AIC/AIV coordination
struct EventHandshake {
    static constexpr uint32_t QK_READY = static_cast<uint32_t>(TokenSignal::QK_READY);
    static constexpr uint32_t P_READY = static_cast<uint32_t>(TokenSignal::P_READY);
    static constexpr uint32_t PV_READY = static_cast<uint32_t>(TokenSignal::PV_READY);
    static constexpr uint32_t PV_RELEASE = static_cast<uint32_t>(TokenSignal::PV_RELEASE);
    static constexpr uint32_t Q_READY = static_cast<uint32_t>(TokenSignal::Q_READY);
    static constexpr uint32_t AIV_FLAG_OFFSET = static_cast<uint32_t>(TokenSignal::AIV_FLAG_OFFSET);

    static constexpr uint32_t QK_EVENT = 2;
    static constexpr uint32_t PV_EVENT = 3;
    static constexpr uint32_t SCALE_EVENT_BASE = 4;
    static constexpr uint32_t QK_SECOND_FIX_EVENT = 1;

#if defined(__CCE__)
    // Explicit cross-core synchronization primitives
    template <auto PIPE>
    __aicore__ static inline void NotifyAiv(TokenSignal sig, uint32_t aivIdx = 0) {
        AscendC::CrossCoreSetFlag<4, PIPE>(static_cast<uint32_t>(sig) + aivIdx * AIV_FLAG_OFFSET);
    }

    template <auto PIPE>
    __aicore__ static inline void NotifyBothAiv(TokenSignal sig) {
        AscendC::CrossCoreSetFlag<4, PIPE>(static_cast<uint32_t>(sig));
        AscendC::CrossCoreSetFlag<4, PIPE>(static_cast<uint32_t>(sig) + AIV_FLAG_OFFSET);
    }

    template <auto PIPE>
    __aicore__ static inline void WaitFromCube(TokenSignal sig) {
        uint32_t subBlock = AscendC::GetSubBlockIdx();
        AscendC::CrossCoreWaitFlag<4, PIPE>(static_cast<uint32_t>(sig) + subBlock * AIV_FLAG_OFFSET);
    }

    template <auto PIPE>
    __aicore__ static inline void NotifyCube(TokenSignal sig) {
        uint32_t subBlock = AscendC::GetSubBlockIdx();
        AscendC::CrossCoreSetFlag<4, PIPE>(static_cast<uint32_t>(sig) + subBlock * AIV_FLAG_OFFSET);
    }

    template <auto PIPE>
    __aicore__ static inline void WaitFromBothAiv(TokenSignal sig) {
        AscendC::CrossCoreWaitFlag<4, PIPE>(static_cast<uint32_t>(sig));
        AscendC::CrossCoreWaitFlag<4, PIPE>(static_cast<uint32_t>(sig) + AIV_FLAG_OFFSET);
    }
#endif
};

} // namespace sage::pipeline
