#pragma once

#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#endif

namespace sage::orchestration {

// ============================================================================
// DeepEP & DeepGEMM Asynchronous Credit Handshake Architecture:
// 1. Cross-Core Ticket Handshake (DeepEP Inspired):
//    On Atlas 950PR (1 AIC + 2 AIV), inter-core synchronization is decoupled
//    into unidirectional ticket signals. Low bits select the logical channel,
//    while AIV_FLAG_OFFSET (16) partitions tickets across the two Vector cores.
//    This guarantees deadlock-free asynchronous signaling across heterogeneous pipes.
//
// 2. Decoupled SubPart Pipeline Credits (DeepGEMM Inspired):
//    SubPart 0 (QK_EVENT = 2U) and SubPart 1 (QK_SECOND_FIX_EVENT = 1U) utilize
//    independent hardware event registers. This decouples the Cube matrix engine
//    (PIPE_M) from the dequantization/fixpipe engine (PIPE_FIX), enabling true
//    hardware-level overlap where SubPart 0 Fixpipe and SubPart 1 MMAD execute
//    concurrently without pipeline stalls.
// ============================================================================

// Cross-core ticket numbers. The low bits select the channel, the AIV offset adds the
// per-sub-block stride so both AIVs of one AIC own distinct tickets.
enum class TokenSignal : uint32_t {
    QK_READY = 0U,
    P_READY = 1U,
    PV_READY = 2U,
    PV_RELEASE = 3U,
    Q_READY = 4U,
    AIV_FLAG_OFFSET = 16U,
};

// Strong-typed ticket-based cross-core handshake protocol
struct EventHandshake final {
    static inline constexpr uint32_t kAivFlagOffset = static_cast<uint32_t>(TokenSignal::AIV_FLAG_OFFSET);

    static inline constexpr uint32_t QK_EVENT = 2U;             // Primary credit event for SubPart 0 (PIPE_M <-> PIPE_FIX)
    static inline constexpr uint32_t PV_EVENT = 3U;             // PV GEMM pipeline credit event
    static inline constexpr uint32_t SCALE_EVENT_BASE = 4U;     // Dynamic scaling synchronization event base
    static inline constexpr uint32_t QK_SECOND_FIX_EVENT = 1U;  // Independent credit event for SubPart 1 (PIPE_M <-> PIPE_FIX)
    static inline constexpr uint32_t K_PREFETCH_EVENT_BASE = 4U;// Key prefetch double-buffer event base
    static inline constexpr uint32_t V_PREFETCH_EVENT_BASE = 6U;// Value prefetch double-buffer event base

#if defined(__CCE__)
    template <auto PIPE>
    __aicore__ static inline void NotifyAiv(TokenSignal sig, uint32_t aivIdx = 0U) noexcept {
        const uint32_t ticket = static_cast<uint32_t>(sig) + aivIdx * kAivFlagOffset;
        AscendC::CrossCoreSetFlag<4, PIPE>(ticket);
    }

    template <auto PIPE>
    __aicore__ static inline void NotifyBothAiv(TokenSignal sig) noexcept {
        const uint32_t ticketBase = static_cast<uint32_t>(sig);
        AscendC::CrossCoreSetFlag<4, PIPE>(ticketBase);
        AscendC::CrossCoreSetFlag<4, PIPE>(ticketBase + kAivFlagOffset);
    }

    template <auto PIPE>
    __aicore__ static inline void WaitFromCube(TokenSignal sig) noexcept {
        // Per-AIV ticket partitioning is load-bearing: Cube notifies with
        // NotifyBothAiv (base + base+kAivFlagOffset) and each AIV must consume
        // exactly its own half, or one Vector core starves on a flag that is
        // never re-set. Symmetrically, NotifyCube lets Cube's WaitFromBothAiv
        // observe both AIV readiness reports.
        uint32_t subBlock = 0U;
        if ASCEND_IS_AIV {
            subBlock = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        }
        const uint32_t ticket = static_cast<uint32_t>(sig) + subBlock * kAivFlagOffset;
        AscendC::CrossCoreWaitFlag<4, PIPE>(ticket);
    }

    template <auto PIPE>
    __aicore__ static inline void NotifyCube(TokenSignal sig) noexcept {
        uint32_t subBlock = 0U;
        if ASCEND_IS_AIV {
            subBlock = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        }
        const uint32_t ticket = static_cast<uint32_t>(sig) + subBlock * kAivFlagOffset;
        AscendC::CrossCoreSetFlag<4, PIPE>(ticket);
    }

    template <auto PIPE>
    __aicore__ static inline void WaitFromBothAiv(TokenSignal sig) noexcept {
        const uint32_t ticketBase = static_cast<uint32_t>(sig);
        AscendC::CrossCoreWaitFlag<4, PIPE>(ticketBase);
        AscendC::CrossCoreWaitFlag<4, PIPE>(ticketBase + kAivFlagOffset);
    }
#endif
};

} // namespace sage::orchestration

namespace sage {
inline namespace orchestration_forwarding {
using namespace ::sage::orchestration;
}
}
