#pragma once

#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#endif

namespace sage::orchestration {

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

    static inline constexpr uint32_t QK_EVENT = 2U;
    static inline constexpr uint32_t PV_EVENT = 3U;
    static inline constexpr uint32_t SCALE_EVENT_BASE = 4U;
    static inline constexpr uint32_t QK_SECOND_FIX_EVENT = 1U;
    static inline constexpr uint32_t K_PREFETCH_EVENT_BASE = 4U;
    static inline constexpr uint32_t V_PREFETCH_EVENT_BASE = 6U;

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
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t ticket = static_cast<uint32_t>(sig) + subBlock * kAivFlagOffset;
        AscendC::CrossCoreWaitFlag<4, PIPE>(ticket);
    }

    template <auto PIPE>
    __aicore__ static inline void NotifyCube(TokenSignal sig) noexcept {
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
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
