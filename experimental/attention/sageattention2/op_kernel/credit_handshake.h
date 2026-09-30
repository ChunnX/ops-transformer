#pragma once

#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#endif

namespace sage::orchestration {

template <uint32_t kSubblock>
struct IntraBlockEvent final {
    uint32_t base{0};
    constexpr explicit IntraBlockEvent(uint32_t b = 0) noexcept : base(b) {}
    constexpr operator uint32_t() const noexcept { return base; }
};

enum class TokenSignal : uint32_t {
    QK_READY = 0U * 1U,
    P_READY = 1U * 1U,
    PV_READY = 2U * 1U,
    PV_RELEASE = 3U * 1U,
    Q_READY = 4U * 1U,
    AIV_FLAG_OFFSET = 16U * 1U,
};

// Strong-typed ticket-based cross-core handshake protocol
struct EventHandshake final {
    static inline constexpr uint32_t QK_READY = static_cast<uint32_t>(TokenSignal::QK_READY);
    static inline constexpr uint32_t P_READY = static_cast<uint32_t>(TokenSignal::P_READY);
    static inline constexpr uint32_t PV_READY = static_cast<uint32_t>(TokenSignal::PV_READY);
    static inline constexpr uint32_t PV_RELEASE = static_cast<uint32_t>(TokenSignal::PV_RELEASE);
    static inline constexpr uint32_t Q_READY = static_cast<uint32_t>(TokenSignal::Q_READY);
    static inline constexpr uint32_t AIV_FLAG_OFFSET = static_cast<uint32_t>(TokenSignal::AIV_FLAG_OFFSET);

    static inline constexpr uint32_t QK_EVENT = 2U;
    static inline constexpr uint32_t PV_EVENT = 3U;
    static inline constexpr uint32_t SCALE_EVENT_BASE = 4U;
    static inline constexpr uint32_t QK_SECOND_FIX_EVENT = 1U;
    static inline constexpr uint32_t K_PREFETCH_EVENT_BASE = 4U;
    static inline constexpr uint32_t V_PREFETCH_EVENT_BASE = 6U;

    // Strong-typed IntraBlock cross-core events (1 AIC + 2 AIV)
    static inline constexpr IntraBlockEvent<0> kQkReadyAiv0{QK_READY};
    static inline constexpr IntraBlockEvent<1> kQkReadyAiv1{QK_READY + AIV_FLAG_OFFSET};
    static inline constexpr IntraBlockEvent<0> kPReadyAiv0{P_READY};
    static inline constexpr IntraBlockEvent<1> kPReadyAiv1{P_READY + AIV_FLAG_OFFSET};
    static inline constexpr IntraBlockEvent<0> kPvReadyAiv0{PV_READY};
    static inline constexpr IntraBlockEvent<1> kPvReadyAiv1{PV_READY + AIV_FLAG_OFFSET};
    static inline constexpr IntraBlockEvent<0> kPvReleaseAiv0{PV_RELEASE};
    static inline constexpr IntraBlockEvent<1> kPvReleaseAiv1{PV_RELEASE + AIV_FLAG_OFFSET};
    static inline constexpr IntraBlockEvent<0> kQReadyAiv0{Q_READY};
    static inline constexpr IntraBlockEvent<1> kQReadyAiv1{Q_READY + AIV_FLAG_OFFSET};

#if defined(__CCE__)
    template <auto PIPE>
    __aicore__ static inline void NotifyAiv(TokenSignal sig, uint32_t aivIdx = 0U) noexcept {
        const uint32_t ticket = static_cast<uint32_t>(sig) + aivIdx * AIV_FLAG_OFFSET;
        AscendC::CrossCoreSetFlag<4, PIPE>(ticket);
    }

    template <auto PIPE>
    __aicore__ static inline void NotifyBothAiv(TokenSignal sig) noexcept {
        const uint32_t ticketBase = static_cast<uint32_t>(sig);
        AscendC::CrossCoreSetFlag<4, PIPE>(ticketBase);
        AscendC::CrossCoreSetFlag<4, PIPE>(ticketBase + AIV_FLAG_OFFSET);
    }

    template <auto PIPE>
    __aicore__ static inline void WaitFromCube(TokenSignal sig) noexcept {
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t ticket = static_cast<uint32_t>(sig) + subBlock * AIV_FLAG_OFFSET;
        AscendC::CrossCoreWaitFlag<4, PIPE>(ticket);
    }

    template <auto PIPE>
    __aicore__ static inline void NotifyCube(TokenSignal sig) noexcept {
        const uint32_t subBlock = AscendC::GetSubBlockIdx();
        const uint32_t ticket = static_cast<uint32_t>(sig) + subBlock * AIV_FLAG_OFFSET;
        AscendC::CrossCoreSetFlag<4, PIPE>(ticket);
    }

    template <auto PIPE>
    __aicore__ static inline void WaitFromBothAiv(TokenSignal sig) noexcept {
        const uint32_t ticketBase = static_cast<uint32_t>(sig);
        AscendC::CrossCoreWaitFlag<4, PIPE>(ticketBase);
        AscendC::CrossCoreWaitFlag<4, PIPE>(ticketBase + AIV_FLAG_OFFSET);
    }
#endif
};

} // namespace sage::orchestration

namespace sage {
inline namespace orchestration_forwarding {
using namespace ::sage::orchestration;
}
}
