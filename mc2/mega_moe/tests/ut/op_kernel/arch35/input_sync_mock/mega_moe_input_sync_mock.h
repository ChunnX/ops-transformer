// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

// CPU protocol test only. Atomic accesses model scalar bypass; WaitFlag commits
// a pending quant payload. This does not model the NPU interconnect or pipelines.
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <thread>

#define __aicore__
#define __gm__
using GM_ADDR = uint8_t *;

namespace AscendC {
enum CoreType {
    AIC,
    AIV
};
enum class HardEvent {
    MTE3_S
};
inline thread_local int32_t *ownedSequence = nullptr;
inline thread_local std::atomic<uint32_t> *pendingPayload = nullptr;
inline thread_local uint32_t pendingVersion = 0;
inline thread_local bool eventSet = false;
inline std::atomic<int32_t *> watchedAddress{nullptr};
inline std::atomic<bool> watchedRead{false};

inline int32_t ReadGmByPassDCache(int32_t *address)
{
    if (address == watchedAddress.load()) {
        watchedRead.store(true);
    }
    return std::atomic_ref<int32_t>(*address).load();
}

inline void WriteGmByPassDCache(int32_t *address, int32_t value)
{
    assert(address == ownedSequence);  // one writer per 64B slot
    assert(pendingPayload == nullptr); // ready cannot overtake the quant payload
    std::atomic_ref<int32_t>(*address).store(value);
}

template <HardEvent>
inline void SetFlag(uint32_t)
{
    eventSet = true;
}
template <HardEvent>
inline void WaitFlag(uint32_t)
{
    assert(eventSet);
    eventSet = false;
    if (pendingPayload != nullptr) {
        pendingPayload->store(pendingVersion);
        pendingPayload = nullptr;
    }
}

inline int64_t GetSystemCycle()
{
    std::this_thread::yield();
    return std::chrono::steady_clock::now().time_since_epoch().count();
}
} // namespace AscendC

#ifdef TEST_AIC
constexpr auto g_coreType = AscendC::AIC;
#else
constexpr auto g_coreType = AscendC::AIV;
#endif
