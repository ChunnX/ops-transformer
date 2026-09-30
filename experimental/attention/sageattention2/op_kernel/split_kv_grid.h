#pragma once

#include <cstdint>
#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"

namespace sage::dispatch {

SAGE_DEVICE_FUNC constexpr uint32_t CalcKh(uint32_t h, uint32_t hq, uint32_t hk) noexcept {
    if (hq == hk) return h;
    const uint32_t ratio = hq / hk;
    return (ratio == 4) ? (h >> 2) :
           (ratio == 8) ? (h >> 3) :
           (ratio == 2) ? (h >> 1) :
           (ratio == 16) ? (h >> 4) : (h / ratio);
}

SAGE_DEVICE_FUNC constexpr bool UseDieGroups(const SageParams &p) noexcept {
    return p.cores > 0 && p.cores % 2 == 0 && (uint64_t(p.b) * p.hk) % 2 == 0 &&
           p.sq % 128 == 0;
}

SAGE_DEVICE_FUNC constexpr uint64_t DieTask(uint64_t task, uint64_t total, uint32_t cores) noexcept {
    const uint32_t half = cores / 2;
    return (task % cores / half) * (total / 2) + (task / cores) * half + task % half;
}

SAGE_DEVICE_FUNC constexpr uint64_t DieTaskEnd(const SageParams &p, uint64_t total, uint32_t core) noexcept {
    if (!UseDieGroups(p)) return total;
    const uint32_t half = p.cores / 2, rank = core % half;
    const uint64_t local = total / 2;
    const uint64_t count = local > rank ? (local - rank + half - 1) / half : 0;
    return core + count * p.cores;
}

} // namespace sage::dispatch

namespace sage {
using dispatch::CalcKh;
using dispatch::UseDieGroups;
using dispatch::DieTask;
using dispatch::DieTaskEnd;
}
