#pragma once

#include <cstdint>
#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"

namespace sage::dispatch {

// GQA head group mapping: maps query head index h in [0, hq) to key/value head kh in [0, hk)
SAGE_DEVICE_FUNC constexpr uint32_t CalcKh(uint32_t h, uint32_t hq, uint32_t hk) noexcept {
    if (hq == hk) {
        return h;
    }
    const uint32_t groupSpan = hq / hk;
    switch (groupSpan) {
        case 2U:  return h >> 1U;
        case 4U:  return h >> 2U;
        case 8U:  return h >> 3U;
        case 16U: return h >> 4U;
        default:  return h / groupSpan;
    }
}

// Topology coordinator for dual-die Atlas 950PR architectures
struct DualDieTopology {
    static SAGE_DEVICE_FUNC constexpr bool IsEnabled(const SageParams &p) noexcept {
        const bool coreValid = (p.cores > 0U) && ((p.cores & 1U) == 0U);
        const bool headAligned = ((static_cast<uint64_t>(p.b) * p.hk) & 1ULL) == 0ULL;
        const bool seqAligned = (p.sq % sage::core::M) == 0U;
        return coreValid && headAligned && seqAligned;
    }

    static SAGE_DEVICE_FUNC constexpr uint64_t MapTask(uint64_t task, uint64_t totalTasks, uint32_t totalCores) noexcept {
        const uint32_t dieCores = totalCores >> 1U;
        const uint64_t halfTotal = totalTasks >> 1U;
        const uint32_t cid = static_cast<uint32_t>(task % totalCores);
        const uint64_t wave = task / totalCores;
        const uint32_t dieIdx = cid / dieCores;
        const uint32_t localRank = cid % dieCores;
        return static_cast<uint64_t>(dieIdx) * halfTotal + wave * dieCores + localRank;
    }

    static SAGE_DEVICE_FUNC constexpr uint64_t ComputeCoreTaskLimit(const SageParams &p, uint64_t totalTasks, uint32_t coreId) noexcept {
        if (!IsEnabled(p)) {
            return totalTasks;
        }
        const uint32_t dieCores = p.cores >> 1U;
        const uint32_t rankInDie = coreId % dieCores;
        const uint64_t dieWorkload = totalTasks >> 1U;
        if (dieWorkload <= rankInDie) {
            return coreId;
        }
        const uint64_t wavesAssigned = (dieWorkload - rankInDie + dieCores - 1U) / dieCores;
        return coreId + wavesAssigned * p.cores;
    }
};

SAGE_DEVICE_FUNC constexpr bool UseDieGroups(const SageParams &p) noexcept {
    return DualDieTopology::IsEnabled(p);
}

SAGE_DEVICE_FUNC constexpr uint64_t DieTask(uint64_t task, uint64_t total, uint32_t cores) noexcept {
    return DualDieTopology::MapTask(task, total, cores);
}

SAGE_DEVICE_FUNC constexpr uint64_t DieTaskEnd(const SageParams &p, uint64_t total, uint32_t core) noexcept {
    return DualDieTopology::ComputeCoreTaskLimit(p, total, core);
}

} // namespace sage::dispatch
