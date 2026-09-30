#pragma once

#include <cstdint>
#include "../core/platform_traits.h"
#include "../core/numerical_types.h"

namespace sage::dispatch {

// Multi-Head & GQA Head Group Topology Mapping

SAGE_DEVICE_FUNC constexpr uint32_t CalcKh(uint32_t h, uint32_t hq, uint32_t hk) noexcept {
    if (hq == hk) return h;
    const uint32_t ratio = hq / hk;
    return (ratio == 4) ? (h >> 2) :
           (ratio == 8) ? (h >> 3) :
           (ratio == 2) ? (h >> 1) :
           (ratio == 16) ? (h >> 4) : (h / ratio);
}

// Dual-Die Topology Symmetry & Core Task Partitioning

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

// Split-KV 2D Spatial Grid Partitioning (Task 3.1)

// Coordinates 2D grid partitioning: Grid(Q_Tiles, KV_Chunks)
struct Grid2DTask {
    uint32_t batch;
    uint32_t headQ;
    uint32_t headK;
    uint32_t qTile;
    uint32_t kvChunk;
    uint32_t totalKvChunks;
    uint32_t qStart;
    uint32_t kvStart;
    uint32_t kvLength;
};

class Grid2DCoordinator {
public:
    static SAGE_DEVICE_FUNC Grid2DTask Decompose(
        const SageParams &p, uint32_t taskIdx, uint32_t totalCores, uint32_t chunkSize = 512) noexcept
    {
        Grid2DTask task{};
        const uint32_t qTiles = (p.sq + M - 1) / M;
        const uint32_t kvChunks = (p.sk + chunkSize - 1) / chunkSize;
        const uint32_t total2DTasks = p.b * p.hq * qTiles * kvChunks;

        uint32_t mappedTask = taskIdx;
        if (mappedTask >= total2DTasks) return task;

        task.totalKvChunks = kvChunks;
        task.kvChunk = mappedTask % kvChunks;
        uint32_t remaining = mappedTask / kvChunks;

        task.qTile = remaining % qTiles;
        remaining /= qTiles;

        task.headQ = remaining % p.hq;
        task.batch = remaining / p.hq;
        task.headK = CalcKh(task.headQ, p.hq, p.hk);

        task.qStart = task.qTile * M;
        task.kvStart = task.kvChunk * chunkSize;
        task.kvLength = (task.kvStart + chunkSize <= p.sk) ? chunkSize : (p.sk - task.kvStart);

        return task;
    }
};

} // namespace sage::dispatch

namespace sage {
using dispatch::CalcKh;
using dispatch::UseDieGroups;
using dispatch::DieTask;
using dispatch::DieTaskEnd;
using dispatch::Grid2DTask;
using dispatch::Grid2DCoordinator;
}
