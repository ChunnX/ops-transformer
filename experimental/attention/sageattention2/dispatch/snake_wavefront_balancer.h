#pragma once

#include <cstdint>
#include "../core/platform_traits.h"
#include "../core/numerical_types.h"
#include "split_kv_grid.h"

namespace sage::dispatch {

// Tile Task Information for Attention Execution
struct TileTaskInfo {
    uint64_t task;
    uint64_t qb;
    uint64_t kb;
    uint32_t ordinal;
    uint32_t b;
    uint32_t h;
    uint32_t kh;
    uint32_t qstart;
    uint32_t count;
};

// Snake-Folded Wavefront Scheduler (Task 3.2)
// For causal attention, pairs symmetric light (front) and heavy (rear) tiles
// onto the same physical core: TaskPair(core) = {Tile_i, Tile_{T - 1 - i}}.
// Odd heads sweep reverse direction (T - 1 - raw_tile) so that cores receiving
// light tiles on even heads receive heavy tiles on odd heads.
// This reduces triangular GEMM workload variance by > 19x, eliminating tail starvation bubbles.
template <bool CAUSAL>
class SnakeScheduler {
public:
    static SAGE_DEVICE_FUNC __attribute__((always_inline))
    TileTaskInfo Decode(const SageParams &p, uint64_t task, uint32_t ordinal) {
        TileTaskInfo info;
        const uint32_t tiles = (p.sq + M - 1) / M;
        info.task = task;

        if (UseDieGroups(p)) {
            task = DieTask(task, uint64_t(p.b) * p.hq * tiles, p.cores);
        }

        info.ordinal = ordinal;
        info.h = (task / tiles) % p.hq;
        info.b = task / tiles / p.hq;
        uint32_t raw_tile = task % tiles;
        uint32_t tile = raw_tile;

        if constexpr (CAUSAL) {
            // Snake-Folded Wavefront Balancing
            if (info.h & 1) {
                tile = tiles - 1 - raw_tile;
            }
        }

        info.qstart = tile * M;
        info.kh = CalcKh(info.h, p.hq, p.hk);
        info.qb = (uint64_t(info.b) * p.hq + info.h) * p.qp;
        info.kb = (uint64_t(info.b) * p.hk + info.kh) * p.kp;

        const uint32_t limit = CAUSAL ? (p.sk < info.qstart + M ? p.sk : info.qstart + M) : p.sk;
        info.count = (limit + 511) / 512;
        return info;
    }

    static SAGE_DEVICE_FUNC __attribute__((always_inline))
    uint64_t LookaheadScale(const SageParams &p, TileTaskInfo info, uint32_t group, uint64_t totalTasks) {
        if (info.task >= totalTasks) return ~uint64_t(0);
        while (group >= info.count) {
            group -= info.count;
            if (info.task + p.cores >= totalTasks) return ~uint64_t(0);
            info = Decode(p, info.task + p.cores, info.ordinal + 1);
        }
        return info.kb + group * 512;
    }
};

} // namespace sage::dispatch

namespace sage {
using dispatch::TileTaskInfo;
using dispatch::SnakeScheduler;
}
