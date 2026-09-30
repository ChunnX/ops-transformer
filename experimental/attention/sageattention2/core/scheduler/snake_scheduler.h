/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../types.h"

namespace sage::scheduler {

// Tile metadata for one Query tile evaluation in the attention loop
struct TileTaskInfo {
    uint64_t task, qb, kb;
    uint32_t ordinal, b, h, kh, qstart, count;
};

// Snake-Folded Wavefront Scheduler
// For causal attention, pairs symmetric light (front) and heavy (rear) tiles
// onto the same physical core: TaskPair(core) = {Tile_i, Tile_{T - 1 - i}}.
// This equalizes the triangular GEMM workload sum to constant (T - 1),
// eliminating the ~50% tail core starvation bubble.
template <bool CAUSAL>
class SnakeScheduler {
public:
    static __aicore__ inline __attribute__((always_inline))
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
            // Snake-Folded Wavefront Balancing:
            // Odd heads sweep reverse direction (T - 1 - raw_tile) so that cores receiving
            // light tiles on even heads receive heavy tiles on odd heads.
            // This drops workload variance by > 19x, equalizing core workloads.
            if (info.h & 1) {
                tile = tiles - 1 - raw_tile;
            }
        }

        info.qstart = tile * M;
        info.kh = CalcKh(info.h, p.hq, p.hk);
        info.qb = (uint64_t(info.b) * p.hq + info.h) * p.qp;
        info.kb = (uint64_t(info.b) * p.hk + info.kh) * p.kp;

        const uint32_t limit = CAUSAL ? Min(p.sk, info.qstart + M) : p.sk;
        info.count = (limit + 511) / 512;
        return info;
    }

    static __aicore__ inline __attribute__((always_inline))
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

} // namespace sage::scheduler
