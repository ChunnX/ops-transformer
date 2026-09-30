#pragma once

#include <cstdint>
#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"
#include "split_kv_grid.h"

namespace sage::dispatch {

// Tile Task Information for Attention Execution
// `qb`/`kb` are already head-resolved element offsets, so the GQA key head is a
// scheduling-time value only and is not carried in the task.
struct TileTaskInfo {
    uint64_t task{0ULL};
    uint64_t qb{0ULL};
    uint64_t kb{0ULL};
    uint32_t ordinal{0U};
    uint32_t b{0U};
    uint32_t h{0U};
    uint32_t qstart{0U};
    uint32_t count{0U};

    SAGE_DEVICE_FUNC constexpr TileTaskInfo() noexcept = default;

    SAGE_DEVICE_FUNC constexpr TileTaskInfo(
        uint64_t t, uint64_t q, uint64_t k, uint32_t ord,
        uint32_t batch, uint32_t head,
        uint32_t qs, uint32_t cnt) noexcept
        : task(t), qb(q), kb(k), ordinal(ord),
          b(batch), h(head), qstart(qs), count(cnt) {}

    SAGE_DEVICE_FUNC constexpr uint64_t ComputeGroupKb(uint32_t groupIndex) const noexcept {
        return kb + static_cast<uint64_t>(groupIndex) * 512ULL;
    }
};

// Snake-Folded Wavefront Scheduler
// Alternates traversal direction across consecutive attention heads
// to balance triangular causal workloads across all execution cores.
template <bool CAUSAL>
class SnakeScheduler {
public:
    static SAGE_DEVICE_FUNC __attribute__((always_inline))
    TileTaskInfo Decode(const SageParams &p, uint64_t task, uint32_t ordinal) {
        const uint32_t tiles = (p.sq + sage::core::M - 1U) / sage::core::M;
        const uint64_t totalTiles = static_cast<uint64_t>(p.b) * p.hq * tiles;
        const uint64_t virtualTask = UseDieGroups(p) ? DieTask(task, totalTiles, p.cores) : task;

        const uint32_t tileIndexInHead = static_cast<uint32_t>(virtualTask % tiles);
        const uint64_t headLinear = virtualTask / tiles;
        const uint32_t headIdx = static_cast<uint32_t>(headLinear % p.hq);
        const uint32_t batchIdx = static_cast<uint32_t>(headLinear / p.hq);

        const uint32_t scheduledTile = (CAUSAL && ((headIdx & 1U) != 0U))
            ? ((tiles - 1U) - tileIndexInHead)
            : tileIndexInHead;

        const uint32_t qStartOffset = scheduledTile * sage::core::M;
        const uint32_t keyHeadIdx = CalcKh(headIdx, p.hq, p.hk);
        const uint64_t queryBase = (static_cast<uint64_t>(batchIdx) * p.hq + headIdx) * p.qp;
        const uint64_t keyBase = (static_cast<uint64_t>(batchIdx) * p.hk + keyHeadIdx) * p.kp;

        const uint32_t maxReach = CAUSAL ? (qStartOffset + sage::core::M) : p.sk;
        const uint32_t effectiveLimit = (CAUSAL && maxReach < p.sk) ? maxReach : p.sk;
        const uint32_t groupCount = (effectiveLimit + 511U) / 512U;

        return TileTaskInfo{
            task, queryBase, keyBase, ordinal,
            batchIdx, headIdx,
            qStartOffset, groupCount
        };
    }

    static SAGE_DEVICE_FUNC __attribute__((always_inline))
    uint64_t LookaheadScale(const SageParams &p, TileTaskInfo info, uint32_t group, uint64_t totalTasks) {
        for (TileTaskInfo current = info; current.task < totalTasks; ) {
            if (group < current.count) {
                return current.ComputeGroupKb(group);
            }
            group -= current.count;
            const uint64_t nextTask = current.task + static_cast<uint64_t>(p.cores);
            if (nextTask >= totalTasks) {
                break;
            }
            current = Decode(p, nextTask, current.ordinal + 1U);
        }
        return ~0ULL;
    }
};

} // namespace sage::dispatch
