#ifndef URMA_P2P_BENCH_TILING_H
#define URMA_P2P_BENCH_TILING_H

#include <cstdint>

constexpr uint32_t TIMING_SLOTS = 8;
constexpr uint32_t COMPLETION_SQE_BYTES = 32U;

struct UrmaP2pBenchTiling {
    uint32_t rankId;
    uint32_t peer;
    uint32_t numSqe;
    uint64_t sqeBytes;
    uint64_t bufBytes;
    uint32_t dbInterval;
    uint32_t channelSlot;
    uint32_t cqOwnerSlot;
    uint32_t stateStride;
    uint32_t warmupCount;
    uint32_t loopCount;
};

#endif
