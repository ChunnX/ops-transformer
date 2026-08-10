#ifndef URMA_P2P_BENCH_TILING_H
#define URMA_P2P_BENCH_TILING_H

#include <cstdint>

constexpr uint32_t TIMING_SLOTS = 8;

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
};

#endif
