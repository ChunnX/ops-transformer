#ifndef URMA_P2P_BENCH_TILING_H
#define URMA_P2P_BENCH_TILING_H

#include <cstdint>

constexpr uint32_t TIMING_SLOTS = 8;
constexpr uint32_t COMPLETION_SQE_BYTES = 32U;

// timing slot语义:
// [0]=assemble累计 [1]=issue累计 [2]=wait累计 [3]=total累计
// [4]=最后一轮assemble [5]=最后一轮issue [6]=最后一轮wait [7]=最后一轮total

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
    uint32_t loopCount;
};

#endif
