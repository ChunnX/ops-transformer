#ifndef URMA_P2P_BENCH_H
#define URMA_P2P_BENCH_H

#include <cstdint>
#include "urma_p2p_bench_tiling.h"

void sqe_write_bench_launch(uint32_t blockDim, void *stream,
                            uint64_t channelPtrTable, uint64_t stateGm,
                            uint64_t timingGm, UrmaP2pBenchTiling *tiling);

void sqe_write_with_notify_bench_launch(uint32_t blockDim, void *stream,
                                        uint64_t channelPtrTable, uint64_t stateGm,
                                        uint64_t timingGm, UrmaP2pBenchTiling *tiling);

void hcomm_bench_launch(uint32_t blockDim, void *stream,
                        uint64_t channelPtrTable, uint64_t stateGm,
                        uint64_t timingGm, UrmaP2pBenchTiling *tiling);

#endif
