#ifndef SQE_WRITE_LAUNCH_H
#define SQE_WRITE_LAUNCH_H

#include <cstdint>
#include "../common/urma_p2p_bench_tiling.h"

void sqe_write_bench_launch(uint32_t blockDim, void *stream,
                            uint64_t channelPtrTable, uint64_t stateGm,
                            uint64_t timingGm, UrmaP2pBenchTiling *tiling);

#endif
