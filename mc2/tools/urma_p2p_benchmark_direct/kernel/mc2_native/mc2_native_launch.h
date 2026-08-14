#ifndef MC2_NATIVE_LAUNCH_H
#define MC2_NATIVE_LAUNCH_H

#include <cstdint>
#include "../common/urma_p2p_bench_tiling.h"

void mc2_native_bench_launch(uint32_t blockDim, void *stream,
                             uint64_t hcclContextGM, uint64_t workspace,
                             uint64_t timingGm, UrmaP2pBenchTiling *tiling);

#endif
