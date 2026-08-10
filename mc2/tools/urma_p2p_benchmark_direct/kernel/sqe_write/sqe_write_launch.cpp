#include "kernel_operator.h"
#include "../common/urma_p2p_bench_tiling.h"
#include "sqe_write_kernel.h"
#include "sqe_write_launch.h"

using namespace AscendC;
using namespace UrmaP2pBench;

extern "C" __global__ __aicore__ void sqe_write_bench_kernel(
    GM_ADDR channelPtrTable, GM_ADDR stateGm, GM_ADDR timingGm,
    uint32_t peer, uint32_t numSqe, uint64_t sqeBytes, uint64_t bufBytes,
    uint32_t dbInterval, uint32_t channelSlot, uint32_t cqOwnerSlot, uint32_t stateStride)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    UrmaP2pBenchTiling tiling;
    tiling.peer = peer;
    tiling.numSqe = numSqe;
    tiling.sqeBytes = sqeBytes;
    tiling.bufBytes = bufBytes;
    tiling.dbInterval = dbInterval;
    tiling.channelSlot = channelSlot;
    tiling.cqOwnerSlot = cqOwnerSlot;
    tiling.stateStride = stateStride;
    SqeWriteBenchOp op;
    op.Init(channelPtrTable, stateGm, timingGm, &pipe, &tiling);
    op.Process();
}

void sqe_write_bench_launch(uint32_t blockDim, void *stream,
                            uint64_t channelPtrTable, uint64_t stateGm,
                            uint64_t timingGm, UrmaP2pBenchTiling *tiling)
{
    sqe_write_bench_kernel<<<blockDim, nullptr, stream>>>(
        channelPtrTable, stateGm, timingGm,
        tiling->peer, tiling->numSqe, tiling->sqeBytes, tiling->bufBytes,
        tiling->dbInterval, tiling->channelSlot, tiling->cqOwnerSlot, tiling->stateStride);
}
