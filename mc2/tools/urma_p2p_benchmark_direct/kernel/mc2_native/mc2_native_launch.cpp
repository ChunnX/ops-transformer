#include "kernel_operator.h"
#include "../common/urma_p2p_bench_tiling.h"
#include "mc2_native_kernel.h"
#include "mc2_native_launch.h"

using namespace AscendC;
using namespace UrmaP2pBench;

extern "C" __global__ __aicore__ void mc2_native_bench_kernel(
    GM_ADDR hcclContextGM, GM_ADDR workspace, GM_ADDR timingGm,
    uint32_t peer, uint32_t numSqe, uint64_t sqeBytes, uint64_t bufBytes,
    uint32_t dbInterval, uint32_t channelSlot, uint32_t cqOwnerSlot, uint32_t stateStride,
    uint32_t loopCount, uint32_t benchMode)
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
    tiling.loopCount = loopCount;
    tiling.benchMode = benchMode;
    Mc2NativeBenchOp op;
    op.Init(hcclContextGM, workspace, timingGm, &pipe, &tiling);
    op.Process();
}

void mc2_native_bench_launch(uint32_t blockDim, void *stream,
                             uint64_t hcclContextGM, uint64_t workspace,
                             uint64_t timingGm, UrmaP2pBenchTiling *tiling)
{
    mc2_native_bench_kernel<<<blockDim, nullptr, stream>>>(
        (void*)hcclContextGM, (void*)workspace, (void*)timingGm,
        tiling->peer, tiling->numSqe, tiling->sqeBytes, tiling->bufBytes,
        tiling->dbInterval, tiling->channelSlot, tiling->cqOwnerSlot, tiling->stateStride,
        tiling->loopCount, tiling->benchMode);
}
