#include "kernel_operator.h"
#include "urma_p2p_bench_tiling.h"
#include "sqe_write_kernel.h"
#include "sqe_write_with_notify_kernel.h"
#include "hcomm_kernel.h"

using namespace AscendC;
using namespace UrmaP2pBench;

// ---- SQE Write (64B, opcode 0x3, comm_samples style) ----
extern "C" __global__ __aicore__ void sqe_write_bench_kernel(
    GM_ADDR channelPtrTable, GM_ADDR stateGm, GM_ADDR timingGm,
    __gm__ UrmaP2pBenchTiling *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SqeWriteBenchOp op;
    op.Init(channelPtrTable, stateGm, timingGm, &pipe, tiling);
    op.Process();
}

void sqe_write_bench_launch(uint32_t blockDim, void *stream,
                            GM_ADDR channelPtrTable, GM_ADDR stateGm,
                            GM_ADDR timingGm, UrmaP2pBenchTiling *tiling)
{
    sqe_write_bench_kernel<<<blockDim, nullptr, stream>>>(
        channelPtrTable, stateGm, timingGm, tiling);
}

// ---- SQE WriteWithNotify (96B, opcode 0x5, mc2 style) ----
extern "C" __global__ __aicore__ void sqe_write_with_notify_bench_kernel(
    GM_ADDR channelPtrTable, GM_ADDR stateGm, GM_ADDR timingGm,
    __gm__ UrmaP2pBenchTiling *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SqeWriteWithNotifyBenchOp op;
    op.Init(channelPtrTable, stateGm, timingGm, &pipe, tiling);
    op.Process();
}

void sqe_write_with_notify_bench_launch(uint32_t blockDim, void *stream,
                                        GM_ADDR channelPtrTable, GM_ADDR stateGm,
                                        GM_ADDR timingGm, UrmaP2pBenchTiling *tiling)
{
    sqe_write_with_notify_bench_kernel<<<blockDim, nullptr, stream>>>(
        channelPtrTable, stateGm, timingGm, tiling);
}

// ---- Hcomm (WriteNbi + Drain) ----
extern "C" __global__ __aicore__ void hcomm_bench_kernel(
    GM_ADDR channelPtrTable, GM_ADDR stateGm, GM_ADDR timingGm,
    __gm__ UrmaP2pBenchTiling *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    HcommBenchOp op;
    op.Init(channelPtrTable, stateGm, timingGm, &pipe, tiling);
    op.Process();
}

void hcomm_bench_launch(uint32_t blockDim, void *stream,
                        GM_ADDR channelPtrTable, GM_ADDR stateGm,
                        GM_ADDR timingGm, UrmaP2pBenchTiling *tiling)
{
    hcomm_bench_kernel<<<blockDim, nullptr, stream>>>(
        channelPtrTable, stateGm, timingGm, tiling);
}
