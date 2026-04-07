/*!
 * \file mte_all_gather.cpp
 * \brief AllGather核函数入口，通过MTE通信拿到其它卡数据，按rank顺序拼接到output
 */

#include <cstdio>
#include <cstring>
#include "kernel_operator.h"
#include "./kernel/mte_all_gather_tiling_data.h"
#include "./kernel/mte_all_gather.h"

using namespace AscendC;
using namespace MteAllGatherImpl;

template<typename DTYPE_X>
__attribute__((always_inline)) __aicore__ __inline__ void MteAllGatherDemoWitchMc2context(GM_ADDR x, GM_ADDR output, GM_ADDR workspaceGM,
                                                        GM_ADDR mc2Context, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    MteAllGather<DTYPE_X> op;
    op.InitWithMc2Context(x, output, mc2Context, tilingGM, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void MteAllGatherGeneric(
    GM_ADDR x, GM_ADDR output,
    GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingGM)
{
    MteAllGatherDemoWitchMc2context<int8_t>(
                x, output, workspaceGM, mc2Context, tilingGM);
}

void MteAllGatherDemo(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* output,
                           uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM) {
    MteAllGatherGeneric<<<blockDim, nullptr, stream>>>(
        x, output, workspaceGM, mc2Context, tilingGM
    );
}
