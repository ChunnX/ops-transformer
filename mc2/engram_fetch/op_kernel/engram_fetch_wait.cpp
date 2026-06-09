/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * EngramFetchWait kernel implementation (极简骨架版)
 */

#include "kernel_operator.h"
#include "engram_fetch_tiling.h"

using namespace AscendC;

class EngramFetchWaitKernel {
public:
    __aicore__ EngramFetchWaitKernel()
    {
    }

    __aicore__ void Init(GM_ADDR handle, GM_ADDR tiling)
    {
        handleGM = handle;
        tilingGM = tiling;
    }

    __aicore__ void Process()
    {
        // 骨架实现：直接返回
        // 实际实现需要在后续迭代中添加 wait 逻辑
    }

private:
    GM_ADDR handleGM;
    GM_ADDR tilingGM;
};

extern "C" __global__ __aicore__ void engram_fetch_wait(GM_ADDR handle, GM_ADDR workspace, GM_ADDR tiling)
{
    EngramFetchWaitKernel op;
    op.Init(handle, tiling);
    op.Process();
}