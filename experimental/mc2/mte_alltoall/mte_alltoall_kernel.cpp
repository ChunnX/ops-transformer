/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */

// ============================================================================
// MTE AllToAll 算子 - Kernel 入口（<<<>>> 调用）
// ============================================================================

#include <cstdio>
#include <cstring>
#include "kernel_operator.h"
#include "./kernel/mte_alltoall_tiling_data.h"
#include "./kernel/mte_alltoall_kernel.h"
#include "./mte_alltoall_kernel_host.h"

using namespace AscendC;
using namespace MteAllToAllImpl;

template<typename DTYPE_X>
__attribute__((always_inline)) __aicore__ __inline__ void MteAllToAllWithMc2Context(
    GM_ADDR x, GM_ADDR output, GM_ADDR workspaceGM,
    GM_ADDR mc2Context, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    MteAllToAll<DTYPE_X> op;
    op.InitWithMc2Context(x, output, mc2Context, tilingGM, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void MteAllToAllGeneric(
    GM_ADDR x, GM_ADDR output,
    GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingGM)
{
    MteAllToAllWithMc2Context<int8_t>(
                x, output, workspaceGM, mc2Context, tilingGM);
}

// Host 端调用入口，<<<>>> 语法启动 Kernel
void MteAllToAllDemo(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* output,
                         uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM) {
    MteAllToAllGeneric<<<blockDim, nullptr, stream>>>(
        x, output, workspaceGM, mc2Context, tilingGM
    );
}
