/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

#define MTE_ONE_SHOT 1

template<uint32_t commMode, typename DTYPE_X>
__attribute__((always_inline)) __aicore__ __inline__ void mte_all_gather_with_mc2context(GM_ADDR x, GM_ADDR output, GM_ADDR workspaceGM,
                                                        GM_ADDR mc2Context, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    MteAllGather<DTYPE_X> op;
    op.InitWithMc2Context(x, output, mc2Context, tilingGM, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void MteAllGather_Generic(
    GM_ADDR x, GM_ADDR output,
    GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingGM)
{
    mte_all_gather_with_mc2context<MTE_ONE_SHOT, int8_t>(
                x, output, workspaceGM, mc2Context, tilingGM);
}

void mte_all_gather_demo(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* output,
                           uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM) {
    MteAllGather_Generic<<<blockDim, nullptr, stream>>>(
        x, output, workspaceGM, mc2Context, tilingGM
    );
}
