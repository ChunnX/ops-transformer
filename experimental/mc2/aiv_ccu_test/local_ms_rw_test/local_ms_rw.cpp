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
 * \file local_ms_rw.cpp
 * \brief 本地读写 MS Kernel 入口 + <<<>>> 封装
 */

#include <cstdio>
#include <cstring>
#include "kernel_operator.h"
#include "./kernel/local_ms_rw_kernel.h"

using namespace AscendC;
using namespace LocalMsRwImpl;

__attribute__((always_inline)) __aicore__ __inline__ void
local_ms_rw_kernel_impl(uint32_t mode0, uint32_t enableWarmup, uint32_t msId, uint32_t loopCount,
                        uint64_t dataSize,
                        GM_ADDR msAddr0, GM_ADDR msAddr1, GM_ADDR msAddr2, GM_ADDR msAddr3,
                        GM_ADDR timeAddr, uint32_t blockDim)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    GM_ADDR msAddrs[4] = {msAddr0, msAddr1, msAddr2, msAddr3};

    LocalMsRw<uint8_t> op;
    op.Init(mode0, enableWarmup, msId, loopCount, dataSize, msAddrs, timeAddr, blockDim, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void Aivccu_COMM(uint32_t mode0, uint32_t enableWarmup,
                                                    uint32_t msId, uint32_t loopCount, uint64_t dataSize,
                                                    GM_ADDR msAddr0, GM_ADDR msAddr1, GM_ADDR msAddr2,
                                                    GM_ADDR msAddr3, GM_ADDR timeAddr, uint32_t blockDim)
{
    local_ms_rw_kernel_impl(mode0, enableWarmup, msId, loopCount, dataSize, msAddr0, msAddr1, msAddr2,
                            msAddr3, timeAddr, blockDim);
}

void local_ms_rw_demo(uint32_t mode0, uint32_t enableWarmup, uint32_t msId, uint32_t loopCount,
                      uint64_t dataSize, uint32_t blockDim, void *stream, uint8_t *msAddr0, uint8_t *msAddr1,
                      uint8_t *msAddr2, uint8_t *msAddr3, uint8_t *timeAddr)
{
    Aivccu_COMM<<<blockDim, nullptr, stream>>>(mode0, enableWarmup, msId, loopCount, dataSize, msAddr0,
                                                msAddr1, msAddr2, msAddr3, timeAddr, blockDim);
}
