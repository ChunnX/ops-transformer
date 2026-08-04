/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file local_xncke_rw.cpp
 * \brief 本地读写 XN(64bit)/CKE(16bit) Kernel 入口 + <<<>>> 封装
 */

#include "kernel_operator.h"
#include "./kernel/local_xncke_rw_kernel.h"

using namespace AscendC;
using namespace LocalXnCkeRwImpl;

__attribute__((always_inline)) __aicore__ __inline__ void
local_xncke_rw_kernel_impl(uint32_t mode, uint32_t resKind, uint32_t enableWarmup, uint32_t loopCount,
                           uint64_t pattern, GM_ADDR xnAddr, GM_ADDR ckeAddr, GM_ADDR timeAddr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    LocalXnCkeRw op;
    op.Init(mode, resKind, enableWarmup, loopCount, pattern, xnAddr, ckeAddr, timeAddr, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void Aivccu_COMM(uint32_t mode, uint32_t resKind, uint32_t enableWarmup,
                                                   uint32_t loopCount, uint64_t pattern, GM_ADDR xnAddr,
                                                   GM_ADDR ckeAddr, GM_ADDR timeAddr)
{
    local_xncke_rw_kernel_impl(mode, resKind, enableWarmup, loopCount, pattern, xnAddr, ckeAddr, timeAddr);
}

void local_xncke_rw_demo(uint32_t mode, uint32_t resKind, uint32_t enableWarmup, uint32_t loopCount,
                         uint64_t pattern, uint32_t blockDim, void *stream, uint8_t *xnAddr, uint8_t *ckeAddr,
                         uint8_t *timeAddr)
{
    Aivccu_COMM<<<blockDim, nullptr, stream>>>(mode, resKind, enableWarmup, loopCount, pattern, xnAddr, ckeAddr,
                                                timeAddr);
}
