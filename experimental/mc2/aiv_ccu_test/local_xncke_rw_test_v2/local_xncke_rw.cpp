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
 *        五个独立 kernel 入口：XN(Scalar) / CKE(Scalar) / Both(Scalar) / XN(DataCopyPad) / CKE(DataCopyPad)
 */

#include "kernel_operator.h"
#include "./kernel/local_xn_rw_kernel.h"
#include "./kernel/local_cke_rw_kernel.h"
#include "./kernel/local_xncke_rw_kernel.h"
#include "./kernel/local_xn_dcpad_kernel.h"
#include "./kernel/local_cke_dcpad_kernel.h"

using namespace AscendC;

// ==================== XN only (Scalar: WriteGmByPassDCache) ====================

extern "C" __global__ __aicore__ void Aivccu_COMM_XN(uint32_t mode, uint32_t enableWarmup,
                                                      uint32_t loopCount, uint64_t pattern,
                                                      GM_ADDR xnAddr, GM_ADDR timeAddr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    LocalXnRwImpl::LocalXnRw op;
    op.Init(mode, enableWarmup, loopCount, pattern, xnAddr, timeAddr);
    op.Process();
}

// ==================== CKE only (Scalar: WriteGmByPassDCache) ====================

extern "C" __global__ __aicore__ void Aivccu_COMM_CKE(uint32_t mode, uint32_t enableWarmup,
                                                       uint32_t loopCount, uint64_t pattern,
                                                       GM_ADDR ckeAddr, GM_ADDR timeAddr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    LocalCkeRwImpl::LocalCkeRw op;
    op.Init(mode, enableWarmup, loopCount, pattern, ckeAddr, timeAddr);
    op.Process();
}

// ==================== Both (XN + CKE, Scalar) ====================

extern "C" __global__ __aicore__ void Aivccu_COMM_BOTH(uint32_t mode, uint32_t enableWarmup,
                                                        uint32_t loopCount, uint64_t pattern,
                                                        GM_ADDR xnAddr, GM_ADDR ckeAddr, GM_ADDR timeAddr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    LocalXnCkeRwImpl::LocalXnCkeRw op;
    op.Init(mode, enableWarmup, loopCount, pattern, xnAddr, ckeAddr, timeAddr);
    op.Process();
}

// ==================== XN only (DataCopyPad: DMA通路对比) ====================

extern "C" __global__ __aicore__ void Aivccu_COMM_XN_DCPAD(uint32_t mode, uint32_t enableWarmup,
                                                             uint32_t loopCount, uint64_t pattern,
                                                             GM_ADDR xnAddr, GM_ADDR timeAddr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    LocalXnDcPadImpl::LocalXnDcPad op;
    op.Init(mode, enableWarmup, loopCount, pattern, xnAddr, timeAddr, &pipe);
    op.Process();
}

// ==================== CKE only (DataCopyPad: DMA通路对比) ====================

extern "C" __global__ __aicore__ void Aivccu_COMM_CKE_DCPAD(uint32_t mode, uint32_t enableWarmup,
                                                              uint32_t loopCount, uint64_t pattern,
                                                              GM_ADDR ckeAddr, GM_ADDR timeAddr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    LocalCkeDcPadImpl::LocalCkeDcPad op;
    op.Init(mode, enableWarmup, loopCount, pattern, ckeAddr, timeAddr, &pipe);
    op.Process();
}

// ==================== Host 侧统一封装 ====================

void local_xncke_rw_demo(uint32_t mode, uint32_t resKind, uint32_t enableWarmup, uint32_t loopCount,
                         uint64_t pattern, uint32_t blockDim, void *stream,
                         uint8_t *xnAddr, uint8_t *ckeAddr, uint8_t *timeAddr)
{
    if (resKind == 0) {
        Aivccu_COMM_XN<<<blockDim, nullptr, stream>>>(mode, enableWarmup, loopCount, pattern, xnAddr, timeAddr);
    } else if (resKind == 1) {
        Aivccu_COMM_CKE<<<blockDim, nullptr, stream>>>(mode, enableWarmup, loopCount, pattern, ckeAddr, timeAddr);
    } else if (resKind == 3) {
        Aivccu_COMM_XN_DCPAD<<<blockDim, nullptr, stream>>>(mode, enableWarmup, loopCount, pattern, xnAddr, timeAddr);
    } else if (resKind == 4) {
        Aivccu_COMM_CKE_DCPAD<<<blockDim, nullptr, stream>>>(mode, enableWarmup, loopCount, pattern, ckeAddr, timeAddr);
    } else {
        Aivccu_COMM_BOTH<<<blockDim, nullptr, stream>>>(mode, enableWarmup, loopCount, pattern, xnAddr, ckeAddr, timeAddr);
    }
}
