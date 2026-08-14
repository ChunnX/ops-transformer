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
 * \file spr_rw.cpp
 * \brief SPR 读写 Kernel 入口 + <<<>>> 封装
 *
 * 实现 background.md 描述的 SPR→vaddr 映射访问例子。
 * Host 侧通过 halResMap 映射所有核的 SPR vaddr 并传入 GM，
 * Kernel 侧通过 st_dev / get_data_main_base() 访问 SPR 寄存器。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "./kernel/spr_rw_kernel.h"

using namespace AscendC;
using namespace SprRwImpl;

__attribute__((always_inline)) __aicore__ __inline__ void
spr_rw_kernel_impl(uint64_t sprVaddr, GM_ADDR sprVaddrTable, GM_ADDR resultAddr,
                   uint32_t targetCoreId, uint32_t blockDim)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    SprRw<uint8_t> op;
    op.Init(sprVaddr, sprVaddrTable, resultAddr, targetCoreId, blockDim);
    op.Process();
}

extern "C" __global__ __aicore__ void Aivccu_COMM(uint64_t sprVaddr, GM_ADDR sprVaddrTable,
                                                   GM_ADDR resultAddr, uint32_t targetCoreId,
                                                   uint32_t blockDim)
{
    spr_rw_kernel_impl(sprVaddr, sprVaddrTable, resultAddr, targetCoreId, blockDim);
}

void spr_rw_demo(uint64_t sprVaddr, uint32_t blockDim, void *stream,
                 uint8_t *sprVaddrTable, uint8_t *resultAddr, uint32_t targetCoreId)
{
    Aivccu_COMM<<<blockDim, nullptr, stream>>>(sprVaddr, sprVaddrTable, resultAddr,
                                                targetCoreId, blockDim);
}
