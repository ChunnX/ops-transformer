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
 * \file moe_distribute_dispatch_ccu.cpp
 * \brief dispatch-ccu device 侧：teardown AICore 核函数 + <<<>>> 启动包装。
 *        本文件由 ascendc 编译器编译（ascendc_library）。
 *        注：CCU setup kernel 是 hcomm::CcuKernel 子类、为 host 编译 C++，见 dispatch_setup_ccu.*。
 */
#include "kernel_operator.h"
#include "dispatch_ccu_common.h"
#include "moe_distribute_dispatch_ccu.h"

using namespace AscendC;

// teardown 核函数：等各 peer READY_FLAG → 从本卡 HBM DATA 区组装 expand_x → 回写 buffer[0]=1 ack。
extern "C" __global__ __aicore__ void dispatch_teardown_aicore(
    GM_ADDR localBuffer, GM_ADDR expandXOut, GM_ADDR tilingGM)
{
    // ===== TODO(P0 步骤 5)：按设计文档 §9 实现 =====
    //   1. 解析 tilingGM → DispatchCcuTilingData（worldSize / readyFlagsOffset / dataOffset / ...）
    //   2. 轮询 localBuffer + readyFlagsOffset 处的 worldSize 个 READY_FLAG，等全部置位
    //   3. 从 localBuffer + dataOffset 的 DATA 区按 METADATA 组装 token → expandXOut
    //   4. 向 localBuffer[0] 写 1 做收讫 ack
    //   等信号写法对齐现存 ops-transformer/mc2/moe_distribute_dispatch_teardown/op_kernel/arch35/
    // 当前为骨架占位空核。
    (void)localBuffer;
    (void)expandXOut;
    (void)tilingGM;
}

// <<<>>> 启动包装：host 侧（moe_distribute_dispatch_ccu_host.cpp）经此函数普通调用。
void DispatchTeardownLaunch(uint32_t blockDim, void* stream,
                            uint8_t* localBuffer, uint8_t* expandXOut, uint8_t* tilingGM)
{
    dispatch_teardown_aicore<<<blockDim, nullptr, stream>>>(localBuffer, expandXOut, tilingGM);
}
