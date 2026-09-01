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
 * \file apace_qmrs_kernel_launcher.h
 * \brief quantMatmulReduceScatter kernel launcher 入口 (PUT 模式)
 *
 * workspaceGM: AIC 写入 C_i 的缓冲区, AIV PUT 推送到其他 rank win 区
 */

#pragma once

#include "apace/kernel/fusions/matmul_reduce_scatter/quant_matmul_reduce_scatter_urma_impl.h"

__global__ __aicore__ void QuantMatmulReduceScatterKernelE4M3E4M3(__gm__ CommContext *hcommCtx, GM_ADDR aGM,
                                                                  GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM,
                                                                  GM_ADDR workspaceGM, GM_ADDR cGM,
                                                                  QuantMatmulReduceScatterTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::QuantMatmulReduceScatterImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, workspaceGM, cGM, &tilingData);
    impl.Run();
}
