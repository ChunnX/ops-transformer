/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_launcher.h
 * \brief quantMatmulReduceScatter kernel launcher 入口 (GET 模式)
 *
 * 参数增加 workspaceGM: GET 模式下作为 AllToAll 接收缓冲区
 */

#pragma once

#include "apace/kernel/apace_quant_matmul_reduce_scatter/apace_quant_matmul_reduce_scatter_udma_impl.h"

__global__ __aicore__ void  ApaceQuantMatmulReduceScatterKernelE4M3E4M3_Udma(
                          __gm__ CommContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR workspaceGM,
                           GM_ADDR cGM,
                           ApaceQuantMatmulReduceScatterTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  Apace::ApaceQuantMatmulReduceScatterUdmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, workspaceGM, cGM, &tilingData);
  impl.Run();
}
