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
 * \file all_gather_mx_matmul_urma_kernel.h
 * \brief Kernel entry point for AllGatherQuantMatmul (URMA variant).
 *
 * Corresponds to the recipe DSL OpRecipe context manager + kernel launch:
 *
 *   with OpRecipe("AllGatherQuantMatmul_urma",
 *       inputs=[x1, x1_scale, x2, x2_scale, comm_ctx],
 *       outputs=[y],
 *       aiv_cores=32, aic_cores=32, total_blocks=32, workspace=...) as recipe:
 *       ...
 *
 *   # Launch: AllGatherQuantMatmulKernel<<<total_blocks, ..., stream>>>(
 *   #     devContext, devA, devScaleA, devB, devScaleB, devOutput, nullptr, tilingData)
 *
 * The entry function is intentionally thin: it instantiates the impl class,
 * calls Init() then Process(), mirroring the recipe's lifecycle.
 */

#pragma once

#include "all_gather_mx_matmul_urma_impl.h"
#include "../all_gather_mx_matmul_urma_tiling_data.h"

__global__ __aicore__ void AllGatherQuantMatmulKernel(__gm__ Apace::AivComm::CommContext *hcommCtx, GM_ADDR aGM,
                                                      GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
                                                      GM_ADDR biasGM, AllGatherMxMatmulUrmaTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllGatherMxMatmulUrmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> impl;
    impl.Init(hcommCtx, aGM, aScaleGM, bGM, bScaleGM, cGM, biasGM, &tilingData);
    impl.Process();
}
