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
 * \file apace_a2a_mm_kernel_launcher.h
 * \brief AllToAllMatmul（非量化，URMA 版）kernel launcher 入口
 */

#pragma once

#include "apace/kernel/matmul/batch_matmul/all_to_all_mm_urma_impl.h"

__global__ __aicore__ void AllToAllMmKernel_Udma(__gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR bGM,
                                                 GM_ADDR cGM, GM_ADDR biasGM, allToAllMmMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMmUrmaImpl<half, half, half, false, true> impl;
    impl.Init(hcommCtx, aGM, bGM, cGM, biasGM, &tilingData);
    impl.Run();
}

__global__ __aicore__ void AllToAllMmKernel_Bf16_Udma(__gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR bGM,
                                                      GM_ADDR cGM, GM_ADDR biasGM,
                                                      allToAllMmMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMmUrmaImpl<bfloat16_t, bfloat16_t, bfloat16_t, false, true> impl;
    impl.Init(hcommCtx, aGM, bGM, cGM, biasGM, &tilingData);
    impl.Run();
}
