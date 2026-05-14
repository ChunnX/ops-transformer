/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dispatch_ffn_combine.cpp
 * \brief Unified DispatchFFNCombine entry kernel.
 *        Supports W4A8 (int4 weight), INT8 (int8 weight) and BF16/FP16 weight paths
 *        for both Atlas A2 (910B) and A3 (910_93) platforms.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dispatch_ffn_combine_tiling.h"
#include "dispatch_ffn_combine_tiling_key.h"
#include "dispatch_ffn_combine.h"

using namespace AscendC;
using namespace DispatchFFNCombineImpl;

template <bool TPL_IS_TRANSPOSE_W1, bool TPL_IS_TRANSPOSE_W2, bool TPL_IS_A2>
__global__ __aicore__ void dispatch_ffn_combine(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
    GM_ADDR weight1, GM_ADDR weight2, GM_ADDR weightScales1, GM_ADDR weightScales2,
    GM_ADDR xActiveMask, GM_ADDR scales,
    GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(DispatchFFNCombineTilingData);

    constexpr bool isW1Nz = (FORMAT_WEIGHT1 == FORMAT_FRACTAL_NZ);
    constexpr bool isW2Nz = (FORMAT_WEIGHT2 == FORMAT_FRACTAL_NZ);

    DispatchFFNCombine<DTYPE_X, DTYPE_WEIGHT1, DTYPE_Y,
                       TPL_IS_TRANSPOSE_W1, TPL_IS_TRANSPOSE_W2,
                       isW1Nz, isW2Nz, TPL_IS_A2> op;
    op.Init(context, x, topkIds, topkWeights, weight1, weight2,
            weightScales1, weightScales2, yOut, expertTokenNumsOut, workspaceGM, tilingGM);
    op.Process();
}
