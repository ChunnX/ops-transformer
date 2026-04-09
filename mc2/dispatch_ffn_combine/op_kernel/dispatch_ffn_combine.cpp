/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "dispatch_ffn_combine_tiling.h"
#include "dispatch_ffn_combine.h"

using namespace AscendC;
using namespace DispatchFFNCombineImpl;

__global__ __aicore__ void dispatch_ffn_combine(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
    GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
    GM_ADDR scales,
    GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DispatchFFNCombineTilingData);
    DispatchFFNCombine<bfloat16_t> op;
    op.Init(context, x, topkIds, topkWeights, weight1, weight2,
            xActiveMask, weightScales1, weightScales2,
            scales, yOut, expertTokenNumsOut,
            workspaceGM, tilingGM);
    op.Process();
}
