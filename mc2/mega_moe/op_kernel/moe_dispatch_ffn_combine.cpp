/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mega_moe.h
 * \brief
 */
#ifndef MEGA_MOE_H
#define MEGA_MOE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "moe_dispatch_ffn_combine.h"
#include "moe_dispatch_ffn_combine_tiling.h"

using namespace AscendC;
using namespace MoeDispatchFFNCombineImpl;

__global__ __aicore__ void moe_dispatch_ffn_combine(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    Tpipe pipe;
    REGISTER_TILING_DEFAULT(MoeDispatchFFNCombineTilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDispatchFFNCombineTilingData, tilingData, tilingGM);
    MoeDispatchFFNCombine<DTYPE_X, DTYPE_Y, 3> op;
}