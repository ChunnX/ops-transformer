/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dispatch_ffn_combine_tiling.h"
#include "dispatch_ffn_combine.h"

using namespace AscendC;
using namespace DispatchFFNCombineImpl;
extern "C" __global__ __aicore__ void dispatch_ffn_combine(
    GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR expertId, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR probs,
    GM_ADDR c, GM_ADDR expertTokenNums, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DispatchFFNCombineTilingData);
    if (TILING_KEY_IS(1000010)) {
        KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineTilingData, tilingData, tilingGM);
        DispatchFFNCombine<int8_t, DTYPE_W1, DTYPE_OUT, false, true, false> op;
        op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(1000110)) {
        KERNEL_TASK_TYPE(1000110, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineTilingData, tilingData, tilingGM);
        DispatchFFNCombine<int8_t, DTYPE_W1, DTYPE_OUT, false, true, true> op;
        op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingGM);
        op.Process();
    }
}
