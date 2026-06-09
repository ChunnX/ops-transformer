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
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dispatch_ffn_combine_bf16_tiling.h"
#include "dispatch_ffn_combine_bf16.h"

using namespace AscendC;
using namespace DispatchFFNCombineBF16Impl;
extern "C" __global__ __aicore__ void dispatch_ffn_combine_bf16(
    GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR expertId,
    GM_ADDR scale1, GM_ADDR scale2, GM_ADDR probs,
    GM_ADDR c, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DispatchFFNCombineBF16TilingData);
    if (TILING_KEY_IS(1000000)) {
        KERNEL_TASK_TYPE(1000000, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineBF16TilingData, tilingData, tilingGM);
        if (tilingData.dispatchFFNCombineBF16Info.isA2) {
            DispatchFFNCombineBF16<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true, true> op;
            op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingGM);
            op.Process();
        } else {
            DispatchFFNCombineBF16<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true, false> op;
            op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingGM);
            op.Process();
        }
    } else if (TILING_KEY_IS(1000001)) {
        KERNEL_TASK_TYPE(1000001, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineBF16TilingData, tilingData, tilingGM);
        if (tilingData.dispatchFFNCombineBF16Info.isA2) {
            DispatchFFNCombineBF16<DTYPE_A, DTYPE_W1, DTYPE_OUT, true, false, true> op;
            op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingGM);
            op.Process();
        } else {
            DispatchFFNCombineBF16<DTYPE_A, DTYPE_W1, DTYPE_OUT, true, false, false> op;
            op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingGM);
            op.Process();
        }
    } else if (TILING_KEY_IS(1000010)) {
        KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineBF16TilingData, tilingData, tilingGM);
        if (tilingData.dispatchFFNCombineBF16Info.isA2) {
            DispatchFFNCombineBF16<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true, true> op;
            op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingGM);
            op.Process();
        } else {
            DispatchFFNCombineBF16<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true, false> op;
            op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingGM);
            op.Process();
        }
    } else if (TILING_KEY_IS(1000011)) {
        KERNEL_TASK_TYPE(1000011, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineBF16TilingData, tilingData, tilingGM);
        if (tilingData.dispatchFFNCombineBF16Info.isA2) {
            DispatchFFNCombineBF16<DTYPE_A, DTYPE_W1, DTYPE_OUT, true, true, true> op;
            op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingGM);
            op.Process();
        } else {
            DispatchFFNCombineBF16<DTYPE_A, DTYPE_W1, DTYPE_OUT, true, true, false> op;
            op.Init(x, w1, w2, expertId, scale1, scale2, probs, c, expertTokenNums, workspaceGM, tilingGM);
            op.Process();
        }
    }
}
