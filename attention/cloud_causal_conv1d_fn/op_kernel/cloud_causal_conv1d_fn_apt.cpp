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
 * \file cloud_causal_conv1d_fn_apt.cpp
 * \brief CloudCausalConv1dFn kernel entry point
 */
#include "./arch35/cloud_causal_conv1d_fn_cut_bh.h"

#define TILING_KEY_BH_BF16 20000
#define TILING_KEY_BH_FP16 20001

extern "C" __global__ __aicore__ void
cloud_causal_conv1d_fn(GM_ADDR x,                           // 输入0: x
                    GM_ADDR weight,                      // 输入1: weight
                    GM_ADDR bias,                        // 输入2: bias (optional)
                    GM_ADDR convStates,                  // 输入3: convStates, updated in-place
                    GM_ADDR queryStartLoc,               // 输入4: queryStartLoc (optional)
                    GM_ADDR cacheIndices,                // 输入5: cacheIndices (optional)
                    GM_ADDR numAcceptedToken,            // 输入6: numAcceptedToken (optional)
                    GM_ADDR y,                           // 输出0: y
                    GM_ADDR workspace,                   // workspace
                    GM_ADDR tiling)                      // tiling
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(CloudCausalConv1dFnCutBHTilingData);
    REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR >= 20000)", CloudCausalConv1dFnCutBHTilingData);
    TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_BH_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(CloudCausalConv1dFnCutBHTilingData, tilingData, tiling);
        CloudCausalConv1dFnCutBH<bfloat16_t> op(&pipe);
        op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, numAcceptedToken, y, &tilingData);
        op.Process();
    }

    if (TILING_KEY_IS(TILING_KEY_BH_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(CloudCausalConv1dFnCutBHTilingData, tilingData, tiling);
        CloudCausalConv1dFnCutBH<half> op(&pipe);
        op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, numAcceptedToken, y, &tilingData);
        op.Process();
    }
}
